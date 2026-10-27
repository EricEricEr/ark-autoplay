"""轮换验证后全量拟合：每局均有覆盖记录，严格分开折外成绩和已见数据成绩。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..datapipe.stages_meta import family_of
from .imitate_v1 import train
from .kfold import _build_pretrained_snapshot, make_folds
from .run_v1 import _make_train_cfg
from .state_action_data import base_stage_id, build_dataset


def write_report(path: Path, value: Any) -> None:
    """报告原子替换，重启时不能把半份报告当作完成。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def stamp_ids(path: Path) -> list[dict[str, Any]]:
    """给计算副本附加逐记录身份，重复episode_id也分别覆盖，原始文件不修改。"""
    with np.load(path, allow_pickle=True) as archive:
        arrays = {key: archive[key] for key in archive.files}
    records = list(arrays["records"])
    for index, record in enumerate(records):
        record["coverage_id"] = f"{index}:{record['stage_id']}:{record['episode_id']}"
    arrays["records"] = np.asarray(records, dtype=object)
    np.savez_compressed(path, **arrays)
    return records


def grouped_folds(records: list[dict], k: int, seed: int) -> list[set[str]]:
    """按章节/活动家族轮换；同关卡及难度变体保持同侧，整份数据不抽样。"""
    labeled = [r for r in records if len(r["type"])]
    groups = [family_of(base_stage_id(str(r["stage_id"]))) for r in labeled]
    if len(set(groups)) < k:
        raise ValueError(f"有标签关卡家族只有{len(set(groups))}个，无法组成{k}折")
    families = make_folds(groups, k, seed)
    return [
        {
            base_stage_id(str(r["stage_id"]))
            for r in records
            if family_of(base_stage_id(str(r["stage_id"]))) in group
        }
        for group in families
    ]


def pooled_metrics(reports: list[dict]) -> dict:
    """按全体折外预测计数汇总，避免将小折与大折等权平均。"""
    metrics = [r["best_holdout"] for r in reports]
    classes = sorted({name for row in metrics for name in row["per_class"]})
    per_class = {}
    for name in classes:
        counts = {
            key: sum(row["per_class"].get(name, {}).get(key, 0) for row in metrics)
            for key in ("support", "pred_n", "tp")
        }
        precision = counts["tp"] / counts["pred_n"] if counts["pred_n"] else 0.0
        recall = counts["tp"] / counts["support"] if counts["support"] else 0.0
        per_class[name] = {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    tokens = sum(row["n_tokens"] for row in metrics)
    active = [row["f1"] for row in per_class.values() if row["support"]]
    result = {
        "n_episodes": sum(row["n_episodes"] for row in metrics),
        "n_tokens": tokens,
        "type_acc": sum(row["tp"] for row in per_class.values()) / tokens if tokens else 0.0,
        "macro_f1": float(np.mean(active)) if active else 0.0,
        "per_class": per_class,
    }
    for kind, denominator in (("cell", "cell_total"), ("dir", "dir_total"), ("target", "n_tokens")):
        total = sum(row[denominator] for row in metrics)
        result[kind + "_total"] = total
        result[kind + "_acc"] = (
            sum(row[kind + "_acc"] * row[denominator] for row in metrics) / total if total else 0.0
        )
    return result


def run_rotation(config: dict, log: Any = print) -> dict:
    """每折从独立初始化开始，固定末轮评测；最后全量训练可复用预训练权重。"""
    data, rotation = config["data"], config["rotation"]
    processed = Path(os.path.expandvars(data["processed_dir"]))
    episodes = Path(os.path.expandvars(data["episodes_root"]))
    root = Path(os.path.expandvars(config["evaluation"]["runs_dir"]))
    root.mkdir(parents=True, exist_ok=True)
    cfg = _make_train_cfg(config)
    if cfg.eval_train_batches != 0:
        raise ValueError("轮换验证及全量拟合要求 eval_train_batches=0，逐局全量评测")
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    marker = root / "configuration.sha256"
    if marker.exists() and marker.read_text(encoding="utf-8") != config_hash:
        raise ValueError("输出目录已有不同配置，拒绝混用训练产物")
    marker.write_text(config_hash, encoding="utf-8")
    full_npz = root / "all_records.npz"
    _, stats, _ = build_dataset(processed, episodes, out_path=full_npz, holdout_stages=set())
    records = stamp_ids(full_npz)
    folds = grouped_folds(records, int(rotation["k"]), cfg.seed)
    labeled = [r for r in records if len(r["type"])]
    expected = Counter(str(r["coverage_id"]) for r in labeled)
    expected_ids = [str(r["coverage_id"]) for r in records]
    write_report(
        root / "plan.json",
        {
            "records": len(records),
            "labeled_records": len(labeled),
            "unlabeled_records": len(records) - len(labeled),
            "stats": {
                name: dict(value) if isinstance(value, Counter) else value
                for name, value in vars(stats).items()
            },
            "folds": [sorted(fold) for fold in folds],
            "grouping": "chapter_or_event_family",
            "fixed_epochs": cfg.epochs,
            "validation_initialization": "fresh_per_fold",
            "pretrained_weights_used_in_validation": False,
            "pretrained_weights_used_in_final_fit": bool(rotation.get("final_pretrain_ckpt")),
        },
    )
    reports = []
    device = str(config.get("experiment", {}).get("device", "cuda"))
    for index, hold in enumerate(folds):
        directory = root / f"fold_{index + 1}"
        report_path = directory / "train_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            dataset = directory / "records.npz"
            build_dataset(processed, episodes, out_path=dataset, holdout_stages=hold)
            current = stamp_ids(dataset)
            if [str(r["coverage_id"]) for r in current] != expected_ids:
                raise ValueError("折内数据记录发生变化，拒绝漏评或换样本")
            log(f"[rotation] fold={index + 1}/{len(folds)}，预训练不注入验证轮")
            report = train(
                dataset,
                cfg,
                holdout_stages=hold,
                out_dir=directory,
                device_str=device,
                log=log,
                checkpoint_selection="last",
            )
            write_report(report_path, report)
        reports.append(report)
        write_report(
            root / "rotation_progress.json",
            {"finished_folds": len(reports), "total_folds": len(folds)},
        )
    trained = Counter(item for report in reports for item in report["train_episode_ids"])
    evaluated = Counter(item for report in reports for item in report["holdout_episode_ids"])
    if evaluated != expected or trained != Counter(
        {key: n * (len(folds) - 1) for key, n in expected.items()}
    ):
        raise ValueError("轮换覆盖检查失败，未达到每份验证一次、训练k-1次")
    oof = pooled_metrics(reports)
    write_report(
        root / "out_of_fold_report.json",
        {
            "metrics": oof,
            "training_coverage_complete": True,
            "evaluation_coverage_complete": True,
            "selection": "fixed_final_epoch",
            "initialization": "fresh_no_pretraining_exposure",
            "per_record": [
                {"id": key, "train_folds": trained[key], "eval_folds": evaluated[key]}
                for key in expected
            ],
        },
    )
    log(f"[rotation] 全部折外评测已完成：{oof['n_episodes']}局/{oof['n_tokens']}动作")
    final_dir = root / "final_full"
    final_path = final_dir / "train_report.json"
    if final_path.exists():
        final_report = json.loads(final_path.read_text(encoding="utf-8"))
    else:
        resume = None
        if rotation.get("final_pretrain_ckpt"):
            resume, _ = _build_pretrained_snapshot(
                full_npz,
                Path(os.path.expandvars(rotation["final_pretrain_ckpt"])),
                final_dir,
                cfg,
                log,
            )
        final_report = train(
            full_npz,
            cfg,
            holdout_stages=set(),
            out_dir=final_dir,
            device_str=device,
            log=log,
            full_data=True,
            checkpoint_selection="last",
            resume_from=resume,
            ckpt_name="policy_full.pt",
        )
        if Counter(final_report["train_episode_ids"]) != expected:
            raise ValueError("最终全量训练覆盖检查失败")
        write_report(final_path, final_report)
    result = {
        "phase": "complete",
        "labeled_records": len(labeled),
        "folds": len(folds),
        "out_of_fold": oof,
        "final_full_checkpoint": final_report["checkpoint"],
        "final_fit_in_sample": final_report["train_at_best"],
        "note": "折外成绩来自无预训练暴露的独立模型；"
        "最终全量模型使用预训练初始化，已见数据成绩单列。",
    }
    write_report(root / "result.json", result)
    return result


def main() -> None:
    """维护者训练入口；路径来自配置与环境变量，真实数据不进入仓库。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_rotation(config)


if __name__ == "__main__":
    main()
