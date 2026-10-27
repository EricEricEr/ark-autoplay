"""539类异构资料的全量用途验收、家族轮换与最终共享模型训练。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..datapipe.stages_meta import family_of
from .all_fields_data import encode_sources, load_sources
from .all_fields_train import train_all_fields
from .kfold import make_folds
from .rotation import pooled_metrics, write_report
from .run_v1 import _make_train_cfg
from .state_action_data import STATE_DIM, STATE_NAMES


def _path(value: str) -> Path:
    """仅展开维护者本地配置，公开配置不包含本机路径。"""
    return Path(os.path.expandvars(value))


def _pooled_auxiliary(reports: list[dict]) -> dict:
    """按真实目标数量合并各折，各任务分母单独保留。"""
    result: dict[str, Counter] = {}
    for report in reports:
        for key, task in report["evaluation"]["tasks"].items():
            values = {field: task[field] for field in ("n", "correct", "loss_sum") if field in task}
            result.setdefault(key, Counter()).update(values)
    out = {}
    for key, counts in result.items():
        out[key] = dict(counts)
        if counts["n"]:
            if "correct" in counts:
                out[key]["accuracy"] = counts["correct"] / counts["n"]
            if "loss_sum" in counts:
                out[key]["mean_huber"] = counts["loss_sum"] / counts["n"]
    return out


def run_all_fields(config: dict, log: Callable = print) -> dict:
    """每份资料都有实际优化步骤与折外评测，不把失败局用作成功动作标签。"""
    import torch

    manifest = _path(config["data"]["manifest"])
    processed = _path(config["data"]["processed_dir"])
    output = _path(config["evaluation"]["runs_dir"])
    output.mkdir(parents=True, exist_ok=True)
    sources = load_sources(manifest)
    expected_count = int(config["data"]["expected_records"])
    if len(sources) != expected_count:
        raise ValueError(f"要求全量{expected_count}份，实际清单为{len(sources)}份，拒绝少用")
    identities = {source["id"] for source in sources}
    signature = hashlib.sha256(
        manifest.read_bytes() + json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    marker = output / "input_signature.sha256"
    if marker.exists() and marker.read_text(encoding="utf-8") != signature:
        raise ValueError("运行目录对应另一份资料清单或配置，拒绝混合")
    marker.write_text(signature, encoding="utf-8")
    cfg = _make_train_cfg(config)
    auxiliary = config["auxiliary"]
    if cfg.eval_train_batches or cfg.batch_size != 1:
        raise ValueError("本入口逐记录更新、全量评测；batch_size须为1，eval_train_batches须为0")
    device = config["experiment"]["device"]
    full, featvec, vocab, availability = encode_sources(sources, processed, identities)
    if len(full) != expected_count:
        raise ValueError("编码没有覆盖全部资料")
    write_report(output / "available_fields.json", availability)
    np.savez_compressed(
        output / "all_records.npz",
        records=np.asarray(full, object),
        featvec=featvec,
        vocab=np.asarray(vocab, object),
        n=len(full),
        state_dim=STATE_DIM,
        state_names=np.asarray(STATE_NAMES, object),
        grid_channels=int(full[0]["cells"].shape[-1]),
        training_mode=np.asarray("all_available_fields"),
    )
    # 先用全部资料跑一轮实际训练/评测验收，权重供最终模型继承，不注入验证折。
    warmup_dir = output / "usage_check"
    warmup_path = warmup_dir / "report.json"
    if not warmup_path.exists():
        pretrain = (
            _path(config["rotation"]["final_pretrain_ckpt"])
            if config["rotation"].get("final_pretrain_ckpt")
            else None
        )
        log(f"[phase] 全量用途验收：{len(full)}份全部进行优化和评测")
        train_all_fields(
            full,
            featvec,
            vocab,
            availability,
            replace(cfg, epochs=1),
            auxiliary,
            train_ids=identities,
            eval_ids=identities,
            output=warmup_dir,
            device_name=device,
            log=log,
            initialize=pretrain,
        )
    families = make_folds(
        [family_of(source["stage_id"]) for source in sources],
        int(config["rotation"]["k"]),
        cfg.seed,
    )
    if any(not group for group in families):
        raise ValueError("存在空折，不能声称资料获得轮换覆盖")
    write_report(
        output / "folds.json",
        {
            "families": families,
            "fixed_epochs": cfg.epochs,
            "validation_initialization": "fresh_no_warmup_or_pretrain_exposure",
        },
    )
    reports = []
    for index, family in enumerate(families):
        evaluated = {source["id"] for source in sources if family_of(source["stage_id"]) in family}
        trained = identities - evaluated
        directory = output / f"fold_{index + 1}"
        report = directory / "report.json"
        if report.exists():
            current = json.loads(report.read_text(encoding="utf-8"))
        else:
            records, features, fold_vocab, fields = encode_sources(sources, processed, trained)
            log(
                f"[phase] 第{index + 1}/{len(families)}折："
                f"训练{len(trained)}份，验证{len(evaluated)}份"
            )
            current = train_all_fields(
                records,
                features,
                fold_vocab,
                fields,
                cfg,
                auxiliary,
                train_ids=trained,
                eval_ids=evaluated,
                output=directory,
                device_name=device,
                log=log,
            )
        if set(current["train_ids"]) != trained or set(current["eval_ids"]) != evaluated:
            raise ValueError("已有折报告的数据身份与当前划分不符")
        reports.append(current)
        write_report(
            output / "progress.json",
            {"completed_folds": len(reports), "total_folds": len(families)},
        )
    training_counts = Counter(key for report in reports for key in report["train_ids"])
    evaluation_counts = Counter(key for report in reports for key in report["eval_ids"])
    if (
        set(training_counts) != identities
        or set(evaluation_counts) != identities
        or any(
            training_counts[key] != len(families) - 1 or evaluation_counts[key] != 1
            for key in identities
        )
    ):
        raise ValueError("全资料轮换覆盖不完整")
    policy_reports = [{"best_holdout": report["evaluation"]["policy"]} for report in reports]
    rotation = {
        "records": len(identities),
        "auxiliary": _pooled_auxiliary(reports),
        "policy": pooled_metrics(policy_reports),
        "per_record": [
            {
                "id": key,
                "training_folds": training_counts[key],
                "evaluation_folds": evaluation_counts[key],
            }
            for key in sorted(identities)
        ],
    }
    write_report(output / "out_of_fold.json", rotation)
    final_dir = output / "final_full"
    final_path = final_dir / "report.json"
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding="utf-8"))
    else:
        log("[phase] 轮换完成，开始所有资料的最终共享模型训练")
        final = train_all_fields(
            full,
            featvec,
            vocab,
            availability,
            cfg,
            auxiliary,
            train_ids=identities,
            eval_ids=identities,
            output=final_dir,
            device_name=device,
            log=log,
            initialize=warmup_dir / "last.pt",
        )
    if set(final["train_ids"]) != identities or set(final["eval_ids"]) != identities:
        raise ValueError("最终模型没有利用全部资料")
    result = {
        "phase": "complete",
        "records": len(identities),
        "availability": availability,
        "out_of_fold": rotation,
        "final_in_sample": final["evaluation"],
        "checkpoint": final["checkpoint"],
        "scope": "all_records_each_by_observed_fields_no_fabricated_labels",
    }
    write_report(output / "result.json", result)
    # 显式释放CUDA缓存，供顺序执行的本地任务继续使用。
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    """维护者入口；现有PyTorch依赖，无新增账号、服务或系统配置。"""
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_all_fields(yaml.safe_load(args.config.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
