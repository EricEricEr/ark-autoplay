"""状态-动作行为克隆 v1/v2/v3 CLI：构建数据集 / 训练 / K 折 / DIR 审计。

用法::

    python -m ark_core.training.run_v1 build   --config configs/imitation_v1.yaml
    python -m ark_core.training.run_v1 train   --config configs/imitation_v1.yaml
    python -m ark_core.training.run_v1 all     --config configs/imitation_v2.yaml
    python -m ark_core.training.run_v1 kfold   --config configs/imitation_v3_kfold.yaml
    python -m ark_core.training.run_v1 audit_dir --config configs/imitation_v3_kfold.yaml
    python -m ark_core.training.run_v1 pretrain --config configs/pretrain_ssl_v0.yaml
    python -m ark_core.training.run_v1 all     --config configs/imitation_v2.yaml \
        --increment-from ${ARK_ML_ROOT}/runs/imitation_v2

step 一览：
- ``build/train/all``：v1/v2 口径构建与训练；v2 加胜负口径清洗
  （``data.exclude_by_result`` / ``data.exclude_scan_file``，明细入 build_report
  的 ``excluded``；不配则完全等同 v1 行为）；
- ``kfold``：K 折关卡轮换留出，均值±std 汇总（v3 评测口径硬化，见 ``kfold`` 模块）；
- ``audit_dir``：DIR 朝向标签四项纯数据审计（见 ``dir_label_audit`` 模块）；
- ``--increment-from``：增量续训钩子——build 只编码旧 run 未见过的新局
  （词表/featvec 冻结，见 ``state_action_data.build_dataset_delta``），train 加载
  旧 run 的 checkpoint 继续训（``training.resume_from_ckpt`` 可显式覆盖权重来源）。

配置见 ``configs/imitation_v*.yaml``（一切超参入配置，禁止代码内魔法数字）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .run import _expand, _load_cfg, _repo_root


def _resolve_holdout_stages(d: dict[str, Any], counts: dict[str, int], log: Any) -> list[str]:
    """确定 holdout 关卡列表：配置显式给定则复核；为空则确定性自动选取。"""
    min_eps = int(d.get("holdout_min_eps", 3))
    explicit = [str(s) for s in (d.get("holdout_stages") or [])]
    if not explicit:
        auto_n = int(d.get("holdout_auto_n", 4))
        from .state_action_data import choose_holdout_stages

        picked = choose_holdout_stages(counts, auto_n, min_eps)
        log(
            f"[build] holdout_stages 未配置，按确定性规则自动选取 "
            f"(min_eps={min_eps}, n={auto_n}): {picked}"
        )
        return picked
    bad = [s for s in explicit if counts.get(s, 0) < min_eps]
    if bad:
        raise SystemExit(
            f"holdout_stages 复核失败：{bad} 的有效局数 < {min_eps}"
            f"（实扫 { {s: counts.get(s, 0) for s in bad} }）"
        )
    return explicit


def _collect_exclusion_from_cfg(d: dict[str, Any], episodes_root: Path, log: Any) -> Any:
    """按配置扫 v2 清洗排除集（未配置任一开关时返回 None = v1 行为）。"""
    from .state_action_data import collect_exclusions

    want_field = bool(d.get("exclude_by_result"))
    scan_cfg = d.get("exclude_scan_file")
    if not (want_field or scan_cfg):
        return None
    scan_path: Path | None = None
    if scan_cfg:
        # 相对路径以**单仓根**（core 的上一级）为基解析，机器绝对路径不入库
        p = Path(str(scan_cfg))
        if not p.is_absolute():
            p = (_repo_root().parent / str(scan_cfg)).resolve()
        if not p.is_file():
            raise SystemExit(f"exclude_scan_file 不存在: {p}")
        scan_path = p
    excl = collect_exclusions(episodes_root, exclude_by_result=want_field, scan_file=scan_path)
    log(
        f"[build] 清洗排除：字段侧 {len(excl.by_field)} 局 "
        f"{ {v: list(excl.by_field.values()).count(v) for v in set(excl.by_field.values())} }；"
        f"名单侧 {len(excl.by_scan)} 局；并集 {len(excl.union())} 局"
        + (f"（名单 {scan_path}）" if scan_path else "")
    )
    return excl


def _valid_dir_names(episodes_root: Path) -> list[str]:
    """有效局目录名全量列表（写入 build_report 供增量集差/审计参考）。"""
    from .state_action_data import iter_episode_dirs

    return sorted(ep_dir.name for _stage, ep_dir in iter_episode_dirs(episodes_root, None))


def _excluded_dir_names(episodes_root: Path, exclusion: Any) -> list[str]:
    """清洗并集 ∩ 有效局 的目录名列表（exclusion=None 时空表）。"""
    if exclusion is None:
        return []
    union = exclusion.union()
    return [n for n in _valid_dir_names(episodes_root) if n in union]


def _do_build(
    cfg: dict[str, Any], log: Any = print, increment_from: Path | None = None
) -> dict[str, Any]:
    """构建 v1/v2 数据集（词表只从训练侧建；build_report 落盘）。

    ``increment_from``：增量模式。指向旧 run 目录（含 build_report.json 与
    dataset_npz），本次只编码其未见过的新局，词表/featvec 冻结（详见
    :func:`ark_core.training.state_action_data.build_dataset_delta` 的不变式）。
    """
    from .state_action_data import (
        build_dataset,
        build_dataset_delta,
        scan_stage_counts,
    )

    d = cfg["data"]
    processed = _expand(d["processed_dir"])
    episodes_root = _expand(d["episodes_root"])
    out = _expand(d["dataset_npz"])
    grid_channels = int(d.get("grid_channels", 5))

    t0 = time.time()
    counts = scan_stage_counts(episodes_root)
    log(f"[build] 扫描 {len(counts)} 个关卡、{sum(counts.values())} 局有效 episode")
    holdout = _resolve_holdout_stages(d, counts, log)
    log(
        f"[build] holdout 关卡 {len(holdout)} 个: {holdout}（局数 "
        f"{ {s: counts.get(s, 0) for s in holdout} }）"
    )
    exclusion = _collect_exclusion_from_cfg(d, episodes_root, log)

    delta: dict[str, Any] | None = None
    if increment_from is not None:
        base_report_path = increment_from / "build_report.json"
        if not base_report_path.is_file():
            raise SystemExit(f"--increment-from 目录缺 build_report.json: {increment_from}")
        with base_report_path.open("r", encoding="utf-8") as f:
            base_report = json.load(f)
        base_npz_raw = base_report.get("dataset_npz")
        if not base_npz_raw:
            raise SystemExit(f"{base_report_path} 无 dataset_npz 字段，无法做增量基底")
        base_npz = Path(str(base_npz_raw))
        if not base_npz.is_file():
            raise SystemExit(f"增量基底 npz 不存在: {base_npz}")
        log(f"[build] 增量模式：基底 {base_npz}")
        vocab, stats, delta = build_dataset_delta(
            processed_dir=processed,
            episodes_root=episodes_root,
            base_npz=base_npz,
            out_path=out,
            holdout_stages=set(holdout),
            limit=d.get("limit"),
            vocab_top_k=int(d.get("vocab_top_k", 0)),
            exclusion=exclusion,
            grid_channels=grid_channels,
        )
        n = int(delta["total_records"])
    else:
        vocab, stats, n = build_dataset(
            processed_dir=processed,
            episodes_root=episodes_root,
            out_path=out,
            holdout_stages=set(holdout),
            limit=d.get("limit"),
            vocab_top_k=int(d.get("vocab_top_k", 0)),
            exclusion=exclusion,
            grid_channels=grid_channels,
        )
    rep = {
        "n_records": n,
        "grid_channels": grid_channels,
        "stages_no_route_feat": stats.n_stages_no_route_feat,
        "vocab_size": vocab.size,
        "vocab_tokens": vocab.extra_tokens,
        "elapsed_s": round(time.time() - t0, 1),
        "seen": stats.n_seen,
        "kept": stats.n_kept,
        "drop": {
            "bad_files": stats.drop_bad_files,
            "no_stage": stats.drop_no_stage,
            "no_squad": stats.drop_no_squad,
            "no_actions": stats.drop_no_actions,
            "too_many_cells": stats.drop_too_many_cells,
            "too_many_spawns": stats.drop_too_many_spawns,
            "too_many_actions": stats.drop_too_many_actions,
        },
        "anomaly": {
            "episode_len_mismatch": stats.n_episode_len_mismatch,
            "actions_dropped_tail": stats.n_actions_dropped_tail,
            "extra_states_dropped": stats.n_extra_states_dropped,
            "unpaired_actions_retained": stats.n_unpaired_actions,
            "unpaired_states_retained": stats.n_unpaired_states,
            "entity_only_records": stats.n_entity_only,
            "skill_daemon_filtered": stats.n_skill_daemon,
            "unknown_job_type": stats.n_unknown_job_type,
            "align_check_mismatch": stats.n_align_check_mismatch,
        },
        "excluded": (
            {
                "by_field": dict(sorted(exclusion.by_field.items())),
                "by_scan": dict(sorted(exclusion.by_scan.items())),
                "excluded_by_field": dict(sorted(Counter(exclusion.by_field.values()).items())),
                "excluded_by_scan": len(exclusion.by_scan),
                "union_n": len(exclusion.union()),
                "applied_to_valid_pool": stats.n_excluded,
            }
            if exclusion is not None
            else None
        ),
        "n_actions_used": stats.n_actions_used,
        "type_hist": dict(stats.type_hist),
        "squad_name_unk": stats.n_squad_name_unk,
        "deploy_operator": stats.n_operator_deploy,
        "deploy_external": stats.n_external_deploy,
        "target_unk": stats.n_target_unk,
        "cell_oob": stats.n_cell_oob,
        "dir_unknown": stats.n_dir_unknown,
        "holdout_stages": holdout,
        "holdout_stage_counts": {s: counts.get(s, 0) for s in holdout},
        "stage_counts": counts,
        "dataset_npz": str(out),
        "incremental": delta,
        # 供 --increment-from 集差/审计参考：有效局目录名全量 + 清洗实际命中的交集
        "episode_dirs_valid_seen": _valid_dir_names(episodes_root),
        "episode_dirs_excluded_applied": _excluded_dir_names(episodes_root, exclusion),
    }
    log(
        f"[build] 保留 {n} 局（扫描 {stats.n_seen}）→ {out}  "
        f"词表 {vocab.size}  ({rep['elapsed_s']}s)"
    )
    if delta is not None:
        log(
            f"[build] 增量：基底 {delta['n_base_records']} + 新增 {delta['n_new_records']}"
            f"（跳过已见 {delta['n_seen_skipped']}；新局目录 {delta['new_episode_dirs']}）"
        )
    log(f"[build] 丢弃明细: {rep['drop']}")
    log(f"[build] 异常明细: {rep['anomaly']}")
    log(
        f"[build] 动作 {stats.n_actions_used} 条 {dict(stats.type_hist)}；"
        f"deploy 干员 {stats.n_operator_deploy} / 非干员 {stats.n_external_deploy}；"
        f"squad_name_unk {stats.n_squad_name_unk}；target_unk {stats.n_target_unk}；"
        f"cell_oob {stats.n_cell_oob}"
    )
    out_report = (
        _expand(cfg["evaluation"]["runs_dir"])
        / str(cfg.get("experiment", {}).get("name", "run"))
        / "build_report.json"
    )
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    log(f"[build] build_report → {out_report}")
    return rep


def _make_train_cfg(cfg: dict[str, Any]) -> Any:
    """从实验配置组装 TrainConfig（train / kfold 共用同一处，避免两路漂移）。"""
    from .imitate import TrainConfig

    m = cfg.get("model", {})
    t = cfg.get("training", {})
    return TrainConfig(
        d_model=int(m.get("hidden_dim", 256)),
        n_heads=int(m.get("num_heads", 8)),
        n_layers=int(m.get("num_layers", 4)),
        n_dec_layers=int(m.get("num_dec_layers", 2)),
        dropout=float(m.get("dropout", 0.1)),
        batch_size=int(t.get("batch_size", 16)),
        lr=float(t.get("lr", 3e-4)),
        weight_decay=float(t.get("weight_decay", 0.01)),
        epochs=int(t.get("epochs", 20)),
        grad_clip=float(t.get("grad_clip", 1.0)),
        seed=int(cfg.get("experiment", {}).get("seed", 42)),
        w_type=float(t.get("w_type", 1.0)),
        w_target=float(t.get("w_target", 1.0)),
        w_cell=float(t.get("w_cell", 1.0)),
        w_dir=float(t.get("w_dir", 0.5)),
        eval_train_batches=int(t.get("eval_train_batches", 0)),
    )


def _resolve_resume_ckpt(cfg: dict[str, Any], increment_from: Path | None, log: Any) -> Path | None:
    """续训权重路径：``training.resume_from_ckpt`` 优先；否则取 --increment-from run
    目录里唯一的 *.pt（多个时报错，不猜）。"""
    t = cfg.get("training", {})
    explicit = t.get("resume_from_ckpt")
    if explicit:
        p = _expand(str(explicit))
        if not p.is_file():
            raise SystemExit(f"resume_from_ckpt 不存在: {p}")
        return p
    if increment_from is not None:
        pts = sorted(increment_from.glob("*.pt"))
        if not pts:
            raise SystemExit(f"--increment-from 目录无 checkpoint: {increment_from}")
        if len(pts) > 1:
            raise SystemExit(f"{increment_from} 有多个 *.pt，请用 resume_from_ckpt 显式指定")
        return pts[0]
    return None


def _do_train(
    cfg: dict[str, Any], log: Any = print, increment_from: Path | None = None
) -> dict[str, Any]:
    """训练（v1：状态注入模型 + 关卡留出 + 常数基线；可增量续训）。"""
    from .imitate_v1 import train

    d = cfg["data"]
    t = cfg.get("training", {})
    tcfg = _make_train_cfg(cfg)
    out_dir = _expand(cfg["evaluation"]["runs_dir"]) / str(
        cfg.get("experiment", {}).get("name", "run")
    )
    holdout = {str(s) for s in (d.get("holdout_stages_resolved") or d.get("holdout_stages") or [])}
    resume = _resolve_resume_ckpt(cfg, increment_from, log)
    result = train(
        _expand(d["dataset_npz"]),
        tcfg,
        holdout_stages=holdout,
        out_dir=out_dir,
        device_str=str(cfg.get("experiment", {}).get("device", "cuda")),
        log=log,
        ckpt_name=str(t.get("ckpt_name", "track_a_v1.pt")),
        resume_from=resume,
    )
    out_report = out_dir / "train_report.json"
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"[train] train_report → {out_report}")
    return result


def _do_kfold(cfg: dict[str, Any], log: Any = print) -> dict[str, Any]:
    """K 折关卡轮换留出（v3 评测口径硬化，见 ``kfold`` 模块 docstring）。"""
    from .kfold import run_kfold

    d = cfg["data"]
    k_cfg = cfg.get("kfold", {})
    k = int(k_cfg.get("k", 5))
    seed = int(k_cfg.get("seed", cfg.get("experiment", {}).get("seed", 42)))
    arms = k_cfg.get("arms") or [{"name": "full", "dir_loss": "full"}]
    # arm 的 pretrain_ckpt 允许 ${VAR} 形式（机器路径不入库），此处展开成绝对路径
    for _i, _a in enumerate(arms):
        if _a.get("pretrain_ckpt"):
            arms[_i] = {**_a, "pretrain_ckpt": str(_expand(str(_a["pretrain_ckpt"])))}
    out_dir = _expand(cfg["evaluation"]["runs_dir"]) / str(
        cfg.get("experiment", {}).get("name", "run")
    )
    result = run_kfold(
        _expand(d["dataset_npz"]),
        _make_train_cfg(cfg),
        k=k,
        seed=seed,
        out_dir=out_dir,
        device_str=str(cfg.get("experiment", {}).get("device", "cuda")),
        arms=[dict(a) for a in arms],
        w_dir_weighted=float(cfg.get("training", {}).get("dir_loss_weighted", 0.1)),
        log=log,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_summary = out_dir / str(k_cfg.get("summary_out", "metrics_summary.json"))
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"[kfold] metrics_summary → {out_summary}")
    return result


def _do_audit_dir(cfg: dict[str, Any], log: Any = print) -> dict[str, Any]:
    """DIR 标签审计（先审数据再谈模型，见 ``dir_label_audit`` 模块 docstring）。"""
    from .dir_label_audit import audit_dir_labels

    d = cfg["data"]
    a_cfg = cfg.get("audit_dir", {})
    episodes_root = _expand(d["episodes_root"])
    exclusion = _collect_exclusion_from_cfg(d, episodes_root, log)
    rep = audit_dir_labels(
        _expand(d["processed_dir"]),
        episodes_root,
        holdout_stages={str(s) for s in (d.get("holdout_stages") or [])},
        exclusion=exclusion,
        limit=d.get("limit"),
        same_job_inconsistency_dirty=float(a_cfg.get("same_job_inconsistency_dirty", 0.05)),
        unknown_alias_dirty=float(a_cfg.get("unknown_alias_dirty", 0.01)),
    )
    out_path = _expand(a_cfg.get("out", "${ARK_ML_ROOT}/dir_label_audit.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    log(f"[audit] episodes={rep['episodes']} actions={rep['actions']} dirty={rep['dirty']}")
    log(f"[audit] 结论: {rep['conclusion']}")
    log(f"[audit] 报告 → {out_path}")
    return rep


def _do_pretrain(cfg: dict[str, Any], log: Any = print) -> dict[str, Any]:
    """自监督预训练（掩码实体建模）+ 可选下游 3 epoch 微调烟测。"""
    from .imitate import TrainConfig
    from .pretrain_ssl import PretextConfig, finetune_smoke, pretrain

    d = cfg["data"]
    m = cfg.get("model", {})
    p = cfg.get("pretrain", {})
    pcfg = PretextConfig(
        d_model=int(m.get("hidden_dim", 256)),
        n_heads=int(m.get("num_heads", 8)),
        n_layers=int(m.get("num_layers", 4)),
        n_dec_layers=int(m.get("num_dec_layers", 2)),
        dropout=float(m.get("dropout", 0.1)),
        batch_size=int(p.get("batch_size", 32)),
        lr=float(p.get("lr", 3e-4)),
        weight_decay=float(p.get("weight_decay", 0.01)),
        epochs=int(p.get("epochs", 12)),
        grad_clip=float(p.get("grad_clip", 1.0)),
        seed=int(cfg.get("experiment", {}).get("seed", 42)),
        limit_train=int(p.get("limit_train", 0)),
        holdout_eval_batches=int(p.get("holdout_eval_batches", 0)),
        mask_rate=float(p.get("mask_rate", 0.15)),
        ratio_mask=float(p.get("ratio_mask", 0.8)),
        ratio_random=float(p.get("ratio_random", 0.1)),
        huber_delta=float(p.get("huber_delta", 1.0)),
        w_recon_cell=float(p.get("w_recon_cell", 1.0)),
        w_recon_spawn=float(p.get("w_recon_spawn", 1.0)),
        w_cls_squad=float(p.get("w_cls_squad", 1.0)),
        w_act_type=float(p.get("w_act_type", 1.0)),
        w_act_target=float(p.get("w_act_target", 1.0)),
        w_act_cell=float(p.get("w_act_cell", 1.0)),
        w_act_dir=float(p.get("w_act_dir", 0.5)),
    )
    out_dir = _expand(cfg["evaluation"]["runs_dir"]) / str(
        cfg.get("experiment", {}).get("name", "run")
    )
    device_str = str(cfg.get("experiment", {}).get("device", "cuda"))
    result: dict[str, Any] = {}
    rep = pretrain(
        _expand(d["dataset_npz"]),
        pcfg,
        holdout_families={str(s) for s in (d.get("holdout_families") or [])},
        out_dir=out_dir,
        device_str=device_str,
        log=log,
    )
    result["pretrain"] = rep
    with (out_dir / "pretrain_report.json").open("w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    log(f"[pretrain] pretrain_report → {out_dir / 'pretrain_report.json'}")

    s = cfg.get("downstream_smoke", {})
    if s.get("enabled"):
        scfg = TrainConfig(
            d_model=int(m.get("hidden_dim", 256)),
            n_heads=int(m.get("num_heads", 8)),
            n_layers=int(m.get("num_layers", 4)),
            n_dec_layers=int(m.get("num_dec_layers", 2)),
            dropout=float(m.get("dropout", 0.1)),
            batch_size=int(s.get("batch_size", 16)),
            lr=float(s.get("lr", 3e-4)),
            weight_decay=float(s.get("weight_decay", 0.01)),
            epochs=int(s.get("epochs", 3)),
            grad_clip=float(s.get("grad_clip", 1.0)),
            seed=int(cfg.get("experiment", {}).get("seed", 42)),
            w_type=float(s.get("w_type", 1.0)),
            w_target=float(s.get("w_target", 1.0)),
            w_cell=float(s.get("w_cell", 1.0)),
            w_dir=float(s.get("w_dir", 0.5)),
            eval_train_batches=int(s.get("eval_train_batches", 0)),
        )
        smoke = finetune_smoke(
            Path(str(rep["checkpoint"])),
            _expand(s["dataset_npz"]),
            scfg,
            holdout_stages={str(x) for x in (s.get("holdout_stages") or [])},
            out_dir=out_dir / "finetune_smoke",
            device_str=device_str,
            log=log,
        )
        result["finetune_smoke"] = smoke
        with (out_dir / "finetune_smoke_report.json").open("w", encoding="utf-8") as f:
            json.dump(smoke, f, ensure_ascii=False, indent=2)
        log(f"[smoke] finetune_smoke_report → {out_dir / 'finetune_smoke_report.json'}")
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    p = argparse.ArgumentParser(prog="python -m ark_core.training.run_v1")
    p.add_argument("step", choices=["build", "train", "all", "kfold", "audit_dir", "pretrain"])
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="实验配置（默认 <repo>/configs/imitation_v1.yaml）",
    )
    p.add_argument("--report", type=Path, default=None, help="把结果 JSON 写到该路径")
    p.add_argument(
        "--increment-from",
        type=Path,
        default=None,
        help="增量模式：旧 run 目录（含 build_report.json 与 *.pt ckpt）。"
        "build 只编码其未见过的新局（词表/featvec 冻结），train 加载旧权重续训",
    )
    args = p.parse_args(argv)

    # 行缓冲 + 立即 flush：训练耗时长，输出被缓冲会让进度不可观测（约定同 run.py）
    def log(msg: str) -> None:
        print(msg, flush=True)

    cfg_path = args.config or (_repo_root() / "configs" / "imitation_v1.yaml")
    cfg = _load_cfg(cfg_path)
    log(f"[cfg] {cfg_path}")

    increment_from = args.increment_from
    if increment_from is not None and not increment_from.is_dir():
        raise SystemExit(f"--increment-from 目录不存在: {increment_from}")

    result: dict[str, Any] = {"config": str(cfg_path)}
    if args.step in ("build", "all"):
        result["build"] = _do_build(cfg, log, increment_from)
        # all 模式下把 build 复核后的 holdout 列表传给 train（显式配置时原样透传）
        cfg.setdefault("data", {})["holdout_stages_resolved"] = result["build"]["holdout_stages"]
    if args.step in ("train", "all"):
        result["train"] = _do_train(cfg, log, increment_from)
    if args.step == "kfold":
        result["kfold"] = _do_kfold(cfg, log)
    if args.step == "audit_dir":
        result["audit_dir"] = _do_audit_dir(cfg, log)
    if args.step == "pretrain":
        result["pretrain"] = _do_pretrain(cfg, log)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        log(f"[report] {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
