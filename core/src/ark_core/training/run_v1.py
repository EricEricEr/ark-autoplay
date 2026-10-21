"""状态-动作行为克隆 v1/v2 CLI：构建数据集 / 训练 / 评测。

用法::

    python -m ark_core.training.run_v1 build   --config configs/imitation_v1.yaml
    python -m ark_core.training.run_v1 train   --config configs/imitation_v1.yaml
    python -m ark_core.training.run_v1 all     --config configs/imitation_v2.yaml

与 v0 ``run`` 的差异：数据源是真机 episodes 目录树（非 episodes_v0.jsonl），
留出为**关卡级**（``holdout_stages``），空列表时按确定性规则自动选取
（见 ``state_action_data.choose_holdout_stages``，结果写入 build_report）。
v2 增加胜负口径清洗：配置给 ``data.exclude_by_result`` / ``data.exclude_scan_file``
时按双侧并集剔除判负/阻断局（详见 state_action_data 模块 docstring 的清洗小节，
排除明细写入 build_report 的 ``excluded``）；不配则完全等同 v1 行为。
配置见 ``configs/imitation_v1.yaml`` / ``configs/imitation_v2.yaml``（一切超参入
配置，禁止代码内魔法数字）。
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


def _resolve_holdout_stages(
    d: dict[str, Any], counts: dict[str, int], log: Any
) -> list[str]:
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


def _collect_exclusion_from_cfg(
    d: dict[str, Any], episodes_root: Path, log: Any
) -> Any:
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


def _do_build(cfg: dict[str, Any], log: Any = print) -> dict[str, Any]:
    """构建 v1/v2 数据集（词表只从训练侧建；build_report 落盘）。"""
    from .state_action_data import build_dataset, scan_stage_counts

    d = cfg["data"]
    processed = _expand(d["processed_dir"])
    episodes_root = _expand(d["episodes_root"])
    out = _expand(d["dataset_npz"])

    t0 = time.time()
    counts = scan_stage_counts(episodes_root)
    log(f"[build] 扫描 {len(counts)} 个关卡、{sum(counts.values())} 局有效 episode")
    holdout = _resolve_holdout_stages(d, counts, log)
    log(f"[build] holdout 关卡 {len(holdout)} 个: {holdout}（局数 "
        f"{ {s: counts.get(s, 0) for s in holdout} }）")
    exclusion = _collect_exclusion_from_cfg(d, episodes_root, log)

    vocab, stats, n = build_dataset(
        processed_dir=processed,
        episodes_root=episodes_root,
        out_path=out,
        holdout_stages=set(holdout),
        limit=d.get("limit"),
        vocab_top_k=int(d.get("vocab_top_k", 256)),
        exclusion=exclusion,
    )
    rep = {
        "n_records": n,
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
            "skill_daemon_filtered": stats.n_skill_daemon,
            "unknown_job_type": stats.n_unknown_job_type,
            "align_check_mismatch": stats.n_align_check_mismatch,
        },
        "excluded": (
            {
                "by_field": dict(sorted(exclusion.by_field.items())),
                "by_scan": dict(sorted(exclusion.by_scan.items())),
                "excluded_by_field": dict(
                    sorted(Counter(exclusion.by_field.values()).items())
                ),
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
    }
    log(f"[build] 保留 {n} 局（扫描 {stats.n_seen}）→ {out}  "
        f"词表 {vocab.size}  ({rep['elapsed_s']}s)")
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


def _do_train(cfg: dict[str, Any], log: Any = print) -> dict[str, Any]:
    """训练（v1：状态注入模型 + 关卡留出 + 常数基线）。"""
    from .imitate import TrainConfig
    from .imitate_v1 import train

    d = cfg["data"]
    m = cfg.get("model", {})
    t = cfg.get("training", {})
    tcfg = TrainConfig(
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
        eval_train_batches=int(t.get("eval_train_batches", 64)),
    )
    out_dir = _expand(cfg["evaluation"]["runs_dir"]) / str(
        cfg.get("experiment", {}).get("name", "run")
    )
    holdout = {str(s) for s in (d.get("holdout_stages_resolved") or d.get("holdout_stages") or [])}
    result = train(
        _expand(d["dataset_npz"]),
        tcfg,
        holdout_stages=holdout,
        out_dir=out_dir,
        device_str=str(cfg.get("experiment", {}).get("device", "cuda")),
        log=log,
        ckpt_name=str(t.get("ckpt_name", "track_a_v1.pt")),
    )
    out_report = out_dir / "train_report.json"
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"[train] train_report → {out_report}")
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    p = argparse.ArgumentParser(prog="python -m ark_core.training.run_v1")
    p.add_argument("step", choices=["build", "train", "all"])
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="实验配置（默认 <repo>/configs/imitation_v1.yaml）",
    )
    p.add_argument("--report", type=Path, default=None, help="把结果 JSON 写到该路径")
    args = p.parse_args(argv)

    # 行缓冲 + 立即 flush：训练耗时长，输出被缓冲会让进度不可观测（约定同 run.py）
    def log(msg: str) -> None:
        print(msg, flush=True)

    cfg_path = args.config or (_repo_root() / "configs" / "imitation_v1.yaml")
    cfg = _load_cfg(cfg_path)
    log(f"[cfg] {cfg_path}")

    result: dict[str, Any] = {"config": str(cfg_path)}
    if args.step in ("build", "all"):
        result["build"] = _do_build(cfg, log)
        # all 模式下把 build 复核后的 holdout 列表传给 train（显式配置时原样透传）
        cfg.setdefault("data", {})["holdout_stages_resolved"] = result["build"]["holdout_stages"]
    if args.step in ("train", "all"):
        result["train"] = _do_train(cfg, log)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        log(f"[report] {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
