"""模仿学习 v0 CLI：构建数据集 / 训练 / 评测。

用法::

    python -m ark_core.training.run build   --config configs/imitation_v0.yaml
    python -m ark_core.training.run train   --config configs/imitation_v0.yaml
    python -m ark_core.training.run all     --config configs/imitation_v0.yaml

配置见 ``configs/imitation_v0.yaml``（一切超参入配置，禁止代码内魔法数字）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def _load_cfg(path: Path) -> dict[str, Any]:
    """读实验配置并展开环境变量（机器路径不入库，见 configs/imitation_v0.yaml 头注）。"""
    import os

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"配置不是 mapping: {path}")
    if not os.environ.get("ARK_ML_ROOT"):
        raise SystemExit(
            "未设置环境变量 ARK_ML_ROOT。\n"
            "它是本管线所有数据/输出路径的根（机器路径不入库）。\n"
            '  PowerShell:  $env:ARK_ML_ROOT = "E:\\path\\to\\workdir"\n'
            "  目录约定见 configs/imitation_v0.yaml 头部注释。"
        )
    return cfg


def _expand(p: str) -> Path:
    """展开 ${VAR} 与 ~，返回绝对路径。"""
    import os

    return Path(os.path.expanduser(os.path.expandvars(str(p)))).resolve()


def _repo_root() -> Path:
    """仓库根（src 布局：本文件上三级）。"""
    return Path(__file__).resolve().parents[3]


def _do_build(cfg: dict[str, Any], log=print) -> dict[str, Any]:
    """构建数据集（词表只从训练家族建，避免泄漏）。"""
    from ..datapipe.stages_meta import family_of
    from .imitation_data import build_dataset

    d = cfg["data"]
    processed = _expand(d["processed_dir"])
    episodes = _expand(d["episodes_jsonl"])
    out = _expand(d["dataset_npz"])
    holdout = set(d.get("holdout_families") or [])

    # 训练家族 = 数据里出现过的家族 − holdout
    fams: set[str] = set()
    with episodes.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            sid = str((ep.get("stage") or {}).get("stage_id") or "")
            if sid:
                fams.add(family_of(sid))
    train_fams = fams - holdout
    log(f"[build] 家族总 {len(fams)}，训练用 {len(train_fams)}，holdout {len(holdout)}")

    t0 = time.time()
    vocab, stats, n = build_dataset(
        processed_dir=processed,
        episodes_path=episodes,
        train_families=train_fams,
        out_path=out,
        limit=d.get("limit"),
        vocab_top_k=int(d.get("vocab_top_k", 256)),
    )
    rep = {
        "n_records": n,
        "vocab_size": vocab.size,
        "elapsed_s": round(time.time() - t0, 1),
        "seen": stats.n_seen,
        "kept": stats.n_kept,
        "drop": {
            "no_stage": stats.drop_no_stage,
            "no_squad": stats.drop_no_squad,
            "no_actions": stats.drop_no_actions,
            "too_many_cells": stats.drop_too_many_cells,
            "too_many_spawns": stats.drop_too_many_spawns,
            "too_many_actions": stats.drop_too_many_actions,
        },
        "type_hist": dict(stats.type_hist),
        "deploy_operator": stats.n_operator_deploy,
        "deploy_external": stats.n_external_deploy,
        "target_unk": stats.n_target_unk,
        "cell_oob": stats.n_cell_oob,
        "dir_unknown": stats.n_dir_unknown,
        "dataset_npz": str(out),
    }
    log(
        f"[build] 保留 {n} 局（扫描 {stats.n_seen}）→ {out}  "
        f"词表 {vocab.size}  ({rep['elapsed_s']}s)"
    )
    log(f"[build] 丢弃明细: {rep['drop']}")
    log(
        f"[build] deploy: 干员 {stats.n_operator_deploy} / 非干员 {stats.n_external_deploy}；"
        f"目标落 UNK {stats.n_target_unk}；格子越界 {stats.n_cell_oob}"
    )
    return rep


def _do_train(cfg: dict[str, Any], log=print) -> dict[str, Any]:
    """训练。"""
    from .imitate import TrainConfig, train

    d = cfg["data"]
    m = cfg.get("model", {})
    t = cfg.get("training", {})
    tcfg = TrainConfig(
        d_model=int(m.get("hidden_dim", 256)),
        n_heads=int(m.get("num_heads", 8)),
        n_layers=int(m.get("num_layers", 4)),
        n_dec_layers=int(m.get("num_dec_layers", 2)),
        dropout=float(m.get("dropout", 0.1)),
        batch_size=int(t.get("batch_size", 32)),
        lr=float(t.get("lr", 3e-4)),
        weight_decay=float(t.get("weight_decay", 0.01)),
        epochs=int(t.get("epochs", 10)),
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
    return train(
        _expand(d["dataset_npz"]),
        tcfg,
        holdout_families=set(d.get("holdout_families") or []),
        out_dir=out_dir,
        device_str=str(cfg.get("experiment", {}).get("device", "cuda")),
        log=log,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    p = argparse.ArgumentParser(prog="python -m ark_core.training.run")
    p.add_argument("step", choices=["build", "train", "all"])
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="实验配置（默认 <repo>/configs/imitation_v0.yaml）",
    )
    p.add_argument("--report", type=Path, default=None, help="把结果 JSON 写到该路径")
    args = p.parse_args(argv)

    # 行缓冲 + 立即 flush：训练耗时长，输出被缓冲会让进度不可观测（实测踩过）
    def log(msg: str) -> None:
        print(msg, flush=True)

    cfg_path = args.config or (_repo_root() / "configs" / "imitation_v0.yaml")
    cfg = _load_cfg(cfg_path)
    log(f"[cfg] {cfg_path}")

    result: dict[str, Any] = {"config": str(cfg_path)}
    if args.step in ("build", "all"):
        result["build"] = _do_build(cfg, log)
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
