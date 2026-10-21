"""K 折关卡轮换留出（v3 评测口径硬化）。

动机（诚实口径）
----------------
v1/v2 的 holdout 固定 4 关（v2 清洗后 14 局 / 79 tokens），指标随单帧抖动，
「提升/回退」分不清是信号还是噪声。本模块把同一份 npz 按**基础关卡 id**切成
k 折**不重叠关卡组**，每折拿该组当 holdout、其余当 train，同超参训 k 轮，
汇总各指标的 **均值 ± 标准差** 进 ``metrics_summary.json``。

口径约束
--------
- 划分单位是基础关卡 id（``#f#`` 变体随基础关同折，与 v1/v2 留出语义一致）；
- 折内 holdout 为空直接判失败（folds 由本模块构造，正常不会发生）；
- ``dir_loss`` 旋钮（``full | weighted | masked | eval_only``）只改 DIR 头的
  训练损失权重，四头指标照常全量评测——这是对「DIR 标签疑似脏」的处置开关：

  ============  ==========================================================
  取值          语义
  ============  ==========================================================
  ``full``      DIR 损失按 ``w_dir`` 全量计入（v1/v2 行为）
  ``weighted``  DIR 损失按 ``dir_loss_weighted`` 降权计入
  ``masked``    DIR 损失完全屏蔽（w_dir=0）；dir_acc 仍逐折上报作记录
  ``eval_only`` 同 masked 的损失语义；仅命名上强调「训练不碰、只读指标」
  ============  ==========================================================

  注：masked 与 eval_only 当前实现等价（都是 w_dir=0 + 指标照报），语义区别
  留给将来「masked 连指标也不汇报」的口径；如实记录，不夸大差异。
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .imitate import TrainConfig
from .imitate_v1 import train
from .state_action_data import base_stage_id

DIR_LOSS_MODES: tuple[str, ...] = ("full", "weighted", "masked", "eval_only")
"""dir_loss 合法取值（语义见模块 docstring 的表）。"""

# 汇总时做均值±std 的指标路径（best_holdout 摘要内的键名）
_SUMMARY_KEYS: tuple[str, ...] = ("type_acc", "macro_f1", "cell_acc", "dir_acc", "target_acc")


def resolve_w_dir(mode: str, w_dir: float, w_dir_weighted: float) -> float:
    """``dir_loss`` 模式 → 实际 DIR 损失权重（未知模式报错，不静默）。"""
    if mode == "full":
        return w_dir
    if mode == "weighted":
        return w_dir_weighted
    if mode in ("masked", "eval_only"):
        return 0.0
    raise ValueError(f"未知 dir_loss 模式: {mode!r}（合法: {DIR_LOSS_MODES}）")


def make_folds(
    stage_ids: list[str], k: int, seed: int
) -> list[list[str]]:
    """把（可含重复的）关卡 id 列表按**基础关**切成 k 折不重叠组。

    确定性：先按基础关名排序，再用 ``seed`` 洗牌，最后按「当前局数最少的折」
    贪心装入——同种子同列表必然同结果，不许随机漂移。
    """
    counts: Counter = Counter(base_stage_id(s) for s in stage_ids)
    if not counts:
        raise ValueError("records 为空，无法分折")
    if k < 2:
        raise ValueError(f"k 必须 >= 2，收到 {k}")
    groups = sorted(counts)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    shuffled = [groups[i] for i in order]
    # 大组优先装，贪心均衡各折局数
    shuffled.sort(key=lambda g: (-counts[g], g))
    folds: list[list[str]] = [[] for _ in range(k)]
    fold_n = [0] * k
    for g in shuffled:
        i = min(range(k), key=lambda j: (fold_n[j], j))
        folds[i].append(g)
        fold_n[i] += counts[g]
    return [sorted(f) for f in folds]


def summarize_fold_metrics(fold_summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """把各折 best_holdout 摘要汇总成 均值±std（样本 std，ddof=1；附逐折值与折数）。"""
    out: dict[str, dict[str, Any]] = {}
    for key in _SUMMARY_KEYS:
        vals = np.asarray([float(s.get(key, 0.0)) for s in fold_summaries], dtype=np.float64)
        out[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "values": [float(v) for v in vals],
            "n_folds": int(len(vals)),
        }
    return out


def run_kfold(
    npz_path: Path,
    cfg: TrainConfig,
    *,
    k: int,
    seed: int,
    out_dir: Path,
    device_str: str,
    arms: list[dict[str, Any]],
    w_dir_weighted: float,
    log: Any = print,
) -> dict[str, Any]:
    """逐臂逐折训练并汇总。``arms`` 元素形如 ``{"name": "full", "dir_loss": "full"}``。

    产物布局：``out_dir/<arm.name>/fold_<i>/{track_a_fold.pt, train_report.json}``；
    汇总由调用方写 ``out_dir/metrics_summary.json``。per-epoch 行日志被折叠
    （只保留折级摘要行），否则 k×arms×epochs 行级噪音淹没关键信息。
    """
    from .imitate_v1 import load_records_v1  # 延迟导入，保持模块可被无 torch 环境加载文档

    records, _featvec, _vocab_tokens, _state_dim = load_records_v1(npz_path)
    folds = make_folds([str(r["stage_id"]) for r in records], k, seed)
    log(f"[kfold] {len(records)} 局 → k={k} 折；各折 holdout 关卡数: "
        f"{[len(f) for f in folds]}")

    def fold_log(msg: str) -> None:
        # 折内 epoch 行逐条太吵，降到折级：只放行 [data]/[model]/[baseline]/[save] 等
        if msg.startswith(("[ep ", "    ↑")):
            return
        log(msg)

    result: dict[str, Any] = {
        "k": k,
        "seed": seed,
        "npz": str(npz_path),
        "folds": [{"fold": i, "holdout_stages": f} for i, f in enumerate(folds)],
        "arms": {},
    }
    t0 = time.time()
    for arm in arms:
        arm_name = str(arm["name"])
        mode = str(arm["dir_loss"])
        w_dir = resolve_w_dir(mode, cfg.w_dir, w_dir_weighted)
        arm_cfg = replace(cfg, w_dir=w_dir)
        log(f"[kfold] === arm {arm_name}（dir_loss={mode}, w_dir={w_dir}）===")
        fold_reports: list[dict[str, Any]] = []
        for i, hold_stages in enumerate(folds):
            fold_dir = out_dir / arm_name / f"fold_{i}"
            rep = train(
                npz_path,
                arm_cfg,
                holdout_stages=set(hold_stages),
                out_dir=fold_dir,
                device_str=device_str,
                log=fold_log,
                ckpt_name="track_a_fold.pt",
            )
            h = rep["best_holdout"]
            fold_reports.append(rep)
            with (fold_dir / "train_report.json").open("w", encoding="utf-8") as f:
                json.dump(rep, f, ensure_ascii=False, indent=2)
            log(
                f"[kfold] {arm_name}/fold_{i} hold={rep['n_holdout']}局/{h['n_tokens']}tok "
                f"type={h['type_acc']:.3f} f1={h['macro_f1']:.3f} "
                f"cell={h['cell_acc']:.3f} dir={h['dir_acc']:.3f} tgt={h['target_acc']:.3f} "
                f"(best_ep {rep['best_epoch']})"
            )
        summaries = [r["best_holdout"] for r in fold_reports]
        result["arms"][arm_name] = {
            "dir_loss": mode,
            "w_dir": w_dir,
            "fold_metrics": [
                {
                    "fold": i,
                    "n_train": r["n_train"],
                    "n_holdout": r["n_holdout"],
                    "best_epoch": r["best_epoch"],
                    "best_holdout": r["best_holdout"],
                    "baselines": r["baselines"],
                }
                for i, r in enumerate(fold_reports)
            ],
            "mean_std": summarize_fold_metrics(summaries),
        }
        ms = result["arms"][arm_name]["mean_std"]
        parts = "  ".join(
            f"{key}={ms[key]['mean']:.3f}±{ms[key]['std']:.3f}" for key in _SUMMARY_KEYS
        )
        log(f"[kfold] arm {arm_name} 汇总: {parts}")
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result
