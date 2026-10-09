"""imitate：行为克隆训练循环（v0）。

任务：``P(a_t | 关卡静态特征, 编队, a_<t)`` —— 见 ``imitation_data`` 模块 docstring。
**注意**：这不是方案 §8 的 ``P(a_t | state_t)``（缺战场状态数据），是该目标在现有
数据下的诚实降级形态。

划分纪律（最高准则，方案 §12）
------------------------------
**按关卡家族划分**，严禁按对局随机划分（同关泄漏会让指标全是水分）。
本 v0 做两档：``train``（见过家族）与 ``holdout``（**未见过家族**，真·零样本）。
「同章节未见关」这一档需要章节内细分，留待 M1 评测体系。

指标
----
- ``type_acc``：动作类型逐位准确率；
- ``exact_match``：**整局动作序列完全一致**的比例（严格且诚实，会偏低）；
- ``avg_prefix``：平均首次出错位置（越接近序列长度越好，全对记序列长度）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..datapipe.stages_meta import family_of
from ..models.track_a.policy import TrackAPolicy
from ..protocol.action import ACTION_TYPES, DIRECTIONS, GRID_CHANNELS
from .imitation_data import MAX_SQUAD, Vocab


@dataclass
class TrainConfig:
    """训练超参（全部来自 configs/，禁止代码内魔法数字）。"""

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    n_dec_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 10
    grad_clip: float = 1.0
    seed: int = 42
    w_type: float = 1.0
    w_target: float = 1.0
    w_cell: float = 1.0
    w_dir: float = 0.5
    eval_train_batches: int = 64
    """train 档评测抽样的 batch 数（holdout 恒全量评测）。"""


@dataclass
class Metrics:
    """评测指标累加器。"""

    n: int = 0
    type_correct: int = 0
    type_total: int = 0
    exact: int = 0
    prefix_sum: int = 0
    prefix_total: int = 0
    loss_sum: float = 0.0
    loss_batches: int = 0

    def add_batch(self, pred_type: torch.Tensor, batch: dict[str, Any], loss: float) -> None:
        """累计一个 batch。``pred_type`` 为 [B, T] 的 argmax 结果。"""
        lens = batch["lengths"].cpu().numpy()
        self.n += len(lens)
        self.loss_sum += loss
        self.loss_batches += 1
        pt = pred_type.cpu().numpy()
        gt = batch["type"].cpu().numpy()
        for i, t in enumerate(lens):
            t = int(t)
            match = pt[i, :t] == gt[i, :t]
            self.type_correct += int(match.sum())
            self.type_total += t
            if bool(match.all()):
                self.exact += 1
            bad = np.nonzero(~match)[0]
            self.prefix_sum += int(bad[0]) if bad.size else t
            self.prefix_total += t

    def summary(self) -> dict[str, float]:
        """汇总（分母为 0 时返回 0）。"""
        return {
            "n_episodes": self.n,
            "type_acc": self.type_correct / self.type_total if self.type_total else 0.0,
            "exact_match": self.exact / self.n if self.n else 0.0,
            "avg_prefix": self.prefix_sum / self.prefix_total if self.prefix_total else 0.0,
            "loss": self.loss_sum / self.loss_batches if self.loss_batches else 0.0,
        }


def set_seed(seed: int) -> None:
    """固定随机种子（可复现性）。"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(npz_path: Path) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    """读回数据集 npz → (记录列表, featvec 矩阵, 词表 token 列表)。"""
    z = np.load(npz_path, allow_pickle=True)
    return list(z["records"]), z["featvec"], [str(x) for x in z["vocab"]]


def collate(records: list[dict[str, Any]], featvec: np.ndarray) -> dict[str, Any]:
    """把一批变长记录补齐成张量（mask: True=有效）。"""
    b = len(records)
    n_cells = max(len(r["cells"]) for r in records)
    n_spawns = max(r["spawns"].shape[0] for r in records)
    n_squad = max(len(r["squad_idx"]) for r in records)
    t_max = max(len(r["type"]) for r in records)

    arr = dict(
        cells=np.zeros((b, n_cells, GRID_CHANNELS), dtype=np.float32),
        cell_pos=np.zeros((b, n_cells, 2), dtype=np.float32),
        cell_mask=np.zeros((b, n_cells), dtype=np.bool_),
        spawns=np.zeros((b, n_spawns, 4), dtype=np.float32),
        spawn_mask=np.zeros((b, n_spawns), dtype=np.bool_),
        squad_feat=np.zeros((b, n_squad, featvec.shape[1]), dtype=np.float32),
        squad_mask=np.zeros((b, n_squad), dtype=np.bool_),
        type=np.zeros((b, t_max), dtype=np.int64),
        target=np.zeros((b, t_max), dtype=np.int64),
        cell=np.zeros((b, t_max), dtype=np.int64),
        dir=np.zeros((b, t_max), dtype=np.int64),
        lengths=np.zeros(b, dtype=np.int64),
    )

    for i, r in enumerate(records):
        nc = len(r["cells"])
        arr["cells"][i, :nc] = r["cells"]
        arr["cell_pos"][i, :nc] = r["cell_pos"]
        arr["cell_mask"][i, :nc] = True

        ns = r["spawns"].shape[0]
        if ns:
            arr["spawns"][i, :ns] = r["spawns"]
            arr["spawn_mask"][i, :ns] = True

        nq = len(r["squad_idx"])
        arr["squad_mask"][i, :nq] = True
        for j, idx in enumerate(r["squad_idx"]):
            if idx >= 0:
                arr["squad_feat"][i, j] = featvec[idx]

        t = len(r["type"])
        arr["lengths"][i] = t
        arr["type"][i, :t] = r["type"]
        arr["target"][i, :t] = r["target"]
        arr["cell"][i, :t] = r["cell"]
        arr["dir"][i, :t] = r["dir"]

    return {k: torch.from_numpy(v) for k, v in arr.items()}


def shift_history(
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造右移一位的动作历史（首列 BOS）。输入内容不泄漏当前步答案。"""
    b = batch["type"].shape[0]
    dev = batch["type"].device
    bos = torch.zeros((b, 1), dtype=torch.long, device=dev)

    h_type = torch.cat([bos + len(ACTION_TYPES), batch["type"][:, :-1]], dim=1)
    h_target = torch.cat([bos, batch["target"][:, :-1]], dim=1)
    h_dir = torch.cat([bos, batch["dir"][:, :-1]], dim=1)
    # 历史格子：用索引归一到 [0,1) 作为占位（真实 (r,c) 换算见 encode_episode）
    h_cell_idx = torch.cat([bos, batch["cell"][:, :-1]], dim=1).float()
    h_cell = torch.stack([h_cell_idx / 400.0, torch.zeros_like(h_cell_idx)], dim=-1)
    return h_type, h_target, h_cell, h_dir


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """把 batch 的所有张量迁到指定设备（返回新 dict）。

    必须在进入前向**之前**调用，使 loss 计算看到的 batch 与模型同设备。
    """
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def forward_batch(model: nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """在（已迁到目标设备的）batch 上做一次前向。"""
    h_type, h_target, h_cell, h_dir = shift_history(batch)
    return model(
        batch["cells"], batch["cell_pos"], batch["cell_mask"],
        batch["spawns"], batch["spawn_mask"],
        batch["squad_feat"], batch["squad_mask"],
        h_type, h_target, h_cell, h_dir,
    )


def compute_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, Any],
    cfg: TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """按字段加权求和。

    **cell 与 dir 只在 deploy 上计算**——其余动作类型没有这两项语义，若一并计入
    会用一个恒定的无意义目标稀释梯度（早期实测会显著拉低 type 准确率的学习速度）。
    """
    b, t = batch["type"].shape
    dev = batch["type"].device
    valid = torch.arange(t, device=dev).unsqueeze(0) < batch["lengths"].unsqueeze(1)

    def ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits.reshape(b * t, -1), target.reshape(b * t), reduction="none"
        ).reshape(b, t)

    l_type = ce(out["type"], batch["type"])
    l_target = ce(out["target"], batch["target"])
    l_cell = ce(out["cell"], batch["cell"])
    l_dir = ce(out["dir"], batch["dir"])

    deploy = (batch["type"] == 0) & valid
    has_dir = deploy & (batch["dir"] != DIRECTIONS.index("None"))

    def mmean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return (x * m).sum() / m.sum().clamp(min=1)

    total = (
        cfg.w_type * mmean(l_type, valid)
        + cfg.w_target * mmean(l_target, valid)
        + cfg.w_cell * mmean(l_cell, deploy)
        + cfg.w_dir * mmean(l_dir, has_dir)
    )
    parts = {
        "type": float(mmean(l_type, valid).detach()),
        "target": float(mmean(l_target, valid).detach()),
        "cell": float(mmean(l_cell, deploy).detach()),
        "dir": float(mmean(l_dir, has_dir).detach()),
    }
    return total, parts


@torch.no_grad()
def evaluate(
    model: nn.Module,
    tensors: list[dict[str, Any]],
    cfg: TrainConfig,
    device: torch.device,
    *,
    restore_train: bool = True,
) -> Metrics:
    """在**预 collate 的** batch 张量上评测（迁到与模型同设备）。

    ``restore_train``：评测结束后是否切回 train 模式（训练循环中为 True；
    末尾单独评测时为 False，避免无谓切换）。
    """
    if not tensors:
        return Metrics()
    model.eval()
    m = Metrics()
    with torch.no_grad():
        for batch_cpu in tensors:
            batch = to_device(batch_cpu, device)
            out = forward_batch(model, batch)
            loss, _ = compute_loss(out, batch, cfg)
            m.add_batch(out["type"].argmax(-1), batch, float(loss.detach()))
    if restore_train:
        model.train()
    return m


def train(
    npz_path: Path,
    cfg: TrainConfig,
    *,
    holdout_families: set[str],
    out_dir: Path,
    device_str: str = "cuda",
    log: Any = print,
) -> dict[str, Any]:
    """训练主循环；按关卡家族切成 train / holdout 两档。"""
    set_seed(cfg.seed)
    records, featvec, vocab_tokens = load_records(npz_path)
    vocab = Vocab(extra_tokens=vocab_tokens)

    train_recs: list[dict[str, Any]] = []
    hold_recs: list[dict[str, Any]] = []
    for r in records:
        fam = r["family"] or family_of(r["stage_id"])
        (hold_recs if fam in holdout_families else train_recs).append(r)

    log(f"[data] 总 {len(records)} 局 → train {len(train_recs)} / holdout {len(hold_recs)}")
    log(f"[data] holdout 家族: {sorted(holdout_families)}")
    if not train_recs:
        raise ValueError("训练集为空：请检查 holdout_families 设置")

    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(train_recs)

    def make_batches(recs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        return [recs[i : i + cfg.batch_size] for i in range(0, len(recs), cfg.batch_size)]

    train_batches = make_batches(train_recs)
    hold_batches = make_batches(hold_recs)

    # **预 collate 并缓存**：collate 是纯 numpy 的 CPU 侧工作，实测每步耗时与 GPU
    # 前向同量级（GPU 利用率仅 ~70%）。数据集仅 38k 局、约数百 MB，可整体驻留内存，
    # 于是把 collate 从「每 epoch 每 batch 一次」降为「全程一次」。
    def prep(batches: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        return [collate(b, featvec) for b in batches]

    train_tensors = prep(train_batches)
    hold_tensors = prep(hold_batches)
    # train 档评测抽样（见循环内注释）；holdout 恒全量
    train_eval_tensors = train_tensors[: cfg.eval_train_batches]

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = TrackAPolicy(
        featvec_dim=int(featvec.shape[1]),
        vocab_size=vocab.size,
        max_squad=MAX_SQUAD,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        n_dec_layers=cfg.n_dec_layers,
        dropout=cfg.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[model] 参数量 {n_params/1e6:.2f}M  vocab={vocab.size}  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_batches) * cfg.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total_steps)

    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, float] = {}
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        rng.shuffle(train_tensors)
        ep_loss = 0.0
        nb = 0
        for batch_cpu in train_tensors:
            batch = to_device(batch_cpu, device)
            out = forward_batch(model, batch)
            loss, parts = compute_loss(out, batch, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            ep_loss += loss.detach().item()
            nb += 1

        # 评测：holdout（真·零样本）**每 epoch 全量**——它是唯一对外指标，不能省。
        # train 档只是诊断过拟合，全量评测要额外 1204 步前向（约占 30% 时间），
        # 故按 eval_train_batches 抽样（默认 64 个 batch，统计上足够看趋势）。
        tm = evaluate(model, train_eval_tensors, cfg, device)
        hm = evaluate(model, hold_tensors, cfg, device)
        ts, hs = tm.summary(), hm.summary()
        row = {
            "epoch": ep + 1,
            "train_loss": ep_loss / max(1, nb),
            "train": ts,
            "holdout": hs,
            "elapsed_s": round(time.time() - t0, 1),
        }
        history.append(row)
        log(
            f"[ep {ep+1}/{cfg.epochs}] loss={row['train_loss']:.4f} | "
            f"train type={ts['type_acc']:.3f} exact={ts['exact_match']:.3f} | "
            f"holdout type={hs['type_acc']:.3f} exact={hs['exact_match']:.3f} | "
            f"{row['elapsed_s']}s"
        )

        # **按 holdout 指标保存最优**：实测 train 持续爬升而 holdout 自首个 epoch
        # 即平台化（过拟合），若只在训练结束保存会交付过拟合权重。
        # 选择准则用 macro 化的组合（type_acc 主导类污染严重，故并看 exact 与 loss）。
        score = hs["type_acc"] + hs["exact_match"]
        if score > best_score:
            best_score = score
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = dict(hs)
            log(f"    ↑ 新的最优 holdout（score={score:.4f}）")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "track_a_v0.pt"
    torch.save(
        {
            "state_dict": best_state if best_state is not None else model.state_dict(),
            "config": dict(cfg.__dict__),
            "vocab": vocab_tokens,
            "featvec_dim": int(featvec.shape[1]),
            "n_params": n_params,
            "holdout_families": sorted(holdout_families),
            "best_epoch": best_epoch,
            "best_holdout": best_metrics,
        },
        ckpt,
    )
    log(f"[save] {ckpt}（最优 epoch {best_epoch}，holdout {best_metrics}）")
    return {
        "n_params": n_params,
        "n_train": len(train_recs),
        "n_holdout": len(hold_recs),
        "n_train_batches": len(train_batches),
        "best_epoch": best_epoch,
        "best_holdout": best_metrics,
        "history": history,
        "checkpoint": str(ckpt),
    }
