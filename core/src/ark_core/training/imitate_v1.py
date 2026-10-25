"""imitate_v1：带战场状态的行为克隆训练循环（v1，链路冒烟）。

任务：``P(a_t | 关卡静态特征, 编队, a_<t, state_t)`` —— ``state_t`` 为第 t 个动作
之前的战场快照（对齐语义见 ``state_action_data`` / ``policy_v1`` 的 docstring）。

与 v0（``imitate``）的关系
---------------------------
损失（``compute_loss``）、历史右移（``shift_history``）、预 collate 缓存、
按 holdout 选优等机制**原样复用** v0；差异仅在三处：

1. batch 增加 ``state`` 张量（``collate_v1``），前向多喂一路（``forward_batch_v1``）；
2. 划分从「家族留出」改为「**关卡留出**」（真机数据只有 main/camp 两族，
   家族留出会砍掉大半数据；关卡级留出是数据现状下的诚实降级，同章泄漏风险
   在报告中如实标注）；
3. 指标加厚（``MetricsV1``）：type Acc / macro-F1 / 逐类 P/R/F1 / cell/dir/target
   命中率，并输出**同口径常数基线**（见 :func:`constant_baselines`）并排对照。

诚实口径（禁令）
----------------
数据量极小（~1k 状态-动作对），本管线产出**只用于链路冒烟验证，不作能力证据**；
对外只报 holdout 指标，train 档仅作过拟合诊断。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..models.track_a.policy_v1 import TrackAPolicyV1
from ..protocol.action import ACTION_TYPES, DIRECTIONS, GRID_CHANNELS
from .imitate import (
    TrainConfig,
    compute_loss,
    set_seed,
    shift_history,
    to_device,
)
from .imitation_data import MAX_SQUAD, Vocab
from .state_action_data import base_stage_id

_DEPLOY_IDX = ACTION_TYPES.index("deploy")
_DIR_NONE_IDX = DIRECTIONS.index("None")


def load_records_v1(
    npz_path: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, list[str], int, int]:
    """读回 v1 数据集 npz → (记录列表, featvec 矩阵, 词表 token 列表, 状态维数,
    格子通道数)。

    老 npz（v1/v2 三件套）无 ``grid_channels`` 键，按 GRID_CHANNELS=5 回落；
    返回的通道数必须与实际 record 的 cells 末维一致（否则本函数直接报错，
    不把不一致带到训练侧）。
    """
    z = np.load(npz_path, allow_pickle=True)
    grid_channels = int(z["grid_channels"]) if "grid_channels" in z else GRID_CHANNELS
    records = list(z["records"])
    if records:
        actual = int(records[0]["cells"].shape[-1])
        if actual != grid_channels:
            raise ValueError(
                f"{npz_path} 元数据 grid_channels={grid_channels} 与 record "
                f"cells 通道 {actual} 不一致（npz 被混淆，请重新 build）"
            )
    return (
        records,
        z["featvec"],
        [str(x) for x in z["vocab"]],
        int(z["state_dim"]),
        grid_channels,
    )


def split_records(
    records: list[dict[str, Any]], holdout_stages: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按**关卡**（基础 id，``#f#`` 变体随基础关同侧）切成 train / holdout 两档。"""
    train_recs: list[dict[str, Any]] = []
    hold_recs: list[dict[str, Any]] = []
    for r in records:
        if base_stage_id(str(r["stage_id"])) in holdout_stages:
            hold_recs.append(r)
        else:
            train_recs.append(r)
    return train_recs, hold_recs


def collate_v1(
    records: list[dict[str, Any]], featvec: np.ndarray, state_dim: int
) -> dict[str, Any]:
    """把一批变长记录补齐成张量（mask: True=有效）；比 v0 多 ``state`` 一项。

    cells 通道数由 record 自身末维派生（5 = v1 地形口径，8 = v1.1 路线几何），
    不再假定协议常量 GRID_CHANNELS。
    """
    b = len(records)
    n_cells = max(len(r["cells"]) for r in records)
    n_cell_channels = max(int(r["cells"].shape[-1]) for r in records)
    n_spawns = max(r["spawns"].shape[0] for r in records)
    n_squad = max(len(r["squad_idx"]) for r in records)
    t_max = max(len(r["type"]) for r in records)

    arr = dict(
        cells=np.zeros((b, n_cells, n_cell_channels), dtype=np.float32),
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
        state=np.zeros((b, t_max, state_dim), dtype=np.float32),
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
        arr["state"][i, :t] = r["state_feat"]

    return {k: torch.from_numpy(v) for k, v in arr.items()}


def forward_batch_v1(model: nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """在（已迁到目标设备的）batch 上做一次前向。

    动作历史右移一位（BOS），``state`` **不右移**（对齐语义见 policy_v1 docstring）。
    """
    h_type, h_target, h_cell, h_dir = shift_history(batch)
    return model(
        batch["cells"], batch["cell_pos"], batch["cell_mask"],
        batch["spawns"], batch["spawn_mask"],
        batch["squad_feat"], batch["squad_mask"],
        h_type, h_target, h_cell, h_dir,
        batch["state"],
    )


@dataclass
class MetricsV1:
    """评测指标累加器（type 逐类计数 + cell/dir/target 命中 + loss）。

    - ``type_*``：三数组均为长度 ``len(ACTION_TYPES)`` 的计数（预测为 c / 真值为 c /
      且预测也为 c），逐类 P/R/F1 由其导出；
    - ``cell`` 只在 deploy 帧上统计（其余帧无格子语义）；``dir`` 只在「deploy 且
      真值朝向非 None」帧上统计（与 ``compute_loss`` 的掩码口径一致）；``target``
      在全部有效帧上统计（含 UNK 语义：speed 等无目标动作的真值即 UNK 槽）。
    """

    n: int = 0
    loss_sum: float = 0.0
    loss_batches: int = 0
    type_pred: np.ndarray = field(
        default_factory=lambda: np.zeros(len(ACTION_TYPES), dtype=np.int64)
    )
    type_true: np.ndarray = field(
        default_factory=lambda: np.zeros(len(ACTION_TYPES), dtype=np.int64)
    )
    type_tp: np.ndarray = field(
        default_factory=lambda: np.zeros(len(ACTION_TYPES), dtype=np.int64)
    )
    cell_hit: int = 0
    cell_total: int = 0
    dir_hit: int = 0
    dir_total: int = 0
    target_hit: int = 0
    target_total: int = 0

    def add_batch(self, out: dict[str, torch.Tensor], batch: dict[str, Any], loss: float) -> None:
        """累计一个 batch（``out`` 为模型 logits dict）。"""
        lens = batch["lengths"].cpu().numpy()
        gt_type = batch["type"].cpu().numpy()
        p_type = out["type"].argmax(-1).cpu().numpy()
        p_target = out["target"].argmax(-1).cpu().numpy()
        p_cell = out["cell"].argmax(-1).cpu().numpy()
        p_dir = out["dir"].argmax(-1).cpu().numpy()
        gt_target = batch["target"].cpu().numpy()
        gt_cell = batch["cell"].cpu().numpy()
        gt_dir = batch["dir"].cpu().numpy()
        self.n += len(lens)
        self.loss_sum += loss
        self.loss_batches += 1
        for i, t in enumerate(lens):
            t = int(t)
            g, p = gt_type[i, :t], p_type[i, :t]
            for c in range(len(ACTION_TYPES)):
                self.type_pred[c] += int((p == c).sum())
                self.type_true[c] += int((g == c).sum())
                self.type_tp[c] += int(((p == c) & (g == c)).sum())
            deploy = g == _DEPLOY_IDX
            self.cell_hit += int((p_cell[i, :t][deploy] == gt_cell[i, :t][deploy]).sum())
            self.cell_total += int(deploy.sum())
            has_dir = deploy & (gt_dir[i, :t] != _DIR_NONE_IDX)
            self.dir_hit += int((p_dir[i, :t][has_dir] == gt_dir[i, :t][has_dir]).sum())
            self.dir_total += int(has_dir.sum())
            self.target_hit += int((p_target[i, :t] == gt_target[i, :t]).sum())
            self.target_total += t

    def summary(self) -> dict[str, Any]:
        """汇总（分母为 0 时对应率值记 0；macro-F1 只平均 holdout 真值出现过的类）。"""
        per_class: dict[str, dict[str, float | int]] = {}
        f1s: list[float] = []
        for c, name in enumerate(ACTION_TYPES):
            pred_n = int(self.type_pred[c])
            true_n = int(self.type_true[c])
            tp = int(self.type_tp[c])
            p = tp / pred_n if pred_n else 0.0
            r = tp / true_n if true_n else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            if true_n:
                f1s.append(f1)
            per_class[name] = {
                "support": true_n, "pred_n": pred_n, "tp": tp,
                "precision": p, "recall": r, "f1": f1,
            }
        type_total = int(self.type_true.sum())
        return {
            "n_episodes": self.n,
            "n_tokens": type_total,
            "type_acc": int(self.type_tp.sum()) / type_total if type_total else 0.0,
            "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
            "per_class": per_class,
            "cell_acc": self.cell_hit / self.cell_total if self.cell_total else 0.0,
            "cell_total": self.cell_total,
            "dir_acc": self.dir_hit / self.dir_total if self.dir_total else 0.0,
            "dir_total": self.dir_total,
            "target_acc": self.target_hit / self.target_total if self.target_total else 0.0,
            "loss": self.loss_sum / self.loss_batches if self.loss_batches else 0.0,
        }


def constant_baselines(
    train_recs: list[dict[str, Any]], hold_recs: list[dict[str, Any]]
) -> dict[str, Any]:
    """同口径常数基线（常数预测器无需训练，直接在对应帧上结算）。

    - ``always_deploy_type_acc``：恒预测 deploy 在 holdout 的 type_acc
      （= holdout deploy 帧占比）；
    - ``majority_from_train``: 训练侧占比最大的动作类型及其在 holdout 的
      type_acc（恒预测该类）；
    - ``dir_majority_from_train``：训练侧 has_dir 帧上占比最大的朝向，及其在
      holdout has_dir 帧上的命中率（与 ``MetricsV1.dir_acc`` 同口径）；
    - ``uniform_cell_expected_hit``：均匀随机选格的期望命中率
      （holdout 每个 deploy 帧贡献 1/n_cells，取均值）。
    """
    train_type_hist = np.zeros(len(ACTION_TYPES), dtype=np.int64)
    train_dir_hist = np.zeros(len(DIRECTIONS), dtype=np.int64)
    for r in train_recs:
        g = np.asarray(r["type"])
        for c in range(len(ACTION_TYPES)):
            train_type_hist[c] += int((g == c).sum())
        deploy = g == _DEPLOY_IDX
        d = np.asarray(r["dir"])
        has_dir = deploy & (d != _DIR_NONE_IDX)
        for c in range(len(DIRECTIONS)):
            train_dir_hist[c] += int((d[has_dir] == c).sum())

    hold_tokens = sum(int(len(r["type"])) for r in hold_recs)
    hold_dep = 0
    hold_dep_dir = 0
    major_type = int(train_type_hist.argmax())
    hold_major = 0
    major_dir = int(train_dir_hist.argmax())
    hold_major_dir_hit = 0
    cell_uni_sum = 0.0
    for r in hold_recs:
        g = np.asarray(r["type"])
        d = np.asarray(r["dir"])
        deploy = g == _DEPLOY_IDX
        hold_dep += int(deploy.sum())
        hold_major += int((g == major_type).sum())
        has_dir = deploy & (d != _DIR_NONE_IDX)
        hold_dep_dir += int(has_dir.sum())
        hold_major_dir_hit += int((d[has_dir] == major_dir).sum())
        n_cells = max(1, len(r["cells"]))
        cell_uni_sum += float(deploy.sum()) / n_cells

    return {
        "always_deploy_type_acc": hold_dep / hold_tokens if hold_tokens else 0.0,
        "majority_from_train": {
            "type": ACTION_TYPES[major_type],
            "type_acc": hold_major / hold_tokens if hold_tokens else 0.0,
        },
        "dir_majority_from_train": {
            "dir": DIRECTIONS[major_dir],
            "dir_acc": hold_major_dir_hit / hold_dep_dir if hold_dep_dir else 0.0,
        },
        "uniform_cell_expected_hit": cell_uni_sum / hold_dep if hold_dep else 0.0,
    }


@torch.no_grad()
def evaluate_v1(
    model: nn.Module,
    tensors: list[dict[str, Any]],
    cfg: TrainConfig,
    device: torch.device,
    *,
    restore_train: bool = True,
) -> MetricsV1:
    """在**预 collate 的** batch 张量上评测（语义同 v0 ``evaluate``）。"""
    if not tensors:
        return MetricsV1()
    model.eval()
    m = MetricsV1()
    for batch_cpu in tensors:
        batch = to_device(batch_cpu, device)
        out = forward_batch_v1(model, batch)
        loss, _ = compute_loss(out, batch, cfg)
        m.add_batch(out, batch, float(loss.detach()))
    if restore_train:
        model.train()
    return m


def train(
    npz_path: Path,
    cfg: TrainConfig,
    *,
    holdout_stages: set[str],
    out_dir: Path,
    device_str: str = "cuda",
    log: Any = print,
    ckpt_name: str = "track_a_v1.pt",
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """训练主循环；按关卡切成 train / holdout 两档，输出常数基线并排对照。

    ``ckpt_name``：checkpoint 文件名（默认 track_a_v1.pt 保持 v1 产物路径不变；
    v2 实验经配置显式给 track_a_v2.pt，避免产物混淆）。
    ``resume_from``：增量续训的旧 checkpoint 路径（strict 加载，词表/维度不符即报错）。
    """
    set_seed(cfg.seed)
    records, featvec, vocab_tokens, state_dim, grid_channels = load_records_v1(npz_path)
    vocab = Vocab(extra_tokens=vocab_tokens)

    train_recs, hold_recs = split_records(records, holdout_stages)
    log(f"[data] 总 {len(records)} 局 → train {len(train_recs)} / holdout {len(hold_recs)}")
    log(f"[data] holdout 关卡: {sorted(holdout_stages)}（关卡级留出）")
    if not train_recs:
        raise ValueError("训练集为空：请检查 holdout_stages 设置")
    if not hold_recs:
        raise ValueError("holdout 为空：请检查 holdout_stages 设置")

    baselines = constant_baselines(train_recs, hold_recs)
    log(f"[baseline] {baselines}")

    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(train_recs)

    def make_batches(recs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        return [recs[i : i + cfg.batch_size] for i in range(0, len(recs), cfg.batch_size)]

    train_batches = make_batches(train_recs)
    hold_batches = make_batches(hold_recs)

    def prep(batches: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        return [collate_v1(b, featvec, state_dim) for b in batches]

    train_tensors = prep(train_batches)
    hold_tensors = prep(hold_batches)
    train_eval_tensors = train_tensors[: cfg.eval_train_batches]

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = TrackAPolicyV1(
        featvec_dim=int(featvec.shape[1]),
        vocab_size=vocab.size,
        max_squad=MAX_SQUAD,
        state_dim=state_dim,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        n_dec_layers=cfg.n_dec_layers,
        dropout=cfg.dropout,
        grid_channels=grid_channels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(
        f"[model] 参数量 {n_params/1e6:.2f}M  vocab={vocab.size}  "
        f"state_dim={state_dim}  grid_channels={grid_channels}  device={device}"
    )
    if resume_from is not None:
        # 续训：加载旧权重，优化器/调度器重新起步（strict=True，形状不符直接报错）
        blob = torch.load(resume_from, map_location=device, weights_only=False)
        state_dict = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        model.load_state_dict(state_dict, strict=True)
        log(f"[model] 已加载旧权重续训: {resume_from}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_batches) * cfg.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total_steps)

    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, Any] = {}
    best_train_metrics: dict[str, Any] = {}
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        rng.shuffle(train_tensors)
        ep_loss = 0.0
        nb = 0
        for batch_cpu in train_tensors:
            batch = to_device(batch_cpu, device)
            out = forward_batch_v1(model, batch)
            loss, _parts = compute_loss(out, batch, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            ep_loss += loss.detach().item()
            nb += 1

        tm = evaluate_v1(model, train_eval_tensors, cfg, device)
        hm = evaluate_v1(model, hold_tensors, cfg, device, restore_train=False)
        model.train()
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
            f"train type={ts['type_acc']:.3f} f1={ts['macro_f1']:.3f} | "
            f"holdout type={hs['type_acc']:.3f} f1={hs['macro_f1']:.3f} "
            f"cell={hs['cell_acc']:.3f} dir={hs['dir_acc']:.3f} tgt={hs['target_acc']:.3f} | "
            f"{row['elapsed_s']}s"
        )

        # 选优准则：type_acc 主导类污染严重（deploy 占多数），并看 macro-F1
        score = float(hs["type_acc"]) + float(hs["macro_f1"])
        if score > best_score:
            best_score = score
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = dict(hs)
            best_train_metrics = dict(ts)
            log(f"    ↑ 新的最优 holdout（score={score:.4f}）")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / ckpt_name
    torch.save(
        {
            "state_dict": best_state if best_state is not None else model.state_dict(),
            "config": dict(cfg.__dict__),
            "vocab": vocab_tokens,
            "featvec_dim": int(featvec.shape[1]),
            "state_dim": state_dim,
            "grid_channels": grid_channels,
            "n_params": n_params,
            "holdout_stages": sorted(holdout_stages),
            "best_epoch": best_epoch,
            "best_holdout": best_metrics,
        },
        ckpt,
    )
    log(f"[save] {ckpt}（最优 epoch {best_epoch}）")
    log(
        "[诚实口径] 本结果为 ~1k 状态-动作对上的**链路冒烟**验证，"
        "证明 (battle_state, action) 数据→模型→指标链路可走通，不作能力证据。"
    )
    return {
        "n_params": n_params,
        "n_train": len(train_recs),
        "n_holdout": len(hold_recs),
        "n_train_batches": len(train_batches),
        "best_epoch": best_epoch,
        "best_holdout": best_metrics,
        "train_at_best": best_train_metrics,
        "baselines": baselines,
        "history": history,
        "checkpoint": str(ckpt),
    }
