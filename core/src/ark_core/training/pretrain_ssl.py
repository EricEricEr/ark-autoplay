"""pretrain_ssl：自监督预训练 v0（掩码实体建模，BERT 式）。

定位（方案 §11，轨道 A 必做）
------------------------------
离线轨迹 ``imitation_v0.npz``（38,526 局：关卡 grid/spawns + 编队 + 动作序列，
**无战场状态**）上做掩码实体重构，预训练出的权重按键名/形状直接初始化
``models/track_a/policy.py`` 的 ``TrackAPolicy``（``EntityEncoder`` 原样吻合），
并经键级匹配加载进 ``policy_v1.TrackAPolicyV1`` 做下游微调烟测。

预置任务（pretext）
-------------------
每个 episode 的四组实体（格子 / 出怪 / 编队 / 动作历史）统一抽
``mask_rate``（默认 15%）实体做重构目标；被抽中的实体按 80/10/10 处理：

- **80%** 换成**可学习 mask token**（每组一个 ``nn.Parameter(d_model)``，
  在投影/嵌入**之后**叠加，故 policy 原有嵌入表形状不变、键级可移植）；
- **10%** 换成**同组随机实体**（格子/出怪/格索引取同局随机有效实体，
  编队取随机 featvec 行，type/target/dir 取均匀随机类）；
- **10%** 保留原样（仍计入重构 loss）。

重构头：
- 数值通道回归（huber）：格子 ``5ch grid + 2ch 归一化 (r,c) = 7``、出怪 ``4ch``；
- 离散分类（CE）：编队槽位的 featvec 行号（458 类）、动作历史四头
  （type/target/dir/cell 指针，与 policy 同口径——cell/dir 只在 gold 为
  deploy / dir≠None 的帧上计 loss，照抄 ``imitate.compute_loss`` 的掩码语义）。

因果与安全
----------
动作历史沿用 policy 的**右移 BOS**约定：被遮的是输入侧实体，重构发生在
``out[j]``（只读 ≤j 的历史输入），不泄漏当前步答案；同 v1 一样无战场状态，
grid/spawns/编队全是动作发生前已知的静态信息，掩码重构不引入标签泄漏。

诚实口径
--------
预训练指标高 ≠ 下游能力高（重构静态地形近似查表）。本 v0 只回答两件事：
1）掩码重构在 holdout（v0 的 holdout_families，家族级零样本）上是否**显著赢过
随机初始化与恒预测基线**（证明编码器学到了可迁移的实体结构）；
2）预训练初始化在 v2 npz 上跑 **3 epoch** 的早期指标，与 v2 随机初始化
同 epoch 历史并排（只出早期信号，不下能力结论）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..models.track_a.policy import SPAWN_IN, TrackAPolicy
from ..protocol.action import ACTION_TYPES, DIRECTIONS, GRID_CHANNELS
from .imitate import (
    TrainConfig,
    collate,
    load_records,
    set_seed,
    shift_history,
    to_device,
)
from .imitate_v1 import train as train_downstream
from .imitation_data import MAX_SQUAD, Vocab

CELL_TOKEN_DIM = GRID_CHANNELS + 2
"""格子 token 数值维数（5 通道 + 归一化 (r,c)），回归头宽。"""


@dataclass
class PretextConfig:
    """预训练超参（全部来自 configs/，禁止代码内魔法数字）。"""

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    n_dec_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 12
    grad_clip: float = 1.0
    seed: int = 42
    limit_train: int = 16000
    """训练侧截断（4060 8G 单轮 1~2h 封顶的成本阀；holdout 恒全量评测）。"""
    holdout_eval_batches: int = 64
    mask_rate: float = 0.15
    ratio_mask: float = 0.8
    """被抽中实体换成可学习 mask token 的比例（其余 = 随机 / 保留各半残）。"""
    ratio_random: float = 0.1
    huber_delta: float = 1.0
    w_recon_cell: float = 1.0
    w_recon_spawn: float = 1.0
    w_cls_squad: float = 1.0
    w_act_type: float = 1.0
    w_act_target: float = 1.0
    w_act_cell: float = 1.0
    w_act_dir: float = 0.5


# ---------------------------------------------------------------- 掩码采样


def _draw_sel(
    valid: torch.Tensor, cfg: PretextConfig, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按 mask_rate 在有效实体中抽重构目标 → (换 mask token, 换随机, 保留原样)。

    返回三个 bool 张量（与 valid 同形，CPU）；loss 位置 = 三者并集。
    """
    u = torch.rand(valid.shape, generator=gen)
    sel = (u < cfg.mask_rate) & valid
    v = torch.rand(valid.shape, generator=gen)
    m_mask = sel & (v < cfg.ratio_mask)
    m_rand = sel & (v >= cfg.ratio_mask) & (v < cfg.ratio_mask + cfg.ratio_random)
    m_keep = sel & ~(m_mask | m_rand)
    return m_mask, m_rand, m_keep


def _rand_fill_rows(
    src_valid: torch.Tensor, m_rand: torch.Tensor, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """同组随机实体替换（10% 分支通用）：对 mask 位取同 episode 内的随机有效行。

    返回 ``(row_i, dst_j, src_j)`` 三个索引张量，调用方据此 gather 替换。
    batch 级小循环（B≤batch_size）可接受，避免复杂的向量化。
    """
    rows: list[int] = []
    dsts: list[int] = []
    srcs: list[int] = []
    for i in range(src_valid.shape[0]):
        vi = src_valid[i].nonzero(as_tuple=False).flatten()
        mj = m_rand[i].nonzero(as_tuple=False).flatten()
        if len(vi) == 0 or len(mj) == 0:
            continue
        j = torch.randint(len(vi), (len(mj),), generator=gen)
        rows.extend([i] * len(mj))
        dsts.extend(int(x) for x in mj)
        srcs.extend(int(vi[x]) for x in j)
    t = torch.tensor
    return (
        t(rows, dtype=torch.long),
        t(dsts, dtype=torch.long),
        t(srcs, dtype=torch.long),
    )


def mask_batch(
    batch: dict[str, torch.Tensor],
    featvec: torch.Tensor,
    n_targets: int,
    cfg: PretextConfig,
    gen: torch.Generator,
) -> dict[str, Any]:
    """对 collate 后的 batch 做掩码采样与替换（纯输入侧操作，device 无关）。

    返回 dict：
    - ``inputs``：被替换后的输入张量（cells/cell_pos/spawns/squad_feat/type/target/cell/dir）；
    - ``m80_*``：应换成可学习 mask token 的位置（传 forward_ssl 做叠加）；
    - ``sel_*``：重构 loss 的位置（80/10/10 并集）；
    - ``gold_*``：重构真值（数值直接来自原 batch，离散为原索引）。
    """
    dev = batch["cells"].device
    inputs: dict[str, torch.Tensor] = {}
    out: dict[str, Any] = {"inputs": inputs}

    # ---- 格子（回归：grid + pos）----
    m80, mrand, mkeep = _draw_sel(batch["cell_mask"].cpu(), cfg, gen)
    cells_m = batch["cells"].clone()
    pos_m = batch["cell_pos"].clone()
    ri, dj, sj = _rand_fill_rows(batch["cell_mask"].cpu(), mrand, gen)
    if len(ri):
        cells_m[ri.to(dev), dj.to(dev)] = batch["cells"][ri.to(dev), sj.to(dev)]
        pos_m[ri.to(dev), dj.to(dev)] = batch["cell_pos"][ri.to(dev), sj.to(dev)]
    inputs["cells"], inputs["cell_pos"] = cells_m, pos_m
    out["m80_cell"] = m80.to(dev)
    out["sel_cell"] = (m80 | mrand | mkeep).to(dev)

    # ---- 出怪（回归：4ch）----
    m80, mrand, mkeep = _draw_sel(batch["spawn_mask"].cpu(), cfg, gen)
    spawns_m = batch["spawns"].clone()
    ri, dj, sj = _rand_fill_rows(batch["spawn_mask"].cpu(), mrand, gen)
    if len(ri):
        spawns_m[ri.to(dev), dj.to(dev)] = batch["spawns"][ri.to(dev), sj.to(dev)]
    inputs["spawns"] = spawns_m
    out["m80_spawn"] = m80.to(dev)
    out["sel_spawn"] = (m80 | mrand | mkeep).to(dev)

    # ---- 编队（分类：featvec 行号）；未知槽（-1 零向量）不参与掩码 ----
    squad_valid = batch["squad_mask"].cpu() & (batch["squad_gold_idx"].cpu() >= 0)
    m80, mrand, mkeep = _draw_sel(squad_valid, cfg, gen)
    squad_feat_m = batch["squad_feat"].clone()
    mj = mrand.nonzero(as_tuple=False)
    if len(mj):
        rid = torch.randint(len(featvec), (len(mj),), generator=gen)
        squad_feat_m[mj[:, 0].to(dev), mj[:, 1].to(dev)] = featvec[rid.to(dev)]
    inputs["squad_feat"] = squad_feat_m
    out["m80_squad"] = m80.to(dev)
    out["sel_squad"] = (m80 | mrand | mkeep).to(dev)
    out["gold_squad"] = batch["squad_gold_idx"]

    # ---- 动作历史（联合遮整个动作实体；输入侧右移约定不变）----
    # 可遮位置 j：其输入列 j+1 必须落在序列内 → j <= len-2（末位动作永不遮，见 docstring）
    t_max = batch["type"].shape[1]
    jj = torch.arange(t_max).unsqueeze(0)
    act_valid = jj < (batch["lengths"].cpu() - 1).unsqueeze(1)
    m80, mrand, mkeep = _draw_sel(act_valid, cfg, gen)
    type_m = batch["type"].clone()
    target_m = batch["target"].clone()
    dir_m = batch["dir"].clone()
    cellmid_m = batch["cell"].clone()
    mj = mrand.nonzero(as_tuple=False)
    if len(mj):
        b_, j_ = mj[:, 0], mj[:, 1]
        type_m[b_.to(dev), j_.to(dev)] = torch.randint(
            len(ACTION_TYPES), (len(b_),), generator=gen
        ).to(dev)
        target_m[b_.to(dev), j_.to(dev)] = torch.randint(
            n_targets, (len(b_),), generator=gen
        ).to(dev)
        dir_m[b_.to(dev), j_.to(dev)] = torch.randint(
            len(DIRECTIONS), (len(b_),), generator=gen
        ).to(dev)
        # 动作 cell 通道的 10% 随机替换：替换值 = 同局随机有效格子的「格子指针索引」
        # （两个轴长度不同：动作轴 T vs 格子轴 n_cells，混用索引会越界，实测踩过）
        for bi, ji in mj.tolist():
            vc = batch["cell_mask"][bi].nonzero(as_tuple=False).flatten()
            if len(vc) == 0:
                continue
            cellmid_m[bi, ji] = int(vc[int(torch.randint(len(vc), (1,), generator=gen))])
    inputs["type"], inputs["target"] = type_m, target_m
    inputs["dir"], inputs["cell"] = dir_m, cellmid_m
    out["m80_hist"] = m80.to(dev)
    out["sel_hist"] = (m80 | mrand | mkeep).to(dev)
    return out


# ---------------------------------------------------------------- 模型


class SSLMaskedPolicy(TrackAPolicy):
    """掩码实体建模版 track_a（**键级兼容** ``TrackAPolicy``）。

    只增加：每组一个可学习 mask token（d_model，投影/嵌入后叠加）与三个重构头
    （格子回归 / 出怪回归 / 编队分类）——policy 原有全部参数名与形状不变，
    ``state_dict`` 可按键名+形状直接初始化 ``TrackAPolicy`` / ``TrackAPolicyV1``。
    """

    def __init__(self, *args: Any, n_featvec_rows: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        d = self.d_model
        zero = lambda: nn.Parameter(torch.zeros(d))  # noqa: E731
        self.mask_cell = zero()
        self.mask_spawn = zero()
        self.mask_squad = zero()
        self.mask_hist_type = zero()
        self.mask_hist_target = zero()
        self.mask_hist_dir = zero()
        self.mask_hist_cell = zero()
        self.cell_recon = nn.Linear(d, CELL_TOKEN_DIM)
        self.spawn_recon = nn.Linear(d, SPAWN_IN)
        self.squad_cls = nn.Linear(d, n_featvec_rows)

    def forward_ssl(
        self,
        cells: torch.Tensor,
        cell_pos: torch.Tensor,
        cell_mask: torch.Tensor,
        spawns: torch.Tensor,
        spawn_mask: torch.Tensor,
        squad_feat: torch.Tensor,
        squad_mask: torch.Tensor,
        hist_type: torch.Tensor,
        hist_target: torch.Tensor,
        hist_cell: torch.Tensor,
        hist_dir: torch.Tensor,
        m80_cell: torch.Tensor,
        m80_spawn: torch.Tensor,
        m80_squad: torch.Tensor,
        m80_hist_in: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """预置前向。``m80_*`` 为叠加可学习 mask token 的位置（其余输入已被
        ``mask_batch`` 做完 80/10/10 的值侧替换）。

        编码器逐行逻辑与 ``EntityEncoder.forward`` 相同（多三步 mask 叠加）；
        动作历史沿用右移 BOS（``hist_*`` 已由 ``shift_history`` 处理）。
        """
        enc_mod = self.encoder
        cell_tok = enc_mod.cell_proj(torch.cat([cells, cell_pos], dim=-1)) + (
            enc_mod.type_emb.weight[0]
        )
        spawn_tok = enc_mod.spawn_proj(spawns) + enc_mod.type_emb.weight[1]
        squad_tok = self.op_proj(squad_feat) + enc_mod.type_emb.weight[2]
        cell_tok = torch.where(m80_cell.unsqueeze(-1), self.mask_cell, cell_tok)
        spawn_tok = torch.where(m80_spawn.unsqueeze(-1), self.mask_spawn, spawn_tok)
        squad_tok = torch.where(m80_squad.unsqueeze(-1), self.mask_squad, squad_tok)

        seq = enc_mod.norm_in(torch.cat([cell_tok, spawn_tok, squad_tok], dim=1))
        valid = torch.cat([cell_mask, spawn_mask, squad_mask], dim=1)
        enc = enc_mod.encoder(seq, src_key_padding_mask=~valid)
        n_cells = cells.shape[1]
        n_spawns = spawns.shape[1]

        h = (
            torch.where(m80_hist_in.unsqueeze(-1), self.mask_hist_type, self.hist_type(hist_type))
            + torch.where(
                m80_hist_in.unsqueeze(-1), self.mask_hist_target, self.hist_target(hist_target)
            )
            + torch.where(m80_hist_in.unsqueeze(-1), self.mask_hist_dir, self.hist_dir(hist_dir))
            + torch.where(
                m80_hist_in.unsqueeze(-1), self.mask_hist_cell, self.hist_cell(hist_cell)
            )
        )
        h = self.hist_norm(h)
        t = hist_type.shape[1]
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=h.device), diagonal=1)
        dec = self.decoder(h, mask=causal)

        cell_enc = enc[:, :n_cells, :]
        q = self.cell_query(dec)
        k = self.cell_key(cell_enc)
        cell_logits = torch.einsum("btd,bsd->bts", q, k) / (self.d_model**0.5)
        # 与 policy.forward 同口径：padding 格/编队 padding 槽置 -inf，评测指标可对比
        cell_logits = cell_logits.masked_fill(~cell_mask.unsqueeze(1), float("-inf"))
        target_logits = self.head_target(dec)
        squad_start = 1 + self.vocab_size
        if self.max_squad:
            slot_valid = squad_mask[:, : self.max_squad]
            pad = torch.ones(
                target_logits.shape[:2] + (self.max_squad,), dtype=torch.bool, device=dec.device
            )
            pad[:, :, : slot_valid.shape[1]] = slot_valid.unsqueeze(1)
            target_logits = target_logits.clone()
            target_logits[:, :, squad_start:] = target_logits[:, :, squad_start:].masked_fill(
                ~pad, float("-inf")
            )
        return {
            "cell_recon": self.cell_recon(enc[:, :n_cells, :]),
            "spawn_recon": self.spawn_recon(enc[:, n_cells : n_cells + n_spawns, :]),
            "squad_cls": self.squad_cls(enc[:, n_cells + n_spawns :, :]),
            "type": self.head_type(dec),
            "target": target_logits,
            "dir": self.head_dir(dec),
            "cell_ptr": cell_logits,
        }


# ---------------------------------------------------------------- 损失


def compute_ssl_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    masked: dict[str, Any],
    cfg: PretextConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """掩码重构损失（只在 ``sel_*`` 位置计；分头返回 parts 供报表）。"""
    parts: dict[str, float] = {}
    dev = batch["cells"].device

    def huber(pred: torch.Tensor, gold: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
        if not bool(sel.any()):
            return torch.zeros((), device=dev)
        return nn.functional.huber_loss(
            pred[sel], gold[sel], delta=cfg.huber_delta, reduction="mean"
        )

    gold_cell = torch.cat([batch["cells"], batch["cell_pos"]], dim=-1)
    l_cell = huber(out["cell_recon"], gold_cell, masked["sel_cell"])
    l_spawn = huber(out["spawn_recon"], batch["spawns"], masked["sel_spawn"])

    def ce(logits: torch.Tensor, gold: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
        if not bool(sel.any()):
            return torch.zeros((), device=dev)
        return nn.functional.cross_entropy(logits[sel], gold[sel], reduction="mean")

    l_squad = ce(out["squad_cls"], masked["gold_squad"], masked["sel_squad"])

    sel_hist = masked["sel_hist"]
    l_type = ce(out["type"], batch["type"], sel_hist)
    l_target = ce(out["target"], batch["target"], sel_hist)
    # cell 指针 / dir 只在 gold 语义帧上计（照抄 imitate.compute_loss 掩码口径）
    sel_deploy = sel_hist & (batch["type"] == 0)
    l_cellptr = ce(out["cell_ptr"], batch["cell"], sel_deploy)
    sel_dir = sel_deploy & (batch["dir"] != DIRECTIONS.index("None"))
    l_dir = ce(out["dir"], batch["dir"], sel_dir)

    total = (
        cfg.w_recon_cell * l_cell
        + cfg.w_recon_spawn * l_spawn
        + cfg.w_cls_squad * l_squad
        + cfg.w_act_type * l_type
        + cfg.w_act_target * l_target
        + cfg.w_act_cell * l_cellptr
        + cfg.w_act_dir * l_dir
    )
    parts = {
        "recon_cell": float(l_cell.detach()),
        "recon_spawn": float(l_spawn.detach()),
        "cls_squad": float(l_squad.detach()),
        "act_type": float(l_type.detach()),
        "act_target": float(l_target.detach()),
        "act_cell": float(l_cellptr.detach()),
        "act_dir": float(l_dir.detach()),
    }
    return total, parts


def _mk_to_device(mk: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """把 mask_batch 产出的张量迁到 device（``inputs`` 是嵌套 dict，须递归迁）。"""
    out: dict[str, Any] = {}
    for k, v in mk.items():
        if k == "inputs":
            out[k] = {
                ik: (iv.to(device) if torch.is_tensor(iv) else iv) for ik, iv in v.items()
            }
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------- 评测


@torch.no_grad()
def evaluate_ssl(
    model: SSLMaskedPolicy,
    tensors: list[dict[str, Any]],
    eval_masks: list[dict[str, Any]],
    featvec: torch.Tensor,
    n_targets: int,
    cfg: PretextConfig,
    device: torch.device,
) -> dict[str, Any]:
    """在**固定掩码**的 holdout batch 上评测（掩码跨 epoch 固定，曲线可比）。

    指标：分头损失 + 离散头命中率 + 回归头的 MAE（比 huber 更直观）。
    """
    acc: dict[str, int] = {}
    tot: dict[str, int] = {}
    mae: dict[str, float] = {}
    mae_n: dict[str, int] = {}
    loss_sum, nb = 0.0, 0
    for batch_cpu, mk in zip(tensors, eval_masks, strict=True):
        batch = to_device(batch_cpu, device)
        mk = _mk_to_device(mk, device)
        out = _forward_masked(model, batch, mk)
        loss, parts = compute_ssl_loss(out, batch, mk, cfg)
        loss_sum += float(loss.detach())
        nb += 1

        def _acc(name: str, logits: torch.Tensor, gold: torch.Tensor, sel: torch.Tensor) -> None:
            n = int(sel.sum())
            if n:
                acc[name] = acc.get(name, 0) + int((logits.argmax(-1)[sel] == gold[sel]).sum())
                tot[name] = tot.get(name, 0) + n

        _acc("cls_squad", out["squad_cls"], mk["gold_squad"], mk["sel_squad"])
        _acc("type", out["type"], batch["type"], mk["sel_hist"])
        _acc("target", out["target"], batch["target"], mk["sel_hist"])
        sel_dep = mk["sel_hist"] & (batch["type"] == 0)
        _acc("cell", out["cell_ptr"], batch["cell"], sel_dep)
        sel_dir = sel_dep & (batch["dir"] != DIRECTIONS.index("None"))
        _acc("dir", out["dir"], batch["dir"], sel_dir)

        for name, pred, gold, sel in (
            ("mae_cell", out["cell_recon"], torch.cat([batch["cells"], batch["cell_pos"]], -1),
             mk["sel_cell"]),
            ("mae_spawn", out["spawn_recon"], batch["spawns"], mk["sel_spawn"]),
        ):
            n = int(sel.sum())
            if n:
                mae[name] = mae.get(name, 0.0) + float((pred[sel] - gold[sel]).abs().sum())
                mae_n[name] = mae_n.get(name, 0) + n * pred.shape[-1]

    res: dict[str, Any] = {"loss": loss_sum / max(1, nb), "n_batches": nb}
    for name in acc:
        res[name + "_acc"] = acc[name] / tot[name]
        res[name + "_n"] = tot[name]
    for name in mae:
        res[name] = mae[name] / mae_n[name]
    return res


def _forward_masked(
    model: SSLMaskedPolicy, batch: dict[str, torch.Tensor], mk: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """用 ``mask_batch`` 产出的替换输入跑预置前向（train/eval 共用）。"""
    inp = mk["inputs"]
    mb = {
        **batch,
        "type": inp["type"],
        "target": inp["target"],
        "dir": inp["dir"],
        "cell": inp["cell"],
    }
    h_type, h_target, h_cell, h_dir = shift_history(mb)
    m80_hist = mk["m80_hist"]
    bos = torch.zeros((m80_hist.shape[0], 1), dtype=torch.bool, device=m80_hist.device)
    m80_hist_in = torch.cat([bos, m80_hist[:, :-1]], dim=1)
    return model.forward_ssl(
        inp["cells"], inp["cell_pos"], batch["cell_mask"],
        inp["spawns"], batch["spawn_mask"],
        inp["squad_feat"], batch["squad_mask"],
        h_type, h_target, h_cell, h_dir,
        mk["m80_cell"], mk["m80_spawn"], mk["m80_squad"], m80_hist_in,
    )


# ---------------------------------------------------------------- 基线


def _train_side_stats(train_tensors: list[dict[str, Any]]) -> dict[str, Any]:
    """train 侧众数/均值统计（恒预测基线的拟合侧，只看不与评测同分布的 train）。"""
    type_hist = torch.zeros(len(ACTION_TYPES), dtype=torch.long)
    dir_hist = torch.zeros(len(DIRECTIONS), dtype=torch.long)
    squad_hist: torch.Tensor | None = None
    cell_sum: torch.Tensor | None = None
    cell_n = 0
    spawn_sum: torch.Tensor | None = None
    spawn_n = 0
    for b in train_tensors:
        t_, l_ = b["type"], b["lengths"]
        for i in range(t_.shape[0]):
            n = int(l_[i])
            type_hist += torch.bincount(t_[i, :n], minlength=len(ACTION_TYPES))
            dep = t_[i, :n] == 0
            dirs = b["dir"][i, :n][dep & (b["dir"][i, :n] != DIRECTIONS.index("None"))]
            dir_hist += torch.bincount(dirs, minlength=len(DIRECTIONS))
            sq = b["squad_gold_idx"][i][b["squad_mask"][i] & (b["squad_gold_idx"][i] >= 0)]
            if len(sq):
                c = torch.bincount(sq, minlength=int(sq.max()) + 1)
                if squad_hist is None:
                    squad_hist = c
                else:
                    if len(c) > len(squad_hist):
                        squad_hist = torch.cat(
                            [squad_hist, torch.zeros(len(c) - len(squad_hist), dtype=torch.long)]
                        )
                    squad_hist[: len(c)] += c
        nc = int(b["cell_mask"].sum())
        gold = torch.cat([b["cells"], b["cell_pos"]], -1)
        s = gold[b["cell_mask"]].sum(0)
        cell_sum = s if cell_sum is None else cell_sum + s
        cell_n += nc
        ns = int(b["spawn_mask"].sum())
        if ns:
            s2 = b["spawns"][b["spawn_mask"]].sum(0)
            spawn_sum = s2 if spawn_sum is None else spawn_sum + s2
            spawn_n += ns
    return {
        "type_majority": int(type_hist.argmax()),
        "dir_majority": int(dir_hist.argmax()),
        "squad_majority": int(squad_hist.argmax()) if squad_hist is not None else 0,
        "cell_mean": (
            cell_sum / max(1, cell_n)
            if cell_sum is not None
            else torch.zeros(CELL_TOKEN_DIM)
        ),
        "spawn_mean": (
            spawn_sum / max(1, spawn_n)
            if spawn_sum is not None
            else torch.zeros(SPAWN_IN)
        ),
    }


def constant_baseline_eval(
    tensors: list[dict[str, Any]],
    eval_masks: list[dict[str, Any]],
    stats: dict[str, Any],
) -> dict[str, Any]:
    """恒预测基线在**评测侧**（holdout 固定掩码位）的命中率 / MAE。

    离散头恒预测 train 侧众数类；回归头恒预测 train 通道均值。给出的是「不用
    任何学习的常数预测器」在同一掩码评测集上的得分——预训练必须显著越过它。
    """
    hit = {"type": 0, "dir": 0, "squad": 0}
    tot = {"type": 0, "dir": 0, "squad": 0}
    mae = {"cell": 0.0, "spawn": 0.0}
    n_mae = {"cell": 0, "spawn": 0}
    for b, mk in zip(tensors, eval_masks, strict=True):
        t_ = b["type"]
        sel_h = mk["sel_hist"].cpu()
        hit["type"] += int((t_[sel_h] == stats["type_majority"]).sum())
        tot["type"] += int(sel_h.sum())
        sel_dep = sel_h & (t_ == 0)
        sel_dir = sel_dep & (b["dir"] != DIRECTIONS.index("None"))
        hit["dir"] += int((b["dir"].cpu()[sel_dir] == stats["dir_majority"]).sum())
        tot["dir"] += int(sel_dir.sum())
        sel_sq = mk["sel_squad"].cpu()
        hit["squad"] += int((mk["gold_squad"].cpu()[sel_sq] == stats["squad_majority"]).sum())
        tot["squad"] += int(sel_sq.sum())
        sel_c = mk["sel_cell"].cpu()
        gold_c = torch.cat([b["cells"], b["cell_pos"]], -1)
        mae["cell"] += float((gold_c[sel_c] - stats["cell_mean"]).abs().sum())
        n_mae["cell"] += int(sel_c.sum()) * gold_c.shape[-1]
        sel_s = mk["sel_spawn"].cpu()
        if int(sel_s.sum()):
            mae["spawn"] += float((b["spawns"][sel_s] - stats["spawn_mean"]).abs().sum())
            n_mae["spawn"] += int(sel_s.sum()) * b["spawns"].shape[-1]
    return {
        "type_acc": hit["type"] / tot["type"] if tot["type"] else 0.0,
        "dir_acc": hit["dir"] / tot["dir"] if tot["dir"] else 0.0,
        "cls_squad_acc": hit["squad"] / tot["squad"] if tot["squad"] else 0.0,
        "mae_cell": mae["cell"] / n_mae["cell"] if n_mae["cell"] else 0.0,
        "mae_spawn": mae["spawn"] / n_mae["spawn"] if n_mae["spawn"] else 0.0,
        "note": "cell/target 头无常数基线（指针空间逐局变化），其对照看随机初始化行",
    }


# ---------------------------------------------------------------- 预训练主循环


def pretrain(
    npz_path: Path,
    cfg: PretextConfig,
    *,
    holdout_families: set[str],
    out_dir: Path,
    device_str: str = "cuda",
    log: Any = print,
) -> dict[str, Any]:
    """掩码实体建模预训练主循环（轨道 A）。"""
    set_seed(cfg.seed)
    records, featvec_np, vocab_tokens = load_records(npz_path)
    featvec = torch.from_numpy(np.asarray(featvec_np, dtype=np.float32))

    train_recs = [r for r in records if (r["family"] or "") not in holdout_families]
    hold_recs = [r for r in records if (r["family"] or "") in holdout_families]
    if not train_recs or not hold_recs:
        raise ValueError("train/holdout 任一侧为空：请检查 holdout_families 设置")
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(train_recs)
    train_recs = train_recs[: cfg.limit_train]
    log(
        f"[data] 总 {len(records)} 局 → train {len(train_recs)}（截断上限 "
        f"{cfg.limit_train}）/ holdout {len(hold_recs)}（家族 {sorted(holdout_families)}）"
    )

    def prep(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """collate + 附 ``squad_gold_idx``（v0 collate 不落 featvec 行号，重构真值要用）。"""
        out: list[dict[str, Any]] = []
        for i in range(0, len(recs), cfg.batch_size):
            chunk = recs[i : i + cfg.batch_size]
            c = collate(chunk, featvec_np)
            b = len(chunk)
            nq = max(len(r["squad_idx"]) for r in chunk)
            g = np.full((b, nq), -1, dtype=np.int64)
            for j, r in enumerate(chunk):
                idx = np.asarray(r["squad_idx"])
                g[j, : len(idx)] = idx
            c["squad_gold_idx"] = torch.from_numpy(g)
            out.append(c)
        return out

    train_tensors = prep(train_recs)
    hold_tensors = prep(hold_recs)[: cfg.holdout_eval_batches]
    log(f"[data] train batches {len(train_tensors)} / holdout 评测 batches {len(hold_tensors)}")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    # vocab_size 口径与 imitate/imitate_v1 一致：Vocab.size = len(extra_tokens) + 1（含 UNK），
    # 指针布局 [0]=UNK | [1,1+size)=全局词表 | 编队槽位——错位 1 会让训练集 gold 全部 -inf
    vocab = Vocab(extra_tokens=vocab_tokens)
    model = SSLMaskedPolicy(
        featvec_dim=int(featvec.shape[1]),
        vocab_size=vocab.size,
        max_squad=MAX_SQUAD,
        n_featvec_rows=int(featvec.shape[0]),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        n_dec_layers=cfg.n_dec_layers,
        dropout=cfg.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[model] 参数量 {n_params/1e6:.2f}M  vocab={vocab.size}  device={device}")

    n_targets = 1 + vocab.size + MAX_SQUAD
    gen = torch.Generator().manual_seed(cfg.seed)
    eval_gen = torch.Generator().manual_seed(cfg.seed + 1)
    eval_masks = [mask_batch(b, featvec, n_targets, cfg, eval_gen) for b in hold_tensors]

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_tensors) * cfg.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total_steps)

    major = _train_side_stats(train_tensors[: cfg.holdout_eval_batches])
    const_base = constant_baseline_eval(hold_tensors, eval_masks, major)
    log(
        f"[baseline] 恒预测（train 众数/均值 → holdout 评测位）: "
        f"type={const_base['type_acc']:.3f} dir={const_base['dir_acc']:.3f} "
        f"squad={const_base['cls_squad_acc']:.3f} mae_cell={const_base['mae_cell']:.4f}"
    )

    base = evaluate_ssl(model, hold_tensors, eval_masks, featvec, n_targets, cfg, device)
    log(f"[ep 0/{cfg.epochs}] 【随机初始化基线】holdout {base}")

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "pretrain_ssl.pt"

    def _save_ckpt() -> None:
        """把 best-so-far 权重落盘（每次刷新即覆写：宿主/断线事故 = 损失止于当前 epoch）。"""
        torch.save(
            {
                "state_dict": (
                    best_state if best_state is not None else model.state_dict()
                ),
                "config": dict(cfg.__dict__),
                "vocab": vocab_tokens,
                "featvec_dim": int(featvec.shape[1]),
                "n_featvec_rows": int(featvec.shape[0]),
                "n_params": n_params,
                "holdout_families": sorted(holdout_families),
                "best_epoch": best_epoch,
            },
            ckpt,
        )

    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        rng.shuffle(train_tensors)
        ep_loss, nb = 0.0, 0
        for batch_cpu in train_tensors:
            batch = to_device(batch_cpu, device)
            mk = mask_batch(batch_cpu, featvec, n_targets, cfg, gen)
            mk = _mk_to_device(mk, device)
            out = _forward_masked(model, batch, mk)
            loss, _parts = compute_ssl_loss(out, batch, mk, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            ep_loss += float(loss.detach())
            nb += 1
        hs = evaluate_ssl(model, hold_tensors, eval_masks, featvec, n_targets, cfg, device)
        history.append({"epoch": ep + 1, "train_loss": ep_loss / max(1, nb), "holdout": hs,
                        "elapsed_s": round(time.time() - t0, 1)})
        log(
            f"[ep {ep+1}/{cfg.epochs}] train_loss={ep_loss/max(1, nb):.4f} | "
            f"holdout loss={hs['loss']:.4f} squad={hs.get('cls_squad_acc', 0):.3f} "
            f"type={hs.get('type_acc', 0):.3f} cell={hs.get('cell_acc', 0):.3f} "
            f"dir={hs.get('dir_acc', 0):.3f} | {history[-1]['elapsed_s']}s"
        )
        if hs["loss"] < best_loss:
            best_loss, best_epoch = hs["loss"], ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _save_ckpt()
            log(f"    ↑ 新的最优 holdout loss（已即时落盘 {ckpt.name}）")

    log(f"[save] {ckpt}（最优 epoch {best_epoch}，holdout loss {best_loss:.4f}）")
    _save_ckpt()
    return {
        "n_params": n_params,
        "n_train": len(train_recs),
        "n_holdout": len(hold_recs),
        "best_epoch": best_epoch,
        "baseline_random_init": base,
        "baseline_constant": const_base,
        "final_holdout": history[-1]["holdout"] if history else base,
        "history": history,
        "checkpoint": str(ckpt),
    }


# ---------------------------------------------------------------- 下游微调烟测


def load_pretrained_into(
    model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> tuple[int, list[str]]:
    """按键名+形状匹配把预训练权重灌入下游模型（宽松键级搬运，不匹配即跳过）。

    返回 (加载键数, 跳过键列表)。``TrackAPolicyV1`` 比 ``SSLMaskedPolicy``
    多出的 ``state_proj.*``（v1 战场状态投影）保持随机初始化；维度不符的
    （如 hist_target/head_target 因两侧词表不同）如实记录进跳过表。
    """
    own = model.state_dict()
    loaded = 0
    skipped: list[str] = []
    for k, v in own.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            own[k] = state_dict[k]
            loaded += 1
        else:
            skipped.append(k)
    model.load_state_dict(own)
    return loaded, skipped


def finetune_smoke(
    ckpt_path: Path,
    downstream_npz: Path,
    cfg: TrainConfig,
    *,
    holdout_stages: set[str],
    out_dir: Path,
    device_str: str = "cuda",
    log: Any = print,
) -> dict[str, Any]:
    """下游 3 epoch 微调烟测：预训练编码器初始化 policy_v1，只出早期信号。"""
    from .imitate_v1 import load_records_v1

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob["state_dict"]
    _records, featvec_np, vocab_tokens, state_dim, grid_channels = load_records_v1(
        downstream_npz
    )

    from ..models.track_a.policy_v1 import TrackAPolicyV1

    model = TrackAPolicyV1(
        featvec_dim=int(featvec_np.shape[1]),
        # 同 pretrain 的口径说明：Vocab.size = len(extra)+1，必须与下游 train() 内部构造一致
        vocab_size=Vocab(extra_tokens=vocab_tokens).size,
        max_squad=MAX_SQUAD,
        state_dim=state_dim,
        grid_channels=grid_channels,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        n_dec_layers=cfg.n_dec_layers,
        dropout=cfg.dropout,
    )
    loaded, skipped = load_pretrained_into(model, sd)
    log(f"[smoke] 预训练权重键级注入：加载 {loaded} 键，跳过 {len(skipped)} 键: {skipped}")
    init_path = out_dir / "init_extended.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "source": str(ckpt_path),
                "loaded_keys": loaded, "skipped_keys": skipped}, init_path)
    log(f"[smoke] 初始化快照 → {init_path}")

    rep = train_downstream(
        downstream_npz,
        cfg,
        holdout_stages=holdout_stages,
        out_dir=out_dir,
        device_str=device_str,
        log=log,
        ckpt_name="track_a_finetune_smoke.pt",
        resume_from=init_path,
    )
    return rep
