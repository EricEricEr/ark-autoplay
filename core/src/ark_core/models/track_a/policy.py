"""track_a：实体 Transformer 策略网络（v0）。

设计对齐方案 §8 轨道 A，但**输入实体集不同**（诚实反映数据现状）：

| 方案 §8 的实体 | v0 是否具备 | 说明 |
|---|---|---|
| 格子 token | ✅ | stagefeat 的 grid（5 通道）+ 归一化 (r,c) |
| 敌人 token | ⚠️ 降级 | 只有**计划出怪表**（spawns），无实时位置/血量 |
| 干员 token | ⚠️ 降级 | 只有**编队**（手上有谁），无场上实时状态 |
| 待部署栏 / 已部署 / 费用 | ❌ | 无状态数据 |
| 动作历史 token | ✅ | 已执行动作序列（当前 step 的"局势"代理） |

因此本网络学的是 ``P(a_t | 关卡, 编队, a_<t)``，**不是**方案 §8 的 ``P(a_t | state_t)``。
当感知层就绪，只需向实体集注入新 token 类型与通道，编码器/指针头结构无需重写——
这是本 v0 的核心工程价值：**把训练链路先打通并钉死形状**。

指针头索引布局（与 imitation_data 严格一致）
--------------------------------------------
``target`` 指针取值域 = ``[0]=UNK`` ｜ ``[1, 1+vocab)``=全局装置/类别 ｜
``[1+vocab, 1+vocab+MAX_SQUAD)``=编队槽位。输出维度 = ``1 + vocab + max_squad``。

变长处理
--------
格子 / 出怪 / 编队三组的长度**逐局不同**，由 collate 补齐到 batch 内最大值，
配套 ``*_mask``（True=有效）屏蔽 padding。格子指针 logits 在 padding 位置 ``-inf``。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...protocol.action import ACTION_TYPES, DIRECTIONS, GRID_CHANNELS

CELL_IN = GRID_CHANNELS + 2
"""格子 token 输入：5 通道 + 归一化 (r, c)。"""

SPAWN_IN = 4
"""出怪 token 输入：t / wave / route / matched。"""

N_GROUPS = 3
"""实体组数：0=cell 1=spawn 2=squad。"""


class EntityEncoder(nn.Module):
    """把三组实体拼成一个 token 序列做自注意力。

    组间用可学习 **type embedding** 区分；格子自带 (r,c) 归一化坐标（等价于空间位置
    编码），出怪自带归一化时刻，两者都无需额外位置编码。
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.cell_proj = nn.Linear(CELL_IN, d_model)
        self.spawn_proj = nn.Linear(SPAWN_IN, d_model)
        self.type_emb = nn.Embedding(N_GROUPS, d_model)
        self.norm_in = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)

    def forward(
        self,
        cells: Tensor,
        cell_pos: Tensor,
        cell_mask: Tensor,
        spawns: Tensor,
        spawn_mask: Tensor,
        squad_tok: Tensor,
        squad_mask: Tensor,
    ) -> tuple[Tensor, int, int, int]:
        """返回 (编码序列 [B,L,D], n_cells, n_spawns, n_squad)。"""
        n_cells = cells.shape[1]
        n_spawns = spawns.shape[1]

        cell_tok = self.cell_proj(torch.cat([cells, cell_pos], dim=-1)) + self.type_emb.weight[0]
        spawn_tok = self.spawn_proj(spawns) + self.type_emb.weight[1]
        squad_t = squad_tok + self.type_emb.weight[2]

        seq = self.norm_in(torch.cat([cell_tok, spawn_tok, squad_t], dim=1))
        valid = torch.cat([cell_mask, spawn_mask, squad_mask], dim=1)
        out = self.encoder(seq, src_key_padding_mask=~valid)
        return out, n_cells, n_spawns, squad_t.shape[1]


class TrackAPolicy(nn.Module):
    """自回归策略：每步预测 (type, target, cell, dir)。"""

    def __init__(
        self,
        featvec_dim: int,
        vocab_size: int,
        max_squad: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        n_dec_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_squad = max_squad
        self.n_types = len(ACTION_TYPES)
        self.n_dirs = len(DIRECTIONS)
        self.n_targets = 1 + vocab_size + max_squad

        self.encoder = EntityEncoder(d_model, n_heads, n_layers, dropout)

        # 干员 featvec → d_model（编队 token 与历史干员共用）
        self.op_proj = nn.Sequential(
            nn.Linear(featvec_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # 动作历史编码。索引 +1 给 BOS；target 用上一动作的值（推理时已知，不泄漏）
        self.hist_type = nn.Embedding(self.n_types + 1, d_model)
        self.hist_target = nn.Embedding(self.n_targets + 1, d_model)
        self.hist_dir = nn.Embedding(self.n_dirs, d_model)
        self.hist_cell = nn.Linear(2, d_model)
        self.hist_norm = nn.LayerNorm(d_model)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(
            dec_layer, num_layers=n_dec_layers, enable_nested_tensor=False
        )

        self.head_type = nn.Linear(d_model, self.n_types)
        self.head_target = nn.Linear(d_model, self.n_targets)
        self.head_dir = nn.Linear(d_model, self.n_dirs)
        # 格子指针（加性注意力式）
        self.cell_query = nn.Linear(d_model, d_model)
        self.cell_key = nn.Linear(d_model, d_model)

    def forward(
        self,
        cells: Tensor,
        cell_pos: Tensor,
        cell_mask: Tensor,
        spawns: Tensor,
        spawn_mask: Tensor,
        squad_feat: Tensor,
        squad_mask: Tensor,
        hist_type: Tensor,
        hist_target: Tensor,
        hist_cell: Tensor,
        hist_dir: Tensor,
    ) -> dict[str, Tensor]:
        """前向。hist_* 已右移一位（首列为 BOS）。"""
        squad_tok = self.op_proj(squad_feat)
        enc, n_cells, _n_spawns, _n_squad = self.encoder(
            cells, cell_pos, cell_mask, spawns, spawn_mask, squad_tok, squad_mask
        )
        t = hist_type.shape[1]

        h = (
            self.hist_type(hist_type)
            + self.hist_target(hist_target)
            + self.hist_dir(hist_dir)
            + self.hist_cell(hist_cell)
        )
        h = self.hist_norm(h)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=h.device), diagonal=1)
        dec = self.decoder(h, mask=causal)

        cell_enc = enc[:, :n_cells, :]
        q = self.cell_query(dec)
        k = self.cell_key(cell_enc)
        cell_logits = torch.einsum("btd,bsd->bts", q, k) / (self.d_model**0.5)
        # padding 格子置 -inf，使指针只能落在真实格子上
        cell_logits = cell_logits.masked_fill(~cell_mask.unsqueeze(1), float("-inf"))
        # 目标指针同理：编队 padding 槽位置 -inf
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
            "type": self.head_type(dec),
            "target": target_logits,
            "cell": cell_logits,
            "dir": self.head_dir(dec),
        }
