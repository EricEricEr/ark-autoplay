"""track_a：实体 Transformer 策略网络（v1，注入战场状态）。

与 v0（:mod:`ark_core.models.track_a.policy`）的唯一差异：动作历史序列 ``h``
额外叠加一项 **战场状态投影**——``h = hist_type + hist_target + hist_dir +
hist_cell + state_proj(hist_state)``。实体编码器（格子/出怪/编队）与四个输出头
原样复用 v0 组件，不改动 ``policy.py``。

对齐语义（**关键，勿改**）
--------------------------
第 t 步输入的状态取**第 t 个动作之前**的战场快照（battle_states.jsonl 第 t 条，
见 state_action_data 模块 docstring）。因此：

- 动作历史（type/target/cell/dir）按 v0 约定**右移一位**、首列 BOS——不能让
  当前步答案泄漏进输入；
- ``hist_state`` **不右移**：它描述的是动作 a_t 发生前的局势，本身不含 a_t
  的标签信息（state_feat 由 cost/kills/手牌/已部署等数值构成，无动作类别），
  放在第 t 位是因果安全的输入，反而正是本任务相对 v0 新增的条件变量。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...protocol.action import ACTION_TYPES, DIRECTIONS, GRID_CHANNELS
from .policy import EntityEncoder


class TrackAPolicyV1(nn.Module):
    """自回归策略（带战场状态）：每步预测 (type, target, cell, dir)。

    指针头索引布局与 v0 严格一致：``[0]=UNK ｜ [1, 1+vocab)=全局词表 ｜
    [1+vocab, 1+vocab+max_squad)=编队槽位``。
    """

    def __init__(
        self,
        featvec_dim: int,
        vocab_size: int,
        max_squad: int,
        state_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        n_dec_layers: int = 2,
        dropout: float = 0.1,
        grid_channels: int = GRID_CHANNELS,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_squad = max_squad
        self.state_dim = state_dim
        self.grid_channels = grid_channels
        self.n_types = len(ACTION_TYPES)
        self.n_dirs = len(DIRECTIONS)
        self.n_targets = 1 + vocab_size + max_squad

        self.encoder = EntityEncoder(
            d_model, n_heads, n_layers, dropout, grid_channels=grid_channels
        )

        # 干员 featvec → d_model（编队 token 与历史干员共用，同 v0）
        self.op_proj = nn.Sequential(
            nn.Linear(featvec_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # 动作历史编码（同 v0：索引 +1 给 BOS）
        self.hist_type = nn.Embedding(self.n_types + 1, d_model)
        self.hist_target = nn.Embedding(self.n_targets + 1, d_model)
        self.hist_dir = nn.Embedding(self.n_dirs, d_model)
        self.hist_cell = nn.Linear(2, d_model)
        # 战场状态投影（v1 新增；不右移，对齐语义见模块 docstring）
        self.state_proj = nn.Linear(state_dim, d_model)
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
        hist_state: Tensor,
    ) -> dict[str, Tensor]:
        """前向。hist_type/target/cell/dir 已右移一位（BOS 首列）；hist_state **不右移**。

        ``hist_state``: float32 [B, T, state_dim]，第 t 位 = 第 t 个动作前的状态。
        """
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
            + self.state_proj(hist_state)
        )
        h = self.hist_norm(h)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=h.device), diagonal=1)
        dec = self.decoder(h, mask=causal)

        cell_enc = enc[:, :n_cells, :]
        q = self.cell_query(dec)
        k = self.cell_key(cell_enc)
        cell_logits = torch.einsum("btd,bsd->bts", q, k) / (self.d_model**0.5)
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
            "type": self.head_type(dec),
            "target": target_logits,
            "cell": cell_logits,
            "dir": self.head_dir(dec),
        }
