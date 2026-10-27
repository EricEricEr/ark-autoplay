"""共享策略骨干的静态重构、观测状态重构与结果识别；辅助权重可导出至同一策略。"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .policy_v1 import TrackAPolicyV1


class AllFieldsPolicy(nn.Module):
    """所有资料更新同一编码器；失败记录只走辅助目标，不模仿其动作。"""

    def __init__(self, *args: Any, n_plan_types: int, n_failure_types: int, **kwargs: Any) -> None:
        super().__init__()
        self.policy = TrackAPolicyV1(*args, **kwargs)
        d, state_dim = self.policy.d_model, self.policy.state_dim
        self.cell_recon = nn.Linear(d, self.policy.grid_channels)
        self.spawn_recon = nn.Linear(d, 4)
        feature_projection = self.policy.op_proj[0]
        assert isinstance(feature_projection, nn.Linear)
        self.squad_recon = nn.Linear(d, feature_projection.in_features)
        self.state_recon = nn.Linear(d, state_dim)
        self.state_available = nn.Linear(state_dim, d, bias=False)
        self.state_hidden = nn.Linear(state_dim, d, bias=False)
        self.outcome_head = nn.Linear(d, 2)
        self.marker_head = nn.Linear(d, 2)
        self.failure_head = nn.Linear(d, n_failure_types)
        self.plan_embedding = nn.Embedding(n_plan_types + 1, d)
        self.plan_recon = nn.Linear(d, n_plan_types)
        self.plan_context = nn.Parameter(torch.zeros(d))

    def auxiliary(
        self,
        batch: dict[str, Tensor],
        states: Tensor,
        observed: Tensor,
        selections: dict[str, Tensor],
        plan_type: Tensor,
    ) -> dict[str, Tensor]:
        """只遮输入观测；返回重构与分类预测，缺失状态不创建伪造帧。"""
        p = self.policy
        cells = batch["cells"].masked_fill(selections["cells"].unsqueeze(-1), 0)
        spawns = batch["spawns"].masked_fill(selections["spawns"].unsqueeze(-1), 0)
        squad = batch["squad_feat"].masked_fill(selections["squad"].unsqueeze(-1), 0)
        encoded, nc, ns, _ = p.encoder(
            cells,
            batch["cell_pos"],
            batch["cell_mask"],
            spawns,
            batch["spawn_mask"],
            p.op_proj(squad),
            batch["squad_mask"],
        )
        valid = torch.cat([batch["cell_mask"], batch["spawn_mask"], batch["squad_mask"]], dim=1)
        pooled = (encoded * valid.unsqueeze(-1)).sum(1) / valid.sum(1).clamp(min=1).unsqueeze(-1)
        predicted_states = states
        state_pool = torch.zeros_like(pooled)
        if states.shape[1]:
            masked = states.masked_fill(selections["states"] | ~observed, 0)
            masked[..., 9] = states[..., 9]  # 序号是已知上下文，仍不作为观测重构标签。
            h = p.state_proj(masked) + self.state_available(observed.float())
            h = h + self.state_hidden(selections["states"].float()) + pooled.unsqueeze(1)
            h = p.hist_norm(h)
            count = h.shape[1]
            causal = torch.triu(
                torch.ones(count, count, dtype=torch.bool, device=h.device), diagonal=1
            )
            decoded = p.decoder(h, mask=causal)
            predicted_states = self.state_recon(decoded)
            has_frame = observed.any(-1)
            clean_state = states.masked_fill(~observed, 0)
            clean_state[..., 9] = states[..., 9]
            clean = p.state_proj(clean_state)
            clean = p.hist_norm(
                clean + self.state_available(observed.float()) + pooled.unsqueeze(1)
            )
            classified = p.decoder(clean, mask=causal)
            state_pool = (classified * has_frame.unsqueeze(-1)).sum(1) / has_frame.sum(1).clamp(
                min=1
            ).unsqueeze(-1)
        context = pooled + state_pool
        plan_logits = torch.empty(1, 0, self.plan_recon.out_features, device=context.device)
        if plan_type.shape[1]:
            tokens = plan_type.masked_fill(selections["plan"], self.plan_recon.out_features)
            h = self.plan_embedding(tokens) + pooled.unsqueeze(1) + self.plan_context
            count = h.shape[1]
            position = torch.arange(count, device=h.device).float()
            frequency = torch.exp(
                torch.arange(0, p.d_model, 2, device=h.device).float()
                * (-9.210340371976184 / p.d_model)
            )
            positional = torch.zeros(count, p.d_model, device=h.device)
            positional[:, 0::2] = torch.sin(position[:, None] * frequency)
            positional[:, 1::2] = torch.cos(position[:, None] * frequency)
            causal = torch.triu(
                torch.ones(count, count, dtype=torch.bool, device=h.device), diagonal=1
            )
            plan_logits = self.plan_recon(p.decoder(p.hist_norm(h + positional), mask=causal))
        return {
            "cells": self.cell_recon(encoded[:, :nc]),
            "spawns": self.spawn_recon(encoded[:, nc : nc + ns]),
            "squad": self.squad_recon(encoded[:, nc + ns :]),
            "states": predicted_states,
            "outcome": self.outcome_head(context),
            "marker": self.marker_head(context),
            "plan": plan_logits,
            "failure": self.failure_head(context),
        }
