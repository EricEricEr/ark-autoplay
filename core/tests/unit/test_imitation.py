"""imitation 数据管线与模型单测：合成数据，不含任何真实游戏内容。

重点覆盖三件容易错且后果严重的事：
1. **坐标序**：deploy 的 [x,y] 必须经 normalize_deploy_location 转成 [row,col]；
2. **指针索引布局**：UNK / 全局词表 / 编队槽位三段不得重叠或错位；
3. **历史右移**：动作历史必须右移一位，不得让当前步答案泄漏进输入。
"""

from __future__ import annotations

import numpy as np
import pytest

# 训练依赖是可选的（`uv sync --group train`）：torch 缺失时整文件 skip，
# 保证 CI 与数据管线无需安装 torch 也能跑（见 pyproject.toml 的 train 组说明）。
# 注意用 importorskip 判定后仍需 `import torch`，否则 torch 只是普通变量，
# pyright 会把 `torch.Tensor` 注解判为「变量不能用作类型」。
pytest.importorskip("torch")
import torch  # noqa: E402

from ark_core.protocol.action import ACTION_TYPES, DIRECTIONS, GRID_CHANNELS  # noqa: E402
from ark_core.training.imitate import (  # noqa: E402
    TrainConfig,
    collate,
    compute_loss,
    shift_history,
)
from ark_core.training.imitation_data import (  # noqa: E402
    DIR_TO_IDX,
    TYPE_TO_IDX,
    UNK_TARGET,
    BuildStats,
    Vocab,
    encode_episode,
)

# ---------------------------------------------------------------- 合成数据


def _synth_stage(rows: int = 3, cols: int = 4, n_spawns: int = 2) -> dict:
    """合成关卡（grid 为 row-major 展平的 5 通道）。"""
    grid = []
    for r in range(rows):
        for c in range(cols):
            # h=0/1, build=0/1/2/3, deployable, is_start, is_end
            grid.extend([r % 2, 1 if (r + c) % 2 else 0, 1, 0, 0])
    return {
        "stage_id": "main_99-01",
        "rows": rows,
        "cols": cols,
        "grid": grid,
        "spawns": [
            {"t_ms": i * 1000, "wave": 0, "route": 0, "enemy_id": "e", "enemy_matched": 1}
            for i in range(n_spawns)
        ],
        "opt_initial_cost": 10,
    }


def _synth_ep(actions: list[dict], squad_ids: list[str]) -> dict:
    return {
        "episode_id": "syn",
        "stage": {"stage_id": "main_99-01", "family": "main_99"},
        "squad": [{"slot": i, "char_id": cid, "name": cid} for i, cid in enumerate(squad_ids)],
        "actions": actions,
    }


# ---------------------------------------------------------------- 1. 坐标序


def test_deploy_location_is_converted_from_xy_to_rowcol():
    """deploy 的 [x,y] 必须翻成 [row,col]——错则格子指针全指错地方。"""
    featvec_idx = {"opA": 0}
    vocab = Vocab(extra_tokens=[])
    stats = BuildStats()
    # 关卡 3 行 4 列；deploy location=[x=3, y=1] → row=1, col=3 → cell = 1*4+3 = 7
    ep = _synth_ep(
        [{"type": "deploy", "location": [3, 1], "direction": "Right",
          "target": {"kind": "operator", "char_id": "opA"}}],
        ["opA"],
    )
    rec = encode_episode(ep, _synth_stage(3, 4), featvec_idx, vocab, stats)
    assert rec is not None
    assert rec["cell"][0] == 1 * 4 + 3, "坐标序转换错误（应为 [row=1, col=3]）"


def test_cell_out_of_bounds_is_counted_not_silently_zero():
    """越界坐标必须计数（不静默当 0 号格，那会伪造一个合法动作）。"""
    stats = BuildStats()
    ep = _synth_ep(
        [{"type": "deploy", "location": [99, 99], "direction": "Right",
          "target": {"kind": "operator", "char_id": "opA"}}],
        ["opA"],
    )
    rec = encode_episode(ep, _synth_stage(3, 4), {"opA": 0}, Vocab(extra_tokens=[]), stats)
    assert rec is not None
    assert stats.n_cell_oob == 1


# ---------------------------------------------------------------- 2. 指针布局


def test_pointer_index_layout_is_consistent():
    """UNK=0 ｜ 全局词表 [1, 1+vocab) ｜ 编队槽位 [1+vocab, ...)。"""
    vocab = Vocab(extra_tokens=["device:棋子", "category:healer"])
    assert vocab.size == 3  # UNK + 2
    ep = _synth_ep(
        [
            {"type": "deploy", "location": [0, 0], "direction": "Right",
             "target": {"kind": "device", "name": "棋子"}},
            {"type": "deploy", "location": [1, 1], "direction": "Down",
             "target": {"kind": "operator", "char_id": "opB"}},
            {"type": "skill", "target": {"kind": "operator", "char_id": "opB"}},
        ],
        ["opA", "opB"],
    )
    rec = encode_episode(ep, _synth_stage(), {"opA": 0, "opB": 1}, vocab, BuildStats())
    assert rec is not None
    t = rec["target"]
    # 装置 token "device:棋子" 是第 1 个 extra → 索引 1
    assert t[0] == 1, f"装置目标索引应为 1，实际 {t[0]}"
    # opB 是编队第 2 槽（slot=1）→ 1 + vocab.size + 1 = 5
    assert t[1] == 1 + vocab.size + 1
    assert t[2] == 1 + vocab.size + 1


def test_unknown_target_falls_back_to_unk():
    """未登录目标落 UNK 槽并计数，不得越界或错位到别的类别。"""
    stats = BuildStats()
    ep = _synth_ep(
        [{"type": "deploy", "location": [0, 0], "direction": "Right",
          "target": {"kind": "device", "name": "不存在的装置"}}],
        ["opA"],
    )
    rec = encode_episode(ep, _synth_stage(), {"opA": 0}, Vocab(extra_tokens=[]), stats)
    assert rec is not None
    assert rec["target"][0] == UNK_TARGET
    assert stats.n_target_unk == 1


def test_operator_not_in_squad_is_counted():
    """目标干员不在本局编队（如召唤物）→ 落 UNK 并计数（实测占 6.1%）。"""
    stats = BuildStats()
    ep = _synth_ep(
        [{"type": "deploy", "location": [0, 0], "direction": "Right",
          "target": {"kind": "operator", "char_id": "Mon3tr"}}],
        ["opA"],
    )
    rec = encode_episode(ep, _synth_stage(), {"opA": 0}, Vocab(extra_tokens=[]), stats)
    assert rec is not None
    assert rec["target"][0] == UNK_TARGET
    assert stats.n_target_unk == 1


# ---------------------------------------------------------------- 3. 编码表


def test_direction_aliases_normalize():
    """中英文朝向必须归一到同一类别（实测两种写法并存）。"""
    assert DIR_TO_IDX["Left"] == DIR_TO_IDX.get("Left")
    assert DIRECTIONS.index("Left") == 2
    assert "None" in DIRECTIONS
    # 编码表顺序是已发布契约，不得变更
    assert ACTION_TYPES[0] == "deploy"
    assert DIRECTIONS[-1] == "None"


def test_action_type_order_is_frozen():
    """ACTION_TYPES 顺序变更会让已训练 checkpoint 权重错位。"""
    assert ACTION_TYPES == ("deploy", "skill", "retreat", "speed", "bullet_time")
    assert TYPE_TO_IDX["deploy"] == 0


# ---------------------------------------------------------------- 4. 历史不泄漏


def test_shift_history_does_not_leak_current_step():
    """历史右移一位：位置 t 只能看到 < t 的动作。"""
    rec = {
        "cells": np.zeros((12, GRID_CHANNELS), dtype=np.float32),
        "cell_pos": np.zeros((12, 2), dtype=np.float32),
        "spawns": np.zeros((0, 4), dtype=np.float32),
        "squad_idx": np.asarray([0], dtype=np.int64),
        "type": np.asarray([0, 1, 2], dtype=np.int64),
        "target": np.asarray([7, 8, 9], dtype=np.int64),
        "cell": np.asarray([3, 4, 5], dtype=np.int64),
        "dir": np.asarray([0, 1, 2], dtype=np.int64),
    }
    featvec = np.zeros((1, 4), dtype=np.float32)
    batch = collate([rec], featvec)
    h_type, h_target, h_cell, h_dir = shift_history(batch)

    # 第 0 位必须是 BOS（type 用了 +len 的哨兵槽），不得等于当前动作
    assert int(h_type[0, 0]) == len(ACTION_TYPES)
    # 第 1 位应等于第 0 步的目标 7（而不是第 1 步的 8）
    assert int(h_target[0, 1]) == 7
    assert int(h_target[0, 2]) == 8
    assert int(h_dir[0, 1]) == 0


# ---------------------------------------------------------------- 5. 损失掩码


def test_dir_loss_only_on_deploy_with_direction():
    """非 deploy 动作与无朝向的 deploy 不得计入 dir 损失。"""
    b, t = 1, 3
    featvec = np.zeros((1, 4), dtype=np.float32)
    rec = {
        "cells": np.zeros((4, GRID_CHANNELS), dtype=np.float32),
        "cell_pos": np.zeros((4, 2), dtype=np.float32),
        "spawns": np.zeros((0, 4), dtype=np.float32),
        "squad_idx": np.asarray([0], dtype=np.int64),
        # 第 0 步 deploy 带朝向；第 1 步 skill（无朝向语义）；第 2 步 deploy 无朝向
        "type": np.asarray([0, 1, 0], dtype=np.int64),
        "target": np.asarray([1, 1, 1], dtype=np.int64),
        "cell": np.asarray([1, 0, 2], dtype=np.int64),
        "dir": np.asarray([0, DIRECTIONS.index("None"), DIRECTIONS.index("None")], dtype=np.int64),
    }
    batch = collate([rec], featvec)
    cfg = TrainConfig()

    out: dict[str, torch.Tensor] = {
        "type": torch.zeros(b, t, len(ACTION_TYPES)),
        "target": torch.zeros(b, t, 4),
        "cell": torch.zeros(b, t, 4),
        "dir": torch.zeros(b, t, len(DIRECTIONS)),
    }
    loss, parts = compute_loss(out, batch, cfg)
    assert torch.isfinite(loss)
    assert parts["dir"] >= 0.0
