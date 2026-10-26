"""infer.track_a_v1 单测：合成数据为主，不依赖真机。

覆盖四件容易错的事：
1. **坐标互逆**：作业 [x,y] ↔ 扁平格子 idx 的往返恒等（ADR-0003 的坑）；
2. **解码映射**：四头 logits → 作业形态动作 dict（type/target/cell/dir）；
3. **合法性过滤**：费用不足 / 冷却中 / 格子被占时 argmax 被正确跳过到次优，
   全部被屏蔽时返回 None；
4. **训练同口径**：encode_action_for_history 与 encode_episode 的动作编码一致。

真 checkpoint + 真数据集 npz 的链路冒烟仅在 ``ARK_ML_ROOT`` 存在时运行。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# 训练依赖是可选的（`uv sync --group train`）：torch 缺失时整文件 skip，
# 与 tests/unit/test_imitation.py 同一模式。
pytest.importorskip("torch")
import torch  # noqa: E402

from ark_core.infer.track_a_v1 import (  # noqa: E402
    InferConfig,
    StageContext,
    cell_to_xy,
    decode_action,
    encode_action_for_history,
    greedy_step,
    load_track_a_v1,
    xy_to_cell,
)
from ark_core.models.track_a.policy_v1 import TrackAPolicyV1  # noqa: E402
from ark_core.protocol.action import ACTION_TYPES, DIRECTIONS, GRID_CHANNELS  # noqa: E402
from ark_core.training.imitation_data import (  # noqa: E402
    DIR_TO_IDX,
    MAX_SQUAD,
    TYPE_TO_IDX,
    Vocab,
)
from ark_core.training.state_action_data import STATE_DIM  # noqa: E402

# ---------------------------------------------------------------- 合成夹具

_ROWS, _COLS, _FEAT_DIM = 3, 4, 4
_N_CELLS = _ROWS * _COLS


def _ctx() -> StageContext:
    """合成关卡 3 行 4 列、全部格可部署；编队两槽；词表含一名支援装置。"""
    return StageContext(
        stage_id="syn_01",
        rows=_ROWS,
        cols=_COLS,
        vocab=Vocab(extra_tokens=["name:支援装置"]),
        squad_names=["玫兰莎", "克洛丝"],
        squad_tokens=["opRose", "opKroos"],
        name2cid={"玫兰莎": "opRose", "克洛丝": "opKroos"},
        cells=np.ones((_N_CELLS, GRID_CHANNELS), dtype=np.float32),
        cell_pos=np.zeros((_N_CELLS, 2), dtype=np.float32),
        spawns=np.zeros((0, 4), dtype=np.float32),
        squad_feat=np.zeros((2, _FEAT_DIM), dtype=np.float32),
    )


def _logits(n: int, ranking: list[int], low: float = -100.0) -> np.ndarray:
    """按 ranking 依次递减的 logits（ranking[0] 最大；未点名槽压到极低）。"""
    out = np.full(n, low, dtype=np.float64)
    for rank, idx in enumerate(ranking):
        out[idx] = 100.0 - rank * 50.0
    return out


def _bs(**over) -> dict:
    """合成 battle_state：两干员在手可用、费用 99、无部署。"""
    bs = {
        "in_battle": True,
        "in_speedup": False,
        "costs": 99.0,
        "kills": 0,
        "total_kills": 10,
        "hand": [
            {"name": "玫兰莎", "cost": 5, "available": True, "cooling": False},
            {"name": "克洛丝", "cost": 3, "available": True, "cooling": False},
        ],
        "deployed": [],
    }
    bs.update(over)
    return bs


def _four_heads(
    ctx: StageContext,
    type_rank: list[int],
    target_rank: list[int],
    cell_rank: list[int],
    dir_rank: list[int],
):
    n_targets = 1 + ctx.vocab.size + len(ctx.squad_names)
    return (
        _logits(len(ACTION_TYPES), type_rank),
        _logits(n_targets, target_rank),
        _logits(_N_CELLS, cell_rank),
        _logits(len(DIRECTIONS), dir_rank),
    )


# 指针布局：vocab.size=2（UNK + 1 个 extra），编队槽位从 1 + 2 = 3 开始
_SLOT0 = 3
_SLOT1 = 4
_VOCAB_DEV = 1

# ---------------------------------------------------------------- 1. 坐标互逆


def test_xy_cell_roundtrip_identity():
    """作业 [x,y] 与扁平格子 idx 的编解码必须往返恒等（坐标序的坑在案）。"""
    for r in range(_ROWS):
        for c in range(_COLS):
            idx = xy_to_cell([c, r], _ROWS, _COLS)
            assert idx == r * _COLS + c
            assert cell_to_xy(idx, _COLS) == [c, r]


# ---------------------------------------------------------------- 2. 解码映射


def test_decode_deploy_to_job_action():
    """deploy 四头：槽位目标 → 中文名；cell 7 → [x=3, y=1]；dir→朝向字符串。"""
    ctx = _ctx()
    heads = _four_heads(ctx, [TYPE_TO_IDX["deploy"]], [_SLOT1], [7], [DIR_TO_IDX["Down"]])
    act = decode_action(*heads, ctx, _bs())
    assert act == {
        "type": "Deploy",
        "name": "克洛丝",
        "location": [3, 1],
        "direction": "Down",
    }


def test_decode_vocab_target_strips_name_prefix():
    """全局词表目标解码去掉 ``name:`` 前缀（训练词表 token 即该前缀形态）。"""
    ctx = _ctx()
    heads = _four_heads(ctx, [TYPE_TO_IDX["deploy"]], [_VOCAB_DEV], [0], [0])
    act = decode_action(*heads, ctx, _bs())
    assert act is not None
    assert act["type"] == "Deploy"
    assert act["name"] == "支援装置"


def test_decode_speed_and_skill():
    """speed 无目标语义映射到 SpeedUp；skill 目标取已部署干员槽位。

    speed 的纯映射语义用关闭语义闸门的配置测（本用例测的是解码映射，
    闸门行为由下方专门用例覆盖）：默认闸门下手牌可部署 = 非死亡时间，
    SpeedUp 会被正确屏蔽。
    """
    ctx = _ctx()
    heads = _four_heads(ctx, [TYPE_TO_IDX["speed"]], [0], [0], [0])
    cfg = InferConfig(speed_semantic_gate=False)
    assert decode_action(*heads, ctx, _bs(), cfg) == {"type": "SpeedUp"}

    heads = _four_heads(ctx, [TYPE_TO_IDX["skill"]], [_SLOT0], [0], [0])
    bs = _bs(deployed=[{"name": "玫兰莎", "row": 0, "col": 0}])
    assert decode_action(*heads, ctx, bs) == {"type": "Skill", "name": "玫兰莎"}


# ---------------------------------------------------------------- 3. 合法性过滤


def test_cost_insufficient_skips_to_next_target():
    """模型首选的干员费用不够（含余量）时，必须受限 argmax 到次优合法槽位。"""
    ctx = _ctx()
    heads = _four_heads(ctx, [TYPE_TO_IDX["deploy"]], [_SLOT0, _SLOT1], [7], [0])
    # 玫兰莎 cost=10 > costs=5（margin=0）被判非法；克洛丝 cost=3 可行
    bs = _bs(
        costs=5.0,
        hand=[
            {"name": "玫兰莎", "cost": 10, "available": True, "cooling": False},
            {"name": "克洛丝", "cost": 3, "available": True, "cooling": False},
        ],
    )
    act = decode_action(*heads, ctx, bs)
    assert act is not None and act["type"] == "Deploy"
    assert act["name"] == "克洛丝", "费用不足的首选目标未被屏蔽到次优"


def test_cost_margin_relaxes_affordability():
    """cost_margin 余量：cost <= costs + margin 视为可部署。"""
    ctx = _ctx()
    heads = _four_heads(ctx, [TYPE_TO_IDX["deploy"]], [_SLOT0, _SLOT1], [7], [0])
    cfg = InferConfig(cost_margin=6.0)  # 10 <= 5 + 6 → 玫兰莎可行
    bs = _bs(
        costs=5.0,
        hand=[
            {"name": "玫兰莎", "cost": 10, "available": True, "cooling": False},
            {"name": "克洛丝", "cost": 3, "available": True, "cooling": False},
        ],
    )
    act = decode_action(*heads, ctx, bs, cfg)
    assert act is not None and act["name"] == "玫兰莎"


def test_cooling_and_unavailable_skipped():
    """冷却中 / 不可用的手牌同样被屏蔽，回退到下一合法槽位。"""
    ctx = _ctx()
    heads = _four_heads(ctx, [TYPE_TO_IDX["deploy"]], [_SLOT0, _SLOT1], [7], [0])
    bs = _bs(
        hand=[
            {"name": "玫兰莎", "cost": 5, "available": True, "cooling": True},
            {"name": "克洛丝", "cost": 3, "available": True, "cooling": False},
        ]
    )
    act = decode_action(*heads, ctx, bs)
    assert act is not None and act["name"] == "克洛丝"


def test_occupied_cell_skipped_to_next_best():
    """被 deployed 占用的格子被屏蔽，cell 头跳到次优格。"""
    ctx = _ctx()
    heads = _four_heads(ctx, [TYPE_TO_IDX["deploy"]], [_SLOT0], [7, 2], [0])
    # 任意干员占了 (row=1, col=3) → cell 7 非法 → 落到次优 cell 2 → [x=2, y=0]
    bs = _bs(deployed=[{"name": "支援干员", "row": 1, "col": 3}])
    act = decode_action(*heads, ctx, bs)
    assert act is not None
    assert act["location"] == [2, 0]


def test_skill_retreat_target_must_be_deployed():
    """skill 的模型首选（未部署槽位）非法 → 回退到已部署槽位。"""
    ctx = _ctx()
    heads = _four_heads(ctx, [TYPE_TO_IDX["skill"]], [_SLOT1, _SLOT0], [0], [0])
    bs = _bs(deployed=[{"name": "玫兰莎", "row": 0, "col": 0}])
    act = decode_action(*heads, ctx, bs)
    assert act == {"type": "Skill", "name": "玫兰莎"}


def test_type_switch_when_preferred_type_unfeasible():
    """首选类型整体不可行（无任何合法目标）时换下一类型，不是返回非法动作。

    语义闸门下 speed 需要「纯等待死亡时间」：bullet_time 是有意义动作之一，
    故本用例一并关掉它，构造真·死亡时间。
    """
    ctx = _ctx()
    # deploy 首选，但手牌全空、词表目标被关 → deploy 判死 → 退到 speed
    heads = _four_heads(
        ctx,
        [TYPE_TO_IDX["deploy"], TYPE_TO_IDX["speed"]],
        [_SLOT0],
        [7],
        [0],
    )
    cfg = InferConfig(allow_vocab_targets=False, allow_bullet_time=False)
    act = decode_action(*heads, ctx, _bs(hand=[]), cfg)
    assert act == {"type": "SpeedUp"}


def test_all_illegal_returns_none():
    """全部候选类型判死（含 speed/bullet_time 被开关关闭）→ None。"""
    ctx = _ctx()
    heads = _four_heads(ctx, [TYPE_TO_IDX["deploy"]], [_SLOT0], [7], [0])
    cfg = InferConfig(allow_speedup=False, allow_bullet_time=False, allow_vocab_targets=False)
    act = decode_action(*heads, ctx, _bs(hand=[], deployed=[]), cfg)
    assert act is None


def test_undeployable_grid_cell_masked():
    """grid 第 DEPLOYABLE_CHANNEL 通道为 0 的格子不可部署，必须屏蔽。"""
    ctx = _ctx()
    ctx.cells[7, 2] = 0.0  # cell 7 标为不可部署
    heads = _four_heads(ctx, [TYPE_TO_IDX["deploy"]], [_SLOT0], [7, 2], [0])
    act = decode_action(*heads, ctx, _bs())
    assert act is not None
    assert act["location"] == [2, 0]


# ---------------------------------------------------------------- 3.5 推理噪声闸门（v1.1）


def test_speed_gate_dead_time_allows_upshift():
    """① 纯等待死亡时间（无任何合法有意义动作）允许升 2x。

    死亡时间构造：手牌空 + 词表目标通道关闭（deploy 死）、无部署（skill/
    retreat 死）、bullet_time 关闭 → 只剩 SpeedUp 可走。
    """
    ctx = _ctx()
    heads = _four_heads(ctx, [TYPE_TO_IDX["speed"]], [0], [0], [0])
    cfg = InferConfig(allow_bullet_time=False, allow_vocab_targets=False)
    bs = _bs(hand=[], deployed=[], in_speedup=False)
    assert decode_action(*heads, ctx, bs, cfg) == {"type": "SpeedUp"}


def test_speed_gate_downshift_before_real_action():
    """② 2x 下模型正要下发真实动作（deploy 可行）→ 本拍先降回 1x。"""
    ctx = _ctx()
    # type logits 首选 deploy：模型「正要」下发真实动作
    heads = _four_heads(ctx, [TYPE_TO_IDX["deploy"]], [_SLOT0], [7], [0])
    bs = _bs(in_speedup=True)
    assert decode_action(*heads, ctx, bs) == {"type": "SpeedUp"}
    # slow_for_action 关掉则不拦：动作照常在 2x 下下发
    cfg = InferConfig(slow_for_action=False)
    act = decode_action(*heads, ctx, bs, cfg)
    assert act is not None and act["type"] == "Deploy"


def test_speed_gate_hysteresis_blocks_second_switch():
    """③ 迟滞内禁止二次换速（距上次换速不足 speed_min_interval_s）。"""
    ctx = _ctx()
    heads = _four_heads(ctx, [TYPE_TO_IDX["speed"]], [0], [0], [0])
    # speed_min_interval_s 默认 5s / 节拍 1s；词表目标关掉做真·死亡时间
    cfg = InferConfig(allow_bullet_time=False, allow_vocab_targets=False)
    bs = _bs(hand=[], deployed=[], in_speedup=False)
    history = [{"type": "SpeedUp"}]
    # 距上次换速 1 拍 = 1s < 5s → speed 被屏蔽 → 全部判死返回 None
    assert decode_action(*heads, ctx, bs, cfg, history=history, step_index=1) is None
    # 间隔足够（10s ≥ 5s）→ 允许再次换速
    act = decode_action(*heads, ctx, bs, cfg, history=history, step_index=10)
    assert act == {"type": "SpeedUp"}


def test_speed_gate_masks_speed_when_meaningful_action_available():
    """④ 有合法有意义动作时（非死亡时间）speed 候选被屏蔽。"""
    ctx = _ctx()
    # speed 首选、deploy 次选：手牌可部署 → 非死亡时间 → 必须落到 deploy
    heads = _four_heads(
        ctx,
        [TYPE_TO_IDX["speed"], TYPE_TO_IDX["deploy"]],
        [_SLOT0],
        [7],
        [0],
    )
    act = decode_action(*heads, ctx, _bs(in_speedup=False))
    assert act is not None and act["type"] == "Deploy"
    # 闸门关闭 → 旧行为：按 logit 顺序走 speed
    cfg = InferConfig(speed_semantic_gate=False)
    act2 = decode_action(*heads, ctx, _bs(in_speedup=False), cfg)
    assert act2 == {"type": "SpeedUp"}


def test_speed_gate_upshift_still_needs_hysteresis_and_not_double():
    """已在 2x 且无降速理由时 speed 被屏蔽（不允许无理由来回切）。"""
    ctx = _ctx()
    heads = _four_heads(ctx, [TYPE_TO_IDX["speed"]], [0], [0], [0])
    cfg = InferConfig(allow_bullet_time=False, allow_vocab_targets=False)
    # 2x + 死亡时间：降速没有理由（没有真实动作要下发）→ speed 屏蔽 → None
    assert decode_action(*heads, ctx, _bs(hand=[], in_speedup=True), cfg) is None


def test_retreat_logit_margin_gate():
    """撤退余量闸门：retreat logit − 其余类型最高 logit < 余量 → 屏蔽。"""
    ctx = _ctx()
    bs = _bs(hand=[], deployed=[{"name": "玫兰莎", "row": 0, "col": 0}])
    tgt = _logits(1 + ctx.vocab.size + len(ctx.squad_names), [_SLOT0])
    cell = _logits(_N_CELLS, [0])
    d = _logits(len(DIRECTIONS), [0])
    # retreat 0.5 vs deploy 0.0：margin 0.5 < 默认 1.0（词表目标关掉，否则
    # 空手牌时 deploy 仍借词表装置可行，测不到「全部判死」）
    tl = np.asarray([0.0, -10.0, 0.5, -10.0, -10.0])  # deploy/skill/retreat/speed/bt
    cfg = InferConfig(allow_bullet_time=False, allow_vocab_targets=False)
    act = decode_action(tl, tgt, cell, d, ctx, bs, cfg)
    # retreat 被余量闸门屏蔽 → 受限 argmax 回落到唯一合法的 skill（同一已部署干员）
    assert act == {"type": "Skill", "name": "玫兰莎"}, "余量不足的撤退未被屏蔽"
    act2 = decode_action(
        tl,
        tgt,
        cell,
        d,
        ctx,
        bs,
        InferConfig(
            retreat_min_logit_margin=0.0,
            allow_bullet_time=False,
            allow_vocab_targets=False,
        ),
    )
    assert act2 == {"type": "Retreat", "name": "玫兰莎"}


def test_retreat_min_interval_gate():
    """两次撤退最小步数间隔：间隔不足时屏蔽、足够时放行。"""
    ctx = _ctx()
    bs = _bs(hand=[], deployed=[{"name": "玫兰莎", "row": 0, "col": 0}])
    heads = _four_heads(ctx, [TYPE_TO_IDX["retreat"]], [_SLOT0], [0], [0])
    cfg = InferConfig(
        allow_bullet_time=False,
        allow_vocab_targets=False,
        retreat_min_logit_margin=0.0,
    )
    history = [{"type": "Retreat", "name": "玫兰莎"}]
    # 间隔 2 < 3：撤退被屏蔽，受限 argmax 回落到同样合法的 skill
    act_blocked = decode_action(*heads, ctx, bs, cfg, history=history, step_index=2)
    assert act_blocked == {"type": "Skill", "name": "玫兰莎"}
    act = decode_action(*heads, ctx, bs, cfg, history=history, step_index=5)
    assert act == {"type": "Retreat", "name": "玫兰莎"}


def test_repeat_guard_masks_repeated_target():
    """重复护栏：同一 (type, 目标) 连续 ≥ repeat_guard 次 → 该目标候选被剔除。"""
    ctx = _ctx()
    heads = _four_heads(ctx, [TYPE_TO_IDX["deploy"]], [_SLOT0, _SLOT1], [7], [0])
    history = [
        {"type": "Deploy", "name": "玫兰莎"},
        {"type": "Deploy", "name": "玫兰莎"},
        {"type": "Deploy", "name": "玫兰莎"},
    ]
    act = decode_action(*heads, ctx, _bs(), InferConfig(), history=history)
    assert act is not None and act["type"] == "Deploy"
    assert act["name"] == "克洛丝", "连击 3 次的 玫兰莎 未被重复护栏剔除"
    # 连击 2 次 < 3 → 不拦
    act2 = decode_action(*heads, ctx, _bs(), InferConfig(), history=history[:2])
    assert act2 is not None and act2["name"] == "玫兰莎"


def test_deploy_duplicate_deployed_name_masked():
    """同名干员已在 deployed 时屏蔽该名 deploy 候选（hand 快照滞后场景）。"""
    ctx = _ctx()
    heads = _four_heads(ctx, [TYPE_TO_IDX["deploy"]], [_SLOT0, _SLOT1], [7], [0])
    bs = _bs(deployed=[{"name": "玫兰莎", "row": 0, "col": 0}])  # hand 快照仍带着她
    act = decode_action(*heads, ctx, bs)
    assert act is not None and act["name"] == "克洛丝"


# ---------------------------------------------------------------- 4. 历史编码同口径


def test_encode_action_for_history_matches_training_layout():
    """encode_action_for_history 的指针布局必须与 encode_episode 一致。"""
    ctx = _ctx()
    # 编队槽位：1 + vocab.size + slot = 1 + 2 + 1 = 4
    assert encode_action_for_history(
        {"type": "Deploy", "name": "克洛丝", "location": [3, 1], "direction": "右"},
        ctx,
    ) == (TYPE_TO_IDX["deploy"], _SLOT1, 7, DIR_TO_IDX["Right"])
    # 全局词表目标（非编队干员）走 index_of 段
    assert (
        encode_action_for_history({"type": "Deploy", "name": "支援装置", "location": [0, 0]}, ctx)[
            1
        ]
        == _VOCAB_DEV
    )
    # 未登录目标落 UNK；无 location 的 skill 落到 cell 0；缺省朝向落 None
    t, tgt, cell, d = encode_action_for_history({"type": "Skill", "name": "不存在的人"}, ctx)
    assert (t, tgt, cell, d) == (TYPE_TO_IDX["skill"], 0, 0, DIR_TO_IDX["None"])


# ---------------------------------------------------------------- 5. greedy_step 冒烟（合成模型）


_JOB_TYPES = {"Deploy", "Skill", "Retreat", "SpeedUp", "BulletTime"}


def test_greedy_step_smoke_with_random_model():
    """随机初始化的迷你 TrackAPolicyV1 跑两步贪心解码不炸，输出是作业形态动作。"""
    torch.manual_seed(0)
    ctx = _ctx()
    model = TrackAPolicyV1(
        featvec_dim=_FEAT_DIM,
        vocab_size=ctx.vocab.size,
        max_squad=MAX_SQUAD,
        state_dim=STATE_DIM,
        d_model=32,
        n_heads=4,
        n_layers=1,
        n_dec_layers=1,
    ).eval()
    cfg = InferConfig()

    act = greedy_step(model, ctx, _bs(), [], t=0, config=cfg)
    assert act is None or act["type"] in _JOB_TYPES
    if act is None:
        return
    if act["type"] == "Deploy":
        assert "location" in act and "direction" in act
    # 把模型给出的动作喂回历史再推一步：自回归历史编码链路（含右移）不炸
    act2 = greedy_step(model, ctx, _bs(), [act], t=1, config=cfg)
    assert act2 is None or act2["type"] in _JOB_TYPES


def test_greedy_step_returns_none_outside_battle():
    """in_battle 为假时直接 None（不做前向，无动作可下）。"""
    ctx = _ctx()
    model = TrackAPolicyV1(
        featvec_dim=_FEAT_DIM,
        vocab_size=ctx.vocab.size,
        max_squad=MAX_SQUAD,
        state_dim=STATE_DIM,
        d_model=32,
        n_heads=4,
        n_layers=1,
        n_dec_layers=1,
    ).eval()
    assert greedy_step(model, ctx, _bs(in_battle=False), [], t=0) is None


def test_greedy_step_channel_mismatch_fails_fast():
    """ctx.cells 通道数与模型 grid_channels 不符必须 fail fast（不许静默错位前向）。"""
    ctx = _ctx()  # 合成 5 通道
    model = TrackAPolicyV1(
        featvec_dim=_FEAT_DIM,
        vocab_size=ctx.vocab.size,
        max_squad=MAX_SQUAD,
        state_dim=STATE_DIM,
        d_model=32,
        n_heads=4,
        n_layers=1,
        n_dec_layers=1,
        grid_channels=8,
    ).eval()
    with pytest.raises(ValueError, match="grid_channels"):
        greedy_step(model, ctx, _bs(), [], t=0)


def test_infer_config_yaml_loads():
    """configs/infer_v1.yaml 必须能被 InferConfig.from_yaml 解析且字段吻合。"""
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "infer_v1.yaml"
    cfg = InferConfig.from_yaml(cfg_path)
    assert cfg.decision_interval_s == 1.0
    assert cfg.total_proxy == 30
    assert cfg.cost_margin == 0.0
    assert cfg.allow_speedup is True
    assert cfg.allow_bullet_time is True
    assert cfg.allow_vocab_targets is True
    assert cfg.device == "cpu"
    # v1.1 噪声闸门（yaml noise_gate 段）与 InferConfig 默认值吻合
    assert cfg.speed_semantic_gate is True
    assert cfg.slow_for_action is True
    assert cfg.speed_min_interval_s == 5.0
    assert cfg.retreat_min_logit_margin == 1.0
    assert cfg.retreat_min_interval_steps == 3
    assert cfg.repeat_guard == 3


# ---------------------------------------------------------------- 6. 真 ckpt 冒烟

_ARK_ML_ROOT = os.environ.get("ARK_ML_ROOT")


@pytest.mark.skipif(_ARK_ML_ROOT is None, reason="ARK_ML_ROOT 未设置，跳过真数据冒烟")
def test_real_checkpoint_first_record_smoke():
    """真 checkpoint + 真 npz 首条 record 过一遍 greedy_step（只验证链路不炸）。

    预测输出实打印、如实记录即可（冒烟模型，输出数字不作能力解读、不许调参）。
    """
    root = Path(str(_ARK_ML_ROOT))
    ckpt_path = root / "runs" / "imitation_v1" / "track_a_v1.pt"
    npz_path = root / "processed" / "state_action_v1.npz"
    if not ckpt_path.is_file() or not npz_path.is_file():
        pytest.skip("checkpoint 或数据集 npz 不存在")

    loaded = load_track_a_v1(ckpt_path)
    assert len(loaded.vocab.extra_tokens) > 0

    z = np.load(npz_path, allow_pickle=True)
    rec = z["records"][0]
    ctx = StageContext.from_record(rec, z["featvec"], loaded.vocab)
    assert ctx.squad_feat.shape[1] == int(loaded.meta["featvec_dim"])

    # record 不含实名 squad：hand 用占位槽名，至少让链路走到合法性层
    bs = {
        "in_battle": True,
        "in_speedup": False,
        "costs": 99.0,
        "kills": 0,
        "total_kills": 10,
        "hand": [
            {"name": nm, "cost": 1, "available": True, "cooling": False} for nm in ctx.squad_names
        ],
        "deployed": [],
    }
    act = greedy_step(loaded.model, ctx, bs, [], t=0)
    assert act is None or act["type"] in _JOB_TYPES
    print(f"\n[真 ckpt 冒烟] stage={ctx.stage_id} 首帧预测: {act}")
