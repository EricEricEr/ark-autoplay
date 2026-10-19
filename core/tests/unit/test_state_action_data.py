"""state_action_data（v1 状态-动作数据集）单测：合成数据，不含任何真实游戏内容。

重点覆盖 v1 相对 v0 新增、且错了后果严重的四件事：
1. **1:1 对齐与截断**：SkillDaemon 必须过滤（不产生 battle_state），过滤后动作与
   battle_states 逐位配对，条数不一致按 min 截断且计数；
2. **坐标序**：作业 deploy 的 [x,y] 必须经 normalize_deploy_location 转成 (row,col)；
3. **中文名映射**：opers 中文名 → char_id 命中进编队槽位，查不到计 UNK 统计；
4. **关卡留出**：holdout_stages（含 #f# 变体随基础关同侧）被排除出训练侧。
"""

from __future__ import annotations

import numpy as np
import pytest

# 训练依赖是可选的（`uv sync --group train`）：torch 缺失时整文件 skip（约定同
# test_imitation.py）；importorskip 判定后仍需 `import torch`，否则 pyright 会把
# `torch.Tensor` 注解判为「变量不能用作类型」。
pytest.importorskip("torch")
import torch  # noqa: E402

from ark_core.models.track_a.policy_v1 import TrackAPolicyV1  # noqa: E402
from ark_core.protocol.action import ACTION_TYPES, DIRECTIONS  # noqa: E402
from ark_core.training.imitate import TrainConfig, compute_loss  # noqa: E402
from ark_core.training.imitate_v1 import collate_v1, forward_batch_v1, split_records  # noqa: E402
from ark_core.training.imitation_data import MAX_SQUAD, Vocab  # noqa: E402
from ark_core.training.state_action_data import (  # noqa: E402
    STATE_DIM,
    BuildStatsV1,
    base_stage_id,
    encode_episode,
    state_vector,
)

ROWS, COLS = 8, 12
NAME2CID = {"能天使": "char_103_angel", "风笛": "char_222_bpipe"}
FEATVEC_IDX = {"char_103_angel": 0, "char_222_bpipe": 1}


def _sf() -> dict:
    """合成关卡静态特征（8×12，row-major 展平 5 通道，无出怪）。"""
    grid = []
    for r in range(ROWS):
        for c in range(COLS):
            grid.extend([r % 2, 1, 1, 0, 0])
    return {"stage_id": "main_99-01", "rows": ROWS, "cols": COLS, "grid": grid, "spawns": []}


def _bs(cost: float, kills: float = 0.0, total_kills: float = 10.0) -> dict:
    """合成一条 battle_state（cost 可区分用于校验对齐位置）。"""
    return {
        "costs": cost,
        "kills": kills,
        "total_kills": total_kills,
        "in_speedup": False,
        "hand": [{"cost": 11, "available": True, "cooling": False}],
        "deployed": [],
    }


def _job(actions: list[dict], opers: list[str] | None = None) -> dict:
    """合成作业文件内容。"""
    return {
        "stage_name": "9-1",
        "opers": [{"name": n, "skill": 1} for n in (opers or ["能天使", "风笛"])],
        "actions": actions,
    }


def _encode(
    job: dict,
    states: list[dict],
    stats: BuildStatsV1,
    vocab: Vocab | None = None,
    events: list[dict] | None = None,
) -> dict | None:
    """本文件 encode_episode 调用糖（全部合成输入，不落盘）。"""
    return encode_episode(
        episode_id="job-0-20261019-000000",
        stage_id="main_99-01",
        job=job,
        battle_states=states,
        action_events=events,
        sf=_sf(),
        name2cid=NAME2CID,
        featvec_idx=FEATVEC_IDX,
        vocab=vocab or Vocab(),
        stats=stats,
    )


# ---------------------------------------------------------------- 对齐与截断


def test_alignment_skill_daemon_filtered_and_truncated() -> None:
    """SkillDaemon 过滤 + 动作多于状态按 min 截断；state 行必须对位。"""
    stats = BuildStatsV1()
    rec = _encode(
        _job(
            [
                {"type": "SpeedUp"},
                {"type": "Deploy", "name": "风笛", "location": [9, 2], "direction": "Down"},
                {"type": "SkillDaemon"},
                {"type": "Deploy", "name": "能天使", "location": [8, 6], "direction": "Up"},
                {"type": "Skill", "name": "能天使"},
            ]
        ),
        [_bs(5.0), _bs(7.0)],
        stats,
    )
    assert rec is not None
    # 过滤 SkillDaemon 后 4 动作 vs 2 状态 → T=2：speed + deploy(风笛)
    assert rec["type"].tolist() == [
        ACTION_TYPES.index("speed"),
        ACTION_TYPES.index("deploy"),
    ]
    assert rec["state_feat"].shape == (2, STATE_DIM)
    # 第 0/1 行的 cost 维必须分别来自第 0/1 条 battle_state（对位不错位）
    assert rec["state_feat"][0, 0] == pytest.approx(5.0 / 100.0)
    assert rec["state_feat"][1, 0] == pytest.approx(7.0 / 100.0)
    assert stats.n_skill_daemon == 1
    assert stats.n_episode_len_mismatch == 1
    assert stats.n_actions_dropped_tail == 2
    assert stats.n_extra_states_dropped == 0


def test_extra_battle_states_dropped() -> None:
    """状态多于动作（结算后尾部状态）时丢弃多余状态并计数。"""
    stats = BuildStatsV1()
    rec = _encode(
        _job([{"type": "Deploy", "name": "风笛", "location": [1, 1], "direction": "Left"}]),
        [_bs(1.0), _bs(2.0), _bs(3.0)],
        stats,
    )
    assert rec is not None
    assert len(rec["type"]) == 1
    assert rec["state_feat"].shape == (1, STATE_DIM)
    assert rec["state_feat"][0, 0] == pytest.approx(1.0 / 100.0)
    assert stats.n_episode_len_mismatch == 1
    assert stats.n_extra_states_dropped == 2
    assert stats.n_actions_dropped_tail == 0


def test_alignment_check_reports_mismatch() -> None:
    """CopilotAction 事件流类型校验：错位计入 n_align_check_mismatch。"""
    stats = BuildStatsV1()
    events = [
        {"details": {"what": "CopilotAction", "details": {"action": "SwitchSpeed"}}},
        {"details": {"what": "CopilotAction", "details": {"action": "UseSkill"}}},
    ]
    rec = _encode(
        _job(
            [
                {"type": "SpeedUp"},
                {"type": "Deploy", "name": "风笛", "location": [1, 1], "direction": "Up"},
            ]
        ),
        [_bs(1.0), _bs(2.0)],
        stats,
        events=events,
    )
    assert rec is not None
    assert stats.n_align_check_mismatch == 1  # 第二位 Deploy 被事件流报成 UseSkill


# ---------------------------------------------------------------- 坐标转换


def test_deploy_location_xy_to_row_col() -> None:
    """[x,y]=[col,row] → cell = row*cols + col；越界计数。"""
    stats = BuildStatsV1()
    rec = _encode(
        _job(
            [
                {"type": "Deploy", "name": "风笛", "location": [9, 2], "direction": "Down"},
                {"type": "Deploy", "name": "能天使", "location": [99, 99], "direction": "Up"},
            ]
        ),
        [_bs(1.0), _bs(2.0)],
        stats,
    )
    assert rec is not None
    assert rec["cell"][0] == 2 * COLS + 9
    assert stats.n_cell_oob == 1  # [99,99] 越界落 0 并计数
    assert rec["cell"][1] == 0
    assert rec["dir"][0] == DIRECTIONS.index("Down")


# ---------------------------------------------------------------- 中文名映射


def test_name_to_char_id_hit_and_unk() -> None:
    """opers 命中 char_id 进 squad 槽位；未登录名保留槽位但 featvec 置 -1 并计数。"""
    stats = BuildStatsV1()
    vocab = Vocab(extra_tokens=["name:障碍物"])
    rec = _encode(
        _job(
            [
                {"type": "Deploy", "name": "能天使", "location": [1, 1], "direction": "Up"},
                {"type": "Deploy", "name": "障碍物", "location": [2, 2], "direction": "Up"},
                {"type": "Deploy", "name": "不存在的装置", "location": [3, 3], "direction": "Up"},
            ],
            opers=["能天使", "未登录干员"],
        ),
        [_bs(1.0), _bs(2.0), _bs(3.0)],
        stats,
        vocab=vocab,
    )
    assert rec is not None
    # 编队：能天使 → featvec 行 0；未登录干员 → -1（零向量槽），UNK 计数 1
    assert rec["squad_idx"].tolist() == [0, -1]
    assert stats.n_squad_name_unk == 1
    # 干员目标 → 编队槽位：1 + vocab.size + 0
    assert rec["target"][0] == 1 + vocab.size + 0
    # 词表内装置 → 词表索引（extra_tokens[0] → 1）
    assert rec["target"][1] == 1
    # 词表外装置 → UNK
    assert rec["target"][2] == 0
    assert stats.n_target_unk == 1
    assert stats.n_operator_deploy == 1
    assert stats.n_external_deploy == 2


def test_operator_not_in_squad_goes_unk() -> None:
    """featvec 里有、但不在本营编队的干员名 → UNK（v0 口径照抄）。"""
    stats = BuildStatsV1()
    rec = _encode(
        _job(
            [{"type": "Deploy", "name": "风笛", "location": [1, 1], "direction": "Up"}],
            opers=["能天使"],
        ),
        [_bs(1.0)],
        stats,
    )
    assert rec is not None
    assert rec["target"][0] == 0
    assert stats.n_target_unk == 1


# ---------------------------------------------------------------- 关卡留出


def test_split_records_excludes_holdout_stages() -> None:
    """holdout_stages 按基础 id 匹配，#f# 变体随基础关进 holdout。"""
    recs = [
        {"stage_id": "main_01-05"},
        {"stage_id": "main_01-05#f#"},
        {"stage_id": "main_02-02"},
        {"stage_id": "camp_02"},
    ]
    train, hold = split_records(recs, {"main_01-05"})
    assert [r["stage_id"] for r in hold] == ["main_01-05", "main_01-05#f#"]
    assert [r["stage_id"] for r in train] == ["main_02-02", "camp_02"]
    assert base_stage_id("main_01-05#f#") == "main_01-05"


# ---------------------------------------------------------------- 状态向量语义


def test_state_vector_dims() -> None:
    """state_vector 各维按 docstring 定义取值（kills clip、空手牌、步进）。"""
    v = state_vector(_bs(cost=50.0, kills=15.0, total_kills=10.0), t=0, total=4, n_squad_job=2)
    assert v.shape == (STATE_DIM,)
    assert v[0] == pytest.approx(0.5)            # cost/100
    assert v[1] == pytest.approx(1.0)            # kills_ratio clip 到 1
    assert v[3] == pytest.approx(1.0 / 12.0)     # hand_size
    assert v[6] == pytest.approx(11.0 / 30.0)    # mean_hand_cost
    assert v[9] == pytest.approx(0.0)            # step=0/4
    assert v[10] == pytest.approx(11.0 / 100.0)  # remain_hand_cost
    assert v[11] == 0.0                          # reserved

    empty = state_vector({"costs": 0, "hand": [], "deployed": []}, t=3, total=4, n_squad_job=0)
    assert empty[6] == 0.0 and empty[8] == 0.0 and empty[9] == pytest.approx(0.75)


# ---------------------------------------------------------------- 链路冒烟（collate + 模型）


def test_collate_and_policy_v1_forward() -> None:
    """collate_v1 补齐 + policy_v1 前向：形状钉死，loss 可反传。"""
    stats = BuildStatsV1()
    rec1 = _encode(
        _job(
            [
                {"type": "SpeedUp"},
                {"type": "Deploy", "name": "风笛", "location": [9, 2], "direction": "Down"},
            ]
        ),
        [_bs(1.0), _bs(2.0)],
        stats,
    )
    rec2 = _encode(
        _job([{"type": "Deploy", "name": "能天使", "location": [8, 6], "direction": "Up"}]),
        [_bs(3.0)],
        BuildStatsV1(),
    )
    assert rec1 is not None and rec2 is not None

    featvec = np.asarray([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]], dtype=np.float32)
    batch = collate_v1([rec1, rec2], featvec, STATE_DIM)
    assert batch["state"].shape == (2, 2, STATE_DIM)
    assert batch["lengths"].tolist() == [2, 1]
    # state padding 行为 0，且第 0 局第 1 步 state 保留原值
    assert batch["state"][0, 1, 0] == pytest.approx(2.0 / 100.0)
    assert batch["state"][1, 1].abs().sum() == 0.0

    model = TrackAPolicyV1(
        featvec_dim=3,
        vocab_size=1,
        max_squad=MAX_SQUAD,
        state_dim=STATE_DIM,
        d_model=32,
        n_heads=4,
        n_layers=1,
        n_dec_layers=1,
        dropout=0.0,
    )
    out = forward_batch_v1(model, batch)
    b, t = batch["type"].shape
    assert out["type"].shape == (b, t, len(ACTION_TYPES))
    assert out["target"].shape == (b, t, 1 + 1 + MAX_SQUAD)
    assert out["dir"].shape == (b, t, len(DIRECTIONS))
    assert out["cell"].shape == (b, t, batch["cells"].shape[1])

    cfg = TrainConfig()
    loss, _parts = compute_loss(out, batch, cfg)
    loss.backward()
    grad = model.state_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
