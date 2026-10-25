"""state_action_data（v1 状态-动作数据集）单测：合成数据，不含任何真实游戏内容。

重点覆盖 v1 相对 v0 新增、且错了后果严重的四件事：
1. **1:1 对齐与截断**：SkillDaemon 必须过滤（不产生 battle_state），过滤后动作与
   battle_states 逐位配对，条数不一致按 min 截断且计数；
2. **坐标序**：作业 deploy 的 [x,y] 必须经 normalize_deploy_location 转成 (row,col)；
3. **中文名映射**：opers 中文名 → char_id 命中进编队槽位，查不到计 UNK 统计；
4. **关卡留出**：holdout_stages（含 #f# 变体随基础关同侧）被排除出训练侧。
"""

from __future__ import annotations

import json
from pathlib import Path

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
    ExclusionInfo,
    base_stage_id,
    build_dataset,
    collect_exclusions,
    encode_episode,
    parse_scan_blockers,
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


def _sf_with_route() -> dict:
    """带路线几何的合成关卡：起点 (0,0)，终点 (7,11)，单行直线路线。"""
    sf = _sf()
    grid = sf["grid"]
    for r in range(ROWS):
        for c in range(COLS):
            i = (r * COLS + c) * 5
            grid[i + 3] = int((r, c) == (0, 0))
            grid[i + 4] = int((r, c) == (7, 11))
    sf["routes"] = [
        {
            "mode": 0,
            "start": [0, 0],
            "end": [7, 11],
            "n_points": 2,
            "points": [0, 0, 7, 11],
            "wait_ms": [],
        }
    ]
    return sf


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
    sf: dict | None = None,
    grid_channels: int = 5,
) -> dict | None:
    """本文件 encode_episode 调用糖（全部合成输入，不落盘）。"""
    return encode_episode(
        episode_id="job-0-20261019-000000",
        stage_id="main_99-01",
        job=job,
        battle_states=states,
        action_events=events,
        sf=sf or _sf(),
        name2cid=NAME2CID,
        featvec_idx=FEATVEC_IDX,
        vocab=vocab or Vocab(),
        stats=stats,
        grid_channels=grid_channels,
    )


# ---------------------------------------------------------------- v1.1 路线几何通道


def test_grid_channels_v1_1_appends_route_feats() -> None:
    """grid_channels=8：前 5 通道与 v1 完全一致，后 3 通道是路线几何派生量。"""
    stats = BuildStatsV1()
    rec8 = _encode(
        _job([{"type": "Deploy", "name": "风笛", "location": [1, 1], "direction": "Up"}]),
        [_bs(1.0)],
        stats,
        sf=_sf_with_route(),
        grid_channels=8,
    )
    assert rec8 is not None
    assert rec8["cells"].shape == (ROWS * COLS, 8)
    # 前 5 通道与 5 通道口径逐格一致
    stats5 = BuildStatsV1()
    rec5 = _encode(
        _job([{"type": "Deploy", "name": "风笛", "location": [1, 1], "direction": "Up"}]),
        [_bs(1.0)],
        stats5,
        sf=_sf_with_route(),
    )
    assert rec5 is not None
    assert rec5["cells"].shape == (ROWS * COLS, 5)
    assert np.array_equal(rec8["cells"][:, :5], rec5["cells"])
    # 路线格子：起点 dist_start=0 / dist_end=1；终点反之；on_route=1
    start = rec8["cells"][0 * COLS + 0, 5:]
    end = rec8["cells"][7 * COLS + 11, 5:]
    assert start.tolist() == [pytest.approx(0.0), pytest.approx(1.0), 1.0]
    assert end.tolist() == [pytest.approx(1.0), pytest.approx(0.0), 1.0]
    assert stats.n_stages_no_route_feat == 0


def test_grid_channels_v1_1_no_routes_zero_padded() -> None:
    """无路线几何的关：3 通道全零 + 计数（剿灭等无路线关的降级口径）。"""
    stats = BuildStatsV1()
    rec = _encode(
        _job([{"type": "Deploy", "name": "风笛", "location": [1, 1], "direction": "Up"}]),
        [_bs(1.0)],
        stats,
        grid_channels=8,
    )
    assert rec is not None
    assert rec["cells"].shape == (ROWS * COLS, 8)
    assert float(rec["cells"][:, 5:].sum()) == 0.0
    assert stats.n_stages_no_route_feat == 1


def test_grid_channels_rejects_unknown_value() -> None:
    """只支持 5/8：其它值报错而不是静默渲染。"""
    with pytest.raises(ValueError, match="grid_channels"):
        _encode(
            _job([{"type": "SpeedUp"}]),
            [_bs(1.0)],
            BuildStatsV1(),
            grid_channels=6,
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


# ---------------------------------------------------------------- 胜负口径清洗（v2）


def test_parse_scan_blockers_hit_lines() -> None:
    """HIT 行解析：抽 job 目录名与原因；非 HIT 行忽略；broad sample 区段也算。"""
    content = """group                                            n  prts  lost  sync   none  skip
4-7/job-22241-20261020-072148                   71     0     2     0     69     0
   HIT job-22241-20261020-072148/072_tick.png:settle_lost {'kind': 'settle_lost'}
   HIT job-22241-20261020-072148/073_settle.png:settle_lost {'kind': 'settle_lost'}
   HIT job-55778-20261020-072643/064_settle.png:prts_level_low {'kind': 'prts_level_low'}
valid main_01-05 job-23200                      79     0     0     0     79     0
quarantine sample (50 eps)                     250     0     0     0    250     0
broad sample                                   413     4    29     0    379     1
   HIT job-21097-20261011-180350/031_settle.png:settle_lost {'kind': 'settle_lost'}
   HIT job-99999-20260101-000000/001_tick.png:sync_updated {'kind': 'sync_updated'}
"""
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".txt", delete=False
    ) as f:
        f.write(content)
        path = Path(f.name)
    hits = parse_scan_blockers(path)
    path.unlink()
    assert hits == {
        "job-22241-20261020-072148": "settle_lost",      # 同 job 多帧去重
        "job-55778-20261020-072643": "prts_level_low",
        "job-21097-20261011-180350": "settle_lost",      # broad sample 区段
        "job-99999-20260101-000000": "sync_updated",
    }


def _mk_episode_dir(root: Path, stage: str, name: str, meta: dict | None) -> Path:
    """在合成 episodes 树下建一局（``meta=None`` 则是个空壳目录）。"""
    ed = root / stage / name
    ed.mkdir(parents=True)
    if meta is not None:
        (ed / "episode.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
    return ed


def test_collect_exclusions_field_side(tmp_path: Path) -> None:
    """字段侧：lost/unknown 命中；win/无字段/坏 JSON 不命中；名单文件可缺席。"""
    root = tmp_path / "episodes"
    _mk_episode_dir(root, "main_01-01", "job-a", {"battle_result": "win"})
    _mk_episode_dir(root, "main_01-01", "job-b", {"battle_result": "lost"})
    _mk_episode_dir(root, "main_01-02", "job-c", {"battle_result": "unknown"})
    _mk_episode_dir(root, "main_01-02", "job-d", {"stage_id": "main_01-02"})  # 老局无字段
    bad = _mk_episode_dir(root, "main_01-03", "job-e", None)
    (bad / "episode.json").write_text("{坏掉的", encoding="utf-8")
    _mk_episode_dir(root, "tmp", "job-junk", {"battle_result": "lost"})  # 垃圾目录跳过

    info = collect_exclusions(root, exclude_by_result=True, scan_file=None)
    assert info.by_field == {"job-b": "lost", "job-c": "unknown"}
    assert info.by_scan == {}
    assert info.union() == {"job-b", "job-c"}

    # 全关：不启用任何一侧 → 空排除集（v1 等价行为由 build_dataset 的 None 承担）
    off = collect_exclusions(root, exclude_by_result=False, scan_file=None)
    assert off.union() == set()


def test_collect_exclusions_union_dedup(tmp_path: Path) -> None:
    """名单/字段冲突以并集为准且去重；无 episode.json 的空壳局字段侧不出现。"""
    root = tmp_path / "episodes"
    _mk_episode_dir(root, "main_01-01", "job-both", {"battle_result": "lost"})
    _mk_episode_dir(root, "main_01-01", "job-field-only", {"battle_result": "unknown"})
    _mk_episode_dir(root, "main_01-01", "job-scan-only", {"battle_result": "win"})
    scan = tmp_path / "scan.txt"
    scan.write_text(
        "   HIT job-both/058_settle.png:settle_lost {}\n"
        "   HIT job-scan-only/031_settle.png:settle_lost {}\n",
        encoding="utf-8",
    )
    info = collect_exclusions(root, exclude_by_result=True, scan_file=scan)
    assert info.by_field == {"job-both": "lost", "job-field-only": "unknown"}
    assert info.by_scan == {"job-both": "settle_lost", "job-scan-only": "settle_lost"}
    assert info.union() == {"job-both", "job-field-only", "job-scan-only"}
    assert len(info.union()) == 3  # job-both 双侧命中只算一次


def _write_min_processed(processed: Path) -> None:
    """写最小可用 stagefeat / featvec parquet（供 build_dataset 集成路径用）。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    processed.mkdir(parents=True, exist_ok=True)
    sf = pa.table(
        {
            "stage_id": ["main_99-01"],
            "rows": [ROWS],
            "cols": [COLS],
            "grid": [_sf()["grid"]],
            "spawns": [[]],
            "opt_initial_cost": [10],
        }
    )
    pq.write_table(sf, processed / "stagefeat.parquet")
    fv = pa.table(
        {
            "char_id": ["char_103_angel", "char_222_bpipe"],
            "name": ["能天使", "风笛"],
            "rarity_tier": [6, 6],
            "sub_profession": ["x", "y"],
            "v0": [1.0, 0.0],
            "v1": [0.0, 1.0],
        }
    )
    pq.write_table(fv, processed / "featvec_operators.parquet")


def _mk_real_episode(root: Path, stage: str, name: str, tmp_path: Path) -> None:
    """建一语法完整的合成局（episode.json / battle_states.jsonl / 作业文件）。"""
    job_path = tmp_path / f"job_{name}.json"
    job_path.write_text(
        json.dumps(
            _job(
                [
                    {"type": "SpeedUp"},
                    {"type": "Deploy", "name": "风笛", "location": [1, 1], "direction": "Up"},
                ],
                opers=["能天使", "风笛"],
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    ed = _mk_episode_dir(
        root,
        stage,
        name,
        {"episode_id": name, "stage_id": stage, "job_file": str(job_path)},
    )
    (ed / "battle_states.jsonl").write_text(
        json.dumps(_bs(1.0)) + "\n" + json.dumps(_bs(2.0)) + "\n", encoding="utf-8"
    )


def test_build_dataset_accepts_moved_bundles_with_local_job(tmp_path: Path) -> None:
    """真实构建链可读采集端转换包，不依赖对方机器的绝对路径。"""
    root, processed = tmp_path / "moved/episodes", tmp_path / "processed"
    _write_min_processed(processed)
    name = "job-portable-20261026"
    _mk_real_episode(root, "main_99-01", name, tmp_path)
    ep = root / "main_99-01" / name
    old_job = tmp_path / f"job_{name}.json"
    old_job.rename(ep / "job.json")
    job_text = (ep / "job.json").read_text(encoding="utf-8")
    (ep / "job.json").write_text(job_text, encoding="utf-8-sig")
    state_text = (ep / "battle_states.jsonl").read_text(encoding="utf-8")
    (ep / "battle_states.jsonl").write_text(state_text, encoding="utf-8-sig")
    meta = json.loads((ep / "episode.json").read_text(encoding="utf-8"))
    meta["job_file"] = "job.json"
    (ep / "episode.json").write_text(json.dumps(meta), encoding="utf-8")
    _v, stats, n = build_dataset(
        processed, root, out_path=tmp_path / "portable.npz", holdout_stages=set(),
    )
    assert n == 1 and stats.n_kept == 1 and stats.drop_bad_files == 0


def test_build_dataset_applies_exclusion(tmp_path: Path) -> None:
    """集成：排除集在 build 主循环生效且计数；``exclusion=None`` 时 v1 行为不回归。"""
    root = tmp_path / "episodes"
    processed = tmp_path / "processed"
    _write_min_processed(processed)
    _mk_real_episode(root, "main_99-01", "job-keep-20261020-000001", tmp_path)
    _mk_real_episode(root, "main_99-01", "job-drop-20261020-000002", tmp_path)

    excl = ExclusionInfo(by_scan={"job-drop-20261020-000002": "settle_lost"})
    _vocab, stats, n = build_dataset(
        processed,
        root,
        out_path=tmp_path / "v2.npz",
        holdout_stages=set(),
        exclusion=excl,
    )
    assert (n, stats.n_kept, stats.n_seen, stats.n_excluded) == (1, 1, 2, 1)

    _vocab2, stats2, n2 = build_dataset(
        processed,
        root,
        out_path=tmp_path / "v1.npz",
        holdout_stages=set(),
        exclusion=None,
    )
    assert (n2, stats2.n_kept, stats2.n_excluded) == (2, 2, 0)


# ---------------------------------------------------------------- 增量构建（v3 钩子）


def test_build_dataset_delta_only_new(tmp_path: Path) -> None:
    """增量：base npz 旧记录原样保留，只编码新局，词表/featvec 冻结。"""
    from ark_core.training.state_action_data import build_dataset_delta

    root = tmp_path / "episodes"
    processed = tmp_path / "processed"
    _write_min_processed(processed)
    _mk_real_episode(root, "main_99-01", "job-old1-20261020-000001", tmp_path)
    _mk_real_episode(root, "main_99-01", "job-old2-20261020-000002", tmp_path)
    base_npz = tmp_path / "base.npz"
    _v, base_stats, base_n = build_dataset(
        processed, root, out_path=base_npz, holdout_stages=set()
    )
    assert (base_n, base_stats.n_kept) == (2, 2)

    # 后到 3 局（其中 1 局关卡在 stagefeat 查不到 → drop_no_stage，验证统计口径）
    _mk_real_episode(root, "main_99-01", "job-new1-20261021-000001", tmp_path)
    _mk_real_episode(root, "main_99-01", "job-new2-20261021-000002", tmp_path)
    _mk_real_episode(root, "main_99-01", "job-new3-20261021-000003", tmp_path)
    bad = _mk_episode_dir(
        root, "main_77-77", "job-nostage-20261021-000004", None
    )
    (bad / "battle_states.jsonl").write_text("{}\n", encoding="utf-8")
    bad_job = tmp_path / "job_bad.json"
    bad_job.write_text(json.dumps(_job([], opers=["能天使"])), encoding="utf-8")
    (bad / "episode.json").write_text(
        json.dumps(
            {"episode_id": "job-nostage-20261021-000004", "stage_id": "main_77-77",
             "job_file": str(bad_job)}
        ),
        encoding="utf-8",
    )

    out_npz = tmp_path / "delta.npz"
    vocab, stats, delta = build_dataset_delta(
        processed,
        root,
        base_npz=base_npz,
        out_path=out_npz,
        holdout_stages=set(),
    )
    assert delta["n_base_records"] == 2
    assert delta["n_new_records"] == 3
    assert delta["n_seen_skipped"] == 2
    assert delta["total_records"] == 5
    assert stats.n_kept == 3 and stats.drop_no_stage == 1
    # 旧记录原样在前、新记录依次在后；词表与 base 完全一致（冻结）
    z = np.load(out_npz, allow_pickle=True)
    ids = [str(r["episode_id"]) for r in z["records"]]
    assert ids[:2] == ["job-old1-20261020-000001", "job-old2-20261020-000002"]
    assert sorted(ids[2:]) == [
        "job-new1-20261021-000001",
        "job-new2-20261021-000002",
        "job-new3-20261021-000003",
    ]
    zb = np.load(base_npz, allow_pickle=True)
    assert [str(x) for x in z["vocab"]] == [str(x) for x in zb["vocab"]]
    assert np.array_equal(z["featvec"], zb["featvec"])
    # 再跑一遍同基底：没有新局则零增量（无 stage 的坏局不算 seen，每次都被重试并计数）
    _v2, _s2, delta2 = build_dataset_delta(
        processed, root, base_npz=out_npz, out_path=tmp_path / "delta2.npz",
        holdout_stages=set(),
    )
    assert delta2["n_new_records"] == 0 and delta2["n_seen_skipped"] == 5
    assert _s2.drop_no_stage == 1


# ---------------------------------------------------------------- 增量续训（v3 resume 冒烟）


def test_train_resume_from_ckpt(tmp_path: Path) -> None:
    """resume_from 加载旧权重续训：形状匹配时链路通畅（2 局 / 2 epoch 冒烟）。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ark_core.models.track_a.policy_v1 import TrackAPolicyV1
    from ark_core.training.imitate_v1 import train

    root = tmp_path / "episodes"
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    # 两关卡 stagefeat（第二关进 holdout，满足 train() 的非空断言）
    grid = _sf()["grid"]
    pq.write_table(
        pa.table(
            {
                "stage_id": ["main_99-01", "main_99-02"],
                "rows": [ROWS, ROWS],
                "cols": [COLS, COLS],
                "grid": [grid, grid],
                "spawns": [[], []],
                "opt_initial_cost": [10, 10],
            }
        ),
        processed / "stagefeat.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "char_id": ["char_103_angel", "char_222_bpipe"],
                "name": ["能天使", "风笛"],
                "rarity_tier": [6, 6],
                "sub_profession": ["x", "y"],
                "v0": [1.0, 0.0],
                "v1": [0.0, 1.0],
            }
        ),
        processed / "featvec_operators.parquet",
    )
    _mk_real_episode(root, "main_99-01", "job-t-20261021-000001", tmp_path)
    _mk_real_episode(root, "main_99-02", "job-h-20261021-000002", tmp_path)
    npz = tmp_path / "ds.npz"
    vocab, _stats, n = build_dataset(processed, root, out_path=npz, holdout_stages=set())
    assert n == 2

    ckpt = tmp_path / "old.pt"
    model0 = TrackAPolicyV1(
        featvec_dim=2, vocab_size=vocab.size, max_squad=MAX_SQUAD, state_dim=STATE_DIM,
        d_model=32, n_heads=4, n_layers=1, n_dec_layers=1, dropout=0.0,
    )
    torch.save({"state_dict": model0.state_dict()}, ckpt)

    rep = train(
        npz,
        TrainConfig(d_model=32, n_heads=4, n_layers=1, n_dec_layers=1, dropout=0.0, epochs=2),
        holdout_stages={"main_99-02"},
        out_dir=tmp_path / "run",
        device_str="cpu",
        log=lambda *_: None,
        ckpt_name="new.pt",
        resume_from=ckpt,
    )
    assert rep["n_train"] == 1 and rep["n_holdout"] == 1
    assert rep["best_epoch"] >= 1
    assert (tmp_path / "run" / "new.pt").is_file()
