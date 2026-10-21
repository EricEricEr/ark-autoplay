"""轨迹写出器纯逻辑单测：目录、清单哈希、必填自检。"""

import json
from pathlib import Path

import pytest

from bridge.episode_writer import PROTOCOL_VERSION, EpisodeWriter, sha256_tag


def _fake_episode() -> dict:
    return {
        "episode_id": "replay-1-20260101-000000",
        "stage_id": "main_00-01",
        "stage_code": "0-1",
        "source": {"type": "prts.plus", "job_id": 1, "note": "ut"},
        "job_file": "/tmp/job_1.json",
        "result_screen_files": ["shots/001_settle.png"],
        "action_events_file": "action_events.jsonl",
        "shot_files": None,
        "duration_ms": 1234,
    }


def test_begin_creates_run_dir(tmp_path: Path) -> None:
    """begin 生成 episodes/<stage_id>/job-<id>-<ts>/shots 目录。"""
    writer = EpisodeWriter(tmp_path)
    run_dir = writer.begin("main_00-01", 48628)
    assert run_dir.parent.parent == tmp_path / "episodes"
    assert run_dir.parent.name == "main_00-01"
    assert run_dir.name.startswith("job-48628-")
    assert (run_dir / "shots").is_dir()


def test_events_and_episode_roundtrip(tmp_path: Path) -> None:
    """事件 jsonl 与 episode.json 写出可回读，清单带 sha256 短戳。"""
    writer = EpisodeWriter(tmp_path)
    run_dir = writer.begin("main_00-01", 1)
    (run_dir / "shots" / "001_tick.png").write_bytes(b"\x89PNG-synthetic-a")
    (run_dir / "shots" / "002_settle.png").write_bytes(b"\x89PNG-synthetic-b")

    writer.write_events(run_dir, [{"t_ms": 0, "msg": "TaskChainStart"}])
    ep_path = writer.write_episode(run_dir, _fake_episode())

    lines = (run_dir / "action_events.jsonl").read_text(encoding="utf-8")
    assert lines.count("\n") == 1 and "TaskChainStart" in lines

    ep = json.loads(ep_path.read_text(encoding="utf-8"))
    assert ep["protocol_version"] == PROTOCOL_VERSION
    assert set(ep["shot_files"]) == {"shots/001_tick.png", "shots/002_settle.png"}
    for rel, tag in ep["shot_files"].items():
        assert tag.startswith("sha256:") and len(tag) == len("sha256:") + 16
        assert tag == sha256_tag(run_dir / rel)


def test_validate_missing_fields(tmp_path: Path) -> None:
    """缺必填字段/空截图清单抛 ValueError。"""
    writer = EpisodeWriter(tmp_path)
    run_dir = writer.begin("main_00-01", 1)
    (run_dir / "shots" / "001_tick.png").write_bytes(b"\x89PNG-a")
    bad = _fake_episode()
    del bad["stage_code"]
    with pytest.raises(ValueError):
        writer.write_episode(run_dir, bad)


def test_validate_empty_shots_rejected(tmp_path: Path) -> None:
    """shots 为空时拒绝落盘（一局必须有帧）。"""
    writer = EpisodeWriter(tmp_path)
    run_dir = writer.begin("main_00-01", 1)
    with pytest.raises(ValueError):
        writer.write_episode(run_dir, _fake_episode())


def test_write_battle_states_roundtrip(tmp_path: Path) -> None:
    """战场状态 jsonl 写出可回读（每行一条，含 t_ms 与费用）。"""
    writer = EpisodeWriter(tmp_path)
    run_dir = writer.begin("main_00-01", 1)
    (run_dir / "shots" / "001_tick.png").write_bytes(b"\x89PNG-a")

    recs = [
        {"protocol_version": "0.1.0-battlestate-draft", "t_ms": 0, "costs": 10},
        {"protocol_version": "0.1.0-battlestate-draft", "t_ms": 1500, "costs": 14},
    ]
    path = writer.write_battle_states(run_dir, recs)
    assert path is not None and path.name == "battle_states.jsonl"
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert [x["costs"] for x in lines] == [10, 14]


def test_write_battle_states_empty_creates_no_file(tmp_path: Path) -> None:
    """空记录**不创建文件**——让"没采到状态"与"采到空序列"在文件层面可区分。

    这条很重要：用官方 MaaCore（无 BattleState 回调）时应表现为"文件不存在"，
    而不是"存在但为空的文件"——后者会被误读成"这局状态全丢"。
    """
    writer = EpisodeWriter(tmp_path)
    run_dir = writer.begin("main_00-01", 1)
    assert writer.write_battle_states(run_dir, []) is None
    assert not (run_dir / "battle_states.jsonl").exists()


def test_validate_shots_dropped_black_is_optional_nonneg_int(tmp_path: Path) -> None:
    """D9 标注字段：缺省合法（旧 bundle 兼容）；给出时必须是真整数且非负。"""
    writer = EpisodeWriter(tmp_path)
    run_dir = writer.begin("main_00-01", 1)
    (run_dir / "shots" / "001_tick.png").write_bytes(b"\x89PNG-a")

    ep = _fake_episode()
    ep["shots_dropped_black"] = 3
    written = json.loads(writer.write_episode(run_dir, ep).read_text(encoding="utf-8"))
    assert written["shots_dropped_black"] == 3

    ep_absent = _fake_episode()
    saved = json.loads(
        writer.write_episode(run_dir, ep_absent).read_text(encoding="utf-8")
    )
    assert "shots_dropped_black" not in saved

    for bad in (-1, True, "3", 1.5):
        ep_bad = _fake_episode()
        ep_bad["shots_dropped_black"] = bad
        with pytest.raises(ValueError):
            writer.write_episode(run_dir, ep_bad)


def test_validate_fail_stage_enum_and_orthogonality(tmp_path: Path) -> None:
    """fail_stage：限枚举；与 battle_result 正交（lost+done / lost+battle 均合法）。"""
    writer = EpisodeWriter(tmp_path)
    run_dir = writer.begin("main_00-01", 1)
    (run_dir / "shots" / "001_tick.png").write_bytes(b"\x89PNG-a")

    for combo in (
        ("lost", "done"),  # 打完但战败
        ("lost", "battle"),  # 未进结算的战斗失败
        ("win", "done"),
        ("unknown", "battle"),
        ("unknown", "formation"),
    ):
        ep = _fake_episode()
        ep["battle_result"], ep["fail_stage"] = combo
        writer.write_episode(run_dir, ep)  # 不抛即通过

    for bad in ("victory", "nav_failed", 3):
        ep_bad = _fake_episode()
        ep_bad["fail_stage"] = bad
        with pytest.raises(ValueError):
            writer.write_episode(run_dir, ep_bad)

    # 无字段 = 老局，合法
    ep_absent = _fake_episode()
    saved = json.loads(
        writer.write_episode(run_dir, ep_absent).read_text(encoding="utf-8")
    )
    assert "fail_stage" not in saved


def test_validate_battle_result_and_blockers(tmp_path: Path) -> None:
    """新字段：battle_result 限枚举；blockers_dismissed 限 [{kind,t_ms,n}] 形态。"""
    writer = EpisodeWriter(tmp_path)
    run_dir = writer.begin("main_00-01", 1)
    (run_dir / "shots" / "001_tick.png").write_bytes(b"\x89PNG-a")

    ep = _fake_episode()
    ep["battle_result"] = "lost"
    ep["blockers_dismissed"] = [
        {"kind": "settle_lost", "t_ms": 61234, "n": 1},
        {"kind": "prts_level_low", "t_ms": 1234, "n": 1},
    ]
    written = json.loads(writer.write_episode(run_dir, ep).read_text(encoding="utf-8"))
    assert written["battle_result"] == "lost"
    assert written["blockers_dismissed"][0]["kind"] == "settle_lost"

    for bad_result in ("victory", 1, True):
        ep_bad = _fake_episode()
        ep_bad["battle_result"] = bad_result
        with pytest.raises(ValueError):
            writer.write_episode(run_dir, ep_bad)

    for bad_blockers in (
        "settle_lost",
        [{"kind": "settle_lost"}],  # 缺 t_ms/n
        [{"kind": "settle_lost", "t_ms": "1", "n": 1}],
        [{"kind": 1, "t_ms": 1, "n": 1}],
    ):
        ep_bad = _fake_episode()
        ep_bad["blockers_dismissed"] = bad_blockers
        with pytest.raises(ValueError):
            writer.write_episode(run_dir, ep_bad)

