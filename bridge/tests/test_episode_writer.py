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

