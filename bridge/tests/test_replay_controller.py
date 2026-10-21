"""replay_controller 纯函数单测：skip-done 收紧判定 + fail_stage 归类。

不构造 ReplayController（那个要连 MAA），只测模块级纯函数。
"""

from __future__ import annotations

from pathlib import Path

from bridge.replay_controller import classify_fail_stage, has_completed_episode


def _mk_run_dir(stage_dir: Path, name: str, *, with_episode: bool) -> None:
    d = stage_dir / name
    (d / "shots").mkdir(parents=True)
    (d / "shots" / "001_tick.png").write_bytes(b"\x89PNG-synthetic")
    if with_episode:
        (d / "episode.json").write_text("{}", encoding="utf-8")


def test_skip_done_requires_episode_json(tmp_path: Path) -> None:
    """半成品（仅 shots/）不再算完成；episode.json 存在才算；跨 job 不误认。"""
    stage = tmp_path / "main_00-01"
    _mk_run_dir(stage, "job-111-20260101-000000", with_episode=False)
    assert has_completed_episode(stage, "job-111-") is False

    _mk_run_dir(stage, "job-111-20260101-010000", with_episode=True)
    assert has_completed_episode(stage, "job-111-") is True
    assert has_completed_episode(stage, "job-222-") is False
    assert has_completed_episode(tmp_path / "不存在", "job-111-") is False


def test_classify_fail_stage() -> None:
    """fail_stage 归类：done/编队/进战前/作战中 四支。"""
    assert classify_fail_stage("TaskChainCompleted", []) == "done"
    assert classify_fail_stage("TaskChainCompleted", ["BattleFormationTask:"]) == "done"
    assert (
        classify_fail_stage("TaskChainError", ["BattleFormationTask:OperatorMissing"])
        == "formation"
    )
    start1 = classify_fail_stage("TaskChainError", ["ProcessTask:BattleStartPre"])
    start2 = classify_fail_stage("TaskChainError", ["ProcessTask:BattleStartAll"])
    assert (start1, start2) == ("start", "start")
    assert classify_fail_stage("TaskChainError", ["BattleProcessTask:"]) == "battle"
    assert classify_fail_stage("Timeout", []) == "battle"
    assert classify_fail_stage(None, []) == "battle"
