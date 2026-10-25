"""控制动作可有回调和状态，过滤时必须保持剩余动作、状态的原始索引一致。"""

from bridge.convert_episodes import paired_samples


def _event(kind: str, time: int) -> dict:
    return {
        "t_ms": time,
        "details": {"what": "CopilotAction", "details": {"action": kind}},
    }


def test_control_state_and_event_are_removed_together() -> None:
    meta = {"episode_id": "synthetic", "stage_id": "synthetic", "battle_result": "win"}
    job = {
        "actions": [{"type": "SpeedUp"}, {"type": "SkillDaemon"}, {"type": "SpeedUp"}]
    }
    states = [
        {"t_ms": 1, "costs": 10},
        {"t_ms": 2, "costs": 99},
        {"t_ms": 3, "costs": 20},
    ]
    events = [
        _event("SwitchSpeed", 1),
        _event("SkillDaemon", 2),
        _event("SwitchSpeed", 3),
    ]
    samples, _ = paired_samples(meta, job, states, events)
    assert [s["source_state_index"] for s in samples] == [0, 2]
    assert [s["state"]["costs"] for s in samples] == [10, 20]


def test_event_only_control_does_not_consume_a_real_state() -> None:
    meta = {"episode_id": "synthetic", "stage_id": "synthetic", "battle_result": "win"}
    job = {"actions": [{"type": "SpeedUp"}, {"type": "SkillDaemon"}]}
    samples, _ = paired_samples(
        meta,
        job,
        [{"t_ms": 1, "costs": 10}],
        [_event("SwitchSpeed", 0), _event("SkillDaemon", 2)],
    )
    assert len(samples) == 1 and samples[0]["source_state_index"] == 0
