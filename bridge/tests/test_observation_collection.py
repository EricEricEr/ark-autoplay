"""零动作采集使用真实终局证据；观察数据不能伪装为动作监督数据。"""

from bridge.collection_quality import collection_status
from bridge.loop_runner import TickSample, observe_battle


def test_observation_waits_through_long_unchanging_and_missing_states():
    samples = [TickSample(state=None, heartbeat_ok=False) for _ in range(60)]
    samples += [TickSample(state={"costs": 10}, battle_active=True) for _ in range(80)]
    samples += [TickSample(state=None, terminal_result="win")]
    result = observe_battle(samples, deadline=0)
    assert result["entered_battle"]
    assert result["observed_terminal_result"] == "win"
    assert result["measurement_actions"] == 141
    assert result["measurement_failures"] == 60


def test_observation_win_is_counted_separately_from_labeled_win():
    summary = {
        "battle_result": "win",
        "battle_states": 100,
        "copilot_actions": 0,
        "observation_only": True,
        "entered_battle": True,
    }
    assert collection_status(summary) == {
        "valid_nonempty_win": False,
        "valid_observation_win": True,
        "valid_collected_win": True,
    }
    assert not collection_status({**summary, "prts_detected": ["UsePrts"]})[
        "valid_collected_win"
    ]
    assert not collection_status({**summary, "battle_result": "unknown"})[
        "valid_collected_win"
    ]


def test_observation_preserves_existing_expert_chain_timeout():
    result = observe_battle([TickSample(state=None)], deadline=10, clock=lambda: 11)
    assert result["observation_cause"] == "expert_chain_timeout"
    assert result["observed_terminal_result"] is None
