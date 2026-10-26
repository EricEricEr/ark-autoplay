"""采集结果分账：空作业观察与含回放动作的轨迹分别验收，不伪造标签。"""

from typing import Any


def collection_status(summary: dict[str, Any]) -> dict[str, bool]:
    """真实胜利且有状态、无代理污染，才可计入有效采集。"""
    states = summary.get("battle_states", 0)
    count = states.get("n", 0) if isinstance(states, dict) else states
    won = (
        summary.get("battle_result") == "win"
        and count > 0
        and not summary.get("prts_detected")
    )
    observation = bool(summary.get("observation_only"))
    labeled = bool(won and not observation and summary.get("copilot_actions", 0) > 0)
    observed = bool(won and observation and summary.get("entered_battle"))
    return {
        "valid_nonempty_win": labeled,
        "valid_observation_win": observed,
        "valid_collected_win": labeled or observed,
    }
