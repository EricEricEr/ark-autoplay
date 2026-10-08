"""jobs_tiers 阵容练度分桶测试（合成阵容，不含真实游戏内容）。

5 桶语义（与真实数据验证计数对齐的规则，见 docs/adr/0002）：
核心干员 = 稀有度 ≥ TIER_5；单核 = 至多 1 名核心；标精二 = 任一槽位
req.elite ≥ 2 或 module > 0；无阵容信息 = 空阵容或全部槽位无法解析。
"""

from typing import Any

from ark_core.datapipe.jobs_tiers import (
    TIER_ALL_LOW_RARITY,
    TIER_MULTI_CORE,
    TIER_NO_ROSTER,
    TIER_SINGLE_CORE_BUDGET,
    TIER_SINGLE_CORE_MARKED_E2,
    tier_of_lineup,
)

# 合成干员稀有度：char_syn_t3 / t4 / t5 / t6 对应 3/4/5/6 星
_RARITY = {"char_syn_t3": 3, "char_syn_t4": 4, "char_syn_t5": 5, "char_syn_t6": 6}


def _rarity_of(cid: str) -> int | None:
    return _RARITY.get(cid)


def _slot(tier_id: str | None, elite: int = 0, module: int = 0) -> dict[str, Any]:
    """造一个合成阵容槽位；tier_id 为 None 表示无法解析的占位槽。"""
    return {
        "name": "占位",
        "id": tier_id,
        "skill": 1,
        "skill_usage": 0,
        "req": {"elite": elite, "level": 1, "skill_level": 1, "module": module, "potentiality": 0},
    }


def test_no_roster_empty() -> None:
    """空阵容 → 无阵容信息。"""
    assert tier_of_lineup([], _rarity_of) == TIER_NO_ROSTER
    assert tier_of_lineup(None, _rarity_of) == TIER_NO_ROSTER


def test_no_roster_all_unresolved() -> None:
    """全部槽位都是占位符（id 为空 / 查不到）→ 无阵容信息。"""
    assert tier_of_lineup([_slot(None), _slot("char_syn_gone")], _rarity_of) == TIER_NO_ROSTER


def test_all_low_rarity() -> None:
    """全员 ≤ 4 星 → 全 4 星以下。"""
    lineup = [_slot("char_syn_t4"), _slot("char_syn_t3")]
    assert tier_of_lineup(lineup, _rarity_of) == TIER_ALL_LOW_RARITY


def test_single_core_budget() -> None:
    """1 名核心 + 其余 ≤ 4 星 + 全队未标精二 → 单核平民。"""
    lineup = [_slot("char_syn_t6"), _slot("char_syn_t4"), _slot("char_syn_t3")]
    assert tier_of_lineup(lineup, _rarity_of) == TIER_SINGLE_CORE_BUDGET


def test_single_core_budget_five_star_core() -> None:
    """核心按 ≥ 5 星计数：单 5 星 + 全 4 星 → 单核平民（核心不止 TIER_6）。"""
    lineup = [_slot("char_syn_t5"), _slot("char_syn_t4")]
    assert tier_of_lineup(lineup, _rarity_of) == TIER_SINGLE_CORE_BUDGET


def test_single_core_marked_e2() -> None:
    """单核但有槽位标了精二 / 模组 → 单核但标精二。"""
    lineup_elite = [_slot("char_syn_t6"), _slot("char_syn_t4", elite=2)]
    assert tier_of_lineup(lineup_elite, _rarity_of) == TIER_SINGLE_CORE_MARKED_E2
    lineup_module = [_slot("char_syn_t6"), _slot("char_syn_t4", module=1)]
    assert tier_of_lineup(lineup_module, _rarity_of) == TIER_SINGLE_CORE_MARKED_E2


def test_multi_core_two_six_stars() -> None:
    """2 名 TIER_6 → 多核高配。"""
    lineup = [_slot("char_syn_t6"), _slot("char_syn_t6")]
    assert tier_of_lineup(lineup, _rarity_of) == TIER_MULTI_CORE


def test_multi_core_six_plus_five() -> None:
    """1 名 6 星 + 1 名 5 星 → 多核高配（5 星同样计核心，验证口径关键 case）。"""
    lineup = [_slot("char_syn_t6"), _slot("char_syn_t5")]
    assert tier_of_lineup(lineup, _rarity_of) == TIER_MULTI_CORE


def test_unresolved_slot_neutral() -> None:
    """占位槽不参与稀有度判断 / 不计核心：单 6 星 + 占位槽仍是单核。"""
    lineup = [_slot("char_syn_t6"), _slot(None)]
    assert tier_of_lineup(lineup, _rarity_of) == TIER_SINGLE_CORE_BUDGET


def test_req_null_means_unmarked() -> None:
    """req 为 null 视为未标精二。"""
    slot = _slot("char_syn_t6")
    slot["req"] = None
    assert tier_of_lineup([slot, _slot("char_syn_t4")], _rarity_of) == TIER_SINGLE_CORE_BUDGET
