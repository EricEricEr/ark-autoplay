"""battle_state 单测：MaaCore BattleState 回调 → 协议化状态序列。

重点覆盖：
1. 字段白名单（多余字段被剔除，缺失字段不补 0）；
2. 坐标序（上游 Point 的 x/y 与落盘 row/col 的对应——**最易错**）；
3. 空输入不伪造数据（"没采到"≠"采到 0"）。
"""

from __future__ import annotations

from bridge.battle_state import (
    STATE_PROTOCOL_VERSION,
    battle_state_to_record,
    summarize_states,
)


def _sample_details() -> dict:
    """一份典型的 BattleState details（字段名与 patch 0001 一致）。"""
    return {
        "in_battle": True,
        "in_speedup": False,
        "costs": 23,
        "kills": 12,
        "total_kills": 45,
        "camera_count": 1,
        "stage_name": "1-7",
        "camera_shift": [0.5, -0.25],
        "hand": [
            {
                "index": 0,
                "name": "棘刺",
                "role": "Warrior_Lord",
                "role_id": 12,
                "cost": 18,
                "available": True,
                "cooling": False,
                "rect": [100, 200, 60, 60],
                "is_usual_location": True,
                # 上游未来可能新增的字段：应被白名单剔除
                "some_future_field": "should_be_dropped",
            }
        ],
        "deployed": [
            {"name": "芬", "role": "Pioneer_Charger", "role_id": 1, "row": 3, "col": 5},
        ],
    }


def test_record_keeps_whitelisted_fields_and_drops_unknown():
    """白名单之外的上游字段必须剔除（避免内部细节泄漏进训练数据）。"""
    rec = battle_state_to_record(
        _sample_details(), t_ms=1234, job_id=95540, stage_code="0-1"
    )
    assert rec["protocol_version"] == STATE_PROTOCOL_VERSION
    assert rec["t_ms"] == 1234
    assert rec["job_id"] == 95540
    assert rec["stage_code"] == "0-1"
    assert rec["costs"] == 23
    assert rec["kills"] == 12
    assert rec["total_kills"] == 45
    assert rec["in_battle"] is True
    # 白名单剔除
    assert "some_future_field" not in rec["hand"][0]


def test_missing_fields_are_not_filled_with_zero():
    """缺失字段必须不出现——补 0 会让下游把"没识别到"当成"费用为 0"。"""
    rec = battle_state_to_record({"in_battle": True}, t_ms=0)
    assert "costs" not in rec
    assert "kills" not in rec
    assert "hand" not in rec
    assert "deployed" not in rec


def test_deployed_coordinates_map_x_to_col_y_to_row():
    """坐标序：上游 Point.x → col、Point.y → row。

    patch 0001 落盘时写的是 ``{"row": loc.y, "col": loc.x}``，故本模块只需透传；
    本用例固化"上游 x 是列、y 是行"这一约定，防止有人"顺手"把它们换过来。
    """
    details = {
        "deployed": [
            {"name": "A", "role": "R", "role_id": 1, "row": 3, "col": 5},
        ]
    }
    rec = battle_state_to_record(details, t_ms=0)
    d = rec["deployed"][0]
    assert d["row"] == 3, "row 必须来自上游 y"
    assert d["col"] == 5, "col 必须来自上游 x"


def test_hand_rect_is_preserved_as_list():
    """手牌 rect 保留为 4 元列表 [x, y, w, h]（下游据此算像素位置）。"""
    rec = battle_state_to_record(_sample_details(), t_ms=0)
    assert rec["hand"][0]["rect"] == [100, 200, 60, 60]


def test_summarize_empty_is_n_zero_and_does_not_fake_ranges():
    """空列表 → 只有 n=0，不得出现伪造的 range 字段。"""
    s = summarize_states([])
    assert s == {"n": 0}


def test_summarize_reports_ranges_and_coverage():
    """有数据时给出费用/击杀范围与时间跨度（供快速判断这局采到没有）。"""
    recs = [
        battle_state_to_record({"costs": 10, "kills": 0, "hand": []}, t_ms=0),
        battle_state_to_record(
            {"costs": 30, "kills": 7, "hand": [{"name": "a"}, {"name": "b"}]}, t_ms=5000
        ),
    ]
    s = summarize_states(recs)
    assert s["n"] == 2
    assert s["has_costs"] is True
    assert s["has_kills"] is True
    assert s["costs_range"] == [10, 30]
    assert s["kills_range"] == [0, 7]
    assert s["hand_size_max"] == 2
    assert s["t_ms_range"] == [0, 5000]


def test_summarize_marks_absent_scalars():
    """若这局从未采到费用（如官方 MaaCore 不发回调而误传空 dict），has_costs=False。"""
    s = summarize_states([battle_state_to_record({}, t_ms=0)])
    assert s["n"] == 1
    assert s["has_costs"] is False
    assert s["has_kills"] is False
    assert "costs_range" not in s
