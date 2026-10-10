"""groups schema 归一化测试（2026-10-10：prts options -> MAA opers）。"""

from bridge.replay_controller import normalize_groups


def test_options_renamed_to_opers_and_id_dropped():
    raw = [
        {
            "name": "奶盾",
            "options": [
                {"name": "黍", "id": "char_2025_shu", "skill": 1},
                {"name": "吽", "id": "char_226_hmau", "skill": None},
            ],
        }
    ]
    assert normalize_groups(raw) == [
        {"name": "奶盾", "opers": [{"name": "黍", "skill": 1}, {"name": "吽"}]}
    ]


def test_already_maa_shape_kept():
    raw = [{"name": "组", "opers": [{"name": "A", "skill": 2, "skill_usage": 1}]}]
    want = [{"name": "组", "opers": [{"name": "A", "skill": 2, "skill_usage": 1}]}]
    assert normalize_groups(raw) == want


def test_empty_and_junk_dropped():
    assert normalize_groups([]) == []
    assert normalize_groups(None) == []
    assert normalize_groups([{"name": "空", "options": []}]) == []
    assert normalize_groups([{"name": "坏", "options": [{"id": "x"}, "junk"]}]) == []
