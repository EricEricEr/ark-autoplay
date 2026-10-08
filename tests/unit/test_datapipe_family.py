"""family_of 关卡家族函数测试（全部为合成 stage_id，不含真实数据）。

规则见 src/ark_core/datapipe/stages_meta.py 模块 docstring 与 docs/adr/0002：
main/tough/hard → main_<章节>；act 前缀 → act 家族；camp|wk|sub|pro|tr|a|rogue
前缀 → token；其余取第一个 _ 分段。验证基线：copilot 作业覆盖 102 个家族。
"""

from ark_core.datapipe.stages_meta import family_of


def test_main_chapters() -> None:
    """主线章节归族：普通关直接取章节号。"""
    assert family_of("main_00-01") == "main_00"
    assert family_of("main_12-7") == "main_12"


def test_tough_and_hard_join_chapter() -> None:
    """磨难 / 绝境并入对应主线章节（验证过的关键 case：tough_05 → main_05）。"""
    assert family_of("tough_05-11") == "main_05"
    assert family_of("hard_12-03") == "main_12"


def test_main_variant_suffix() -> None:
    """带 #f# 等变体后缀的主线关仍归到章节。"""
    assert family_of("main_00-01#f#") == "main_00"


def test_act_families() -> None:
    """活动关前缀归族（含数字字母混合尾）。"""
    assert family_of("act23side_06") == "act23side"
    assert family_of("act1break_01") == "act1break"
    assert family_of("act17d7_01") == "act17d7"


def test_token_families() -> None:
    """独立 token 家族；a001/a003 剿灭轮换统一并入 a（验证基线的关键口径）。"""
    assert family_of("camp_01") == "camp"
    assert family_of("wk_melee_1") == "wk"
    assert family_of("sub_02-01") == "sub"
    assert family_of("pro_v_01") == "pro"
    assert family_of("tr_10") == "tr"
    assert family_of("a001_01") == "a"
    assert family_of("a003_02") == "a"
    assert family_of("rogue_mist_01") == "rogue"


def test_fallback_first_segment() -> None:
    """未命中前缀规则的 stage_id：取第一个下划线分段。"""
    assert family_of("lt_tr_01") == "lt"
    assert family_of("bi_foo_01") == "bi"


def test_act_rule_precedes_token_a() -> None:
    """act 规则先于 a-token：act23side 不得被 a 前缀吞掉。"""
    assert family_of("act23side_06").startswith("act")
    assert family_of("act23side_06") != "a"
