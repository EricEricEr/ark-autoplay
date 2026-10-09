"""maa_stages 判据单测：两个判据**必须分开**，且不互相污染。

背景（重要纠错，见 docs/adr/0006）：曾把 ``Fight`` 的 ``Episode{N}`` 规则
当成"可导航"判据，导致活动关（占数据集 71%）被整片误杀，
队列从本该有的数千条缩到 1,012 条。用户质疑"这些作业都是从 MAA 作业网站
下载的"后复查源码，确认真正判据是**地图数据（Arknights-Tile-Pos）**。
"""

from __future__ import annotations

from bridge.maa_stages import (
    fight_accepts,
    has_map_data,
    load_tile_keys,
)

# 模拟 task 表：只有 Episode0 / Episode5 / Episode8
TASKS = {"Episode0", "Episode5", "Episode8", "1-7", "SR-5"}
# 模拟 Tile-Pos key
TILES = {"main_00-01", "main_00-02", "0-1", "act33side_07", "camp_01"}


def test_main_stage_with_episode_and_map_data() -> None:
    """主线关卡：两个判据都通过。"""
    assert has_map_data("main_00-01", "0-1", TILES) is True
    assert fight_accepts("0-1", TASKS) is True


def test_activity_stage_has_map_data_but_fight_rejects() -> None:
    """**核心回归**：活动关有地图数据（Copilot 可处理），但 Fight 拒收。

    这正是当初误判的场景——若用 fight_accepts 过滤就会丢掉活动关。
    """
    # act33side_07 → code=IW-7（假设）；IW-7 不匹配正则、无 Episode7
    assert has_map_data("act33side_07", "IW-7", TILES) is True
    assert fight_accepts("IW-7", TASKS) is False


def test_fight_requires_matching_chapter() -> None:
    """Fight 判据：正则匹配但章节 task 不存在时仍拒收。"""
    # 3-1 匹配正则，但 TASKS 里没有 Episode3
    assert fight_accepts("3-1", TASKS) is False
    # 0-1 与 Episode0 都在
    assert fight_accepts("0-1", TASKS) is True


def test_fight_rejects_activity_shape() -> None:
    """GT-1 形态（前缀+单数字，无 `数字-数字`）不被 Fight 接受。"""
    assert fight_accepts("GT-1", TASKS) is False


def test_fight_accepts_direct_task_hit() -> None:
    """task 表直接命中时可用（如 SR-5 已注册）。"""
    assert fight_accepts("SR-5", TASKS) is True


def test_has_map_data_ignores_hash_modifier() -> None:
    """复刻关 stage_id 带 `#f#` 修饰，剥掉后仍应命中。"""
    assert has_map_data("main_00-01#f#", "0-1", TILES) is True


def test_empty_inputs_do_not_restrict() -> None:
    """未加载判据表时一律放行（便于无 MAA 环境下跑纯逻辑测试）。"""
    assert has_map_data("whatever", "X-1", set()) is True
    assert fight_accepts("whatever", set()) is True


def test_load_tile_keys_reads_real_dir(tmp_path) -> None:
    """load_tile_keys 能从 Tile-Pos 目录解析 key（用临时目录构造样本）。"""
    tp = tmp_path / "resource" / "Arknights-Tile-Pos"
    tp.mkdir(parents=True)
    for name in (
        "a001_01-activities-a001-level_a001_01.json",
        "main_00-01-obt-main-level_main_00-01.json",
        "act33side_07-activities-act33side-level_act33side_07.json",
    ):
        (tp / name).write_text("{}", encoding="utf-8")
    keys = load_tile_keys(tmp_path)
    assert "a001_01" in keys
    assert "main_00-01" in keys, "含连字符的 key 不能被截断"
    assert "act33side_07" in keys
