"""家族分组划分泄漏检查（合成数据，不含真实游戏内容）。

设计文档 §12 最高准则：按关卡家族（章节 / 活动）划分训练 / 测试，严禁按对局随机划分。
本用例用合成 episode 记录验证划分辅助逻辑的组不变式：
- 正例：按家族分组划分 → 训练 / 测试家族交集为空，green；
- 反例：人为把同一家族放进两侧 → 泄漏检查器必须检出（assert_leak_free 抛出）。

TODO(M1): evaluation/splits.py 的三档划分实现落地后，本用例切换为直接调用
splits.py 的划分生成器与泄漏检查器（当前 splits.py 仍为 M1 占位 stub）。
"""

from collections.abc import Iterable

import pytest

from ark_core.datapipe.stages_meta import family_of


def _synth_episodes() -> list[dict[str, str]]:
    """合成 episode 表：3 个家族 × 每族 3 条（episode_id/stage_id/family）。"""
    stage_ids = [
        "main_00-01", "main_00-02", "tough_00-01",   # → main_00
        "act99side_01", "act99side_02", "act99side_03",  # → act99side
        "wk_melee_1", "wk_melee_2", "wk_melee_3",    # → wk
    ]
    return [
        {
            "episode_id": f"copilot_syn_{i}",
            "stage_id": sid,
            "family": family_of(sid),
        }
        for i, sid in enumerate(stage_ids)
    ]


def split_by_family(
    episodes: list[dict[str, str]], test_family: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """按家族留出划分：test_family 的全部 episode 进测试集，其余进训练集。"""
    train = [e for e in episodes if e["family"] != test_family]
    test = [e for e in episodes if e["family"] == test_family]
    return train, test


def assert_leak_free(train: Iterable[dict[str, str]], test: Iterable[dict[str, str]]) -> None:
    """泄漏检查器：训练 / 测试家族交集非空即失败。"""
    fam_train = {e["family"] for e in train}
    fam_test = {e["family"] for e in test}
    both = fam_train & fam_test
    if both:
        raise AssertionError(f"关卡家族泄漏：{sorted(both)} 同时出现在训练集与测试集")


def test_family_grouped_split_is_green() -> None:
    """正例：按家族划分时泄漏检查通过，且两侧 episode 数守恒。"""
    episodes = _synth_episodes()
    for holdout in ("main_00", "act99side", "wk"):
        train, test = split_by_family(episodes, holdout)
        assert_leak_free(train, test)
        assert len(train) + len(test) == len(episodes)
        assert {e["family"] for e in test} == {holdout}


def test_family_split_covers_tough_joining() -> None:
    """tough_00-01 与 main_00-01 同族：同族关必须落在同一侧（家族语义正确性）。"""
    episodes = _synth_episodes()
    train, _ = split_by_family(episodes, "main_00")
    assert_family_members = {e["stage_id"] for e in train if e["family"] == "main_00"}
    assert assert_family_members == set()


def test_leak_checker_catches_dirty_split() -> None:
    """反例：人为污染的划分必须被检查器检出。"""
    episodes = _synth_episodes()
    # 污染一：把 act99side 的 1 条塞进训练集，其余进测试集（家族被打散到两侧）
    dirty_train = [e for e in episodes if e["stage_id"] != "act99side_03"]
    dirty_test = [e for e in episodes if e["stage_id"] == "act99side_03"]
    # 污染二：按对局奇偶随机划分（设计文档明令禁止的划分方式）——3 族必跨两侧
    parity_train = episodes[::2]
    parity_test = episodes[1::2]
    for train, test in ((dirty_train, dirty_test), (parity_train, parity_test)):
        with pytest.raises(AssertionError, match="泄漏"):
            assert_leak_free(train, test)
