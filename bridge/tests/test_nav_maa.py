"""MaaNativeNavigator 单测：只测纯逻辑（不连真机、不装 MAA）。

导航本身依赖 MAA 与模拟器，无法在单测里跑；这里覆盖的是**判定逻辑与安全约束**：
- 准备界面标志的识别（按任务名，而非像素）；
- Fight 参数必须禁用付费资源消耗（碎石）与自动嗑药；
- 未到位时不抛异常，而是返回 ok=False（便于批量流程继续）。
"""

from __future__ import annotations

import pytest

from bridge.nav_maa import (
    FIGHT_PARAMS_SAFE,
    READY_TASKS,
    MaaNativeNavigator,
    NavigationError,
)


class _FakeDriver:
    """假驱动：按脚本回放事件，用于测导航判定逻辑。"""

    def __init__(self, task_sequence: list[str], append_ret: int = 1) -> None:
        self._seq = task_sequence
        self._append_ret = append_ret
        self._cursor = 0
        self.stopped = False
        self.appended: list[tuple[str, dict]] = []

    def append_task(self, task_type: str, params: dict | None = None) -> int:
        self.appended.append((task_type, params or {}))
        return self._append_ret

    def mark(self) -> int:
        return self._cursor

    def events_since(self, mark: int):
        # 每次把剩余任务当成新事件吐出来
        out = []
        while self._cursor < len(self._seq):
            name = self._seq[self._cursor]
            self._cursor += 1
            fake_evt = type(
                "E", (), {"msg": "SubTaskStart", "details": {"task": name}}
            )()
            out.append(fake_evt)
        return out, self._cursor

    def start(self) -> bool:
        return True

    def stop(self) -> bool:
        self.stopped = True
        return True


def test_ready_tasks_include_start_button() -> None:
    """到位判定必须包含 StartButton1（实测该步 score=1.000，即"开始行动"按钮）。

    用**任务名**判定而非像素阈值，是 ADR-0002 的核心决策。
    """
    assert "StartButton1" in READY_TASKS


def test_fight_params_forbid_paid_resources() -> None:
    """Fight 参数必须禁用药与碎石——付费资源绝不允许自动消耗。"""
    assert FIGHT_PARAMS_SAFE["stone"] == 0, "禁止自动碎石（付费资源）"
    assert FIGHT_PARAMS_SAFE["medicine"] == 0, "禁止自动使用理智药"
    assert FIGHT_PARAMS_SAFE["times"] == 1, "单次执行，避免连打"


def test_navigator_stops_immediately_on_ready_task() -> None:
    """一旦出现到位标志，必须立刻 stop（避免 MAA 继续点"开始行动"开战）。"""
    drv = _FakeDriver(
        ["StageBegin", "Fight", "ClickStageName", "StartButton1", "FightBegin"]
    )
    nav = MaaNativeNavigator(drv, timeout_s=5.0)  # type: ignore[arg-type]
    res = nav.goto_stage_briefing("0-2")
    assert res.ok is True
    assert res.ready_task == "StartButton1"
    assert res.stopped is True
    assert drv.stopped is True
    # 不应把后续任务也消费掉（说明及时中断）
    assert "FightBegin" not in res.tasks_seen


def test_navigator_reports_not_ok_on_timeout() -> None:
    """未到位时返回 ok=False（不抛异常），便于批量流程记摘要后继续。"""
    drv = _FakeDriver(["StageBegin", "Fight"])
    nav = MaaNativeNavigator(drv, timeout_s=0.3)  # type: ignore[arg-type]
    res = nav.goto_stage_briefing("0-2")
    assert res.ok is False
    assert res.ready_task is None
    assert drv.stopped is True, "超时也应主动停掉，避免任务继续跑"


def test_navigator_raises_when_append_rejected() -> None:
    """append_task 返回 0（参数被拒）时抛 NavigationError。"""
    drv = _FakeDriver([], append_ret=0)
    nav = MaaNativeNavigator(drv, timeout_s=1.0)  # type: ignore[arg-type]
    with pytest.raises(NavigationError):
        nav.goto_stage_briefing("0-2")


def test_navigator_passes_safe_params_and_stage() -> None:
    """下发的 Fight 任务必须带上安全参数与目标关卡。"""
    drv = _FakeDriver(["StartButton1"])
    nav = MaaNativeNavigator(drv, timeout_s=2.0)  # type: ignore[arg-type]
    nav.goto_stage_briefing("1-7")
    assert drv.appended, "应下发任务"
    task_type, params = drv.appended[0]
    assert task_type == "Fight"
    assert params["stage"] == "1-7"
    assert params["stone"] == 0
