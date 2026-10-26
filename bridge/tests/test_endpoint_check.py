"""终点屏校验单测（2026-10-21，1-11 假成功 RCA 修复方案 A）。

纯逻辑覆盖：briefing_blue_ok 合成图三支；navigate_only 集成打桩
（假 driver，不碰真机）验证链完成后 True/False 两支与 Home 兜底告警。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from bridge.nav_maa import navigate_only
from bridge.navigator import briefing_blue_ok, briefing_button_ok

REGION = [1660, 955, 200, 90]  # 沿用 configs/nav_main.yaml brief_blue_check
TH = 0.25
CFG = {"region": REGION, "threshold": TH}


def _frame(*, blue: bool, fill: tuple[int, int, int] = (20, 120, 220)) -> np.ndarray:
    """1920x1080 画布：在蓝钮区域放/不放"行动蓝"色块。"""
    a = np.empty((1080, 1920, 3), dtype=np.uint8)
    a[:] = (60, 55, 50)
    if blue:
        x, y, w, h = REGION
        a[y : y + h, x : x + w] = fill
    return a


def test_briefing_blue_ok_hit_and_miss() -> None:
    """命中（蓝块）/不命中（暗块）/近似但错误色相，PIL 与 numpy 都认。"""
    assert briefing_blue_ok(_frame(blue=True), REGION, TH) is True
    assert briefing_blue_ok(Image.fromarray(_frame(blue=True)), REGION, TH) is True
    assert briefing_blue_ok(_frame(blue=False), REGION, TH) is False
    # 近似但错误色相（红块）：判定式 B-R>50 / B-G>30 天然否决
    assert briefing_blue_ok(_frame(blue=True, fill=(220, 60, 40)), REGION, TH) is False
    # 半蓝区域 0.5 >= 0.25 通过（阈值语义）
    a = _frame(blue=False)
    x, y, w, h = REGION
    a[y : y + h, x : x + w // 2] = (20, 120, 220)
    assert briefing_blue_ok(a, REGION, TH) is True
    # 异常输入 fail-closed
    assert briefing_blue_ok(np.zeros((10, 10), dtype=np.uint8), REGION, TH) is False
    assert briefing_blue_ok(np.zeros((64, 64, 3), dtype=np.uint8), REGION, TH) is False


class _FakeDriver:
    """按预排事件走完导航链；screencap 返回固定终点帧。"""

    def __init__(self, events: list, frame: np.ndarray) -> None:
        self._events = list(events)
        self._frame = frame
        self.taps: list[tuple[int, int]] = []

    def append_task(self, task_type: str, params: dict | None = None) -> int:
        return 1

    def start(self) -> bool:
        return True

    def running(self) -> bool:
        return False  # wait_idle 用：立即空闲

    def mark(self) -> int:
        return 0

    def events_since(self, mark: int) -> tuple[list[Any], int]:
        if mark == 0:
            return self._events, 1
        return [], 1

    def screencap(self) -> np.ndarray:
        return self._frame

    def tap(self, x: int, y: int) -> None:
        self.taps.append((x, y))


def _completed(taskid: int = 1) -> SimpleNamespace:
    return SimpleNamespace(msg="TaskChainCompleted", details={"taskid": taskid})


def test_navigate_only_true_when_briefing_shown(monkeypatch) -> None:
    """链完成 + 终点帧带蓝钮 → True（正常导航路径）。"""
    monkeypatch.delenv("ARK_NAV_SKIP", raising=False)
    drv = _FakeDriver([_completed()], _frame(blue=True))
    assert navigate_only(drv, "1-11", endpoint_cfg=CFG) is True


def test_navigate_only_false_when_home_fallback(monkeypatch, capsys) -> None:
    """链完成但终点帧是主页（无蓝钮）→ False；
    事件流里带 <stage>@Home@QuickSwitch 时打印兜底告警（RCA 同签名）。"""
    monkeypatch.delenv("ARK_NAV_SKIP", raising=False)
    home_evt = SimpleNamespace(
        msg="SubTaskStart",
        details={"cur_task": "1-11@Home@QuickSwitch@ToHome@Open"},
    )
    drv = _FakeDriver([home_evt, _completed()], _frame(blue=False))
    assert navigate_only(drv, "1-11", endpoint_cfg=CFG) is False
    out = capsys.readouterr().out
    assert "Home 兜底" in out
    assert "endpoint_check_failed" in out


def test_navigate_only_no_check_when_cfg_absent(monkeypatch) -> None:
    """endpoint_cfg 不给 = 旧行为：链完成即 True，不校验终点屏。"""
    monkeypatch.delenv("ARK_NAV_SKIP", raising=False)
    drv = _FakeDriver([_completed()], _frame(blue=False))
    assert navigate_only(drv, "1-11", endpoint_cfg=None) is True


def test_navigate_only_screencap_failure_is_false(monkeypatch) -> None:
    """终点校验时截图挂掉 → fail-closed 返回 False（宁判失败不放假成功）。"""

    class _BrokenDriver(_FakeDriver):
        def screencap(self) -> np.ndarray:
            raise RuntimeError("adb 挂了")

    monkeypatch.delenv("ARK_NAV_SKIP", raising=False)
    drv = _BrokenDriver([_completed()], _frame(blue=True))
    assert navigate_only(drv, "1-11", endpoint_cfg=CFG) is False


def test_dark_red_start_button_is_valid_for_hard_mode_only() -> None:
    frame = _frame(blue=True, fill=(95, 26, 17))
    assert briefing_button_ok(frame, REGION, TH, hard_mode=True)
    assert briefing_button_ok(Image.fromarray(frame), REGION, TH, hard_mode=True)
    assert not briefing_button_ok(frame, REGION, TH)
    assert not briefing_button_ok(_frame(blue=True), REGION, TH, hard_mode=True)


def test_hard_mode_rejects_invalid_or_non_briefing_frames() -> None:
    assert not briefing_button_ok(_frame(blue=False), REGION, TH, hard_mode=True)
    assert not briefing_button_ok(np.zeros((10, 10)), REGION, TH, hard_mode=True)
    assert not briefing_button_ok(np.zeros((10, 10, 3)), REGION, TH, hard_mode=True)


def test_hard_navigation_accepts_dark_red_button(monkeypatch) -> None:
    monkeypatch.delenv("ARK_NAV_SKIP", raising=False)
    driver = _FakeDriver([_completed()], _frame(blue=True, fill=(95, 26, 17)))
    assert navigate_only(driver, "11-2-Hard", endpoint_cfg=CFG)
