"""阻断画面看门狗单测（2026-10-20 卡死自愈层）。

全部用 numpy 程序化合成假图（配色/几何仿真实特征，素材红线：真实截图
一个像素都不进 git）。覆盖三类阻断器命中、负样本不误报、消解点映射、
哨兵双确认/点屏预算/截图失败容错，以及 battle_result 三支推导。
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from bridge.blocker_watchdog import (
    BLOCKER_KINDS,
    BLOCKER_PRTS_LEVEL_LOW,
    BLOCKER_SETTLE_LOST,
    BLOCKER_SYNC_UPDATED,
    DISMISS_TAPS,
    BlockerSentry,
    detect_blocker,
    resolve_battle_result,
)

# ---------------------------------------------------------------------------
# 合成假图（1920x1080 基线）
# ---------------------------------------------------------------------------


def _canvas(color: tuple[int, int, int]) -> np.ndarray:
    """纯色底画布。"""
    a = np.empty((1080, 1920, 3), dtype=np.uint8)
    a[:] = color
    return a


def prts_img() -> np.ndarray:
    """仿「PRTS提示：干员等级过低」：暗底 + 暗红通栏横幅 + 黄字块 + 警示图标。"""
    a = _canvas((45, 42, 40))
    a[355:690, :] = (135, 18, 12)  # 暗红横幅带（实测 y355-690）
    a[400:600, 470:560] = (255, 170, 30)  # 橙黄圆环（警示图标）
    a[480:620, 560:610] = (240, 240, 240)  # 白三角（警示图标）
    a[545:635, 660:1220] = (255, 205, 0)  # 黄字黑底（"干员等级过低"）
    return a


def lost_img() -> np.ndarray:
    """仿判负结算屏：暖色暗底 + 左中大白字块 + 右侧提示白字块。"""
    a = _canvas((68, 52, 44))  # 暖暗模糊底（实测场景色）
    a[455:545, 190:660] = (235, 235, 235)  # "任务失败" 大白字
    a[560:595, 100:460] = (225, 225, 225)  # "MISSION FAILED" 次行
    a[400:560, 1150:1620] = (230, 230, 230)  # 右侧提示框白字
    return a


def sync_img() -> np.ndarray:
    """仿「数据已更新」：暗底 + 中央灰底白字框 + (960,980) 圆形 ✓ 按钮。"""
    a = _canvas((55, 52, 58))
    a[380:560, 640:1280] = (95, 95, 100)  # 灰底对话框
    a[415:425, 740:1180] = (245, 245, 245)  # 白字行 1
    a[445:455, 740:1180] = (245, 245, 245)  # 白字行 2
    yy, xx = np.mgrid[940:1020, 920:1000]
    d2 = (yy - 980) ** 2 + (xx - 960) ** 2
    patch = a[940:1020, 920:1000]
    patch[(d2 >= 26**2) & (d2 <= 44**2)] = (205, 205, 210)  # 圆环
    patch[d2 < 20**2] = (70, 70, 75)  # 按钮内底
    a[975:985, 952:970] = (245, 245, 245)  # 勾（少量白像素）
    return a


def win_img() -> np.ndarray:
    """仿胜利结算亮屏（负样本）：高亮度 + 暖色 + 大块白/金面板。"""
    a = _canvas((175, 160, 140))
    a[300:500, 200:900] = (240, 240, 240)  # 亮面板
    a[500:700, 1000:1700] = (255, 215, 90)  # 金色饰带
    return a


def home_img() -> np.ndarray:
    """仿游戏主页（负样本）：亮、冷色倾向、左上有小块红色新闻条。"""
    a = _canvas((150, 145, 155))
    a[850:950, 20:320] = (190, 40, 40)  # 小块红新闻条（非通栏）
    a[200:700, 560:960] = (45, 45, 60)  # 看板立绘暗块
    return a


def night_warm_img() -> np.ndarray:
    """仿暖色夜战帧（负样本）：暖且暗但没有任何白字块。"""
    a = _canvas((70, 55, 50))
    a[600:800, 300:900] = (95, 60, 40)  # 火光/岩浆色斑
    return a


# ---------------------------------------------------------------------------
# detect_blocker：命中与误报
# ---------------------------------------------------------------------------


def test_detect_prts_level_low() -> None:
    """红横幅假图命中 prts_level_low（PIL 与 numpy 输入都支持）。"""
    assert detect_blocker(prts_img()) == BLOCKER_PRTS_LEVEL_LOW
    assert detect_blocker(Image.fromarray(prts_img())) == BLOCKER_PRTS_LEVEL_LOW


def test_detect_settle_lost() -> None:
    """判负结算假图命中 settle_lost。"""
    assert detect_blocker(lost_img()) == BLOCKER_SETTLE_LOST


def test_detect_sync_updated() -> None:
    """同步弹窗假图命中 sync_updated。"""
    assert detect_blocker(sync_img()) == BLOCKER_SYNC_UPDATED


def test_detect_negatives() -> None:
    """胜利亮屏 / 主页 / 暖色夜战帧：全部不误报。"""
    assert detect_blocker(win_img()) is None
    assert detect_blocker(home_img()) is None
    assert detect_blocker(night_warm_img()) is None


def test_detect_partial_features_not_enough() -> None:
    """缺一两个特征时不命中（多条件组合的防误报语义）。"""
    # 只有红横幅，没有黄字块与图标 → 不算 prts 弹窗
    a = _canvas((45, 42, 40))
    a[355:690, :] = (135, 18, 12)
    assert detect_blocker(a) is None
    # 暖暗底 + 只有左白块 → 不算判负屏
    a2 = _canvas((68, 52, 44))
    a2[455:545, 190:660] = (235, 235, 235)
    assert detect_blocker(a2) is None
    # 只有圆环，没有对话框 → 不算同步弹窗
    a3 = _canvas((55, 52, 58))
    yy, xx = np.mgrid[940:1020, 920:1000]
    d2 = (yy - 980) ** 2 + (xx - 960) ** 2
    a3[940:1020, 920:1000][(d2 >= 26**2) & (d2 <= 44**2)] = (205, 205, 210)
    a3[940:1020, 920:1000][d2 < 20**2] = (70, 70, 75)
    assert detect_blocker(a3) is None


def test_detect_wrong_size_returns_none() -> None:
    """非 1920x1080 基线一律不检出（宁可漏检不可误检）。"""
    small = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert detect_blocker(small) is None
    gray = np.zeros((1080, 1920), dtype=np.uint8)
    assert detect_blocker(gray) is None


def test_dismiss_tap_mapping() -> None:
    """每个阻断器都有消解点；同步弹窗必须点 ✓ 按钮位，其余点右下空旷区。"""
    assert set(DISMISS_TAPS) == set(BLOCKER_KINDS)
    assert DISMISS_TAPS[BLOCKER_SYNC_UPDATED] == (960, 980)
    assert DISMISS_TAPS[BLOCKER_PRTS_LEVEL_LOW] == (1660, 1000)
    assert DISMISS_TAPS[BLOCKER_SETTLE_LOST] == (1660, 1000)


# ---------------------------------------------------------------------------
# BlockerSentry：双确认 / 预算 / 容错 / 记账
# ---------------------------------------------------------------------------


class _FakeDriver:
    """按脚本依次返回帧（脚本耗尽后重复最后一帧）；记录所有点屏。"""

    def __init__(self, frames: list) -> None:
        self._frames = list(frames)
        self._calls = 0
        self.taps: list[tuple[int, int]] = []

    def screencap(self):
        frame = self._frames[min(self._calls, len(self._frames) - 1)]
        self._calls += 1
        if isinstance(frame, Exception):
            raise frame
        return frame

    def tap(self, x: int, y: int) -> None:
        self.taps.append((x, y))


def _sentry(frames: list, **kw) -> tuple[BlockerSentry, _FakeDriver]:
    drv = _FakeDriver(frames)
    kw.setdefault("min_interval_s", 0.0)
    return BlockerSentry(drv, 1_000_000, **kw), drv


def test_sentry_double_confirm_before_tap() -> None:
    """同一阻断器须连续两轮命中才点屏；事件记账 {kind,t_ms,n}；n 递增。"""
    frames = [lost_img(), lost_img(), lost_img(), lost_img(), home_img()]
    sentry, drv = _sentry(frames)
    assert sentry.poll() is None  # 第 1 轮：只是候选
    assert drv.taps == []
    assert sentry.poll() == BLOCKER_SETTLE_LOST  # 第 2 轮：确认并消解
    assert drv.taps == [(1660, 1000)]
    assert sentry.poll() is None  # 消解后再见同屏：重新走双确认
    assert sentry.poll() == BLOCKER_SETTLE_LOST
    assert drv.taps == [(1660, 1000), (1660, 1000)]
    assert sentry.poll() is None  # 主页帧清掉候选
    events = sentry.events()
    assert [e["kind"] for e in events] == [BLOCKER_SETTLE_LOST, BLOCKER_SETTLE_LOST]
    assert [e["n"] for e in events] == [1, 2]
    assert all(isinstance(e["t_ms"], int) for e in events)


def test_sentry_tap_budget_per_kind() -> None:
    """同类阻断器点屏次数有预算，超预算后只 pending 不再点。"""
    frames = [lost_img()] * 8
    sentry, drv = _sentry(frames, max_taps_per_kind=1)
    for _ in range(8):
        sentry.poll()
    assert drv.taps == [(1660, 1000)]
    assert sentry.count() == 1
    assert sentry.hits_since(0, BLOCKER_SETTLE_LOST) == sentry.events()
    assert sentry.hits_since(1) == []
    assert sentry.hits_since(0, BLOCKER_SYNC_UPDATED) == []


def test_sentry_screencap_failure_tolerated() -> None:
    """截图异常只计数不抛出（不打断作战主流程）。"""
    sentry, _ = _sentry([RuntimeError("adb 挂了"), home_img()])
    assert sentry.poll() is None
    assert sentry.cap_failures == 1
    assert sentry.poll() is None
    assert sentry.events() == []


def test_sentry_throttle_skips_poll() -> None:
    """节流：距上次截图不足间隔时直接跳过（不增截图次数）。"""
    sentry, drv = _sentry([lost_img(), lost_img(), lost_img()], min_interval_s=999)
    assert sentry.poll() is None  # 第 1 次实际截图（上线即查一次）
    assert drv._calls == 1
    assert sentry.poll() is None  # 被节流，不再截图
    assert drv._calls == 1


def test_sentry_kinds_independent_pending() -> None:
    """不同阻断器交替出现时不构成"连续两轮"，不会误点。"""
    frames = [prts_img(), lost_img(), prts_img(), lost_img()]
    sentry, drv = _sentry(frames)
    for _ in frames:
        assert sentry.poll() is None
    assert drv.taps == []


# ---------------------------------------------------------------------------
# battle_result 三支
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lost", "chain_ok", "settled", "expect"),
    [
        (True, True, True, "lost"),  # 判负屏是最高证据，链状态不问
        (True, False, False, "lost"),
        (False, True, True, "win"),
        (False, True, False, "unknown"),  # 收尾没成功 → 不确定
        (False, False, True, "unknown"),  # 链异常（如超时）→ 不确定
        (False, False, False, "unknown"),
    ],
)
def test_resolve_battle_result(
    lost: bool, chain_ok: bool, settled: bool, expect: str
) -> None:
    """win/lost/unknown 三支推导（判负优先，win 需要链路+收尾双保险）。"""
    got = resolve_battle_result(
        lost_confirmed=lost, chain_completed=chain_ok, settled=settled
    )
    assert got == expect
