"""阻断画面看门狗：无 OCR 像素特征检测 + 自动点屏消解（2026-10-20 卡死自愈层）。

背景（三次实测卡死）
--------------------
工厂在以下三类画面会静止等待（任务链/收尾都认不得这些屏），必须有人点屏：

1. ``prts_level_low``：进战斗前的「PRTS提示：干员等级过低」全屏红横幅
   （暗红横幅带 + 黄字黑底 + 左侧等级警示图标，底部"点击画面以继续"）。
   实测样本：2026-10-20 main_04-07 job-55778 局，Copilot 链在弹窗上挂死
   直至链超时。消解 = 点屏任意处。
2. ``settle_lost``：判负结算屏（"任务失败 MISSION FAILED"，暖色暗底 +
   左中大白字块 + 右侧提示框）。实测样本：2026-10-20 main_04-07 job-22241
   局 settle 帧。消解 = 点屏任意处推进。
3. ``sync_updated``：「数据已更新，即将进行同步」弹窗（中央灰底白字 +
   底部圆形 ✓ 按钮，圆心约 (960,980)，日更后首启常见）。
   **规格驱动**：暂无落盘样本，特征按形态描述设计并经全库零误报验证；
   消解 = 点 ✓ 按钮（点任意处可能不生效，故必须点按钮位）。

设计要点
--------
- **纯像素特征**（numpy）：横幅色带占比 / 固定区域白字密度 / 圆形按钮
  签名；不用 OCR、不引入模板图片素材（素材红线，阈值均为模块常量并注明来源）。
- **二次命中才点屏**（双确认）：连续两轮（间隔 ≥1s）检出同一阻断器才执行
  消解动作，防止转场瞬间的单一帧误判把屏幕点到别处。
- **每类阻断器有消解点预算**：同一 episode 内同类最多点
  ``MAX_TAPS_PER_KIND`` 次，超出后只记不再点（防止检测误报时的无限点屏）。
- **截图/点屏失败不打断主流程**：poll 内吞掉异常（与 state_logger.snap 同戒律）。

接入方式（replay_controller / nav_maa）：所有"等屏幕变化"的循环里每次迭代
调用 ``sentry.poll()``（内部节流），事件经 ``sentry.events()`` 汇总进
episode.json 的 ``blockers_dismissed``；结算段检出 ``settle_lost`` 供
``battle_result`` 判定（见 :func:`resolve_battle_result`）。
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np


class _Screencapper(Protocol):
    """哨兵所需的最小驱动接口（便于单测用假 driver）。"""

    def screencap(self) -> Any: ...
    def tap(self, x: int, y: int) -> None: ...


# ---------------------------------------------------------------------------
# 阻断器 id 与消解点
# ---------------------------------------------------------------------------

BLOCKER_PRTS_LEVEL_LOW = "prts_level_low"
BLOCKER_SETTLE_LOST = "settle_lost"
BLOCKER_SYNC_UPDATED = "sync_updated"

BLOCKER_KINDS: tuple[str, str, str] = (
    BLOCKER_PRTS_LEVEL_LOW,
    BLOCKER_SETTLE_LOST,
    BLOCKER_SYNC_UPDATED,
)
"""看门狗首批支持的三类阻断画面 id。"""

DISMISS_TAPS: dict[str, tuple[int, int]] = {
    # 红横幅弹窗：任意点可消解；点右下空旷区，远离中部可能叠出的二级弹窗。
    BLOCKER_PRTS_LEVEL_LOW: (1660, 1000),
    # 判负结算屏：任意点推进；同上取右下空旷区（右上为倍速/暂停区，避开）。
    BLOCKER_SETTLE_LOST: (1660, 1000),
    # 同步弹窗：只有 ✓ 圆形按钮可消解；按钮圆心 (960,980) 来自 2026-10-20 实测形态描述。
    BLOCKER_SYNC_UPDATED: (960, 980),
}
"""阻断器 id → 消解点触坐标（1920x1080 基线）。"""

MAX_TAPS_PER_KIND = 5
"""同一 episode 内同类阻断器最多点屏次数（兜底防误报死循环）。"""

BASELINE: tuple[int, int] = (1920, 1080)
"""特征标定基线分辨率（MuMu 1920x1080/DPI 320）；非基线帧一律不检出。"""

# ---------------------------------------------------------------------------
# 像素特征阈值（来源：2026-10-20 对 ark-data 真实截图的实测标定 + 全库零误报验证）
#   prts 样本 = episodes/main_04-07/job-55778-20261020-072643/shots/064_settle.png
#   lost 样本 = episodes/main_04-07/job-22241-20261020-072148/shots/073_settle.png
#   同步弹窗 = 规格驱动（无落盘样本，见模块 docstring）
# ---------------------------------------------------------------------------

# prts_level_low：暗红横幅带（实测 y355-690）暗红像素占比
_PRTS_BAND_Y = (350, 660)
_PRTS_BAND_RED_MIN = 0.45
"""横幅带暗红占比阈值（实测命中 0.687，判负屏 0.069，噪声帧 <0.1）。"""

# prts_level_low：黄字黑底块（"干员等级过低"黄字区）
_PRTS_YELLOW_BOX = (530, 650, 630, 1320)  # y1,y2,x1,x2
_PRTS_YELLOW_MIN = 0.12
"""黄字占比阈值（实测 0.279）。"""

# prts_level_low：左侧等级警示图标区（橙黄圆环 + 白三角）
_PRTS_ICON_ZONE = (370, 650, 460, 670)  # y1,y2,x1,x2
_PRTS_ICON_MIN = 0.12
"""图标区亮色占比阈值（实测约 0.26）。"""

# settle_lost：整体昏暗 + 暖色调（暖色暗底是判负屏与昼夜战场/亮结算屏的分界）
_LOST_SCENE_Y = (200, 900)
_LOST_SCENE_LUM_MAX = 110.0
"""场景亮度上限（实测判负屏 66，主页 115+，编队/助战屏 145+）。"""
_LOST_SCENE_WARM_MIN = 8.0
"""暖色下限 R-B 均值差（实测判负屏 20.5，主页约 -12，夜战帧可负可小正）。"""

# settle_lost：左中大白字块（"任务失败 MISSION FAILED"）
_LOST_LEFT_ZONE = (430, 640, 160, 700)  # y1,y2,x1,x2
_LOST_LEFT_WHITE_MIN = 0.06
"""左区白色占比阈值（实测 0.174；同屏暖色战场帧 ≤0.03）。"""

# settle_lost：右侧提示框白字
_LOST_RIGHT_ZONE = (360, 580, 1100, 1650)  # y1,y2,x1,x2
_LOST_RIGHT_WHITE_MIN = 0.08
"""右区白色占比阈值（实测 0.237；红横幅弹窗同区仅 0.025）。"""

# settle_lost 的排除条件：不能同时命中红横幅特征
_LOST_BAND_RED_MAX = 0.40

# sync_updated：底部圆形 ✓ 按钮签名（规格驱动）
_SYNC_CENTER = (980, 960)  # (cy, cx)
_SYNC_RING_R = (26, 44)
_SYNC_RING_GRAYBRIGHT_MIN = 0.50
"""按钮圆环上亮灰像素占比下限。"""
_SYNC_INNER_LUM_MAX = 130.0
"""按钮内部亮度上限（深色底 + 少量白勾）。"""

# sync_updated：中央灰底白字对话框
_SYNC_DLG_ZONE = (390, 500, 640, 1280)  # y1,y2,x1,x2
_SYNC_DLG_LUM_RANGE = (40.0, 150.0)
"""对话框区域亮度范围（灰底）。"""
_SYNC_DLG_TEXT_ZONE = (400, 480, 700, 1220)  # y1,y2,x1,x2
_SYNC_DLG_WHITE_MIN = 0.02
"""对话框文字白色占比下限。"""


def _zone(arr: np.ndarray, y1: int, y2: int, x1: int, x2: int) -> np.ndarray:
    return arr[y1:y2, x1:x2]


_SYNC_MASK_CACHE: dict[str, np.ndarray] = {}
"""✓ 按钮环/内圆布尔掩码惰性缓存（全分辨率 mgrid 每次现算太浪费）。"""


def _sync_masks() -> tuple[np.ndarray, np.ndarray]:
    if "ring" not in _SYNC_MASK_CACHE:
        cy, cx = _SYNC_CENTER
        r1, r2 = _SYNC_RING_R
        ys, xs = np.mgrid[0 : BASELINE[1], 0 : BASELINE[0]]
        dist2 = (ys - cy) ** 2 + (xs - cx) ** 2
        _SYNC_MASK_CACHE["ring"] = (dist2 >= r1 * r1) & (dist2 <= r2 * r2)
        _SYNC_MASK_CACHE["inner"] = dist2 < 20 * 20
    return _SYNC_MASK_CACHE["ring"], _SYNC_MASK_CACHE["inner"]


def detect_blocker(img: Any) -> str | None:
    """检测一帧截图命中的阻断器 id，无命中返回 None。

    输入为 PIL Image 或 numpy 数组（RGB）。非 1920x1080 基线的帧一律返回
    None（阈值按基线标定，宁可漏检不可误检——误点屏比漏点屏代价高）。
    """
    a = np.asarray(img, dtype=np.int32)
    if a.ndim != 3 or a.shape[2] < 3:
        return None
    if (a.shape[1], a.shape[0]) != BASELINE:
        return None
    rr, gg, bb = a[..., 0], a[..., 1], a[..., 2]
    lum = (rr + gg + bb) / 3.0
    white = (rr > 200) & (gg > 200) & (bb > 200)
    yellow = (rr > 200) & (gg > 150) & (bb < 70)
    darkred = (rr > 70) & (rr < 200) & (rr - gg > 25) & (rr - bb > 25)

    band_red = float(_zone(darkred, *_PRTS_BAND_Y, 0, BASELINE[0]).mean())

    # --- prts_level_low：红横幅 + 黄字黑底 + 警示图标，三条件缺一不可 ---
    if band_red >= _PRTS_BAND_RED_MIN:
        yellow_box = float(_zone(yellow, *_PRTS_YELLOW_BOX).mean())
        icon_zone = _zone(a, *_PRTS_ICON_ZONE)
        ir, ig, ib = icon_zone[..., 0], icon_zone[..., 1], icon_zone[..., 2]
        icon_bright = (
            ((ir > 200) & (ig > 200) & (ib > 200))  # 白三角
            | ((ir > 180) & (ig > 120) & (ig < 200) & (ib < 80))  # 橙黄圆环
            | ((ir > 200) & (ig > 150) & (ib < 70))  # 黄数字
        )
        icon_frac = float(icon_bright.mean())
        if yellow_box >= _PRTS_YELLOW_MIN and icon_frac >= _PRTS_ICON_MIN:
            return BLOCKER_PRTS_LEVEL_LOW
        # 大面积暗红带但缺黄字/图标 → 不明变体，按未知屏处理（不误判其他类）

    # --- settle_lost：暖暗底 + 左右白字块；先排除红横幅屏 ---
    if band_red < _LOST_BAND_RED_MAX:
        scene = _zone(a, *_LOST_SCENE_Y, 0, BASELINE[0])
        scene_lum = float(
            (scene[..., 0] + scene[..., 1] + scene[..., 2]).mean() / 3.0
        )
        warm = float(scene[..., 0].mean() - scene[..., 2].mean())
        if scene_lum < _LOST_SCENE_LUM_MAX and warm > _LOST_SCENE_WARM_MIN:
            left_white = float(_zone(white, *_LOST_LEFT_ZONE).mean())
            right_white = float(_zone(white, *_LOST_RIGHT_ZONE).mean())
            left_ok = left_white >= _LOST_LEFT_WHITE_MIN
            right_ok = right_white >= _LOST_RIGHT_WHITE_MIN
            if left_ok and right_ok:
                return BLOCKER_SETTLE_LOST

    # --- sync_updated：✓ 圆形按钮签名 + 中央灰底白字框（规格驱动）---
    ring_mask, inner_mask = _sync_masks()
    spread = np.maximum.reduce([rr, gg, bb]) - np.minimum.reduce([rr, gg, bb])
    graybright = (lum > 150) & (spread < 40)
    ring_hit = float(graybright[ring_mask].mean())
    inner_lum = float(lum[inner_mask].mean())
    if ring_hit >= _SYNC_RING_GRAYBRIGHT_MIN and inner_lum < _SYNC_INNER_LUM_MAX:
        dlg = _zone(lum, *_SYNC_DLG_ZONE)
        dlg_lum = float(dlg.mean())
        dlg_white = float(_zone(white, *_SYNC_DLG_TEXT_ZONE).mean())
        if (
            _SYNC_DLG_LUM_RANGE[0] <= dlg_lum <= _SYNC_DLG_LUM_RANGE[1]
            and dlg_white >= _SYNC_DLG_WHITE_MIN
        ):
            return BLOCKER_SYNC_UPDATED
    return None


class BlockerSentry:
    """阻断器哨兵：节流轮询截图、双确认后点屏消解、记账阻断事件。

    用法::

        sentry = BlockerSentry(driver, t0_wall_ms)
        ... 每个等待循环迭代里: sentry.poll()
        episode["blockers_dismissed"] = sentry.events()

    ``t0_wall_ms`` 为 episode 起点墙钟毫秒，事件 ``t_ms`` 与之对齐
    （与 action_events.jsonl 同基准）。
    """

    def __init__(
        self,
        driver: _Screencapper,
        t0_wall_ms: int,
        *,
        min_interval_s: float = 1.0,
        max_taps_per_kind: int = MAX_TAPS_PER_KIND,
    ) -> None:
        """``driver`` 仅需 screencap/tap 两方法；轮询间隔默认 1s。"""
        self._drv = driver
        self._t0 = t0_wall_ms
        self._min_interval = min_interval_s
        self._max_taps = max_taps_per_kind
        self._events: list[dict[str, Any]] = []
        self._pending: str | None = None
        # -inf：保证实例化后的第一次 poll 总能真实截图（上线即查一次）
        self._last_poll = float("-inf")
        self._tap_count: dict[str, int] = {}
        self.cap_failures = 0
        """截图异常次数（诊断用，只计数不抛出）。"""

    def poll(self) -> str | None:
        """过一遍看门狗；返回本轮执行了消解动作的阻断器 id（无则 None）。

        节流：距上次实际截图不足 ``min_interval_s`` 直接返回 None。
        双确认：同一阻断器须**连续两轮**检出才点屏（防转场单帧误判）。
        """
        now = time.monotonic()
        if now - self._last_poll < self._min_interval:
            return None
        self._last_poll = now
        try:
            img = self._drv.screencap()
        except Exception:  # noqa: BLE001  # 截图失败不打断主流程，只计数
            self.cap_failures += 1
            return None
        kind = detect_blocker(img)
        if kind is None:
            self._pending = None
            return None
        if self._pending != kind:
            self._pending = kind
            return None
        # 连续两轮命中：执行消解（预算内）
        used = self._tap_count.get(kind, 0)
        if used >= self._max_taps:
            return None
        try:
            tx, ty = DISMISS_TAPS[kind]
            self._drv.tap(tx, ty)
        except Exception:  # noqa: BLE001  # 点屏失败同样只跳过
            return None
        self._tap_count[kind] = used + 1
        self._events.append(
            {
                "kind": kind,
                "t_ms": int(time.time() * 1000) - self._t0,
                "n": used + 1,
            }
        )
        self._pending = None
        return kind

    def events(self) -> list[dict[str, Any]]:
        """阻断事件列表（``[{kind, t_ms, n}]``，按发生序），返回副本。"""
        return list(self._events)

    def count(self) -> int:
        """已记录的阻断事件总数（配合 :meth:`hits_since` 做结算段切片）。"""
        return len(self._events)

    def hits_since(self, mark: int, kind: str | None = None) -> list[dict[str, Any]]:
        """返回第 ``mark`` 条之后的事件（可按 kind 过滤）。"""
        tail = self._events[mark:]
        if kind is None:
            return list(tail)
        return [e for e in tail if e["kind"] == kind]


def resolve_battle_result(
    *, lost_confirmed: bool, chain_completed: bool, settled: bool
) -> str:
    """推导 episode 的 ``battle_result`` 字段（"win"/"lost"/"unknown"）。

    口径（2026-10-20 拍板）：

    - 作战+结算窗口内检出判负屏（``settle_lost`` 消解成功）→ ``"lost"``
      （窗口从 Copilot 下发起算，避开上局残屏错记，见 replay_controller）；
    - 否则任务链正常完成 **且** 收尾成功回到可操作态 → ``"win"``
      （best-effort：当前无胜利屏正向校验，判负屏检测是主要证据）；
    - 其余（链超时/收尾失败/截图通道故障等）→ ``"unknown"``。

    数据口径提醒（方案 §10）：训练只留通关局 → 下游按
    ``battle_result == "win"`` 过滤即可，"lost"/"unknown" 一律不进训练集。
    """
    if lost_confirmed:
        return "lost"
    if chain_completed and settled:
        return "win"
    return "unknown"
