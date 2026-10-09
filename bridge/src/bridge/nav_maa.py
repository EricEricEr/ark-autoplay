"""MAA 原生任务导航：把游戏带到目标关卡的"开始行动"准备界面。

设计（见 ``docs/adr/0002-maa-native-navigation.md``）
----------------------------------------------------
**不使用自建几何导航**（固定点位 + 钳位滑动 + 像素阈值）——它在真机联调中反复
失败（0-2 报"节点点击未打开 briefing"），且需逐关手工标定。改为**复用 MAA 自己
的导航**：

1. 下发 ``Fight(stage=<code>)``——MAA 自带的完整关卡导航（含终端→曲谱→章节→
   节点点击→确认"开始行动"），每步都有 template score；
2. 监听事件流，一旦出现准备界面标志（``StartButton1`` 等）立即 ``AsstStop``，
   避免真的开打消耗理智；
3. 返回是否到位。

为什么可行（实测，ADR-0002）
----------------------------
MAA 的 ``Fight`` 任务实测 24.7 秒把游戏从主页带到 0-2 的准备界面，关键步骤
score 达 0.999~1.000；自建导航器在同一关卡直接失败。

已知边界（务必知悉）
--------------------
- **抢停是粗粒度中断**：``AsstStop`` 后 ``Fight`` 任务链不会正常收尾（会挂到
  超时）。故**不能**用 ``wait_chain`` 的终态判定"是否到位"，本模块改为按
  **任务名出现**判定。
- 抢停与"MAA 自动点开始行动"之间存在竞态：MAA 在 ``StartButton1`` 之后可能立刻
  开战。本模块在检测到标志的同一次事件循环内立即 ``stop()``，实测可抢到；
  但仍非 100% 保证，极端情况下会真打一局（消耗理智）。
- 需用参数约束 MAA 的默认行为：``times=1`` / ``medicine=0`` / ``stone=0``，
  防止自动嗑药或碎石。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bridge.maa_driver import MaaDriver

READY_TASKS: tuple[str, ...] = (
    "StartButton1",
    "StartButton2",
    "BattleStartNormal",
    "BattleQuickFormation",
    "BattlePrepare",
)
"""出现这些**任务名**即视为已到准备界面（MAA 的 task 名，语义明确）。

优先用 ``StartButton1``：实测 0-2 上该步骤 score=1.000，且是"开始行动"按钮本身。
"""

FIGHT_PARAMS_SAFE: dict[str, object] = {
    "times": 1,
    "medicine": 0,  # 不自动使用理智药
    "stone": 0,  # 不自动碎石（付费资源，绝不允许自动消耗）
    "report_to_penguin": False,
    "DrGrandet": False,
}
"""``Fight`` 任务的安全参数：限制单次、禁用药与碎石、关闭上报。"""


class NavigationError(RuntimeError):
    """导航失败（未能在超时内到达准备界面）。"""


@dataclass
class NavResult:
    """导航结果。"""

    ok: bool
    elapsed_s: float
    ready_task: str | None
    """触发到位的任务名（None 表示超时未到位）。"""
    tasks_seen: list[str]
    """观察到的任务名序列（诊断用）。"""
    stopped: bool
    """是否成功抢停（False 表示可能已进入作战）。"""


class MaaNativeNavigator:
    """用 MAA 的 Fight 任务导航到关卡准备界面，并截停。"""

    def __init__(self, driver: MaaDriver, timeout_s: float = 120.0) -> None:
        """``driver`` 须已 connect。"""
        self._drv = driver
        self._timeout = timeout_s

    def goto_stage_briefing(self, stage_code: str) -> NavResult:
        """导航到 ``stage_code``（如 ``"0-2"``）的准备界面并截停。

        返回 :class:`NavResult`；未到位时 ``ok=False``（不抛异常，便于批量流程
        记摘要后继续下一局）。
        """
        params = {"stage": stage_code, **FIGHT_PARAMS_SAFE}
        tid = self._drv.append_task("Fight", params)
        if not tid:
            raise NavigationError(f"append_task(Fight) 返回 0，参数被拒: {params}")

        mark = self._drv.mark()
        t0 = time.monotonic()
        self._drv.start()

        seen: list[str] = []
        seen_set: set[str] = set()
        ready: str | None = None
        stopped = False
        deadline = t0 + self._timeout

        while time.monotonic() < deadline:
            evts, mark = self._drv.events_since(mark)
            for evt in evts:
                d = evt.details if isinstance(evt.details, dict) else {}
                det = d.get("details") if isinstance(d.get("details"), dict) else {}
                task = str(d.get("task") or det.get("task") or "")
                if task and task not in seen_set:
                    seen_set.add(task)
                    seen.append(task)
                if task in READY_TASKS:
                    ready = task
                    # 立即抢停：同一事件循环内，避免 MAA 继续点下"开始行动"
                    self._drv.stop()
                    stopped = True
                    break
            if ready:
                break
            time.sleep(0.1)
        else:
            # 超时：主动停掉，避免任务继续跑
            self._drv.stop()

        return NavResult(
            ok=ready is not None,
            elapsed_s=round(time.monotonic() - t0, 1),
            ready_task=ready,
            tasks_seen=seen,
            stopped=stopped,
        )

    def wait_battle_settle(self, timeout_s: float = 300.0) -> bool:
        """作战收尾：等待回到可操作界面。

        简化实现：用 MAA 的 ``StartUp`` 任务把游戏带回主界面（它自己会识别并
        处理结算/公告等弹窗）。返回是否成功。

        相比原先"轮询蓝钮像素点屏幕"的做法，这里交给 MAA 的界面识别，
        不依赖像素阈值。
        """
        tid = self._drv.append_task("StartUp", {"client_type": "Official"})
        if not tid:
            return False
        mark = self._drv.mark()
        self._drv.start()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            evts, mark = self._drv.events_since(mark)
            for evt in evts:
                if evt.msg in ("TaskChainCompleted", "AllTasksCompleted"):
                    return True
                if evt.msg == "TaskChainError":
                    return False
            time.sleep(0.2)
        return False
