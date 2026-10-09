"""Copilot 直达：用 MAA 的 ``Copilot`` 任务一次完成"导航 + 执行作业"。

为什么不再用 ``Fight`` 导航（2026-10-09 纠错，重要）
-----------------------------------------------------
此前用 ``Fight(stage=...)`` 做导航，理由是"它自带关卡导航"。**但实测发现
``Fight`` 会走进游戏内的代理作战（PRTS）**，而自抽号的代理记录是号商机械刷出的
非正常数值、对训练毫无价值甚至有害。

证据（真机日志，``Fight`` 逐章测试期间）：

- ``Fight@PRTS1`` 出现 **13 次** —— 确实进入了代理作战；
- ``Fight@PrtsErrorConfirm`` 4 次、``Fight@FightMissionFailed`` 73 次
  —— 号商代理记录连自己都跑不通，直接失败；
- ``tasks.json`` 中 ``UsePrts`` 的 ``next`` 含 ``StartButton1``，即"有代理就点代理
  直接开打、没有才走正常流程"；``FightTask`` 只把 ``PRTS1/2/3`` 次数限 0，
  **并未禁掉 ``UsePrts``**。

对比 ``Copilot`` 路径（``CopilotTask`` / ``MultiCopilotTaskPlugin``）：

- 起手是 ``BattleStartPre`` → ``BattleQuickFormation`` → ``BattleStartAll``，
  **全程不碰 PRTS**；
- 显式执行 ``ProcessTask(*this, {"NotUsePrts"})`` —— **主动关闭代理**；
- **自带导航**：``navigate_to_stage()`` 先试模板匹配、无模板则走图像 OCR 读关名，
  因此不需要"先导航到 briefing 再抢停"的两段式。

结论：``Copilot`` 一步到位（导航 + 按玩家作业执行），且天然避开代理，
**是数据工厂唯一正确的路径**；``Fight`` 的 ``UsePrts`` 分支不适合采数据。

关于数据来源的定位（用户 2026-10-09 澄清，勿再误解）
---------------------------------------------------
prts.plus 作业是**玩家先手动打通、再转成作业分享**的，有真实价值，部分质量很高。
本项目用 MAA 只是**忠实回放这些作业**；MAA 自身不产出策略数据。
（主线关卡 MAA 也不会用自有策略去打。）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bridge.maa_driver import MaaDriver

# Copilot 任务链里代表"已进入作战"的任务名（用于诊断，不用于截停）
BATTLE_TASKS: tuple[str, ...] = (
    "BattleStartPre",
    "BattleQuickFormation",
    "BattleStartAll",
    "BattleProcessTask",
)

# 一旦出现这些任务名，说明走了代理作战（应该永远不会出现；出现即报警）
PRTS_TASKS: tuple[str, ...] = (
    "UsePrts",
    "PRTS1",
    "PRTS2",
    "PRTS3",
    "PrtsErrorConfirm",
    "ClickCornerAfterPRTS",
)
"""代理相关任务名。**正常 Copilot 路径下不应出现任何一项**。

本模块把它作为**运行时护栏**：一旦观察到即判定该局数据被污染，
在 episode 中标记并可选择中止——自抽号的代理记录无训练价值（用户明确）。
"""


class NavigationError(RuntimeError):
    """作业链失败（未能在超时内正常完成）。"""


@dataclass
class CopilotResult:
    """一次 Copilot 作业的执行结果。"""

    ok: bool
    terminal: str | None
    """终态事件名（``TaskChainCompleted`` 等）。"""
    elapsed_s: float
    tasks_seen: list[str] = field(default_factory=list)
    """观察到的任务名序列（诊断用）。"""
    prts_detected: list[str] = field(default_factory=list)
    """命中的代理相关任务名（**非空即代表本局可能被代理污染**）。"""


class CopilotRunner:
    """下发 Copilot 作业并等待其完成（导航由 MAA 内部完成）。

    与旧 ``MaaNativeNavigator`` 的区别：不再有"导航到 briefing 再抢停"的两段式，
    也不需要 ``StartButton1`` 之类的到位判定——Copilot 自己会导航并执行作业。
    """

    def __init__(self, driver: MaaDriver, timeout_s: float = 600.0) -> None:
        """``driver`` 须已 connect。"""
        self._drv = driver
        self._timeout = timeout_s

    def run_job(self, job_path: str, formation: bool = True) -> CopilotResult:
        """下发并执行一个 Copilot 作业文件，阻塞至终态或超时。"""
        tid = self._drv.append_copilot(job_path, formation=formation)
        if not tid:
            raise NavigationError(f"append_copilot 返回 0（参数被拒）: {job_path}")

        mark = self._drv.mark()
        t0 = time.monotonic()
        self._drv.start()

        seen: list[str] = []
        seen_set: set[str] = set()
        prts: list[str] = []
        terminal: str | None = None
        deadline = t0 + self._timeout

        while time.monotonic() < deadline:
            evts, mark = self._drv.events_since(mark)
            for evt in evts:
                d = evt.details if isinstance(evt.details, dict) else {}
                det = d.get("details") if isinstance(d.get("details"), dict) else {}
                task = str(d.get("task") or det.get("task") or d.get("cur_task") or "")
                if task:
                    # cur_task 形如 "Fight@UsePrts"，取末段比对
                    tail = task.rsplit("@", 1)[-1]
                    if task not in seen_set:
                        seen_set.add(task)
                        seen.append(task)
                    if tail in PRTS_TASKS and tail not in prts:
                        prts.append(tail)
                if evt.msg in (
                    "TaskChainCompleted",
                    "TaskChainError",
                    "AllTasksCompleted",
                ):
                    terminal = evt.msg
                    break
            if terminal:
                break
            time.sleep(0.15)

        return CopilotResult(
            ok=terminal in ("TaskChainCompleted", "AllTasksCompleted"),
            terminal=terminal,
            elapsed_s=round(time.monotonic() - t0, 1),
            tasks_seen=seen,
            prts_detected=prts,
        )


def safe_copilot_params(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Copilot 的安全参数（禁自动嗑药/碎石；不上报第三方）。

    ``use_sanity_potion`` 保持 False：不自动使用理智药。
    碎石的开关不在此（Copilot 由 ``medicine``/``stone`` 无关参数控制），
    但数据工厂侧仍禁止任何付费资源消耗。
    """
    p: dict[str, Any] = {"use_sanity_potion": False}
    if extra:
        p.update(extra)
    return p


def navigate_only(driver: MaaDriver, stage_code: str, timeout_s: float = 180.0) -> bool:
    """把游戏从主页导航到 ``stage_code`` 的准备界面，**不开打、不碰代理**。

    依赖 **patch 0002**（``patches/0002-fight-nav-only.patch``）：给 ``Fight``
    任务增加 ``nav_only`` 参数，只保留启动与关卡导航子任务、禁用作战
    （``m_fight_task_ptr``），从而绕开 ``UsePrts``（游戏内代理作战）。

    ⚠️ 未打该补丁的 MaaCore 会忽略 ``nav_only`` 并**照常开打**，故使用前
    应确认运行时已应用 patch 0002（见 ``maacore/UPSTREAM.json``）。

    为什么不用 Copilot 自带的导航：``MultiCopilotTaskPlugin::navigate_to_stage``
    **只做地图内滑动找关卡**，没有"主页→章节地图"的章节寻路（``Episode{N}``），
    在主页会卡死（实测其 OCR 反复读到主界面文本）。

    为什么不用普通 ``Fight``：``FightBegin`` 的 next 里 ``UsePrts`` 排在
    ``StartButton1`` **之前**，只要该关有代理记录就必然先点代理；而自抽号的
    代理记录是号商机械刷出的异常数值，对训练无价值。

    返回是否在超时内导航完成（``TaskChainCompleted``）。
    """
    tid = driver.append_task("Fight", {"stage": stage_code, "nav_only": True})
    if not tid:
        raise NavigationError(
            f"append_task(Fight, nav_only=True) 返回 0: {stage_code}"
            "（关卡名不被 MAA 接受，或 MaaCore 未打 patch 0002）"
        )
    mark = driver.mark()
    if not driver.start():
        return False
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        evts, mark = driver.events_since(mark)
        for evt in evts:
            if evt.msg in ("TaskChainCompleted", "AllTasksCompleted"):
                return True
            if evt.msg == "TaskChainError":
                return False
        time.sleep(0.2)
    return False


def wait_settle(driver: MaaDriver, timeout_s: float = 300.0) -> bool:
    """作战收尾：把游戏带回可操作状态（用 MAA 的 ``StartUp``）。

    链收尾 ≠ 作战结束（Copilot 任务链在作业动作执行完即返回），故需再推一次
    结算。交给 MAA 的 ``StartUp``：它会自己识别并处理结算画面、公告等弹窗，
    不依赖像素阈值（原做法是轮询蓝钮像素点屏幕，已废弃）。

    返回是否在超时内完成。
    """
    tid = driver.append_task("StartUp", {"client_type": "Official"})
    if not tid:
        return False
    mark = driver.mark()
    if not driver.start():
        return False
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        evts, mark = driver.events_since(mark)
        for evt in evts:
            if evt.msg in ("TaskChainCompleted", "AllTasksCompleted"):
                return True
            if evt.msg == "TaskChainError":
                return False
        time.sleep(0.2)
    return False
