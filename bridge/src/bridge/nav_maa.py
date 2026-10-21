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
    from bridge.blocker_watchdog import BlockerSentry
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

    def run_job(
        self, job_path: str, formation: bool = True, support_unit_usage: int = 0
    ) -> CopilotResult:
        """下发并执行一个 Copilot 作业文件，阻塞至终态或超时。"""
        tid = self._drv.append_copilot(
            job_path, formation=formation, support_unit_usage=support_unit_usage
        )
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


# 剿灭列表条目在 1920x1080 下的点击 y 坐标（2026-10-13 截图 OCR 实测；
# 名字来自游戏内「剿灭委托 / 剿灭战模拟」列表）。x 固定 1466（条目卡片中部）。
_CAMP_ENTRY_Y: dict[str, int] = {
    "camp_01": 657,    # 乌萨斯 - 切尔诺伯格
    "camp_02": 357,    # 炎国 - 龙门外环
    "camp_03": 507,    # 炎国 - 龙门市区
    "camp_r_03": 1023, # 乌萨斯 - 废弃矿区（剿灭战模拟）
    "camp_r_15": 873,  # 炎国 - 龙门商业街（剿灭战模拟）
    "camp_r_33": 134,  # 拉特兰 - 默祷圣祠（剿灭委托）
}


def _tap_camp_entry(driver: MaaDriver, stage_code: str) -> None:
    """把剿灭列表滚回顶部后点选目标关卡。

    两层保险：先把列表连续下拉回顶部（游戏会记住上次的滚动位置），
    再按 _CAMP_ENTRY_Y 的固定 y 点击；y 坐标只在列表处于顶部时才确定。
    """
    try:
        import subprocess as _sp
        _adb = r"E:\Program Files\Netease\MuMu\nx_main\adb.exe"
        for _ in range(3):
            _sp.run([_adb, "-s", "127.0.0.1:16416", "shell", "input", "swipe",
                     "1466", "380", "1466", "1000", "260"], capture_output=True, timeout=25)
            time.sleep(0.6)
        time.sleep(1.0)
        print("[nav] 剿灭列表已滚回顶部", flush=True)
    except Exception as _ex:
        print(f"[nav] 列表滚顶失败：{_ex!r}", flush=True)
    y = _CAMP_ENTRY_Y.get(str(stage_code))
    if y:
        driver.tap(1466, y)
        time.sleep(2.5)
        print(f"[nav] 剿灭列表点选 {stage_code} @(1466,{y})", flush=True)


def navigate_only(
    driver: MaaDriver,
    stage_code: str,
    timeout_s: float = 180.0,
    watchdog: BlockerSentry | None = None,
) -> bool:
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

    ``watchdog``（2026-10-20 卡死自愈）：等待循环里每次迭代过一遍阻断器
    哨兵（同步弹窗等可在导航途中弹出）；不传则行为与旧版完全一致。

    返回是否在超时内导航完成（``TaskChainCompleted``）。
    """
    import os as _os
    if _os.environ.get("ARK_NAV_SKIP") == "1":
        # 人工把游戏停在目标关卡的准备界面时用（2026-10-14 加入）：
        # 直接认为导航已完成，避免 MAA 把用户摆好的界面挪走。
        print(f"[nav] ARK_NAV_SKIP=1，跳过导航，假定已停在 {stage_code} 准备界面", flush=True)
        return True
    camp_fallback = False
    # 2026-10-14 修根因(2)：剿灭关卡一律直接走 Annihilation 兜底。
    # 实测 append_task(Fight, {"stage": "camp_XX"}) 偶尔返回非 0 并"完成"，
    # 于是既没进兜底分支、也从未点选列表 → 游戏留在上一场的图（默祷圣祠），
    # 而调用方仍收到"导航成功"（静默假成功）。强制走兜底可保证一定点选。
    tid = 0 if str(stage_code).startswith("camp_") else driver.append_task("Fight", {"stage": stage_code, "nav_only": True})
    if not tid and str(stage_code).startswith("camp_"):
        # 剿灭兜底（2026-10-13 实测）：StageNavigationTask::set_stage_name 只认
        # 「同名 task」或主线正则（字母0-3位 + 数字-数字），camp_01/02/03 两者都不满足
        # → append 返回 0、整局被判「导航失败」。但剿灭的内部标识 Annihilation 是被
        # 接受的（MaaCore 源码注释：内部标识与显示代号不一致的关卡，如剿灭 Annihilation），
        # nav_only 下 MAA 会把游戏导航到剿灭作战地图；之后由 Copilot 自带的
        # navigate_to_stage 在地图内滑到具体剿灭关卡。
        tid = driver.append_task("Fight", {"stage": "Annihilation", "nav_only": True})
        camp_fallback = bool(tid)
    if not tid:
        raise NavigationError(
            f"append_task(Fight, nav_only=True) 返回 0: {stage_code}"
            "（关卡名不被 MAA 接受，或 MaaCore 未打 patch 0002）"
        )
    if not driver.start():
        # 上一个任务链尚未完全空闲时 start() 会直接返回 False。
        # 实测：连续「StartUp → Fight」时若不等 AllTasksCompleted，
        # Fight 会被 append 成功但 start 立即失败（日志 `Start | block` +
        # `leave, 0 ms`），表现为导航"秒失败"。
        # 这里再等一次空闲后重试一次，仍失败才认输。
        if not wait_idle(driver, timeout_s=30.0):
            return False
        if not driver.start():
            return False

    mark = driver.mark()
    t_start = time.monotonic()
    deadline = t_start + timeout_s
    while time.monotonic() < deadline:
        if watchdog is not None:
            watchdog.poll()
        evts, mark = driver.events_since(mark)
        for evt in evts:
            if evt.msg in ("TaskChainCompleted", "AllTasksCompleted"):
                # 2026-10-14：剿灭兜底即使"完成"也必须点选，否则会静默留在旧图。
                if camp_fallback and stage_code in _CAMP_ENTRY_Y:
                    _tap_camp_entry(driver, stage_code)
                wait_idle(driver, timeout_s=20.0)
                return True
            if evt.msg in ("TaskChainError", "TaskChainStopped"):
                if camp_fallback:
                    # 剿灭兜底：MAA 的 Annihilation 任务链必然以
                    # Annihilation@UnableToAgent2 → StageNavigationTask error 收尾
                    # （账号没有 PRTS 代理指挥卡），但报错前它已经点开了剿灭地图。
                    # 故把「点到地图」视为导航成功，剩余滑找交给 Copilot。
                    grace = 18.0 - (time.monotonic() - t_start)
                    if grace > 0:
                        time.sleep(grace)
                    # 关键一步（2026-10-13 实测踩坑）：剿灭列表默认停在第一个条目
                    # （默祷圣祠）。若不点选，Copilot 会拿「这张图」的点位去打「另一张图」
                    # → 干员放下却零击杀、约 100 秒被冲爆，且不报错。
                    _tap_camp_entry(driver, stage_code)
                    wait_idle(driver, timeout_s=20.0)
                    return True
                return False
        time.sleep(0.2)
    return False


def wait_idle(driver: MaaDriver, timeout_s: float = 30.0) -> bool:
    """等待 MAA 完全空闲（可安全下发并 start 下一个任务）。

    为什么需要它（实测教训）：``TaskChainCompleted`` 只表示**该链的任务跑完**，
    之后还会依次发出 ``AllTasksCompleted`` 与 ``TaskChainStopped``；
    在这两个事件落地前 assistant 仍处于忙碌态，此时 ``start()`` 会**直接返回
    False**（日志表现为 ``Start | block`` 紧跟 ``leave, 0 ms``），
    导致同一 driver 连续跑多任务时"秒失败"。

    故凡是"一个 driver 连续下发多个任务"的流程，切换任务前都要等到空闲。
    返回是否在超时内观察到空闲信号。
    """
    mark = driver.mark()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        evts, mark = driver.events_since(mark)
        for evt in evts:
            if evt.msg in ("AllTasksCompleted", "TaskChainStopped"):
                # 再排空一拍，确保没有紧随其后的事件
                time.sleep(0.3)
                driver.events_since(driver.mark())
                return True
        if not driver.running():
            time.sleep(0.3)
            return True
        time.sleep(0.15)
    return False


def wait_settle(
    driver: MaaDriver,
    timeout_s: float = 300.0,
    watchdog: BlockerSentry | None = None,
) -> bool:
    """作战收尾：把游戏带回可操作状态（用 MAA 的 ``StartUp``）。

    链收尾 ≠ 作战结束（Copilot 任务链在作业动作执行完即返回），故需再推一次
    结算。交给 MAA 的 ``StartUp``：它会自己识别并处理结算画面、公告等弹窗，
    不依赖像素阈值（原做法是轮询蓝钮像素点屏幕，已废弃）。

    ``watchdog``（2026-10-20 卡死自愈）：StartUp 不认识的阻断画面（判负
    结算变体 / 同步弹窗等）由哨兵在等待循环里像素检出并点屏消解；
    不传则行为与旧版完全一致。

    返回是否在超时内完成。**返回前会等到 MAA 完全空闲**（见 :func:`wait_idle`），
    以便调用方紧接着下发下一个任务（否则 `start()` 会因助手仍忙碌而秒失败）。
    """
    tid = driver.append_task("StartUp", {"client_type": "Official"})
    if not tid:
        return False
    mark = driver.mark()
    if not driver.start():
        # 与 navigate_only 同理：助手未空闲时 start 直接失败，等一下再试
        if not wait_idle(driver, timeout_s=30.0):
            return False
        if not driver.start():
            return False
    def _terminal(evts: list[Any]) -> bool | None:
        # 按 taskid 过滤，避免上一链（如 Copilot）的终态落进本链窗口造成跨链误判；
        # 再「先扫 Completed、后扫 Stopped/Error」，同一批里以 Completed 优先。
        def _mine(evt: Any) -> bool:
            d = evt.details if isinstance(evt.details, dict) else {}
            t = d.get("taskid")
            return t is None or str(t) == str(tid)

        completed = False
        halted = False
        for evt in evts:
            # AllTasksCompleted 表示队列已空；用 finished_tasks/taskid 确认本链在其中，
            # 避免上一链的 AllTasksCompleted 落进本窗被误判 True（两者都缺则放行）。
            if evt.msg == "AllTasksCompleted":
                dd = evt.details if isinstance(evt.details, dict) else {}
                fin = dd.get("finished_tasks")
                has_id = dd.get("taskid") is not None or fin is not None
                mine_fin = isinstance(fin, list) and any(
                    str(x) == str(tid) for x in fin
                )
                if not has_id or str(dd.get("taskid")) == str(tid) or mine_fin:
                    completed = True
                continue
            if not _mine(evt):
                continue
            if evt.msg == "TaskChainCompleted":
                completed = True
            elif evt.msg in ("TaskChainError", "TaskChainStopped"):
                halted = True
        if completed:
            return True
        if halted:
            return False
        return None

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if watchdog is not None:
            watchdog.poll()
        evts, mark = driver.events_since(mark)
        verdict = _terminal(evts)
        if verdict is not None:
            if verdict:
                wait_idle(driver, timeout_s=20.0)
            return verdict
        if not driver.running():
            # 驱动已停：终态事件**可能晚于** running() 变 False 才到（实测教训），
            # 故再排空一次缓冲，而不是立刻判失败、也不要空等满 300s。
            time.sleep(0.5)
            evts, mark = driver.events_since(mark)
            verdict = _terminal(evts)
            if verdict:
                wait_idle(driver, timeout_s=20.0)
            # TaskChainStopped / Error / 无终态：收尾未确认，快速返回 False
            return verdict if verdict is not None else False
        time.sleep(0.2)
    return False
