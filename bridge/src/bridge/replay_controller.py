"""作业重放调度模块（数据工厂主循环）。

职责（设计文档 §5.4 / §10）：按 jobs 文件逐个重放 prts.plus 作业：
构建 MAA 作业 JSON（保留来源元数据）→ 固定点位导航到关卡 briefing
→ 下发 Copilot 接管作战 → 全程采集（回调事件流 + 战场状态 + 1.5s 节拍截图 +
CopilotAction 即时补拍 + 结算抓帧）→ 写 episode 束（bundle 结构沿用
``data-validation/capture_episode.py`` 已验证格式）。

输入为 ``configs/jobs_main_v1.json``（内嵌转换好的 MAA 形态作业 +
来源署名，控制器离线可用）；输出交给 ``episode_writer`` 落盘。

战场状态采集（2026-10-09 新增）
-------------------------------
使用**打过补丁的 MaaCore**（``patches/0001-battle-state-callback.patch``）时，
每个作业动作执行前会收到 ``what="BattleState"`` 回调，含费用/击杀/待部署栏/
已部署干员；本模块将其转为 ``battle_states.jsonl`` 并写进 episode 清单
（``battle_states_file`` / ``battle_states`` 摘要）。

用**官方发行版**时不发此回调 → ``battle_states.jsonl`` 不生成、
``battle_states_file`` 为 None、``battle_states.n = 0``。
**两种运行方式产出的 bundle 结构兼容**，可混存；靠 ``battle_states_file``
是否为 None 区分这批数据有没有状态。

已知边界（如实记录不求完美）：

- Copilot 任务链在作业动作执行完即收尾，作战自然结束靠
  ``navigator.wait_battle_settle`` 轮询蓝钮推进（期间桌面上会闪过
  结算画面并被抓帧）；胜负判定属 TODO(M0.x)；
- 断点续跑仅按"episode 目录已存在即跳过"（``--skip-done``）；
- 中途失败（导航/下发失败）记摘要后继续下一局。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from bridge.battle_state import battle_state_to_record, summarize_states
from bridge.blocker_watchdog import (
    BLOCKER_SETTLE_LOST,
    BlockerSentry,
    resolve_battle_result,
)
from bridge.episode_writer import EpisodeWriter
from bridge.maa_driver import MaaDriver, load_instance_config
from bridge.nav_maa import (
    PRTS_TASKS,
    NavigationError,
    navigate_only,
    wait_idle,
    wait_settle,
)
from bridge.state_logger import StateLogger, is_dropped_shot

# 数据集动作类型 → MAA 作业动作白名单（其余如 Output/MoveCamera 丢弃）
_MAA_ACTION_TYPES = ("Deploy", "Skill", "Retreat", "SpeedUp", "SkillDaemon")

_CHAIN_TIMEOUT_S = 480.0
"""任务链最长执行时间（防卡死保护）。"""

_SETTLE_TIMEOUT_S = 300.0

# D4（2026-10-10 用户批准）：助战借用。1 = WhenNeeded——MAA 在“恰好缺 1 组”时
# 自动借入作业缺失的那名干员（名字由 MAA 推导）。用户原话：很多地图就是要靠助战借用的。
# 理论可达 v3 767 中的 721 条；实测 510/617 可直接命中，107 条需 D5 groups 支持。
# 置 0 可回到“不借用”的旧行为（对账/复现旧数据时用）。
_SUPPORT_UNIT_USAGE = 1
"""作战自然结束+结算推进的最长等待。"""


def stage_id_to_code(stage_id: str) -> str:
    """数据集 stage_id → 游戏关卡短码（MAA ``Fight`` 的 ``stage`` 参数）。

    **优先**用数据集 ``stages.jsonl`` 的 ``code`` 字段（权威、覆盖全部类型）；
    本函数只在拿不到 code 时按 stage_id 兜底推断。

    历史（重要，别重蹈）：本函数原为
    ``re.fullmatch(r"main_(\\d{2})-(\\d{2})")``，**只接受两位数的主线关卡**，
    其余直接抛 ``ValueError``——这使活动关（占数据集 71%）根本无法进队列，
    是"队列只有 6 条"的代码级根因（2026-10-09 查明）。

    现在支持的类型（依 ``stages.jsonl`` 的 ``type``）：
    - ``MAIN``：``main_00-01`` → ``0-1``（去掉前导零）
    - ``ACTIVITY``/``SUB``/``CAMPAIGN``/``DAILY``/``CLIMB_TOWER``/``GUIDE``：
      形如 ``act33side_01``、``sub_03-1-1``、``camp_01``、``wk_melee_1``——
      这些 **stage_id 本身往往就是 MAA 认的关名**（或与 code 一致），故原样返回，
      不再用正则拒绝。

    ⚠️ 关卡能否被 MAA 导航，取决于 MAA 资源版本与账号进度，**不由本函数保证**；
    无法导航时 ``Fight`` 任务会失败，由导航层如实记录（不静默跳过）。
    """
    m = re.fullmatch(r"main_(\d+)-(\d+)(#.*)?", stage_id)
    if m:
        return f"{int(m.group(1))}-{int(m.group(2))}"
    # 非 main_ 前缀：stage_id 去掉可能的 #f# 修饰后即可作为关名尝试
    return re.sub(r"#.*$", "", stage_id)


def dataset_record_to_job_entry(rec: dict[str, Any]) -> dict[str, Any]:
    """数据集 copilot.jsonl 单条 → 内嵌作业条目（离线可回放）。

    返回 ``{"source": {...来源署名}, "maa_job": {...MAA 作业形态}}``；
    编队要求整体省略（实测 MAA 会把 ``module: 0`` 当作模组硬要求，
    对无模组低星干员直接判 OperatorMissing；原始 req 留在 source 溯源）。
    """
    stage_id = rec["stage_id"]
    # 优先用数据集自带的权威 code（stages.jsonl 给出，覆盖全部关卡类型）；
    # 缺失时按 stage_id 兜底推断。见 stage_id_to_code 的历史说明。
    code = str(rec.get("stage_code") or "") or stage_id_to_code(stage_id)
    opers = []
    for op in rec.get("lineup", []):
        opers.append(
            {
                "name": op["name"],
                "skill": op.get("skill") or 1,
                "skill_usage": op.get("skill_usage") or 0,
            }
        )
    actions = []
    for act in rec.get("actions", []):
        if act["type"] not in _MAA_ACTION_TYPES:
            continue
        out: dict[str, Any] = {"type": act["type"]}
        for key in ("name", "location", "direction", "pre_delay", "post_delay"):
            val = act.get(key)
            if val not in (None, "", "None"):
                out[key] = val
        if act.get("kills"):
            out["kills"] = act["kills"]
        if act.get("costs"):
            out["costs"] = act["costs"]
        actions.append(out)
    source = {
        "type": "prts.plus",
        "job_id": rec["id"],
        "title": rec.get("title", ""),
        "like": rec.get("like", 0),
        "views": rec.get("views", 0),
        "rating_ratio": rec.get("rating_ratio"),
        "upload_time": rec.get("upload_time", ""),
        "stage_id_raw": stage_id,
        "lineup_raw": [
            {"name": op["name"], "id": op.get("id"), "req": op.get("req")}
            for op in rec.get("lineup", [])
        ],
        "comment": (
            f"来源 prts.plus 作业 #{rec['id']}（ak_dataset_v0.1 copilot.jsonl），"
            "编队要求已置零回放，原始 req 见 lineup_raw"
        ),
    }
    maa_job = {
        "stage_name": code,
        "opers": opers,
        "groups": rec.get("groups", []),
        "actions": actions,
        "doc": {
            "title": rec.get("title", "") or f"prts.plus #{rec['id']}",
            "details": (
                f"来源: prts.plus 作业 #{rec['id']}（ark-autoplay 数据工厂重放）"
            ),
        },
        "minimum_required": "v4.0.0",
    }
    return {"source": source, "maa_job": maa_job}


def assert_patched_runtime(inst: dict[str, Any], cfg_path: str | Path) -> None:
    """拒绝用未打补丁的官方运行时跑批（2026-10-10 事故，见 docs/adr/0006）。

    官方 MaaCore 不含 patch 0001-0004：``nav_only`` 失效 → Fight 走完整刷图链
    直接开打，并**启用游戏内代理指挥**（自抽号的号商存档）——正是用户红线：
    代理数据零价值且会污染数据集。默认实例曾经指错目录，导致 09:45 之后的所有
    批次都在跑官方 DLL，且现场无任何报错。这里在装配阶段就 fail fast。
    """
    maa_dir = str(inst.get("maa_dir", "")).replace("\\", "/")
    if "maa_runtime_patched" not in maa_dir:
        raise RuntimeError(
            f"instance={cfg_path} 的 maa_dir={maa_dir!r} 不是打过补丁的运行时，"
            "拒绝启动：未打补丁时 nav_only 失效、Fight 会直接开打并启用游戏内代理指挥。"
            "请使用 configs/instances/mumu_local_patched.yaml。"
        )


def normalize_groups(groups: Any) -> list[dict[str, Any]]:
    """把 prts 形态的候选干员组修成 MAA Copilot schema（2026-10-10 实测缺陷）。

    prts.plus 把候选组写成 ``{"name":..., "options":[{name,id,skill},...]}``；
    MAA 的 CopilotConfig 读的是 ``group_info.at("opers")``（CopilotConfig.cpp:168）。
    元素是 OperUsage。原样下发会被 set_params 拒绝 —— AsstAppendTask 返回 0、
    asst.log 出现 ``CopilotConfig parse failed``，整局白跑（实测 job_64196）。
    这里统一改名并只保留 MAA 认识的字段（``id`` 是 prts 内部字段，剔掉；
    req 维持与顶层 opers 同口径 = 不带 requirements，即不卡练度）。
    """
    out: list[dict[str, Any]] = []
    for g in groups or []:
        if not isinstance(g, dict):
            continue
        src = g.get("opers") if g.get("opers") is not None else (g.get("options") or [])
        ops: list[dict[str, Any]] = []
        for op in src:
            if not isinstance(op, dict) or not op.get("name"):
                continue
            item: dict[str, Any] = {"name": op["name"]}
            if isinstance(op.get("skill"), int) and op["skill"] > 0:
                item["skill"] = op["skill"]
            if op.get("skill_usage") is not None:
                item["skill_usage"] = op["skill_usage"]
            ops.append(item)
        if ops:
            out.append({"name": g.get("name") or "组", "opers": ops})
    return out


def load_jobs(path: str | Path) -> list[dict[str, Any]]:
    """读取 jobs 定义文件（支持 {"jobs":[...]} 或裸 list）。"""
    with open(path, encoding="utf-8") as f:
        jobs = json.load(f)
    if isinstance(jobs, dict) and "jobs" in jobs:
        jobs = jobs["jobs"]
    if not isinstance(jobs, list):
        raise ValueError(f"jobs 文件结构非法: {path}")
    for j in jobs:
        if "source" not in j or "maa_job" not in j:
            raise ValueError("jobs 条目缺 source/maa_job")
    return jobs


class ReplayController:
    """作业重放调度器：队列消费 jobs 文件，逐局产出 episode。"""

    def __init__(
        self,
        instance_cfg_path: str | Path,
        nav_cfg_path: str | Path,
        work_root: str | Path,
    ) -> None:
        """装配驱动/导航/写出；``work_root`` 为数据落盘根（机器本地）。"""
        self._work = Path(work_root)
        (self._work / "tmp" / "maa_jobs").mkdir(parents=True, exist_ok=True)
        maa_user = self._work / "debug" / "maa_user"
        maa_user.mkdir(parents=True, exist_ok=True)
        inst = load_instance_config(instance_cfg_path)
        assert_patched_runtime(inst, instance_cfg_path)
        self._driver = MaaDriver(inst, maa_user)
        # 终点屏校验标定（2026-10-21，1-11 假成功 RCA 修复方案 A）：
        # 从 nav_cfg（configs/nav_main.yaml）取 brief_blue_check 蓝钮区域/阈值，
        # 传给 nav_maa.navigate_only 做"链完成后再验终点屏"。配置不可用则
        # 关闭校验并显式告警（不拦工厂启动，但日志必须看得见）。
        self._brief_check: dict | None = None
        try:
            from bridge.navigator import Navigator  # noqa: PLC0415  # 装配期才用

            nav_cfg = Navigator.load_nav_config(nav_cfg_path)
            self._brief_check = dict(nav_cfg["brief_blue_check"])
        except Exception as exc:  # noqa: BLE001
            print(
                f"[factory] ⚠ 终点屏校验配置不可用（{exc!r}），导航不做终点屏校验",
                flush=True,
            )
        self._writer = EpisodeWriter(self._work)

    def _write_maa_job(self, entry: dict[str, Any]) -> Path:
        path = self._work / "tmp" / "maa_jobs" / f"job_{entry['source']['job_id']}.json"
        mj = dict(entry["maa_job"])
        # groups 必须归一化成 MAA schema，否则 CopilotConfig 直接解析失败、整局白跑
        if mj.get("groups"):
            mj["groups"] = normalize_groups(mj["groups"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mj, f, ensure_ascii=False, indent=1)
        return path

    def _run_one(self, entry: dict[str, Any]) -> dict[str, Any]:
        """重放一个作业，返回结果摘要。"""
        src = entry["source"]
        stage_code = entry["maa_job"]["stage_name"]
        stage_id = src["stage_id_raw"]
        job_id = src["job_id"]
        print(f"[job {job_id}] {stage_id} ({stage_code}) 开始", flush=True)
        summary: dict[str, Any] = {"job_id": job_id, "stage_code": stage_code}

        t0 = time.time()
        t0_wall = int(t0 * 1000)
        records: list[dict[str, Any]] = []
        states: list[dict[str, Any]] = []
        # 卡死自愈哨兵（2026-10-20）：贯穿全 episode 的阻断画面轮询/消解/记账，
        # 事件最终写进 episode.json 的 blockers_dismissed，判负屏供 battle_result。
        sentry = BlockerSentry(self._driver, t0_wall)

        def rec(msg: str, **kw: Any) -> None:
            records.append(
                {"t_ms": int(time.time() * 1000) - t0_wall, "msg": msg, **kw}
            )

        def rec_state(details: dict[str, Any]) -> None:
            """把 MaaCore 的 BattleState 回调转成协议的战场状态记录。

            需 patched MaaCore（``patches/0001-battle-state-callback.patch``）；
            官方发行版不会发此回调，届时本函数永不触发、``states`` 保持为空。
            """
            states.append(
                battle_state_to_record(
                    details,
                    int(time.time() * 1000) - t0_wall,
                    job_id=job_id,
                    stage_code=stage_code,
                )
            )

        # 两段式（对应 ADR-0002 的正确版）：
        #   0) 先把游戏带回主界面（StartUp）——**每局都必须做**；
        #   1) 用 Fight(nav_only=True) 把游戏从主页导航到关卡准备界面
        #      —— 依赖 patch 0002，只导航不开打、不碰游戏内代理作战；
        #   2) 交给 Copilot 忠实回放 prts.plus 玩家作业。
        # 为何必须分两段：Copilot 的 navigate_to_stage 只做「地图内滑动找关卡」，
        # 缺「主页→章节地图」的章节寻路（Episode{N}），在主页会卡死；
        # 而普通 Fight 会走 UsePrts（实测 Fight@PRTS1 出现 13 次），
        # 自抽号的代理记录是号商机械刷出的异常数值，对训练无价值。
        # 详见 bridge/docs/adr/0003~0005 与 nav_maa 模块 docstring。
        #
        # ⚠️ 第 0 步是**批量场景的必需项**（实测教训）：`Fight` 的 `StageBegin`
        # 要求从已知界面起步；上一局结束后游戏可能停在结算/地图/任意界面，
        # 直接 nav_only 会让 `StageBegin` 立即 SubTaskError、整批全灭
        # （实测 20 局批量 ok 0/20，每局仅 6 秒即失败）。
        if not ensure_home(self._driver, sentry, log_prefix=f"[job {job_id}]"):
            summary.update(ok=False, error="pre: StartUp 失败（已重试 2 次）")
            return summary

        # 2026-10-10：预导航（Fight nav_only）是**唯一**能导航到关卡准备界面的路径——
        # 实测去掉它后 Copilot 链直接 StartUp→BattleStartAll 不导航（卡在章节地图）。
        # 但它同时是**游戏内代理指挥的源头**（MAA 的 Fight 是刷图任务，会点开
        # 关卡详情页的“代理指挥”勾选框）。红线要求该勾选框必须为 OFF —— 由 DLL 侧
        # 根治（见 bridge/patches/0004 任务），桥侧无法在详情页插手中途步骤。
        nav_ok = False
        try:
            nav_ok = navigate_only(
                self._driver,
                stage_code,
                watchdog=sentry,
                endpoint_cfg=self._brief_check,
            )
        except NavigationError as exc:
            print(f"[job {job_id}] 导航失败: {exc}", flush=True)
            summary.update(ok=False, error=f"nav: {exc}")
            return summary
        if not nav_ok:
            print(f"[job {job_id}] 导航未完成（{stage_code}）", flush=True)
            summary.update(ok=False, error="nav: 未到准备界面(endpoint_check_failed)")
            return summary
        print(f"[job {job_id}] 已导航到 {stage_code} 准备界面（nav_only）", flush=True)

        run_dir = self._writer.begin(stage_id, job_id)
        job_file = self._write_maa_job(entry)

        def on_shot(tag: str, rel: str) -> None:
            if tag == "tick":
                rec("ShotTick", file=rel)
            elif tag == "action":
                rec("ShotOnAction", file=rel)
            else:
                rec("Shot", tag=tag, file=rel)

        logger = StateLogger(self._driver, run_dir, on_shot=on_shot)
        logger.start()
        logger.snap("briefing")
        # 判负证据窗口起点（Copilot 下发前）：窗口内哨兵消解的 settle_lost
        # 是本局判负的直接证据。窗口不从 episode 头起算——避开上一局残留的
        # 判负屏在 pre-nav 阶段被消解后**错记到本局头上**。
        battle_mark = sentry.count()

        tid = self._driver.append_copilot(
            job_file, formation=True, support_unit_usage=_SUPPORT_UNIT_USAGE
        )
        if tid <= 0:
            logger.stop()
            print(f"[job {job_id}] append_copilot 被拒绝 (tid={tid})", flush=True)
            summary.update(ok=False, error=f"append tid={tid}")
            (run_dir / "error.txt").write_text(f"append tid={tid}", encoding="utf-8")
            return summary

        mark = self._driver.mark()
        started = self._driver.start()
        if not started:
            # 助手未空闲时 start() 会直接返回 False（见 nav_maa.wait_idle 说明）。
            # 由 navigate_only 返回前已等到空闲，此处属兜底重试。
            started = wait_idle(self._driver, timeout_s=30.0) and self._driver.start()
        if not started:
            logger.stop()
            summary.update(ok=False, error="asst.start 失败")
            return summary

        # 作战阶段开始：允许哨兵做剧情冻结停滞检测（主页/地图合法静滞期
        # 绝不许触发连点；story_freeze 窗口从激活后起算）。
        sentry.set_battle_active(True)

        # 作战过程：轮询事件流，CopilotAction 触发补拍
        cursor = mark
        terminal: str | None = None
        seen_actions = 0
        prts_seen: list[str] = []
        # 战后阶段（Copilot 链结束之后）命中的 PRTS 任务名，单独记账供下游判断
        prts_after: list[str] = []
        # 失败子任务留痕：汇总里必须能直接看出失败阶段与原因，
        # 否则 Copilot 阶段失败（如 BattleFormationTask OperatorMissing）只能翻日志。
        failures: list[str] = []
        deadline = time.monotonic() + _CHAIN_TIMEOUT_S

        def scan_prts(evt: Any, *, after: bool) -> None:
            """代理护栏（整局扫描）：prts_detected 必须覆盖全流程。

            2026-10-10 实测缺陷：原先只在 Copilot 阶段扫（drain 内），收尾/
            StartUp 阶段的事件虽被 rec() 记录却不扫描，导致 22/42 局命中
            PRTS_TASKS 而 prts_detected 仍为空（漏报）。after=True 表示该事件
            发生在 Copilot 链结束之后，单独记账便于下游区分战后清理分支。
            """
            d = evt.details if isinstance(evt.details, dict) else {}
            det = d.get("details") if isinstance(d.get("details"), dict) else {}
            t = str(d.get("task") or det.get("task") or d.get("cur_task") or "")
            if not t:
                return
            tail = t.rsplit("@", 1)[-1]
            if tail not in PRTS_TASKS:
                return
            if tail not in prts_seen:
                prts_seen.append(tail)
            if after and tail not in prts_after:
                prts_after.append(tail)

        def drain(fresh: list[Any]) -> None:
            """处理一批事件（收尾时也要跑，见下方说明）。"""
            nonlocal terminal, seen_actions
            for evt in fresh:
                rec(evt.msg, details=evt.details)
                if evt.msg == "TaskChainStart":
                    logger.set_active(True)
                elif evt.msg == "SubTaskExtraInfo":
                    what = evt.details.get("what")
                    if what == "CopilotAction":
                        seen_actions += 1
                        logger.snap("action")
                    elif what == "BattleState":
                        # patched MaaCore 才有；官方发行版不会走到这里
                        rec_state(evt.details.get("details") or {})
                elif evt.msg == "SubTaskError":
                    d0 = evt.details if isinstance(evt.details, dict) else {}
                    raw_inner = d0.get("details")
                    inner = raw_inner if isinstance(raw_inner, dict) else {}
                    cls = str(d0.get("class", "")).rsplit("::", 1)[-1]
                    first = d0.get("first") or []
                    why = inner.get("why") or d0.get("why") or ""
                    tag = f"{cls}:{'/'.join(str(x) for x in first)}"
                    failures.append(f"{tag} why={why}" if why else tag)
                elif evt.msg in ("TaskChainCompleted", "TaskChainError"):
                    terminal = evt.msg
                scan_prts(evt, after=False)

        while self._driver.running() and terminal is None:
            # 卡死自愈：作战中弹出的阻断画面（如"干员等级过低"红横幅会让
            # MAA 链静止）在此轮询消解；哨兵内部 1s 节流，不加 adb 压力。
            sentry.poll()
            fresh, cursor = self._driver.events_since(cursor)
            drain(fresh)
            if time.monotonic() > deadline:
                print(f"[job {job_id}] 链超时，stop", flush=True)
                self._driver.stop()
                time.sleep(2.0)
                terminal = terminal or "Timeout"
                break
            time.sleep(0.2)

        # 关键：`running()` 变 False 是**先于**我们读到终态事件的——
        # 任务链一结束驱动即停，而含 `TaskChainCompleted` 的那批事件尚未被取走。
        # 若不在此补取，terminal 会永远停在 None、把成功误判为失败
        # （实测：episode 里明明有 TaskChainCompleted，却报"任务链收尾=None"）。
        fresh, cursor = self._driver.events_since(cursor)
        drain(fresh)
        # 作战阶段结束：结算推进期改由画面型阻断器值守（settle_lost 等），
        # 避免把"等待中的静止结算屏"误判成剧情冻结乱点。
        sentry.set_battle_active(False)

        print(
            f"[job {job_id}] 任务链收尾={terminal} 动作事件={seen_actions}", flush=True
        )
        if prts_seen:
            print(
                f"[job {job_id}] ⚠ 检测到代理作战痕迹 {prts_seen}"
                "——本局数据来源不纯，episode 已标记 prts_detected",
                flush=True,
            )

        # 作战自然结束 + 结算推进：链收尾≠作战结束。
        # 交给 MAA 的 StartUp 回主界面——它会自己识别并处理结算/公告等弹窗，
        # 不依赖像素阈值（原做法是轮询蓝钮像素点屏幕，已随自建导航一并废弃）。
        logger.set_active(False)
        result_files: list[str] = []
        rel = logger.snap("settle")
        result_files.append(rel)
        settled = wait_settle(
            self._driver, timeout_s=_SETTLE_TIMEOUT_S, watchdog=sentry
        )
        # 判负证据窗口（Copilot 下发起算，见上 battle_mark）：覆盖作战中途
        # 判负与结算推进判负两种时序；消解动作本身经过哨兵双确认。
        lost_hits = sentry.hits_since(battle_mark, BLOCKER_SETTLE_LOST)
        logger.stop()
        # D9 接线（2026-10-10）：黑帧已由 state_logger 丢弃，snap 会返回
        # "BLACK_DROPPED:<n>" 标记；结果帧列表里不能混入这种标记。
        result_files = [r for r in result_files if not is_dropped_shot(r)]
        fresh, cursor = self._driver.events_since(cursor)
        for evt in fresh:
            rec(evt.msg, details=evt.details)
            scan_prts(evt, after=True)
        print(
            f"[job {job_id}] 结算推进 settled={settled} snaps={len(result_files)}",
            flush=True,
        )

        self._writer.write_events(run_dir, records)
        states_path = self._writer.write_battle_states(run_dir, states)
        state_summary = summarize_states(states)
        result_first = result_files[0] if result_files else ""
        result_last = result_files[-1] if result_files else ""
        episode = {
            "episode_id": f"replay-{job_id}-{time.strftime('%Y%m%d-%H%M%S')}",
            "stage_id": stage_id,
            "stage_code": stage_code,
            "source": {
                "type": "prts.plus",
                "job_id": job_id,
                "note": src.get("comment", ""),
            },
            "job_file": str(job_file),
            "result_screen_files": [result_first, result_last],
            "action_events_file": "action_events.jsonl",
            # 卡死自愈（2026-10-20）：全 episode 阻断画面消解记录
            # [{kind, t_ms, n}]，t_ms 与 action_events.jsonl 同基准；
            # 空列表 = 全程无阻断，不省略该字段（与旧 bundle 区分口径）。
            "blockers_dismissed": sentry.events(),
            # 胜负标记（bridge 自有字段，不动 core/protocol）：
            # "win"/"lost"/"unknown"，规则见 blocker_watchdog.resolve_battle_result。
            # 方案 §10「只留通关局」→ 下游按 battle_result == "win" 过滤。
            "battle_result": resolve_battle_result(
                lost_confirmed=bool(lost_hits),
                chain_completed=(terminal == "TaskChainCompleted"),
                settled=settled,
            ),
            # 失败阶段摘要（可选字段，bridge 自有；与 battle_result 正交）：
            # nav|formation|start|battle|done，推导见 classify_fail_stage。
            "fail_stage": classify_fail_stage(terminal, failures),
            # 战场状态（需 patched MaaCore）：无记录时为 None 且不落文件
            "battle_states_file": states_path.name if states_path else None,
            "battle_states": state_summary,
            # 数据来源纯净度：正常应为空列表。非空表示本局出现了游戏内代理作战
            # 痕迹（自抽号代理是号商机械刷出的异常数值，无训练价值）——
            # 下游按此字段剔除即可，不必重跑或人工翻日志。
            "prts_detected": prts_seen,
            # 战后阶段单独记账（护栏整局覆盖后的新增维度）
            "prts_detected_after_battle": prts_after,
            "shot_files": None,
            "duration_ms": int((time.time() - t0) * 1000),
            "copilot_actions": seen_actions,
            "settled_back_to_briefing": settled,
            # D9：本局丢弃的黑帧数（黑帧=过场/转场伪影，不落盘只计数）
            "shots_dropped_black": logger.shots_dropped_black,
        }
        ep_path = self._writer.write_episode(run_dir, episode)
        if terminal != "TaskChainCompleted":
            summary.update(
                error=(
                    f"copilot 链失败 terminal={terminal}; "
                    + ("; ".join(failures[-3:]) or "无 SubTaskError 记录")
                )
            )
        summary.update(
            ok=terminal == "TaskChainCompleted",
            episode=str(ep_path),
            copilot_actions=seen_actions,
            battle_states=state_summary["n"],
            prts_detected=prts_seen,
            prts_detected_after_battle=prts_after,
            settled=settled,
            shots_dropped_black=logger.shots_dropped_black,
            battle_result=episode["battle_result"],
            fail_stage=episode["fail_stage"],
            blockers_dismissed=len(sentry.events()),
        )
        print(
            f"[job {job_id}] episode -> {ep_path}  "
            f"战场状态 {state_summary['n']} 条{state_summary}",
            flush=True,
        )
        return summary

    def run(
        self,
        jobs_path: str | Path,
        only: set[str] | None = None,
        limit: int | None = None,
        skip_done: bool = False,
    ) -> list[dict[str, Any]]:
        """消费 jobs 队列；``only`` 按 job_id 过滤，``skip_done`` 跳过已落盘局。"""
        jobs = load_jobs(jobs_path)
        if only:
            jobs = [j for j in jobs if str(j["source"]["job_id"]) in only]
        if skip_done:
            jobs = [j for j in jobs if not self._has_episode(j)]
        if limit:
            jobs = jobs[:limit]
        if not jobs:
            print("[factory] 无待跑作业")
            return []
        if not self._driver.connect():
            raise RuntimeError("MAA 连接失败")
        summaries = []
        try:
            for entry in jobs:
                summaries.append(self._run_one(entry))
        finally:
            self._driver.close()
        out = self._work / "debug" / f"factory_summary_{int(time.time())}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=1)
        print(f"[factory] 完成 {len(summaries)} 局 -> {out}")
        return summaries

    def _has_episode(self, entry: dict[str, Any]) -> bool:
        ep_dir = self._work / "episodes" / entry["source"]["stage_id_raw"]
        marker = f"job-{entry['source']['job_id']}-"
        return has_completed_episode(ep_dir, marker)

def ensure_home(
    driver: MaaDriver,
    sentry: BlockerSentry,
    *,
    log_prefix: str = "[factory]",
    attempts: int = 2,
    settle_timeout_s: float = _SETTLE_TIMEOUT_S,
    idle_wait_s: float = 20.0,
) -> bool:
    """把游戏带回主界面（StartUp），失败做有界重试；供工厂与闭环运行器共用。

    实测（2026-10-10 批量）：收尾瞬间的弹窗/非主页态会让 StartUp 链偶发失败，
    而**紧接重试往往 8 秒内即成功**。故这里做有界重试，避免把可恢复的局
    直接记成 pre 失败（v6 批量里这类损失占 3/20）。重试次数/等待均为参数，
    调用方按需从自己的配置注入。
    """
    if wait_settle(driver, timeout_s=settle_timeout_s, watchdog=sentry):
        return True
    for attempt in range(1, attempts + 1):
        if wait_idle(driver, timeout_s=idle_wait_s) and wait_settle(
            driver, timeout_s=settle_timeout_s, watchdog=sentry
        ):
            print(f"{log_prefix} 前置回主页第 {attempt} 次重试成功", flush=True)
            return True
        print(f"{log_prefix} 前置回主页重试 {attempt} 仍失败", flush=True)
    print(f"{log_prefix} 前置回主页失败（StartUp）", flush=True)
    return False


def classify_fail_stage(terminal: str | None, failures: list[str]) -> str:
    """推导 episode 的 ``fail_stage`` 字段（nav|formation|start|battle|done）。

    口径（2026-10-21 拍板，与 RCA 报表同源）：

    - 任务链正常收尾 → ``"done"``（流程走完；胜负由 ``battle_result`` 承担，
      两字段正交——``lost+done``、``lost+battle`` 都合法）；
    - 编队被拒（``BattleFormationTask``，如 OperatorMissing）→ ``"formation"``；
    - 进战斗前序失败（``BattleStartPre``/``BattleStartAll``）→ ``"start"``；
    - 其余（作战中链错误/超时/无失败子任务留痕）→ ``"battle"``。

    ``"nav"`` 预留在枚举内：当前导航失败会在写 episode 之前提前返回，
    不产生 episode.json，故线上暂时不会出现该值。
    """
    if terminal == "TaskChainCompleted":
        return "done"
    tags = ";".join(failures)
    if "BattleFormationTask" in tags:
        return "formation"
    if "BattleStartPre" in tags or "BattleStartAll" in tags:
        return "start"
    return "battle"


def has_completed_episode(ep_dir: str | Path, marker: str) -> bool:
    """断点续跑判定：``ep_dir`` 下存在带 ``episode.json`` 的 ``marker*`` 目录。

    杀进程会留下只有 ``shots/`` 的半成品目录——按"目录存在即完成"的旧口径
    这类局会被永久跳过且永不可见。收紧为以 ``episode.json`` 为准
    （它是 _run_one 最后写出的文件，存在即代表本局流程走完）。
    """
    ep_dir = Path(ep_dir)
    return ep_dir.is_dir() and any(
        p.is_dir()
        and p.name.startswith(marker)
        and (p / "episode.json").is_file()
        for p in ep_dir.iterdir()
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    """工厂 CLI：``python -m bridge.replay_controller --work <数据根>``。"""
    root = _repo_root()
    parser = argparse.ArgumentParser(description="ark-autoplay 重放数据工厂 v1WIP")
    parser.add_argument(
        "--jobs", default=str(root / "configs" / "jobs_main_v1.json"), help="作业队列"
    )
    parser.add_argument(
        "--instance",
        default=str(root / "configs" / "instances" / "mumu_local_patched.yaml"),
        help="实例定义",
    )
    parser.add_argument(
        "--nav", default=str(root / "configs" / "nav_main.yaml"), help="导航点位表"
    )
    parser.add_argument("--work", required=True, help="数据落盘根（本地目录）")
    parser.add_argument("--only", default=None, help="只跑指定 job_id（逗号分隔）")
    parser.add_argument("--limit", type=int, default=None, help="最多跑 N 个")
    parser.add_argument(
        "--skip-done", action="store_true", help="跳过已有 episode 目录的作业"
    )
    args = parser.parse_args()

    # Windows 控制台默认 GBK：日志里的对钩/叉号会抛 UnicodeEncodeError 并终止整批
    # （2026-10-10 实测：H5-1 导航成功后死在 print 上，零 episode 落盘）。
    # 把编码错误降级为替换字符，保证批量不因日志字符中断。
    for _stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            _stream.reconfigure(errors="replace")

    controller = ReplayController(args.instance, args.nav, args.work)
    only = set(args.only.split(",")) if args.only else None
    summaries = controller.run(
        args.jobs, only=only, limit=args.limit, skip_done=args.skip_done
    )
    ok = sum(1 for s in summaries if s.get("ok"))
    print(f"[factory] ok {ok}/{len(summaries)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
