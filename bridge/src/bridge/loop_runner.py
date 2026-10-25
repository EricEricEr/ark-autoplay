"""模型闭环运行器：模型实时看状态、实时下动作、自行打关（v1 冒烟演示链路）。

定位与诚实口径（**不是能力证据**）
----------------------------------
本模块是「模型闭环打关」里程碑的第三环：core 侧 ``track_a_v1.pt`` 是约 1k
状态-动作对上的链路冒烟模型（训练侧已声明不作能力证据），本模块同样只证明
「battle_state → greedy_step → SingleStep 动态下发 → 战斗自然结束」这条
**在线闭环链路可走通**。任何对外表述不得夸大为自主作战能力。

跑法（必读）
------------
bridge 与 core 分属不同 uv 环境：推理依赖 torch，bridge venv **无** torch
（约定不往 bridge pyproject 加 torch）。故从 **core 的 uv 环境**运行本模块，
用 PYTHONPATH 注入 ``bridge/src``::

    cd <repo>/core
    PYTHONPATH="<repo>/bridge/src" ARK_ML_ROOT="<local-data>" \
        uv run python -m bridge.loop_runner \
        --work "<local-loop-data>" \
        --config "../bridge/configs/loop_milestone_v1.json"

ark_core 与模型/torch 的 import 全部惰性（``load_core_policy`` 首次调用时发生），
bridge venv 无 torch 也能 import 本模块、跑单测；``--bridge-src`` 方案不可行
（``-m`` 的包解析早于 main 的参数注入），已如实放弃。bridge 侧的 PIL 依赖
同样已全部惰性化：core venv 无 Pillow 可完成 import 与离线决策冒烟；**真机运行
截图/像素校验仍需 Pillow**，在 core venv 机器本地补装 ``uv pip install Pillow``
即可（不改 core pyproject；长期入依赖请走 core 仓自己的流程）。

技术链路（依据 pin 版 MaaCore 源码与 patches 0001/0002/0005）
-------------------------------------------------------------
1. **进图编队**（原样复用 replay_controller 链路）：``ensure_home`` 回主页 →
   ``nav_maa.navigate_only``（``Fight nav_only``，patch 0002）到 briefing →
   空作业 Copilot 壳（opers 保留、actions 空数组）完成自动编队+开打，链秒收尾，
   之后战斗**自由运行**（不由 MAA 托管）。
2. **心跳取状态**：动作 ``{"type": "Output"}`` 是真 no-op，patch 0001 的
   BattleState 回调在每个动作执行前无条件触发（patch 0005 会先取帧刷新
   costs/kills）。**字段现实（2026-10-24 实测澄清）**：Output 心跳链上只有
   ``costs``/``kills``/``total_kills`` 被刷新，``in_battle``/``hand``/
   ``deployed``/``stage_name`` 全是开链时的陈旧值——上游
   ``SingleStepBattleProcessTask._run`` 不做 ``update_deployment``（正常
   Copilot 的 ``BattleProcessTask._run`` 开头才做）。不改 MaaCore 的前提下，
   本模块以**数值签名 (costs,kills,total_kills) 的鲜活度**做战斗开闸/结束
   判据（``_vitals_sig``），并向模型喂台账合成的完整状态
   （``synthesize_battle_state``，hand/deployed 按已下发动作推导）。
3. **动态下动作**：``SingleStep`` 任务（接口源码
   ``maacore/_src/src/MaaCore/Task/Interface/SingleStepTask.cpp``）——append 时
   用 ``{"type":"copilot","subtype":"stage","details":{"stage_name": ...}}``
   声明关卡，再以 ``set_task_params(tid, {"type":"copilot","subtype":
   "action","details":{"actions": [...]}})`` 挂动作子任务。
   **pin 版 ``CopilotConfig::parse_actions`` 读 ``details.at("actions")``，
   details 必须是 ``{"actions": […]}`` 包裹形态**（裸数组会被拒）。
   通道实现有 A/B 两路（``channel_mode`` 配置），**默认 B（append_per_action，
   一推一链）**：A 路（set_params 原地续喂）2026-10-24 真机实测会丢心跳——
   stage-only 链同毫秒完成、set_params 落到已完成任务上被静默拒收
   （asst.log 10:45:45.565 TaskChainStart→Completed 全毫秒并排、全程零
   SubTaskStart/BattleState），两连拒 → channel_dead；B 路「先参后启」
   无此竞态（细节见两通道类的 docstring）。
4. **空闲恢复**：单链跑完 assistant 自动停；下一推重新 append + start，
   语义沿用 ``nav_maa.wait_idle``（A 路通道内部处理，B 路天然一推一链）。
5. **判终/结算**：复用 ``blocker_watchdog`` 与 StartUp 结算推进
   （``nav_maa.wait_settle``）。判终全部基于数值签名：**开闸** = 签名连续
   ``gate_change_ticks`` 拍变化（编队壳链收尾 ≠ 战斗打响——首跑 0-3 用
   in_battle 判死在加载屏就收工放跑了防御；用 in_battle 字段本身则可证
   在本链路上恒脏）；**自然结束** = 开闸后签名连续 ``freeze_ticks`` 拍冻结
   （结算屏无计数 UI、数值停更）；兜底是无状态超时、整链超时
   （闸门未开记 ``battle_never_started``）。动作下发被拒不再中止循环，
   记 ``actions_lost`` 继续（Deploy 自等待超回收窗口属可恢复情形）。
6. **落盘**：沿用 ``episode_writer`` 写 bundle；``source.type`` 用新值
   ``"model_loop"``；episode 额外带 ``model_actions``（模型实际下发的作业
   形态动作序列）与 ``model_loop`` 统计段（动作数/节拍数/拦截次数等）。
   （episode_writer 的动作落盘接口 = ``write_events`` 的 action_events.jsonl，
   **无需改动**：沿用事件流落盘整局，模型动作另以独立字段进 episode.json。）

测试边界：本模块单测为纯逻辑（假 driver / 假 policy / 假状态流），不连 MAA；
core 侧 import 全部惰性，bridge venv 可无 torch import 本模块。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

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
from bridge.replay_controller import assert_patched_runtime, ensure_home
from bridge.state_logger import StateLogger, is_dropped_shot

if TYPE_CHECKING:
    from bridge.maa_driver import MaaEvent

HEARTBEAT_ACTION: dict[str, Any] = {"type": "Output"}
"""心跳动作：上游真 no-op，触发一次 BattleState 回调即取到新鲜快照。"""

_EVENT_DRAIN_S = 0.1
"""真机节拍流内事件排空的睡眠粒度（秒）。"""

_IDLE_RESTART_WAIT_S = 30.0
"""空闲恢复时 start() 失败后等待 idle 再重试的上限（秒），与 navigate_only 同款。"""


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class LoopRunConfig:
    """节拍/超时/重试上限（全部来自 configs/loop_milestone_v1.json）。"""

    heartbeat_s: float
    chain_timeout_s: float
    shell_timeout_s: float
    settle_timeout_s: float
    pre_retry_attempts: int
    pre_retry_idle_wait_s: float
    no_state_timeout_ticks: int
    max_model_actions: int
    support_unit_usage: int
    channel_mode: str = "append_per_action"
    gate_change_ticks: int = 2
    """战斗开幕闸门：连续 N 个状态帧数值签名发生变化即判定战斗已打响。"""
    freeze_ticks: int = 6
    """战斗结束判据：战斗打响后连续 N 个状态帧数值签名冻结即自然结束。"""


@dataclass
class MilestoneStage:
    """一个里程碑关卡定义（参考 bundle 溯源 + 编队 + 动作数代理）。"""

    stage_id: str
    stage_code: str
    ref_job_id: int
    ref_episode_dir: str
    total_proxy: int
    opers: list[dict[str, Any]]
    note: str = ""


@dataclass
class LoopConfig:
    """里程碑配置整体：运行参数 + 路径 + 关卡列表。"""

    run: LoopRunConfig
    stages: list[MilestoneStage]
    ckpt_path: str | None = None
    npz_path: str | None = None
    infer_cfg_path: str | None = None
    infer_overrides: dict[str, Any] | None = None
    """core ``InferConfig`` 字段覆盖（``dataclasses.replace`` 粒度）——里程碑
    关卡专属调参留在 bridge 侧，不污染 core 的 infer_v1.yaml（连带其单测）。"""


def load_loop_config(path: str | Path) -> LoopConfig:
    """读取里程碑配置（``configs/loop_milestone_v1.json``）。"""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"loop 配置不是 mapping: {path}")
    r = raw.get("run") or {}
    if not r:
        raise ValueError("loop 配置缺 run 段")
    run = LoopRunConfig(
        heartbeat_s=float(r["heartbeat_s"]),
        chain_timeout_s=float(r["chain_timeout_s"]),
        shell_timeout_s=float(r["shell_timeout_s"]),
        settle_timeout_s=float(r["settle_timeout_s"]),
        pre_retry_attempts=int(r["pre_retry_attempts"]),
        pre_retry_idle_wait_s=float(r["pre_retry_idle_wait_s"]),
        no_state_timeout_ticks=int(r["no_state_timeout_ticks"]),
        max_model_actions=int(r["max_model_actions"]),
        support_unit_usage=int(r.get("support_unit_usage", 0)),
        channel_mode=str(r.get("channel_mode", "append_per_action")),
        gate_change_ticks=int(r.get("gate_change_ticks", 2)),
        freeze_ticks=int(r.get("freeze_ticks", 6)),
    )
    stages = []
    for s in raw.get("stages") or []:
        stages.append(
            MilestoneStage(
                stage_id=str(s["stage_id"]),
                stage_code=str(s["stage_code"]),
                ref_job_id=int(s["ref_job_id"]),
                ref_episode_dir=str(s.get("ref_episode_dir", "")),
                total_proxy=int(s["total_proxy"]),
                opers=[dict(o) for o in s.get("opers") or []],
                note=str(s.get("note", "")),
            )
        )
    if not stages:
        raise ValueError("loop 配置缺 stages 段")
    paths = raw.get("paths") or {}
    # 相对路径以配置文件所在目录为根；环境变量在此展开，避免发布开发机绝对路径。
    def config_path(key: str) -> str | None:
        value = paths.get(key)
        if not value:
            return None
        expanded = os.path.expandvars(str(value))
        if Path(expanded).is_absolute() or expanded.startswith("/"):
            return expanded
        return str((Path(path).resolve().parent / expanded).resolve())
    overrides = raw.get("infer_overrides")
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("infer_overrides 必须是 mapping")
    if overrides:
        overrides = {k: v for k, v in overrides.items() if not k.startswith("_")}
    return LoopConfig(
        run=run,
        stages=stages,
        ckpt_path=config_path("ckpt"),
        npz_path=config_path("npz"),
        infer_cfg_path=config_path("infer_cfg"),
        infer_overrides=overrides,
    )


# ---------------------------------------------------------------------------
# 决策与下发通道的抽象（纯逻辑测试的注入点）
# ---------------------------------------------------------------------------


class LoopPolicy(Protocol):
    """模型侧决策接口（core track_a_v1 的适配签名；单测用假 policy）。"""

    def decide(
        self,
        battle_state: dict[str, Any],
        history: list[dict[str, Any]],
        *,
        t: int,
    ) -> dict[str, Any] | None:
        """看一帧状态 + 动作历史，给一个作业形态动作；无动作返回 None。"""
        ...


class RecordingPolicy:
    """LoopPolicy 透传适配器：把模型动作在决策当下回调出去（真时间戳记账用）。"""

    def __init__(
        self, inner: LoopPolicy, on_action: Callable[[dict[str, Any], int], None]
    ) -> None:
        """``on_action(action, tick)`` 在每次 inner 决策出动作时同步调用。"""
        self._inner = inner
        self._on_action = on_action

    def decide(
        self,
        battle_state: dict[str, Any],
        history: list[dict[str, Any]],
        *,
        t: int,
    ) -> dict[str, Any] | None:
        act = self._inner.decide(battle_state, history, t=t)
        if act:
            self._on_action(act, t)
        return act


class SingleStepChannel(Protocol):
    """动作下发通道接口（路径 A 原喂 / 路径 B 骨架 / 单测假实现共用）。"""

    opens: int
    """累计开张次数（链路 idle 后重开的计数，闭环统计用）。"""

    def open_stage(self) -> int:
        """确保下发链路存活（idle 则重开），返回 task id（0=失败）。"""
        ...

    def push_actions(self, actions: list[dict[str, Any]]) -> bool:
        """把一批作业形态动作喂给运行中的链路，返回是否被接受。"""
        ...


@dataclass
class TickSample:
    """一个心跳节拍的采样：本拍心跳是否受理 + 采到的最新状态（None=未采到）。"""

    state: dict[str, Any] | None
    heartbeat_ok: bool = True


@dataclass
class LoopStats:
    """战斗段闭环统计（结果 JSON 与 episode.model_loop 段共用口径）。"""

    ticks: int = 0
    """采到状态的节拍数（含战斗开幕前的等待拍）。"""
    heartbeats: int = 0
    """心跳（Output）下发次数。"""
    actions_sent: int = 0
    """模型动作下发成功次数。"""
    actions_lost: int = 0
    """模型动作下发被拒/回收超时次数（不计入 history，不中止循环）。"""
    idle_restarts: int = 0
    """链路 idle 后重开次数（= 累计开张数 - 1；append 模式约为链数-1）。"""
    no_state_max_streak: int = 0
    """最长连续未采到状态的节拍数。"""
    battle_started: bool = False
    """战斗开幕闸门是否已开（见 battle_decision_loop 的闸门说明）。"""
    terminal_cause: str | None = None
    """收尾原因：out_of_battle / battle_never_started / no_state_timeout /
    timeout / channel_dead / action_cap / stream_end。"""
    model_actions: list[dict[str, Any]] = field(default_factory=list)
    """模型实际下发的作业形态动作序列（episode.model_actions 原样落盘）。"""


def _vitals_sig(state: dict[str, Any]) -> tuple[Any, Any, Any]:
    """状态帧的「鲜活度签名」：(costs, kills, total_kills)。

    注意：Patch 版 BattleState 在 Output 心跳链上**只有这三项会被刷新**
    （patch 0005 每拍取帧更新 costs/kills），``in_battle``/``hand``/
    ``deployed``/``stage_name`` 一律是开链时的陈旧值（上游
    ``SingleStepBattleProcessTask._run`` 不调 ``update_deployment``，
    与正常 Copilot 的 ``BattleProcessTask._run`` 不同——这是 pin 版源码事实，
    不允许改 MaaCore，只能桥侧绕过）。签名的"变化/冻结"因此成为战斗
    进程的唯一可靠探针：战斗打响后 costs 随时间自增（kills 随击杀步进）；
    战斗结束后计数 UI 消失、数值冻结。
    """
    return (
        state.get("costs"),
        state.get("kills"),
        state.get("total_kills"),
    )


def synthesize_battle_state(
    emit: dict[str, Any],
    squad_names: list[str],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """原始心跳快照 + 编队名册 + 已下发动作台账 → 喂给模型的完整状态。

    为什么需要它（诚实口径）：Output 心跳链上的 hand/deployed 恒为空列表
    （上游不刷新，见 ``_vitals_sig`` 说明），直接喂模型会让 deploy/skill
    全部被合法性层判死。桥侧按**自己下发的动作台账**推导：

    - ``hand`` = 编队中尚未部署的干员（available=True、cost 缺失——core 的
      口径是缺 cost 按可负担处理；真正在 CD / 阵亡离场无法感知，由 MAA 侧
      Deploy 动作自等待兜底，见 :class:`MaaAppendPerActionChannel` 动作链
      自等待语义）；
    - ``deployed`` = 台账里 Deploy 且未 Retreat 的干员**名字**（无格子坐标，
      因而可部署格不被占用扣减——模型可能选到已占格，由游戏自身拒绝兜底）；
    - Retreat 后的干员按 ``cooling=True`` 放回 hand（保守：不再怂恿重部署）。

    此外把 ``in_battle`` 强制置 True：**开闸判断已由 battle_decision_loop
    的数值鲜活度完成**，喂给模型时战斗必然在进行中。
    """
    deployed_names: list[str] = []
    for act in history:
        name = str(act.get("name") or "")
        if not name or name not in squad_names:
            continue
        atype = str(act.get("type") or "")
        if atype == "Deploy" and name not in deployed_names:
            deployed_names.append(name)
        elif atype == "Retreat" and name in deployed_names:
            deployed_names.remove(name)
    hand = [
        {
            "name": n,
            "available": True,
            "cooling": _was_retreated(n, history),
        }
        for n in squad_names
        if n not in deployed_names
    ]
    deployed = [{"name": n} for n in deployed_names]
    return {
        **emit,
        "in_battle": True,
        "hand": hand,
        "deployed": deployed,
        "synthesized": True,
    }


def _was_retreated(name: str, history: list[dict[str, Any]]) -> bool:
    """台账里最近一个针对该干员的动作是否 Retreat（once CD 不可知，保守记冷却）。"""
    outcome = False
    for act in history:
        if str(act.get("name") or "") != name:
            continue
        t = str(act.get("type") or "")
        if t == "Retreat":
            outcome = True
        elif t == "Deploy":
            outcome = False
    return outcome


def battle_decision_loop(
    *,
    channel: SingleStepChannel,
    policy: LoopPolicy,
    tick_samples: Iterable[TickSample],
    cfg: LoopRunConfig,
    squad_names: list[str],
    clock: Callable[[], float] = time.monotonic,
    on_tick: Callable[[], None] | None = None,
) -> LoopStats:
    """战斗段决策循环状态机（**纯逻辑**，假 policy + 假状态流可全程驱动）。

    每拍固定顺序：

    1. ``on_tick`` 钩（真机侧挂哨兵 ``poll``）与整链超时检查
       （闸门未开就超时 → ``battle_never_started``）；
    2. 读本拍 ``TickSample``（心跳被拒 → ``channel_dead``）；
    3. 无状态计数与超时判终（``no_state_timeout``）；
    4. **战斗开幕闸门（数值鲜活度，不依赖 patch 的 in_battle）**：
       连续 ``cfg.gate_change_ticks`` 拍签名变化 → 开闸；一直是同一组
       陈旧数值 = 还在编队/加载屏。开闸前不决策、不发动作；
    5. **自然结束判据（签名冻结）**：开闸后连续 ``cfg.freeze_ticks`` 拍
       签名不变 → 战斗结束（``out_of_battle``），去走结算；
    6. ``policy.decide`` 吃的是 :func:`synthesize_battle_state` 合成的完整
       状态（台账 hand/deployed + ``in_battle=True``）；出动作则下发并入
       台账，**下发被拒/回收超时不中止循环**（``actions_lost`` 计数）；
       动作上限兜底（``action_cap``）。

    判终分支为正交枚举，便于下游按原因记账；``states`` 不由本函数持有，
    调用方在生成 ``tick_samples`` 时就地落盘。
    """
    stats = LoopStats()
    t0 = clock()
    history: list[dict[str, Any]] = []
    no_state = 0
    prev_sig: tuple[Any, Any, Any] | None = None
    change_streak = 0
    frozen_streak = 0
    for sample in tick_samples:
        if on_tick is not None:
            on_tick()
        if clock() - t0 > cfg.chain_timeout_s:
            stats.terminal_cause = "timeout" if stats.battle_started else (
                "battle_never_started"
            )
            break
        if not sample.heartbeat_ok:
            stats.terminal_cause = "channel_dead"
            break
        stats.heartbeats += 1
        state = sample.state
        if state is None:
            no_state += 1
            stats.no_state_max_streak = max(stats.no_state_max_streak, no_state)
            if no_state >= cfg.no_state_timeout_ticks:
                stats.terminal_cause = "no_state_timeout"
                break
            continue
        no_state = 0
        stats.ticks += 1
        sig = _vitals_sig(state)
        if not stats.battle_started:
            # 上游明确报 in_battle=true（理论上 Output 链不会发生，防御性放行）
            # 或签名连续 N 拍变化 → 判定战斗已打响
            if state.get("in_battle"):
                stats.battle_started = True
            elif prev_sig is not None and sig != prev_sig:
                change_streak += 1
                if change_streak >= cfg.gate_change_ticks:
                    stats.battle_started = True
            else:
                change_streak = 0
            prev_sig = sig
            if not stats.battle_started:
                continue
        else:
            if sig == prev_sig:
                frozen_streak += 1
                if frozen_streak >= cfg.freeze_ticks:
                    stats.terminal_cause = "out_of_battle"
                    break
            else:
                frozen_streak = 0
            prev_sig = sig
        if stats.actions_sent + stats.actions_lost >= cfg.max_model_actions:
            stats.terminal_cause = "action_cap"
            break
        action = policy.decide(
            synthesize_battle_state(state, squad_names, history),
            list(history),
            t=stats.ticks,
        )
        if not action:
            continue
        if not channel.push_actions([action]):
            # 动作链被拒/回收超时（如 Deploy 等费自等待超窗口）：
            # 计入丢失、不入台账，循环继续等下一拍
            stats.actions_lost += 1
            continue
        stats.actions_sent += 1
        stats.model_actions.append(action)
        history.append(action)
    else:
        stats.terminal_cause = "stream_end"
    stats.idle_restarts = max(0, channel.opens - 1)
    return stats


# ---------------------------------------------------------------------------
# SingleStep 通道实现（运行侧）
# ---------------------------------------------------------------------------

_TERMINAL_MSGS = ("TaskChainCompleted", "TaskChainError", "TaskChainStopped")
"""链路终态事件（``AllTasksCompleted`` 另行单独处理——队列全空语义）。"""


class SingleStepParams:
    """SingleStep 参数形态的纯函数集（两路通道共用，与 pin 版源码对齐）。

    事实依据（``_src/…/Interface/SingleStepTask.cpp`` 与
    ``CopilotConfig.cpp``，勿凭记忆改）：

    - append 参数 ``{"type":"copilot","subtype":"stage",
      "details":{"stage_name": ...}}``——``Tile.find(stage_name)`` 查不到关卡时
      **append 直接返回 0**（早期 fail-fast 点）；
    - 追加动作 ``set_task_params(tid, {"type":"copilot","subtype":"action",
      "details":{"actions": [...]}})``；``CopilotConfig::parse_actions`` 读
      ``details.at("actions")``，**只能是包裹形态**，裸数组会被拒。
    """

    @staticmethod
    def stage_params(stage_code: str) -> dict[str, Any]:
        """SingleStep 开链参数（subtype=stage 包裹形态）。"""
        return {
            "type": "copilot",
            "subtype": "stage",
            "details": {"stage_name": stage_code},
        }

    @staticmethod
    def action_params(actions: list[dict[str, Any]]) -> dict[str, Any]:
        """追加动作参数（**details 必须 {"actions":[…]} 包裹**）。"""
        return {
            "type": "copilot",
            "subtype": "action",
            "details": {"actions": list(actions)},
        }


class MaaAppendPerActionChannel:
    """路径 B（**默认**）：一推一链——先参后启，TaskChainCompleted 天然回执。

    为什么默认它（2026-10-24 真机首跑 0/3 的铁证）：

    - 路径 A 的 stage-only 链（``append stage`` 后立即 ``start``）零 subtask、
      **同毫秒完成**（asst.log 10:45:45.565 TaskChainStart→Completed→
      AllTasksCompleted，其间无任何 SubTaskStart）；随后
      ``set_task_params(tid, Output)`` 落在已从 Assistant 工作集移除的已完成
      任务上被静默拒收——心跳动作从未执行、零 BattleState，两连拒 →
      ``channel_dead``（0-1/0-2 实跑全程 0 心跳 0 状态）。
    - 本路径把顺序焊死为 ``append(stage) → set_task_params(action) →
      start``：链起跑时动作子任务已在 ``m_subtasks`` 里，无竞态；链收尾的
      ``TaskChainCompleted`` 就是动作执行完成的天然回执（``wait_chain`` 收取，
      游标从 append 前起算，杜绝"快链在 wait 前完成被漏读"）。

    动作链自等待语义（上游 ``wait_condition`` 的既有能力）：

    - Deploy 动作的 ``wait_condition`` 会在自有循环里反复
      ``update_deployment`` 并等待目标干员 available+费用足够——所以模型下单
      不必精确等费；代价是动作链可能阻塞数十秒，故**动作链的回收上限
      （``action_timeout_s``）大于心跳链**，回收超时时链继续在后台执行，
      本通道如实返回 False 让循环记 actions_lost。
    - 心跳链（Output）不经过任何等待分支，秒级收尾。
    """

    def __init__(
        self,
        driver: MaaDriver,
        stage_code: str,
        heartbeat_timeout_s: float,
        action_timeout_s: float,
    ) -> None:
        """两条回收上限：心跳链秒级，动作链允许 Deploy 自等待数十秒。"""
        self._drv = driver
        self._stage = stage_code
        self._hb_timeout = heartbeat_timeout_s
        self._act_timeout = action_timeout_s
        self._tid = 0
        self.opens = 0

    def open_stage(self) -> int:
        """本通道下 open_stage 是幂等空操作（真正的开链发生在 push）。"""
        return 1

    def push_actions(self, actions: list[dict[str, Any]]) -> bool:
        is_heartbeat = actions == [HEARTBEAT_ACTION]
        timeout = self._hb_timeout if is_heartbeat else self._act_timeout
        mark = self._drv.mark()
        tid = self._drv.append_task(
            "SingleStep", SingleStepParams.stage_params(self._stage)
        )
        if tid <= 0:
            return False
        self._tid = tid
        self.opens += 1
        # 先参后启：链起跑时动作子任务已就位，不存在"零 subtask 速收"竞态
        if not self._drv.set_task_params(
            tid, SingleStepParams.action_params(actions)
        ):
            return False
        if not self._drv.start():
            # 上一条链可能未完全 idle（沿用 wait_idle 恢复语义）
            if not wait_idle(self._drv, timeout_s=_IDLE_RESTART_WAIT_S):
                return False
            if not self._drv.start():
                return False
        res = self._drv.wait_chain(timeout, mark=mark)
        return res.terminal_msg == "TaskChainCompleted"


class MaaSingleStepChannel:
    """路径 A（备用）：对同一个 ``SingleStep`` 任务原地 ``set_task_params`` 续喂。

    ⚠️ 2026-10-24 真机实测**会丢心跳**（stage-only 链同毫秒完成、set_params
    落空，证据见 :class:`MaaAppendPerActionChannel` docstring），仅留作与
    上游 SingleStep 原设计对照的调试路径——要让它真的工作，需要额外的
    保活子任务（如常驻 SkillDaemon）撑住 PackageTask，未经验证，勿默认使用。
    """

    def __init__(self, driver: MaaDriver, stage_code: str) -> None:
        """``driver`` 须已 connect 且战斗已经在打。"""
        self._drv = driver
        self._stage = stage_code
        self._tid = 0
        self._alive = False
        self._mark = driver.mark()
        self.opens = 0

    stage_params = SingleStepParams.stage_params
    action_params = SingleStepParams.action_params

    def _observe(self) -> None:
        """消费自有事件游标：观察到本链/全队列终态即标记链路死亡。"""
        fresh, self._mark = self._drv.events_since(self._mark)
        for evt in fresh:
            if evt.msg == "AllTasksCompleted":
                self._alive = False
                continue
            if evt.msg not in _TERMINAL_MSGS:
                continue
            d = evt.details if isinstance(evt.details, dict) else {}
            tid = d.get("taskid")
            if tid is None or str(tid) == str(self._tid):
                self._alive = False

    def open_stage(self) -> int:
        """链路活着就直接复用；死了（或不曾开）重新 append + start。"""
        self._observe()
        if self._alive and self._tid:
            return self._tid
        tid = self._drv.append_task(
            "SingleStep", SingleStepParams.stage_params(self._stage)
        )
        if tid <= 0:
            return 0
        self._tid = tid
        self._alive = True
        self.opens += 1
        if not self._drv.start():
            # 助手未空闲时 start 直接 False（沿用 wait_idle 的同款恢复）
            idle = wait_idle(self._drv, timeout_s=_IDLE_RESTART_WAIT_S)
            if not idle or not self._drv.start():
                self._alive = False
        return tid if self._alive else 0

    def push_actions(self, actions: list[dict[str, Any]]) -> bool:
        """喂一批动作；链死则先重开。``set_task_params`` 被拒再重开重试一次。"""
        if not self.open_stage():
            return False
        if self._drv.set_task_params(
            self._tid, SingleStepParams.action_params(actions)
        ):
            return True
        # 竞态：链在我们 observe 后刚收尾，task id 已失效 → 重开一次再试
        self._alive = False
        if not self.open_stage():
            return False
        return self._drv.set_task_params(
            self._tid, SingleStepParams.action_params(actions)
        )


def make_channel(
    mode: str,
    driver: MaaDriver,
    stage_code: str,
    heartbeat_timeout_s: float,
    action_timeout_s: float,
) -> SingleStepChannel:
    """按 ``channel_mode`` 造下发通道（默认 append_per_action）；未知模式 fail fast。"""
    if mode == "append_per_action":
        return MaaAppendPerActionChannel(
            driver, stage_code, heartbeat_timeout_s, action_timeout_s
        )
    if mode == "single_step_inplace":
        return MaaSingleStepChannel(driver, stage_code)
    raise ValueError(f"未知 channel_mode: {mode!r}")


# ---------------------------------------------------------------------------
# 真机节拍流与 core 策略装载（惰性 import，bridge venv 无 torch 也能测）
# ---------------------------------------------------------------------------


def maa_tick_stream(
    *,
    channel: SingleStepChannel,
    driver: MaaDriver,
    sentry: BlockerSentry,
    t0_wall_ms: int,
    job_tag: str,
    stage_code: str,
    heartbeat_s: float,
    on_event: Callable[[MaaEvent], None],
    on_state: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[TickSample]:
    """真机节拍流生成器：每拍 = 一次 Output 心跳 + 该拍窗口内最新 BattleState。

    - 先喂心跳动作，再在 ``heartbeat_s`` 窗口内轮询事件流：全部事件交给
      ``on_event`` 落盘/护栏，``what="BattleState"`` 的转成协议记录；
    - 窗口内的多帧状态只保最新一帧喂模型（旧帧仍会经 ``on_state`` 全部落盘）；
    - 哨兵 ``poll`` 在窗口循环里每轮过一遍（内部 1s 节流，不加 adb 压力）；
    - **无限生成**：收尾判据全在 :func:`battle_decision_loop` 一侧。
    """
    cursor = driver.mark()
    while True:
        hb_ok = channel.push_actions([dict(HEARTBEAT_ACTION)])
        deadline = time.monotonic() + heartbeat_s
        latest: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            sentry.poll()
            fresh, cursor = driver.events_since(cursor)
            for evt in fresh:
                on_event(evt)
                if evt.msg != "SubTaskExtraInfo":
                    continue
                d = evt.details if isinstance(evt.details, dict) else {}
                if d.get("what") != "BattleState":
                    continue
                details = d.get("details") if isinstance(d.get("details"), dict) else {}
                rec = battle_state_to_record(
                    details or {},
                    int(time.time() * 1000) - t0_wall_ms,
                    job_id=job_tag,
                    stage_code=stage_code,
                )
                latest = rec
                if on_state is not None:
                    on_state(rec)
            time.sleep(_EVENT_DRAIN_S)
        yield TickSample(state=latest, heartbeat_ok=hb_ok)


def load_core_policy(
    *,
    ckpt_path: str,
    npz_path: str | None,
    infer_cfg_path: str,
    stage: MilestoneStage,
    infer_overrides: dict[str, Any] | None = None,
) -> LoopPolicy:
    """装载 core track_a_v1 冒烟模型并包成 :class:`LoopPolicy`。

    **惰性 import**：torch/ark_core 只在本函数首次调用时加载——bridge venv
    （无 torch）import 本模块不受影响，单测保持纯逻辑。checkpoint/featvec 冻结
    矩阵/推理配置路径全部来自调用参数（配置或 CLI 覆盖），函数内不留默认值；
    ``infer_overrides`` 以 ``dataclasses.replace`` 粒度覆盖 InferConfig 字段
    （未知字段 fail fast）。
    """
    try:
        from ark_core.infer.track_a_v1 import (  # noqa: PLC0415
            InferConfig,
            StageContext,
            greedy_step,
            load_track_a_v1,
        )
    except ImportError as exc:
        raise RuntimeError(
            "无法导入 ark_core（本模块需从 core 的 uv 环境运行，并用 "
            "PYTHONPATH 注入 bridge/src；跑法见模块 docstring）"
        ) from exc

    cfg = InferConfig.from_yaml(infer_cfg_path)
    if infer_overrides:
        import dataclasses  # noqa: PLC0415  # 仅覆盖路径用

        known = {f.name for f in dataclasses.fields(cfg)}
        unknown = set(infer_overrides) - known
        if unknown:
            raise ValueError(f"infer_overrides 含 InferConfig 未知字段: {unknown}")
        cfg = dataclasses.replace(cfg, **infer_overrides)
    loaded = load_track_a_v1(ckpt_path, device=cfg.device)
    npz = Path(npz_path) if npz_path else None
    squad_names = [str(o["name"]) for o in stage.opers]
    ctx = StageContext.from_stage_id(stage.stage_id, squad_names, npz_path=npz)

    core_cfg = cfg
    total_proxy = stage.total_proxy

    def decide(
        battle_state: dict[str, Any],
        history: list[dict[str, Any]],
        *,
        t: int,
    ) -> dict[str, Any] | None:
        return greedy_step(
            loaded.model,
            ctx,
            battle_state,
            history,
            t=t,
            config=core_cfg,
            total_proxy=total_proxy,
        )

    class _Policy:
        """把包好的 decide 适配到 LoopPolicy 协议（闭包持有模型/ctx）。"""

        def __init__(self) -> None:
            self.decide = decide

    return _Policy()


# ---------------------------------------------------------------------------
# 闭环运行器（运行侧，结构与 replay_controller 同风格）
# ---------------------------------------------------------------------------


class LoopRunner:
    """模型闭环打关运行器：进图编队 → 心跳取状态 → 贪心推理 → SingleStep 下发。"""

    def __init__(
        self,
        instance_cfg_path: str | Path,
        nav_cfg_path: str | Path,
        work_root: str | Path,
        loop_config: LoopConfig,
    ) -> None:
        """装配驱动/导航/写出/看门狗线；防线与 replay_controller 完全同款。"""
        self._work = Path(work_root)
        (self._work / "tmp" / "maa_jobs").mkdir(parents=True, exist_ok=True)
        maa_user = self._work / "debug" / "maa_user"
        maa_user.mkdir(parents=True, exist_ok=True)
        inst = load_instance_config(instance_cfg_path)
        assert_patched_runtime(inst, instance_cfg_path)
        self._driver = MaaDriver(inst, maa_user)
        self._brief_check: dict | None = None
        try:
            from bridge.navigator import Navigator  # noqa: PLC0415  # 装配期才用

            nav_cfg = Navigator.load_nav_config(nav_cfg_path)
            self._brief_check = dict(nav_cfg["brief_blue_check"])
        except Exception as exc:  # noqa: BLE001
            print(
                f"[loop] ⚠ 终点屏校验配置不可用（{exc!r}），导航不做终点屏校验",
                flush=True,
            )
        self._writer = EpisodeWriter(self._work)
        self._cfg = loop_config

    # -- 小队壳：空动作 Copilot 作业（opers 保留，only 自动编队+开打） --

    def _write_shell_job(self, stage: MilestoneStage) -> Path:
        path = self._work / "tmp" / "maa_jobs" / f"loop_shell_{stage.stage_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "stage_name": stage.stage_code,
                    "opers": stage.opers,
                    "actions": [],
                    "doc": {
                        "title": f"model_loop 壳 {stage.stage_id}",
                        "details": (
                            "闭环运行器编队壳：actions 为空，仅自动编队+开打；"
                            f"参考作业 prts.plus #{stage.ref_job_id}"
                        ),
                    },
                    "minimum_required": "v4.0.0",
                },
                f,
                ensure_ascii=False,
                indent=1,
            )
        return path

    def _wait_shell(
        self, sentry: BlockerSentry, mark: int, rec: Callable[..., None]
    ) -> tuple[str | None, list[str]]:
        """等编队壳链收尾（期间哨兵轮询+事件落盘），返回 (终态, 失败摘要)。"""
        cursor = mark
        terminal: str | None = None
        failures: list[str] = []
        deadline = time.monotonic() + self._cfg.run.shell_timeout_s
        while self._driver.running() and terminal is None:
            sentry.poll()
            fresh, cursor = self._driver.events_since(cursor)
            for evt in fresh:
                rec(evt.msg, details=evt.details)
                if evt.msg == "SubTaskError":
                    d = evt.details if isinstance(evt.details, dict) else {}
                    raw_inner = d.get("details")
                    inner = raw_inner if isinstance(raw_inner, dict) else {}
                    cls = str(d.get("class", "")).rsplit("::", 1)[-1]
                    why = inner.get("why") or d.get("why") or ""
                    failures.append(f"{cls} why={why}" if why else cls)
                elif evt.msg in ("TaskChainCompleted", "TaskChainError"):
                    terminal = evt.msg
            if time.monotonic() > deadline:
                print("[loop] 编队壳链超时，stop", flush=True)
                self._driver.stop()
                time.sleep(2.0)
                terminal = terminal or "Timeout"
                break
            time.sleep(0.2)
        fresh, _ = self._driver.events_since(cursor)
        for evt in fresh:
            rec(evt.msg, details=evt.details)
            if evt.msg in ("TaskChainCompleted", "TaskChainError"):
                terminal = terminal or evt.msg
        return terminal, failures

    # -- 一关的完整闭环 --

    def _run_stage(self, stage: MilestoneStage, policy: LoopPolicy) -> dict[str, Any]:
        """跑一关闭环；返回结果 JSON 摘要（battle_result/动作数/节拍数/拦截次数）。"""
        cfg = self._cfg.run
        tag = f"[loop {stage.stage_id}]"
        print(
            f"{tag} 开始（参考作业 #{stage.ref_job_id}，"
            f"预期动作≈{stage.total_proxy}）",
            flush=True,
        )
        summary: dict[str, Any] = {
            "stage_id": stage.stage_id,
            "stage_code": stage.stage_code,
            "ref_job_id": stage.ref_job_id,
        }
        t0 = time.time()
        t0_wall = int(t0 * 1000)
        records: list[dict[str, Any]] = []
        states: list[dict[str, Any]] = []
        sentry = BlockerSentry(self._driver, t0_wall)

        def rec(msg: str, **kw: Any) -> None:
            t_ms = int(time.time() * 1000) - t0_wall
            records.append({"t_ms": t_ms, "msg": msg, **kw})

        # 进图：回主页 → nav_only 到 briefing（原样复用 replay_controller 两段式
        # 与 retry 逻辑；patch 0002 + 终点屏校验防线同工厂线）
        if not ensure_home(
            self._driver,
            sentry,
            log_prefix=tag,
            attempts=cfg.pre_retry_attempts,
            settle_timeout_s=cfg.settle_timeout_s,
            idle_wait_s=cfg.pre_retry_idle_wait_s,
        ):
            summary.update(ok=False, error="pre: StartUp 失败（含重试）")
            return summary
        try:
            nav_ok = navigate_only(
                self._driver,
                stage.stage_code,
                watchdog=sentry,
                endpoint_cfg=self._brief_check,
            )
        except NavigationError as exc:
            summary.update(ok=False, error=f"nav: {exc}")
            return summary
        if not nav_ok:
            summary.update(ok=False, error="nav: 未到准备界面(endpoint_check_failed)")
            return summary

        run_dir = self._writer.begin(stage.stage_id, f"model-{stage.ref_job_id}")
        job_file = self._write_shell_job(stage)

        def on_shot(tag_s: str, rel: str) -> None:
            if tag_s == "tick":
                rec("ShotTick", file=rel)
            else:
                rec("Shot", tag=tag_s, file=rel)

        logger = StateLogger(self._driver, run_dir, on_shot=on_shot)
        logger.start()
        logger.snap("briefing")

        # 编队壳：actions 空数组的 Copilot（自动编队+开打，链秒收尾、战斗自由运行）
        shell_tid = self._driver.append_copilot(
            job_file, formation=True, support_unit_usage=cfg.support_unit_usage
        )
        if shell_tid <= 0:
            logger.stop()
            summary.update(ok=False, error=f"shell append tid={shell_tid}")
            (run_dir / "error.txt").write_text(
                f"shell append tid={shell_tid}", encoding="utf-8"
            )
            return summary
        mark = self._driver.mark()
        shell_started = self._driver.start() or (
            wait_idle(self._driver, timeout_s=_IDLE_RESTART_WAIT_S)
            and self._driver.start()
        )
        if not shell_started:
            logger.stop()
            summary.update(ok=False, error="asst.start 失败（编队壳）")
            return summary
        logger.set_active(True)
        shell_terminal, shell_failures = self._wait_shell(sentry, mark, rec)
        print(f"{tag} 编队壳收尾={shell_terminal} failures={shell_failures}",
              flush=True)
        wait_idle(self._driver, timeout_s=_IDLE_RESTART_WAIT_S)
        if shell_terminal != "TaskChainCompleted":
            logger.stop()
            self._writer.write_events(run_dir, records)
            summary.update(
                ok=False,
                episode=str(run_dir),
                error=(
                    f"shell 链失败 terminal={shell_terminal}; "
                    + ("; ".join(shell_failures[-3:]) or "无 SubTaskError 记录")
                ),
            )
            return summary

        # 战斗闭环段（判负证据窗口从战斗起手算，口径同 replay_controller）
        battle_mark = sentry.count()
        sentry.set_battle_active(True)
        # 心跳链回收上限秒级（两拍冗余）；动作链放宽到 Deploy 自等待窗口
        # （等费/等 CD 由上游 wait_condition 处理，上限内仍算链未回收）
        hb_timeout_s = max(2.0 * cfg.heartbeat_s, 15.0)
        act_timeout_s = max(20.0 * cfg.heartbeat_s, 90.0)
        channel = make_channel(
            cfg.channel_mode, self._driver, stage.stage_code,
            hb_timeout_s, act_timeout_s,
        )
        if cfg.channel_mode == "single_step_inplace" and not channel.open_stage():
            sentry.set_battle_active(False)
            logger.stop()
            summary.update(ok=False, error="SingleStep 开链被拒绝（stage_name?）")
            return summary

        prts_seen: list[str] = []

        def on_event(evt: MaaEvent) -> None:
            rec(evt.msg, details=evt.details)
            d = evt.details if isinstance(evt.details, dict) else {}
            det = d.get("details") if isinstance(d.get("details"), dict) else {}
            t = str(d.get("task") or det.get("task") or d.get("cur_task") or "")
            if t:
                tail = t.rsplit("@", 1)[-1]
                if tail in PRTS_TASKS and tail not in prts_seen:
                    prts_seen.append(tail)

        samples = maa_tick_stream(
            channel=channel,
            driver=self._driver,
            sentry=sentry,
            t0_wall_ms=t0_wall,
            job_tag=f"model-{stage.ref_job_id}",
            stage_code=stage.stage_code,
            heartbeat_s=cfg.heartbeat_s,
            on_event=on_event,
            on_state=states.append,
        )
        recorded = RecordingPolicy(
            policy, lambda act, tick: rec("LoopAction", action=act, tick=tick)
        )
        stats = battle_decision_loop(
            channel=channel,
            policy=recorded,
            tick_samples=samples,
            cfg=cfg,
            squad_names=[str(o["name"]) for o in stage.opers],
            on_tick=sentry.poll,
        )
        print(
            f"{tag} 闭环收尾 cause={stats.terminal_cause} 采状态拍={stats.ticks} "
            f"心跳={stats.heartbeats} 模型动作={stats.actions_sent} "
            f"动作丢失={stats.actions_lost} "
            f"idle重开={stats.idle_restarts} "
            f"最长无状态连拍={stats.no_state_max_streak}",
            flush=True,
        )
        sentry.set_battle_active(False)
        # 战斗大概率已自然结束；若仍在打（timeout/token 等），停链再推进结算
        self._driver.stop()
        wait_idle(self._driver, timeout_s=_IDLE_RESTART_WAIT_S)

        # 结算推进 + 判负证据（原样复用 replay_controller 尾段）
        logger.set_active(False)
        rel = logger.snap("settle")
        result_files = [rel] if not is_dropped_shot(rel) else []
        settled = wait_settle(
            self._driver, timeout_s=cfg.settle_timeout_s, watchdog=sentry
        )
        lost_hits = sentry.hits_since(battle_mark, BLOCKER_SETTLE_LOST)
        logger.stop()

        self._writer.write_events(run_dir, records)
        states_path = self._writer.write_battle_states(run_dir, states)
        state_summary = summarize_states(states)
        battle_result = resolve_battle_result(
            lost_confirmed=bool(lost_hits),
            chain_completed=(stats.terminal_cause == "out_of_battle"),
            settled=settled,
        )
        # fail_stage 口径（本运行器）：闭环观测到战斗自然结束 → done；否则战斗段
        # 失败（超时/断流/通道拒收）。编队壳失败在上方提前返回、不产生 episode，
        # 与 replay_controller 的 nav 提前返回同款口径。
        fail_stage = "done" if stats.terminal_cause == "out_of_battle" else "battle"
        ep_ts = time.strftime("%Y%m%d-%H%M%S")
        episode = {
            "episode_id": f"model-loop-{stage.ref_job_id}-{ep_ts}",
            "stage_id": stage.stage_id,
            "stage_code": stage.stage_code,
            "source": {
                "type": "model_loop",
                "ref_job_id": stage.ref_job_id,
                "note": stage.note,
            },
            "job_file": str(job_file),
            "result_screen_files": [
                result_files[0] if result_files else "",
                result_files[-1] if result_files else "",
            ],
            "action_events_file": "action_events.jsonl",
            "blockers_dismissed": sentry.events(),
            "battle_result": battle_result,
            "fail_stage": fail_stage,
            "battle_states_file": states_path.name if states_path else None,
            "battle_states": state_summary,
            "prts_detected": prts_seen,
            # 模型闭环附加段：实际下发动作序列 + 闭环统计（结果 JSON 同口径）
            "model_actions": stats.model_actions,
            "model_loop": {
                "terminal_cause": stats.terminal_cause,
                "ticks": stats.ticks,
                "heartbeats": stats.heartbeats,
                "actions_sent": stats.actions_sent,
                "idle_restarts": stats.idle_restarts,
                "no_state_max_streak": stats.no_state_max_streak,
                "battle_started": stats.battle_started,
                "actions_lost": stats.actions_lost,
                "total_proxy": stage.total_proxy,
            },
            "shot_files": None,
            "duration_ms": int((time.time() - t0) * 1000),
            "copilot_actions": 0,
            "settled_back_to_briefing": settled,
            "shots_dropped_black": logger.shots_dropped_black,
        }
        ep_path = self._writer.write_episode(run_dir, episode)
        summary.update(
            ok=True,
            episode=str(ep_path),
            battle_result=battle_result,
            fail_stage=fail_stage,
            terminal_cause=stats.terminal_cause,
            actions_sent=stats.actions_sent,
            ticks=stats.ticks,
            heartbeats=stats.heartbeats,
            idle_restarts=stats.idle_restarts,
            battle_started=stats.battle_started,
            actions_lost=stats.actions_lost,
            blockers_dismissed=len(sentry.events()),
            settled=settled,
        )
        print(f"{tag} episode -> {ep_path} battle_result={battle_result}", flush=True)
        return summary

    def run(
        self,
        *,
        policy_for: Callable[[MilestoneStage], LoopPolicy],
        only: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """按里程碑列表逐关跑闭环；``policy_for`` 按关构造策略（调方装载模型）。"""
        stages = self._cfg.stages
        if only:
            stages = [s for s in stages if s.stage_id in only or s.stage_code in only]
        if not stages:
            print("[loop] 无待跑关卡")
            return []
        if not self._driver.connect():
            raise RuntimeError("MAA 连接失败")
        summaries = []
        try:
            for stage in stages:
                summaries.append(self._run_stage(stage, policy_for(stage)))
        finally:
            self._driver.close()
        out = self._work / "debug" / f"loop_summary_{int(time.time())}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=1)
        print(f"[loop] 完成 {len(summaries)} 关 -> {out}")
        win = sum(1 for s in summaries if s.get("battle_result") == "win")
        print(f"[loop] win {win}/{len(summaries)}（口径勿夸大：链路冒烟不是能力证据）")
        return summaries


def _bridge_repo_root() -> Path:
    """bridge 仓根（模块位于 bridge/src/bridge/loop_runner.py）。"""
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    """CLI：从 core 环境运行，PYTHONPATH 注入 bridge/src（跑法见模块 docstring）。

    ``-m bridge.loop_runner`` 的包解析发生在本模块 main 之前，故注入只能经
    环境变量完成（不存在可工作的 ``--bridge-src``，踩过的坑如实说明）。
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    root = _bridge_repo_root()
    parser = argparse.ArgumentParser(description="模型闭环打关运行器（v1 冒烟链路）")
    parser.add_argument(
        "--config",
        default=str(root / "configs" / "loop_milestone_v1.json"),
        help="里程碑配置",
    )
    parser.add_argument(
        "--instance",
        default=str(root / "configs" / "instances" / "mumu_local_patched.yaml"),
        help="实例定义（必须 patched 运行时）",
    )
    parser.add_argument(
        "--nav", default=str(root / "configs" / "nav_main.yaml"), help="导航点位表"
    )
    parser.add_argument("--work", required=True, help="数据落盘根（本地目录）")
    parser.add_argument(
        "--only", default=None, help="只跑指定关卡（stage_id/code，逗号分隔）"
    )
    parser.add_argument("--ckpt", default=None, help="覆盖配置里的 ckpt 路径")
    parser.add_argument("--npz", default=None, help="覆盖配置里的 featvec npz 路径")
    parser.add_argument(
        "--infer-cfg", default=None, help="覆盖配置里的 InferConfig yaml"
    )
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="replace")

    loop_cfg = load_loop_config(args.config)
    ckpt = args.ckpt or loop_cfg.ckpt_path
    infer_cfg = args.infer_cfg or loop_cfg.infer_cfg_path
    npz = args.npz or loop_cfg.npz_path
    if not ckpt or not infer_cfg:
        raise SystemExit("ckpt/infer_cfg 必须用配置 paths 或 CLI 提供（无内置默认值）")

    runner = LoopRunner(args.instance, args.nav, args.work, loop_cfg)

    def policy_for(stage: MilestoneStage) -> LoopPolicy:
        return load_core_policy(
            ckpt_path=ckpt,
            npz_path=npz,
            infer_cfg_path=infer_cfg,
            stage=stage,
            infer_overrides=loop_cfg.infer_overrides,
        )

    only = set(args.only.split(",")) if args.only else None
    summaries = runner.run(policy_for=policy_for, only=only)
    ok = sum(1 for s in summaries if s.get("ok"))
    print(f"[loop] 流程成功 {ok}/{len(summaries)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
