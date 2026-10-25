"""loop_runner 纯逻辑单测（不连 MAA、不依赖 torch）。

覆盖四类（对应交付约定）：

1. SingleStep params 序列化/包裹形态（pin 版 ``CopilotConfig::parse_actions``
   要求 details = ``{"actions":[…]}``；裸数组会被拒）；
2. 决策循环状态机（假 policy + 假状态流驱动全部判终分支）；
3. 判终分支（out_of_battle / no_state_timeout / timeout / channel_dead /
   action_cap / stream_end）逐项断言；
4. idle 恢复逻辑（假 driver + 脚本化事件流：链收尾后重开、set_task_params
   竞态被拒后重开重试）。
"""

from __future__ import annotations

import collections
import itertools
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from bridge.loop_runner import (
    HEARTBEAT_ACTION,
    LoopRunConfig,
    MaaAppendPerActionChannel,
    MaaSingleStepChannel,
    SingleStepParams,
    TickSample,
    battle_decision_loop,
    load_loop_config,
    make_channel,
)
from bridge.loop_runner import (
    synthesize_battle_state as lr_synth,
)
from bridge.maa_driver import ChainResult, MaaEvent


def _cfg(**over: Any) -> LoopRunConfig:
    """测试用配置（默认阈值都压在个位数，节拍参数无关真假机）。"""
    base: dict[str, Any] = {
        "heartbeat_s": 1.5,
        "chain_timeout_s": 600.0,
        "shell_timeout_s": 180.0,
        "settle_timeout_s": 300.0,
        "pre_retry_attempts": 2,
        "pre_retry_idle_wait_s": 20.0,
        "no_state_timeout_ticks": 3,
        "max_model_actions": 64,
        "support_unit_usage": 0,
    }
    base.update(over)
    return LoopRunConfig(**base)


def _state(
    in_battle: bool | None = None,
    costs: Any = 30,
    kills: Any = 0,
    total_kills: Any = 11,
    **kw: Any,
) -> dict[str, Any]:
    """假状态帧：默认 in_battle 缺省（模拟 Output 心跳链上的陈旧字段）。

    sig = (costs, kills, total_kills) 由参数控制，供开闸/冻结测试驱动。
    """
    st: dict[str, Any] = {"costs": costs, "kills": kills, "total_kills": total_kills}
    if in_battle is not None:
        st["in_battle"] = in_battle
    st.update(kw)
    return st


_DEPLOY = {"type": "Deploy", "name": "芬", "location": [3, 2], "direction": "Right"}

_SQUAD = ["芬", "阿米娅"]


def _loop(
    ch: Any,
    policy: Any,
    samples: Iterable[Any],
    cfg: LoopRunConfig | None = None,
    clock: Any = None,
) -> Any:
    kw: dict[str, Any] = {
        "channel": ch,
        "policy": policy,
        "tick_samples": samples,
        "cfg": cfg or _cfg(),
        "squad_names": list(_SQUAD),
    }
    if clock is not None:
        kw["clock"] = clock
    return battle_decision_loop(**kw)


class _FakePolicy:
    """脚本化 policy：每次 decide 弹一个脚本动作，记 (t, history 副本, state)。"""

    def __init__(self, script: list[dict[str, Any] | None]) -> None:
        self._script = collections.deque(script)
        self.calls: list[tuple[int, list[dict[str, Any]]]] = []
        self.states: list[dict[str, Any]] = []
        self.mutate_history = False

    def decide(
        self,
        battle_state: dict[str, Any],
        history: list[dict[str, Any]],
        *,
        t: int,
    ) -> dict[str, Any] | None:
        self.calls.append((t, list(history)))
        self.states.append(dict(battle_state))
        if self.mutate_history:
            history.append({"type": "Skill", "name": "测试污染"})
        return self._script.popleft() if self._script else None


class _FakeChannel:
    """假下发通道：记全部 push；``fail_at`` 里的第 n 次 push 返回 False。"""

    def __init__(self, fail_at: tuple[int, ...] = (), opens: int = 1) -> None:
        self.opens = opens
        self.pushes: list[list[dict[str, Any]]] = []
        self._fail_at = set(fail_at)
        self._n = 0

    def open_stage(self) -> int:
        return 1

    def push_actions(self, actions: list[dict[str, Any]]) -> bool:
        self._n += 1
        if self._n in self._fail_at:
            return False
        self.pushes.append(list(actions))
        return True


class _FakeDriver:
    """MaaDriver 最小假实现：事件日志 + append/set_params/start/wait_chain 脚本化。"""

    def __init__(self) -> None:
        self._events: list[MaaEvent] = []
        self.append_calls: list[tuple[str, dict[str, Any]]] = []
        self.set_calls: list[tuple[int, dict[str, Any]]] = []
        self.start_calls = 0
        self.call_order: list[str] = []
        self.wait_chain_calls: list[int | None] = []
        self._running = False
        self._tid = 0
        self.set_results: collections.deque[bool] = collections.deque()
        self.chain_results: collections.deque = collections.deque()

    def mark(self) -> int:
        return len(self._events)

    def events_since(self, mark: int) -> tuple[list[MaaEvent], int]:
        return list(self._events[mark:]), len(self._events)

    def post(self, msg: str, details: dict[str, Any] | None = None) -> None:
        self._events.append(MaaEvent(0, 0, msg, details or {}))

    def append_task(self, task_type: str, params: dict[str, Any]) -> int:
        self.call_order.append("append_task")
        self.append_calls.append((task_type, params))
        self._tid += 1
        self._running = True
        return self._tid

    def set_task_params(self, task_id: int, params: dict[str, Any]) -> bool:
        self.call_order.append("set_task_params")
        self.set_calls.append((task_id, params))
        return self.set_results.popleft() if self.set_results else True

    def start(self) -> bool:
        self.call_order.append("start")
        self.start_calls += 1
        return True

    def wait_chain(self, timeout_s: float, mark: int | None = None) -> ChainResult:
        self.call_order.append("wait_chain")
        self.wait_chain_calls.append(mark)
        if self.chain_results:
            return self.chain_results.popleft()
        return ChainResult(True, "TaskChainCompleted")

    def running(self) -> bool:
        return self._running


# ---------------------------------------------------------------------------
# 1. SingleStep params 序列化 / 包裹形态
# ---------------------------------------------------------------------------


def test_stage_params_form() -> None:
    expected = {
        "type": "copilot",
        "subtype": "stage",
        "details": {"stage_name": "0-1"},
    }
    assert SingleStepParams.stage_params("0-1") == expected
    # 备用路径 A 的参数形态与共享实现一致
    assert MaaSingleStepChannel.stage_params("0-1") == expected


def test_action_params_wrapped_form() -> None:
    """details 必须 {"actions":[…]} 包裹（pin 版 parse_actions 读 .at("actions")）。"""
    params = SingleStepParams.action_params([_DEPLOY])
    assert params == {
        "type": "copilot",
        "subtype": "action",
        "details": {"actions": [_DEPLOY]},
    }
    # 传入列表不被外部后续改动污染（通道持有副本）
    src = [dict(_DEPLOY)]
    out = MaaSingleStepChannel.action_params(src)
    src.append({"type": "Skill", "name": "阿米娅"})
    assert out["details"]["actions"] == [_DEPLOY]


def test_heartbeat_action_is_output_noop() -> None:
    assert HEARTBEAT_ACTION == {"type": "Output"}


def test_make_channel_unknown_mode_fails_fast() -> None:
    drv = _FakeDriver()
    with pytest.raises(ValueError, match="channel_mode"):
        make_channel("bogus", drv, "0-1", 600.0, 600.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. 决策循环状态机（假 policy + 假状态流；判终 = 数值签名鲜活度/冻结）
# ---------------------------------------------------------------------------


def test_loop_happy_path_records_history_and_actions() -> None:
    ch = _FakeChannel()
    policy = _FakePolicy([_DEPLOY, None])
    samples = [
        TickSample(_state(in_battle=True, costs=10)),  # 上游明确报战斗中 → 直接开闸
        TickSample(_state(costs=11)),                  # 鲜活（签名变化，冻结计数清零）
        TickSample(_state(costs=11)),
        TickSample(_state(costs=11)),                  # freeze_ticks=2 → 自然结束
    ]
    stats = _loop(ch, policy, samples, _cfg(freeze_ticks=2))
    assert stats.terminal_cause == "out_of_battle"
    assert stats.ticks == 4
    assert stats.heartbeats == 4
    assert stats.actions_sent == 1
    assert stats.model_actions == [_DEPLOY]
    assert ch.pushes == [[_DEPLOY]]
    # 冻结计数期间仍继续决策（直到冻结阈值触发）；history 逐拍累积
    assert policy.calls == [(1, []), (2, [_DEPLOY]), (3, [_DEPLOY])]


def test_loop_policy_sees_synthesized_state() -> None:
    """policy 吃进的是台账合成状态：in_battle=True + 台账 hand/deployed。"""
    ch = _FakeChannel()
    policy = _FakePolicy([_DEPLOY, None])
    samples = [
        TickSample(_state(in_battle=True, hand=[], deployed=[])),
        TickSample(_state(costs=31)),
        TickSample(_state(costs=31)),
        TickSample(_state(costs=31)),
    ]
    _loop(ch, policy, samples, _cfg(freeze_ticks=2, gate_change_ticks=1))
    s1, s2 = policy.states[0], policy.states[1]
    assert s1["in_battle"] is True
    assert s1["synthesized"] is True
    assert [(h["name"], h["available"]) for h in s1["hand"]] == [
        ("芬", True),
        ("阿米娅", True),
    ]
    # 第二拍：芬 已在台账 deployed，hand 只剩阿米娅
    assert [d["name"] for d in s2["deployed"]] == ["芬"]
    assert [h["name"] for h in s2["hand"]] == ["阿米娅"]


def test_loop_policy_cannot_corrupt_internal_history() -> None:
    ch = _FakeChannel()
    policy = _FakePolicy([_DEPLOY, None])
    policy.mutate_history = True
    samples = [TickSample(_state(in_battle=True)), TickSample(_state(costs=31))]
    stats = _loop(ch, policy, samples)
    assert stats.model_actions == [_DEPLOY]
    assert all(e["name"] != "测试污染" for e in stats.model_actions)


def test_loop_no_state_timeout() -> None:
    ch = _FakeChannel()
    policy = _FakePolicy([_DEPLOY])
    samples = [TickSample(None) for _ in range(5)]
    cfg = _cfg(no_state_timeout_ticks=3)
    stats = _loop(ch, policy, samples, cfg)
    assert stats.terminal_cause == "no_state_timeout"
    assert stats.heartbeats == 3
    assert stats.no_state_max_streak == 3
    assert stats.ticks == 0
    assert not policy.calls


def test_loop_no_state_streak_resets_on_state() -> None:
    ch = _FakeChannel()
    policy = _FakePolicy([])
    samples = [
        TickSample(None),
        TickSample(None),
        TickSample(_state(in_battle=True)),
        TickSample(None),
        TickSample(None),
        TickSample(_state(costs=9)),
    ]
    cfg = _cfg(no_state_timeout_ticks=3)
    stats = _loop(ch, policy, samples, cfg)
    assert stats.terminal_cause == "stream_end"
    assert stats.no_state_max_streak == 2


def test_loop_chain_timeout_via_clock() -> None:
    ch = _FakeChannel()
    policy = _FakePolicy([])
    clock = itertools.count(0, 5).__next__  # 每次调用 +5s（t0=0 由循环内部消费）
    stats = _loop(
        ch, policy, itertools.repeat(TickSample(_state(in_battle=True))),
        _cfg(chain_timeout_s=20.0), clock=clock,
    )
    assert stats.terminal_cause == "timeout"
    # t0=0 之后时钟 5/10/15/20 各处理一拍，25 越界收尾：共 4 拍
    assert stats.heartbeats == 4


def test_loop_channel_dead_on_heartbeat_reject() -> None:
    ch = _FakeChannel()
    stats = _loop(
        ch, _FakePolicy([]),
        [TickSample(_state(in_battle=True), heartbeat_ok=False)],
    )
    assert stats.terminal_cause == "channel_dead"
    assert stats.heartbeats == 0
    assert stats.actions_sent == 0


def test_loop_action_reject_is_lost_not_dead() -> None:
    """动作下发被拒/回收超时：记 actions_lost 继续循环，不入台账不中止。"""
    ch = _FakeChannel(fail_at=(1,))
    policy = _FakePolicy([_DEPLOY, None])
    samples = [
        TickSample(_state(in_battle=True, costs=10)),
        TickSample(_state(costs=11)),
    ]
    stats = _loop(ch, policy, samples)
    assert stats.terminal_cause == "stream_end"
    assert stats.actions_sent == 0
    assert stats.actions_lost == 1
    assert stats.model_actions == []


def test_loop_action_cap() -> None:
    ch = _FakeChannel()
    policy = _FakePolicy([_DEPLOY, dict(_DEPLOY), dict(_DEPLOY)])
    samples = [TickSample(_state(in_battle=True)) for _ in range(4)]
    stats = _loop(ch, policy, samples, _cfg(max_model_actions=2))
    assert stats.terminal_cause == "action_cap"
    assert stats.actions_sent == 2
    assert len(ch.pushes) == 2


def test_loop_stream_end_when_samples_exhausted() -> None:
    ch = _FakeChannel()
    stats = _loop(ch, _FakePolicy([]), [TickSample(_state(in_battle=True))])
    assert stats.terminal_cause == "stream_end"


def test_loop_idle_restart_count_from_channel_opens() -> None:
    ch = _FakeChannel(opens=3)
    stats = _loop(ch, _FakePolicy([]), [TickSample(_state(in_battle=True))])
    assert stats.idle_restarts == 2


def test_loop_on_tick_hook_runs_each_sample() -> None:
    ch = _FakeChannel()
    seen: list[int] = []
    samples = [
        TickSample(_state(in_battle=True)),
        TickSample(None), TickSample(None), TickSample(None), TickSample(None),
    ]
    battle_decision_loop(
        channel=ch,
        policy=_FakePolicy([]),
        tick_samples=samples,
        cfg=_cfg(no_state_timeout_ticks=3),
        squad_names=list(_SQUAD),
        on_tick=lambda: seen.append(1),
    )
    # 第 4 拍触发 no_state_timeout（阈值 3 已在第 3 个 None 满足）→ 4 次钩
    assert len(seen) == 4


# ---------------------------------------------------------------------------
# 2b. 战斗开幕闸门：数值签名鲜活度（patch 的 in_battle/hand 在 Output 链上恒脏）
# ---------------------------------------------------------------------------


def test_gate_waits_for_vitals_change_before_deciding() -> None:
    """签名连续 2 拍变化才开闸；开闸前不决策、开闸后冻结 N 拍判结束。"""
    ch = _FakeChannel()
    policy = _FakePolicy([_DEPLOY, None])
    samples = [
        TickSample(_state(costs=10, kills=0)),  # 陈旧（编队/加载屏）
        TickSample(_state(costs=10, kills=0)),  # 不变 → streak 0
        TickSample(_state(costs=11, kills=0)),  # 变化 → streak 1
        TickSample(_state(costs=13, kills=0)),  # 变化 → streak 2 → 开闸
        TickSample(_state(costs=13, kills=0)),  # 冻结 1（仍决策）
        TickSample(_state(costs=13, kills=0)),  # 冻结 2 → 自然结束
    ]
    stats = _loop(ch, policy, samples, _cfg(gate_change_ticks=2, freeze_ticks=2))
    assert stats.terminal_cause == "out_of_battle"
    assert stats.battle_started is True
    assert stats.ticks == 6  # 等待拍也计数
    # policy 只在开闸后的第 4、5 拍被调用
    assert [c[0] for c in policy.calls] == [4, 5]


def test_gate_change_breaks_frozen_then_reends() -> None:
    """冻结计数被打断后重新累计（击杀步进/费用回升视为鲜活）。"""
    ch = _FakeChannel()
    policy = _FakePolicy([None, None, None, None])
    samples = [
        TickSample(_state(costs=90, kills=5)),   # streak 0
        TickSample(_state(costs=91, kills=5)),   # 变 → streak1
        TickSample(_state(costs=92, kills=5)),   # 变 → 开闸
        TickSample(_state(costs=92, kills=5)),   # 冻结 1
        TickSample(_state(costs=92, kills=6)),   # 击杀步进 → 冻结清零
        TickSample(_state(costs=92, kills=6)),   # 冻结 1
        TickSample(_state(costs=92, kills=6)),   # 冻结 2 → 结束
    ]
    stats = _loop(ch, policy, samples, _cfg(gate_change_ticks=2, freeze_ticks=2))
    assert stats.terminal_cause == "out_of_battle"
    assert stats.ticks == 7


def test_gate_battle_never_started_on_timeout() -> None:
    """签名恒冻结 + 整链超时：记 battle_never_started 且从不调用 policy。"""
    ch = _FakeChannel()
    policy = _FakePolicy([_DEPLOY])
    clock = itertools.count(0, 10).__next__
    stats = _loop(
        ch, policy,
        itertools.repeat(TickSample(_state(costs=14, kills=0, total_kills=23))),
        _cfg(no_state_timeout_ticks=99, chain_timeout_s=30.0, gate_change_ticks=2),
        clock=clock,
    )
    assert stats.terminal_cause == "battle_never_started"
    assert stats.battle_started is False
    assert stats.actions_sent == 0
    assert not policy.calls


def test_gate_opens_immediately_on_in_battle_true() -> None:
    """上游明确报 in_battle=true（正常 Copilot 路径才有）→ 立即开闸（防御性）。"""
    ch = _FakeChannel()
    policy = _FakePolicy([_DEPLOY])
    samples = [TickSample(_state(in_battle=True))]
    stats = _loop(ch, policy, samples)
    assert stats.battle_started is True
    assert policy.calls == [(1, [])]


def test_loop_timeout_after_battle_started_still_timeout() -> None:
    """开闸后再超时：维持 timeout 语义（区别于 battle_never_started）。"""
    ch = _FakeChannel()
    clock = itertools.count(0, 10).__next__
    stats = _loop(
        ch, _FakePolicy([]),
        itertools.repeat(TickSample(_state(in_battle=True, costs=99))),
        _cfg(chain_timeout_s=30.0),
        clock=clock,
    )
    assert stats.terminal_cause == "timeout"
    assert stats.battle_started is True


# ---------------------------------------------------------------------------
# 2c. 台账合成状态（synthesize_battle_state 纯函数）
# ---------------------------------------------------------------------------


def test_shadow_hand_excludes_deployed() -> None:
    st = lr_synth(
        {"costs": 25, "kills": 2},
        ["芬", "阿米娅"],
        [{"type": "Deploy", "name": "芬", "location": [3, 2]}],
    )
    assert st["costs"] == 25 and st["kills"] == 2
    assert st["in_battle"] is True
    assert [h["name"] for h in st["hand"]] == ["阿米娅"]
    assert st["hand"][0]["available"] is True
    assert st["hand"][0]["cooling"] is False
    assert [d["name"] for d in st["deployed"]] == ["芬"]


def test_shadow_retreat_marks_cooling() -> None:
    st = lr_synth(
        {},
        ["芬"],
        [{"type": "Deploy", "name": "芬"}, {"type": "Retreat", "name": "芬"}],
    )
    assert st["deployed"] == []
    assert st["hand"] == [{"name": "芬", "available": True, "cooling": True}]


def test_shadow_ignores_actions_outside_squad() -> None:
    """非编队干员的动作（如未来借兵）不影响台账推导。"""
    st = lr_synth(
        {}, ["芬"], [{"type": "Deploy", "name": "阿米娅", "location": [1, 1]}]
    )
    assert [h["name"] for h in st["hand"]] == ["芬"]
    assert st["deployed"] == []


# ---------------------------------------------------------------------------
# 3/4. 判终与 idle 恢复（假 driver + 脚本化事件的通道层）
# ---------------------------------------------------------------------------


def test_channel_open_appends_wrapped_stage_params() -> None:
    drv = _FakeDriver()
    ch = MaaSingleStepChannel(drv, "0-1")  # type: ignore[arg-type]
    assert ch.open_stage() == 1
    assert ch.opens == 1
    task_type, params = drv.append_calls[0]
    assert task_type == "SingleStep"
    assert params == MaaSingleStepChannel.stage_params("0-1")
    assert drv.start_calls == 1


def test_channel_push_uses_set_task_params_wrapped() -> None:
    drv = _FakeDriver()
    ch = MaaSingleStepChannel(drv, "0-1")  # type: ignore[arg-type]
    assert ch.push_actions([_DEPLOY]) is True
    tid, params = drv.set_calls[-1]
    assert tid == 1
    assert params == MaaSingleStepChannel.action_params([_DEPLOY])


def test_channel_idle_recovery_reopens_after_chain_completed() -> None:
    """链收尾（TaskChainCompleted + AllTasksCompleted）后下一次 push 必须重开。"""
    drv = _FakeDriver()
    ch = MaaSingleStepChannel(drv, "0-1")  # type: ignore[arg-type]
    assert ch.push_actions([dict(HEARTBEAT_ACTION)]) is True
    assert ch.opens == 1
    drv.post("TaskChainCompleted", {"taskid": 1})
    drv.post("AllTasksCompleted", {})
    drv._running = False
    assert ch.push_actions([_DEPLOY]) is True
    assert ch.opens == 2
    assert len(drv.append_calls) == 2
    tid, params = drv.set_calls[-1]
    assert tid == 2
    assert params["details"]["actions"] == [_DEPLOY]


def test_channel_other_task_terminal_does_not_kill_chain() -> None:
    """别的 taskid 的终态事件不应误判本链死亡。"""
    drv = _FakeDriver()
    ch = MaaSingleStepChannel(drv, "0-1")  # type: ignore[arg-type]
    ch.open_stage()
    drv.post("TaskChainCompleted", {"taskid": 999})
    assert ch.open_stage() == 1
    assert ch.opens == 1


def test_channel_set_params_race_reopens_and_retries_once() -> None:
    """竞态（observe 后链才收尾 → set_task_params 被拒）：重开后再试一次。"""
    drv = _FakeDriver()
    ch = MaaSingleStepChannel(drv, "0-1")  # type: ignore[arg-type]
    assert ch.push_actions([dict(HEARTBEAT_ACTION)]) is True
    drv.set_results.append(False)  # 下一次 set_task_params 模拟竞态被拒
    assert ch.push_actions([_DEPLOY]) is True
    assert ch.opens == 2
    assert len(drv.append_calls) == 2
    assert drv.set_calls[-1][0] == 2


def test_channel_set_params_double_reject_gives_up() -> None:
    drv = _FakeDriver()
    ch = MaaSingleStepChannel(drv, "0-1")  # type: ignore[arg-type]
    ch.open_stage()
    drv.set_results.extend([False, False])
    assert ch.push_actions([_DEPLOY]) is False
    assert len(drv.set_calls) == 2


# ---------------------------------------------------------------------------
# 3b. 默认通道（append_per_action）：一推一链、先参后启、回执收取
# ---------------------------------------------------------------------------


def test_append_channel_order_is_params_before_start() -> None:
    """首跑 channel_dead 的修复点：动作子任务必须先于 start 就位（无速收竞态）。"""
    drv = _FakeDriver()
    ch = MaaAppendPerActionChannel(drv, "0-1", 15.0, 90.0)  # type: ignore[arg-type]
    assert ch.push_actions([dict(HEARTBEAT_ACTION)]) is True
    assert drv.call_order == ["append_task", "set_task_params", "start", "wait_chain"]
    task_type, stage_p = drv.append_calls[0]
    assert task_type == "SingleStep"
    assert stage_p == SingleStepParams.stage_params("0-1")
    tid, act_p = drv.set_calls[0]
    assert tid == 1
    assert act_p == SingleStepParams.action_params([dict(HEARTBEAT_ACTION)])
    # 回执游标从 append 前起算：快链在 wait_chain 前完成也不会漏读终态
    assert drv.wait_chain_calls == [0]
    assert ch.opens == 1


def test_append_channel_each_push_opens_new_chain() -> None:
    """一推一链：心跳拍与动作拍各开一条链（TaskChainCompleted 即天然回执）。"""
    drv = _FakeDriver()
    ch = MaaAppendPerActionChannel(drv, "0-1", 15.0, 90.0)  # type: ignore[arg-type]
    assert ch.push_actions([dict(HEARTBEAT_ACTION)]) is True
    assert ch.push_actions([_DEPLOY]) is True
    assert len(drv.append_calls) == 2
    assert ch.opens == 2
    assert drv.set_calls[1][0] == 2
    assert drv.set_calls[1][1]["details"]["actions"] == [_DEPLOY]


def test_append_channel_chain_error_receipt_is_reject() -> None:
    drv = _FakeDriver()
    drv.chain_results.append(ChainResult(False, "TaskChainError"))
    ch = MaaAppendPerActionChannel(drv, "0-1", 15.0, 90.0)  # type: ignore[arg-type]
    assert ch.push_actions([_DEPLOY]) is False


def test_append_channel_set_params_reject_skips_start() -> None:
    drv = _FakeDriver()
    drv.set_results.append(False)
    ch = MaaAppendPerActionChannel(drv, "0-1", 15.0, 90.0)  # type: ignore[arg-type]
    assert ch.push_actions([_DEPLOY]) is False
    assert drv.start_calls == 0


def test_make_channel_defaults_and_modes() -> None:
    drv = _FakeDriver()
    ch = make_channel("append_per_action", drv, "0-1", 600.0, 600.0)  # type: ignore[arg-type]
    assert isinstance(ch, MaaAppendPerActionChannel)
    ch2 = make_channel("single_step_inplace", drv, "0-1", 600.0, 600.0)  # type: ignore[arg-type]
    assert isinstance(ch2, MaaSingleStepChannel)


# ---------------------------------------------------------------------------
# 配置装载
# ---------------------------------------------------------------------------


def test_load_loop_config_roundtrip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "loop.json"
    cfg_path.write_text(
        json.dumps(
            {
                "schema": "loop_milestone_v1",
                "paths": {"ckpt": "/c.pt", "infer_cfg": "/i.yaml", "npz": "/n.npz"},
                "run": {
                    "heartbeat_s": 1.5,
                    "chain_timeout_s": 600,
                    "shell_timeout_s": 180,
                    "settle_timeout_s": 300,
                    "pre_retry_attempts": 2,
                    "pre_retry_idle_wait_s": 20,
                    "no_state_timeout_ticks": 40,
                    "max_model_actions": 64,
                    "support_unit_usage": 0,
                    "channel_mode": "single_step_inplace",
                },
                "stages": [
                    {
                        "stage_id": "main_00-01",
                        "stage_code": "0-1",
                        "ref_job_id": 38233,
                        "ref_episode_dir": "ark-data/episodes/main_00-01/job-x",
                        "total_proxy": 1,
                        "opers": [{"name": "芬", "skill": 1, "skill_usage": 0}],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cfg = load_loop_config(cfg_path)
    assert cfg.ckpt_path == "/c.pt"
    assert cfg.run.heartbeat_s == 1.5
    assert cfg.run.channel_mode == "single_step_inplace"  # 显式覆盖优先
    assert cfg.stages[0].stage_id == "main_00-01"
    assert cfg.stages[0].total_proxy == 1
    assert cfg.stages[0].opers[0]["name"] == "芬"


def test_shipping_milestone_config_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """仓内 configs/loop_milestone_v1.json 必须可被装载器吃进去（不读路径内容）。"""
    monkeypatch.setenv("ARK_ML_ROOT", str(tmp_path))
    root = Path(__file__).resolve().parents[1]
    cfg = load_loop_config(root / "configs" / "loop_milestone_v1.json")
    assert len(cfg.stages) == 3
    assert {s.stage_id for s in cfg.stages} == {
        "main_00-01",
        "main_00-02",
        "main_00-03",
    }
    # 首跑 channel_dead 后拍板的默认通道：一推一链（先参后启）
    assert cfg.run.channel_mode == "append_per_action"
    assert Path(cfg.ckpt_path or "") == tmp_path / "runs/imitation_v1/track_a_v1.pt"
    assert Path(cfg.infer_cfg_path or "") == root.parent / "core/configs/infer_v1.yaml"
