"""CopilotRunner / 代理护栏 单测（纯逻辑，不连真机）。

背景（重要，勿重蹈）：曾用 MAA 的 ``Fight`` 任务做导航，但实测它会走进
**游戏内代理作战**（``Fight@PRTS1`` 出现 13 次，``PrtsErrorConfirm`` 4 次），
而自抽号的代理记录是号商机械刷出的异常数值、对训练无价值。故改为
``Copilot`` 路径（自带导航 + 显式 ``NotUsePrts``）。

本测试覆盖：
1. 代理护栏能识别 ``UsePrts``/``PRTS1`` 等任务名（**尾段匹配**，因 MAA 报的是
   ``Fight@UsePrts`` 这种带链前缀的形式）；
2. 正常 Copilot 路径不误报；
3. CopilotRunner 的终态判定、超时行为与参数透传。
"""

from __future__ import annotations

from bridge.nav_maa import PRTS_TASKS, CopilotRunner


class _FakeDriver:
    """假驱动：按脚本回放事件。"""

    def __init__(self, tasks: list[str], append_ret: int = 1) -> None:
        self._tasks = tasks
        self._append_ret = append_ret
        self._cursor = 0
        self.appended: list[tuple[str, str, bool]] = []

    def append_copilot(self, job_path, formation: bool = True) -> int:
        self.appended.append((str(job_path), "Copilot", formation))
        return self._append_ret

    def mark(self) -> int:
        return self._cursor

    def events_since(self, mark: int):
        out = []
        while self._cursor < len(self._tasks):
            raw = self._tasks[self._cursor]
            self._cursor += 1
            if raw == "__DONE__":
                done = type(
                    "E", (), {"msg": "TaskChainCompleted", "details": {}}
                )()
                out.append(done)
                continue
            # 模拟 MAA 真实格式：cur_task 形如 "Fight@UsePrts"
            out.append(
                type("E", (), {"msg": "SubTaskStart", "details": {"cur_task": raw}})()
            )
        return out, self._cursor

    def start(self) -> bool:
        return True


def test_prts_tasks_covers_the_observed_ones() -> None:
    """护栏必须覆盖实测观察到的代理任务名（否则等于没护栏）。"""
    for t in ("UsePrts", "PRTS1", "PRTS2", "PRTS3", "PrtsErrorConfirm"):
        assert t in PRTS_TASKS, f"{t} 必须被护栏覆盖（实测出现过）"


def test_runner_detects_prts_contamination() -> None:
    """出现 Fight@UsePrts / Fight@PRTS1 时必须标记污染（尾段匹配）。"""
    drv = _FakeDriver(
        ["Copilot@BattleStartPre", "Fight@UsePrts", "Fight@PRTS1", "__DONE__"]
    )
    res = CopilotRunner(drv, timeout_s=5.0).run_job("job.json")  # type: ignore[arg-type]
    assert res.ok is True
    assert "UsePrts" in res.prts_detected
    assert "PRTS1" in res.prts_detected


def test_runner_clean_path_has_no_prts_flag() -> None:
    """正常 Copilot 路径不得误报污染（用 BattleStartPre/QuickFormation 链）。"""
    drv = _FakeDriver(
        [
            "Copilot@BattleStartPre",
            "Copilot@BattleQuickFormation",
            "Copilot@BattleStartAll",
            "Copilot@BattleProcessTask",
            "__DONE__",
        ]
    )
    res = CopilotRunner(drv, timeout_s=5.0).run_job("job.json")  # type: ignore[arg-type]
    assert res.ok is True
    assert res.prts_detected == [], "正常路径不应出现代理标记"


def test_runner_reports_timeout_as_not_ok() -> None:
    """任务链不结束时 ok=False（不抛异常，便于批量流程继续）。"""
    drv = _FakeDriver(["Copilot@BattleStartPre"])
    res = CopilotRunner(drv, timeout_s=0.3).run_job("job.json")  # type: ignore[arg-type]
    assert res.ok is False
    assert res.terminal is None


def test_runner_passes_job_path_and_formation() -> None:
    """作业路径与 formation 参数须透传给 append_copilot。"""
    drv = _FakeDriver(["__DONE__"])
    CopilotRunner(drv, timeout_s=2.0).run_job("j.json", formation=False)  # type: ignore[arg-type]
    assert drv.appended == [("j.json", "Copilot", False)]
