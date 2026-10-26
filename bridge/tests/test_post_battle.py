"""战后剧情与收尾回归：空队列不等于已回主页，确认按钮必须有剧情上下文。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bridge import nav_maa
from bridge.maa_driver import MaaEvent


class _Driver:
    """逐任务回放真实格式回调，不连接游戏。"""

    def __init__(self, scripts: list[list[tuple[str, dict]]]) -> None:
        self.scripts = scripts
        self.appended: list[tuple[str, dict]] = []
        self.current: list[MaaEvent] = []
        self.active = False
        self.stopped = False

    def append_task(self, kind: str, params: dict) -> int:
        self.appended.append((kind, params))
        tid = len(self.appended)
        script = self.scripts.pop(0) if self.scripts else []
        self.current = [
            MaaEvent(0, 0, name, {"taskid": tid, "taskchain": kind, **body})
            for name, body in script
        ]
        return tid

    def mark(self) -> int:
        return 0

    def start(self) -> bool:
        self.active = True
        return True

    def events_since(self, mark: int) -> tuple[list[MaaEvent], int]:
        events, self.current = self.current, []
        self.active = False
        return events, mark + len(events)

    def running(self) -> bool:
        return self.active

    def stop(self) -> None:
        self.active = False
        self.stopped = True


def _task(name: str) -> tuple[str, dict]:
    return "SubTaskCompleted", {"details": {"task": name}}


def _home() -> list[tuple[str, dict]]:
    return [
        _task("ArkPostBattle@Home"),
        _task("ArkPostBattle@HomeStable"),
        ("TaskChainCompleted", {}),
        ("AllTasksCompleted", {}),
    ]


@pytest.fixture(autouse=True)
def _no_real_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nav_maa.time, "sleep", lambda _: None)
    monkeypatch.setattr(nav_maa, "wait_idle", lambda *args, **kwargs: True)


def test_story_skip_confirm_then_home_is_verified() -> None:
    driver = _Driver(
        [
            [
                _task("ArkPostBattle@SkipStory"),
                _task("ArkPostBattle@StoryConfirmButton"),
                _task("ArkPostBattle@ResultContinue"),
                ("TaskChainCompleted", {}),
            ],
            [("TaskChainCompleted", {})],
            _home(),
        ]
    )
    assert nav_maa.wait_settle(driver)  # type: ignore[arg-type]
    assert [kind for kind, _ in driver.appended] == ["Custom", "StartUp", "Custom"]


def test_already_home_needs_no_further_startup() -> None:
    driver = _Driver([_home()])
    assert nav_maa.wait_settle(driver)  # type: ignore[arg-type]
    assert len(driver.appended) == 1


def test_native_callbacks_with_stripped_prefix_verify_home() -> None:
    events = [
        (
            "SubTaskCompleted",
            {"first": ["ArkPostBattle@Entry"], "details": {"task": name}},
        )
        for name in ("Home", "HomeStable")
    ]
    driver = _Driver([events + [("TaskChainCompleted", {})]])
    assert nav_maa.wait_settle(driver)  # type: ignore[arg-type]


def test_error_followed_by_empty_queue_is_not_settled() -> None:
    failure = [("TaskChainError", {}), ("AllTasksCompleted", {})]
    driver = _Driver([failure.copy() for _ in range(3)])
    assert not nav_maa.wait_settle(driver)  # type: ignore[arg-type]


def test_empty_queue_alone_is_not_settled() -> None:
    driver = _Driver([[("AllTasksCompleted", {})] for _ in range(3)])
    assert not nav_maa.wait_settle(driver)  # type: ignore[arg-type]


def test_completion_without_home_evidence_is_not_settled() -> None:
    driver = _Driver([[("TaskChainCompleted", {})] for _ in range(3)])
    assert not nav_maa.wait_settle(driver)  # type: ignore[arg-type]


def test_one_home_frame_is_not_enough() -> None:
    single_home = [_task("ArkPostBattle@Home"), ("TaskChainCompleted", {})]
    driver = _Driver([single_home.copy() for _ in range(3)])
    assert not nav_maa.wait_settle(driver)  # type: ignore[arg-type]


def test_foreign_home_callbacks_cannot_finish_current_chain() -> None:
    foreign = [(name, {**body, "taskid": 999}) for name, body in _home()]
    driver = _Driver([foreign.copy() for _ in range(3)])
    assert not nav_maa.wait_settle(driver)  # type: ignore[arg-type]


def test_timeout_stops_current_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter(range(1000))
    monkeypatch.setattr(nav_maa.time, "monotonic", lambda: float(next(clock)))
    driver = _Driver([[]])
    monkeypatch.setattr(driver, "running", lambda: True)
    assert not nav_maa.wait_settle(driver, timeout_s=5)  # type: ignore[arg-type]
    assert driver.stopped


def test_confirmation_and_result_tasks_require_specific_evidence() -> None:
    from bridge import maa_driver

    resource = (
        Path(maa_driver.__file__).parent
        / "resources/post_battle/resource/tasks/ark_post_battle.json"
    )
    tasks = json.loads(resource.read_text(encoding="utf-8"))
    entry_next = tasks["ArkPostBattle@Entry"]["next"]
    assert "ArkPostBattle@StoryConfirmButton" not in entry_next
    assert tasks["ArkPostBattle@StoryConfirmText"]["algorithm"] == "OcrDetect"
    confirm_text = tasks["ArkPostBattle@StoryConfirmText"]["text"]
    assert all("跳过" in text for text in confirm_text)
    assert tasks["ArkPostBattle@Result"]["action"] == "DoNothing"
    assert tasks["ArkPostBattle@ResultContinue"]["template"] == "EndOfAction.png"
    assert not any(
        name.startswith(("StartButton", "BattleStart", "UseStone", "UsePrts"))
        for task in tasks.values()
        for name in task["next"]
    )
