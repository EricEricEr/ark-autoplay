"""名册采集回归：空/未完成快照不能冒充成功，失败不能覆盖已有名册。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bridge import collect_operbox
from bridge.maa_driver import ChainResult, MaaEvent


def _event(own: list[dict[str, Any]], done: bool = True) -> MaaEvent:
    return MaaEvent(0, 0, "SubTaskExtraInfo", {
        "what": "OperBoxInfo", "details": {"done": done, "own_opers": own},
    })


def test_empty_and_partial_snapshots_are_not_final() -> None:
    assert collect_operbox.select_snapshot([
        _event([]), _event([{"name": "合成干员"}], done=False),
    ]) is None


def test_largest_nonempty_final_snapshot_is_preserved() -> None:
    full = _event([{"name": "合成甲"}, {"name": "合成乙"}])
    result = collect_operbox.select_snapshot([_event([]), full, _event([])])
    assert result is not None
    assert result["own_opers"] == full.details["details"]["own_opers"]


def test_failed_chain_keeps_previous_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Driver:
        def __init__(self, *_: Any) -> None:
            self.closed = False

        def connect(self) -> bool:
            return True

        def mark(self) -> int:
            return 0

        def append_task(self, *_: Any) -> int:
            return 1

        def start(self) -> bool:
            return True

        def wait_chain(self, **_: Any) -> ChainResult:
            return ChainResult(False, "TaskChainError", [_event([{"name": "合成甲"}])])

        def close(self) -> None:
            self.closed = True

    old = tmp_path / "roster.json"
    old.write_bytes(b'{"previous": true}')
    monkeypatch.setattr(collect_operbox, "MaaDriver", Driver)
    monkeypatch.setattr(collect_operbox, "load_instance_config", lambda _: {})
    monkeypatch.setattr(collect_operbox, "wait_settle", lambda *a, **k: True)
    result = collect_operbox.main([
        "--instance", str(tmp_path / "unused.yaml"), "--out", str(old),
        "--user-dir", str(tmp_path / "logs"),
    ])
    assert result == 4
    assert old.read_bytes() == b'{"previous": true}'
