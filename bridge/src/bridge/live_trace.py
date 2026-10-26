"""采集中即时保存原始事件和状态，中断时也保留已收到的数据。"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class LiveTrace:
    """独立旁路 JSONL；与最终整理文件分开，不在收尾时覆盖旁路原件。"""

    def __init__(self, directory: Path, metadata: dict[str, Any]) -> None:
        self.directory = directory
        self._lock = threading.Lock()
        (directory / "episode.inprogress.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _append(self, name: str, value: dict[str, Any]) -> None:
        line = json.dumps(value, ensure_ascii=False) + "\n"
        with self._lock, (self.directory / name).open("a", encoding="utf-8") as stream:
            stream.write(line)

    def event(self, value: dict[str, Any]) -> None:
        """原始事件原样逐条落盘。"""
        self._append("events_live.jsonl", value)

    def state(self, value: dict[str, Any]) -> None:
        """每个收到的状态落盘，不仅保存喂给模型的最后一帧。"""
        self._append("states_live.jsonl", value)
