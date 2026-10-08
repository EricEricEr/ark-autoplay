"""战场状态采集落盘模块（v1：帧级采集）。

职责（对应设计文档 §5.4）：作战期间以固定节拍截图落盘 PNG，并同时保存
费用/击杀两个状态区域的裁剪图，写入 ``frames_index.jsonl`` 帧索引
（``{t_rel_ms, file, crops}``）。

v1 边界：回调不含战场状态（费用/击杀），故状态以"画面 + 区域裁剪"形式
落盘；OCR 数值标注为 **TODO(M0.x)**（本 PR 不引入 OCR 依赖）。
截图本体只落本机数据目录，入仓内容仅限哈希清单（素材红线 §6.3）。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bridge.maa_driver import MaaDriver


class StateLogger:
    """作战期帧采集线程。

    用法::

        logger = StateLogger(driver, run_dir, interval_s=1.5,
                             crops={"cost": [x, y, w, h], "kills": [x, y, w, h]})
        logger.start()          # 后台线程开始节拍截图
        logger.extra_shot("deploy_芬")   # 事件触发的补拍
        logger.stop()           # 收尾落盘
    """

    def __init__(
        self,
        driver: MaaDriver,
        run_dir: str | Path,
        interval_s: float = 1.5,
        crops: dict[str, list[int]] | None = None,
    ) -> None:
        """初始化采集器；``crops`` 为 {区域名: [x, y, w, h]}（1920x1080 基线）。"""
        self._driver = driver
        self._run_dir = Path(run_dir)
        self._shots_dir = self._run_dir / "shots"
        self._interval = interval_s
        self._crops = crops or {}
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._idx = 0
        self._t0_mono = 0.0
        self._index_fp: Any = None
        self._index_lock = threading.Lock()

    def _rel_ms(self) -> int:
        return int((time.monotonic() - self._t0_mono) * 1000)

    def _append_index(self, entry: dict[str, Any]) -> None:
        with self._index_lock:
            if self._index_fp is not None:
                self._index_fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._index_fp.flush()

    def _capture_once(self, tag: str | None) -> None:
        """截一帧并按需保存状态裁剪，追加帧索引。"""
        self._idx += 1
        img = self._driver.screencap()
        stem = f"f_{self._idx:05d}" + (f"_{tag}" if tag else "")
        frame_file = f"shots/{stem}.png"
        img.save(self._run_dir / frame_file)
        crop_files: dict[str, str] = {}
        for name, (x, y, w, h) in self._crops.items():
            crop_file = f"shots/{stem}_{name}.png"
            img.crop((x, y, x + w, y + h)).save(self._run_dir / crop_file)
            crop_files[name] = crop_file
        entry: dict[str, Any] = {"t_rel_ms": self._rel_ms(), "file": frame_file}
        if tag:
            entry["tag"] = tag
        if crop_files:
            entry["crops"] = crop_files
        self._append_index(entry)

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            next_tick = time.monotonic() + self._interval
            try:
                self._capture_once(tag=None)
            except Exception:  # noqa: BLE001  # 采集线程不允许崩掉整局
                self._append_index({"t_rel_ms": self._rel_ms(), "error": "capture"})
            wait = next_tick - time.monotonic()
            if wait > 0:
                self._stop_evt.wait(wait)

    def start(self) -> None:
        """启动后台节拍采集（幂等；重启需新建实例）。"""
        if self._thread is not None:
            return
        self._shots_dir.mkdir(parents=True, exist_ok=True)
        self._t0_mono = time.monotonic()
        self._index_fp = open(self._run_dir / "frames_index.jsonl", "a", encoding="utf-8")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def extra_shot(self, tag: str) -> None:
        """事件触发补拍一帧（如每次 CopilotAction、作战开始/结束）。"""
        try:
            self._capture_once(tag=tag)
        except Exception:  # noqa: BLE001
            self._append_index({"t_rel_ms": self._rel_ms(), "error": "extra_shot"})

    def stop(self) -> None:
        """停止采集并关闭帧索引。"""
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None
        with self._index_lock:
            if self._index_fp is not None:
                self._index_fp.close()
                self._index_fp = None
