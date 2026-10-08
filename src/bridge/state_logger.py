"""作战期节拍截图模块（沿用 capture_episode.py 已验证节拍）。

职责（设计文档 §5.4）：任务链激活期间以固定节拍（默认 1.5s）adb 截屏
落盘 PNG（序号_标签 命名），并向调用方回报 (tag, 相对路径) 供写入
事件流；支持事件触发的即时补拍（CopilotAction / result 等）。

v1WIP 边界：回调不含费用/击杀等数值战场状态，状态以截图本体落盘；
数值 OCR 标注属 TODO(M0.x)。截图只落本机数据目录，入仓内容仅限
sha256 清单（素材红线 §6.3）。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bridge.maa_driver import MaaDriver


class StateLogger:
    """节拍截图线程。

    用法::

        logger = StateLogger(driver, run_dir, on_shot=cb)
        logger.set_active(True)     # TaskChainStart 时
        logger.snap("result")       # 事件触发补拍（线程安全）
        logger.stop()
    """

    def __init__(
        self,
        driver: MaaDriver,
        run_dir: str | Path,
        on_shot: Callable[[str, str], None] | None = None,
        interval_s: float = 1.5,
    ) -> None:
        """初始化；``on_shot(tag, relpath)`` 在每帧落盘后触发。"""
        self._driver = driver
        self._dir = Path(run_dir)
        self._shots = self._dir / "shots"
        self._on_shot = on_shot
        self._interval = interval_s
        self._active = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._seq = 0

    def set_active(self, active: bool) -> None:
        """作战是否激活（激活期间节拍帧才落盘）。"""
        if active:
            self._active.set()
        else:
            self._active.clear()

    def start(self) -> None:
        """启动后台节拍线程（幂等）。"""
        if self._thread is not None:
            return
        self._shots.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def snap(self, tag: str) -> str:
        """立即截一帧落盘，返回 shots 相对路径（POSIX 分隔符）。"""
        with self._lock:
            self._seq += 1
            name = f"{self._seq:03d}_{tag}.png"
        rel = f"shots/{name}"
        try:
            img = self._driver.screencap()
            img.save(self._shots / name)
            out = rel
        except Exception as exc:  # noqa: BLE001  # 截图失败不打断作战
            out = f"SNAP_FAIL:{exc}"
        if self._on_shot is not None:
            self._on_shot(tag, out)
        return out

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._active.is_set():
                self.snap("tick")
            self._stop.wait(self._interval)

    def stop(self) -> None:
        """停止节拍线程。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None
