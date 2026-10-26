"""作战期节拍截图模块（沿用 capture_episode.py 已验证节拍）。

职责（设计文档 §5.4）：任务链激活期间以固定节拍（默认 1.5s）adb 截屏
落盘 PNG（序号_标签 命名），并向调用方回报 (tag, 相对路径) 供写入
事件流；支持事件触发的即时补拍（CopilotAction / result 等）。

v1WIP 边界：回调不含费用/击杀等数值战场状态，状态以截图本体落盘；
数值 OCR 标注属 TODO(M0.x)。截图只落本机数据目录，入仓内容仅限
sha256 清单（素材红线 §6.3）。

黑帧策略（用户裁决 2026-10-10，D9）
-----------------------------------
verifier 已定性：tick 帧黑率 4.4%，黑帧字节恒等（6109B、灰度均值 0、
sha256 前缀 0f52608dd021），96.6% 相位锁定在 BattleStart* 过场。
裁决 = **丢弃 + 标注**，不重拍、不降节拍（节拍保持 1.5s）。因此：

- 截到黑帧时**不落盘**（免下游把黑帧当有效画面）；
- 仍经 on_shot 回报一条标注事件，值为 BLACK_DROPPED:<本局累计丢弃数>
  （写进 action_events.jsonl 的 Shot* 记录里即可见）；
- 累计丢弃数经 shots_dropped_black 属性暴露给调用方，供 episode.json
  增加 shots_dropped_black 字段（由 replay_controller 接线）。

接口稳定性：shots_dropped_black（int，线程安全只读属性）与
BLACK_DROPPED_PREFIX 是外部契约，改名属破坏性变更。
SNAP_FAIL: 与 BLACK_DROPPED: 都不是落盘路径，用 is_dropped_shot()
或前缀判断区分；build_shot_manifest 只扫目录，故黑帧天然不进清单。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from bridge.maa_driver import MaaDriver


BLACK_DROPPED_PREFIX: str = "BLACK_DROPPED:"
"""黑帧丢弃标记前缀；snap 返回 / on_shot 收到 BLACK_DROPPED:<累计数>。"""

BLACK_MEAN_MAX: float = 0.5
"""判黑的灰度均值上限（0-255）。实测黑帧均值恒为 0，留少量余量抗噪声。"""


def grayscale_mean(img: Image.Image) -> float:
    """灰度均值（0-255）；用直方图统计，不依赖 numpy。"""
    hist = img.convert("L").histogram()
    total = sum(hist)
    if not total:
        return 0.0
    return sum(i * n for i, n in enumerate(hist)) / total


def is_black_frame(img: Image.Image, mean_max: float = BLACK_MEAN_MAX) -> bool:
    """是否黑帧：灰度均值 <= mean_max。

    实测黑帧（BattleStart 过场期的 adb 截图）灰度均值恒为 0；阈值取 0.5
    而非严格 0 是为了容忍个别像素噪声，同时远低于任何真实战场画面
    （正常画面含 UI/干员，均值通常远大于 10）。
    """
    return grayscale_mean(img) <= mean_max


def is_dropped_shot(rel: str) -> bool:
    """snap 返回值是否为"黑帧已丢弃"标记（而非落盘相对路径）。"""
    return rel.startswith(BLACK_DROPPED_PREFIX)


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
        on_black_frame: Callable[[str, int], None] | None = None,
        interval_s: float = 1.5,
        on_storage_error: Callable[[OSError], None] | None = None,
    ) -> None:
        """初始化；``on_shot(tag, relpath)`` 在每帧落盘后触发。

        ``on_black_frame(tag, dropped_total)`` 仅在该帧被判为黑帧并丢弃时触发
        （可选，供调用方额外记账；不传时标注事件仍会经 on_shot 发出）。
        """
        self._driver = driver
        self._dir = Path(run_dir)
        self._shots = self._dir / "shots"
        self._on_shot = on_shot
        self._on_black_frame = on_black_frame
        self._interval = interval_s
        self._on_storage_error = on_storage_error
        self._active = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._seq = 0
        self._black_dropped = 0
        self._black_dropped_by_tag: dict[str, int] = {}

    def set_active(self, active: bool) -> None:
        """作战是否激活（激活期间节拍帧才落盘）。"""
        if active:
            self._active.set()
        else:
            self._active.clear()

    @property
    def shots_dropped_black(self) -> int:
        """本局累计丢弃的黑帧数（稳定接口，供 episode.json 的 shots_dropped_black）。

        与采集线程并发安全：可在 `snap` 进行中读取。
        """
        with self._lock:
            return self._black_dropped

    # 与 episode.json 字段同名的别名，调用方可直接取用而无需记两套名字
    black_dropped_count = shots_dropped_black

    @property
    def black_dropped_by_tag(self) -> dict[str, int]:
        """按标签分类的丢弃计数（诊断用，返回副本，不暴露内部可变状态）。"""
        with self._lock:
            return dict(self._black_dropped_by_tag)

    def start(self) -> None:
        """启动后台节拍线程（幂等）。"""
        if self._thread is not None:
            return
        self._shots.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def snap(self, tag: str) -> str:
        """立即截一帧：非黑帧落盘并返回 shots 相对路径（POSIX 分隔符）。

        黑帧（见模块 docstring 的黑帧策略）**不落盘**，返回
        `BLACK_DROPPED:<本局累计丢弃数>` 标记，并同样触发 on_shot，
        使 action_events.jsonl 留下一条带累计数的标注事件。
        """
        with self._lock:
            self._seq += 1
            name = f"{self._seq:03d}_{tag}.png"
        rel = f"shots/{name}"
        try:
            img = self._driver.screencap()
            if is_black_frame(img):
                with self._lock:
                    self._black_dropped += 1
                    self._black_dropped_by_tag[tag] = (
                        self._black_dropped_by_tag.get(tag, 0) + 1
                    )
                    dropped = self._black_dropped
                out = f"{BLACK_DROPPED_PREFIX}{dropped}"
                if self._on_black_frame is not None:
                    self._on_black_frame(tag, dropped)
            else:
                try:
                    img.save(self._shots / name)
                except OSError as error:
                    if self._on_storage_error is not None:
                        self._on_storage_error(error)
                    raise
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
