"""StateLogger 黑帧丢弃单测（D9）：黑帧不落盘、计数可读、标注事件带累计数。

不启动采集线程、不碰 adb（设备单写者纪律）：用假 driver 现场生成 PNG，
只验证 snap 的判定/落盘/记账/回调四条路径。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from bridge.state_logger import (
    BLACK_DROPPED_PREFIX,
    StateLogger,
    grayscale_mean,
    is_black_frame,
    is_dropped_shot,
)


class _FakeDriver:
    """按脚本依次返回帧；脚本项为 Exception 时抛错（模拟 adb 失败）。"""

    def __init__(self, frames: list[Image.Image | Exception]) -> None:
        self._frames = list(frames)
        self.calls = 0

    def screencap(self) -> Image.Image:
        self.calls += 1
        item = self._frames.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _black() -> Image.Image:
    return Image.new("RGB", (1920, 1080), (0, 0, 0))


def _bright() -> Image.Image:
    img = Image.new("RGB", (1920, 1080), (40, 60, 80))
    img.putpixel((10, 10), (255, 255, 255))
    return img


def _logger(tmp_path: Path, frames: list, **kw: object) -> StateLogger:
    (tmp_path / "shots").mkdir(parents=True, exist_ok=True)
    return StateLogger(_FakeDriver(frames), tmp_path, **kw)  # type: ignore[arg-type]


def test_grayscale_mean_and_black_detection() -> None:
    """黑帧判定：纯黑均值 0；正常画面均值远高；极暗但非全黑不误判。"""
    assert grayscale_mean(_black()) == 0.0
    assert is_black_frame(_black()) is True
    assert is_black_frame(_bright()) is False
    assert grayscale_mean(_bright()) > 10
    near_black = Image.new("RGB", (100, 100), (1, 1, 1))
    assert is_black_frame(near_black) is False


def test_black_frame_is_not_written_and_is_annotated(tmp_path: Path) -> None:
    """黑帧：不落盘、返回带累计数的标记、on_shot 收到标注、计数属性可读。"""
    shots: list[tuple[str, str]] = []
    logger = _logger(tmp_path, [_black()], on_shot=lambda t, r: shots.append((t, r)))
    out = logger.snap("tick")
    assert out == f"{BLACK_DROPPED_PREFIX}1"
    assert is_dropped_shot(out) is True
    assert shots == [("tick", f"{BLACK_DROPPED_PREFIX}1")]
    assert logger.shots_dropped_black == 1
    assert logger.black_dropped_count == 1
    assert logger.black_dropped_by_tag == {"tick": 1}
    assert list((tmp_path / "shots").iterdir()) == []


def test_bright_frame_is_written_unchanged(tmp_path: Path) -> None:
    """正常帧路径语义不变：落盘 shots/ 并返回相对路径，计数不涨。"""
    shots: list[tuple[str, str]] = []
    logger = _logger(tmp_path, [_bright()], on_shot=lambda t, r: shots.append((t, r)))
    out = logger.snap("tick")
    assert out == "shots/001_tick.png"
    assert is_dropped_shot(out) is False
    assert (tmp_path / "shots" / "001_tick.png").exists()
    assert shots == [("tick", out)]
    assert logger.shots_dropped_black == 0


def test_mixed_sequence_counts_and_callback(tmp_path: Path) -> None:
    """混合序列：序号连续（丢弃也占号）、累计数递增、按 tag 分类正确。"""
    events: list[tuple[str, int]] = []
    logger = _logger(
        tmp_path,
        [_black(), _bright(), _black(), _black()],
        on_black_frame=lambda t, n: events.append((t, n)),
    )
    outs = [
        logger.snap("tick"),
        logger.snap("action"),
        logger.snap("tick"),
        logger.snap("settle"),
    ]
    assert outs == [
        f"{BLACK_DROPPED_PREFIX}1",
        "shots/002_action.png",
        f"{BLACK_DROPPED_PREFIX}2",
        f"{BLACK_DROPPED_PREFIX}3",
    ]
    assert events == [("tick", 1), ("tick", 2), ("settle", 3)]
    assert logger.black_dropped_by_tag == {"tick": 2, "settle": 1}
    assert sorted(p.name for p in (tmp_path / "shots").iterdir()) == ["002_action.png"]


def test_screencap_failure_keeps_snapfail_semantics(tmp_path: Path) -> None:
    """截图异常仍走 SNAP_FAIL（不打断作战），且不计入黑帧丢弃。"""
    logger = _logger(tmp_path, [RuntimeError("adb 挂了")])
    out = logger.snap("tick")
    assert out.startswith("SNAP_FAIL:") and "adb 挂了" in out
    assert is_dropped_shot(out) is False
    assert logger.shots_dropped_black == 0
