"""采集器恢复补充：严格按账号授权补给，未知同步确认交给玩家。"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bridge.blocker_watchdog import BLOCKER_SYNC_UPDATED, BlockerSentry
from bridge.resource_policy import originite_allowed


def locate_template(
    frame: np.ndarray, template: np.ndarray, roi: tuple[int, int, int, int]
) -> tuple[float, tuple[int, int]]:
    """归一化互相关返回匹配中心，避免用猜测坐标确认补给。"""
    x, y, width, height = roi
    image = frame[y : y + height, x : x + width].astype(np.float64)
    target = template.astype(np.float64)
    th, tw = target.shape[:2]
    ih, iw = image.shape[:2]
    if th > ih or tw > iw or not th or not tw:
        return 0.0, (0, 0)
    centered = target - target.mean(axis=(0, 1), keepdims=True)
    energy = float((centered**2).sum())
    if energy <= 0:
        return 0.0, (0, 0)
    shape = (ih + th - 1, iw + tw - 1)
    numerator = np.zeros((ih - th + 1, iw - tw + 1))
    for channel in range(3):
        correlation = np.fft.irfft2(
            np.fft.rfft2(image[..., channel], s=shape)
            * np.conj(np.fft.rfft2(centered[..., channel], s=shape)),
            s=shape,
        )
        numerator += correlation[: ih - th + 1, : iw - tw + 1]

    def sums(array: np.ndarray) -> np.ndarray:
        integral = np.pad(array, ((1, 0), (1, 0), (0, 0))).cumsum(0).cumsum(1)
        return (
            integral[th:, tw:]
            - integral[:-th, tw:]
            - integral[th:, :-tw]
            + integral[:-th, :-tw]
        )

    total = sums(image)
    variance = np.maximum(0.0, sums(image**2).sum(-1) - (total**2).sum(-1) / (th * tw))
    denominator = np.sqrt(variance * energy)
    scores = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12
    )
    row, col = np.unravel_index(int(scores.argmax()), scores.shape)
    return float(np.clip(scores[row, col], -1.0, 1.0)), (
        x + int(col) + tw // 2,
        y + int(row) + th // 2,
    )


class CollectorSentry(BlockerSentry):
    """只确认识别充分且已授权的补给，不自动同意未知更新/同步弹窗。"""

    def __init__(
        self,
        driver: Any,
        t0_wall_ms: int,
        *,
        on_message: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(driver, t0_wall_ms)
        self._on_message = on_message or (lambda _: None)
        self._originite = originite_allowed(driver._cfg)
        root = Path(driver._cfg["maa_dir"]) / "resource/template"
        self._templates = {}
        for name in ("UseStone.png", "MedicineConfirm.png"):
            path = next(root.rglob(name), None)
            if path is not None:
                self._templates[name] = np.asarray(Image.open(path).convert("RGB"))
        self._last_extra = float("-inf")
        self._stone_pending = 0
        self._latched = False
        self._last_click = float("-inf")

    def detect(self, image: Any) -> str | None:
        kind = super().detect(image)
        if kind == BLOCKER_SYNC_UPDATED:
            self._last_sig = None
            self._still_since = self._clock()
            self._on_message(
                "游戏可能需要处理更新或同步，请在模拟器内确认；软件不会自动同意未知弹窗"
            )
            return None
        return kind

    def poll(self) -> str | None:
        current = time.monotonic()
        if current - self._last_extra >= 2 and len(self._templates) == 2:
            self._last_extra = current
            try:
                image = self._drv.screencap()
            except Exception:
                return super().poll()
            if image.size != (1920, 1080):
                return super().poll()
            frame = np.asarray(image.convert("RGB").resize((1280, 720)))
            stone, _ = locate_template(
                frame, self._templates["UseStone.png"], (600, 130, 420, 420)
            )
            confirm, center = locate_template(
                frame, self._templates["MedicineConfirm.png"], (1000, 500, 200, 150)
            )
            if stone >= 0.94 and confirm >= 0.94:
                self._stone_pending += 1
                if not self._originite:
                    self._on_message(
                        "理智不足，当前账号未授权使用源石。已有数据保留，可等待恢复或调整账号补给选项"
                    )
                elif (
                    self._stone_pending >= 2
                    and not self._latched
                    and current - self._last_click > 20
                ):
                    self._drv.tap(round(center[0] * 1.5), round(center[1] * 1.5))
                    self._last_click, self._latched = current, True
                    self._events.append(
                        {
                            "kind": "authorized_originite_confirm",
                            "n": 1,
                            "t_ms": int(time.time() * 1000) - self._t0,
                        }
                    )
                    self._on_message("已按当前账号授权使用一颗已有源石补理智")
                    return "authorized_originite_confirm"
            else:
                self._stone_pending, self._latched = 0, False
        return super().poll()
