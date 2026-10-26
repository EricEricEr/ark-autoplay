"""真实战斗生命周期识别：按 HUD/结算/主页证据判定，不用数值冻结代替结束。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bridge.blocker_watchdog import BLOCKER_SETTLE_LOST, detect_blocker


def normalized_template_score(image: np.ndarray, template: np.ndarray) -> float:
    """FFT归一化互相关；不新增OpenCV依赖，不改变或裁掉原始采样帧。"""
    image = np.asarray(image, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    height, width = template.shape[:2]
    ih, iw = image.shape[:2]
    if height > ih or width > iw or not height or not width:
        return 0.0
    centered = template - template.mean(axis=(0, 1), keepdims=True)
    energy = float((centered * centered).sum())
    if energy <= 0:
        return 0.0
    shape = (ih + height - 1, iw + width - 1)
    numerator = np.zeros((ih - height + 1, iw - width + 1))
    for channel in range(3):
        correlation = np.fft.irfft2(
            np.fft.rfft2(image[..., channel], s=shape)
            * np.conj(np.fft.rfft2(centered[..., channel], s=shape)),
            s=shape,
        )
        numerator += correlation[: ih - height + 1, : iw - width + 1]

    def window_sum(array):
        integral = np.pad(array, ((1, 0), (1, 0), (0, 0))).cumsum(0).cumsum(1)
        return (
            integral[height:, width:]
            - integral[:-height, width:]
            - integral[height:, :-width]
            + integral[:-height, :-width]
        )

    sums = window_sum(image)
    squared = window_sum(image * image)
    variance = np.maximum(
        0.0, squared.sum(-1) - (sums * sums).sum(-1) / (height * width)
    )
    denominator = np.sqrt(variance * energy)
    scores = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12
    )
    return float(np.clip(scores.max(), -1.0, 1.0))


@dataclass
class LifecycleObservation:
    """本帧真实画面证据及原图，所有证据均可随轨迹保存。"""

    active: bool
    result: str | None
    home: bool
    scores: dict[str, Any]
    frame: Image.Image


class BattleLifecycleObserver:
    """复用当前MAA资产识别真实终局，依赖只有项目已有Pillow与numpy。"""

    def __init__(self, maa_dir: str | Path) -> None:
        self.config = json.loads(
            (Path(__file__).parent / "resources/battle_lifecycle.json").read_text(
                encoding="utf-8"
            )
        )
        root = Path(maa_dir) / "resource/template"
        self.templates = {}
        for group in self.config["templates"].values():
            for name in group["names"]:
                path = next(root.rglob(name), None)
                if path is None:
                    raise RuntimeError(f"战斗生命周期识别缺少运行时模板：{name}")
                self.templates[name] = np.asarray(Image.open(path).convert("RGB"))

    def observe(self, frame: Image.Image) -> LifecycleObservation:
        normalized = np.asarray(frame.convert("RGB").resize((1280, 720)))
        scores = {}
        matches = {}
        for kind, group in self.config["templates"].items():
            x, y, width, height = group["roi"]
            patch = normalized[y : y + height, x : x + width]
            score = max(
                normalized_template_score(patch, self.templates[name])
                for name in group["names"]
            )
            scores[kind] = score
            matches[kind] = score >= group["threshold"]
        pause = self.config["pause"]
        x, y, width, height = pause["roi"]
        crop = Image.fromarray(normalized[y : y + height, x : x + width]).convert("L")
        pause_count = int((np.asarray(crop) > pause["luminance"]).sum())
        scores["pause_bright_pixels"] = pause_count
        won = matches["stars"] and matches["result_marker"]
        lost = detect_blocker(frame) == BLOCKER_SETTLE_LOST
        active = (
            pause_count > pause["pixels"]
            and matches["cost_marker"]
            and not won
            and not lost
        )
        return LifecycleObservation(
            active,
            "lost" if lost else "win" if won else None,
            matches["home"],
            scores,
            frame,
        )
