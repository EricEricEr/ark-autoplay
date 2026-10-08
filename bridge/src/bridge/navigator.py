"""主线导航模块（策略 A：逐关固定点位 + 像素校验）。

职责（设计文档 §10 数据工厂的"进图"环节）：从已知界面出发，把游戏导航到
目标主线关卡的 briefing（"开始行动"准备界面），交给 Copilot 接管。

链路（1920x1080 MuMu 实测校准，坐标全部入 ``configs/nav_main.yaml``）::

    任意页 → 快捷导航图标 → 浮层"终端" → 曲谱 tab → EP00 卡片
    → 前往章节 → 章节地图（目标章>0 时点 EPISODE 下一章）
    → 锚定滑动到地图钳位端 → 点关卡节点 → 蓝色像素校验"开始行动"

实测依据：选关地图两端是钳位死端，钳位态坐标完全确定；中端滚动有
±140px 惯性漂移，故只收录钳位态可见关卡，点空时按补偿偏移重试。
v1WIP 仅服务 ``configs/jobs_main_v1.json`` 涉及关卡（0/1 章钳位关卡），
任意关通用导航与 OCR 校验属 TODO(M0.x)。

兜底策略 B（未启用，TODO）：MAA Fight 任务代导航 + 回调抢停 AsstStop。
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from PIL import Image

if TYPE_CHECKING:
    from bridge.maa_driver import MaaDriver


class NavigationError(RuntimeError):
    """导航失败（无法到达目标关卡 briefing）。"""


def blue_fraction(img: Image.Image, region: list[int]) -> float:
    """区域内"行动蓝"像素占比（纯函数，可单测）。

    判定式：B>140 且 B-R>50 且 B-G>30（对 0-1 briefing 实测=0.52，
    对地图/终端/主页同区域实测=0.00）。
    """
    x, y, w, h = region
    arr = np.asarray(img.convert("RGB"), dtype=np.int32)[y : y + h, x : x + w]
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return float(np.mean((b > 140) & (b - r > 50) & (b - g > 30)))


class Navigator:
    """固定点位导航器（一个模拟器实例一个 Navigator）。"""

    def __init__(self, driver: MaaDriver, nav_cfg: dict[str, Any]) -> None:
        """以导航配置初始化（配置结构见 configs/nav_main.yaml 注释）。"""
        self._driver = driver
        self._cfg = nav_cfg
        self._ui = nav_cfg["ui"]

    @staticmethod
    def load_nav_config(path: str | Path) -> dict[str, Any]:
        """读取导航配置；``ui/stages/brief_blue_check`` 缺一报错。"""
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for key in ("ui", "stages", "brief_blue_check"):
            if not isinstance(cfg, dict) or key not in cfg:
                raise ValueError(f"导航配置缺少字段 {key}: {path}")
        return cfg

    def _tap(self, key: str) -> None:
        x, y = self._ui[key]
        self._driver.tap(int(x), int(y))

    def briefing_open(self) -> bool:
        """当前是否处于 briefing（开始行动蓝钮像素校验）。"""
        chk = self._cfg["brief_blue_check"]
        img = self._driver.screencap()
        return blue_fraction(img, chk["region"]) >= float(chk["threshold"])

    def _swipe_to_clamp(self, anchor: str) -> None:
        """锚定滑动：过量滑向地图钳位端（左锚=向右滑出早期关卡）。"""
        x1, y1 = self._ui["anchor_swipe_from"]
        x2, y2 = self._ui["anchor_swipe_to"]
        n = int(self._cfg.get("anchor_swipes", 3))
        for _ in range(n):
            if anchor == "left":
                self._driver.swipe(x2, y2, x1, y1)  # 手指右移 → 内容右移回左端
            else:
                self._driver.swipe(x1, y1, x2, y2)
            time.sleep(1.0)

    def _enter_chapter_map(self, chapter: int) -> None:
        """盲序列：快捷导航 → 终端 → 曲谱 → EP00 卡 → 前往章节 → 切章。"""
        self._tap("quicknav_icon")
        time.sleep(1.5)
        self._tap("quicknav_terminal")
        time.sleep(3.0)  # 终端有暗场过场
        self._tap("story_tab")
        time.sleep(2.5)
        self._tap("saga_card_0")
        time.sleep(2.0)
        self._tap("ep_goto")
        time.sleep(3.0)  # 章节地图落场
        for _ in range(chapter):  # v1 仅 0/1 章：0 次或 1 次
            self._tap("chapter_next")
            time.sleep(2.0)

    def _tap_node(self, stage_code: str) -> None:
        """锚定后点关卡节点；点空按补偿偏移重试，每次以蓝色校验裁决。"""
        spec = self._cfg["stages"].get(stage_code)
        if spec is None:
            raise NavigationError(f"关卡未标定（不在 nav_main.yaml）: {stage_code}")
        self._swipe_to_clamp(spec["anchor"])
        x, y = spec["node"]
        offsets = [[0, 0], *self._ui.get("node_retry_offsets", [])]
        for dx, dy in offsets:
            self._driver.tap(int(x) + dx, int(y) + dy)
            time.sleep(2.0)
            if self.briefing_open():
                return
        raise NavigationError(f"节点点击未打开 briefing: {stage_code}")

    def goto_stage_briefing(self, stage_code: str) -> None:
        """总入口：任意已知界面 → 目标关卡 briefing；整体失败重试一次。"""
        # 前置清理：连点跳过残留结算链（任务失败/物资结算等"点击继续"屏），
        # 若落回 briefing 则退回章节地图，交由盲序列统一重进。
        for _ in range(4):
            if self.briefing_open():
                self._driver.key_back()
                time.sleep(1.5)
                break
            x, y = self._ui["settle_tap"]
            self._driver.tap(int(x), int(y))
            time.sleep(2.0)
        for attempt in range(2):
            try:
                spec = self._cfg["stages"].get(stage_code)
                if spec is None:
                    raise NavigationError(f"关卡未标定: {stage_code}")
                self._enter_chapter_map(int(spec["chapter"]))
                self._tap_node(stage_code)
                return
            except NavigationError:
                if attempt == 1:
                    raise
                time.sleep(2.0)

    def wait_battle_settle(self, timeout_s: float = 300.0) -> list[str]:
        """作战收尾：链结束后轮询，点到结算画面直到回到 briefing。

        返回推进过程抓到的结算帧文件路径列表（由调用方落盘补拍；
        本方法只负责推进）。v1WIP：以蓝色校验回到 briefing 为准，
        胜负判定/OCR 属 TODO(M0.x)。
        """
        deadline = time.monotonic() + timeout_s
        tapped = 0
        while time.monotonic() < deadline:
            if self.briefing_open():
                return []
            x, y = self._ui["settle_tap"]
            self._driver.tap(int(x), int(y))
            tapped += 1
            time.sleep(4.0)
            if tapped > 60:
                break
        return []

    def advance_after_battle(self) -> None:
        """战后推进到 briefing（best-effort；失败由下一局导航兜底）。"""
        with contextlib.suppress(Exception):
            self.wait_battle_settle()
