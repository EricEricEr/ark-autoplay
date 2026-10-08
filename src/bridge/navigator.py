"""主线导航模块（自研 ADB 点触 + 模板校验）。

职责（对应设计文档 §10 数据工厂的"进图"环节）：从任意已知界面出发，
把游戏导航到目标主线关卡的 briefing（"开始行动"）界面，交给 Copilot 接管。

已确认的 UI 链路（1920x1080 / DPI 320 MuMu，开发机实测校准）::

    任意页 → 快捷导航(home 图标) → 终端 → 曲谱页(底部 tab) → 主旋律卡片
    → 章节 intro → 前往章节 → 章节地图 → 章节切换(prev/next 按钮)
    → 关卡节点（模板搜索 + 横滑扫描）→ briefing（标题模板校验）

模板均为开发机自采自裁的小块 UI 裁剪（见 ``configs/nav_templates/`` 说明），
几何与坐标一律入 ``configs/nav_main.yaml``，代码内无魔法坐标。

兜底策略（v1 未启用，TODO(M0.x)）：``maa_fight`` —— 向 MAA 下发 Fight 任务
代导航，回调出现关卡节点点击时 ``AsstStop`` 抢停在 briefing，再移交 Copilot。
抢停与 MAA 下一步点击存在竞争，故仅留 ``strategy`` 配置位与本文档，
取舍记录见 ``docs/adr/0001-pure-python-driver.md``。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import yaml
from PIL import Image

if TYPE_CHECKING:
    from bridge.maa_driver import MaaDriver

MATCH_THRESHOLD = 0.72
"""模板匹配默认阈值（TM_CCOEFF_NORMED）。"""

RESULT_TIMEOUT_S = 300.0
"""作战自然结束（结算画面出现）的最长等待。"""


class NavigationError(RuntimeError):
    """导航失败（状态机无法到达目标界面 / 模板未校准）。"""


@dataclass
class Match:
    """一次模板命中。"""

    name: str
    score: float
    center: tuple[int, int]


def to_gray(img: Image.Image | np.ndarray, scale: float = 1.0) -> np.ndarray:
    """PIL/BGR 图转灰度 ndarray；``scale`` 可做加速缩放。"""
    arr = np.asarray(img) if isinstance(img, Image.Image) else img
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if scale != 1.0:
        arr = cv2.resize(arr, None, fx=scale, fy=scale)
    return arr


def match_score(img_gray: np.ndarray, tmpl_gray: np.ndarray) -> float:
    """整图滑动模板匹配的最高分（TM_CCOEFF_NORMED）。"""
    if (
        img_gray.shape[0] < tmpl_gray.shape[0]
        or img_gray.shape[1] < tmpl_gray.shape[1]
    ):
        return 0.0
    res = cv2.matchTemplate(img_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    return float(res.max())


def locate(
    img_gray: np.ndarray, tmpl_gray: np.ndarray, threshold: float = MATCH_THRESHOLD
) -> Match | None:
    """整图搜索模板，命中返回中心坐标（原图坐标系需调用方按 scale 换算）。"""
    h, w = tmpl_gray.shape[:2]
    res = cv2.matchTemplate(img_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < threshold:
        return None
    return Match("", float(score), (loc[0] + w // 2, loc[1] + h // 2))


def classify_result(
    img_gray: np.ndarray, templates: dict[str, np.ndarray], threshold: float = 0.70
) -> tuple[str | None, float]:
    """结算画面判定：返回 ("result_win" | "result_fail" | None, 得分)。

    纯函数（注入灰度图与模板集），供单测用合成图验证。
    """
    best_name: str | None = None
    best_score = 0.0
    for name in ("result_win", "result_fail"):
        tmpl = templates.get(name)
        if tmpl is None:
            continue
        score = match_score(img_gray, tmpl)
        if score > best_score:
            best_name, best_score = name, score
    if best_name is not None and best_score >= threshold:
        return best_name, best_score
    return None, best_score


@dataclass
class ScreenBank:
    """模板仓库：加载 ``configs/nav_templates/*.png`` 为灰度图。"""

    templates: dict[str, np.ndarray] = field(default_factory=dict)
    """模板名（文件名去扩展名）→ 灰度 ndarray。"""

    @classmethod
    def load(cls, templates_dir: str | Path) -> ScreenBank:
        """从模板目录加载全部 PNG。"""
        bank: dict[str, np.ndarray] = {}
        for p in sorted(Path(templates_dir).glob("*.png")):
            img = Image.open(p).convert("L")
            bank[p.stem] = np.asarray(img)
        return cls(bank)

    def require(self, name: str) -> np.ndarray:
        """取模板，缺模板即视为未校准，抛 NavigationError。"""
        tmpl = self.templates.get(name)
        if tmpl is None:
            raise NavigationError(f"缺少未校准模板: {name}")
        return tmpl


class Navigator:
    """主线导航器（一个模拟器实例一个 Navigator）。"""

    def __init__(
        self,
        driver: MaaDriver,
        nav_cfg: dict[str, Any],
        templates: ScreenBank,
    ) -> None:
        """以导航配置 + 模板仓库初始化。"""
        self._driver = driver
        self._cfg = nav_cfg
        self._ui = nav_cfg["ui"]
        self._bank = templates
        self._last_cap: Image.Image | None = None

    # ---- 基础件 ----

    @staticmethod
    def load_nav_config(path: str | Path) -> dict[str, Any]:
        """读取 ``configs/nav_main.yaml``。"""
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict) or "chapters" not in cfg or "ui" not in cfg:
            raise ValueError(f"导航配置缺少 ui/chapters: {path}")
        return cfg

    def _cap(self) -> Image.Image:
        img = self._driver.screencap()
        self._last_cap = img
        return img

    def _hit(self, name: str, img: Image.Image) -> Match | None:
        tmpl = self._bank.templates.get(name)
        if tmpl is None:
            return None
        return locate(to_gray(img), tmpl)

    def _tap_xy(self, key: str) -> None:
        x, y = self._ui[key]
        self._driver.tap(int(x), int(y))

    def _wait_for(
        self, names: list[str], timeout_s: float, poll_s: float = 1.0
    ) -> tuple[str | None, Image.Image]:
        """轮询截图直到某模板命中；返回 (模板名, 当时截图)。"""
        deadline = time.monotonic() + timeout_s
        img = self._cap()
        while time.monotonic() < deadline:
            for name in names:
                if self._hit(name, img):
                    return name, img
            time.sleep(poll_s)
            img = self._cap()
        return None, img

    # ---- 界面识别 ----

    def current_chapter(self, img: Image.Image) -> int | None:
        """尽力识别章节地图的章节号（半透明底噪大，仅作日志提示用）。

        数字模板区分度不足（实测真/假命中 0.91/0.89 重叠），
        **不作为导航依据**；章节到位由目标节点模板搜索兜底验证。
        """
        for ch, ch_cfg in self._cfg["chapters"].items():
            digit = ch_cfg.get("geometry", {}).get("digit_template")
            if digit and self._hit(digit, img):
                return int(ch)
        return None

    def _state(self, img: Image.Image) -> str:
        """粗粒度界面分类。"""
        for name in (
            "exit_dialog",
            "popup_x",
            "quicknav_terminal",
            "terminal_tab",
            "story_title",
            "ep_goto",
            "brief_start",
            "home_terminal_tile",
        ):
            hit = self._hit(name, img)
            if hit and (name != "popup_x" or hit.score >= 0.80):
                return name
        return "unknown"

    # ---- 导航主流程 ----

    def goto_terminal(self, timeout_s: float = 60.0) -> None:
        """从任意已知界面恢复到终端界面。"""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            img = self._cap()
            state = self._state(img)
            if state == "terminal_tab":
                return
            if state == "exit_dialog":
                self._tap_xy("exit_dialog_cancel")  # 黑色 ✗ 取消退出
            elif state == "popup_x":
                hit = self._hit("popup_x", img)
                if hit:
                    self._driver.tap(*hit.center)  # 签到/公告弹窗关闭
            elif state == "quicknav_terminal":
                self._tap_xy("quicknav_terminal")
            elif state == "home_terminal_tile":
                self._tap_xy("home_terminal_tile")
            elif state in ("story_title", "ep_goto"):
                self._tap_xy("top_back")
            elif state == "brief_start":
                self._tap_xy("top_back")  # 关闭 briefing 面板回章节地图
            else:
                # 未知界面：优先尝试打开快捷导航，失败再返回键
                icon_tmpl = self._bank.templates.get("quicknav_icon")
                if icon_tmpl is not None and locate(to_gray(img), icon_tmpl):
                    self._tap_xy("quicknav_icon")
                else:
                    self._driver.key_back()
            time.sleep(1.5)
        raise NavigationError("timeout：无法到达终端")

    def enter_chapter_map(self, chapter: int) -> None:
        """从终端进入主旋律，点目标章节卡片并确认进入章节地图。

        章节卡片直通对应章节；落点校验由调用方的节点模板搜索完成
        （数字模板区分度不足，不靠 OCR 判章节号）。
        """
        if str(chapter) not in self._cfg["chapters"]:
            raise NavigationError(f"章节不在导航表: {chapter}")
        name, _ = self._wait_for(["terminal_tab"], timeout_s=10)
        if name is None:
            raise NavigationError("未处于终端界面")
        self._tap_xy("story_tab")
        name, _ = self._wait_for(["story_title"], timeout_s=15)
        if name is None:
            raise NavigationError("打开曲谱页失败")
        card_key = f"saga_card_{chapter}"
        if card_key not in self._ui:
            raise NavigationError(f"章节卡片未校准: {card_key}")
        self._tap_xy(card_key)
        name, _ = self._wait_for(["ep_goto"], timeout_s=15)
        if name is None:
            raise NavigationError("章节 intro 未出现（卡片坐标漂移？）")
        self._tap_xy("ep_goto")
        time.sleep(2.5)  # 章节地图落场动画

    def _swipe_map_left(self) -> None:
        x1, y1 = self._ui["map_swipe_from"]
        x2, y2 = self._ui["map_swipe_to"]
        self._driver.swipe(x2, y2, x1, y1)

    def _swipe_map_right(self) -> None:
        x1, y1 = self._ui["map_swipe_from"]
        x2, y2 = self._ui["map_swipe_to"]
        self._driver.swipe(x1, y1, x2, y2)

    def tap_stage_node(self, stage_code: str, max_sweeps: int = 8) -> None:
        """在当前章节地图上搜索关卡节点模板并点击。

        先向左滑到底（地图端点会钳位），再逐屏右扫；扫描不到抛错。
        """
        tmpl = self._bank.require(f"node_{stage_code}")
        for _ in range(3):
            self._swipe_map_left()
            time.sleep(0.8)
        last_frame: np.ndarray | None = None
        for _ in range(max_sweeps):
            time.sleep(1.2)
            img = self._cap()
            gray = to_gray(img)
            hit = locate(gray, tmpl)
            if hit:
                self._driver.tap(*hit.center)
                return
            if last_frame is not None:
                diff = float(np.mean(cv2.absdiff(gray, last_frame)))
                if diff < 1.5:
                    break  # 已到地图右端，画面不再变化
            last_frame = gray
            self._swipe_map_right()
        raise NavigationError(f"节点未找到: {stage_code}")

    def verify_briefing(self, stage_code: str, timeout_s: float = 12.0) -> None:
        """校验 briefing 面板打开且关卡标题正确。"""
        name, img = self._wait_for(["brief_start"], timeout_s=timeout_s)
        if name is None:
            raise NavigationError("briefing 未打开（开始行动 未识别）")
        title_tmpl = self._bank.require(f"brief_{stage_code}")
        if match_score(to_gray(img), title_tmpl) < MATCH_THRESHOLD:
            raise NavigationError(f"briefing 标题不符: {stage_code}")

    def goto_stage_briefing(self, stage_code: str) -> None:
        """总入口：任意已知界面 → 目标关卡 briefing（开始行动页）。

        章节落点校验 = 目标节点模板搜索；卡片落错相邻章节时
        用 prev/next 章节按钮各补救一次（v1 仅覆盖 0-1 两章）。
        """
        chapter = int(stage_code.split("-")[0])
        img = self._cap()
        if self._hit(f"brief_{stage_code}", img):
            return  # 已在目标 briefing
        self.goto_terminal()
        self.enter_chapter_map(chapter)
        last_exc: NavigationError | None = None
        for rescue in ("", "chapter_next", "chapter_prev_x2"):
            if rescue == "chapter_next":
                self._tap_xy("chapter_next")
            elif rescue == "chapter_prev_x2":
                self._tap_xy("chapter_prev")
                time.sleep(1.5)
                self._tap_xy("chapter_prev")
            if rescue:
                time.sleep(2.5)
            try:
                self.tap_stage_node(stage_code)
                break
            except NavigationError as exc:
                last_exc = exc
        else:
            raise last_exc or NavigationError(f"节点搜索失败: {stage_code}")
        self.verify_briefing(stage_code)

    def is_briefing_open(self) -> bool:
        """当前是否停在任意 briefing（结算后状态确认用）。"""
        return self._hit("brief_start", self._cap()) is not None

    # ---- 作战结果判定与战后推进 ----

    def detect_result(
        self, timeout_s: float = RESULT_TIMEOUT_S, poll_s: float = 2.0
    ) -> tuple[bool | None, Image.Image | None]:
        """轮询结算画面，判定胜负。

        返回 ``(win, settle_img)``；``win=None`` 表示超时未见结算画面
        （轨迹 result 记 null，可按哈希复查）。结算模板缺失时直接
        ``(None, None)``，不阻塞流水线。
        """
        if not any(k.startswith("result_") for k in self._bank.templates):
            return None, None
        deadline = time.monotonic() + timeout_s
        img: Image.Image | None = None
        while time.monotonic() < deadline:
            img = self._cap()
            name, _ = classify_result(to_gray(img, scale=0.5), {
                k: cv2.resize(v, None, fx=0.5, fy=0.5)
                for k, v in self._bank.templates.items()
                if k.startswith("result_")
            })
            if name == "result_win":
                return True, img
            if name == "result_fail":
                return False, img
            time.sleep(poll_s)
        return None, img

    def advance_after_battle(self, max_rounds: int = 12) -> None:
        """战后推进：点击屏幕跳过结算/掉落，回到 briefing 或章节地图。"""
        for _ in range(max_rounds):
            time.sleep(2.0)
            img = self._cap()
            if self._hit("brief_start", img) or self.current_chapter(img) is not None:
                return
            x, y = self._ui["settle_tap"]
            self._driver.tap(int(x), int(y))
        # 推进失败不致命：由调用方记录，下一局 goto 时恢复
