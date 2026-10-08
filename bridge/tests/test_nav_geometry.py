"""导航几何表单测：nav_main.yaml 结构合法性与 jobs 覆盖一致性。"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from bridge.navigator import blue_fraction

REPO_ROOT = Path(__file__).resolve().parents[1]
NAV_YAML = REPO_ROOT / "configs" / "nav_main.yaml"
JOBS_JSON = REPO_ROOT / "configs" / "jobs_main_v1.json"


def _cfg() -> dict:
    with open(NAV_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_ui_points_in_bounds() -> None:
    """所有 ui 点位/补偿偏移都在基线分辨率内。"""
    cfg = _cfg()
    w, h = 1920, 1080  # 基线分辨率（AGENTS.md 铁律：固定基线）
    for key, pt in cfg["ui"].items():
        if key == "node_retry_offsets":
            for dx, dy in pt:
                assert abs(dx) <= w // 4 and abs(dy) <= h // 4
            continue
        x, y = pt
        assert 0 <= x < w and 0 <= y < h, f"{key}={pt} 越界"


def test_brief_check_region_in_bounds() -> None:
    """蓝色校验区域在屏内且阈值介于 (0,1)。"""
    cfg = _cfg()
    x, y, w, h = cfg["brief_blue_check"]["region"]
    threshold = float(cfg["brief_blue_check"]["threshold"])
    assert 0 < threshold < 1.0
    assert x >= 0 and y >= 0 and x + w <= 1920 and y + h <= 1080


def test_stages_table_valid() -> None:
    """关卡表：章节为 int、锚点为 left/right、节点坐标在屏内。"""
    cfg = _cfg()
    assert int(cfg["anchor_swipes"]) >= 2  # 必须过量滑动确保钳位
    for code, spec in cfg["stages"].items():
        ch, num = code.split("-")
        assert spec["chapter"] == int(ch)
        assert spec["anchor"] in ("left", "right")
        x, y = spec["node"]
        assert 0 <= x < 1920 and 0 <= y < 1080, f"{code} 节点越界"


def test_jobs_stages_covered_by_nav() -> None:
    """jobs 队列里的每个关卡都在导航表中有标定。"""
    with open(JOBS_JSON, encoding="utf-8") as f:
        jobs = json.load(f)["jobs"]
    stages = _cfg()["stages"]
    for entry in jobs:
        code = entry["maa_job"]["stage_name"]
        assert code in stages, f"{code} 未在 nav_main.yaml 标定"


def test_blue_fraction_synthetic() -> None:
    """蓝色像素判定：纯蓝区域≈1，灰/红区域≈0。"""
    blue = Image.new("RGB", (40, 30), (20, 120, 220))
    gray = Image.new("RGB", (40, 30), (120, 120, 120))
    red = Image.new("RGB", (40, 30), (220, 40, 40))
    region = [0, 0, 40, 30]
    assert blue_fraction(blue, region) == pytest.approx(1.0)
    assert blue_fraction(gray, region) == pytest.approx(0.0)
    assert blue_fraction(red, region) == pytest.approx(0.0)


def test_blue_fraction_partial() -> None:
    """半蓝半灰区域占比约 0.5（像素级正确性）。"""
    arr = np.zeros((20, 20, 3), dtype=np.uint8) + 120
    arr[:, :10] = (20, 120, 220)
    img = Image.fromarray(arr)
    assert blue_fraction(img, [0, 0, 20, 20]) == pytest.approx(0.5)
