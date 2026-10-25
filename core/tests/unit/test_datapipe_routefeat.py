"""datapipe.routefeat 单测：合成数据，不含任何真实游戏内容。

重点是三件错了后果严重的事：
1. **折线栅格化与 BFS 距离口径**：waypoint 间 L 形填充，dist 沿路线格 4 连通增长，
   不可达格统一记 1.0（归一化后），不可静默成 0；
2. **飞行路线排除**：mode == FLY（1）的路线不贡献任何格子；
3. **无路线几何降级**：整关 3 通道全零 + ``has_routes=False``（调用方据此计数）。
"""

from __future__ import annotations

import numpy as np
import pytest

from ark_core.datapipe.routefeat import (
    ROUTE_CHANNELS,
    WALK_MODE,
    RoutefeatError,
    route_feature_channels,
    walk_route_cells,
)
from ark_core.protocol.action import GRID_CHANNELS

ROWS, COLS = 3, 4


def _sf(routes: list[dict] | None) -> dict:
    """合成关卡：3×4 格子，起点 (0,0)，终点 (2,3)，其余地形通道为 0。"""
    grid = []
    for r in range(ROWS):
        for c in range(COLS):
            grid.extend([0, 1, 1, int((r, c) == (0, 0)), int((r, c) == (2, 3))])
    return {
        "stage_id": "syn_route",
        "rows": ROWS,
        "cols": COLS,
        "grid": grid,
        "routes": routes or [],
    }


_WALK_ROUTE = {
    "mode": WALK_MODE,
    "start": [0, 0],
    "end": [2, 3],
    "n_points": 3,
    "points": [0, 0, 0, 3, 2, 3],
    "wait_ms": [],
}

# L 形栅格化后的路线格：上排 (0,0..3) + 右列 (1,3)(2,3)
_ROUTE_CELLS = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]


def test_walk_route_cells_l_shape_rasterize() -> None:
    """相邻 waypoint 按「先纵后横」L 形填充，端点也在集合内。"""
    cells = walk_route_cells(_sf([_WALK_ROUTE]), ROWS, COLS)
    assert cells == set(_ROUTE_CELLS)


def test_channels_dist_start_end_and_membership() -> None:
    """dist_start 自 (0,0) 递增、dist_end 自 (2,3) 递增，on_route 命中路线格。"""
    feats, has_routes = route_feature_channels(_sf([_WALK_ROUTE]))
    assert has_routes
    assert feats.shape == (ROWS * COLS, ROUTE_CHANNELS)

    def cell(r: int, c: int) -> np.ndarray:
        return feats[r * COLS + c]

    # 路径是 6 格单链：dist_start = 距 (0,0) 的步数 / 5，dist_end = 距 (2,3) 的步数 / 5
    assert cell(0, 0).tolist() == [pytest.approx(0.0), pytest.approx(1.0), 1.0]
    assert cell(2, 3).tolist() == [pytest.approx(1.0), pytest.approx(0.0), 1.0]
    assert cell(0, 2).tolist() == [pytest.approx(2 / 5), pytest.approx(3 / 5), 1.0]
    # 非路线格：两条距离不可达都记 1.0，membership 为 0
    assert cell(1, 0).tolist() == [1.0, 1.0, 0.0]
    assert cell(2, 1).tolist() == [1.0, 1.0, 0.0]


def test_fly_route_excluded() -> None:
    """mode == FLY（1）的路线不贡献格子；只有飞行路线时等同无路线。"""
    fly = {**_WALK_ROUTE, "mode": 1}
    feats, has_routes = route_feature_channels(_sf([fly]))
    assert not has_routes
    assert feats.shape == (ROWS * COLS, ROUTE_CHANNELS)
    assert float(feats.sum()) == 0.0


def test_unknown_mode_route_excluded() -> None:
    """mode == -1（E_NUM 占位等未登记编码）同样排除。"""
    feats, has_routes = route_feature_channels(_sf([{**_WALK_ROUTE, "mode": -1}]))
    assert not has_routes
    assert float(feats.sum()) == 0.0


def test_no_routes_gives_zeros() -> None:
    """路线列表空 / 缺 routes 键：全零通道 + False（调用方计数，不报错）。"""
    feats, has_routes = route_feature_channels(_sf(None))
    assert not has_routes
    assert feats.shape == (ROWS * COLS, ROUTE_CHANNELS)
    assert float(feats.sum()) == 0.0

    sf = _sf(None)
    del sf["routes"]
    feats2, has2 = route_feature_channels(sf)
    assert not has2
    assert float(feats2.sum()) == 0.0


def test_endpoints_off_polyline_still_reachable() -> None:
    """start/end 不在折线上时并入 BFS 可走域：端点 dist = 0 而不是 1.0。"""
    route = {
        "mode": WALK_MODE,
        "start": [0, 0],
        "end": [2, 3],
        "n_points": 2,
        "points": [1, 1, 1, 2],  # 折线不经过 (0,0)/(2,3)
        "wait_ms": [],
    }
    feats, has_routes = route_feature_channels(_sf([route]))
    assert has_routes
    assert feats[0 * COLS + 0, 0] == pytest.approx(0.0)
    assert feats[2 * COLS + 3, 1] == pytest.approx(0.0)


def test_grid_size_mismatch_raises() -> None:
    """grid 长度与 rows*cols*GRID_CHANNELS 不符必须报错（口径不动，fail fast）。"""
    sf = _sf([_WALK_ROUTE])
    sf["grid"] = sf["grid"][: -GRID_CHANNELS]
    with pytest.raises(RoutefeatError):
        route_feature_channels(sf)
