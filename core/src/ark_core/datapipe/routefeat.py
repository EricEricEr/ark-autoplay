"""routefeat：敌方行进路线几何 → 每格 3 条路线通道（v1.1 数据集与在线推理共用）。

动机与口径
----------
v1 格子特征只有地形 5 通道（h / build / deployable / is_start / is_end），cell 头
（「部署到哪」）没有显式路线几何。本模块实验性地从 stagefeat.parquet 的路线
派生特征；离线 K 折尚未证实改善（见 docs/reports/2026-10-24-offline-v1-1.md）。
输入为 ``routes`` 几何（ADR-0003 定义的
``[r, c]`` 折线 waypoint）派生 3 条每格通道，训练（``state_action_data``）与推理
（``infer.track_a_v1``）走**同一函数**，避免两侧漂移。

输出通道（每格 3 维，float32，顺序固定）
------------------------------------------
0. ``dist_start``：到敌方出怪点的最短步行距离，沿**地面路线格子集**做 4 连通
   BFS（多源 = grid 编码第 3 通道 is_start 的格子），归一化到 [0,1]
   （除以可达最大距离；不可达格 = 1.0）；
1. ``dist_end``：到我方防守终点的最短步行距离（同法，多源 = is_end 格）；
2. ``on_route``：是否在任意**地面**敌方行进路线格子集合内（0/1）。

步行路径定义（如实声明的简化）
--------------------------------
- 只消费 ``mode == WALK``（mode 编码 0，见 configs/tile_vocab.yaml）的路线；
  FLY（1）与未登记（-1，E_NUM 占位）一律排除；
- 格子集合 = 每条 WALK 路线的 ``start`` / ``end`` 端点 + 相邻 move waypoint 间的
  **纵向再横向** L 形栅格化中间格（Arknights 路线段实测基本都是轴对齐直线段，
  非轴对齐时 L 形是忠实近似，不声称逐像素还原寻路引擎轨迹）；
- BFS 被限制在该集合上（不在路线上的格子不构成可走边）；不在集合内的格
  dist 记 1.0、on_route 记 0。

无可用路线几何的关（路线条数 0，或没有任何带几何的 WALK 路线，或与 grid 尺寸
不符）：3 条通道全零，由调用方计数记录（不进单测断言的静默失败——调用方的
build_report 里要能看到）。
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ..protocol.action import GRID_CHANNELS

ROUTE_CHANNELS = 3
"""本模块派生的每格通道数（dist_start / dist_end / on_route）。"""

WALK_MODE = 0
"""routes.mode 的地面步行编码（configs/tile_vocab.yaml 的 mode_encoding.WALK）。"""


class RoutefeatError(RuntimeError):
    """路线几何与 grid 尺寸不符等口径错误。"""


def _rasterize_segment(r1: int, c1: int, r2: int, c2: int) -> list[tuple[int, int]]:
    """相邻 waypoint 间 L 形栅格化：先纵向走完行差再走横向列差（含两端点）。"""
    cells: list[tuple[int, int]] = []
    step_r = 1 if r2 >= r1 else -1
    for r in range(r1, r2 + step_r, step_r):
        cells.append((r, c1))
    step_c = 1 if c2 >= c1 else -1
    for c in range(c1 + step_c, c2 + step_c, step_c):
        cells.append((r2, c))
    return cells


def walk_route_cells(sf: dict[str, Any], rows: int, cols: int) -> set[tuple[int, int]]:
    """一关 stagefeat 行 → 全部 WALK 路线的行进格子集合（[r,c]，越界裁剪）。

    输入即 ``stagefeat.parquet`` 一行的 dict（``routes`` 字段可缺省 = 空集）。
    mode 编码见模块 docstring；只认 WALK。返回空集表示该关无可用路线几何，
    由调用方走「全零通道 + 计数」路径。
    """
    cells: set[tuple[int, int]] = set()

    def _put(r: int, c: int) -> None:
        if 0 <= int(r) < rows and 0 <= int(c) < cols:
            cells.add((int(r), int(c)))

    for route in sf.get("routes") or []:
        if route is None or int(route.get("mode", -1)) != WALK_MODE:
            continue
        start = route.get("start") or []
        end = route.get("end") or []
        if len(start) >= 2:
            _put(start[0], start[1])
        if len(end) >= 2:
            _put(end[0], end[1])
        pts = [int(x) for x in (route.get("points") or [])]
        waypoints = [(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
        if not waypoints:
            continue
        prev = waypoints[0]
        _put(*prev)
        for wp in waypoints[1:]:
            for cell in _rasterize_segment(prev[0], prev[1], wp[0], wp[1]):
                _put(*cell)
            prev = wp
    return cells


def _bfs_dists(
    sources: list[tuple[int, int]],
    walkable: set[tuple[int, int]],
    rows: int,
    cols: int,
) -> np.ndarray:
    """多源 4 连通 BFS（限制在 ``walkable`` 集合上）→ 距离矩阵（不可达 = inf）。"""
    dist = np.full((rows, cols), np.inf, dtype=np.float64)
    q: deque[tuple[int, int]] = deque()
    for r, c in sources:
        if (r, c) in walkable and np.isinf(dist[r, c]):
            dist[r, c] = 0.0
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in walkable and np.isinf(dist[nr, nc]):
                dist[nr, nc] = dist[r, c] + 1.0
                q.append((nr, nc))
    return dist


def _norm_dist(dist: np.ndarray) -> np.ndarray:
    """距离矩阵 → [0,1] 归一化：可达格除以最大可达距离，不可达格记 1.0。"""
    finite = np.isfinite(dist)
    if not finite.any():
        return np.zeros(dist.shape, dtype=np.float32)
    max_d = float(dist[finite].max())
    out = np.ones(dist.shape, dtype=np.float32)
    out[finite] = (dist[finite] / max(max_d, 1.0)).astype(np.float32)
    return out


def route_feature_channels(sf: dict[str, Any]) -> tuple[np.ndarray, bool]:
    """一关 stagefeat 行 → (每格 3 通道 float32 [rows*cols, 3], 有无可用路线几何)。

    - 无路线几何（含路线条数 0）→ 全零矩阵 + False（调用方计数记录）；
    - start/end 格取自 grid 编码（第 3/4 通道），与其 is_start/is_end 标记一致；
      起点/终点不在路线集合里时，把它们本身的格子追加进 BFS 可走域
      （地形端点配线是数据常态，例如 start 格不在折线上）；
    - ``grid`` 缺失或与 rows*cols*GRID_CHANNELS 尺寸不符 → :class:`RoutefeatError`。
    """
    rows, cols = int(sf["rows"]), int(sf["cols"])
    n_cells = rows * cols
    grid_flat = np.asarray(sf.get("grid") or [], dtype=np.float32)
    if grid_flat.size != n_cells * GRID_CHANNELS:
        raise RoutefeatError(
            f"{sf.get('stage_id')}: grid 长度 {grid_flat.size} 与 "
            f"{rows}×{cols}×{GRID_CHANNELS} 不符"
        )
    grid = grid_flat.reshape(n_cells, GRID_CHANNELS)

    walkable = walk_route_cells(sf, rows, cols)
    if not walkable:
        return np.zeros((n_cells, ROUTE_CHANNELS), dtype=np.float32), False

    starts = [
        (i // cols, i % cols) for i in range(n_cells) if grid[i, 3] > 0.5
    ]
    ends = [
        (i // cols, i % cols) for i in range(n_cells) if grid[i, 4] > 0.5
    ]
    # 端点不在折线上（常见于 start/end 只在地形端点配线时漏挂）时把端点并进去，
    # 否则 BFS 源全部 0 步终止，整路不可达
    walkable |= set(starts) | set(ends)

    d_start = _norm_dist(_bfs_dists(starts, walkable, rows, cols))
    d_end = _norm_dist(_bfs_dists(ends, walkable, rows, cols))
    on_route = np.zeros((rows, cols), dtype=np.float32)
    for r, c in walkable:
        on_route[r, c] = 1.0

    out = np.stack(
        [d_start.reshape(n_cells), d_end.reshape(n_cells), on_route.reshape(n_cells)],
        axis=1,
    )
    return out.astype(np.float32), True
