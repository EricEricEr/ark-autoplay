"""stagefeat：关卡静态特征转换（stages.jsonl → 模型可消费的关卡特征 parquet）。

背景与动机
----------
stages.jsonl（prts.plus 静态关卡镜像，3279 关）已含完整的地形 / 路线 / 出怪表，
但它是**面向人读的嵌套 JSON**：格子是字符串 token、路线是变长 checkpoint 列表、
出怪是 (t, enemy, route) 三元组。模型需要的是整型、可直接张量化的表示。

本模块把每关转成一行扁平记录，三个核心张量列：

- ``grid``  ：row-major 展平的格子矩阵，每格 ``GRID_CHANNELS`` 个整型通道；
- ``routes``：每条路线的几何（各自变长，配套 ``n_points`` / ``wait_ms``）；
- ``spawns``：出怪时间表（t 升序），含敌人解析结果。

变长而非 padding 到定长
-----------------------
实测网格尺寸 6×9 ~ 40×40（常见 8×11，最大 1600 格），若一律补齐到 40×40
会浪费约 3 倍空间且引入大量 padding 噪音。故 grid 按 ``rows × cols`` 原样展平，
消费方按行内自带的 ``rows`` / ``cols`` 自行 reshape；路线同理按条变长。

坐标系统一（重要）
------------------
真实数据里存在**两套坐标系**，实测证据：

- ``grid`` / ``routes`` 的 pos 是 ``[row, col]``：route.start 按 [row,col] 解读时
  3990 次命中 tile_start，按 [col,row] 只有 342 次；
- MAA copilot 作业的 deploy ``location`` 是 ``[x, y] = [col, row]``：按此解读
  96.8% 落在可部署格上，按 [row,col] 只有 41.7%（28076 条真实 deploy 实测）。

本模块统一输出 ``[row, col]``（与 GameData 原生一致），并提供
:func:`normalize_deploy_location` 做 deploy 坐标的显式转换。混用这两套坐标是
ReviewBench rubric「Correctness」类的高危缺陷，因此转换必须显式、单向、可测
（见 tests/unit/test_datapipe_stagefeat.py）。

数据质量口径（实测，如实记录）
------------------------------
- 网格严格矩形（3279 关无一例外）；
- ``routes`` 含 758 个 ``null`` 占位元素，**从不被 spawns 引用**（实测 0 次），跳过并计数；
- ``spawns`` 只引用 WALK / FLY 两类路线（176993 条全覆盖），E_NUM 是纯占位；
- spawns 的 enemy id 直接命中 featvec enemy_id 175872/176993（99.4%）；
- 关卡变体（``#f#`` 四星 775 / ``#s`` 六星 45）与主关卡共用 level_id，grid / routes /
  spawns / options **完全一致**（888 组变体实测 0 差异），默认不重复入表；
- checkpoint 有字符串与数字两套类型码，数字码语义由形态交叉验证判定
  （见 configs/tile_vocab.yaml 的 checkpoint_kind 与 ADR-0003）。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .sources import DataConfig, ResolvedSources, out_path

STAGEFEAT_VERSION = "0.1.0"

GRID_CHANNELS = 5
"""每格通道：h / build / deployable / is_start / is_end。"""

GRID_UNKNOWN = -1
"""未知编码值（未在词表登记的 h / build / mode）。"""

DEFAULT_MAX_GRID = 64
"""网格边长上限（实测最大 40×40；超出报错而非静默截断）。"""

DEFAULT_MAX_ROUTE_POINTS = 256
"""单条路线路径点上限（实测最大 132；超出报错）。"""

_STRUCT_FLAGS = ("start", "fly_start", "end")


class StagefeatError(RuntimeError):
    """关卡特征口径错误（数据不符合已验证的不变式）。"""


@dataclass(frozen=True)
class TileVocab:
    """地形 / 路线编码表（configs/tile_vocab.yaml）。"""

    numeric_h: dict[str, str]
    numeric_build: dict[str, str]
    numeric_mode: dict[str, str]
    h_encoding: dict[str, int]
    build_encoding: dict[str, int]
    mode_encoding: dict[str, int]
    structural_keys: dict[str, list[str]]
    checkpoint_kind: dict[str, str]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> TileVocab:
        """读取编码表；缺失或字段不全时报错。"""
        p = (path or default_tile_vocab_path()).expanduser().resolve()
        if not p.is_file():
            raise StagefeatError(f"找不到地形编码表：{p}")
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        sec = raw.get("stagefeat") or {}
        if not isinstance(sec, dict):
            raise StagefeatError(f"{p} 的 stagefeat 小节必须是 mapping。")
        need = (
            "numeric_h",
            "numeric_build",
            "numeric_mode",
            "h_encoding",
            "build_encoding",
            "mode_encoding",
            "structural_keys",
            "checkpoint_kind",
        )
        miss = [k for k in need if k not in sec]
        if miss:
            raise StagefeatError(f"{p} 的 stagefeat 缺少条目：{', '.join(miss)}")
        return cls(
            numeric_h={str(k): str(v) for k, v in (sec["numeric_h"] or {}).items()},
            numeric_build={str(k): str(v) for k, v in (sec["numeric_build"] or {}).items()},
            numeric_mode={str(k): str(v) for k, v in (sec["numeric_mode"] or {}).items()},
            h_encoding={str(k): int(v) for k, v in (sec["h_encoding"] or {}).items()},
            build_encoding={str(k): int(v) for k, v in (sec["build_encoding"] or {}).items()},
            mode_encoding={str(k): int(v) for k, v in (sec["mode_encoding"] or {}).items()},
            structural_keys={
                str(k): [str(x) for x in (v or [])]
                for k, v in (sec["structural_keys"] or {}).items()
            },
            checkpoint_kind={
                str(k): str(v) for k, v in (sec["checkpoint_kind"] or {}).items()
            },
            raw=sec,
        )

    # ---- 归一化：把数字码 / 字符串统一成语义名 ----

    def norm_h(self, value: Any) -> str:
        """格子高度 → LOWLAND / HIGHLAND（数字码按表翻译）。"""
        return self.numeric_h.get(str(value), str(value))

    def norm_build(self, value: Any) -> str:
        """格子可部署类型 → NONE / MELEE / RANGED / ALL。"""
        return self.numeric_build.get(str(value), str(value))

    def h_code(self, value: Any) -> int:
        """格子高度 → 整型编码（未登记返回 ``GRID_UNKNOWN``）。"""
        return self.h_encoding.get(self.norm_h(value), GRID_UNKNOWN)

    def build_code(self, value: Any) -> int:
        """格子类型 → 整型编码（未登记返回 ``GRID_UNKNOWN``）。"""
        return self.build_encoding.get(self.norm_build(value), GRID_UNKNOWN)

    def norm_mode(self, value: Any) -> str:
        """路线模式 → WALK / FLY（数字码按表翻译）。"""
        return self.numeric_mode.get(str(value), str(value))

    def mode_code(self, value: Any) -> int:
        """路线模式 → 整型编码（未登记返回 ``GRID_UNKNOWN``）。

        必须先过 :meth:`norm_mode`：实测 232 个全数字编码关卡里 mode 写作 '0'/'1'，
        不翻译会全部落到 UNKNOWN，路线类型就丢了。
        """
        return self.mode_encoding.get(self.norm_mode(value), GRID_UNKNOWN)

    def deployable(self, value: Any) -> int:
        """该格是否可部署干员（build ∈ {MELEE, RANGED, ALL}）。"""
        return int(self.norm_build(value) in ("MELEE", "RANGED", "ALL"))

    def structural_flag(self, key: Any, name: str) -> int:
        """tile key 是否命中结构化标记（start / end / hole / fly_start / impassable）。"""
        return int(str(key) in set(self.structural_keys.get(name, [])))

    def kind_of_checkpoint(self, value: Any) -> str:
        """checkpoint 类型 → move / wait / other（未登记按 other 处理并计数）。"""
        return self.checkpoint_kind.get(str(value), "other")


def default_tile_vocab_path() -> Path:
    """默认编码表路径：<repo>/configs/tile_vocab.yaml。"""
    return Path(__file__).resolve().parents[3] / "configs" / "tile_vocab.yaml"


def normalize_deploy_location(location: list[int] | tuple[int, int]) -> tuple[int, int]:
    """MAA 作业 deploy 的 ``[x, y]`` → 本模块统一的 ``[row, col]``。

    实测依据：28076 条真实 deploy 中，按 [x,y]=[col,row] 解读有 96.8% 落在可部署格，
    按 [row,col] 只有 41.7%。凡消费 episodes 里 deploy 位置的代码都必须先过这里。
    """
    if len(location) < 2:
        raise StagefeatError(f"deploy location 需要两个分量，收到 {location!r}")
    x, y = int(location[0]), int(location[1])
    return y, x


@dataclass
class StagefeatStats:
    """构建期计数（进 build_report）。"""

    n_stages: int = 0
    n_skipped_variant: int = 0
    n_null_routes: int = 0
    n_routes: int = 0
    n_spawns: int = 0
    n_grid_cells: int = 0
    max_grid_rows: int = 0
    max_grid_cols: int = 0
    max_route_points: int = 0
    unknown_h_codes: Counter = field(default_factory=Counter)
    unknown_build_codes: Counter = field(default_factory=Counter)
    unknown_mode_codes: Counter = field(default_factory=Counter)
    unknown_checkpoint_types: Counter = field(default_factory=Counter)
    spawn_enemy_unmatched: Counter = field(default_factory=Counter)


@dataclass(frozen=True)
class StagefeatBuildResult:
    """产物路径与统计。"""

    n_stages: int
    out_path: Path
    stats: StagefeatStats


def is_variant(stage_id: str) -> bool:
    """关卡变体判定：``main_00-01#f#``（四星）/ ``xxx#s``（六星）。"""
    return "#" in stage_id


def _encode_grid(
    grid: list[list[dict[str, Any]]],
    vocab: TileVocab,
    stats: StagefeatStats,
    max_rows: int,
    max_cols: int,
) -> list[int]:
    """二维格子矩阵 → row-major 展平整型数组（长度 ``rows*cols*GRID_CHANNELS``）。

    每格通道顺序：``h`` / ``build`` / ``deployable`` / ``is_start`` / ``is_end``。
    非矩形网格报错（实测 3279 关全部矩形）；超上限报错而非静默截断。
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if any(len(r) != cols for r in grid):
        raise StagefeatError("grid 非矩形：本模块要求每行等长（实测数据全部满足）")
    if rows > max_rows or cols > max_cols:
        raise StagefeatError(
            f"网格 {rows}×{cols} 超出上限 {max_rows}×{max_cols}；"
            "请调大 configs/data.yaml 的 stagefeat.max_grid_rows/cols，不要静默截断"
        )

    out: list[int] = []
    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c] or {}
            h = vocab.h_code(cell.get("h"))
            b = vocab.build_code(cell.get("build"))
            if h == GRID_UNKNOWN:
                stats.unknown_h_codes[str(cell.get("h"))] += 1
            if b == GRID_UNKNOWN:
                stats.unknown_build_codes[str(cell.get("build"))] += 1
            key = cell.get("key")
            is_start = vocab.structural_flag(key, "start") | vocab.structural_flag(key, "fly_start")
            is_end = vocab.structural_flag(key, "end")
            out.extend([h, b, vocab.deployable(cell.get("build")), is_start, is_end])
    stats.n_grid_cells += rows * cols
    return out


def _encode_route(
    route: dict[str, Any],
    vocab: TileVocab,
    stats: StagefeatStats,
    max_points: int,
) -> dict[str, Any]:
    """单条路线 → 几何摘要。

    只把 **move 类** checkpoint 的 ``pos`` 计入路径（wait 类的 pos 是 [0,0] 占位，
    计入会污染几何）；wait 类的时长单独收集到 ``wait_ms``。
    """
    mode_code = vocab.mode_code(route.get("mode"))
    if mode_code == GRID_UNKNOWN:
        stats.unknown_mode_codes[str(route.get("mode"))] += 1

    points: list[int] = []
    wait_ms: list[int] = []
    for cp in route.get("checkpoints") or []:
        raw_type = cp.get("type")
        kind = vocab.kind_of_checkpoint(raw_type)
        if kind == "move":
            pos = cp.get("pos") or [0, 0]
            points.extend([int(pos[0]), int(pos[1])])
        elif kind == "wait":
            wait_ms.append(int(round(float(cp.get("time") or 0.0) * 1000)))
        elif str(raw_type) not in (vocab.checkpoint_kind or {}):
            stats.unknown_checkpoint_types[str(raw_type)] += 1

    n_points = len(points) // 2
    if n_points > max_points:
        raise StagefeatError(
            f"路线路径点 {n_points} 超出上限 {max_points}；"
            "请调大 configs/data.yaml 的 stagefeat.max_route_points，不要静默截断"
        )
    stats.max_route_points = max(stats.max_route_points, n_points)

    start = route.get("start") or [0, 0]
    end = route.get("end") or [0, 0]
    return {
        "mode": mode_code,
        "start": [int(start[0]), int(start[1])],
        "end": [int(end[0]), int(end[1])],
        "n_points": n_points,
        "points": points,
        "wait_ms": wait_ms,
    }


def build_stagefeat(
    cfg: DataConfig,
    resolved: ResolvedSources,
    out_dir: Path,
    *,
    vocab: TileVocab | None = None,
    include_variants: bool = False,
) -> StagefeatBuildResult:
    """构建关卡静态特征 parquet（一关一行）。

    ``include_variants=False`` 时跳过 ``#f#`` / ``#s`` 变体（它们与主关卡共用
    level_id 且 grid/routes/spawns/options 实测完全一致，重复入表只会放大权重）。
    """
    section = dict(cfg.raw.get("stagefeat") or {})
    max_rows = int(section.get("max_grid_rows", DEFAULT_MAX_GRID))
    max_cols = int(section.get("max_grid_cols", DEFAULT_MAX_GRID))
    max_points = int(section.get("max_route_points", DEFAULT_MAX_ROUTE_POINTS))
    vocab = vocab or TileVocab.load()

    # 敌人 id 白名单：spawns 的 enemy 直接对齐 featvec 的 enemy_id（实测 99.4%）
    enemy_ids: set[str] = set()
    enemy_parquet = out_path(resolved, "featvec_enemies_parquet")
    if enemy_parquet.is_file():
        t = pq.read_table(enemy_parquet, columns=["enemy_id"])
        enemy_ids = {str(x) for x in t.column("enemy_id").to_pylist()}

    stats = StagefeatStats()
    rows: list[dict[str, Any]] = []
    stages_jsonl = resolved.require("stages_jsonl")

    with stages_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            sid = str(s.get("stage_id") or "")
            if is_variant(sid) and not include_variants:
                stats.n_skipped_variant += 1
                continue

            grid = s.get("grid") or []
            n_rows = len(grid)
            n_cols = len(grid[0]) if n_rows else 0
            stats.max_grid_rows = max(stats.max_grid_rows, n_rows)
            stats.max_grid_cols = max(stats.max_grid_cols, n_cols)

            routes: list[dict[str, Any]] = []
            for r in s.get("routes") or []:
                if r is None:
                    stats.n_null_routes += 1
                    continue
                routes.append(_encode_route(r, vocab, stats, max_points))
            stats.n_routes += len(routes)

            spawns: list[dict[str, Any]] = []
            for sp in s.get("spawns") or []:
                eid = str(sp.get("enemy") or "")
                matched = (not enemy_ids) or (eid in enemy_ids)
                if not matched:
                    stats.spawn_enemy_unmatched[eid] += 1
                spawns.append(
                    {
                        "t_ms": int(round(float(sp.get("t") or 0.0) * 1000)),
                        "wave": int(sp.get("wave") or 0),
                        "route": int(sp.get("route") or 0),
                        "enemy_id": eid,
                        "enemy_matched": int(matched),
                    }
                )
            spawns.sort(key=lambda x: x["t_ms"])
            stats.n_spawns += len(spawns)

            options = s.get("options") or {}
            rows.append(
                {
                    "stage_id": sid,
                    "code": str(s.get("code") or ""),
                    "name": str(s.get("name") or ""),
                    "type": str(s.get("type") or ""),
                    "difficulty": str(s.get("difficulty") or ""),
                    "level_id": str(s.get("level_id") or ""),
                    "is_variant": int(is_variant(sid)),
                    "rows": n_rows,
                    "cols": n_cols,
                    "n_routes": len(routes),
                    "n_spawns": len(spawns),
                    "opt_character_limit": int(options.get("characterLimit") or 0),
                    "opt_max_life_point": int(options.get("maxLifePoint") or 0),
                    "opt_initial_cost": int(options.get("initialCost") or 0),
                    "opt_max_cost": int(options.get("maxCost") or 0),
                    "opt_cost_increase_time": float(options.get("costIncreaseTime") or 0.0),
                    "grid": _encode_grid(grid, vocab, stats, max_rows, max_cols),
                    "routes": routes,
                    "spawns": spawns,
                    "stagefeat_version": STAGEFEAT_VERSION,
                }
            )
            stats.n_stages += 1

    if not rows:
        raise StagefeatError(f"未从 {stages_jsonl} 读到任何关卡")

    out = out_path(resolved, "stagefeat_parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out)
    return StagefeatBuildResult(n_stages=len(rows), out_path=out, stats=stats)
