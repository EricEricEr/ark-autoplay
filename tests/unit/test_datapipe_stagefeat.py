"""stagefeat 单测：合成数据，不含任何真实游戏内容。

覆盖三件必须钉死的事：
1. 坐标序：deploy 的 [x,y] → [row,col] 转换（错则全盘部署落错格）；
2. 编码：数字码 / 字符串码两套写法归一后必须得到同一结果；
3. checkpoint 分类：wait 类的 [0,0] 占位不得混入路径几何。
"""

from __future__ import annotations

import json

import pytest

from ark_core.datapipe.stagefeat import (
    GRID_CHANNELS,
    GRID_UNKNOWN,
    StagefeatError,
    TileVocab,
    build_stagefeat,
    is_variant,
    normalize_deploy_location,
)


@pytest.fixture
def vocab(tmp_path):
    """最小编码表（口径与 configs/tile_vocab.yaml 一致）。"""
    p = tmp_path / "tile_vocab.yaml"
    p.write_text(
        """
stagefeat:
  numeric_h: {"0": LOWLAND, "1": HIGHLAND}
  numeric_build: {"0": NONE, "1": MELEE, "2": RANGED, "3": ALL}
  numeric_mode: {"0": WALK, "1": FLY}
  h_encoding: {LOWLAND: 0, HIGHLAND: 1}
  build_encoding: {NONE: 0, MELEE: 1, RANGED: 2, ALL: 3}
  mode_encoding: {WALK: 0, FLY: 1}
  structural_keys:
    impassable: [tile_forbidden]
    start: [tile_start]
    end: [tile_end]
    hole: [tile_hole]
    fly_start: [tile_flystart]
  checkpoint_kind:
    MOVE: move
    PATROL_MOVE: move
    APPEAR_AT_POS: move
    MAP_OFFSET_MOVE: move
    WAIT_FOR_SECONDS: wait
    WAIT_CURRENT_FRAGMENT_TIME: wait
    WAIT_CURRENT_WAVE_TIME: wait
    WAIT_BOSSRUSH_WAVE: wait
    DISAPPEAR: other
    "0": move
    "6": move
    "1": wait
    "3": wait
    "4": wait
    "5": other
""",
        encoding="utf-8",
    )
    return TileVocab.load(p)


# ---------------- 1. 坐标序 ----------------


def test_deploy_location_is_x_y_and_converts_to_row_col():
    """deploy 的 [x, y] 必须翻成 [row, col]——这是本模块最危险的转换。"""
    assert normalize_deploy_location([4, 7]) == (7, 4)
    assert normalize_deploy_location((0, 0)) == (0, 0)
    assert normalize_deploy_location([7, 3]) == (3, 7)


def test_deploy_location_rejects_short_input():
    with pytest.raises(StagefeatError):
        normalize_deploy_location([3])


# ---------------- 2. 两套编码写法必须归一 ----------------


def test_numeric_and_string_codes_normalize_identically(vocab):
    """同一语义的字符串码与数字码，编码后必须相等（实测数据两套写法并存）。"""
    assert vocab.h_code("HIGHLAND") == vocab.h_code("1") == 1
    assert vocab.h_code("LOWLAND") == vocab.h_code("0") == 0
    assert vocab.build_code("MELEE") == vocab.build_code("1") == 1
    assert vocab.build_code("RANGED") == vocab.build_code("2") == 2
    assert vocab.build_code("NONE") == vocab.build_code("0") == 0
    assert vocab.build_code("ALL") == vocab.build_code("3") == 3
    assert vocab.mode_code("WALK") == vocab.mode_code("0") == 0
    assert vocab.mode_code("FLY") == vocab.mode_code("1") == 1


def test_deployable_only_for_buildable(vocab):
    """可部署 = build ∈ {MELEE, RANGED, ALL}；NONE 与数字 '0' 都不可部署。"""
    assert vocab.deployable("MELEE") == 1
    assert vocab.deployable("1") == 1
    assert vocab.deployable("RANGED") == 1
    assert vocab.deployable("ALL") == 1
    assert vocab.deployable("NONE") == 0
    assert vocab.deployable("0") == 0


def test_unknown_code_is_flagged_not_defaulted(vocab):
    """未登记的编码必须返回 GRID_UNKNOWN，不得静默当成 0（那会伪造地形）。"""
    assert vocab.h_code("SOMETHING_NEW") == GRID_UNKNOWN
    assert vocab.build_code("99") == GRID_UNKNOWN
    assert vocab.mode_code("TELEPORT") == GRID_UNKNOWN


# ---------------- 3. checkpoint 分类 ----------------


def test_wait_checkpoints_do_not_pollute_path_geometry(vocab):
    """wait 类的 pos 是 [0,0] 占位；若计入路径会把几何拖到左上角。"""
    assert vocab.kind_of_checkpoint("WAIT_FOR_SECONDS") == "wait"
    assert vocab.kind_of_checkpoint("1") == "wait"
    assert vocab.kind_of_checkpoint("MOVE") == "move"
    assert vocab.kind_of_checkpoint("0") == "move"
    assert vocab.kind_of_checkpoint("DISAPPEAR") == "other"
    assert vocab.kind_of_checkpoint("5") == "other"
    # 未登记类型保守归 other，不得当成 move
    assert vocab.kind_of_checkpoint("BRAND_NEW_TYPE") == "other"


# ---------------- 4. 端到端（合成关卡） ----------------


def _write_sources(tmp_path, stages: list[dict]):
    """写出合成 stages.jsonl 与空的 featvec 目录，返回 (cfg, resolved)。"""
    from ark_core.datapipe.sources import DataConfig, ResolvedSources

    stages_p = tmp_path / "stages.jsonl"
    with stages_p.open("w", encoding="utf-8") as f:
        for s in stages:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    cfg = DataConfig(
        config_path=tmp_path / "data.yaml",
        data_version="test",
        data_root=tmp_path,
        sources={},
        optional_sources=frozenset(),
        outputs={"stagefeat_parquet": "stagefeat.parquet"},
        featvec={},
        episodes={},
        raw={"stagefeat": {}},
    )
    resolved = ResolvedSources(
        config=cfg,
        paths={"stages_jsonl": stages_p},
        missing={},
        out_dir=tmp_path,
    )
    return cfg, resolved


def _synth_stage(stage_id: str, rows: int = 2, cols: int = 3) -> dict:
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            if r == 0 and c == 0:
                row.append({"key": "tile_start", "h": "LOWLAND", "build": "NONE"})
            elif r == 0 and c == cols - 1:
                row.append({"key": "tile_end", "h": "LOWLAND", "build": "NONE"})
            elif r == 1:
                row.append({"key": "tile_road", "h": "LOWLAND", "build": "MELEE"})
            else:
                row.append({"key": "tile_wall", "h": "HIGHLAND", "build": "RANGED"})
        grid.append(row)
    return {
        "stage_id": stage_id,
        "code": "0-1",
        "name": "synth",
        "type": "MAIN",
        "difficulty": "NORMAL",
        "level_id": "obt/main/level_synth",
        "options": {
            "characterLimit": 8,
            "maxLifePoint": 20,
            "initialCost": 10,
            "maxCost": 99,
            "costIncreaseTime": 1.0,
        },
        "rows": rows,
        "cols": cols,
        "grid": grid,
        "routes": [
            {
                "mode": "WALK",
                "start": [0, 0],
                "end": [0, cols - 1],
                "checkpoints": [
                    {"type": "MOVE", "pos": [1, 1], "time": 0.0},
                    {"type": "WAIT_FOR_SECONDS", "pos": [0, 0], "time": 3.0},
                    {"type": "MOVE", "pos": [0, 2], "time": 0.0},
                ],
            },
            None,  # 实测存在 null 占位，必须跳过并计数
        ],
        "spawns": [
            {"t": 5.0, "wave": 0, "enemy": "enemy_x", "route": 0},
            {"t": 2.0, "wave": 0, "enemy": "enemy_x", "route": 0},
        ],
    }


def test_build_stagefeat_end_to_end(tmp_path, vocab):
    stage = _synth_stage("main_00-01")
    variant = _synth_stage("main_00-01#f#")
    cfg, resolved = _write_sources(tmp_path, [stage, variant])

    res = build_stagefeat(cfg, resolved, tmp_path, vocab=vocab)
    assert res.n_stages == 1, "变体应被跳过"
    assert res.stats.n_skipped_variant == 1
    assert res.stats.n_null_routes == 1, "null 占位路线应被计数"
    assert res.stats.n_routes == 1

    import pyarrow.parquet as pq

    t = pq.read_table(res.out_path)
    row = t.to_pylist()[0]

    # 网格通道数 = rows*cols*GRID_CHANNELS
    assert len(row["grid"]) == row["rows"] * row["cols"] * GRID_CHANNELS
    # (0,0) 是 tile_start：h=LOWLAND=0, build=NONE=0, deployable=0, is_start=1
    assert row["grid"][0:5] == [0, 0, 0, 1, 0]
    # (1,0) 是 tile_road：build=MELEE=1, deployable=1
    idx = (1 * row["cols"] + 0) * GRID_CHANNELS
    assert row["grid"][idx + 1] == 1
    assert row["grid"][idx + 2] == 1
    # (0,2) 是 tile_end：is_end=1
    idx_end = (0 * row["cols"] + 2) * GRID_CHANNELS
    assert row["grid"][idx_end + 4] == 1

    # 路线：只有 2 个 move 点，wait 的 [0,0] 不得进 points
    route = row["routes"][0]
    assert route["n_points"] == 2
    assert route["points"] == [1, 1, 0, 2]
    assert route["wait_ms"] == [3000]
    assert route["mode"] == 0

    # 出怪按 t 升序
    assert [s["t_ms"] for s in row["spawns"]] == [2000, 5000]


def test_variant_detection():
    assert is_variant("main_00-01#f#")
    assert is_variant("main_00-01#s")
    assert not is_variant("main_00-01")


def test_grid_ragged_rejected(tmp_path, vocab):
    """非矩形网格必须报错——静默容忍会让行列索引全错位。"""
    stage = _synth_stage("main_00-01")
    stage["grid"][0].pop()
    cfg, resolved = _write_sources(tmp_path, [stage])
    with pytest.raises(StagefeatError, match="非矩形"):
        build_stagefeat(cfg, resolved, tmp_path, vocab=vocab)


def test_grid_over_limit_rejected_not_truncated(tmp_path, vocab):
    """超上限必须报错而非静默截断（截断会丢地形且不留痕）。"""
    stage = _synth_stage("main_00-01")
    cfg, resolved = _write_sources(tmp_path, [stage])
    cfg2 = cfg
    cfg2.raw["stagefeat"] = {"max_grid_rows": 1, "max_grid_cols": 1}
    with pytest.raises(StagefeatError, match="超出上限"):
        build_stagefeat(cfg2, resolved, tmp_path, vocab=vocab)
