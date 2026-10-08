"""关卡注册表：stages.jsonl → stages_registry.parquet，含关卡家族 family 标注。

family_of() 规则（与数据探查验证一致：copilot 作业覆盖 102 个家族、main_00..main_17，
见 docs/adr/0002）：
1. ``^(main|tough|hard)_(\\d+)`` → ``main_<章节号>``：tough / hard 并入对应主线章节
   （例：tough_05-11 → main_05，hard_12-03 → main_12）；
2. ``^act[0-9a-z]+`` → 该前缀本身（例：act23side_06 → act23side，act1break_01 → act1break）；
3. ``^(camp|wk|sub|pro|tr|a|rogue)`` 前缀 → 该 token（例：a003_02 → a——
   a001/a003 剿灭轮换合并为同一家族；tr_10 → tr）；
4. 其余 → 第一个 ``_`` 之前的分段（例：lt_tr_01 → lt）。
规则按 1→4 依次尝试，命中即返回。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_MAIN_RE = re.compile(r"^(main|tough|hard)_(\d+)")
_ACT_RE = re.compile(r"^act[0-9a-z]+")
_TOKEN_RE = re.compile(r"^(camp|wk|sub|pro|tr|a|rogue)")


def family_of(stage_id: str) -> str:
    """关卡家族名：规则见模块 docstring（按 1→4 依次匹配）。"""
    m = _MAIN_RE.match(stage_id)
    if m:
        return f"main_{m.group(2)}"
    m = _ACT_RE.match(stage_id)
    if m:
        return m.group(0)
    m = _TOKEN_RE.match(stage_id)
    if m:
        return m.group(1)
    return stage_id.split("_", 1)[0]


@dataclass(frozen=True)
class StagesBuildResult:
    """stages 注册表构建结果统计。"""

    n_stages: int
    n_families: int
    out_path: Path


def _tile_counts(grid: list[list[dict[str, object]]]) -> dict[str, int]:
    """按格子 build 属性统计可部署地面 / 高台格数（build=MELEE/RANGED/NONE）。"""
    melee = ranged = forbidden = 0
    for row in grid:
        for cell in row:
            key = str(cell.get("key", ""))
            build = str(cell.get("build", ""))
            if build == "MELEE":
                melee += 1
            elif build == "RANGED":
                ranged += 1
            if key == "tile_forbidden":
                forbidden += 1
    return {"melee_tiles": melee, "ranged_tiles": ranged, "forbidden_tiles": forbidden}


def build_registry(stages_jsonl: Path, out: Path) -> StagesBuildResult:
    """读取 stages.jsonl，写出注册表 parquet（一关一行）。"""
    rows: list[dict[str, object]] = []
    with stages_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            grid = s.get("grid") or []
            spawns = s.get("spawns") or []
            options = s.get("options") or {}
            counts = _tile_counts(grid)
            rows.append(
                {
                    "stage_id": str(s["stage_id"]),
                    "code": str(s.get("code", "")),
                    "name": str(s.get("name", "")),
                    "type": str(s.get("type", "")),
                    "difficulty": str(s.get("difficulty", "")),
                    "level_id": str(s.get("level_id", "")),
                    "family": family_of(str(s["stage_id"])),
                    "rows": int(s.get("rows") or 0),
                    "cols": int(s.get("cols") or 0),
                    **counts,
                    "n_routes": len(s.get("routes") or []),
                    "n_spawns": len(spawns),
                    "n_distinct_enemies": len({str(sp.get("enemy", "")) for sp in spawns}),
                    "opt_character_limit": int(options.get("characterLimit") or 0),
                    "opt_max_life_point": int(options.get("maxLifePoint") or 0),
                    "opt_initial_cost": int(options.get("initialCost") or 0),
                    "opt_max_cost": int(options.get("maxCost") or 0),
                    "opt_cost_increase_time": float(options.get("costIncreaseTime") or 0.0),
                }
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out)
    return StagesBuildResult(
        n_stages=len(rows),
        n_families=len({r["family"] for r in rows}),
        out_path=out,
    )


def load_stage_index(stages_jsonl: Path) -> dict[str, dict[str, str]]:
    """供 episodes 转换使用的 stage_id → 基本元信息索引（family/code/name/type/difficulty）。"""
    index: dict[str, dict[str, str]] = {}
    with stages_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            sid = str(s["stage_id"])
            index[sid] = {
                "stage_id": sid,
                "code": str(s.get("code", "")),
                "name": str(s.get("name", "")),
                "type": str(s.get("type", "")),
                "difficulty": str(s.get("difficulty", "")),
                "family": family_of(sid),
            }
    return index
