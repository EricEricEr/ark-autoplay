"""数值特征向量（featvec）：官方数值表 → 干员 / 敌人 parquet + 版本化 schema JSON。

设计要点（与设计文档 §6.1 对齐：干员 / 敌人以数值特征向量标识，不用身份 embedding）：
- 干员：character_table 过滤出玩家干员（8 大职业 + TIER_1..6），取最后精英化阶段满级
  白值；profession / subProfessionId / position one-hot + 归一化数值 + 技能关键词标签
  （skill_table 技能名 + 描述的子串启发式；TODO(M1): 换人工整理的技能语义标签表）
  + 是否拥有非 ORIGINAL 模组（uniequip_table）。
- 敌人：enemy_database 每个 (enemy_id, level) 变体一行（levelType NORMAL/ELITE/BOSS），
  motion / applyWay / levelType one-hot + 归一化数值；name / tags 作元信息列。
- 归一化尺度全部来自 configs/data.yaml 的 featvec.*_stat_scales（超参不入代码）。
- 随 parquet 写 featvec_schema.json：featvec_version + 列清单（角色 / dtype），
  下游按 version 消费，列变更必须升 FEATVEC_VERSION。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .sources import DataConfig, ResolvedSources, out_path

FEATVEC_VERSION = "0.1.0"

# 8 个玩家职业（character_table 里大量 TRAP/TOKEN 非干员条目按此过滤）
PLAY_PROFESSIONS: tuple[str, ...] = (
    "PIONEER",
    "WARRIOR",
    "TANK",
    "SNIPER",
    "CASTER",
    "MEDIC",
    "SUPPORT",
    "SPECIAL",
)

# 干员版块的数值属性（character_table attributesKeyFrames.data 里的键）
OP_STAT_KEYS: tuple[str, ...] = (
    "maxHp",
    "atk",
    "def",
    "magicResistance",
    "cost",
    "blockCnt",
    "baseAttackTime",
    "attackSpeed",
    "respawnTime",
)

# 敌人版块的数值属性（enemy_database attributes 里的键）
ENEMY_STAT_KEYS: tuple[str, ...] = (
    "maxHp",
    "atk",
    "def",
    "magicResistance",
    "moveSpeed",
    "attackSpeed",
    "baseAttackTime",
    "massLevel",
    "rangeRadius",
    "lifePointReduce",
)

_ENEMY_LEVEL_TYPES = ("NORMAL", "ELITE", "BOSS")
_ENEMY_MOTIONS = ("WALK", "FLY")
_ENEMY_APPLY_WAYS = ("MELEE", "RANGED", "ALL", "NONE")


@dataclass(frozen=True)
class FeatvecBuildResult:
    """featvec 构建统计（进 build_report）。"""

    n_operators: int
    n_enemy_rows: int
    op_vector_columns: list[str] = field(repr=False)
    enemy_vector_columns: list[str] = field(repr=False)
    op_out: Path
    enemy_out: Path
    schema_out: Path


def _m(node: dict[str, Any] | None) -> Any:
    """enemy_database 的 {m_defined, m_value} 包装取值。"""
    if not isinstance(node, dict):
        return None
    return node.get("m_value")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _skill_tag_columns(
    skills: list[dict[str, Any]],
    skill_table: dict[str, Any],
    keywords: list[str],
) -> dict[str, int]:
    """对单个干员：扫全部技能最后一级的名称 + 描述，做关键词子串打标。"""
    tags = dict.fromkeys(keywords, 0)
    for sk in skills:
        sid = sk.get("skillId")
        entry = skill_table.get(str(sid)) if sid else None
        if not entry:
            continue
        levels = entry.get("levels") or []
        if not levels:
            continue
        last = levels[-1]
        text = str(last.get("name", "")) + "\n" + str(last.get("description", ""))
        for kw in keywords:
            if kw in text:
                tags[kw] = 1
    return tags


def _build_operator_rows(
    character_table: dict[str, Any],
    skill_table: dict[str, Any],
    char_equip: dict[str, Any],
    equip_dict: dict[str, Any],
    scales: dict[str, float],
    keywords: list[str],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """生成干员行与列清单（向量列 + 元信息列）。"""
    subs = sorted(
        {
            str(c.get("subProfessionId") or "none")
            for c in character_table.values()
            if c.get("profession") in PLAY_PROFESSIONS
        }
    )
    prof_cols = [f"op_prof_{p.lower()}" for p in PLAY_PROFESSIONS]
    sub_cols = [f"op_sub_{s}" for s in subs]
    pos_cols = ["op_pos_melee", "op_pos_ranged"]
    stat_cols = [f"op_stat_{k}" for k in OP_STAT_KEYS] + ["op_stat_rarity_tier"]
    tag_cols = [f"op_tag_{kw}" for kw in keywords]
    vector_columns = prof_cols + sub_cols + pos_cols + stat_cols + tag_cols

    rows: list[dict[str, Any]] = []
    for char_id, c in character_table.items():
        if c.get("profession") not in PLAY_PROFESSIONS:
            continue
        rarity = str(c.get("rarity", ""))
        if not rarity.startswith("TIER_"):
            continue
        tier = int(rarity.split("_", 1)[1])
        phases = c.get("phases") or []
        if not phases:
            continue
        frames = phases[-1].get("attributesKeyFrames") or []
        if not frames:
            continue
        data = frames[-1].get("data") or {}

        row: dict[str, Any] = {
            "char_id": char_id,
            "name": str(c.get("name", "")),
            "rarity_tier": tier,
            "sub_profession": str(c.get("subProfessionId") or "none"),
        }
        for p, col in zip(PLAY_PROFESSIONS, prof_cols, strict=True):
            row[col] = int(c.get("profession") == p)
        for s, col in zip(subs, sub_cols, strict=True):
            row[col] = int(row["sub_profession"] == s)
        row["op_pos_melee"] = int(c.get("position") == "MELEE")
        row["op_pos_ranged"] = int(c.get("position") == "RANGED")
        for k, col in zip(OP_STAT_KEYS, stat_cols[: len(OP_STAT_KEYS)], strict=True):
            scale = float(scales.get(k, 1.0)) or 1.0
            row[col] = float(data.get(k, 0.0) or 0.0) / scale
        row["op_stat_rarity_tier"] = tier / float(scales.get("rarityTier") or 6.0)

        equips = char_equip.get(char_id) or []
        equip_types = (str(equip_dict.get(e, {}).get("typeName1", "")).upper() for e in equips)
        row["op_has_module"] = int(any(t != "ORIGINAL" for t in equip_types))
        tags = _skill_tag_columns(c.get("skills") or [], skill_table, keywords)
        for kw, col in zip(keywords, tag_cols, strict=True):
            row[col] = tags[kw]
        rows.append(row)
    meta_columns = ["char_id", "name", "rarity_tier", "sub_profession"]
    return rows, vector_columns, meta_columns


def _build_enemy_rows(
    enemies: list[dict[str, Any]],
    scales: dict[str, float],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """生成敌人行：每个 (enemy_id, level) 变体一行。"""
    lvl_cols = [f"en_lvl_{t.lower()}" for t in _ENEMY_LEVEL_TYPES]
    motion_cols = [f"en_motion_{m.lower()}" for m in _ENEMY_MOTIONS]
    way_cols = [f"en_way_{w.lower()}" for w in _ENEMY_APPLY_WAYS]
    stat_cols = [f"en_stat_{k}" for k in ENEMY_STAT_KEYS]
    vector_columns = lvl_cols + motion_cols + way_cols + stat_cols

    rows: list[dict[str, Any]] = []
    for e in enemies:
        enemy_id = str(e.get("Key", ""))
        for variant in e.get("Value") or []:
            d = variant.get("enemyData") or {}
            attrs = d.get("attributes") or {}

            def attr(key: str) -> float:
                v = _m(attrs.get(key))
                return float(v) if isinstance(v, (int, float)) else 0.0

            level_type = str(_m(d.get("levelType")) or "NORMAL")
            motion = str(_m(d.get("motion")) or "WALK")
            apply_way = str(_m(d.get("applyWay")) or "NONE")
            tags = _m(d.get("enemyTags")) or []
            row: dict[str, Any] = {
                "enemy_id": enemy_id,
                "level_index": int(variant.get("level") or 0),
                "row_id": f"{enemy_id}#{int(variant.get('level') or 0)}",
                "name": str(_m(d.get("name")) or ""),
                "level_type": level_type,
                "motion": motion,
                "apply_way": apply_way,
                "life_point_reduce": int(_m(d.get("lifePointReduce")) or 0),
                "not_count_in_total": bool(_m(d.get("notCountInTotal")) or False),
                "tags": json.dumps(tags, ensure_ascii=False),
            }
            for t, col in zip(_ENEMY_LEVEL_TYPES, lvl_cols, strict=True):
                row[col] = int(level_type == t)
            for mo, col in zip(_ENEMY_MOTIONS, motion_cols, strict=True):
                row[col] = int(motion == mo)
            for w, col in zip(_ENEMY_APPLY_WAYS, way_cols, strict=True):
                row[col] = int(apply_way == w)
            for k, col in zip(ENEMY_STAT_KEYS, stat_cols, strict=True):
                scale = float(scales.get(k, 1.0)) or 1.0
                row[col] = attr(k) / scale
            rows.append(row)
    meta_columns = [
        "enemy_id",
        "level_index",
        "row_id",
        "name",
        "level_type",
        "motion",
        "apply_way",
        "life_point_reduce",
        "not_count_in_total",
        "tags",
    ]
    return rows, vector_columns, meta_columns


def build_featvec(cfg: DataConfig, resolved: ResolvedSources, out_dir: Path) -> FeatvecBuildResult:
    """构建 featvec parquet × 2 + schema JSON 并落盘（out_dir 需已存在）。"""
    character_table = _load_json(resolved.require("character_table"))
    skill_table = _load_json(resolved.require("skill_table"))
    uniequip = _load_json(resolved.require("uniequip_table"))
    enemy_db = _load_json(resolved.require("enemy_database"))
    _ = _load_json(resolved.require("battle_equip_table"))  # 校验存在性；v1 仅经 uniequip 判模组

    keywords = [str(k) for k in (cfg.featvec.get("skill_keywords") or [])]
    op_rows, op_vec, op_meta = _build_operator_rows(
        character_table=character_table,
        skill_table=skill_table,
        char_equip=uniequip.get("charEquip") or {},
        equip_dict=uniequip.get("equipDict") or {},
        scales=dict(cfg.featvec.get("op_stat_scales") or {}),
        keywords=keywords,
    )
    en_rows, en_vec, en_meta = _build_enemy_rows(
        enemies=enemy_db.get("enemies") or [],
        scales=dict(cfg.featvec.get("enemy_stat_scales") or {}),
    )

    op_out = out_path(resolved, "featvec_operators_parquet")
    en_out = out_path(resolved, "featvec_enemies_parquet")
    schema_out = out_path(resolved, "featvec_schema_json")
    pq.write_table(pa.Table.from_pylist(op_rows), op_out)
    pq.write_table(pa.Table.from_pylist(en_rows), en_out)

    schema_doc = {
        "featvec_version": FEATVEC_VERSION,
        "data_version": cfg.data_version,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": "干员/敌人以数值特征向量标识（无身份 embedding）；列变更必须升 featvec_version。",
        "operators": {
            "row_count": len(op_rows),
            "meta_columns": op_meta,
            "vector_columns": op_vec,
        },
        "enemies": {
            "row_count": len(en_rows),
            "meta_columns": en_meta,
            "vector_columns": en_vec,
        },
        "normalization": {
            "op_stat_scales": dict(cfg.featvec.get("op_stat_scales") or {}),
            "enemy_stat_scales": dict(cfg.featvec.get("enemy_stat_scales") or {}),
        },
        "skill_keywords": keywords,
    }
    with schema_out.open("w", encoding="utf-8") as f:
        json.dump(schema_doc, f, ensure_ascii=False, indent=2)
    return FeatvecBuildResult(
        n_operators=len(op_rows),
        n_enemy_rows=len(en_rows),
        op_vector_columns=op_vec,
        enemy_vector_columns=en_vec,
        op_out=op_out,
        enemy_out=en_out,
        schema_out=schema_out,
    )
