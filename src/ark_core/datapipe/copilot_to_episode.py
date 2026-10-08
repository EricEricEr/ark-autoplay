"""copilot 作业 → episode 轨迹的流式转换器（prts.plus / MAA copilot JSONL → episodes JSONL）。

每行一个 episode（protocol_version="0.1.0"），要点：
- 保留来源署名：source 小节带 job_id / author 标题（title）/ upload_time / views / like /
  rating_ratio（设计文档 §6.3 的 source 字段，CC 署名义务）；
- squad：阵容槽位（char_id + skill + req 原文；无 char_id 的占位槽走 deploy_vocab 分类）；
- actions：MAA 动作映射 Deploy→deploy, Skill→skill, Retreat→retreat, SpeedUp→speed,
  BulletTime→bullet_time；MoveCamera/Output/Click/Swipe/SkillUsage/SkillDaemon 属于
  辅助类（aux）全部跳过并按类型计数（保留原始触发条件 kills/costs/... 于 timing）；
- deploy / skill / retreat 的 target 按 deploy_vocab 形式化 subtype：
  operator（解析出 char_id）/ device（关卡装置·召唤物，保留原名）/
  category（类别通配，给 profession + sub_hint）/ unknown_name（计数、保留原文）；
- 全部动作均为 aux 的作业整局丢弃并计数（drop_reasons.all_actions_skipped）。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .deploy_vocab import DeployKind, DeployVocab
from .jobs_tiers import tier_of_lineup
from .stages_meta import family_of

# 与 protocol/version.py 对齐（改动须走 RFC 升版）：本转换器输出的轨迹协议版本
EPISODE_PROTOCOL_VERSION = "0.1.0"

# MAA 动作类型 → 协议动作类型
ACTION_MAP: dict[str, str] = {
    "Deploy": "deploy",
    "Skill": "skill",
    "Retreat": "retreat",
    "SpeedUp": "speed",
    "BulletTime": "bullet_time",
}

# 辅助类动作（跳过并计数）：见图面 / 日志类，不构成决策动作
AUX_ACTION_TYPES: tuple[str, ...] = (
    "MoveCamera",
    "Output",
    "Click",
    "Swipe",
    "SkillUsage",
    "SkillDaemon",
)

_SOURCE_TYPE = "prts.plus_copilot"
_SOURCE_URL_FMT = "https://prts.plus/?op={job_id}"

# timing 保留的原始字段（MAA 触发条件，对齐重放 tick 用）
_TIMING_KEYS: tuple[str, ...] = ("kills", "costs", "cost_changes", "pre_delay", "post_delay")

# direction 的“无效”字面量（MAA 习惯写法）
_DIRECTION_NONE = "None"


class OperatorNameIndex:
    """干员名 → char_id 解析（deploy_vocab.OperatorLookup 协议的实现）。

    operators.jsonl 存在 9 组重名（预备干员-×、郁金香、Sharp 等异格共存名）：
    全局解析只接受**全局唯一**的名字；重名歧义由作业的 lineup 槽位先行消解
    （见 RosterResolver）。
    """

    def __init__(self, operators_jsonl: Path) -> None:
        by_name: dict[str, list[str]] = {}
        self._by_id: dict[str, dict[str, Any]] = {}
        with operators_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                oid = str(o["id"])
                self._by_id[oid] = o
                by_name.setdefault(str(o["name"]), []).append(oid)
        self._unique_name_to_id = {n: ids[0] for n, ids in by_name.items() if len(ids) == 1}
        self._rarity = {
            oid: int(str(o["rarity"]).split("_", 1)[1])
            for oid, o in self._by_id.items()
            if str(o.get("rarity", "")).startswith("TIER_")
        }

    def resolve(self, name: str) -> str | None:
        """全局唯一名解析（别名改写由调用方完成）。"""
        return self._unique_name_to_id.get(name)

    def rarity_of(self, char_id: str) -> int | None:
        """char_id → 稀有度数字（供 jobs_tiers 使用）。"""
        return self._rarity.get(char_id)

    def name_of(self, char_id: str) -> str | None:
        """char_id → 干员名。"""
        o = self._by_id.get(char_id)
        return str(o["name"]) if o else None


@dataclass
class RosterResolver:
    """作业阵容内优先的名字解析：先用别名改写，再在阵容槽位中按名命中 char_id。

    处理 9 组全局重名（如 阿米娅/Sharp 等多 id 同名）时，阵容上下文是唯一的消歧依据。
    """

    roster_by_name: dict[str, str]  # 官方名 → char_id（同一作业阵容内唯一）
    global_index: OperatorNameIndex
    vocab: DeployVocab

    def resolve(self, name: str) -> str | None:
        canonical = self.vocab.canonical_operator_name(name)
        hit = self.roster_by_name.get(canonical)
        if hit is not None:
            return hit
        return self.global_index.resolve(canonical)


def _direction_of(value: Any) -> str | None:
    """方向字段：'None' / 空 → null，其余保留原文。"""
    if value is None:
        return None
    v = str(value)
    return None if v in ("", _DIRECTION_NONE) else v


def _timing_of(action: dict[str, Any]) -> dict[str, Any]:
    """保留 MAA 触发条件字段（缺省补 0）。"""
    return {k: action.get(k, 0) for k in _TIMING_KEYS}


def _target_of(
    name: Any,
    resolver: RosterResolver,
) -> tuple[dict[str, Any], str]:
    """按 deploy_vocab 形式化 target；返回 (target dict, subtype 字符串)。"""
    raw = "" if name is None else str(name)
    cls = resolver.vocab.classify(raw, resolver)
    if cls.kind is DeployKind.OPERATOR:
        char_id = resolver.resolve(cls.matched_form)
        return (
            {
                "kind": DeployKind.OPERATOR.value,
                "char_id": char_id,
                "name": cls.matched_form,
                "name_raw": raw,
            },
            DeployKind.OPERATOR.value,
        )
    if cls.kind is DeployKind.DEVICE:
        return (
            {"kind": DeployKind.DEVICE.value, "name": cls.matched_form, "name_raw": raw},
            DeployKind.DEVICE.value,
        )
    if cls.kind is DeployKind.CATEGORY:
        assert cls.category is not None
        return (
            {
                "kind": DeployKind.CATEGORY.value,
                "category_id": cls.category.category_id,
                "profession": cls.category.profession,
                "sub_hint": cls.category.sub_hint,
                "position_hint": cls.category.position_hint,
                "name_raw": raw,
            },
            DeployKind.CATEGORY.value,
        )
    return (
        {"kind": DeployKind.UNKNOWN_NAME.value, "name_raw": raw},
        DeployKind.UNKNOWN_NAME.value,
    )


@dataclass
class ConvertContext:
    """转换期共享状态：索引 / 词表 / 全量计数器。"""

    op_index: OperatorNameIndex
    vocab: DeployVocab
    stage_index: dict[str, dict[str, str]]
    data_version: str
    action_hist_kept: Counter[str] = field(default_factory=Counter)
    action_hist_aux: Counter[str] = field(default_factory=Counter)
    deploy_subtypes: Counter[str] = field(default_factory=Counter)
    unknown_names: Counter[str] = field(default_factory=Counter)
    stage_join_missing: int = 0


def convert_job(job: dict[str, Any], ctx: ConvertContext) -> dict[str, Any]:
    """单个作业 → episode dict（全为 aux 时 actions 为空，由调用方决定是否丢弃）。"""
    actions_out: list[dict[str, Any]] = []
    aux_skipped: Counter[str] = Counter()
    deploy_subtypes: Counter[str] = Counter()

    resolver: RosterResolver | None = None

    def get_resolver() -> RosterResolver:
        nonlocal resolver
        if resolver is None:
            roster_by_name = {
                ctx.vocab.canonical_operator_name(str(sl["name"])): str(sl["id"])
                for sl in (job.get("lineup") or [])
                if sl.get("name") and sl.get("id")
            }
            resolver = RosterResolver(
                roster_by_name=roster_by_name,
                global_index=ctx.op_index,
                vocab=ctx.vocab,
            )
        return resolver

    seq = 0
    for a in job.get("actions") or []:
        maa_type = str(a.get("type", ""))
        proto = ACTION_MAP.get(maa_type)
        if proto is None:
            if maa_type in AUX_ACTION_TYPES:
                aux_skipped[maa_type] += 1
            # 未登记类型（未来新增）既不算保留也不算 aux：保守起见计入 aux 并留原名
            else:
                aux_skipped[maa_type or "<missing>"] += 1
            continue
        item: dict[str, Any] = {
            "seq": seq,
            "type": proto,
            "timing": _timing_of(a),
        }
        seq += 1
        loc = a.get("location")
        if loc is not None:
            item["location"] = [int(loc[0]), int(loc[1])]
        direction = _direction_of(a.get("direction"))
        if direction is not None:
            item["direction"] = direction
        if proto in ("deploy", "skill", "retreat"):
            target, subtype = _target_of(a.get("name"), get_resolver())
            item["target"] = target
            if proto == "deploy":
                deploy_subtypes[subtype] += 1
                if subtype == DeployKind.UNKNOWN_NAME.value:
                    ctx.unknown_names[str(target["name_raw"])] += 1
        actions_out.append(item)

    ctx.action_hist_kept.update(a["type"] for a in actions_out)
    ctx.action_hist_aux.update(aux_skipped)
    ctx.deploy_subtypes.update(deploy_subtypes)

    sid = str(job.get("stage_id") or "")
    stage_meta = ctx.stage_index.get(sid)
    if stage_meta is None:
        ctx.stage_join_missing += 1
        stage_meta = {
            "stage_id": sid,
            "code": "",
            "name": str(job.get("stage_name_raw") or ""),
            "type": "",
            "difficulty": "",
            "family": family_of(sid),
        }

    job_id = job.get("id")
    squad: list[dict[str, Any]] = []
    for idx, sl in enumerate(job.get("lineup") or []):
        cid = sl.get("id")
        slot: dict[str, Any] = {
            "slot": idx,
            "name": str(sl.get("name") or ""),
            "char_id": str(cid) if cid else None,
            "skill": sl.get("skill"),
            "skill_usage": sl.get("skill_usage"),
            "req": sl.get("req"),
        }
        if not cid:
            _, subtype = _target_of(sl.get("name"), get_resolver())
            slot["placeholder_subtype"] = subtype
        squad.append(slot)

    return {
        "protocol_version": EPISODE_PROTOCOL_VERSION,
        "episode_id": f"copilot_{job_id}",
        "data_version": ctx.data_version,
        "stage": stage_meta,
        "source": {
            "type": _SOURCE_TYPE,
            "job_id": job_id,
            "title": str(job.get("title") or ""),
            "upload_time": str(job.get("upload_time") or ""),
            "views": int(job.get("views") or 0),
            "like": int(job.get("like") or 0),
            "rating_ratio": float(job.get("rating_ratio") or 0.0),
            "url": _SOURCE_URL_FMT.format(job_id=job_id),
        },
        "roster_tier": tier_of_lineup(job.get("lineup"), ctx.op_index.rarity_of),
        "squad": squad,
        "groups": job.get("groups") or [],
        "actions": actions_out,
        "stats": {
            "n_actions_kept": len(actions_out),
            "n_actions_aux_skipped": int(sum(aux_skipped.values())),
            "aux_skipped_by_type": dict(aux_skipped),
            "deploy_subtypes": dict(deploy_subtypes),
        },
    }


@dataclass(frozen=True)
class EpisodesBuildResult:
    """转换统计（进 build_report）。"""

    n_jobs: int
    n_kept: int
    n_dropped: int
    drop_reasons: dict[str, int]
    out_path: Path


def convert_copilot(
    copilot_jsonl: Path,
    ctx: ConvertContext,
    out: Path,
    drop_when_all_skipped: bool = True,
) -> EpisodesBuildResult:
    """流式转换全量作业 → episodes JSONL（一行一局）。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    n_jobs = n_kept = 0
    drop_reasons: Counter[str] = Counter()
    with (
        copilot_jsonl.open("r", encoding="utf-8") as fin,
        out.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_jobs += 1
            job = json.loads(line)
            ep = convert_job(job, ctx)
            if drop_when_all_skipped and not ep["actions"]:
                drop_reasons["all_actions_skipped"] += 1
                continue
            fout.write(json.dumps(ep, ensure_ascii=False) + "\n")
            n_kept += 1
    return EpisodesBuildResult(
        n_jobs=n_jobs,
        n_kept=n_kept,
        n_dropped=int(sum(drop_reasons.values())),
        drop_reasons=dict(drop_reasons),
        out_path=out,
    )
