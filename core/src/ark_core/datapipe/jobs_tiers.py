"""作业阵容练度分桶（roster tiers）：每个 copilot 作业归入 5 个互斥桶。

桶定义（按以下优先级依次判定，已在 ak_dataset_v0.1 上验证计数
726 / 5581 / 2959 / 29740 / 1293，见 docs/adr/0002）：

1. ``no_roster``（无阵容信息）：lineup 为空，或所有槽位都无法解析为干员；
2. ``all_low_rarity``（全 4 星以下）：全部可解析干员稀有度 ≤ TIER_4；
3. ``single_core_budget``（单核平民）：至多 1 名“核心”干员（稀有度 ≥ TIER_5，即 5/6 星，
   其余干员自然 ≤ 4 星），且全队 req 均未标精二（所有槽位 elite < 2 且 module == 0）；
4. ``single_core_marked_e2``（单核但标精二）：同上但存在任一槽位 req 标了精二 / 模组
   （elite ≥ 2 或 module > 0）；
5. ``multi_core``（多核高配）：其余全部情况（>1 名 5 星及以上核心）。

注：所谓“≤1 个核心”中的“核心”按稀有度 ≥ TIER_5 计数——只有该语义能复现验证计数
（若核心仅按 TIER_6 且其余 ≤ 4 星，会得到 5137/2904/30239，与验证集不一致）；lineup 里
无法解析为干员的槽位（类别占位符等）不参与稀有度判断、不计核心。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 桶 id（落盘值）与中文标签
TIER_NO_ROSTER = "no_roster"
TIER_ALL_LOW_RARITY = "all_low_rarity"
TIER_SINGLE_CORE_BUDGET = "single_core_budget"
TIER_SINGLE_CORE_MARKED_E2 = "single_core_marked_e2"
TIER_MULTI_CORE = "multi_core"

TIER_ORDER: tuple[str, ...] = (
    TIER_ALL_LOW_RARITY,
    TIER_SINGLE_CORE_BUDGET,
    TIER_SINGLE_CORE_MARKED_E2,
    TIER_MULTI_CORE,
    TIER_NO_ROSTER,
)

TIER_LABELS: dict[str, str] = {
    TIER_ALL_LOW_RARITY: "全4星以下",
    TIER_SINGLE_CORE_BUDGET: "单核平民",
    TIER_SINGLE_CORE_MARKED_E2: "单核但标精二",
    TIER_MULTI_CORE: "多核高配",
    TIER_NO_ROSTER: "无阵容信息",
}

# 核心干员稀有度下限（TIER_5 = 5 星，见模块 docstring 注）
CORE_TIER_MIN = 5
# 全 4 星以下桶的稀有度上限
LOW_RARITY_MAX = 4


def tier_of_lineup(
    lineup: list[dict[str, Any]] | None,
    rarity_of: Any,  # Callable[[str], int | None]，避免与转换器循环 import
) -> str:
    """判定单个作业阵容所属桶（规则见模块 docstring，TIER_ORDER 即优先级）。"""
    slots = lineup or []
    tiers: list[int] = []
    n_core = 0
    marked_e2 = False
    for sl in slots:
        cid = sl.get("id")
        t = rarity_of(str(cid)) if cid else None
        if t is not None:
            tiers.append(t)
            if t >= CORE_TIER_MIN:
                n_core += 1
        req = sl.get("req") or {}
        if (req.get("elite") or 0) >= 2 or (req.get("module") or 0) > 0:
            marked_e2 = True
    if not tiers:
        return TIER_NO_ROSTER
    if max(tiers) <= LOW_RARITY_MAX:
        return TIER_ALL_LOW_RARITY
    if n_core <= 1:
        return TIER_SINGLE_CORE_MARKED_E2 if marked_e2 else TIER_SINGLE_CORE_BUDGET
    return TIER_MULTI_CORE


def load_rarity_table(operators_jsonl: Path) -> dict[str, int]:
    """operators.jsonl → char_id 到稀有度数字（TIER_n 的 n）的映射。"""
    table: dict[str, int] = {}
    with operators_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            rarity = str(o.get("rarity", ""))
            if rarity.startswith("TIER_"):
                table[str(o["id"])] = int(rarity.split("_", 1)[1])
    return table


@dataclass(frozen=True)
class TiersBuildResult:
    """分桶构建统计。"""

    counts: dict[str, int]
    n_families: int
    out_path: Path


def build_tiers(
    copilot_jsonl: Path,
    operators_jsonl: Path,
    family_of_job: Any,  # Callable[[dict[str, Any]], str]
    out: Path,
    data_version: str,
) -> TiersBuildResult:
    """流式扫 copilot.jsonl 打分桶，输出 jobs_roster_tiers.json。

    family_of_job：由调用方注入的家族函数（episodes 与 tiers 共用同一实现）。
    """
    rarity_table = load_rarity_table(operators_jsonl)

    def rarity_of(cid: str) -> int | None:
        return rarity_table.get(cid)

    counts: Counter[str] = Counter()
    per_family: dict[str, Counter[str]] = defaultdict(Counter)
    by_job: dict[str, str] = {}
    with copilot_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            job = json.loads(line)
            tier = tier_of_lineup(job.get("lineup"), rarity_of)
            job_id = str(job.get("id"))
            counts[tier] += 1
            per_family[family_of_job(job)][tier] += 1
            by_job[job_id] = tier

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "data_version": data_version,
        "tier_order": list(TIER_ORDER),
        "tier_labels": TIER_LABELS,
        "core_tier_min": CORE_TIER_MIN,
        "counts": {k: int(counts.get(k, 0)) for k in TIER_ORDER},
        "per_family": {
            fam: {k: int(cnt.get(k, 0)) for k in TIER_ORDER}
            for fam, cnt in sorted(per_family.items())
        },
        "by_job": by_job,
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    return TiersBuildResult(
        counts={k: int(counts.get(k, 0)) for k in TIER_ORDER},
        n_families=len(per_family),
        out_path=out,
    )
