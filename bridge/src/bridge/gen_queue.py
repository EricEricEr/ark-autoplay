"""生成重放队列：从数据集作业里按"覆盖优先"挑选，产出 jobs JSON。

用户指示（2026-10-09）："尽可能的覆盖多关卡"。
故队列不是随机抽样，而是分层择优：

1. **每关保底**：每个可用关卡至少收录 ``--per-stage`` 条中优先的若干条；
2. **MAA 可导航过滤**（默认开）：按 :mod:`bridge.maa_stages` 的源码规则剔除
   MAA 必然拒收的关卡（活动关 `GT-1` 这类），避免队列里塞满注定失败的任务；
3. **低练优先**：优先 ``all_low_rarity`` / ``single_core_budget`` 桶（更易通关）；
4. **长度可控**：动作数上限，避免超长序列拖慢工厂与训练；
5. **类型与关卡均衡**：截断时按「类型交错 × 关卡 × 深度」取，避免某一类霸榜。

输出结构与 ``configs/jobs_main_v1.json`` 一致（``{"jobs": [...]}``），
每条含 ``source``（来源署名与溯源）与 ``maa_job``（MAA 作业形态），
可直接喂给 ``bridge.replay_controller``。

用法::

    python -m bridge.gen_queue --dataset <ak_dataset_v0.1> --out queue.json \\
        --per-stage 2 --max-actions 30 --limit 2000

⚠️ 本工具**只生成清单**，不代表账号已解锁全部关卡——未解锁的会在运行时
由导航结果如实记为失败（见 ``nav_maa.MaaNativeNavigator``）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bridge.maa_stages import load_maa_tasks, maa_navigable

LOW_TIERS = ("all_low_rarity", "single_core_budget")
"""低练桶：优先收录（实际更容易通关）。"""


def load_stage_index(dataset: Path) -> dict[str, dict[str, Any]]:
    """``stage_id`` → ``stages.jsonl`` 记录（提供权威 ``code`` 与 ``type``）。"""
    idx: dict[str, dict[str, Any]] = {}
    with (dataset / "stages.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            sid = str(s.get("stage_id"))
            # 复刻关（#f#）与本体共用 code，保留本体即可
            if sid not in idx or "#" not in sid:
                idx[sid] = s
    return idx


def _family(stage_id: str) -> str:
    """关卡家族（用于占比限制与统计）。"""
    if stage_id.startswith("act"):
        return stage_id.split("_")[0]
    if "_" in stage_id:
        return "_".join(stage_id.split("_")[:2])
    return stage_id


def build_queue(
    dataset: Path,
    tiers_path: Path | None,
    maa_dir: Path | None,
    per_stage: int,
    max_actions: int,
    limit: int | None,
    types: set[str] | None,
    nav_filter: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """构建队列，返回 ``(entries, stats)``。"""
    from bridge.replay_controller import dataset_record_to_job_entry

    stage_idx = load_stage_index(dataset)
    tasks = load_maa_tasks(maa_dir) if (nav_filter and maa_dir) else set()

    tier_by_job: dict[str, str] = {}
    if tiers_path and tiers_path.is_file():
        t = json.loads(tiers_path.read_text(encoding="utf-8"))
        tier_by_job = {str(k): str(v) for k, v in (t.get("by_job") or {}).items()}

    per_stage_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cnt = Counter()
    with (dataset / "copilot.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cnt["read"] += 1
            rec = json.loads(line)
            sid = str(rec.get("stage_id") or "")
            st = stage_idx.get(sid)
            if not st or not st.get("code"):
                cnt["no_code"] += 1
                continue
            if types and str(st.get("type")) not in types:
                cnt["type"] += 1
                continue
            if max_actions and len(rec.get("actions") or []) > max_actions:
                cnt["too_long"] += 1
                continue
            code = str(st["code"])
            if not maa_navigable(code, tasks):
                cnt["unnavigable"] += 1
                continue
            rec["stage_code"] = code
            rec["stage_type"] = str(st.get("type"))
            per_stage_map[sid].append(rec)

    def rank(rec: dict[str, Any]) -> tuple:
        """同关卡内排序键（越小越优）：低练 → 高评分 → 动作少。"""
        tier = tier_by_job.get(str(rec.get("id")), "zzz")
        low = 0 if tier in LOW_TIERS else 1
        ratio = rec.get("rating_ratio")
        ratio = float(ratio) if isinstance(ratio, (int, float)) else 0.0
        return (low, -ratio, len(rec.get("actions") or []))

    picked: list[dict[str, Any]] = []
    for sid in sorted(per_stage_map):
        rows = sorted(per_stage_map[sid], key=rank)
        picked.extend(rows[:per_stage])

    if limit and len(picked) > limit:
        # 类型交错 × 关卡 × 深度：避免截断时某一类型/章节霸榜
        by_type: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for rec in picked:
            by_type[str(rec.get("stage_type", "?"))][str(rec["stage_id"])].append(rec)
        type_order = sorted(by_type)
        stage_lists = {t: sorted(by_type[t]) for t in type_order}

        out: list[dict[str, Any]] = []
        depth = 0
        while len(out) < limit:
            progressed = False
            for t in type_order:
                stages = stage_lists[t]
                if depth < len(stages):
                    progressed = True
                    rows = sorted(by_type[t][stages[depth]], key=rank)
                    for d in range(per_stage):
                        if d < len(rows):
                            out.append(rows[d])
                            if len(out) >= limit:
                                break
                if len(out) >= limit:
                    break
            if not progressed:
                break
            depth += 1
        picked = out

    jobs = [dataset_record_to_job_entry(rec) for rec in picked]
    stages = {j["source"]["stage_id_raw"] for j in jobs}
    type_of_code = {
        str(s.get("code")): str(s.get("type")) for s in stage_idx.values()
    }
    type_counts = Counter(
        type_of_code.get(j["maa_job"]["stage_name"], "?") for j in jobs
    )
    fam_counts = Counter(_family(j["source"]["stage_id_raw"]) for j in jobs)
    stats = {
        "n_jobs": len(jobs),
        "n_stages": len(stages),
        "n_families": len(fam_counts),
        "filter": dict(cnt),
        "type_counts": dict(type_counts.most_common()),
        "per_stage_cap": per_stage,
        "max_actions": max_actions,
    }
    return jobs, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成重放队列（覆盖优先）")
    ap.add_argument("--dataset", required=True, help="ak_dataset 目录（含 jsonl）")
    ap.add_argument("--out", required=True, help="输出队列 JSON")
    ap.add_argument("--tiers", default=None, help="jobs_roster_tiers.json（可选）")
    ap.add_argument("--maa-dir", default=None, help="MAA 运行时目录（判可导航）")
    ap.add_argument("--per-stage", type=int, default=2)
    ap.add_argument("--max-actions", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--types", default=None, help="只收这些类型（逗号分隔）")
    ap.add_argument("--no-nav-filter", action="store_true")
    args = ap.parse_args(argv)

    jobs, stats = build_queue(
        dataset=Path(args.dataset),
        tiers_path=Path(args.tiers) if args.tiers else None,
        maa_dir=Path(args.maa_dir) if args.maa_dir else None,
        per_stage=args.per_stage,
        max_actions=args.max_actions,
        limit=args.limit,
        types=set(args.types.split(",")) if args.types else None,
        nav_filter=not args.no_nav_filter,
    )
    doc = {
        "note": (
            "由 bridge.gen_queue 生成：每关保底 + MAA 可导航过滤 + 低练优先 + "
            "长度可控 + 类型均衡。关卡名取自数据集 stages.jsonl 的 code 字段。"
        ),
        "stats": stats,
        "jobs": jobs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n输出 → {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
