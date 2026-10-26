"""生成重放队列：从数据集作业里按"覆盖优先"挑选，产出 jobs JSON。

用户指示（2026-10-09）："尽可能的覆盖多关卡"。
故队列不是随机抽样，而是分层择优：

1. **全量候选**：默认保留每关全部合格作业，不限制单份作业动作数；
2. **地图数据过滤**（默认开）：按 :mod:`bridge.maa_stages` 判 MAA 是否有该关的
   地图数据（``Arknights-Tile-Pos``）——**这才是 Copilot 能否处理的权威判据**
   （实测覆盖 92.9% 活动关、作业维度 100%）。
   ⚠️ 曾误用 ``Fight`` 的 ``Episode{N}`` 规则过滤，把活动关整片判死（用户指出
   "这些作业都是从 MAA 作业网站下载的"），现已纠正；
3. **低练优先**：优先 ``all_low_rarity`` / ``single_core_budget`` 桶（更易通关）；
4. **显式截断**：仅在用户指定正数参数时限制每关份数或动作数，0 表示不限；
5. **干员可达过滤**（可选，``--operbox`` 给出时启用）：只收"账号 + 1 助战"
   能凑齐阵容的作业（``缺 ≤1 名``），显著提高实际可跑率；
6. **账号进度过滤**（可选，``--account-progress``）：main 系关卡只保留
   ``<= progress``（章节序）的——账号终端进度之后的关卡在账号上不存在，
   运行时必然失败（m0 队列尾部 91 条 15~17 章作业全军覆没的事故）；
   sub/tough/camp 等无法静态判定解锁规则的类型**保守放行**；
7. **干员已知名过滤**（可选，``--require-ops-known``，默认关）：作业 opers
   必须全部在账号干员盒内，且**每个候选组（groups）至少 1 名持证**——
   2026-10-21 RCA 定性：6~9 章队列事故的失配不在 opers（字节级一致），
   而在候选组整组无持证，lineup-only 过滤完全看不见它（成队必要条件）；
8. **类型与关卡均衡**：截断时按「类型交错 × 关卡 × 深度」取，避免某一类霸榜。

输出结构与 ``configs/jobs_main_v1.json`` 一致（``{"jobs": [...]}``），
每条含 ``source``（来源署名与溯源）与 ``maa_job``（MAA 作业形态），
可直接喂给 ``bridge.replay_controller``。

用法::

    python -m bridge.gen_queue --dataset <ak_dataset_v0.1> --out queue.json \\
        --operbox <operbox.json> --max-missing 1

⚠️ 本工具**只生成清单**，不代表账号已解锁全部关卡——未解锁的会在运行时
由导航结果如实记为失败（见 ``nav_maa.MaaNavigator``）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bridge.maa_stages import (
    fight_accepts,
    has_map_data,
    load_maa_tasks,
    load_tile_keys,
)

LOW_TIERS = ("all_low_rarity", "single_core_budget")
"""低练桶：优先收录（实际更容易通关）。"""

_ASCII_CODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_]*")
"""合法关卡代号形态（ASCII 字母数字开头，可含 - 和 _）。

用于剔除数据集里 code 为中文的关卡（如 camp_01 → '乌萨斯'）。
"""

_MAIN_CODE_RE = re.compile(r"(\d+)-(\d+)")
"""主线数字关码（``14-21``/``6-9``）→ (章节, 小关) 比较键。"""


def main_code_key(code: str) -> tuple[int, int] | None:
    """``14-21`` → ``(14, 21)``；非主线数字码（S9-2 / R8-2 / 活动码）返回 None。"""
    m = _MAIN_CODE_RE.fullmatch(code)
    return (int(m.group(1)), int(m.group(2))) if m else None


def groups_reachable(groups: Any, owned: set[str]) -> bool:
    """每个候选组（groups）至少 1 名账号持证——成队必要条件。

    2026-10-21 RCA 实锤：作业 opers 与干员盒字节级一致，MAA 编队仍
    OperatorMissing 的失配点在 groups——候选组整组无持证时编队必挂，
    而 lineup-only 的可达过滤完全看不见它。空组/非 dict 组视为无约束。
    """
    for g in groups or []:
        if not isinstance(g, dict):
            continue
        src = g.get("opers") if g.get("opers") is not None else (g.get("options") or [])
        names = [
            str(o.get("name")) for o in src if isinstance(o, dict) and o.get("name")
        ]
        if names and not any(n in owned for n in names):
            return False
    return True


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
    operbox_path: Path | None = None,
    max_missing: int | None = None,
    require_fight: bool = False,
    require_req: bool = False,
    account_progress: str | None = None,
    require_ops_known: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """构建队列，返回 ``(entries, stats)``。

    ``per_stage=0`` / ``max_actions=0`` 表示相应维度不设上限。

    ``operbox_path`` + ``max_missing`` 给出时启用**干员可达过滤**：
    只保留"账号 + ``max_missing`` 个助战"能凑齐阵容的作业。

    ``require_fight=True`` 时额外要求关卡能被 ``Fight`` 接受——
    即只收主线/高难/支线（实测可接受率 100%），排除活动关/剿灭/物资筹备
    （实测 0%）。**当前导航走 ``Fight(nav_only)``，故本项应开启**（见 ADR-0007）。

    ``account_progress``（如 ``"14-21"``）：main 系数字关码只保留
    ``<= progress`` 的（章节序），丢弃计入 ``filter.beyond_progress``；
    非 main 系/非数字码保守放行。不传 = 不过滤（零回归）。

    ``require_ops_known=True``（需配 ``--operbox``）：作业 opers 必须全部
    持证，且每个候选组至少 1 名持证（见模块 docstring 第 7 条 RCA）；
    丢弃分别计入 ``filter.opers_unknown`` / ``filter.groups_unreachable``。
    """
    if per_stage < 0 or max_actions < 0:
        raise ValueError("每关作业数和动作数参数必须非负，0 表示不限")
    progress_key = main_code_key(account_progress) if account_progress else None
    if account_progress and progress_key is None:
        raise ValueError(
            f"--account-progress 只接受主线数字码（如 14-21）: {account_progress!r}"
        )
    from bridge.replay_controller import dataset_record_to_job_entry

    stage_idx = load_stage_index(dataset)
    tile_keys = load_tile_keys(maa_dir) if (nav_filter and maa_dir) else set()
    maa_tasks = load_maa_tasks(maa_dir) if (require_fight and maa_dir) else set()

    # 账号拥有的干员名（用于可达过滤）
    owned: set[str] = set()
    owned_map: dict[str, dict[str, Any]] = {}
    if operbox_path and operbox_path.is_file():
        ob = json.loads(operbox_path.read_text(encoding="utf-8"))
        owned = {
            str(o.get("name")) for o in (ob.get("own_opers") or []) if o.get("name")
        }
        # --require-req 用：干员名 -> (elite, level, skill_level)；skill 缺数据记 None
        for o in ob.get("own_opers") or []:
            if o.get("name"):
                owned_map[str(o.get("name"))] = o
        if max_missing is None:
            max_missing = 1  # 默认允许 1 个助战

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
            # 关卡名必须是 ASCII 代号才能下发。
            # 实测数据集里 54 个关卡的 code 是中文而非代号，例如
            #   camp_01 → '乌萨斯'（剿灭，36 个）、act1vautochess_m01 → '新手模式'、
            #   guide_01 → '???'。直接当关卡名下发必然失败（见 ADR-0007 约束 C）。
            if not _ASCII_CODE_RE.fullmatch(code):
                cnt["code_not_ascii"] += 1
                continue
            # 账号进度过滤（可选）：main 系数字关码只收 <= progress；
            # 非 main 系/无法静态判定的码保守放行（见模块 docstring 第 6 条）
            if progress_key is not None:
                key = main_code_key(code) if sid.startswith("main_") else None
                if key is not None and key > progress_key:
                    cnt["beyond_progress"] += 1
                    continue
            # 地图数据过滤（Copilot 能否处理的权威判据）
            if not has_map_data(sid, code, tile_keys):
                cnt["no_map_data"] += 1
                continue
            # Fight 可接受过滤（当前导航走 Fight(nav_only)，故需开启）
            if require_fight and not fight_accepts(code, maa_tasks):
                cnt["fight_rejects"] += 1
                continue
            # 干员可达过滤（可选）：账号 + 助战能否凑齐阵容
            if owned and max_missing is not None:
                names = {
                    str(sl.get("name"))
                    for sl in (rec.get("lineup") or [])
                    if sl.get("name")
                }
                if require_ops_known:
                    # opers 必须全部持证（杜绝任何名称失配漏进队）
                    if any(n not in owned for n in names):
                        cnt["opers_unknown"] += 1
                        continue
                    # 每个候选组至少 1 名持证（成队必要条件，RCA 实锤点）
                    if not groups_reachable(rec.get("groups"), owned):
                        cnt["groups_unreachable"] += 1
                        continue
                miss = len({n for n in names if n not in owned})
                if miss > max_missing:
                    cnt["too_many_missing"] += 1
                    continue
                own_ct = len(names) - miss
                if names and own_ct == 0:
                    # 整队全是缺的且阵容非空——即便 max_missing>=1 也无法开局
                    # （游戏要求主队 >=1 名自有干员）。数据里存在"全队只有助战 1 人"
                    # 的作业（例：job-37809 / 1-1 孤岛），实测 TaskChainError。
                    # 空编队（0 名作业，例 0-1）是合法回放，不受此限。
                    cnt["no_own_ops"] += 1
                    continue
                cnt["reachable"] += 1
                if require_req:
                    bad = 0
                    for sl in rec.get("lineup") or []:
                        nm = str(sl.get("name") or "")
                        o = owned_map.get(nm)
                        if not o:
                            continue  # 缺员算 max_missing，不在此计
                        rq = sl.get("req") or {}
                        re_ = rq.get("elite") or 0
                        rl = rq.get("level") or 0
                        rsk = rq.get("skill_level") or 0
                        if o.get("elite") is not None and (o.get("elite") or 0) < re_:
                            bad += 1
                            continue
                        if (o.get("elite") or 0) == re_ and (o.get("level") or 0) < rl:
                            bad += 1
                            continue
                        sk = o.get("skill_level")
                        if sk is not None and rsk and sk < rsk:
                            bad += 1
                    if bad:
                        cnt["req_insufficient"] += 1
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
        picked.extend(rows[:per_stage] if per_stage else rows)

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
                    for d in range(len(rows)):
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
    type_of_code = {str(s.get("code")): str(s.get("type")) for s in stage_idx.values()}
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
        "per_stage_cap": per_stage or None,
        "max_actions": max_actions or None,
        "total_limit": limit or None,
        "account_progress": account_progress,
        "require_ops_known": require_ops_known,
    }
    return jobs, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成重放队列（覆盖优先）")
    ap.add_argument("--dataset", required=True, help="ak_dataset 目录（含 jsonl）")
    ap.add_argument("--out", required=True, help="输出队列 JSON")
    ap.add_argument("--tiers", default=None, help="jobs_roster_tiers.json（可选）")
    ap.add_argument(
        "--maa-dir",
        default=None,
        help="MAA 运行时目录（读 Tile-Pos 判有无地图数据）",
    )
    ap.add_argument(
        "--per-stage", type=int, default=0, help="每关份数上限，默认 0 不限"
    )
    ap.add_argument(
        "--max-actions", type=int, default=0, help="动作数上限，默认 0 不限"
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--types", default=None, help="只收这些类型（逗号分隔）")
    ap.add_argument("--no-nav-filter", action="store_true")
    ap.add_argument(
        "--operbox",
        default=None,
        help="账号干员盒 JSON（给出则启用干员可达过滤）",
    )
    ap.add_argument(
        "--max-missing",
        type=int,
        default=None,
        help="允许缺几名干员（默认 1 = 可借 1 个助战；0 = 必须全靠自己）",
    )
    ap.add_argument(
        "--require-req",
        action="store_true",
        help="要求账号干员满足 lineup.req 的精级/等级/技能"
        "（skill_level 1-7/8-10 口径；缺数据的 skill 不判）",
    )
    ap.add_argument(
        "--require-fight",
        action="store_true",
        help="只收 Fight 可接受的关卡（主线/高难/支线）。"
        "当前导航走 Fight(nav_only)，**应开启**——否则活动关/剿灭/"
        "物资筹备会因 Fight 拒收而必然失败（见 ADR-0007）",
    )
    ap.add_argument(
        "--account-progress",
        default=None,
        help="账号终端进度关码（如 14-21）：main 系只收 <= 该关；"
        "sub/tough/camp 保守放行并单列 beyond_progress 计数。不传=不过滤",
    )
    ap.add_argument(
        "--require-ops-known",
        action="store_true",
        help="（需配 --operbox）作业 opers 必须全部持证、每个候选组至少 1 名持证；"
        "2026-10-21 RCA:6~9 章 OperatorMissing 失配在 groups 可达性",
    )
    args = ap.parse_args(argv)
    if args.require_ops_known and not args.operbox:
        print(
            "[gen_queue] ⚠ --require-ops-known 需要 --operbox 才能生效（本次忽略）",
            file=sys.stderr,
        )

    jobs, stats = build_queue(
        dataset=Path(args.dataset),
        tiers_path=Path(args.tiers) if args.tiers else None,
        maa_dir=Path(args.maa_dir) if args.maa_dir else None,
        per_stage=args.per_stage,
        max_actions=args.max_actions,
        limit=args.limit,
        types=set(args.types.split(",")) if args.types else None,
        nav_filter=not args.no_nav_filter,
        operbox_path=Path(args.operbox) if args.operbox else None,
        max_missing=args.max_missing,
        require_fight=args.require_fight,
        require_req=args.require_req,
        account_progress=args.account_progress,
        require_ops_known=args.require_ops_known and bool(args.operbox),
    )
    doc = {
        "note": (
            "由 bridge.gen_queue 生成：默认不限制每关作业份数或动作数；"
            "地图数据过滤、低练排序及可选干员可达过滤。"
            "关卡名取自数据集 stages.jsonl 的 code 字段。"
            "注意：地图数据过滤依据 Arknights-Tile-Pos（Copilot 的权威判据），"
            "**不是** Fight 的 Episode{N} 规则（后者会误杀活动关）。"
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
