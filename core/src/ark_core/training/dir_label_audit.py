"""DIR（部署朝向）标签审计：先审数据再谈模型（v3）。

动机
----
v2 holdout 的 dir 命中 0.474 **低于**恒预测 Left 的常数基线 0.526。在怪模型之前
先证伪「标签口径脏」：本模块对作业文件里的 ``direction`` 做四项纯数据审计
（不读截图）：

a) **缺失/未知别名占比**：全池与按动作类型切（Deploy 之外的方向字段本就少语义）；
b) **同格同干员方向不一致率**：按 (row, col, 干员名) 分组统计方向唯一性——
   分「同 job」组（同一作业文件内同格同干员的多次 deploy + 该作业多次回放；
   实测 2026-10-21：不一致来自**作业文件内部**同格同干员方向自相矛盾，
   如 job_45596 米格鲁 [7,2] 一次 Up 一次 Down——对行为克隆是标签噪声，
   而**非**回放漂移）与「跨 job」组（不同作业策略本就允许不同，仅作参考）；
c) **朝向合理性粗查**（不读图）：按 stagefeat 网格看 deploy 朝向的相邻格——
   越界 / 邻出怪门或保护门（必经路）/ 邻可部署格 / 其他（疑似「朝墙」）的占比。
   仅为粗粒度证据：朝向语义与干员射程/职业强相关，不做对错断言；
d) **标签先验上限**：恒预测最强类在 train / holdout 各自的 acc（direction 分布
   本身就偏斜时，模型要赢的不是随机而是这个先验）。

审计结论（``conclusion``）按配置阈值自动给出「脏 / 不脏」判定与人话段，
阈值入 yaml（禁止代码内魔法数字）。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..protocol.action import DIR_ALIASES as _DIR_ALIASES
from .imitation_data import _load_stagefeat
from .state_action_data import (
    JOB_TYPE_TO_ACTION,
    SKIP_JOB_TYPES,
    ExclusionInfo,
    _read_json,
    base_stage_id,
    iter_episode_dirs,
)

_FACING_DELTA: dict[str, tuple[int, int]] = {
    "Right": (0, 1),
    "Down": (1, 0),
    "Left": (0, -1),
    "Up": (-1, 0),
}


def _iter_job_actions(
    episodes_root: Path, excluded: frozenset[str], limit: int | None
):
    """逐局产出 (episode 目录名, stage_id, 作业动作列表)（什么文件坏就跳过该局）。"""
    n = 0
    for stage, ep_dir in iter_episode_dirs(episodes_root, limit):
        if ep_dir.name in excluded:
            continue
        try:
            meta = _read_json(ep_dir / "episode.json")
            job = _read_json(Path(str(meta["job_file"])))
        except (OSError, KeyError, json.JSONDecodeError, TypeError):
            continue
        sid = str(meta.get("stage_id") or stage)
        yield ep_dir.name, sid, job.get("actions") or []
        n += 1


def _norm_dir(raw: Any) -> tuple[str, str]:
    """direction 原值 → (归为哪类, 归一后的方向或原样)。

    类别：``missing``（键缺失或 null）/ ``empty``（空串）/ ``none``（无朝向语义）/
    ``known``（别名表命中四向）/ ``unknown``（未登录写法，标签脏的直接证据）。
    """
    if raw is None:
        return "missing", ""
    s = str(raw)
    if s == "":
        return "empty", s
    d = _DIR_ALIASES.get(s)
    if d is None:
        return "unknown", s
    if d == "None":
        return "none", d
    return "known", d


def audit_dir_labels(
    processed_dir: Path,
    episodes_root: Path,
    *,
    holdout_stages: set[str],
    exclusion: ExclusionInfo | None = None,
    limit: int | None = None,
    same_job_inconsistency_dirty: float = 0.05,
    unknown_alias_dirty: float = 0.01,
) -> dict[str, Any]:
    """执行 a~d 四项审计并返回可落盘 dict（阈值入参，禁止魔法数字）。"""
    excluded = frozenset(exclusion.union()) if exclusion is not None else frozenset()
    stagefeat = _load_stagefeat(processed_dir / "stagefeat.parquet")

    # ---- a) 分布与质量（全池 + 按类型）----
    dir_kind_hist: Counter = Counter()
    by_type: dict[str, Counter] = defaultdict(Counter)
    unknown_aliases: Counter = Counter()
    n_episodes = 0
    n_actions = 0

    # ---- b) 分组数据（值 = 方向计数，供「观测≥2」判定）----
    same_job: dict[tuple, Counter] = defaultdict(Counter)   # (job_id, stage, r, c, name)
    cross_job: dict[tuple, Counter] = defaultdict(Counter)  # (stage, r, c, name)

    # ---- c) 朝向相邻格分类 ----
    facing_hist: Counter = Counter()

    # ---- d) has_dir 帧分布（按侧）----
    train_dir_hist: Counter = Counter()
    hold_dir_hist: Counter = Counter()

    for ep_name, sid, actions in _iter_job_actions(episodes_root, excluded, limit):
        n_episodes += 1
        parts = ep_name.split("-")
        job_id = parts[1] if len(parts) >= 3 else ep_name  # job-<id>-<ts>
        hold = base_stage_id(sid) in holdout_stages
        sf = stagefeat.get(sid) or stagefeat.get(base_stage_id(sid))
        for a in actions:
            t_raw = str(a.get("type"))
            if t_raw in SKIP_JOB_TYPES or t_raw not in JOB_TYPE_TO_ACTION:
                continue
            t = JOB_TYPE_TO_ACTION[t_raw]
            n_actions += 1
            kind, d = _norm_dir(a.get("direction"))
            dir_kind_hist[kind] += 1
            by_type[t][kind] += 1
            if kind == "unknown":
                unknown_aliases[str(a.get("direction"))] += 1
            if t != "deploy":
                continue
            # deploy 帧：未归一到四向的（含 none）不参与 b~d 的一致性与先验统计
            name = str(a.get("name") or "")
            loc = a.get("location")
            r = c = -1
            if loc and len(loc) >= 2:
                # deploy location 是 [x,y]=[col,row]（坐标序约定见 state_action_data）
                x, y = int(loc[0]), int(loc[1])
                r, c = y, x
            if kind == "known":
                same_job[(job_id, sid, r, c, name)][d] += 1
                cross_job[(sid, r, c, name)][d] += 1
                (hold_dir_hist if hold else train_dir_hist)[d] += 1
            # c) 朝向相邻格粗查（只有归一四向且有坐标/地形时可查）
            if sf is not None and kind == "known" and r >= 0:
                rows, cols = int(sf["rows"]), int(sf["cols"])
                dr, dc = _FACING_DELTA[d]
                rr, cc = r + dr, c + dc
                if not (0 <= rr < rows and 0 <= cc < cols):
                    facing_hist["oob"] += 1
                else:
                    grid = sf["grid"]
                    base = (rr * cols + cc) * 5
                    _h, _build, deployable, is_start, is_end = grid[base : base + 5]
                    if is_start or is_end:
                        facing_hist["path_door"] += 1
                    elif deployable:
                        facing_hist["deployable"] += 1
                    else:
                        facing_hist["other_blockedish"] += 1

    def _share(cnt: Counter, key: str) -> float:
        tot = sum(cnt.values())
        return cnt.get(key, 0) / tot if tot else 0.0

    # b) 不一致率：只在「观测≥2 次」的组上判（单次观测无从谈不一致）
    def _inconsistency(groups: dict[tuple, Counter]) -> dict[str, Any]:
        judged = {k: v for k, v in groups.items() if sum(v.values()) >= 2}
        bad = {k: dict(v) for k, v in judged.items() if len(v) > 1}
        return {
            "n_groups_observed": len(groups),
            "n_groups_multi_obs": len(judged),
            "n_inconsistent": len(bad),
            "rate": len(bad) / len(judged) if judged else 0.0,
            "examples": [
                {"key": [str(x) for x in k], "dir_counts": v}
                for k, v in list(bad.items())[:10]
            ],
        }

    same_job_stat = _inconsistency(same_job)
    cross_job_stat = _inconsistency(cross_job)

    def _majority(hist: Counter) -> tuple[str, float]:
        tot = sum(hist.values())
        if not tot:
            return "", 0.0
        d, n = max(hist.items(), key=lambda kv: (kv[1], kv[0]))
        return d, n / tot

    train_maj, train_maj_acc = _majority(train_dir_hist)
    hold_maj, hold_maj_acc = _majority(hold_dir_hist)

    unknown_rate = _share(dir_kind_hist, "unknown")
    dirty = same_job_stat["rate"] > same_job_inconsistency_dirty or unknown_rate > (
        unknown_alias_dirty
    )
    if dirty:
        verdict = "脏：证据越过阈值（见下同 job 不一致率 / 未知别名占比）"
    else:
        verdict = (
            "不脏：同 job 重放方向基本一致、未知别名罕见；v2 的 dir 弱于恒 Left 主要"
            "暴露的是**类不平衡先验 + 小样本**，不是标签口径错误"
        )
    conclusion = (
        f"{verdict}。同 job 同格同干员方向不一致率 "
        f"{same_job_stat['n_inconsistent']}/{same_job_stat['n_groups_multi_obs']}"
        f"={same_job_stat['rate']:.3f}（脏阈值 >{same_job_inconsistency_dirty}，"
        f"分母=观测≥2 的组）；"
        f"未知别名占比 {unknown_rate:.4f}（脏阈值 >{unknown_alias_dirty}）；"
        f"has_dir 帧 train 恒最强类 {train_maj} acc={train_maj_acc:.3f}、"
        f"holdout 恒最强类 {hold_maj} acc={hold_maj_acc:.3f}。"
    )

    return {
        "episodes": n_episodes,
        "actions": n_actions,
        "a_dir_kind": {
            "all": dict(dir_kind_hist),
            "by_type": {t: dict(c) for t, c in sorted(by_type.items())},
            "unknown_aliases": dict(unknown_aliases),
        },
        "b_inconsistency": {"same_job": same_job_stat, "cross_job": cross_job_stat},
        "c_facing_terrain": {
            "hist": dict(facing_hist),
            "note": "oob=朝向越界；path_door=邻出怪/保护门；deployable=邻可部署格；"
            "other_blockedish=邻格既非门也不可部署（疑似朝墙，粗查不下对错断言）",
        },
        "d_prior": {
            "train": {"hist": dict(train_dir_hist), "majority": train_maj,
                      "majority_acc": train_maj_acc},
            "holdout": {"hist": dict(hold_dir_hist), "majority": hold_maj,
                        "majority_acc": hold_maj_acc},
        },
        "dirty": dirty,
        "conclusion": conclusion,
    }
