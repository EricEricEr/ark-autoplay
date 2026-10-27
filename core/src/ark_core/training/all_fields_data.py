"""逐份资料按真实字段编码；未知结局、缺失状态与计划动作不补造为监督标签。"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .imitation_data import Vocab, _load_featvec, _load_stagefeat
from .state_action_data import (
    STATE_DIM,
    BuildStatsV1,
    _load_name_map,
    base_stage_id,
    encode_episode,
    state_vector,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """缺文件明确表示没有此类记录，不创建虚构行。"""
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def observed_state(
    raw: dict[str, Any], step: int, total: int, squad: int
) -> tuple[np.ndarray, np.ndarray]:
    """既有12维状态的观测掩码；缺失填充值仅作输入，不能成为重构真值。"""

    def number(value: Any) -> bool:
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )

    hand = raw.get("hand")
    deployed = raw.get("deployed")
    has_hand = isinstance(hand, list)
    hand = hand if has_hand else []
    has_deployed = isinstance(deployed, list)
    safe = dict(raw)
    safe["hand"] = [
        {**entry, "cost": entry.get("cost") if number(entry.get("cost")) else 0} for entry in hand
    ]
    safe["deployed"] = deployed if has_deployed else []
    for key in ("costs", "kills", "total_kills"):
        if not number(safe.get(key)):
            safe[key] = 0
    values = state_vector(safe, step, total, squad)
    mask = np.asarray(
        [
            number(raw.get("costs")),
            number(raw.get("kills")) and number(raw.get("total_kills")) and raw["total_kills"] > 0,
            isinstance(raw.get("in_speedup"), bool),
            has_hand,
            has_hand and all(isinstance(entry.get("available"), bool) for entry in hand),
            has_hand and all(isinstance(entry.get("cooling"), bool) for entry in hand),
            has_hand and all(number(entry.get("cost")) for entry in hand),
            has_deployed,
            has_deployed and squad > 0,
            False,  # 序号是计算上下文，不冒充游戏观测字段。
            has_hand
            and all(
                isinstance(entry.get("available"), bool)
                and (not entry["available"] or number(entry.get("cost")))
                for entry in hand
            ),
            False,  # reserved没有观测标签。
        ],
        dtype=np.bool_,
    )
    values[~np.isfinite(values)] = 0
    return values, mask


def load_sources(manifest: Path) -> list[dict[str, Any]]:
    """读取完整冻结清单；每份记录保留独立身份和各原件摘要。"""
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    sources = []
    for index, row in enumerate(rows):
        directory = Path(row["snapshot_bundle"])
        for name, expected in (row.get("files") or {}).items():
            if expected is not None:
                actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"冻结资料摘要变化，拒绝混入另一版本：{index}/{name}")
        meta = json.loads((directory / "episode.json").read_text(encoding="utf-8-sig"))
        job = json.loads((directory / "job.json").read_text(encoding="utf-8-sig"))
        source = {
            **row,
            "meta": meta,
            "job": job,
            "states": read_jsonl(directory / "battle_states.jsonl"),
            "events": read_jsonl(directory / "action_events.jsonl"),
            "directory": directory,
            "stage_id": str(meta.get("stage_id") or directory.parent.name),
        }
        source["id"] = f"{index}:" + hashlib.sha256(str(directory).encode()).hexdigest()[:20]
        sources.append(source)
    if len({source["id"] for source in sources}) != len(rows):
        raise ValueError("资料身份重复，拒绝遗漏或合并记录")
    return sources


def encode_sources(
    sources: list[dict[str, Any]],
    processed: Path,
    train_ids: set[str],
) -> tuple[list[dict[str, Any]], np.ndarray, list[str], dict]:
    """全量编码所有资料，目标词表只来自当折训练侧，失败局的策略损失关闭。"""
    stagefeat = _load_stagefeat(processed / "stagefeat.parquet")
    ids, featvec = _load_featvec(processed / "featvec_operators.parquet")
    name_map = _load_name_map(processed / "featvec_operators.parquet")
    index = {name: i for i, name in enumerate(ids)}
    names: Counter = Counter()
    for source in sources:
        if source["id"] not in train_ids:
            continue
        for action in source["job"].get("actions") or []:
            name = action.get("name")
            if action.get("type") == "Deploy" and name and name_map.get(str(name)) is None:
                names[f"name:{name}"] += 1
    vocab = Vocab(extra_tokens=sorted(names))
    plan_types = sorted(
        {
            str(a["type"])
            for source in sources
            if source["id"] in train_ids
            for a in source["job"].get("actions") or []
            if isinstance(a.get("type"), str) and a["type"]
        }
    )
    plan_index = {name: i + 1 for i, name in enumerate(plan_types)}
    failures = sorted(
        {
            source["meta"]["fail_stage"]
            for source in sources
            if source["id"] in train_ids
            and isinstance(source["meta"].get("fail_stage"), str)
            and source["meta"]["fail_stage"]
        }
    )
    failure_index = {name: i + 1 for i, name in enumerate(failures)}
    stats = BuildStatsV1()
    records = []
    for source in sources:
        sid = source["stage_id"]
        sf = stagefeat.get(sid) or stagefeat.get(base_stage_id(sid))
        if sf is None:
            raise ValueError(f"资料缺少关卡静态字段，不能静默丢弃：{sid}")
        record = encode_episode(
            episode_id=source["id"],
            stage_id=sid,
            job=source["job"],
            battle_states=source["states"],
            action_events=source["events"],
            sf=sf,
            name2cid=name_map,
            featvec_idx=index,
            vocab=vocab,
            stats=stats,
        )
        if record is None:
            raise ValueError(f"资料编码失败：{source['id']}")
        states = [
            observed_state(raw, i, len(source["states"]), len(record["squad_idx"]))
            for i, raw in enumerate(source["states"])
        ]
        record["observed_states"] = (
            np.stack([row[0] for row in states]) if states else np.zeros((0, STATE_DIM), np.float32)
        )
        record["observed_mask"] = (
            np.stack([row[1] for row in states]) if states else np.zeros((0, STATE_DIM), np.bool_)
        )
        meta = source["meta"]
        record.update(
            coverage_id=source["id"],
            policy_allowed=bool(source["eligible"] and len(record["type"])),
            legacy=bool(source.get("legacy")),
            outcome={"win": 0, "lost": 1}.get(meta.get("battle_result"), -1),
            marker=(int(bool(meta["prts_detected"])) if "prts_detected" in meta else -1),
            failure=(
                failure_index.get(meta["fail_stage"], 0)
                if isinstance(meta.get("fail_stage"), str) and meta["fail_stage"]
                else -1
            ),
            episode_metadata_raw=meta,
        )
        if not record["policy_allowed"]:
            # 保留候选对齐视图供复核，但通用策略加载器也不能误把失败局当正例。
            keys = ("type", "target", "cell", "dir", "state_feat")
            record["non_policy_aligned_view"] = {key: record[key] for key in keys}
            for key in keys:
                record[key] = record[key][:0].copy()
            record["entity_only"] = True
        planned = source["job"].get("actions") or []
        record["plan_type"] = np.asarray(
            [plan_index.get(a.get("type"), 0) for a in planned], np.int64
        )
        record["plan_mask"] = np.asarray(
            [isinstance(a.get("type"), str) and bool(a["type"]) for a in planned], np.bool_
        )
        record["uses"] = {
            "static": True,
            "raw_state_frames": len(source["states"]),
            "state_targets": int(record["observed_mask"].sum()),
            "policy_actions": len(record["type"]) if record["policy_allowed"] else 0,
            "observed_outcome": record["outcome"] >= 0,
            "observed_marker": record["marker"] >= 0,
            "observed_failure_stage": record["failure"] >= 0,
            "planned_actions_retained": len(source["job"].get("actions") or []),
            "planned_type_targets": int(record["plan_mask"].sum()),
            "planned_types_outside_training_vocab": int(
                ((record["plan_type"] == 0) & record["plan_mask"]).sum()
            ),
            "failure_type_outside_training_vocab": record["failure"] == 0,
        }
        records.append(record)
    report = {
        "records": len(records),
        "state_records": sum(bool(len(r["observed_states"])) for r in records),
        "static_only_records": sum(not len(r["observed_states"]) for r in records),
        "policy_records": sum(r["policy_allowed"] for r in records),
        "state_frames": sum(len(r["observed_states"]) for r in records),
        "state_targets": sum(r["uses"]["state_targets"] for r in records),
        "policy_actions": sum(r["uses"]["policy_actions"] for r in records),
        "outcome_records": sum(r["outcome"] >= 0 for r in records),
        "marker_records": sum(r["marker"] >= 0 for r in records),
        "failure_records": sum(r["failure"] >= 0 for r in records),
        "failure_types": failures,
        "vocab": vocab.extra_tokens,
        "plan_types": plan_types,
        "planned_type_targets": sum(r["uses"]["planned_type_targets"] for r in records),
        "per_record": [
            {"id": r["coverage_id"], "stage_id": r["stage_id"], "legacy": r["legacy"], **r["uses"]}
            for r in records
        ],
    }
    return records, featvec, vocab.extra_tokens, report
