"""原始 MAA 作业补全：镜像字段缺失不可猜填，优先使用缓存的公开原文。"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


def missing_runtime_fields(actions: list[dict]) -> list[dict]:
    """只检查当前 MAA 明确必需的参数，不用默认零位移掩盖源数据缺失。"""
    missing = []
    for index, action in enumerate(actions):
        kind = str(action.get("type", "")).casefold()
        fields = []
        if kind == "movecamera" and action.get("distance") is None:
            fields = ["distance"]
        elif kind == "swipe":
            fields = [key for key in ("begin", "end") if not action.get(key)]
        elif (
            kind == "click"
            and action.get("rect") is None
            and action.get("location") is None
        ):
            fields = ["rect_or_location"]
        elif kind == "setunitlocation":
            fields = [key for key in ("name", "location") if action.get(key) is None]
        if fields:
            missing.append(
                {"action_index": index, "type": action.get("type"), "fields": fields}
            )
    return missing


def hydrate_record(record: dict, script: dict, stages: dict[str, dict]) -> dict:
    """恢复完整动作、分组和要求；关卡改变时重新关联，不在旧图执行新脚本。"""
    result = copy.deepcopy(record)
    original_stage = str(script.get("stage_name") or "")
    match = re.fullmatch(r"(.+)-(Normal|Hard)", original_stage)
    base = match.group(1) if match else original_stage

    def alias(value: str) -> str:
        value = value.replace("\\", "/").rsplit("/", 1)[-1].casefold()
        return value.removeprefix("level_")

    exact = stages.get(original_stage)
    matches = []
    if exact is not None:
        matches = [original_stage]
    else:
        for sid, stage in stages.items():
            level = str(stage.get("level_id") or "")
            aliases = {alias(sid), alias(str(stage.get("code") or "")), alias(level)}
            if alias(base) in aliases:
                if match and match.group(2) == "Hard" and not sid.startswith("tough_"):
                    continue
                if match and match.group(2) == "Normal" and sid.startswith("tough_"):
                    continue
                matches.append(sid)
    previous = str(record.get("stage_id") or "")
    resolved = (
        previous if previous in matches else matches[0] if len(matches) == 1 else None
    )
    result["native_script_raw"] = copy.deepcopy(script)
    result["normalized_record_before_hydration"] = copy.deepcopy(record)
    result["source_quality"] = (
        "original_script" if resolved else "original_stage_unresolved"
    )
    if not resolved:
        # 无法定位新版本地图时，只存新原文。历史动作和历史地图仍保持配套。
        result["unresolved_original_stage"] = original_stage
        return result
    result["stage_id"] = resolved
    if stages[resolved].get("code"):
        result["stage_code"] = stages[resolved]["code"]
    result["actions"] = copy.deepcopy(script.get("actions") or [])
    old_ids = {op.get("name"): op.get("id") for op in record.get("lineup") or []}
    result["lineup"] = []
    for original in script.get("opers") or []:
        operator = copy.deepcopy(original)
        if operator.get("name") in old_ids and old_ids[operator["name"]]:
            operator.setdefault("id", old_ids[operator["name"]])
        if "requirements" in operator:
            operator["req"] = copy.deepcopy(operator["requirements"])
        result["lineup"].append(operator)
    result["groups"] = copy.deepcopy(script.get("groups") or [])
    return result


def load_cached_originals(dataset: Path, cache: Path, out: Path) -> dict[str, Any]:
    """创建新镜像，不覆盖旧派生数据；未恢复项保留原记录并标明来源质量。"""
    stages = {
        row["stage_id"]: row
        for row in (
            json.loads(line)
            for line in (dataset / "stages.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        )
    }
    out.mkdir(parents=True, exist_ok=True)
    counts = {
        "total": 0,
        "restored": 0,
        "normalized_fallback": 0,
        "unresolved_stage": 0,
    }
    temp = out / "copilot.jsonl.tmp"
    with (
        (dataset / "copilot.jsonl").open(encoding="utf-8") as source,
        temp.open("w", encoding="utf-8") as target,
    ):
        for line in source:
            record = json.loads(line)
            counts["total"] += 1
            path = cache / f"{record['id']}.script.json"
            if path.is_file():
                record = hydrate_record(
                    record, json.loads(path.read_text(encoding="utf-8")), stages
                )
                counts["restored"] += 1
                counts["unresolved_stage"] += (
                    record["source_quality"] != "original_script"
                )
            else:
                record["source_quality"] = "normalized_source_fallback"
                counts["normalized_fallback"] += 1
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp.replace(out / "copilot.jsonl")
    for name in ("stages.jsonl", "operators.jsonl"):
        destination = out / name
        if not destination.exists():
            destination.write_bytes((dataset / name).read_bytes())
    (out / "source_quality_report.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8"
    )
    return counts
