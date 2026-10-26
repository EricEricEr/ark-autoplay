"""无固定编队作业的候选覆盖：每名可组成合法队伍的候选至少出现一次。"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from bridge.copilot_schema import canonical_action_type

_ROLES = {
    "先锋": "PIONEER",
    "近卫": "WARRIOR",
    "重装": "TANK",
    "狙击": "SNIPER",
    "术师": "CASTER",
    "医疗": "MEDIC",
    "辅助": "SUPPORT",
    "特种": "SPECIAL",
}


def execution_key(source: dict[str, Any]) -> str:
    """版本/候选变体独立去重，原始 job_id 仍用于来源归属。"""
    return str(source.get("execution_key") or source.get("job_id") or "")


def stamp_execution_identity(entry: dict[str, Any]) -> dict[str, Any]:
    """完整回放按实际输入生成身份，不与旧缺动作回放混为同一次采集。"""
    fields = (
        "name",
        "elite",
        "level",
        "potential",
        "skill_level",
        "main_skill_lvl",
        "skills",
        "modules",
        "module_level",
        "profession",
        "position",
        "borrowed",
    )
    practice = [
        {key: actor.get(key) for key in fields}
        for actor in entry["source"].get("squad_metadata", [])
    ]
    payload = json.dumps(
        {"job": entry["maa_job"], "squad": practice}, ensure_ascii=False, sort_keys=True
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    identifier = str(entry["source"]["job_id"])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
        identifier = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:20]
    entry["source"]["execution_key"] = f"{identifier}-full-{digest}"
    entry["source"]["replay_schema"] = "lossless_copilot/1"
    return entry


def _requirements(option: dict, owned: dict) -> tuple[list[str], list[str]]:
    """只排除已知不满足；缺失练度明确列为待核验，不伪造为满足。"""
    req = option.get("req") or option.get("requirements") or {}
    failed, unknown = [], []
    for field, own_field in (
        ("elite", "elite"),
        ("skill_level", "skill_level"),
        ("potentiality", "potential"),
        ("module", "module_level"),
    ):
        minimum = req.get(field) or 0
        if minimum:
            actual = owned.get(own_field)
            if actual is None:
                unknown.append(field)
            elif actual < minimum:
                failed.append(field)
    level = req.get("level") or 0
    if level:
        elite = owned.get("elite")
        if elite is None or owned.get("level") is None:
            unknown.append("level")
        elif elite == (req.get("elite") or 0) and owned["level"] < level:
            failed.append("level")
    skill = option.get("skill") if option.get("skill") is not None else 0
    if owned.get("skill_count") is not None and skill > owned["skill_count"]:
        failed.append("skill_slot")
    minimum_elite = max(0, skill - 1)
    if owned.get("elite") is None:
        unknown.append("skill_unlock")
    elif owned["elite"] < minimum_elite:
        failed.append("skill_unlock")
    return failed, unknown


def _placement_issues(
    name: str, option: dict, job: dict, stage: dict, owned: dict
) -> list[str]:
    issues = []
    for action in job.get("actions") or []:
        if (
            action.get("name") != name
            or canonical_action_type(str(action.get("type"))) != "Deploy"
        ):
            continue
        role = str(action.get("role") or "").upper()
        role = _ROLES.get(role, role)
        if role and owned.get("profession") and role != owned["profession"]:
            issues.append("profession")
        loc = action.get("location")
        grid = stage.get("grid") or []
        if not loc or not grid:
            continue
        col, row = loc
        if not (0 <= row < len(grid) and 0 <= col < len(grid[row])):
            issues.append("deployment_location")
            continue
        build = str((grid[row][col] or {}).get("build", ""))
        build = {"0": "NONE", "1": "MELEE", "2": "RANGED", "3": "ALL"}.get(build, build)
        if build == "NONE" or (
            build in ("MELEE", "RANGED")
            and owned.get("position")
            and owned["position"] not in (build, "ALL")
        ):
            issues.append("deployment_position")
    return issues


def expand_candidate_coverage(
    entry: dict[str, Any],
    owned: dict[str, dict],
    *,
    stage: dict | None = None,
    known_operator_names: set[str] | None = None,
    non_operator_names: set[str] | None = None,
) -> tuple[list[dict], dict]:
    """覆盖每个合法候选，不枚举全部笛卡尔积，也不设置份数上限。"""
    stage = stage or {}
    job = entry["maa_job"]
    fixed = {op["name"] for op in job.get("opers", []) if op.get("name")}
    groups = copy.deepcopy(job.get("groups") or [])
    report: dict[str, Any] = {
        "job_id": entry["source"]["job_id"],
        "mode": "each_eligible_candidate",
        "excluded": [],
        "unverified_requirements": [],
        "infeasible_candidates": [],
    }
    if fixed and not groups:
        variant = copy.deepcopy(entry)
        variant["source"]["squad_metadata"] = [
            copy.deepcopy(owned[name]) for name in sorted(fixed) if name in owned
        ]
        return [stamp_execution_identity(variant)], report
    if not groups:
        slots = list(
            dict.fromkeys(
                str(action["name"])
                for action in job.get("actions", [])
                if canonical_action_type(str(action.get("type"))) == "Deploy"
                and action.get("name")
                and action.get("name") not in (non_operator_names or set())
            )
        )
        for slot in slots:
            if slot in owned:
                options = [{"name": slot, "skill": 0}]
            elif known_operator_names and slot in known_operator_names:
                report["excluded"].append(
                    {"group": slot, "name": slot, "reason": ["not_owned"]}
                )
                options = []
            else:
                options = [
                    {
                        "name": name,
                        "skill": 0,
                    }
                    for name in sorted(owned)
                ]
            groups.append({"name": slot, "opers": options})
    if not groups:
        groups = [
            {
                "name": "__squad_only__",
                "opers": [
                    {"name": name, "skill": 0}
                    for name in sorted(owned)
                    if not owned[name].get("borrowed")
                ],
            }
        ]
        report["squad_only_coverage"] = True
    choices: dict[str, dict[str, dict]] = {}
    for group in groups:
        group_name = str(group["name"])
        options = (
            group.get("opers")
            if group.get("opers") is not None
            else group.get("options", [])
        )
        legal = {}
        for option in options:
            name = str(option.get("name") or "")
            if not name or name not in owned or name in fixed:
                report["excluded"].append(
                    {"group": group_name, "name": name, "reason": ["not_owned"]}
                )
                continue
            failed, unknown = _requirements(option, owned[name])
            failed += _placement_issues(group_name, option, job, stage, owned[name])
            if failed:
                report["excluded"].append(
                    {"group": group_name, "name": name, "reason": failed}
                )
                continue
            if unknown:
                report["unverified_requirements"].append(
                    {"group": group_name, "name": name, "fields": unknown}
                )
            legal[name] = option
        choices[group_name] = legal

    def assignment(force: tuple[str, str] | None = None) -> dict[str, str] | None:
        owners: dict[str, str] = {}
        answer: dict[str, str] = {}
        if force:
            answer[force[0]] = force[1]
            owners[force[1]] = force[0]

        def augment(group_name: str, seen: set[str]) -> bool:
            for name in sorted(choices[group_name]):
                if name in seen or (force and name == force[1]):
                    continue
                seen.add(name)
                previous = owners.get(name)
                if previous is None or augment(previous, seen):
                    owners[name] = group_name
                    answer[group_name] = name
                    return True
            return False

        for group_name in choices:
            if force and group_name == force[0]:
                continue
            if not augment(group_name, set()):
                return None
        return answer

    mappings: list[dict[str, str]] = []
    covered: set[tuple[str, str]] = set()
    covered_names: set[str] = set()
    if fixed:
        selected = assignment()
        if selected is not None:
            mappings.append(selected)
            covered.update(selected.items())
            covered_names.update(selected.values())
    for group_name, options in choices.items():
        for name in sorted(options):
            if name in covered_names:
                continue
            selected = assignment((group_name, name))
            if selected is None:
                report["infeasible_candidates"].append(
                    {"group": group_name, "name": name}
                )
                continue
            mappings.append(selected)
            covered.update(selected.items())
            covered_names.update(selected.values())
    variants = []
    for selected in mappings:
        variant = copy.deepcopy(entry)
        concrete = variant["maa_job"]
        concrete["opers"] = copy.deepcopy(job.get("opers") or [])
        for group_name, name in selected.items():
            operator = copy.deepcopy(choices[group_name][name])
            operator.pop("req", None)
            concrete["opers"].append(operator)
        concrete["groups"] = []
        for action in concrete.get("actions") or []:
            if action.get("name") in selected:
                action["name"] = selected[action["name"]]
        variant["source"]["group_assignment"] = selected
        variant["source"]["candidate_coverage"] = "each_eligible_candidate"
        variant["source"]["squad_metadata"] = [
            copy.deepcopy(owned[operator["name"]])
            for operator in concrete["opers"]
            if operator["name"] in owned
        ]
        variants.append(stamp_execution_identity(variant))
    report["covered_candidates"] = [
        {"group": group, "name": name} for group, name in sorted(covered)
    ]
    report["covered_names"] = sorted(covered_names)
    report["variant_count"] = len(variants)
    return variants, report
