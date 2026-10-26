"""采集端离线转换：原始 bundle → 可移交状态—动作 JSONL 与轻量训练输入包。

不读取像素、不生成视频、无需 torch/GPU/静态图鉴。输入原件不改动；输出必须是
新目录。按明确 win 标签和严格数量/类型校验筛选，其余局仅记审计报告。最终
词表、关卡划分和 npz 由维护者使用统一配置构建。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from bridge.copilot_schema import ACTION_EVENT

VERSION = "collector_state_action/1.0"


class Rejected(ValueError):
    """单局未过结构验收，原因写进报告，不中止其他局。"""


def read_json(path: Path) -> Any:
    """读取 UTF-8 JSON，兼容有 BOM 的采集端文件。"""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    """读 JSONL；格式错误不能被静默跳过。"""
    rows = [
        json.loads(s)
        for s in path.read_text(encoding="utf-8-sig").splitlines()
        if s.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise Rejected("jsonl_row_not_object")
    return rows


def digest(path: Path) -> str:
    """原文件完整 SHA-256，用于跨机器溯源和去重。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_job(data_root: Path, ep_dir: Path, meta: dict) -> Path:
    """优先取随包作业，不访问采集根外的旧绝对路径或任意元数据路径。"""
    candidates = [ep_dir / "job.json"]
    declared = meta.get("job_file")
    if declared:
        p = Path(str(declared))
        candidates.append(p if p.is_absolute() else ep_dir / p)
    job_id = (meta.get("source") or {}).get("job_id")
    if job_id is not None and re.fullmatch(r"[A-Za-z0-9_-]+", str(job_id)):
        candidates.append(data_root / "tmp/maa_jobs" / f"job_{job_id}.json")
    for candidate in candidates:
        p = candidate.resolve()
        if p.is_relative_to(data_root) and p.is_file():
            return p
    raise Rejected("job_json_missing_inside_data_root")


def _time_order(rows: list[dict], kind: str) -> None:
    times = [row.get("t_ms") for row in rows]
    if any(
        not isinstance(t, (int, float))
        or isinstance(t, bool)
        or not math.isfinite(t)
        or t < 0
        for t in times
    ):
        raise Rejected(f"{kind}_invalid_timestamp")
    if times != sorted(times):
        raise Rejected(f"{kind}_timestamp_not_monotonic")


def paired_samples(
    meta: dict, job: dict, states: list[dict], events: list[dict]
) -> tuple[list[dict], list[str]]:
    """严格按序配对；CopilotAction 可能先于等待条件，不把它当物理执行时间。"""
    if meta.get("battle_result") != "win":
        raise Rejected("result_not_explicit_win")
    if (meta.get("source") or {}).get("type") == "model_loop":
        raise Rejected("model_loop_not_expert_data")
    if not meta.get("episode_id") or not meta.get("stage_id"):
        raise Rejected("missing_episode_or_stage_id")
    if meta.get("stage_code") and job.get("stage_name") != meta["stage_code"]:
        raise Rejected("job_stage_mismatch")
    raw_actions = job.get("actions")
    if not isinstance(raw_actions, list):
        raise Rejected("job_actions_not_list")
    if not all(isinstance(a, dict) for a in raw_actions):
        raise Rejected("job_action_not_object")
    unknown = {a.get("type") for a in raw_actions} - set(ACTION_EVENT) - {"SkillDaemon"}
    if unknown:
        raise Rejected("unsupported_job_action_type")
    kept_positions = [
        i
        for i, a in enumerate(raw_actions)
        if a.get("type") in ACTION_EVENT and a.get("type") != "SkillDaemon"
    ]
    actions = [raw_actions[i] for i in kept_positions]
    raw_observed = [
        e
        for e in events
        if isinstance(e.get("details"), dict)
        and e["details"].get("what") == "CopilotAction"
    ]
    if not states or not actions:
        raise Rejected("empty_states_or_actions")
    state_indices = list(range(len(states)))
    if len(states) == len(raw_actions) == len(raw_observed):
        # pin 运行时也会报告 SkillDaemon，并可能导出其终局状态；不是训练动作。
        expected = dict(ACTION_EVENT, SkillDaemon="SkillDaemon")
        for action, event in zip(raw_actions, raw_observed, strict=True):
            inner = event["details"].get("details")
            if (
                not isinstance(inner, dict)
                or inner.get("action") != expected[action["type"]]
            ):
                raise Rejected("action_type_alignment_mismatch")
        state_indices = kept_positions
        states = [states[i] for i in state_indices]
        observed = [raw_observed[i] for i in kept_positions]
    else:
        # 兼容仅给真实动作导出状态的协议；控制事件同样必须先过滤。
        observed = [
            e
            for e in raw_observed
            if (e["details"].get("details") or {}).get("action") != "SkillDaemon"
        ]
    if not (len(actions) == len(states) == len(observed)):
        raise Rejected("state_action_event_count_mismatch")
    _time_order(states, "state")
    _time_order(observed, "action_event")
    samples = []
    for step, (action, state, event) in enumerate(
        zip(actions, states, observed, strict=True)
    ):
        inner = event["details"].get("details")
        if (
            not isinstance(inner, dict)
            or inner.get("action") != ACTION_EVENT[action["type"]]
        ):
            raise Rejected("action_type_alignment_mismatch")
        if (
            action["type"] in {"Deploy", "Skill", "Retreat"}
            and action.get("name")
            and inner.get("target") != action["name"]
        ):
            raise Rejected("action_target_alignment_mismatch")
        if action["type"] == "Deploy":
            loc = action.get("location")
            if not (
                isinstance(loc, list)
                and len(loc) == 2
                and all(
                    isinstance(v, int) and not isinstance(v, bool) and v >= 0
                    for v in loc
                )
            ):
                raise Rejected("invalid_deploy_location")
            if action.get("direction") not in {
                "Left",
                "Right",
                "Up",
                "Down",
                "左",
                "右",
                "上",
                "下",
                "None",
            }:
                raise Rejected("invalid_deploy_direction")
        samples.append(
            {
                "schema": VERSION,
                "episode_id": meta["episode_id"],
                "step": step,
                "source_action_index": kept_positions[step],
                "source_state_index": state_indices[step],
                "stage_id": meta["stage_id"],
                "stage_code": meta.get("stage_code"),
                "source": meta.get("source"),
                "state_t_ms": state["t_ms"],
                "action_event_t_ms": event["t_ms"],
                "state": state,
                "action": action,
                "action_event": event,
            }
        )
    warnings = []
    if not any(s.get("costs") or s.get("kills") for s in states):
        warnings.append("all_costs_and_kills_zero_or_missing")
    if not meta.get("result_screen_files") or not any(meta["result_screen_files"]):
        warnings.append("no_result_image_reference_win_label_not_visually_verified")
    return samples, warnings


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def convert(data_root: Path, out: Path) -> dict[str, Any]:
    """输出独立、可搬家的转换包；从不追加旧结果或改动/删除原始数据。"""
    data_root, out = data_root.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError("输出目录已存在；请换一个新目录，不能重复追加")
    ep_root = data_root / "episodes"
    if not ep_root.is_dir():
        raise ValueError("--data-root 下没有 episodes 目录")
    if out.is_relative_to(ep_root):
        raise ValueError("输出不能放在原始 episodes 内")
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter(
        {
            "scanned": 0,
            "accepted_episodes": 0,
            "rejected_episodes": 0,
            "samples": 0,
        }
    )
    issues, seen, sources = [], set(), []
    with tempfile.TemporaryDirectory(prefix=".ark-convert-", dir=out.parent) as temp:
        work = Path(temp) / "package"
        work.mkdir()
        with (
            (work / "samples.jsonl").open("w", encoding="utf-8", newline="\n") as sf,
            (work / "episodes_index.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as index,
        ):
            for ep_dir in sorted(p for p in ep_root.glob("*/*") if p.is_dir()):
                relative = ep_dir.relative_to(data_root).as_posix()
                counts["scanned"] += 1
                writing = False
                try:
                    meta_path = ep_dir / "episode.json"
                    meta = read_json(meta_path)
                    if not isinstance(meta, dict):
                        raise Rejected("episode_not_object")
                    if meta.get("battle_result") != "win":
                        raise Rejected("result_not_explicit_win")
                    job_path = resolve_job(data_root, ep_dir, meta)
                    job = read_json(job_path)
                    if not isinstance(job, dict):
                        raise Rejected("job_not_object")
                    states_path = ep_dir / "battle_states.jsonl"
                    events_path = ep_dir / "action_events.jsonl"
                    samples, warnings = paired_samples(
                        meta, job, read_rows(states_path), read_rows(events_path)
                    )
                    if meta["episode_id"] in seen:
                        raise Rejected("duplicate_episode_id")
                    seen.add(meta["episode_id"])
                    writing = True
                    token = hashlib.sha256(
                        str(meta["episode_id"]).encode()
                    ).hexdigest()[:20]
                    stage = (
                        re.sub(r"[^A-Za-z0-9_#-]", "_", str(meta["stage_id"]))
                        or "stage"
                    )
                    # 保留标准 job-<id>-<时间> 名称，使维护者现有排除清单仍可对账。
                    name = (
                        ep_dir.name
                        if re.fullmatch(r"[A-Za-z0-9_#.-]+", ep_dir.name)
                        and ep_dir.name not in {".", ".."}
                        else f"episode-{token}"
                    )
                    target = work / "episodes" / stage / name
                    target.mkdir(parents=True)
                    inputs = {
                        p.name: digest(p)
                        for p in (
                            meta_path,
                            states_path,
                            events_path,
                        )
                    }
                    inputs["job.json"] = digest(job_path)
                    for path in (states_path, events_path):
                        shutil.copyfile(path, target / path.name)
                    shutil.copyfile(job_path, target / "job.json")
                    converted_meta = dict(meta)
                    converted_meta["job_file"] = "job.json"
                    converted_meta["conversion"] = {
                        "schema": VERSION,
                        "input_sha256": inputs,
                        "source_bundle": relative,
                        "images_included": False,
                        "win_verification": "collector_metadata_only",
                        "warnings": warnings,
                        "paired_state_indices": [
                            s["source_state_index"] for s in samples
                        ],
                    }
                    _write_json(target / "episode.json", converted_meta)
                    for sample in samples:
                        sf.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    item = {
                        "episode_id": meta["episode_id"],
                        "stage_id": meta["stage_id"],
                        "bundle": target.relative_to(work).as_posix(),
                        "n_samples": len(samples),
                        "input_sha256": inputs,
                        "warnings": warnings,
                    }
                    index.write(json.dumps(item, ensure_ascii=False) + "\n")
                    sources.append(item)
                    counts["accepted_episodes"] += 1
                    counts["samples"] += len(samples)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    if writing:
                        raise  # 写盘失败必须终止，不能发布残缺包后声称只拒收一局。
                    reason = (
                        str(exc) if isinstance(exc, Rejected) else type(exc).__name__
                    )
                    issues.append({"bundle": relative, "reason": reason})
                    counts["rejected_episodes"] += 1
        report = {
            "schema": VERSION,
            "counts": dict(counts),
            "rejected": issues,
            "note": "win 标签来自采集元数据，未读取结算图片；不作人工通关认证",
        }
        _write_json(work / "quality_report.json", report)
        _write_json(
            work / "MANIFEST.json",
            {
                "schema": VERSION,
                "counts": dict(counts),
                "source_episodes": sources,
                "visuals_included": False,
                "vocabulary_built": False,
                "split_assigned": False,
                "files": {
                    p.relative_to(work).as_posix(): digest(p)
                    for p in sorted(work.rglob("*"))
                    if p.is_file()
                },
                "supplement_required": [
                    "batch_metadata",
                    "roster_snapshots",
                    "queue",
                    "logs",
                ],
            },
        )
        work.rename(out)
    return report


def main(argv: list[str] | None = None) -> int:
    """采集端转换命令；无网络、无模拟器连接。"""
    parser = argparse.ArgumentParser(description="离线转换采集数据（不需要视频/GPU）")
    parser.add_argument(
        "--data-root", type=Path, required=True, help="含 episodes/tmp 的数据根"
    )
    parser.add_argument("--out", type=Path, required=True, help="尚不存在的新输出目录")
    args = parser.parse_args(argv)
    try:
        report = convert(args.data_root, args.out)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"转换失败：{exc}\n")
    print(json.dumps(report["counts"], ensure_ascii=False))
    print(f"转换包 -> {args.out}；完整拒收原因见 quality_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
