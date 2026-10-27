"""手动资料离线整理：原始控制事件和原图成包，不伪造现有策略模型的战术标签。"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from bisect import bisect_right
from pathlib import Path
from typing import Any

from bridge.data_paths import data_path


def _rows(path: Path) -> list[dict]:
    return (
        [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if path.is_file()
        else []
    )


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def convert_manual(data_root: Path, out: Path) -> dict:
    """保存全部可核验手动资料；帧与输入仅近似同轴，明确不作为战术动作监督对。"""
    out = data_path(out, label="手动资料转换位置")
    source_root = data_root.resolve() / "manual_sessions"
    if not source_root.is_dir():
        raise ValueError("没有手动操作资料目录")
    if out.exists() or out.is_relative_to(source_root):
        raise ValueError("请选择原始手动资料目录之外、尚不存在的新输出目录")
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        "sessions": 0,
        "accepted_sessions": 0,
        "rejected_sessions": 0,
        "frames": 0,
        "input_events": 0,
        "gestures": 0,
        "tactical_action_labels": 0,
    }
    rejected = []
    with tempfile.TemporaryDirectory(
        prefix=".manual-convert-", dir=out.parent
    ) as temporary:
        work = Path(temporary) / "package"
        work.mkdir()
        with (work / "manual_input_samples.jsonl").open(
            "w", encoding="utf-8"
        ) as samples:
            for source in sorted(p for p in source_root.iterdir() if p.is_dir()):
                counts["sessions"] += 1
                writing = False
                try:
                    if source.is_symlink() or not source.resolve().is_relative_to(
                        source_root
                    ):
                        raise ValueError("session_outside_data_root")
                    for filename in (
                        "session.json",
                        "frames.jsonl",
                        "input_events.jsonl",
                        "gestures.jsonl",
                        "roster.json",
                    ):
                        member = source / filename
                        if member.exists() and not member.resolve().is_relative_to(
                            source.resolve()
                        ):
                            raise ValueError("metadata_outside_session")
                    meta = json.loads(
                        (source / "session.json").read_text(encoding="utf-8")
                    )
                    if (
                        meta.get("schema") != "manual_session/1"
                        or meta.get("phase") != "complete"
                    ):
                        raise ValueError("session_incomplete_or_failed")
                    frames = _rows(source / "frames.jsonl")
                    events = _rows(source / "input_events.jsonl")
                    gestures = _rows(source / "gestures.jsonl")
                    if any(
                        not isinstance(event.get("t_ms"), (int, float))
                        for event in events
                    ):
                        raise ValueError("input_time_missing")
                    if (
                        not frames
                        or len(frames) != meta["frames"]
                        or len(events) != meta["input_events"]
                    ):
                        raise ValueError("record_count_mismatch")
                    if len(gestures) != meta["gestures"]:
                        raise ValueError("gesture_count_mismatch")
                    files = []
                    times = [frame["received_t_ms"] for frame in frames]
                    if times != sorted(times):
                        raise ValueError("frame_time_not_monotonic")
                    for frame in frames:
                        relative_image = Path(frame["path"])
                        if relative_image.is_absolute() or ".." in relative_image.parts:
                            raise ValueError("frame_path_not_relative")
                        image = (source / frame["path"]).resolve()
                        if (
                            not image.is_relative_to(source.resolve())
                            or not image.is_file()
                        ):
                            raise ValueError("frame_missing_or_outside_session")
                        if (
                            hashlib.sha256(image.read_bytes()).hexdigest()
                            != frame["sha256"]
                        ):
                            raise ValueError("frame_hash_mismatch")
                        files.append((image, frame["path"]))
                    writing = True
                    token = hashlib.sha256(source.name.encode()).hexdigest()[:20]
                    relative = Path("manual_sessions") / token
                    destination = work / relative
                    destination.mkdir(parents=True)
                    for name in (
                        "session.json",
                        "frames.jsonl",
                        "input_events.jsonl",
                        "gestures.jsonl",
                        "roster.json",
                    ):
                        if (source / name).is_file():
                            shutil.copy2(source / name, destination / name)
                    for image, name in files:
                        target = destination / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(image, target)
                    for index, event in enumerate(events):
                        before = bisect_right(times, event["t_ms"]) - 1
                        after = before + 1
                        row = {
                            "schema": "manual_input_frame_sample/1",
                            "session": token,
                            "event_index": index,
                            "input": event,
                            "before_frame": (
                                relative / frames[before]["path"]
                            ).as_posix()
                            if before >= 0
                            else None,
                            "after_frame": (relative / frames[after]["path"]).as_posix()
                            if after < len(frames)
                            else None,
                            "alignment": (
                                "approximate_host_receipt_not_verified_state_action_pair"
                            ),
                            "tactical_action": None,
                            "existing_policy_imitation_eligible": False,
                        }
                        samples.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts["accepted_sessions"] += 1
                    counts["frames"] += len(frames)
                    counts["input_events"] += len(events)
                    counts["gestures"] += len(gestures)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    if writing:
                        raise
                    counts["rejected_sessions"] += 1
                    rejected.append({"session": source.name, "reason": str(error)})
        report = {
            "schema": "manual_conversion/1",
            "counts": counts,
            "rejected": rejected,
            "scope": "raw_pixel_control_for_review_or_pixel_training",
            "vocabulary_built": False,
            "split_assigned": False,
            "tactical_labels_fabricated": False,
            "originals_unchanged": True,
        }
        _write(work / "quality_report.json", report)
        _write(
            work / "MANIFEST.json",
            {
                **report,
                "files": {
                    p.relative_to(work).as_posix(): hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
                    for p in sorted(work.rglob("*"))
                    if p.is_file()
                },
            },
        )
        work.rename(out)
    return report
