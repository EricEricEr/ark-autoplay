"""采集任务导入：保留全部作业，不执行包内程序或采用包内账号授权。"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from bridge.candidate_coverage import execution_key
from bridge.copilot_schema import MAA_ACTION_TYPES, canonical_action_type
from bridge.replay_controller import dataset_record_to_job_entry
from bridge.source_originals import missing_runtime_fields

from .storage import write_json


def load_queue(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """接受维护者任务包、队列JSON或单份MAA作业；不截断条数或动作数。"""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                normalized = PurePosixPath(member.filename.replace("\\", "/"))
                if normalized.is_absolute() or ".." in normalized.parts:
                    raise ValueError("任务包包含不安全路径")
            names = archive.namelist()
            candidates = [
                name
                for name in names
                if PurePosixPath(name).name in ("queue.json", "jobs.json")
            ]
            if len(candidates) != 1:
                raise ValueError("任务包应包含一份 queue.json")
            document = json.loads(archive.read(candidates[0]).decode("utf-8-sig"))
    else:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(document, list):
        metadata: dict[str, Any] = {}
        entries = document
    elif isinstance(document, dict):
        metadata = document
        if "jobs" in document:
            entries = document["jobs"]
        elif "stage_name" in document and isinstance(document.get("actions"), list):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            entries = [
                {
                    "source": {
                        "job_id": f"local-{digest}",
                        "stage_id_raw": document["stage_name"],
                    },
                    "maa_job": document,
                }
            ]
        else:
            raise ValueError("这不是可识别的采集任务，请选择维护者提供的任务包")
    else:
        raise ValueError("任务文件格式不正确")
    if not isinstance(entries, list) or not entries:
        raise ValueError("任务包内没有作业")
    jobs = []
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError("作业条目格式不正确")
        if "maa_job" not in row and "stage_id" in row:
            row = dataset_record_to_job_entry(row)
        source = row.get("source")
        job = row.get("maa_job")
        if not isinstance(source, dict) or not isinstance(job, dict):
            raise ValueError("作业缺少来源或执行内容")
        if not source.get("job_id") or not source.get("stage_id_raw"):
            raise ValueError("作业缺少来源编号或关卡")
        if not isinstance(job.get("actions"), list) or not job.get("stage_name"):
            raise ValueError("作业缺少动作列表或关卡名称")
        if not all(isinstance(action, dict) for action in job["actions"]):
            raise ValueError("动作列表格式不正确")
        source.setdefault("collector_job_original", copy.deepcopy(job))
        normalized = []
        for action in job["actions"]:
            action["type"] = canonical_action_type(str(action.get("type") or "Deploy"))
            normalized.append(action)
        source["unsupported_action_types"] = sorted(
            {
                action["type"]
                for action in normalized
                if action["type"] not in MAA_ACTION_TYPES
            }
        )
        source["incomplete_action_fields"] = missing_runtime_fields(normalized)
        for value in (str(source["stage_id_raw"]), execution_key(source)):
            if (
                not value
                or "/" in value
                or "\\" in value
                or value in (".", "..")
                or ":" in value
            ):
                raise ValueError("作业标识包含不安全路径")
        jobs.append(row)
    return metadata, jobs


def import_queue(source: Path, data_root: Path) -> tuple[Path, dict[str, Any]]:
    """导入不可变本地副本；包内资源授权、路径配置不会进入应用设置。"""
    metadata, jobs = load_queue(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    path = data_root / "tasks" / f"queue_{digest[:20]}.json"
    original = data_root / "tasks" / f"original_{digest[:20]}{source.suffix.casefold()}"
    original.parent.mkdir(parents=True, exist_ok=True)
    if not original.exists():
        shutil.copy2(source, original)
    write_json(
        path,
        {
            "jobs": jobs,
            "source_file_sha256": digest,
            "title": str(metadata.get("title") or source.stem),
        },
    )
    return path, {
        "title": metadata.get("title") or source.stem,
        "jobs": len(jobs),
        "stages": len({row["source"]["stage_id_raw"] for row in jobs}),
        "actions": sum(len(row["maa_job"]["actions"]) for row in jobs),
    }
