"""由玩家主动导出交接包；凭据、本机配置和绝对个人路径不进入分享包。"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from .storage import now

PRIVATE_KEYS = {
    "cred",
    "token",
    "accesstoken",
    "refreshtoken",
    "authorization",
    "cookie",
    "password",
    "passwd",
    "phone",
    "email",
    "scanid",
    "scancode",
}


def redact(value: Any, data_root: Path, path: tuple[str, ...] = ()) -> Any:
    """保留干员/地图字段，移除登录态和玩家身份；本地归档本身不修改。"""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            normalized = re.sub(r"[_\-]", "", str(key)).casefold()
            if normalized in PRIVATE_KEYS:
                continue
            if normalized in ("uid", "userid", "accountid", "nickname") and (
                not path or path[-1] in ("player", "account", "status")
            ):
                continue
            if key == "name" and path and path[-1] in ("player", "account", "status"):
                continue
            result[key] = redact(child, data_root, (*path, str(key)))
        return result
    if isinstance(value, list):
        return [redact(child, data_root, path) for child in value]
    if isinstance(value, str):
        if Path(value).is_absolute():
            try:
                return Path(value).resolve().relative_to(data_root.resolve()).as_posix()
            except ValueError:
                return "<本机路径>"
        value = re.sub(
            r"[A-Za-z]:[\\/]+(?:Users|Documents and Settings)[\\/]+[^\r\n\"<>]+",
            "<本机路径>",
            value,
        )
        return value
    return value


def export_data(
    data_root: Path,
    target: Path,
    *,
    include_images: bool = False,
    redact_personal: bool = True,
    diagnostics_home: Path | None = None,
    diagnostics_only: bool = False,
) -> dict[str, int]:
    """创建新ZIP，原件不删改；分享范围由玩家在导出界面明确选择。"""
    if target.exists():
        raise ValueError("请选择新的导出文件名，已有文件不会被覆盖")
    root = data_root.resolve()
    allowed = (
        "episodes",
        "episodes_quarantine",
        "roster_snapshots",
        "tasks",
        "exports",
    )
    counts = {"files": 0, "images": 0, "redacted_json": 0}
    manifest = {}
    source_hashes = {}
    deferred_manifests = []
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            candidates = (
                []
                if diagnostics_only
                else [
                    path
                    for name in allowed
                    for path in (root / name).rglob("*")
                    if path.is_file()
                ]
            )
            candidates += (
                [root / "collection_results.jsonl"]
                if (root / "collection_results.jsonl").exists() and not diagnostics_only
                else []
            )
            for source in sorted(candidates):
                if source.resolve() == target.resolve():
                    continue
                relative = source.relative_to(root)
                if source.is_symlink() or not source.resolve().is_relative_to(root):
                    continue
                if ".secrets" in relative.parts or source.name.endswith(
                    (".local.json", ".local.yaml")
                ):
                    continue
                suffix = source.suffix.casefold()
                image = suffix in (".png", ".jpg", ".jpeg", ".mp4")
                if image and not include_images:
                    continue
                raw_task_archive = (
                    not redact_personal
                    and relative.parts[0] == "tasks"
                    and suffix in (".zip", ".arktask")
                )
                if (
                    suffix not in (".json", ".jsonl")
                    and not image
                    and not raw_task_archive
                ):
                    continue
                payload = source.read_bytes()
                source_hashes[relative.as_posix()] = hashlib.sha256(payload).hexdigest()
                if source.name == "MANIFEST.json" and redact_personal:
                    deferred_manifests.append(
                        (relative, json.loads(payload.decode("utf-8-sig")))
                    )
                    continue
                if suffix == ".json" and redact_personal:
                    payload = json.dumps(
                        redact(json.loads(payload.decode("utf-8-sig")), root),
                        ensure_ascii=False,
                        indent=2,
                    ).encode()
                    counts["redacted_json"] += 1
                elif suffix == ".jsonl" and redact_personal:
                    lines = []
                    for line in payload.decode("utf-8-sig").splitlines():
                        try:
                            lines.append(
                                json.dumps(
                                    redact(json.loads(line), root), ensure_ascii=False
                                )
                            )
                        except json.JSONDecodeError as error:
                            raise ValueError(
                                f"存在未写完的记录：{relative}。请先暂停采集并保留原件交维护者复核"
                            ) from error
                    payload = ("\n".join(lines) + "\n").encode()
                    counts["redacted_json"] += 1
                archive.writestr(relative.as_posix(), payload)
                manifest[relative.as_posix()] = hashlib.sha256(payload).hexdigest()
                counts["files"] += 1
                counts["images"] += int(image)
            if diagnostics_home is not None:
                for source in sorted((root / "runs").rglob("*")):
                    if not source.is_file() or not source.resolve().is_relative_to(
                        root
                    ):
                        continue
                    allowed = (
                        source.suffix == ".log"
                        or source.name.endswith(
                            (".result.json", ".fatal.json", ".notice.json")
                        )
                        or source.name in ("status.json", "recovery_events.jsonl")
                    )
                    if allowed:
                        relative = "diagnostics/" + source.relative_to(root).as_posix()
                        payload = redact_text(
                            source.read_text(encoding="utf-8", errors="replace"), root
                        ).encode()
                        archive.writestr(relative, payload)
                        manifest[relative] = hashlib.sha256(payload).hexdigest()
                        counts["files"] += 1
                for source in (diagnostics_home / "app_errors.log",):
                    if source.is_file():
                        relative = "diagnostics/" + source.name
                        payload = redact_text(
                            source.read_text(encoding="utf-8", errors="replace"), root
                        ).encode()
                        archive.writestr(relative, payload)
                        manifest[relative] = hashlib.sha256(payload).hexdigest()
                        counts["files"] += 1
            for relative, document in deferred_manifests:
                prefix = relative.parent.as_posix().rstrip(".")
                prefix = prefix + "/" if prefix else ""
                document = redact(document, root)
                document["files"] = {
                    name[len(prefix) :]: digest
                    for name, digest in manifest.items()
                    if name.startswith(prefix)
                }
                document["export_redaction"] = True
                document["source_manifest_sha256"] = source_hashes[relative.as_posix()]
                payload = json.dumps(document, ensure_ascii=False, indent=2).encode()
                archive.writestr(relative.as_posix(), payload)
                manifest[relative.as_posix()] = hashlib.sha256(payload).hexdigest()
                counts["files"] += 1
            archive.writestr(
                "MANIFEST.json",
                json.dumps(
                    {
                        "schema": "collector_export/1",
                        "created_at": now(),
                        "counts": counts,
                        "includes_images": include_images,
                        "login_credentials_included": False,
                        "personal_fields_redacted": redact_personal,
                        "raw_records_not_certified_as_wins": not redact_personal,
                        "files": manifest,
                        "source_file_sha256": source_hashes,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        return counts
    except BaseException:
        # 此文件只由当前函数创建，失败时不留下冒充成功的交付包。
        if target.exists():
            target.unlink()
        raise


def redact_text(text: str, root: Path) -> str:
    """诊断全文保留，移除凭据字段、账号标识和个人目录，不导出配置原文。"""
    text = text.replace(str(root), "<数据目录>").replace(root.as_posix(), "<数据目录>")
    text = re.sub(
        r"[A-Za-z]:[\\/]+(?:Users|Documents and Settings)[\\/]+[^\r\n\"<>]+",
        "<本机路径>",
        text,
    )
    text = re.sub(r"(?i)(?:Bearer\s+)[\w.\-]+", "Bearer <已移除>", text)
    text = re.sub(
        r"""(?ix)(["']?(?:uid|userid|nickname|password|cred|token|access_token|refresh_token|cookie|authorization)["']?\s*[:=]\s*)(["'][^"'\r\n]*["']|[^\s,;}\r\n]+)""",
        r"\1<已移除>",
        text,
    )
    return text
