"""故障状态、子进程健康证明和断电记账核对，不按战斗时长判定失败。"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from .storage import Settings, append_json, now, read_json, read_results, write_json

HEALTH_STALE_S = 30.0
NATIVE_CALL_S = 90.0


class NeedsUser(RuntimeError):
    """需要玩家完成登录或确认后回到主页，不能盲点未知界面。"""


def account_key(settings: Settings) -> str:
    """只在本地用稳定摘要关联同一采集账号。"""
    return hashlib.sha256(
        (settings.account_binding or settings.nickname).encode()
    ).hexdigest()


def writable_probe(root: Path) -> None:
    """实际创建、刷盘和读取小文件；不通过占满磁盘模拟写入错误。"""
    root.mkdir(parents=True, exist_ok=True)
    probe = root / f".collector-write-probe-{uuid.uuid4().hex}"
    try:
        with probe.open("xb") as stream:
            stream.write(b"collector-storage-check")
            stream.flush()
            os.fsync(stream.fileno())
        if probe.read_bytes() != b"collector-storage-check":
            raise OSError("数据写入校验未通过")
    finally:
        with contextlib.suppress(OSError):
            probe.unlink(missing_ok=True)


def load_settings(path: Path) -> tuple[Settings, str]:
    """损坏设置保留副本并恢复备份，恢复时撤销所有补给授权。"""
    if not path.exists():
        return Settings(), ""
    try:
        return Settings.load(path), ""
    except (ValueError, TypeError, AttributeError, UnicodeError):
        shutil.copy2(path, path.with_name(f"settings-damaged-{uuid.uuid4().hex}.json"))
        try:
            recovered = Settings.load(path.with_suffix(".backup.json"))
            message = "设置损坏，已从备份恢复。请核对账号、设备和数据位置后继续。"
        except (OSError, ValueError, TypeError, AttributeError, UnicodeError):
            recovered = Settings()
            message = (
                "设置及备份无法读取，损坏文件已保留。请选择原数据目录和任务后续跑。"
            )
        recovered.allow_medicine = recovered.allow_originite = False
        recovered.resource_authorized_at = ""
        recovered.accepted_notice = False
        return recovered, message


def reconcile(data_root: Path, account: str, *, demo: bool = False) -> list[dict]:
    """已完成但尚未入总账的逐局收据补入一次；半局不猜填胜负。"""
    ledger = data_root / ("demo_results.jsonl" if demo else "collection_results.jsonl")
    rows = read_results(ledger)
    known = {row.get("attempt_id") for row in rows if row.get("attempt_id")}
    # episode已完整落盘而总账尚未写入时，以同一attempt关联真实结算证据。
    completed = {}
    if not demo:
        from bridge.collection_quality import collection_status

        for pool in ("episodes", "episodes_quarantine"):
            for metadata in (data_root / pool).glob("*/*/episode.json"):
                try:
                    episode = read_json(metadata, {})
                    attempt = episode.get("source", {}).get("collector_attempt_id")
                    if attempt:
                        completed[attempt] = (metadata, episode)
                except (OSError, ValueError, AttributeError):
                    continue
        for request in (data_root / "runs").glob("*/jobs/*.json"):
            if request.name.endswith((".result.json", ".notice.json", ".health.json")):
                continue
            try:
                spec = read_json(request, {})
                attempt = spec.get("attempt_id")
                receipt = request.with_suffix(".result.json")
                if attempt not in completed or receipt.exists() or attempt in known:
                    continue
                if account_key(Settings(**spec["settings"])) != account:
                    continue
                metadata, episode = completed[attempt]
                row = {
                    key: episode.get(key)
                    for key in (
                        "battle_result",
                        "battle_states",
                        "copilot_actions",
                        "prts_detected",
                        "entered_battle",
                        "observation_only",
                        "fail_stage",
                    )
                }
                row.update(
                    execution_key=spec["entry"]["source"].get("execution_key")
                    or str(spec["entry"]["source"]["job_id"]),
                    job_id=spec["entry"]["source"]["job_id"],
                    stage_code=episode.get("stage_code"),
                    episode=str(metadata),
                    collector_account=account,
                    attempt_id=attempt,
                    receipt_complete=True,
                    recovered_from_episode=True,
                    completed_at=now(),
                )
                from bridge.candidate_coverage import execution_key

                row["execution_key"] = execution_key(spec["entry"]["source"])
                row.update(collection_status(row))
                write_json(receipt, row)
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
    for receipt in sorted((data_root / "runs").glob("*/jobs/*.result.json")):
        try:
            row = read_json(receipt, {})
        except (OSError, ValueError, UnicodeError):
            continue
        attempt = row.get("attempt_id") if isinstance(row, dict) else None
        if (
            not attempt
            or attempt in known
            or row.get("collector_account") != account
            or not row.get("receipt_complete")
            or row.get("retry_pending")
            or row.get("retry_before_battle")
        ):
            continue
        append_json(ledger, row)
        known.add(attempt)
        rows.append(row)
    return rows


class HealthReporter:
    """健康线程与实际API调用分开记录，长战斗不等于挂死。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.operations: dict[int, dict] = {}
        self.fault: Exception | None = None
        self.main_thread = threading.get_ident()
        self.last_main_at = time.time()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.done.is_set():
            with self.lock:
                snapshot = dict(self.operations)
            try:
                write_json(
                    self.path,
                    {
                        "pid": os.getpid(),
                        "at": time.time(),
                        "operations": snapshot,
                        "last_main_at": self.last_main_at,
                    },
                )
            except OSError as error:
                self.fault = error
                return
            self.done.wait(1)

    def wrap(self, name: str, function: Callable, *, seconds: float = NATIVE_CALL_S):
        """只约束单次系统/API调用的响应，不约束战斗或动作数量。"""

        def invoke(*args, **kwargs):
            if self.fault and name != "close":
                raise self.fault
            key = threading.get_ident()
            if key == self.main_thread:
                self.last_main_at = time.time()
            with self.lock:
                previous = self.operations.get(key)
                self.operations[key] = {
                    "name": name,
                    "since": time.time(),
                    "deadline_s": seconds,
                }
            try:
                return function(*args, **kwargs)
            finally:
                with self.lock:
                    if previous is None:
                        self.operations.pop(key, None)
                    else:
                        self.operations[key] = previous
                if key == self.main_thread:
                    self.last_main_at = time.time()

        return invoke

    def close(self) -> None:
        self.done.set()
        self.thread.join(timeout=6)


def unhealthy(path: Path, launched_at: float, *, clock: float | None = None) -> str:
    """发现没有健康应答或确切API阻塞；不读取画面静止时长。"""
    current = time.time() if clock is None else clock
    try:
        health = read_json(path, {}) or {}
    except (ValueError, OSError):
        health = {}
    if current - health.get("at", launched_at) > HEALTH_STALE_S:
        return "采集子进程失去健康应答"
    for operation in health.get("operations", {}).values():
        if current - operation["since"] > operation["deadline_s"]:
            return f"单次设备调用未返回：{operation['name']}"
    if (
        health.get("last_main_at")
        and not health.get("operations")
        and current - health["last_main_at"] > NATIVE_CALL_S
    ):
        return "采集主循环未再响应，原始数据保留待复核"
    return ""
