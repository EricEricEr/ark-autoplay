"""采集故障闭环回归：真实文件写盘、断电收据、设备恢复和无时长截断健康判断。"""

import errno
import subprocess
import threading
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import pytest

from bridge.collector import supervisor, worker
from bridge.collector.discovery import Device
from bridge.collector.exporting import export_data, redact_text
from bridge.collector.resilience import (
    account_key,
    load_settings,
    reconcile,
    unhealthy,
    writable_probe,
)
from bridge.collector.service import CollectorService
from bridge.collector.storage import Settings, read_json, read_results, write_json


def test_damage_restores_settings_without_inheriting_resource_authority(tmp_path):
    file = tmp_path / "settings.local.json"
    settings = Settings(data_dir=str(tmp_path / "original-data"), account_binding="SYN")
    settings.save(file)
    settings.allow_originite = settings.allow_medicine = True
    settings.resource_authorized_at = "SYN-confirmed"
    settings.save(file)
    file.write_text('{"interrupted":', encoding="utf-8")
    restored, message = load_settings(file)
    assert restored.data_dir == settings.data_dir and message
    assert not restored.allow_originite and not restored.allow_medicine
    assert list(tmp_path.glob("settings-damaged-*.json"))


def test_completed_receipt_reconciles_exactly_once_and_leaves_partial_bytes(tmp_path):
    settings = Settings(account_binding="SYN")
    receipt = tmp_path / "runs/run/jobs/j0.result.json"
    write_json(
        receipt,
        {
            "attempt_id": "SYN_ATTEMPT",
            "execution_key": "SYN_JOB",
            "collector_account": account_key(settings),
            "receipt_complete": True,
            "battle_result": "unknown",
        },
    )
    partial = tmp_path / "collection_results.jsonl"
    partial.write_bytes(b'{"interrupted":')
    reconcile(tmp_path, account_key(settings))
    reconcile(tmp_path, account_key(settings))
    assert len(read_results(partial)) == 1
    assert b'{"interrupted":' in partial.read_bytes()


def test_final_episode_recovers_crash_between_episode_and_receipt(tmp_path):
    settings = Settings(account_binding="SYN_ACCOUNT")
    request = tmp_path / "runs/one/jobs/j0.json"
    write_json(
        request,
        {
            "attempt_id": "SYN_ATTEMPT",
            "settings": asdict(settings),
            "entry": {"source": {"job_id": "SYN_JOB", "execution_key": "SYN_VARIANT"}},
        },
    )
    write_json(
        tmp_path / "episodes/SYN/job/episode.json",
        {
            "source": {"collector_attempt_id": "SYN_ATTEMPT"},
            "stage_code": "SYN-1",
            "battle_result": "win",
            "battle_states": {"n": 2},
            "copilot_actions": 1,
            "prts_detected": [],
            "entered_battle": True,
            "observation_only": False,
        },
    )
    reconcile(tmp_path, account_key(settings))
    rows = reconcile(tmp_path, account_key(settings))
    assert len(rows) == 1 and rows[0]["execution_key"] == "SYN_VARIANT"
    assert rows[0]["valid_collected_win"] and rows[0]["recovered_from_episode"]


def test_stale_alive_supervisor_cannot_start_duplicate(tmp_path, monkeypatch):
    service = CollectorService(tmp_path)
    service.settings.last_run_dir = str(tmp_path / "run")
    write_json(tmp_path / "run/run.json", {"run_token": "SYN"})
    write_json(
        tmp_path / "run/status.json",
        {"run_token": "SYN", "pid": 123, "status": "running", "heartbeat_epoch": 0},
    )
    monkeypatch.setattr("bridge.collector.service.process_alive", lambda _: True)
    assert service.status()["status"] == "unresponsive" and service.is_running()
    with pytest.raises(RuntimeError, match="进行"):
        service.start()


def test_native_call_deadlock_detected_while_long_healthy_battle_is_unlimited(tmp_path):
    file = tmp_path / "health.json"
    write_json(file, {"at": 10**9, "operations": {}})
    assert unhealthy(file, 0, clock=10**9) == ""
    write_json(
        file,
        {
            "at": 1000,
            "operations": {"1": {"name": "connect", "since": 0, "deadline_s": 90}},
        },
    )
    assert "connect" in unhealthy(file, 0, clock=1000)
    assert "健康应答" in unhealthy(tmp_path / "missing", 0, clock=1000)


def test_device_manager_failure_recovers_and_discovers_changed_port(
    tmp_path, monkeypatch
):
    settings = Settings(mumu_manager="SYN.exe", instance_index="1", adb_path="SYN-adb")
    attempts = []

    def devices(_):
        attempts.append(True)
        if len(attempts) == 1:
            raise subprocess.TimeoutExpired("synthetic-manager", 1)
        return [Device("1", "SYN", "127.0.0.1:19000", True)]

    monkeypatch.setattr(worker, "list_devices", devices)
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)

    def command(args, **kwargs):
        output = b"device" if args[-1] == "get-state" else b""
        if args[-1] == "size":
            output = b"Physical size: 1080x1920"
        if args[-1] == "density":
            output = b"Physical density: 320"
        return subprocess.CompletedProcess(args, 0, output)

    monkeypatch.setattr(worker, "run_command", command)
    worker.prepare_device(settings, lambda **kwargs: None, lambda: False)
    assert len(attempts) == 2 and settings.device_addr.endswith(":19000")


def test_active_game_exit_is_detected_without_visual_freeze_limit(monkeypatch):
    settings = Settings(adb_path="SYN_ADB", device_addr="127.0.0.1:19000")

    def command(args, **kwargs):
        if args[-1] == "get-state":
            return subprocess.CompletedProcess(args, 0, b"device")
        return subprocess.CompletedProcess(args, 1, b"")

    monkeypatch.setattr(worker, "run_command", command)
    assert "游戏进程已退出" in worker.environment_failure(settings)


def test_write_failure_stops_before_any_job_and_recovers_without_new_queue(
    tmp_path, monkeypatch
):
    service = CollectorService(tmp_path / "profile", demo=True)
    data = Path(service.settings.data_dir)
    run = data / "runs/fault"
    write_json(run / "control.json", {"command": "continue"})
    request = run / "run.json"
    write_json(
        request,
        {
            "run_token": "SYN",
            "settings": asdict(service.settings),
            "run_dir": str(run),
            "live_file": str(tmp_path / "live.json"),
        },
    )

    def deny(_):
        raise OSError(errno.ENOSPC, "synthetic storage full")

    monkeypatch.setattr(supervisor, "writable_probe", deny)
    monkeypatch.setattr(
        worker.subprocess, "Popen", lambda *a, **k: pytest.fail("不得在写盘失败后开打")
    )
    assert worker.run_session(request) == 1
    state = read_json(tmp_path / "live.json")
    assert state["status"] == "storage_error" and state["attempted"] == 0
    monkeypatch.setattr(supervisor, "writable_probe", writable_probe)
    writable_probe(data)


def test_diagnostics_keeps_logs_but_excludes_config_and_credentials(tmp_path):
    data, home = tmp_path / "data", tmp_path / "profile"
    log = data / "runs/run/worker.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        'UID=SYN_PRIVATE token="SYN_SECRET" failure=connection\n', encoding="utf-8"
    )
    write_json(log.with_name("run.json"), {"settings": {"account": "SYN_PRIVATE"}})
    write_json(home / ".secrets/cred.json", {"cred": "SYN_SECRET"})
    output = tmp_path / "diagnostics.zip"
    export_data(data, output, diagnostics_home=home, diagnostics_only=True)
    with zipfile.ZipFile(output) as archive:
        text = archive.read("diagnostics/runs/run/worker.log").decode()
        assert "failure=connection" in text
        assert "SYN_SECRET" not in text and "SYN_PRIVATE" not in text
        assert not any(
            name.endswith("run.json") or ".secrets" in name
            for name in archive.namelist()
        )
    escaped = "C:" + "\\\\" + "Users" + "\\\\" + "SYN_PRIVATE" + "\\\\" + "logs"
    assert "SYN_PRIVATE" not in redact_text(escaped, data)


def test_operator_confirmation_is_fresh_and_resume_checks_actual_home(
    tmp_path, monkeypatch
):
    # 实际等待与控制文件，避免旧continue在没有玩家确认时自动放行。
    service = CollectorService(tmp_path / "profile", demo=True)
    data = Path(service.settings.data_dir)
    run = data / "runs/user"
    request = run / "run.json"
    write_json(run / "control.json", {"command": "resume_checked", "at": "2000"})
    write_json(
        request,
        {"run_token": "SYN", "settings": asdict(service.settings), "run_dir": str(run)},
    )
    seen = []

    def start(_):
        if not seen:
            seen.append("blocked")
            raise supervisor.NeedsUser("SYN needs confirmation")
        raise OSError(errno.ENOSPC, "end synthetic test before any battle")

    monkeypatch.setattr(worker, "start_game_if_needed", start)
    thread = threading.Thread(target=lambda: worker.run_session(request))
    thread.start()
    try:
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            state = read_json(run / "status.json", {})
            if state.get("status") == "needs_user":
                break
            time.sleep(0.05)
        assert state["status"] == "needs_user" and thread.is_alive()
        write_json(run / "control.json", {"command": "stop_after"})
    finally:
        write_json(run / "control.json", {"command": "stop_after"})
        thread.join(timeout=6)
    assert not thread.is_alive()
