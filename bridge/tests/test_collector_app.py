"""采集桌面应用的实质回归：账号授权隔离、完整队列、安全导出与故障继续。"""

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

import pytest

from bridge.collector import worker
from bridge.collector.discovery import parse_devices, validate_address
from bridge.collector.exporting import export_data
from bridge.collector.service import CollectorService
from bridge.collector.storage import (
    Settings,
    append_json,
    read_json,
    read_results,
    write_json,
)
from bridge.collector.task_package import import_queue, load_queue


def entries(count=3, action_count=1):
    return [
        {
            "source": {"job_id": index + 1, "stage_id_raw": "SYN-1"},
            "maa_job": {
                "stage_name": "SYN-1",
                "opers": [],
                "actions": [{"type": "Output"} for _ in range(action_count)],
            },
        }
        for index in range(count)
    ]


def test_resource_permissions_default_off_and_reset_on_account_switch(tmp_path):
    settings = Settings()
    assert not settings.allow_medicine and not settings.allow_originite
    settings.bind_account("SYN_ACCOUNT_A")
    settings.allow_medicine = settings.allow_originite = True
    settings.resource_authorized_at = "confirmed"
    settings.bind_account("SYN_ACCOUNT_B")
    assert not settings.allow_medicine and not settings.allow_originite
    assert not settings.resource_authorized_at
    file = tmp_path / "settings.json"
    write_json(file, {"allow_originite": "true"})
    with pytest.raises(ValueError):
        Settings.load(file)


def test_task_import_keeps_all_jobs_actions_and_ignores_package_authorization(tmp_path):
    source = tmp_path / "tasks.json"
    jobs = entries(140, 150)
    write_json(source, {"jobs": jobs, "resource_policy": {"allow_originite": True}})
    target, summary = import_queue(source, tmp_path / "data")
    _, actual = load_queue(target)
    assert len(actual) == len(jobs) == 140
    assert [job["maa_job"] for job in actual] == [job["maa_job"] for job in jobs]
    assert summary["jobs"] == 140 and summary["actions"] == 21000
    assert "resource_policy" not in read_json(target)
    assert not Settings().allow_originite


def test_task_archive_rejects_traversal_and_does_not_execute_members(tmp_path):
    archive = tmp_path / "tasks.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../queue.json", json.dumps({"jobs": entries()}))
    with pytest.raises(ValueError, match="不安全"):
        load_queue(archive)


def test_device_discovery_does_not_guess_offline_ports_or_connect_remotely():
    devices = parse_devices(
        {
            "0": {"index": "0", "name": "offline", "is_android_started": False},
            "1": {
                "index": "1",
                "name": "online",
                "adb_port": 16000,
                "is_android_started": True,
            },
        }
    )
    assert devices[0].address == ""
    assert devices[1].address == "127.0.0.1:16000"
    with pytest.raises(ValueError):
        validate_address("example.com:5555")
    with pytest.raises(ValueError):
        validate_address("127.0.0.1:70000")


def test_export_preserves_data_semantics_redacts_identity_and_regenerates_hashes(
    tmp_path,
):
    data = tmp_path / "data"
    private = data / ".secrets/cred.json"
    write_json(private, {"cred": "FAKE_CREDENTIAL"})
    episode = data / "episodes/SYN/job/episode.json"
    write_json(
        episode,
        {
            "player": {"uid": "SYN_ACCOUNT", "name": "private_name"},
            "source": {"author": {"name": "public_author"}},
            "job_file": str(data / "jobs/job.json"),
            "deployed": [{"uid": "SYN_UNIT", "name": "synthetic_operator"}],
        },
    )
    image = episode.parent / "shots/one.png"
    image.parent.mkdir()
    image.write_bytes(b"synthetic image placeholder")
    converted = data / "exports/batch"
    write_json(
        converted / "sample.json", {"player": {"uid": "SYN_ACCOUNT"}, "value": 42}
    )
    write_json(converted / "MANIFEST.json", {"files": {"sample.json": "old"}})
    archive = tmp_path / "export.zip"
    export_data(data, archive)
    with zipfile.ZipFile(archive) as stream:
        names = stream.namelist()
        assert not any(".secrets" in name or name.endswith(".png") for name in names)
        actual = json.loads(stream.read("episodes/SYN/job/episode.json"))
        assert "uid" not in actual["player"] and "name" not in actual["player"]
        assert actual["deployed"] == [{"uid": "SYN_UNIT", "name": "synthetic_operator"}]
        assert actual["source"]["author"]["name"] == "public_author"
        assert actual["job_file"] == "jobs/job.json"
        manifest = json.loads(stream.read("MANIFEST.json"))
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(stream.read(name)).hexdigest() == digest
        inner = json.loads(stream.read("exports/batch/MANIFEST.json"))
        assert (
            inner["files"]["sample.json"]
            == hashlib.sha256(stream.read("exports/batch/sample.json")).hexdigest()
        )
    assert read_json(private)["cred"] == "FAKE_CREDENTIAL"
    assert read_json(episode)["player"]["uid"] == "SYN_ACCOUNT"


def test_export_does_not_silently_drop_partial_rows(tmp_path):
    source = tmp_path / "data/episodes/SYN/job/events.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text('{"complete":true}\n{"partial":', encoding="utf-8")
    output = tmp_path / "export.zip"
    with pytest.raises(ValueError, match="未写完"):
        export_data(tmp_path / "data", output)
    assert source.exists() and not output.exists()


def test_raw_export_keeps_incomplete_original_bytes_and_identity_by_explicit_choice(
    tmp_path,
):
    data = tmp_path / "data"
    path = data / "episodes/SYN/job/events.jsonl"
    path.parent.mkdir(parents=True)
    original = b'{"player":{"uid":"SYN_ACCOUNT"}}\n{"unfinished":'
    path.write_bytes(original)
    write_json(data / ".secrets/cred.json", {"cred": "FAKE_CREDENTIAL"})
    archive = tmp_path / "raw.zip"
    export_data(data, archive, redact_personal=False)
    with zipfile.ZipFile(archive) as stream:
        assert stream.read("episodes/SYN/job/events.jsonl") == original
        assert not any(".secrets" in name for name in stream.namelist())
        assert (
            json.loads(stream.read("MANIFEST.json"))["personal_fields_redacted"]
            is False
        )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("resume", []), ("retry_failed", [1, 2, 5]), ("restart_all", list(range(6)))],
)
def test_session_keeps_going_after_child_failure_and_native_crash(
    tmp_path, monkeypatch, mode, expected
):
    service = CollectorService(tmp_path / "profile", demo=True)
    data = Path(service.settings.data_dir)
    run = data / "runs/test"
    run.mkdir(parents=True)
    write_json(run / "control.json", {"command": "continue"})
    write_json(
        run / "run.json",
        {
            "run_token": "SYN_RUN",
            "settings": asdict(service.settings),
            "run_dir": str(run),
        },
    )
    seen = []

    class Child:
        def __init__(self, command, **kwargs):
            spec = read_json(Path(command[-1]))
            ordinal = int(spec["entry"]["source"]["job_id"].rsplit("-", 1)[-1])
            seen.append(ordinal)
            self.returncode = 99 if ordinal == 1 else 0
            if ordinal != 1:
                write_json(
                    Path(spec["result_file"]),
                    {"demo": True, "demo_passed": ordinal % 3 != 2},
                )

        def poll(self):
            return self.returncode

    monkeypatch.setattr(worker.subprocess, "Popen", Child)

    class Group:
        def assign(self, process):
            pass

        def close(self):
            pass

    monkeypatch.setattr(worker, "ProcessTree", Group)
    assert worker.run_session(run / "run.json") == 0
    state = read_json(run / "status.json")
    results = read_results(data / "demo_results.jsonl")
    assert seen == list(range(6))
    assert state["status"] == "finished" and state["attempted"] == 6
    assert len(results) == 6 and results[1]["fail_stage"] == "process_exit"
    assert not any(row.get("valid_collected_win") for row in results)
    seen.clear()
    spec = read_json(run / "run.json")
    spec[mode] = True
    write_json(run / "run.json", spec)
    assert worker.run_session(run / "run.json") == 0
    assert seen == expected
    assert len(read_results(data / "demo_results.jsonl")) == 6 + len(expected)


def test_stale_running_file_is_not_shown_as_active(tmp_path, monkeypatch):
    service = CollectorService(tmp_path)
    run = tmp_path / "run"
    service.settings.last_run_dir = str(run)
    write_json(run / "run.json", {"run_token": "SYN_RUN"})
    write_json(
        run / "status.json", {"run_token": "SYN_RUN", "status": "running", "pid": 1}
    )
    monkeypatch.setattr("bridge.collector.service.process_alive", lambda _: False)
    assert service.status()["status"] == "interrupted"


@pytest.mark.skipif(os.name != "nt", reason="Windows作业子进程隔离")
def test_owned_process_group_closes_only_its_child(tmp_path):
    from bridge.collector.storage import ProcessTree

    group = ProcessTree()
    child = subprocess.Popen(
        [
            getattr(sys, "_base_executable", sys.executable),
            "-c",
            "import time,sys; from pathlib import Path; "
            "time.sleep(30); Path(sys.argv[1]).write_text('completed')",
            str(tmp_path / "completed.txt"),
        ],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        group.assign(child)
        assert child.poll() is None
        group.close()
        child.wait(timeout=5)
        assert not (tmp_path / "completed.txt").exists()
    finally:
        group.close()
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_append_after_partial_record_preserves_new_completed_row(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('{"unfinished":', encoding="utf-8")
    append_json(path, {"completed": True})
    assert read_results(path) == [{"completed": True}]
    assert '"unfinished"' in path.read_text(encoding="utf-8")
