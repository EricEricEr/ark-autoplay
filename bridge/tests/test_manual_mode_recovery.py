"""手动录制的恢复与界面入口不允许悄悄转成自动作业。"""

from types import SimpleNamespace

import pytest

from bridge.collector.app import CollectorApp
from bridge.collector.service import CollectorService


def test_manual_recovery_cannot_replay_previously_imported_jobs(tmp_path, monkeypatch):
    service = CollectorService(tmp_path / "profile")
    service.set_data_directory(tmp_path / "data")
    service.settings.extras.update(
        last_capture_mode="manual", manual_stage_label="备注"
    )
    service.settings.queue_file = str(tmp_path / "old-queue.json")
    calls = []
    monkeypatch.setattr(service, "start_manual", lambda stage: calls.append(stage))
    monkeypatch.setattr(service, "start", lambda: pytest.fail("不能恢复成自动作业"))
    service.recover()
    assert calls == ["备注"]


def test_job_import_explicitly_switches_back_to_job_mode(tmp_path, monkeypatch):
    service = CollectorService(tmp_path / "profile")
    service.set_data_directory(tmp_path / "data")
    service.settings.extras["last_capture_mode"] = "manual"
    monkeypatch.setattr(
        "bridge.collector.service.import_queue",
        lambda *args: (tmp_path / "queue.json", {"jobs": 1}),
    )
    service.import_tasks(tmp_path / "provided-queue.json")
    assert service.settings.extras["last_capture_mode"] == "jobs"


@pytest.mark.parametrize("flags", [{}, {"retry_failed": True}, {"restart_all": True}])
def test_all_generic_start_buttons_stay_in_manual_mode(tmp_path, flags):
    service = CollectorService(tmp_path / "profile")
    service.settings.extras["last_capture_mode"] = "manual"
    opened = []
    window = SimpleNamespace(service=service, open_manual=lambda: opened.append(True))
    CollectorApp.start(window, **flags)
    assert opened == [True]


def test_generic_service_start_cannot_replay_old_queue_in_manual_mode(tmp_path):
    service = CollectorService(tmp_path / "profile")
    service.settings.extras["last_capture_mode"] = "manual"
    with pytest.raises(ValueError, match="手动录制模式"):
        service.start(retry_failed=True)
