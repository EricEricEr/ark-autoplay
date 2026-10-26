"""标准采集入口故障隔离：单局异常继续、人工停止生效、截图线程不串局。"""

import json

import pytest

from bridge import replay_controller as replay


class Driver:
    """不连接设备的采集驱动替身。"""

    def __init__(self, *, connect=True, stop_error=False):
        self.connected = connect
        self.stop_error = stop_error
        self.stops = 0
        self.closed = False

    def connect(self):
        return self.connected

    def stop(self):
        self.stops += 1
        if self.stop_error:
            raise RuntimeError("disconnected")

    def close(self):
        self.closed = True


def make_controller(tmp_path, monkeypatch, count=3, **driver_options):
    """只装配队列循环，不加载DLL、游戏或真实配置。"""
    controller = replay.ReplayController.__new__(replay.ReplayController)
    controller._work = tmp_path
    controller._driver = Driver(**driver_options)
    entries = [
        {
            "source": {"job_id": index, "stage_id_raw": "synthetic"},
            "maa_job": {"stage_name": "SYN-1"},
        }
        for index in range(count)
    ]
    monkeypatch.setattr(replay, "load_jobs", lambda _: entries)
    monkeypatch.setattr(replay, "wait_idle", lambda *args, **kwargs: True)
    return controller


@pytest.mark.parametrize("stop_error", [False, True])
def test_exception_and_failed_recovery_do_not_stop_later_jobs(
    tmp_path, monkeypatch, stop_error
):
    controller = make_controller(tmp_path, monkeypatch, stop_error=stop_error)
    attempted = []

    def run(entry):
        index = entry["source"]["job_id"]
        attempted.append(index)
        if index == 0:
            raise RuntimeError("synthetic screencap failure")
        return {"job_id": index, "ok": True}

    monkeypatch.setattr(controller, "_run_one", run)
    results = controller.run("unused")
    assert attempted == [0, 1, 2]
    assert [result["ok"] for result in results] == [False, True, True]
    assert results[0]["fail_stage"] == "exception"
    assert ("recovery_error" in results[0]) == stop_error
    journal = next((tmp_path / "debug").glob("factory_summary_*.jsonl"))
    assert [
        json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()
    ] == results
    assert controller._driver.closed


def test_manual_interrupt_preserves_prior_results_and_is_not_swallowed(
    tmp_path, monkeypatch
):
    controller = make_controller(tmp_path, monkeypatch)
    attempted = []

    def run(entry):
        index = entry["source"]["job_id"]
        attempted.append(index)
        if index == 1:
            raise KeyboardInterrupt
        return {"job_id": index, "ok": True}

    monkeypatch.setattr(controller, "_run_one", run)
    with pytest.raises(KeyboardInterrupt):
        controller.run("unused")
    assert attempted == [0, 1]
    summary = next((tmp_path / "debug").glob("factory_summary_*.json"))
    assert json.loads(summary.read_text(encoding="utf-8")) == [
        {"job_id": 0, "ok": True}
    ]
    assert controller._driver.closed


def test_logger_is_stopped_even_when_episode_raises(tmp_path, monkeypatch):
    controller = make_controller(tmp_path, monkeypatch)
    stops = []

    class Logger:
        def stop(self):
            stops.append(True)

    def body(entry):
        controller._episode_logger = Logger()
        raise RuntimeError("synthetic capture failure")

    monkeypatch.setattr(controller, "_run_one_body", body)
    with pytest.raises(RuntimeError, match="synthetic capture failure"):
        controller._run_one({})
    assert stops == [True]
    assert controller._episode_logger is None


def test_connection_failure_is_reported_and_driver_closed(tmp_path, monkeypatch):
    controller = make_controller(tmp_path, monkeypatch, connect=False)
    with pytest.raises(RuntimeError, match="MAA"):
        controller.run("unused")
    assert controller._driver.closed
