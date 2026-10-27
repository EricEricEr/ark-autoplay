"""手动旁路采集回归：真实输入解析、保留缺失、结算证据、导出与只读后端。"""

import io
import json
import subprocess
import threading
import zipfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from bridge.battle_lifecycle import LifecycleObservation
from bridge.collector.exporting import export_data
from bridge.collector.service import CollectorService
from bridge.collector.storage import Settings, read_json, write_json
from bridge.convert_episodes import convert
from bridge.manual_capture import ManualSession, foreground_package, run_manual
from bridge.manual_convert import convert_manual
from bridge.manual_input import GestureDecoder, input_devices, input_event

CAPABILITIES = """add device 1: /dev/input/event4
  name: "Synthetic Touchscreen"
    ABS_MT_SLOT : value 0, min 0, max 31, fuzz 0
    ABS_MT_POSITION_X : value 0, min 0, max 1080, fuzz 0
    ABS_MT_POSITION_Y : value 0, min 0, max 1920, fuzz 0
add device 2: /dev/input/event5
  name: "Synthetic Keyboard"
    KEY_A KEY_ENTER
add device 3: /dev/input/event0
  name: "Power Button"
    KEY_POWER
"""


def event(code, value, t=1, kind="EV_ABS"):
    result = input_event(
        f"[ 123.000001] {kind} {code} {value}\n", "/dev/input/event4", t
    )
    assert result is not None
    return result


def observation(*, active=False, result=None):
    return LifecycleObservation(
        active, result, False, {"synthetic": 1.0}, Image.new("RGB", (8, 8), "white")
    )


def session_fixture(tmp_path, *, finalize=True):
    root = tmp_path / "data"
    source = root / "manual_sessions/synthetic"
    session = ManualSession(
        source,
        "合成关卡备注",
        input_devices(CAPABILITIES),
        {"result_confirmation_frames": 2},
    )
    session.frame(observation(active=True), 0, 10)
    session.input(event("ABS_MT_POSITION_X", "00000064", t=15), [])
    session.frame(observation(active=True), 20, 30)
    if finalize:
        session.finish("user_stopped")
    return root, source, session


def test_native_capabilities_preserve_axes_and_filter_power_button():
    devices = input_devices(CAPABILITIES)
    assert [d["kind"] for d in devices] == ["multitouch", "keyboard"]
    assert devices[0]["axes"]["ABS_MT_POSITION_Y"] == {"min": 0, "max": 1920}
    assert (
        foreground_package(
            "topResumedActivity=ActivityRecord{x u0 com.hypergryph.arknights/.Main t3}"
        )
        == "com.hypergryph.arknights"
    )
    assert foreground_package("other application list") is None


def test_native_event_keeps_both_clocks_and_unknown_values():
    raw = "[ 27.125000] /dev/input/event4: EV_ABS ABS_MT_TRACKING_ID ffffffff"
    row = input_event(raw, "/dev/input/event4", 190)
    assert row["value"] == -1 and row["device_t_s"] == 27.125
    assert row["t_ms"] == 190 and row["raw"] == raw
    assert input_event("device header", "/dev/input/event4", 0) is None


def test_long_touch_trace_has_no_action_or_point_cap():
    decoder = GestureDecoder(input_devices(CAPABILITIES)[0])
    decoder.feed(event("ABS_MT_TRACKING_ID", "00000000", 0))
    decoder.feed(event("ABS_MT_POSITION_Y", "00000032", 0))
    for index in range(350):
        decoder.feed(event("ABS_MT_POSITION_X", f"{index:08x}", index))
        assert decoder.feed(event("SYN_REPORT", "00000000", index, "EV_SYN")) == []
    decoder.feed(event("ABS_MT_TRACKING_ID", "ffffffff", 351))
    trace = decoder.feed(event("SYN_REPORT", "00000000", 351, "EV_SYN"))[0]
    assert len(trace["points"]) == 351 and trace["complete"]
    assert trace["points"][349]["x"] == 349
    assert trace["semantic_action"] is None and not trace["screen_coordinates_verified"]


def test_unreleased_or_dropped_touch_is_not_completed_by_guessing():
    decoder = GestureDecoder(input_devices(CAPABILITIES)[0])
    decoder.feed(event("ABS_MT_TRACKING_ID", "00000001"))
    decoder.feed(event("SYN_REPORT", "00000000", 2, "EV_SYN"))
    trace = decoder.feed(event("SYN_DROPPED", "00000000", 3, "EV_SYN"))[0]
    assert not trace["complete"] and trace["points"][0]["x"] is None
    assert decoder.dropped == 1 and decoder.flush("stop") == []


def test_tracking_id_change_keeps_previous_contact_incomplete():
    decoder = GestureDecoder(input_devices(CAPABILITIES)[0])
    decoder.feed(event("ABS_MT_TRACKING_ID", "00000001"))
    decoder.feed(event("SYN_REPORT", "00000000", 2, "EV_SYN"))
    incomplete = decoder.feed(event("ABS_MT_TRACKING_ID", "00000002", 3))[0]
    assert not incomplete["complete"] and incomplete["tracking_id"] == 1
    assert decoder.flush("user_stopped")[0]["tracking_id"] == 2


def test_keyboard_events_are_preserved_without_inventing_skills(tmp_path):
    root, source, session = session_fixture(tmp_path, finalize=False)
    keyboard = input_devices(CAPABILITIES)[1]
    row = input_event("[ 123.1] EV_KEY KEY_A DOWN", keyboard["path"], 40)
    assert row is not None
    session.input(row, GestureDecoder(keyboard).feed(row))
    session.finish("user_stopped")
    report = convert_manual(root, tmp_path / "out")
    assert report["counts"]["input_events"] == 2
    assert report["counts"]["gestures"] == 0
    assert report["counts"]["tactical_action_labels"] == 0
    assert "KEY_A" in (source / "input_events.jsonl").read_text(encoding="utf-8")


def test_real_result_requires_battle_and_confirmed_frames(tmp_path):
    session = ManualSession(
        tmp_path / "session", "", [], {"result_confirmation_frames": 2}
    )
    assert not session.frame(observation(result="win"), 0, 1)
    assert not session.frame(observation(active=True), 2, 3)
    assert not session.frame(observation(result="win"), 4, 5)
    assert session.frame(observation(result="win"), 6, 7)
    meta = session.finish("observed_battle_result")
    assert meta["battle_result"] == "win" and len(meta["result_evidence"]) == 2
    assert meta["tactical_action_labels"] == 0 and not meta["policy_imitation_ready"]


def test_received_png_bytes_are_preserved_without_reencoding(tmp_path):
    session = ManualSession(
        tmp_path / "session", "", [], {"result_confirmation_frames": 2}
    )
    buffer = io.BytesIO()
    Image.new("RGBA", (8, 8), (3, 4, 5, 255)).save(
        buffer, format="PNG", compress_level=0
    )
    payload = buffer.getvalue()
    session.frame(observation(active=True), 0, 1, original_png=payload)
    assert (session.directory / "shots/00000001.png").read_bytes() == payload


def test_long_frozen_battle_does_not_end_by_frame_count(tmp_path):
    session = ManualSession(
        tmp_path / "session", "", [], {"result_confirmation_frames": 2}
    )
    for index in range(135):
        assert not session.frame(observation(active=True), index, index + 1)
    assert session.finish("user_stopped")["battle_result"] == "unknown"


def test_returning_home_ends_unknown_without_fabricating_a_victory(tmp_path):
    session = ManualSession(
        tmp_path / "session", "", [], {"result_confirmation_frames": 2}
    )
    session.frame(observation(active=True), 0, 1)
    home = observation()
    home.home = True
    assert not session.frame(home, 2, 3)
    assert session.frame(home, 4, 5)
    meta = session.finish(session.terminal_reason)
    assert meta["battle_result"] == "unknown" and meta["result_evidence"] == []
    assert len(meta["terminal_evidence"]) == 2


def test_top_resumed_activity_wins_over_old_resumed_game_entry():
    text = (
        "mResumedActivity: ActivityRecord{x com.hypergryph.arknights/.Main}\n"
        "topResumedActivity: ActivityRecord{y synthetic.other/.Main}"
    )
    assert foreground_package(text) == "synthetic.other"


def test_manual_conversion_pairs_only_approximate_references_and_keeps_originals(
    tmp_path,
):
    root, source, _ = session_fixture(tmp_path)
    original = (source / "input_events.jsonl").read_bytes()
    out = tmp_path / "converted"
    report = convert(root, out)
    assert report["counts"]["accepted_sessions"] == 1
    sample = json.loads(
        (out / "manual_input_samples.jsonl").read_text(encoding="utf-8")
    )
    assert sample["before_frame"].endswith("00000001.png")
    assert sample["after_frame"].endswith("00000002.png")
    assert (
        sample["tactical_action"] is None
        and not sample["existing_policy_imitation_eligible"]
    )
    assert (source / "input_events.jsonl").read_bytes() == original
    assert not list(out.rglob("job.json"))


@pytest.mark.parametrize("fault", ["unfinalized", "missing_frame", "corrupt_frame"])
def test_incomplete_or_changed_manual_material_is_rejected_without_fake_labels(
    tmp_path, fault
):
    root, source, _ = session_fixture(tmp_path, finalize=fault != "unfinalized")
    image = source / "shots/00000001.png"
    if fault == "missing_frame":
        image.unlink()
    elif fault == "corrupt_frame":
        image.write_bytes(b"synthetic-corruption")
    report = convert_manual(root, tmp_path / "converted")
    assert report["counts"]["accepted_sessions"] == 0
    assert report["counts"]["rejected_sessions"] == 1
    assert report["counts"]["tactical_action_labels"] == 0
    assert source.is_dir()


def test_manual_export_has_frames_and_no_expert_queue_or_credentials(tmp_path):
    root, _, _ = session_fixture(tmp_path)
    write_json(root / ".secrets/credentials.json", {"cred": "SYN_PRIVATE"})
    write_json(root / "episodes/other/episode.json", {"synthetic": True})
    archive = tmp_path / "manual.zip"
    export_data(root, archive, include_images=True, manual_only=True)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert any(name.endswith(".png") for name in names)
        assert not any(
            name.startswith("episodes/") or ".secrets" in name for name in names
        )
        assert not json.loads(bundle.read("MANIFEST.json"))["manual_images_omitted"]


def test_manual_images_omission_is_explicit(tmp_path):
    root, _, _ = session_fixture(tmp_path)
    export_data(root, tmp_path / "without-images.zip")
    with zipfile.ZipFile(tmp_path / "without-images.zip") as bundle:
        assert json.loads(bundle.read("MANIFEST.json"))["manual_images_omitted"]


def test_absolute_frame_name_cannot_overwrite_original_during_conversion(tmp_path):
    root, source, _ = session_fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in (source / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    original = source / rows[0]["path"]
    before = original.read_bytes()
    rows[0]["path"] = str(original.resolve())
    (source / "frames.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    report = convert_manual(root, tmp_path / "out")
    assert report["counts"]["rejected_sessions"] == 1
    assert original.read_bytes() == before


def test_manual_start_does_not_require_queue_or_start_maa(tmp_path, monkeypatch):
    service = CollectorService(tmp_path / "profile")
    service.set_data_directory(tmp_path / "data")
    service.settings.adb_path, service.settings.device_addr = (
        "synthetic-adb.exe",
        "127.0.0.1:19999",
    )
    service.settings.maa_dir = str(tmp_path / "templates")
    calls = []

    def spawn(arguments, **kwargs):
        calls.append(arguments)
        return SimpleNamespace(pid=123, poll=lambda: None)

    monkeypatch.setattr("bridge.collector.service.subprocess.Popen", spawn)
    run = service.start_manual("合成备注")
    assert "--manual" in calls[0]
    spec = read_json(run / "run.json")
    assert spec["manual"] and spec["settings"]["queue_file"] == ""
    assert not Path(spec["session_dir"]).exists()


def test_readonly_backend_records_actual_stream_and_result_with_fake_device(
    tmp_path, monkeypatch
):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        adb_path="synthetic-adb",
        device_addr="127.0.0.1:19998",
        maa_dir=str(tmp_path / "runtime"),
    )
    run = tmp_path / "run"
    spec = {
        "run_token": "SYN_MANUAL",
        "settings": asdict(settings),
        "run_dir": str(run),
        "session_dir": str(tmp_path / "data/manual_sessions/one"),
        "profile_home": str(tmp_path / "profile"),
        "live_file": str(tmp_path / "live.json"),
    }
    write_json(run / "run.json", spec)
    image = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(image, format="PNG")
    commands = []

    def command(arguments, **kwargs):
        commands.append(arguments)
        if "getevent" in arguments:
            stdout = CAPABILITIES.split("add device 2:")[0].encode()
        elif "dumpsys" in arguments:
            stdout = (
                b"mResumedActivity: ActivityRecord{x com.hypergryph.arknights/.Main}"
            )
        else:
            assert "screencap" in arguments
            stdout = image.getvalue()
        return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr=b"")

    class InputProcess:
        def __init__(self, arguments, **kwargs):
            assert arguments[-2:] == ["-lt", "/dev/input/event4"]
            self.done = threading.Event()
            self.stdout = self.lines()

        def lines(self):
            yield "[ 123.000000] EV_ABS ABS_MT_TRACKING_ID 00000001\n"
            yield "[ 123.000000] EV_ABS ABS_MT_POSITION_X 00000010\n"
            yield "[ 123.000000] EV_ABS ABS_MT_POSITION_Y 00000020\n"
            yield "[ 123.000000] EV_SYN SYN_REPORT 00000000\n"
            yield "[ 123.100000] EV_ABS ABS_MT_TRACKING_ID ffffffff\n"
            yield "[ 123.100000] EV_SYN SYN_REPORT 00000000\n"
            self.done.wait(10)

        def poll(self):
            return 0 if self.done.is_set() else None

        def terminate(self):
            self.done.set()

        def wait(self, **kwargs):
            self.done.wait(1)
            return 0

    states = iter(
        [observation(active=True), observation(result="win"), observation(result="win")]
    )
    monkeypatch.setattr("bridge.manual_capture.subprocess.run", command)
    monkeypatch.setattr("bridge.manual_capture.subprocess.Popen", InputProcess)
    monkeypatch.setattr(
        "bridge.manual_capture.BattleLifecycleObserver",
        lambda _: SimpleNamespace(observe=lambda frame: next(states)),
    )
    assert run_manual(run / "run.json") == 0
    meta = read_json(Path(spec["session_dir"]) / "session.json")
    assert meta["frames"] == 3 and meta["input_events"] == 6 and meta["gestures"] == 1
    assert meta["battle_result"] == "win" and meta["tactical_action_labels"] == 0
    assert read_json(tmp_path / "live.json")["status"] == "finished"
    assert all("input" not in args and "am" not in args for args in commands)
