"""授权补给及人工确认边界回归，全部使用合成画面和假设备。"""

import numpy as np
from PIL import Image

from bridge import blocker_watchdog
from bridge.collector import recovery
from bridge.collector.recovery import CollectorSentry, locate_template


def test_template_match_reports_actual_center():
    rng = np.random.default_rng(42)
    frame = rng.uniform(0, 255, (40, 50, 3))
    template = rng.uniform(0, 255, (8, 9, 3))
    frame[10:18, 20:29] = template
    score, point = locate_template(frame, template, (0, 0, 50, 40))
    assert score > 0.999
    assert point == (24, 14)


class Device:
    def __init__(self, root, authorized):
        self._cfg = {
            "maa_dir": str(root),
            "resource_policy": {"allow_originite": authorized},
        }
        self.taps = []

    def screencap(self):
        return Image.new("RGB", (1920, 1080))

    def tap(self, x, y):
        self.taps.append((x, y))


def make_sentry(tmp_path, monkeypatch, authorized):
    clock = [0.0]
    monkeypatch.setattr(recovery.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(recovery, "locate_template", lambda *args: (0.999, (1100, 575)))
    device = Device(tmp_path, authorized)
    messages = []
    sentry = CollectorSentry(device, 0, on_message=messages.append)
    sentry._templates = {
        "UseStone.png": np.ones((2, 2, 3)),
        "MedicineConfirm.png": np.ones((2, 2, 3)),
    }
    return sentry, device, clock, messages


def test_unapproved_account_never_confirms_originite(tmp_path, monkeypatch):
    sentry, device, clock, messages = make_sentry(tmp_path, monkeypatch, False)
    sentry.poll()
    clock[0] = 3.0
    sentry.poll()
    assert not device.taps
    assert any("未授权" in message for message in messages)


def test_authorized_account_requires_two_confirmations_and_does_not_repeat(
    tmp_path, monkeypatch
):
    sentry, device, clock, _ = make_sentry(tmp_path, monkeypatch, True)
    sentry.poll()
    assert not device.taps
    clock[0] = 3.0
    sentry.poll()
    assert device.taps == [(1650, 862)]
    clock[0] = 25.0
    sentry.poll()
    assert len(device.taps) == 1
    assert sentry.events()[0]["kind"] == "authorized_originite_confirm"


def test_unknown_sync_dialog_is_not_dismissed_as_story_freeze(tmp_path, monkeypatch):
    sentry, device, clock, messages = make_sentry(tmp_path, monkeypatch, False)
    sentry._templates = {}
    sentry.set_battle_active(True)
    monkeypatch.setattr(
        blocker_watchdog,
        "detect_blocker",
        lambda image: blocker_watchdog.BLOCKER_SYNC_UPDATED,
    )
    sentry.poll()
    clock[0] = 30.0
    sentry.poll()
    assert not device.taps
    assert any("同步" in message for message in messages)
