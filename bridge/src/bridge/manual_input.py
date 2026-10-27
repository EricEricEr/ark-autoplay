"""MuMu原始输入解析：保存Android事件与触点轨迹，不推测部署、技能或朝向。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def input_devices(text: str) -> list[dict[str, Any]]:
    """识别虚拟触屏、鼠标、键盘和手柄；电源及休眠按钮不属于操作采集。"""
    result = []
    for block in re.split(r"(?=add device \d+:)", text):
        match = re.search(r"add device \d+:\s*(/dev/input/event\d+)", block)
        name = re.search(r'name:\s*"([^"]+)"', block)
        if not match or not name:
            continue
        axes = {
            m[0]: {"min": int(m[1]), "max": int(m[2])}
            for m in re.findall(
                r"(ABS_\w+)\s*:.*?min\s+(-?\d+),\s*max\s+(-?\d+)", block
            )
        }
        if {"ABS_MT_POSITION_X", "ABS_MT_POSITION_Y"} <= axes.keys():
            kind = "multitouch"
        elif "joystick" in name[1].lower():
            kind = "gamepad"
        elif {"ABS_X", "ABS_Y"} <= axes.keys() and "BTN_MOUSE" in block:
            kind = "absolute_pointer"
        elif "keyboard" in name[1].lower():
            kind = "keyboard"
        elif "mouse" in name[1].lower():
            kind = "relative_pointer"
        else:
            continue
        result.append(
            {
                "path": match[1],
                "name": name[1],
                "kind": kind,
                "axes": axes,
                "capabilities_raw": block,
            }
        )
    return result


def input_event(line: str, device: str, t_ms: int) -> dict[str, Any] | None:
    """解析getevent -lt；设备时钟与主机接收时间分别保留，不伪装成精确同步。"""
    match = re.match(
        r"\s*\[\s*(\d+\.\d+)\]\s*(?:/dev/input/event\d+:\s*)?"
        r"(EV_\w+|[0-9a-f]{4})\s+(\w+)\s+(\w+)\s*$",
        line,
    )
    if not match:
        return None
    value = match[4]
    number = {"DOWN": 1, "UP": 0, "REPEAT": 2}.get(value)
    if number is None:
        try:
            number = int(value, 16)
            if number >= 0x80000000:
                number -= 0x100000000
        except ValueError:
            number = None
    return {
        "device": device,
        "device_t_s": float(match[1]),
        "t_ms": t_ms,
        "time_basis": "host_receipt",
        "type": match[2],
        "code": match[3],
        "value": number,
        "value_raw": value,
        "raw": line.rstrip("\r\n"),
    }


@dataclass
class _Contact:
    x: int | None = None
    y: int | None = None
    active: bool = False
    ending: bool = False
    started: int | None = None
    tracking_id: int | None = None
    points: list[dict] = field(default_factory=list)


class GestureDecoder:
    """按SYN_REPORT组装原始触点；丢包及停止保留不完整轨迹，不补造抬起。"""

    def __init__(self, device: dict[str, Any]) -> None:
        self.device = device
        self.slot = 0
        self.contacts: dict[int, _Contact] = {}
        self.dropped = 0

    def _finish(
        self, slot: int, contact: _Contact, complete: bool, reason: str
    ) -> dict:
        result = {
            "device": self.device["path"],
            "slot": slot,
            "tracking_id": contact.tracking_id,
            "coordinate_space": "android_input_absolute",
            "started_t_ms": contact.started,
            "ended_t_ms": contact.points[-1]["t_ms"] if contact.points else None,
            "points": contact.points,
            "complete": complete,
            "reason": reason,
            "semantic_action": None,
            "screen_coordinates_verified": False,
        }
        contact.active, contact.ending, contact.started, contact.points = (
            False,
            False,
            None,
            [],
        )
        contact.tracking_id = None
        return result

    def feed(self, event: dict[str, Any]) -> list[dict]:
        """消费一条真实输入；所有点原样保留，不截断长拖动或多指轨迹。"""
        code, value = event["code"], event["value"]
        result = []
        if code == "SYN_DROPPED":
            self.dropped += 1
            return self.flush("kernel_syn_dropped")
        if self.device["kind"] not in ("multitouch", "absolute_pointer"):
            return []
        if code == "ABS_MT_SLOT" and isinstance(value, int):
            self.slot = value
        contact = self.contacts.setdefault(self.slot, _Contact())
        if code in ("ABS_MT_POSITION_X", "ABS_X"):
            contact.x = value
        elif code in ("ABS_MT_POSITION_Y", "ABS_Y"):
            contact.y = value
        elif code == "ABS_MT_TRACKING_ID":
            if value == -1:
                contact.ending = contact.active
            elif contact.active and value != contact.tracking_id:
                result.append(
                    self._finish(
                        self.slot, contact, False, "tracking_id_changed_without_release"
                    )
                )
                contact.active, contact.started = True, event["t_ms"]
            elif not contact.active:
                contact.active, contact.started = True, event["t_ms"]
            if value != -1:
                contact.tracking_id = value
        elif self.device["kind"] == "absolute_pointer" and code in (
            "BTN_MOUSE",
            "BTN_TOUCH",
        ):
            if value == 1 and not contact.active:
                contact.active, contact.started = True, event["t_ms"]
            elif value == 0:
                contact.ending = contact.active
        if code == "SYN_REPORT":
            for slot, current in self.contacts.items():
                if current.active:
                    current.points.append(
                        {
                            "t_ms": event["t_ms"],
                            "device_t_s": event["device_t_s"],
                            "x": current.x,
                            "y": current.y,
                        }
                    )
                    if current.ending:
                        result.append(
                            self._finish(slot, current, True, "observed_release")
                        )
        return result

    def flush(self, reason: str) -> list[dict]:
        """保留尚未抬起的原始轨迹，标为不完整而不伪造终点。"""
        return [
            self._finish(slot, value, False, reason)
            for slot, value in self.contacts.items()
            if value.active
        ]
