"""玩家在MuMu直接操作的旁路记录；不启动MAA任务、不发送任何游戏输入。"""

from __future__ import annotations

import hashlib
import io
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

from PIL import Image

from bridge.battle_lifecycle import BattleLifecycleObserver, LifecycleObservation
from bridge.collector.discovery import validate_address
from bridge.collector.storage import (
    DeviceLease,
    Settings,
    append_json,
    now,
    process_birth,
    read_json,
    write_json,
)
from bridge.data_paths import data_path
from bridge.manual_input import GestureDecoder, input_devices, input_event


class ManualSession:
    """原图、原始输入和触点轨迹同轴保存；未知语义与不完整记录保持缺失。"""

    def __init__(
        self, directory: Path, stage_label: str, devices: list[dict], config: dict
    ):
        self.directory = directory
        (directory / "shots").mkdir(parents=True, exist_ok=False)
        self.config = config
        self.frames = self.events = self.gestures = self.incomplete = 0
        self.input_loss_events = 0
        self.active_seen = False
        self.result: str | None = None
        self.result_frames: list[str] = []
        self.home_frames: list[str] = []
        self.terminal_reason: str | None = None
        self.candidate: str | None = None
        self.meta = {
            "schema": "manual_session/1",
            "session_id": directory.name,
            "source": "player_manual",
            "operator_source": "user_declared",
            "input_capture": "android_getevent",
            "stage_label": stage_label or None,
            "started_at": now(),
            "phase": "recording",
            "battle_result": "unknown",
            "input_devices": devices,
            "time_basis": "host_monotonic_receipt",
            "alignment": "approximate_receipt_time_not_verified_action_state_pair",
            "coordinate_space": "android_input_absolute_untransformed",
            "tactical_action_labels": 0,
            "structured_battle_states": False,
            "semantic_actions": None,
        }
        self.lock = threading.Lock()
        write_json(directory / "session.json", self.meta)

    def input(self, event: dict, gestures: list[dict]) -> None:
        """输入按原事件追加；不把触点或按键自动改名成游戏战术动作。"""
        with self.lock:
            append_json(self.directory / "input_events.jsonl", event)
            self.events += 1
            self.input_loss_events += int(event["code"] == "SYN_DROPPED")
            for gesture in gestures:
                append_json(self.directory / "gestures.jsonl", gesture)
                self.gestures += 1
                self.incomplete += int(not gesture["complete"])

    def orphan(self, gestures: list[dict]) -> None:
        """停止、丢包时保留未完成触点，不补造释放事件。"""
        with self.lock:
            for gesture in gestures:
                append_json(self.directory / "gestures.jsonl", gesture)
                self.gestures += 1
                self.incomplete += int(not gesture["complete"])

    def frame(
        self,
        observed: LifecycleObservation,
        requested_ms: int,
        received_ms: int,
        *,
        original_png: bytes | None = None,
    ) -> bool:
        """保存实际采样原图及识别证据，只按真实结算判终，不按时长或冻结判终。"""
        self.frames += 1
        relative = f"shots/{self.frames:08d}.png"
        target = self.directory / relative
        temporary = target.with_suffix(".png.tmp")
        if original_png is None:
            observed.frame.save(temporary, format="PNG")
        else:
            temporary.write_bytes(original_png)
        temporary.replace(target)
        row = {
            "frame_id": self.frames,
            "path": relative,
            "request_t_ms": requested_ms,
            "received_t_ms": received_ms,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "width": observed.frame.width,
            "height": observed.frame.height,
            "active": observed.active,
            "result": observed.result,
            "home": observed.home,
            "scores": observed.scores,
        }
        append_json(self.directory / "frames.jsonl", row)
        self.active_seen |= observed.active
        if observed.result and self.active_seen:
            if self.candidate != observed.result:
                self.candidate, self.result_frames = observed.result, []
            self.result_frames.append(relative)
            if len(self.result_frames) >= self.config["result_confirmation_frames"]:
                self.result = observed.result
                self.terminal_reason = "observed_battle_result"
                return True
        else:
            self.candidate, self.result_frames = None, []
        if self.active_seen and observed.home and not observed.active:
            self.home_frames.append(relative)
            if len(self.home_frames) >= self.config["result_confirmation_frames"]:
                self.terminal_reason = "observed_home_after_battle_result_unknown"
                return True
        else:
            self.home_frames = []
        return False

    def finish(self, reason: str, *, failed: bool = False) -> dict:
        """元数据只陈述已观测事实；人工停止和错误不冒充已确认胜局。"""
        with self.lock:
            result = {
                **self.meta,
                "finished_at": now(),
                "phase": "failed" if failed else "complete",
                "stop_reason": reason,
                "battle_result": self.result or "unknown",
                "result_evidence": self.result_frames if self.result else [],
                "terminal_evidence": self.result_frames
                if self.result
                else self.home_frames,
                "frames": self.frames,
                "input_events": self.events,
                "gestures": self.gestures,
                "incomplete_gestures": self.incomplete,
                "input_loss_events": self.input_loss_events,
                "no_input_loss_observed": not failed
                and not self.input_loss_events
                and not self.incomplete,
                "battle_observed": self.active_seen,
                "policy_imitation_ready": False,
            }
            write_json(self.directory / "session.json", result)
            return result


def foreground_package(text: str) -> str | None:
    """只取Android当前前台包名，不保存其他应用列表。"""
    for field in ("topResumedActivity", "mResumedActivity"):
        for line in text.splitlines():
            if field in line:
                match = re.search(r"\b([A-Za-z][A-Za-z0-9_.]+)/[A-Za-z0-9_.$]+", line)
                if match:
                    return match[1]
    return None


def run_manual(spec_path: Path) -> int:
    """后台旁路录制一局；停止只关闭自身读取进程，不关闭游戏或其他采集任务。"""
    spec = read_json(spec_path)
    settings = Settings(**spec["settings"])
    data = data_path(settings.data_dir)
    directory = data_path(spec["session_dir"])
    if not directory.is_relative_to(data / "manual_sessions"):
        raise ValueError("手动录制目录必须位于所选数据根的manual_sessions中")
    validate_address(settings.device_addr)
    config = read_json(
        Path(__file__).parent / "collector/resources/manual_capture.json"
    )
    run_dir = Path(spec["run_dir"])
    stopped, capture, wake = threading.Event(), threading.Event(), threading.Event()
    faults: queue.Queue[Exception] = queue.Queue()
    state_lock = threading.Lock()
    state = {
        "pid": os.getpid(),
        "process_birth": process_birth(os.getpid()),
        "run_token": spec["run_token"],
        "status": "starting",
        "manual": True,
        "message": "正在准备旁路录制，不会操作游戏",
        "total": 1,
        "attempted": 0,
        "valid": 0,
        "review": 0,
        "current_stage": spec.get("stage_label") or "玩家手动关卡",
    }
    session: ManualSession | None = None
    processes: list[subprocess.Popen] = []
    threads: list[threading.Thread] = []
    start = time.monotonic()
    profile = DeviceLease(f"profile:{spec['profile_home']}")
    identity = (
        f"mumu:{Path(settings.mumu_manager).resolve()}:{settings.instance_index}"
        if settings.instance_index
        else settings.device_addr
    )
    device_lease = DeviceLease(identity)

    def stamp() -> int:
        return int((time.monotonic() - start) * 1000)

    def update(**values):
        with state_lock:
            state.update(values)

    def heartbeat():
        while not stopped.is_set():
            try:
                with state_lock:
                    value = {
                        **state,
                        "heartbeat_at": now(),
                        "heartbeat_epoch": time.time(),
                    }
                write_json(run_dir / "status.json", value)
                write_json(Path(spec["live_file"]), value)
            except Exception as error:
                faults.put(error)
                return
            stopped.wait(1)

    def command(*args: str, binary=False):
        response = subprocess.run(
            [settings.adb_path, "-s", settings.device_addr, *args],
            capture_output=True,
            timeout=config["device_call_timeout_s"],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if response.returncode:
            raise RuntimeError("所选MuMu只读采集命令失败，请检查连接和读取权限")
        return (
            response.stdout
            if binary
            else response.stdout.decode("utf-8", errors="replace")
        )

    def read_input(process, device):
        decoder = GestureDecoder(device)
        try:
            for line in process.stdout:
                if stopped.is_set():
                    break
                event = input_event(line, device["path"], stamp())
                if event is None and line.lstrip().startswith("["):
                    event = {
                        "device": device["path"],
                        "t_ms": stamp(),
                        "code": "UNPARSED",
                        "type": "UNPARSED",
                        "value": None,
                        "raw": line.rstrip(),
                        "time_basis": "host_receipt",
                        "device_t_s": None,
                    }
                if event is not None and capture.is_set():
                    completed = decoder.feed(event)
                    session.input(event, completed)
                    if completed or event["code"] in (
                        "ABS_MT_TRACKING_ID",
                        "BTN_MOUSE",
                    ):
                        wake.set()
            if not stopped.is_set():
                raise RuntimeError("原始输入读取通道退出；记录保留，不能冒充完整操作")
        except Exception as error:
            faults.put(error)
        finally:
            if session:
                try:
                    session.orphan(decoder.flush("recording_stopped_or_input_lost"))
                except Exception as error:
                    faults.put(error)

    pulse = threading.Thread(target=heartbeat, daemon=True)
    pulse.start()
    reason, failed = "user_stopped", False
    try:
        if not profile.acquire() or not device_lease.acquire():
            raise RuntimeError("当前档案或所选MuMu已有采集任务，请先停止自动采集")
        if (
            foreground_package(command("shell", "dumpsys", "activity", "activities"))
            not in config["game_packages"]
        ):
            raise RuntimeError("请先在所选MuMu中打开已登录的明日方舟，再开始录制")
        devices = input_devices(command("shell", "getevent", "-lp"))
        if not devices:
            raise RuntimeError("MuMu没有提供可读取的原始输入通道，录制未开始")
        observer = BattleLifecycleObserver(settings.maa_dir)
        session = ManualSession(directory, spec.get("stage_label", ""), devices, config)
        if settings.roster_file and Path(settings.roster_file).is_file():
            write_json(directory / "roster.json", read_json(Path(settings.roster_file)))
        capture.set()
        for device in devices:
            process = subprocess.Popen(
                [
                    settings.adb_path,
                    "-s",
                    settings.device_addr,
                    "shell",
                    "getevent",
                    "-lt",
                    device["path"],
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            processes.append(process)
            thread = threading.Thread(
                target=read_input, args=(process, device), daemon=True
            )
            threads.append(thread)
            thread.start()
        update(status="running", message="正在记录玩家操作；请直接在MuMu内操作游戏")
        while True:
            if not faults.empty():
                raise faults.get()
            control = read_json(run_dir / "control.json", {}) or {}
            if control.get("command") == "stop_now":
                break
            if control.get("command") in ("pause_after", "stop_after"):
                update(message="本局真实结算后结束录制；如需中止请点停止录制")
            package = foreground_package(
                command("shell", "dumpsys", "activity", "activities")
            )
            if package not in config["game_packages"]:
                raise RuntimeError("游戏已离开前台，停止旁路记录；半局保留待复核")
            requested = stamp()
            original_png = command("exec-out", "screencap", "-p", binary=True)
            frame = Image.open(io.BytesIO(original_png)).convert("RGB")
            received = stamp()
            if (
                foreground_package(
                    command("shell", "dumpsys", "activity", "activities")
                )
                != package
            ):
                raise RuntimeError("取帧期间前台应用改变，该帧未保存")
            if session.frame(
                observer.observe(frame), requested, received, original_png=original_png
            ):
                reason = session.terminal_reason or "observed_battle_result"
                break
            update(
                manual_frames=session.frames,
                manual_events=session.events,
                manual_gestures=session.gestures,
                battle_observed=session.active_seen,
            )
            wake.wait(config["frame_interval_s"])
            wake.clear()
    except Exception as error:
        failed, reason = True, f"{type(error).__name__}: {error}"
    finally:
        capture.clear()
        stopped.set()
        for process in processes:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        for thread in threads:
            thread.join(timeout=5)
        pulse.join(timeout=5)
        if not faults.empty():
            failed, reason = True, str(faults.get())
        try:
            result = session.finish(reason, failed=failed) if session else None
            update(
                status="error" if failed else "finished",
                message=reason
                if failed
                else "手动录制已保存，语义动作未补造，可导出画面与输入进行复核",
                attempted=int(session is not None),
                review=int(session is not None),
                manual_frames=session.frames if session else 0,
                manual_events=session.events if session else 0,
                manual_gestures=session.gestures if session else 0,
                manual_session=str(directory),
                manual_result=result,
                heartbeat_at=now(),
                heartbeat_epoch=time.time(),
            )
            write_json(run_dir / "status.json", state)
            write_json(Path(spec["live_file"]), state)
        finally:
            device_lease.close()
            profile.close()
    return int(failed)
