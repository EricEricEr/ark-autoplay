"""桌面采集执行器：子进程隔离、恢复连接、局末暂停与可靠进度记录。"""

from __future__ import annotations

import _thread
import hashlib
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bridge.candidate_coverage import execution_key
from bridge.collection_quality import collection_status

from .discovery import (
    instance_config,
    list_devices,
    run_command,
    validate_address,
)
from .storage import (
    DeviceLease,
    ProcessTree,
    Settings,
    append_json,
    command_for,
    now,
    process_birth,
    read_json,
    read_results,
    write_json,
)
from .task_package import load_queue


def nav_file() -> Path:
    """获取随软件打包的导航配置，兼容源码运行。"""
    packaged = Path(__file__).parent / "resources/nav_main.yaml"
    return (
        packaged
        if packaged.exists()
        else Path(__file__).resolve().parents[3] / "configs/nav_main.yaml"
    )


def stop_owned_process(process: subprocess.Popen) -> None:
    """仅结束本执行器刚创建的子进程树，不按程序名批量停止用户软件。"""
    if process.poll() is not None:
        return
    if os.name == "nt":
        run_command(["taskkill", "/PID", str(process.pid), "/T", "/F"])
    else:
        process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()


def control(run_dir: Path) -> str:
    """控制文件由界面写入；玩家无需处理文件。"""
    return str(
        (read_json(run_dir / "control.json", {}) or {}).get("command") or "continue"
    )


def prepare_device(
    settings: Settings,
    update: Callable[..., None],
    stopped: Callable[[], bool],
    *,
    check_display: bool = True,
) -> None:
    """连接中断时保留队列，恢复选定实例；不限制重连次数、不确认未知弹窗。"""
    if settings.demo:
        return
    launched = False
    while not stopped():
        if settings.mumu_manager and settings.instance_index:
            devices = list_devices(settings.mumu_manager)
            selected = next(
                (
                    device
                    for device in devices
                    if device.index == settings.instance_index
                ),
                None,
            )
            if selected and selected.address:
                settings.device_addr = selected.address
            if (
                selected
                and not selected.running
                and settings.automatic_recovery
                and not launched
            ):
                if not settings.instance_index.isdigit():
                    raise ValueError("模拟器编号无效")
                update(status="recovering", message="正在启动所选模拟器，请稍候")
                run_command(
                    [
                        settings.mumu_manager,
                        "control",
                        "-v",
                        settings.instance_index,
                        "launch",
                    ],
                    timeout=30,
                )
                launched = True
        if settings.device_addr:
            validate_address(settings.device_addr)
            try:
                run_command([settings.adb_path, "connect", settings.device_addr])
                connected = run_command(
                    [settings.adb_path, "-s", settings.device_addr, "get-state"]
                )
                if connected.returncode == 0 and connected.stdout.strip() == b"device":
                    if not check_display:
                        return
                    sizes = run_command(
                        [
                            settings.adb_path,
                            "-s",
                            settings.device_addr,
                            "shell",
                            "wm",
                            "size",
                        ]
                    )
                    matches = re.findall(
                        rb"(?:Physical|Override) size:\s*(\d+)x(\d+)", sizes.stdout
                    )
                    if not matches or tuple(map(int, matches[-1])) != (1920, 1080):
                        raise ValueError(
                            "模拟器画面需要调整。请点击‘调整模拟器画面’，完成后重试"
                        )
                    density = run_command(
                        [
                            settings.adb_path,
                            "-s",
                            settings.device_addr,
                            "shell",
                            "wm",
                            "density",
                        ]
                    )
                    values = re.findall(
                        rb"(?:Physical|Override) density:\s*(\d+)", density.stdout
                    )
                    if not values or int(values[-1]) != 320:
                        raise ValueError(
                            "模拟器显示规格不匹配，请点击‘调整模拟器画面’后重试"
                        )
                    return
            except (OSError, subprocess.TimeoutExpired):
                pass
        if not settings.automatic_recovery:
            raise RuntimeError("模拟器尚未连接，请连接后点击继续采集")
        update(
            status="waiting_device",
            message="等待模拟器恢复连接，任务和已采数据均已保留",
        )
        for _ in range(10):
            if stopped():
                return
            time.sleep(0.5)


def start_game_if_needed(settings: Settings) -> None:
    """只启动已安装的明日方舟，不填写账号、密码或确认更新/付费界面。"""
    if settings.demo or not settings.automatic_recovery:
        return
    packages = run_command(
        [
            settings.adb_path,
            "-s",
            settings.device_addr,
            "shell",
            "pm",
            "list",
            "packages",
        ]
    )
    installed = packages.stdout.decode("utf-8", errors="replace").splitlines()
    candidates = ("com.hypergryph.arknights", "com.hypergryph.arknights.bilibili")
    available = [name for name in candidates if f"package:{name}" in installed]
    if not available:
        raise ValueError("所选模拟器未安装明日方舟，请先安装并登录游戏")
    for name in available:
        active = run_command(
            [settings.adb_path, "-s", settings.device_addr, "shell", "pidof", name]
        )
        if active.stdout.strip():
            return
    if len(available) > 1:
        raise ValueError(
            "模拟器中装有多个游戏版本。请先打开要采集的版本并登录，再点击继续"
        )
    package = available[0]
    resolved = run_command(
        [
            settings.adb_path,
            "-s",
            settings.device_addr,
            "shell",
            "cmd",
            "package",
            "resolve-activity",
            "--brief",
            package,
        ]
    )
    component = next(
        (
            line.strip()
            for line in resolved.stdout.decode("utf-8", errors="replace").splitlines()
            if re.fullmatch(r"[\w.]+/[\w.$]+", line.strip())
            and line.startswith(package + "/")
        ),
        None,
    )
    if not component:
        raise RuntimeError("无法自动打开游戏，请在模拟器中点击游戏图标后继续")
    run_command(
        [
            settings.adb_path,
            "-s",
            settings.device_addr,
            "shell",
            "am",
            "start",
            "-n",
            component,
        ]
    )


def episode(spec_path: Path) -> int:
    """一次作业一个独立进程，MaaCore异常不会把整个桌面界面带走。"""
    spec = read_json(spec_path)
    settings = Settings(**spec["settings"])
    run_dir = Path(spec["run_dir"])
    result_path = Path(spec["result_file"])
    entry = spec["entry"]
    if settings.demo:
        for _ in range(12):
            if control(run_dir) == "stop_now":
                write_json(
                    result_path, {"ok": False, "demo": True, "interrupted": True}
                )
                return 0
            time.sleep(0.15)
        passed = int(spec.get("ordinal", 0)) % 3 != 1
        write_json(
            result_path,
            {
                "ok": passed,
                "demo": True,
                "demo_passed": passed,
                "battle_result": "unknown",
                "valid_collected_win": False,
                "error": "演示：本份任务模拟失败，后续任务继续" if not passed else "",
            },
        )
        return 0
    from bridge import replay_controller as replay

    from .recovery import CollectorSentry

    replay._CHAIN_TIMEOUT_S = 1200.0
    replay._SUPPORT_UNIT_USAGE = settings.support_unit_usage
    notice_file = spec_path.with_suffix(".notice.json")

    def sentry_factory(driver: Any, t0_wall: int) -> CollectorSentry:
        return CollectorSentry(
            driver,
            t0_wall,
            on_message=lambda message: write_json(
                notice_file, {"message": message, "at": time.time()}
            ),
        )

    replay.BlockerSentry = sentry_factory
    controller = replay.ReplayController(
        spec["instance_file"], nav_file(), settings.data_dir
    )
    finished = threading.Event()

    def watch_stop() -> None:
        while not finished.wait(0.5):
            if control(run_dir) == "stop_now":
                try:
                    controller._driver.stop()
                finally:
                    _thread.interrupt_main()
                return

    monitor = threading.Thread(target=watch_stop, daemon=True)
    monitor.start()
    summary: dict[str, Any] = {}
    try:
        if not controller._driver.connect():
            summary = {
                "ok": False,
                "fail_stage": "connection",
                "error": "连接中断，未开始作战",
                "retry_before_battle": True,
            }
        else:
            summary = controller._run_one(entry)
            summary.update(collection_status(summary))
    except KeyboardInterrupt:
        summary = {
            "ok": False,
            "interrupted": True,
            "battle_result": "unknown",
            "error": "当前局已中止，未完成数据保留",
        }
    except Exception as error:
        summary = {
            "ok": False,
            "battle_result": "unknown",
            "fail_stage": "exception",
            "error": f"{type(error).__name__}: {error}",
        }
        partial = getattr(controller, "_partial_run_dir", None)
        if partial:
            summary["inprogress_bundle"] = str(partial)
    finally:
        finished.set()
        controller._driver.close()
    summary.update(
        job_id=entry["source"]["job_id"],
        execution_key=execution_key(entry["source"]),
        stage_code=entry["maa_job"]["stage_name"],
        completed_at=now(),
    )
    write_json(result_path, summary)
    return 0


def run_session(spec_path: Path) -> int:
    """后台队列：局末暂停/停止、崩溃隔离、真实进度与可恢复记录。"""
    spec = read_json(spec_path)
    settings = Settings(**spec["settings"])
    run_dir = Path(spec["run_dir"])
    data_root = Path(settings.data_dir)
    state_file = run_dir / "status.json"
    results_file = data_root / (
        "demo_results.jsonl" if settings.demo else "collection_results.jsonl"
    )
    _, entries = load_queue(Path(settings.queue_file))
    account = hashlib.sha256(
        (settings.account_binding or settings.nickname).encode()
    ).hexdigest()
    previous = {
        row["execution_key"]: row
        for row in read_results(results_file)
        if row.get("collector_account") == account
    }
    if spec.get("restart_all"):
        pass
    elif spec.get("retry_failed"):
        entries = [
            entry
            for entry in entries
            if execution_key(entry["source"]) in previous
            and not previous[execution_key(entry["source"])].get("valid_collected_win")
            and not previous[execution_key(entry["source"])].get("demo_passed")
        ]
    else:
        entries = [
            entry for entry in entries if execution_key(entry["source"]) not in previous
        ]
    state: dict[str, Any] = {
        "pid": os.getpid(),
        "process_birth": process_birth(os.getpid()),
        "run_token": spec["run_token"],
        "status": "starting",
        "started_at": now(),
        "message": "正在准备采集",
        "total": len(entries),
        "attempted": 0,
        "valid": 0,
        "review": 0,
        "demo": settings.demo,
        "current_stage": "",
        "current_job": "",
    }
    lock = threading.Lock()
    done = threading.Event()

    def update(**values: Any) -> None:
        with lock:
            state.update(values)

    def pulse() -> None:
        while not done.is_set():
            with lock:
                snapshot = {**state, "heartbeat_at": now()}
            try:
                write_json(state_file, snapshot)
            except OSError as error:
                update(status="error", message=f"数据保存失败：{error}")
                return
            done.wait(1)

    heartbeat = threading.Thread(target=pulse, daemon=True)
    heartbeat.start()
    lease: DeviceLease | None = None
    child: subprocess.Popen | None = None
    group = ProcessTree()
    try:
        if not settings.demo:
            prepare_device(
                settings, update, lambda: control(run_dir).startswith("stop")
            )
            if control(run_dir).startswith("stop"):
                update(status="stopped", message="采集已停止，已保存数据保留")
                return 0
            lease = DeviceLease(settings.device_addr)
            instance = instance_config(settings, data_root)
        else:
            lease = DeviceLease(f"demo:{spec['run_token']}")
            instance = run_dir / "demo.local.yaml"
        if not lease.acquire():
            raise RuntimeError("这台模拟器已有采集任务，请回到原窗口查看进度")
        if settings.roster_file and Path(settings.roster_file).is_file():
            roster_hash = hashlib.sha256(
                Path(settings.roster_file).read_bytes()
            ).hexdigest()
            roster_copy = data_root / "roster_snapshots" / f"{roster_hash}.json"
            roster_copy.parent.mkdir(parents=True, exist_ok=True)
            if not roster_copy.exists():
                shutil.copy2(settings.roster_file, roster_copy)
        else:
            roster_copy = None
        for index, entry in enumerate(entries):
            while control(run_dir) == "pause_after":
                update(status="paused", message="已暂停，点击继续采集即可恢复")
                time.sleep(0.3)
            if control(run_dir).startswith("stop"):
                update(status="stopped", message="采集已停止，已保存数据保留")
                return 0
            prepare_device(
                settings, update, lambda: control(run_dir).startswith("stop")
            )
            if control(run_dir).startswith("stop"):
                update(status="stopped", message="采集已停止")
                return 0
            start_game_if_needed(settings)
            update(
                status="running",
                message="正在重放作业并保存数据",
                current_stage=entry["maa_job"]["stage_name"],
                current_job=str(entry["source"]["job_id"]),
                position=index + 1,
            )
            if roster_copy:
                entry["source"]["collector_roster_file"] = roster_copy.relative_to(
                    data_root
                ).as_posix()
            child_spec = run_dir / "jobs" / f"j{index}.json"
            result = child_spec.with_suffix(".result.json")
            write_json(
                child_spec,
                {
                    "settings": asdict(settings),
                    "entry": entry,
                    "run_dir": str(run_dir),
                    "instance_file": str(instance),
                    "result_file": str(result),
                    "ordinal": index,
                },
            )
            with child_spec.with_suffix(".log").open("ab", buffering=0) as log:
                env = os.environ.copy()
                env.update(
                    PYTHONUTF8="1",
                    PYTHONUNBUFFERED="1",
                    PYINSTALLER_RESET_ENVIRONMENT="1",
                )
                child = subprocess.Popen(
                    command_for("--episode", str(child_spec)),
                    cwd=data_root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                group.assign(child)
                stop_at: float | None = None
                while child.poll() is None:
                    command = control(run_dir)
                    notice = read_json(child_spec.with_suffix(".notice.json"), {}) or {}
                    if notice.get("at", 0) > time.time() - 10:
                        update(message=notice["message"])
                    if command == "pause_after":
                        update(message="当前局完成后暂停，数据正在保存")
                    if command == "stop_after":
                        update(message="当前局完成后停止，数据正在保存")
                    if command == "stop_now":
                        stop_at = stop_at or time.monotonic()
                        update(message="正在停止当前局并保留未完成数据")
                        if time.monotonic() - stop_at >= 12:
                            stop_owned_process(child)
                    time.sleep(0.3)
            try:
                summary = read_json(result)
            except (ValueError, OSError):
                summary = None
            if summary is None:
                summary = {
                    "ok": False,
                    "fail_stage": "process_exit",
                    "battle_result": "unknown",
                    "error": "采集子进程意外退出，未完成数据保留",
                    "worker_exit_code": child.returncode,
                }
            if control(run_dir) == "stop_now":
                update(status="stopped", message="已停止当前局，未完成数据留待复核")
                return 0
            summary.update(
                execution_key=execution_key(entry["source"]),
                job_id=entry["source"]["job_id"],
                stage_code=entry["maa_job"]["stage_name"],
                collector_account=account,
                completed_at=now(),
            )
            if summary.get("retry_before_battle") and settings.automatic_recovery:
                # 没有入战的连接故障不冒充已经尝试完该作业，保留在续跑列表。
                append_json(run_dir / "recovery_events.jsonl", summary)
                update(
                    status="waiting_device", message="连接中断，该作业保留待下次续跑"
                )
                continue
            append_json(results_file, summary)
            valid = (
                bool(summary.get("demo_passed"))
                if settings.demo
                else bool(summary.get("valid_collected_win"))
            )
            update(
                attempted=state["attempted"] + 1,
                valid=state["valid"] + valid,
                review=state["review"] + (not valid),
                last_result=summary,
            )
        if control(run_dir).startswith("stop"):
            update(status="stopped", message="已完成当前局并停止，数据已保留")
        else:
            pending = state["total"] - state["attempted"]
            update(
                status="finished",
                pending=pending,
                message=f"本轮已结束，{pending}份未开打任务可续跑"
                if pending
                else "本轮任务已完成，未通过项保留在待复核列表",
            )
        if (
            settings.convert_after
            and not settings.demo
            and state["status"] == "finished"
            and state["attempted"] > 0
        ):
            output = data_root / "exports" / f"converted_{run_dir.name}"
            conversion_spec = run_dir / "conversion.json"
            write_json(
                conversion_spec, {"data_root": str(data_root), "output": str(output)}
            )
            update(status="converting", message="正在转换结构化数据，原件保留在本机")
            with (run_dir / "conversion.log").open("wb") as stream:
                child = subprocess.Popen(
                    command_for("--convert", str(conversion_spec)),
                    cwd=data_root,
                    env=env,
                    stdout=stream,
                    stderr=stream,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                group.assign(child)
                while child.poll() is None:
                    if control(run_dir) == "stop_now":
                        stop_owned_process(child)
                        update(status="stopped", message="转换已停止，原始采集数据保留")
                        return 0
                    time.sleep(0.3)
            if child.returncode:
                raise RuntimeError("转换没有完成，原始采集数据保留，可先导出原始采集包")
            update(
                status="finished",
                converted_dir=str(output),
                message="采集和转换已完成，可点击导出数据",
            )
        return 0
    except KeyboardInterrupt:
        update(status="stopped", message="采集已中断，已保存数据保留")
        return 0
    except Exception as error:
        update(status="error", message=str(error), error_type=type(error).__name__)
        return 1
    finally:
        if child is not None and child.poll() is None:
            stop_owned_process(child)
        group.close()
        if lease:
            lease.close()
        done.set()
        heartbeat.join(timeout=5)
        write_json(state_file, {**state, "heartbeat_at": now(), "finished_at": now()})
