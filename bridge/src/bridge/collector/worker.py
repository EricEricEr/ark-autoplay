"""桌面采集执行器：子进程隔离、恢复连接、局末暂停与可靠进度记录。"""

from __future__ import annotations

import _thread
import contextlib
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bridge.candidate_coverage import execution_key
from bridge.collection_quality import collection_status

from .discovery import instance_config as instance_config
from .discovery import (
    list_devices,
    run_command,
    validate_address,
)
from .resilience import HealthReporter, NeedsUser, account_key
from .storage import (
    DeviceLease as DeviceLease,
)
from .storage import (
    ProcessTree as ProcessTree,
)
from .storage import (
    Settings,
    now,
    read_json,
    write_json,
)
from .storage import (
    command_for as command_for,
)


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
            try:
                devices = list_devices(settings.mumu_manager)
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
                if not settings.automatic_recovery:
                    raise
                update(
                    status="waiting_device", message="正在等待模拟器管理服务恢复响应"
                )
                time.sleep(1)
                continue
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
                with contextlib.suppress(OSError, subprocess.TimeoutExpired):
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
                    # Android按设备自然朝向报告尺寸；MuMu游戏会旋转成横屏。
                    if not matches or sorted(map(int, matches[-1])) != [1080, 1920]:
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
        raise NeedsUser("所选模拟器未安装明日方舟，请安装、登录并回到主页后继续")
    for name in available:
        active = run_command(
            [settings.adb_path, "-s", settings.device_addr, "shell", "pidof", name]
        )
        if active.stdout.strip():
            return
    if len(available) > 1:
        raise NeedsUser(
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
        raise NeedsUser("无法自动打开游戏，请打开游戏并回到主页后继续")
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


def environment_failure(settings: Settings) -> str:
    """战斗中核对真实设备与游戏进程，不用画面冻结或动作数量判断故障。"""
    if settings.demo:
        return ""
    try:
        connected = run_command(
            [settings.adb_path, "-s", settings.device_addr, "get-state"]
        )
        if connected.returncode or connected.stdout.strip() != b"device":
            return "模拟器连接中断，当前局原件保留，正在恢复设备"
        for package in (
            "com.hypergryph.arknights",
            "com.hypergryph.arknights.bilibili",
        ):
            active = run_command(
                [
                    settings.adb_path,
                    "-s",
                    settings.device_addr,
                    "shell",
                    "pidof",
                    package,
                ]
            )
            if active.returncode == 0 and active.stdout.strip():
                return ""
        return "游戏进程已退出，当前局原件保留，正在重新打开游戏"
    except (OSError, subprocess.TimeoutExpired):
        return "设备控制通道失去响应，当前局原件保留，正在重连"


def episode(spec_path: Path) -> int:
    """一次作业一个独立进程，MaaCore异常不会把整个桌面界面带走。"""
    spec = read_json(spec_path)
    settings = Settings(**spec["settings"])
    run_dir = Path(spec["run_dir"])
    result_path = Path(spec["result_file"])
    entry = spec["entry"]
    health = HealthReporter(
        Path(spec.get("health_file", spec_path.with_suffix(".health.json")))
    )

    def save_result(summary: dict) -> None:
        summary.update(
            job_id=entry["source"]["job_id"],
            execution_key=execution_key(entry["source"]),
            stage_code=entry["maa_job"]["stage_name"],
            collector_account=account_key(settings),
            attempt_id=spec.get("attempt_id"),
            receipt_complete=True,
            completed_at=now(),
        )
        write_json(result_path, summary)
        health.close()

    if settings.demo:
        for _ in range(12):
            if control(run_dir) == "stop_now":
                save_result({"ok": False, "demo": True, "interrupted": True})
                return 0
            time.sleep(0.15)
        passed = int(spec.get("ordinal", 0)) % 3 != 1
        save_result(
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
    from bridge.state_logger import StateLogger

    class CheckedLogger(StateLogger):
        def __init__(self, *args, **kwargs):
            kwargs["on_storage_error"] = lambda error: setattr(health, "fault", error)
            super().__init__(*args, **kwargs)

    replay.StateLogger = CheckedLogger
    controller = health.wrap("初始化采集运行库", replay.ReplayController)(
        spec["instance_file"], nav_file(), settings.data_dir
    )
    for name in (
        "connect",
        "running",
        "start",
        "stop",
        "close",
        "append_copilot",
        "append_task",
        "set_task_params",
        "screencap",
        "_adb_shell_input",
        "events_since",
    ):
        function = getattr(controller._driver, name, None)
        if callable(function):
            setattr(controller._driver, name, health.wrap(name, function))
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
    except NeedsUser as error:
        summary = {
            "ok": False,
            "battle_result": "unknown",
            "retry_pending": True,
            "needs_user": True,
            "error": str(error),
        }
    except OSError as error:
        summary = {
            "ok": False,
            "battle_result": "unknown",
            "retry_pending": True,
            "storage_error": True,
            "error": str(error),
        }
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
    save_result(summary)
    return 0


def run_session(spec_path: Path) -> int:
    """采集队列的隔离监督入口，测试与发行版共用恢复实现。"""
    import sys

    from .supervisor import run

    return run(spec_path, sys.modules[__name__])
