"""界面与后台执行器之间的应用服务；恢复上次任务不依赖命令行。"""

from __future__ import annotations

import contextlib
import os
import site
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bridge.data_paths import data_path

from .discovery import check_runtime, validate_address
from .resilience import account_key, load_settings, reconcile, writable_probe
from .storage import (
    app_home,
    command_for,
    now,
    process_alive,
    process_birth,
    read_json,
    write_json,
)
from .task_package import import_queue, load_queue


def child_environment() -> dict[str, str]:
    """子进程使用同一份程序；开发环境的源码路径仅存在本机环境变量中。"""
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYINSTALLER_RESET_ENVIRONMENT="1")
    if not getattr(sys, "frozen", False):
        source = str(Path(__file__).resolve().parents[2])
        paths = [
            source,
            *site.getsitepackages(),
            *env.get("PYTHONPATH", "").split(os.pathsep),
        ]
        env["PYTHONPATH"] = os.pathsep.join(
            dict.fromkeys(path for path in paths if path)
        )
    return env


class CollectorService:
    """维护一个本地采集档案，界面重开后可继续观察同一后台进程。"""

    def __init__(self, home: Path | None = None, *, demo: bool = False) -> None:
        self.home = home or app_home()
        self.home.mkdir(parents=True, exist_ok=True)
        self.settings_file = self.home / "settings.local.json"
        self.settings, self.startup_notice = load_settings(self.settings_file)
        if not self.settings.account_binding:
            self.settings.account_binding = f"local-{uuid.uuid4().hex}"
        self.process: subprocess.Popen | None = None
        if demo:
            self.settings.demo = True
            self.settings.data_dir = str(self.home / "demo-data")
            self.prepare_demo()
        elif self.settings.demo:
            self.settings.demo = False
            self.settings.data_dir = ""
            self.settings.queue_file = ""
            self.settings.last_run_dir = ""

    def save(self) -> None:
        """保存界面已明确选择的选项。"""
        self.settings.save(self.settings_file)

    def data_root(self) -> Path:
        """真实资料必须由用户选择非C盘目录；演示合成资料留在独立演示档案。"""
        if self.settings.demo:
            return Path(self.settings.data_dir).resolve()
        return data_path(self.settings.data_dir)

    def set_data_directory(self, value: str | Path) -> None:
        """保存有效的数据目录；换目录后重新导入任务，原目录和任务原件保留。"""
        selected = data_path(value)
        if self.is_running():
            raise RuntimeError("请先停止当前任务再修改数据位置")
        previous = self.settings.data_dir
        changed = not previous or Path(previous).expanduser().resolve() != selected
        self.settings.data_dir = str(selected)
        if changed:
            self.settings.queue_file = ""
            self.settings.last_run_dir = ""
        self.save()

    def prepare_demo(self) -> None:
        """演示仅使用合成任务，单独的数据目录不会被当成真实轨迹。"""
        demo_queue = self.home / "demo_tasks.json"
        jobs = [
            {
                "source": {
                    "job_id": f"SYNTHETIC-{index}",
                    "stage_id_raw": f"demo_{index}",
                },
                "maa_job": {
                    "stage_name": f"演示关卡 {index + 1}",
                    "opers": [],
                    "actions": [{"type": "Output", "doc": "synthetic"}],
                },
            }
            for index in range(6)
        ]
        write_json(demo_queue, {"title": "演示采集任务（不连接游戏）", "jobs": jobs})
        path, _ = import_queue(demo_queue, Path(self.settings.data_dir))
        self.settings.queue_file = str(path)
        self.settings.last_run_dir = ""
        self.settings.extras["last_capture_mode"] = "jobs"

    def import_tasks(self, source: Path) -> dict[str, Any]:
        """玩家点击导入后，建立本地任务副本。"""
        if self.is_running():
            raise RuntimeError("请先完成当前采集，再更换任务")
        path, details = import_queue(source, self.data_root())
        self.settings.queue_file = str(path)
        self.settings.last_run_dir = ""
        self.settings.extras["last_capture_mode"] = "jobs"
        self.save()
        return details

    def task_summary(self) -> dict[str, Any]:
        """计算全部任务数量，不设置份数上限。"""
        if not self.settings.queue_file:
            return {"jobs": 0, "stages": 0, "title": "还没有导入采集任务"}
        metadata, entries = load_queue(Path(self.settings.queue_file))
        return {
            "jobs": len(entries),
            "stages": len({row["source"]["stage_id_raw"] for row in entries}),
            "title": metadata.get("title") or "采集任务",
        }

    def status(self) -> dict[str, Any]:
        """业务状态与真实进程同时核对，旧running不冒充仍在运行。"""
        if not self.settings.last_run_dir:
            return {
                "status": "idle",
                "message": "准备完成后，点击开始采集",
                "alive": False,
            }
        run_dir = Path(self.settings.last_run_dir)
        try:
            spec = read_json(run_dir / "run.json", {}) or {}
            state = (
                read_json(Path(spec.get("live_file", run_dir / "status.json")), {})
                or {}
            )
        except (OSError, ValueError, UnicodeError):
            spec, state = {}, {}
        if state and state.get("run_token") != spec.get("run_token"):
            return {
                "status": "error",
                "message": "任务状态不匹配，请重新打开软件",
                "alive": False,
            }
        alive = process_alive(int(state.get("pid") or 0))
        if alive and state.get("process_birth"):
            alive = process_birth(int(state["pid"])) == state["process_birth"]
        if not state:
            alive = self.process is not None and self.process.poll() is None
            return {
                "status": "starting" if alive else "interrupted",
                "manual": bool(spec.get("manual"))
                or self.settings.extras.get("last_capture_mode") == "manual",
                "alive": alive,
                "message": "正在启动采集"
                if alive
                else "上次任务未正常结束，可以继续未完成任务",
            }
        if alive and time.time() - state.get("heartbeat_epoch", time.time()) > 30:
            return {
                **state,
                "status": "unresponsive",
                "alive": True,
                "message": "采集监督器失去响应。可点‘停止无响应任务’，"
                "保留数据后重新检查。",
            }
        if (
            state.get("status")
            not in (
                "finished",
                "stopped",
                "error",
                "storage_error",
                "conversion_error",
                "needs_user",
            )
            and not alive
        ):
            return {
                **state,
                "status": "interrupted",
                "alive": False,
                "message": "采集进程已退出，已采数据保留，可点击继续未完成任务",
            }
        return {**state, "alive": alive}

    def is_running(self) -> bool:
        """暂停、重连期间也占用当前采集器。"""
        state = self.status()
        return bool(state.get("alive"))

    def start(
        self,
        *,
        retry_failed: bool = False,
        restart_all: bool = False,
        conversion_only: bool = False,
    ) -> Path:
        """开始新一轮或续跑；运行版本无需Python环境或命令窗口。"""
        if self.is_running():
            raise RuntimeError("采集已经在进行")
        if (
            self.settings.extras.get("last_capture_mode") == "manual"
            and not conversion_only
        ):
            raise ValueError("当前是手动录制模式；如需自动作业，请先重新导入任务")
        data_root = self.data_root()
        if not conversion_only:
            self.task_summary()
        if not conversion_only and not self.settings.queue_file:
            raise ValueError("请先点击导入采集任务")
        if not self.settings.demo and not conversion_only:
            if not self.settings.accepted_notice:
                raise ValueError("请先阅读并确认采集说明")
            if not self.settings.adb_path or not self.settings.maa_dir:
                raise ValueError("请先完成环境检查")
            if not self.settings.device_addr and not self.settings.instance_index:
                raise ValueError("请先选择模拟器")
            check_runtime(Path(self.settings.maa_dir))
            if (
                self.settings.allow_originite or self.settings.allow_medicine
            ) and not self.settings.resource_authorized_at:
                raise ValueError("补给选项尚未得到账号持有者确认")
        data_root.mkdir(parents=True, exist_ok=True)
        writable_probe(data_root)
        reconcile(data_root, account_key(self.settings), demo=self.settings.demo)
        run_dir = data_root / "runs" / ("r-" + uuid.uuid4().hex[:16])
        run_dir.mkdir(parents=True)
        token = uuid.uuid4().hex
        if not conversion_only:
            self.settings.extras["last_capture_mode"] = "jobs"
        write_json(
            run_dir / "run.json",
            {
                "schema": "collector_run/1",
                "run_token": token,
                "settings": asdict(self.settings),
                "run_dir": str(run_dir),
                "retry_failed": retry_failed,
                "restart_all": restart_all,
                "conversion_only": conversion_only,
                "live_file": str(self.home / "live" / f"{token}.json"),
                "profile_home": str(self.home),
            },
        )
        write_json(run_dir / "control.json", {"command": "continue", "at": now()})
        self.settings.last_run_dir = str(run_dir)
        self.save()
        with (run_dir / "worker.log").open("ab", buffering=0) as stream:
            self.process = subprocess.Popen(
                command_for("--run", str(run_dir / "run.json")),
                cwd=data_root,
                env=child_environment(),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=stream,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        return run_dir

    def start_manual(self, stage_label: str = "") -> Path:
        """无队列旁路录制；不加载MAA、不导航、不启动或控制游戏。"""
        if self.is_running():
            raise RuntimeError("请先停止当前采集，再开始手动录制")
        if self.settings.demo:
            raise ValueError("演示模式不连接真实设备，请在正式档案中使用手动录制")
        data = self.data_root()
        validate_address(self.settings.device_addr)
        if not self.settings.adb_path or not self.settings.maa_dir:
            raise ValueError("请先选择MuMu和运行库；运行库仅用于识别战斗与结算")
        token = uuid.uuid4().hex
        run_dir = data / "runs" / ("manual-" + token)
        run_dir.mkdir(parents=True)
        self.settings.extras["last_capture_mode"] = "manual"
        self.settings.extras["manual_stage_label"] = stage_label
        write_json(
            run_dir / "run.json",
            {
                "schema": "manual_run/1",
                "run_token": token,
                "manual": True,
                "settings": asdict(self.settings),
                "run_dir": str(run_dir),
                "session_dir": str(data / "manual_sessions" / token),
                "stage_label": stage_label,
                "profile_home": str(self.home),
                "live_file": str(self.home / "live" / f"{token}.json"),
            },
        )
        write_json(run_dir / "control.json", {"command": "continue", "at": now()})
        self.settings.last_run_dir = str(run_dir)
        self.save()
        with (run_dir / "worker.log").open("ab", buffering=0) as stream:
            self.process = subprocess.Popen(
                command_for("--manual", str(run_dir / "run.json")),
                cwd=data,
                env=child_environment(),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=stream,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        return run_dir

    def send(self, command: str) -> None:
        """界面按钮发送显式控制，不按连续失败次数停止队列。"""
        if command not in (
            "pause_after",
            "continue",
            "stop_after",
            "stop_now",
            "resume_checked",
        ):
            raise ValueError("未知控制操作")
        if not self.settings.last_run_dir:
            return
        write_json(
            Path(self.settings.last_run_dir) / "control.json",
            {
                "command": command,
                "at": now(),
                "challenge": self.status().get("challenge"),
            },
        )

    def recover(self) -> None:
        """界面一键检查磁盘、核对记账并恢复，所有操作仍在本地。"""
        writable_probe(self.data_root())
        if self.settings.extras.get("last_capture_mode") == "manual":
            if self.is_running():
                raise RuntimeError("手动录制仍在运行，可停止后开始新的录制")
            self.start_manual(self.settings.extras.get("manual_stage_label", ""))
            return
        if self.is_running():
            self.send("resume_checked")
        else:
            if self.status().get("status") == "needs_user" and not self.settings.demo:
                from . import worker
                from .supervisor import home_visible

                worker.prepare_device(self.settings, lambda **_: None, lambda: False)
                if not home_visible(self.settings, worker):
                    raise RuntimeError("尚未识别到游戏主页，请处理登录或弹窗后重试")
            self.start()

    def stop_unresponsive(self) -> None:
        """只终止身份已核对的本档案监督进程，其作业组负责清理子进程。"""
        from .worker import run_command

        state = self.status()
        pid = int(state.get("pid") or 0)
        if not pid or not state.get("alive"):
            return
        if (
            not state.get("process_birth")
            or process_birth(pid) != state["process_birth"]
        ):
            raise RuntimeError("进程身份已变化，请重新打开软件核对")
        run_command(["taskkill", "/PID", str(pid), "/T", "/F"])
        if self.process:
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=10)
