"""界面与后台执行器之间的应用服务；恢复上次任务不依赖命令行。"""

from __future__ import annotations

import os
import site
import subprocess
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .discovery import check_runtime
from .storage import (
    Settings,
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
        self.settings = (
            Settings.load(self.settings_file)
            if self.settings_file.exists()
            else Settings()
        )
        if not self.settings.data_dir:
            self.settings.data_dir = str(self.home / "data")
        if not self.settings.account_binding:
            self.settings.account_binding = f"local-{uuid.uuid4().hex}"
        self.process: subprocess.Popen | None = None
        if demo:
            self.settings.demo = True
            self.settings.data_dir = str(self.home / "demo-data")
            self.prepare_demo()
        elif self.settings.demo:
            self.settings.demo = False
            self.settings.data_dir = str(self.home / "data")
            self.settings.queue_file = ""
            self.settings.last_run_dir = ""

    def save(self) -> None:
        """保存界面已明确选择的选项。"""
        self.settings.save(self.settings_file)

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

    def import_tasks(self, source: Path) -> dict[str, Any]:
        """玩家点击导入后，建立本地任务副本。"""
        if self.is_running():
            raise RuntimeError("请先完成当前采集，再更换任务")
        path, details = import_queue(source, Path(self.settings.data_dir))
        self.settings.queue_file = str(path)
        self.settings.last_run_dir = ""
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
        state = read_json(run_dir / "status.json", {}) or {}
        spec = read_json(run_dir / "run.json", {}) or {}
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
                "alive": alive,
                "message": "正在启动采集"
                if alive
                else "上次任务未正常结束，可以继续未完成任务",
            }
        if state.get("status") not in ("finished", "stopped", "error") and not alive:
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
        return bool(state.get("alive")) and state.get("status") not in (
            "finished",
            "stopped",
            "error",
        )

    def start(self, *, retry_failed: bool = False, restart_all: bool = False) -> Path:
        """开始新一轮或续跑；运行版本无需Python环境或命令窗口。"""
        if self.is_running():
            raise RuntimeError("采集已经在进行")
        self.task_summary()
        if not self.settings.queue_file:
            raise ValueError("请先点击导入采集任务")
        if not self.settings.demo:
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
        data_root = Path(self.settings.data_dir)
        data_root.mkdir(parents=True, exist_ok=True)
        run_dir = data_root / "runs" / ("r-" + uuid.uuid4().hex[:16])
        run_dir.mkdir(parents=True)
        token = uuid.uuid4().hex
        write_json(
            run_dir / "run.json",
            {
                "schema": "collector_run/1",
                "run_token": token,
                "settings": asdict(self.settings),
                "run_dir": str(run_dir),
                "retry_failed": retry_failed,
                "restart_all": restart_all,
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

    def send(self, command: str) -> None:
        """界面按钮发送显式控制，不按连续失败次数停止队列。"""
        if command not in ("pause_after", "continue", "stop_after", "stop_now"):
            raise ValueError("未知控制操作")
        if not self.settings.last_run_dir:
            return
        write_json(
            Path(self.settings.last_run_dir) / "control.json",
            {"command": command, "at": now()},
        )
