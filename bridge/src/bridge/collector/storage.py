"""采集应用的本地设置、运行记录和设备互斥；不保存账号密码。"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import stat
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any


def now() -> str:
    """返回带时区的本机时间。"""
    return datetime.now().astimezone().isoformat()


def app_home() -> Path:
    """设置与登录态只放在当前用户的应用数据目录。"""
    return (
        Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share"))
        / "ArkCollector"
    )


def write_json(path: Path, value: Any) -> None:
    """先写临时文件再替换；Windows短暂读锁不应损坏旧文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            if (
                path.exists()
                and path.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY
            ):
                raise
            time.sleep(0.05)


def read_json(path: Path, default: Any = None) -> Any:
    """读取本地JSON；不存在时返回调用方指定的缺省值。"""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return default


def append_json(path: Path, value: Any) -> None:
    """每份作业完成后追加并刷盘，重启不重复覆盖已有结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("ab+") as stream:
        stream.seek(0, 2)
        if stream.tell():
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def read_results(path: Path) -> list[dict[str, Any]]:
    """保留完整记录；强制结束留下的未写完末行不冒充完成记录。"""
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


@dataclass
class Settings:
    """当前账号的本地选项；补给权限从不继承维护者的机器配置。"""

    data_dir: str = ""
    maa_dir: str = ""
    mumu_manager: str = ""
    adb_path: str = ""
    device_addr: str = ""
    instance_index: str = ""
    queue_file: str = ""
    roster_file: str = ""
    nickname: str = "我的采集账号"
    account_binding: str = ""
    allow_originite: bool = False
    allow_medicine: bool = False
    resource_authorized_at: str = ""
    automatic_recovery: bool = True
    convert_after: bool = False
    support_unit_usage: int = 1
    demo: bool = False
    accepted_notice: bool = False
    last_run_dir: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Settings:
        """只接受已知字段；授权必须为真正布尔值。"""
        raw = read_json(path, {})
        allowed = {item.name for item in fields(cls)}
        settings = cls(**{key: value for key, value in raw.items() if key in allowed})
        for name in (
            "allow_originite",
            "allow_medicine",
            "automatic_recovery",
            "convert_after",
            "demo",
            "accepted_notice",
        ):
            if not isinstance(getattr(settings, name), bool):
                raise ValueError(f"配置项 {name} 不是布尔值")
        if settings.support_unit_usage not in (0, 1):
            raise ValueError("助战选项无效")
        return settings

    def save(self, path: Path) -> None:
        """本地保存选项，不向外部服务上传。"""
        write_json(path, asdict(self))

    def bind_account(self, identifier: str) -> None:
        """切换账号时清除上个账号的补给授权。"""
        if self.account_binding and self.account_binding != identifier:
            self.allow_originite = self.allow_medicine = False
            self.resource_authorized_at = ""
        self.account_binding = identifier


def command_for(*arguments: str) -> list[str]:
    """打包版调用自身工作模式，源码版调用同一模块；玩家无需输入命令。"""
    if getattr(sys, "frozen", False):
        return [sys.executable, *arguments]
    executable = getattr(sys, "_base_executable", sys.executable) or sys.executable
    return [executable, "-m", "bridge.collector", *arguments]


class DeviceLease:
    """同一台设备只允许一个采集执行器，进程退出后由系统释放。"""

    def __init__(self, address: str) -> None:
        self.address = address
        self.handle: Any = None

    def acquire(self) -> bool:
        """抢占当前用户范围内的设备互斥量，不结束任何已有进程。"""
        if os.name != "nt":
            return True
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_bool,
            ctypes.c_wchar_p,
        ]
        kernel.CreateMutexW.restype = ctypes.c_void_p
        name = (
            "Local\\ArkCollector_" + hashlib.sha256(self.address.encode()).hexdigest()
        )
        handle = kernel.CreateMutexW(None, False, name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == 183:
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel.CloseHandle(handle)
            return False
        self.handle = handle
        return True

    def close(self) -> None:
        """只释放本执行器持有的句柄。"""
        if self.handle is not None:
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel.CloseHandle(self.handle)
            self.handle = None


def process_alive(pid: int) -> bool:
    """检查进程实际存活，不能仅凭旧状态文件显示采集中。"""
    if not pid:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    code = ctypes.c_ulong()
    try:
        return (
            bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code)))
            and code.value == 259
        )
    finally:
        kernel.CloseHandle(handle)


def process_birth(pid: int) -> str | None:
    """使用进程创建时间防止PID复用被误认成旧采集任务。"""
    if os.name != "nt":
        return None
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        *([ctypes.POINTER(wintypes.FILETIME)] * 4),
    ]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    created, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
    try:
        if not kernel.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            return None
        return str((created.dwHighDateTime << 32) | created.dwLowDateTime)
    finally:
        kernel.CloseHandle(handle)


class ProcessTree:
    """执行器异常退出时由Windows结束它创建的作业子进程，防止旧控制器遗留。"""

    def __init__(self) -> None:
        self.handle: Any = None
        if os.name != "nt":
            return
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        kernel.CreateJobObjectW.restype = ctypes.c_void_p
        kernel.SetInformationJobObject.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        handle = kernel.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        information = ctypes.create_string_buffer(144)
        information[16:20] = (0x2000).to_bytes(4, "little")
        if not kernel.SetInformationJobObject(handle, 9, information, len(information)):
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle = handle

    def assign(self, process: Any) -> None:
        """仅绑定当前执行器刚创建的子进程，不枚举或结束用户其他程序。"""
        if self.handle is None:
            return
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        if (
            not kernel.AssignProcessToJobObject(self.handle, int(process._handle))
            and process.poll() is None
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        """释放句柄；Windows负责清理该组内仍存活的作业进程。"""
        if self.handle is not None:
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel.CloseHandle(self.handle)
            self.handle = None
