"""自动发现MuMu和采集运行库；所有外部程序都以参数列表启动。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .storage import Settings


@dataclass
class Device:
    """模拟器列表中的一项，显示名称与内部连接参数分开。"""

    index: str
    name: str
    address: str = ""
    running: bool = False

    @property
    def label(self) -> str:
        """面向玩家显示，不要求理解ADB端口。"""
        return f"{self.name} · {'运行中' if self.running else '未启动'}"


def run_command(
    arguments: list[str], *, timeout: float = 20
) -> subprocess.CompletedProcess:
    """隐藏子进程窗口，不使用shell拼接，不安装系统服务。"""
    return subprocess.run(
        arguments,
        capture_output=True,
        timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def parse_devices(document: Any) -> list[Device]:
    """兼容单实例和全部实例的MuMu查询响应。"""
    if not isinstance(document, dict):
        raise ValueError("模拟器返回的设备列表无法识别")
    rows = [document] if "index" in document else list(document.values())
    result = []
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("index", "")).isdigit():
            continue
        address = ""
        if row.get("adb_port"):
            host = row.get("adb_host_ip") or "127.0.0.1"
            address = f"{host}:{int(row['adb_port'])}"
            validate_address(address)
        result.append(
            Device(
                str(row["index"]),
                str(row.get("name") or f"模拟器 {row['index']}"),
                address,
                bool(row.get("is_android_started")),
            )
        )
    return sorted(result, key=lambda item: int(item.index))


def validate_address(value: str) -> None:
    """软件仅连接本机MuMu；任务包不能指定远程设备或命令。"""
    match = re.fullmatch(r"(?:127\.0\.0\.1|localhost):(\d+)", value)
    if not match or not 1 <= int(match.group(1)) <= 65535:
        raise ValueError("请选择本机MuMu设备")


def application_base() -> Path:
    """获取软件所在目录，源码运行时使用项目根。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[4]


def find_installations(settings: Settings) -> dict[str, str]:
    """先查已保存路径、软件旁和常见安装位置，不扫描用户整块硬盘。"""
    roots = [
        Path(os.environ.get(name, "C:/Program Files"))
        for name in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)")
    ]
    candidates = [Path(settings.mumu_manager)] if settings.mumu_manager else []
    for root in roots:
        candidates.extend(
            [
                root / "Netease/MuMu/nx_main/MuMuManager.exe",
                root / "Netease/MuMuPlayer-12.0/shell/MuMuManager.exe",
            ]
        )
    if os.name == "nt":
        import winreg

        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for key_name in (
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
            ):
                try:
                    with winreg.OpenKey(hive, key_name) as root_key:
                        for index in range(winreg.QueryInfoKey(root_key)[0]):
                            try:
                                with winreg.OpenKey(
                                    root_key, winreg.EnumKey(root_key, index)
                                ) as subkey:
                                    name = str(
                                        winreg.QueryValueEx(subkey, "DisplayName")[0]
                                    )
                                    if "mumu" not in name.casefold():
                                        continue
                                    for field_name in (
                                        "InstallLocation",
                                        "DisplayIcon",
                                        "UninstallString",
                                    ):
                                        try:
                                            value = str(
                                                winreg.QueryValueEx(subkey, field_name)[
                                                    0
                                                ]
                                            ).strip()
                                        except OSError:
                                            continue
                                        match = re.match(
                                            r'"([^\"]+)"|(.+?\.exe)',
                                            value,
                                            re.IGNORECASE,
                                        )
                                        location = Path(
                                            (match.group(1) or match.group(2))
                                            if match
                                            else value
                                        )
                                        location = (
                                            location.parent
                                            if location.suffix.casefold() == ".exe"
                                            else location
                                        )
                                        candidates.extend(
                                            [
                                                location / "MuMuManager.exe",
                                                location / "nx_main/MuMuManager.exe",
                                                location.parent
                                                / "nx_main/MuMuManager.exe",
                                            ]
                                        )
                            except OSError:
                                continue
                except OSError:
                    continue
    found = {
        "mumu_manager": "",
        "adb_path": settings.adb_path,
        "maa_dir": settings.maa_dir,
        "node_path": "",
    }
    for candidate in candidates:
        if candidate.is_file():
            found["mumu_manager"] = str(candidate)
            adb = candidate.parent / "adb.exe"
            if adb.exists():
                found["adb_path"] = str(adb)
            break
    for candidate in (
        application_base() / "maa_runtime_patched",
        application_base() / "runtime/maa_runtime_patched",
    ):
        if not found["maa_dir"] and (candidate / "MaaCore.dll").exists():
            found["maa_dir"] = str(candidate)
    for candidate in (
        application_base() / "tools/node.exe",
        application_base() / "_internal/tools/node.exe",
    ):
        if candidate.is_file():
            found["node_path"] = str(candidate)
            break
    if not found["node_path"]:
        import shutil

        found["node_path"] = shutil.which("node") or ""
    return found


def list_devices(manager: str) -> list[Device]:
    """读取模拟器列表，不启动或重启实例。"""
    response = run_command([manager, "info", "-v", "all"])
    if response.returncode:
        raise RuntimeError("无法读取模拟器列表，请确认MuMu已正常安装")
    return parse_devices(json.loads(response.stdout.decode("utf-8-sig")))


def runtime_hash(directory: Path) -> str:
    """只校验DLL文件，不读取账号配置或游戏内存。"""
    with (directory / "MaaCore.dll").open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_runtime(directory: Path) -> dict[str, Any]:
    """使用维护者登记的采集DLL摘要核验，不能只凭文件夹名字认定已打补丁。"""
    required = [
        directory / "MaaCore.dll",
        directory / "Python/asst/asst.py",
        directory / "resource",
    ]
    if not all(item.exists() for item in required):
        raise ValueError("这里不是完整的采集运行库，请选择包含 MaaCore.dll 的文件夹")
    reference = Path(__file__).parent / "resources/runtime_compatibility.json"
    known = json.loads(reference.read_text(encoding="utf-8"))
    digest = runtime_hash(directory)
    if digest not in known["accepted_core_sha256"]:
        raise ValueError(
            "该运行库不是此软件配套的采集版本，请使用维护者提供的采集运行库"
        )
    return {"core_sha256": digest, "version": known["display_version"]}


def instance_config(settings: Settings, data_root: Path) -> Path:
    """把界面选项转换为私有实例配置；外部任务无权改变补给权限。"""
    import yaml

    validate_address(settings.device_addr)
    runtime = Path(settings.maa_dir).resolve()
    check_runtime(runtime)
    if "maa_runtime_patched" not in runtime.as_posix():
        alias = data_root / "runtime/maa_runtime_patched"
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.exists():
            if alias.resolve() != runtime:
                raise ValueError("本地运行库关联已改变，请在设置中重新选择数据目录")
        else:
            response = run_command(
                ["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(runtime)]
            )
            if response.returncode or not alias.exists():
                raise PermissionError(
                    "无法建立运行库关联，可尝试点击以管理员权限重新打开"
                )
        runtime = alias
    host, port = settings.device_addr.rsplit(":", 1)
    document = {
        "name": "collector",
        "adb_host": host,
        "adb_port": int(port),
        "adb_path": settings.adb_path,
        "maa_dir": str(runtime),
        "maa_version": "collector-patched",
        "resolution": {"width": 1920, "height": 1080},
        "dpi": 320,
        "resource_policy": {
            "allow_originite": settings.allow_originite,
            "allow_medicine": settings.allow_medicine,
        },
    }
    path = data_root / "configs/collector.local.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
    return path


def collect_diagnostics(settings: Settings) -> dict[str, Any]:
    """生成普通玩家可读的检查结果，不连接MAA、不控制游戏。"""
    found = find_installations(settings)
    rows = [
        {
            "title": "MuMu模拟器",
            "ok": bool(found["mumu_manager"]),
            "detail": "已找到" if found["mumu_manager"] else "请点击选择MuMu安装位置",
        }
    ]
    if found["maa_dir"]:
        try:
            check_runtime(Path(found["maa_dir"]))
            runtime_ok, text = True, "采集运行库校验通过"
        except (ValueError, OSError) as error:
            runtime_ok, text = False, str(error)
    else:
        runtime_ok, text = False, "首次使用请点击选择采集运行库"
    rows.append({"title": "采集运行库", "ok": runtime_ok, "detail": text})
    devices = list_devices(found["mumu_manager"]) if found["mumu_manager"] else []
    return {
        "paths": found,
        "checks": rows,
        "devices": [asdict(device) for device in devices],
    }
