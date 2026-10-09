"""MAA 驱动封装（共享基础设施）。

职责（对应设计文档 §10 数据工厂的执行底座）：对官方 MAA 发行版的 Python
接口（``asst``）做薄封装——动态加载、单实例连接、回调事件流（线程安全）、
Copilot 任务下发与任务链等待、ADB 截图与点触。

红线与约束：

- 只动态加载官方 MAA 安装目录的接口文件，**不复制、不修改其任何内容**
  （协议级复用，依据见 ``docs/adr/0001-pure-python-driver.md``）；
- 一切机器相关路径（MAA 目录、adb 路径、实例地址）均来自
  ``configs/instances/*.yaml``，代码内不出现任何本机路径常量；
- ``asst`` 的导入在首次使用时惰性完成，纯逻辑单测不需要安装 MAA。

回调事实（**2026-10-09 更新**）：

- **官方发行版**：回调包含连接/截图协商、任务链生命周期与作战动作事件
  （``SubTaskExtraInfo`` 的 ``CopilotAction``），但**不含**费用/击杀等战场状态
  （导出的 C API 无 ``GetCost``/``GetKills``）。
- **打过补丁的 MaaCore**（``patches/0001-battle-state-callback.patch``）：额外发出
  ``SubTaskExtraInfo`` + ``what="BattleState"``，含费用/击杀/待部署栏/已部署干员。
  状态转换见 ``battle_state`` 模块。

两者的 ``Asst*`` 接口与 bundle 结构完全兼容，故本驱动无需区分——把实例配置的
``maa_dir`` 指向哪个运行时，就拿到对应能力。

任务链在作业动作全部执行完毕时即完成，**不等作战自然结束**，胜负判定需
在任务链结束后由结算画面模板判定（见 ``navigator.detect_result``）。
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

# asst 模块级缓存：同一进程只允许 Asst.load 一次
_ASST_CTX: dict[str, Any] = {}
"""已加载的 asst 上下文：{"Asst": cls, "Message": cls, "maa_dir": str}。"""

_CHAIN_TERMINAL_MSGS = ("TaskChainCompleted", "TaskChainError", "AllTasksCompleted")
"""任务链终态回调名（实测：Copilot 单任务链以 TaskChainCompleted 收尾）。"""


@dataclass
class MaaEvent:
    """一条 MAA 回调事件（已解码）。"""

    t_wall_ms: int
    """墙钟毫秒时间戳（``time.time() * 1000``），用于跨进程对齐截图。"""
    t_mono_ms: int
    """单调时钟毫秒（进程内相对量，计算间隔专用）。"""
    msg: str
    """回调类型名（``Message`` 枚举名，未知时为 ``msg(<int>)``）。"""
    details: dict[str, Any]
    """回调 JSON 体（解码失败时为 ``{"raw": ...}``）。"""


@dataclass
class ChainResult:
    """任务链等待结果。"""

    finished_ok: bool
    """True=TaskChainCompleted；False=TaskChainError / 超时 / 未收到终态。"""
    terminal_msg: str | None
    """触发收尾的终态回调名；超时未收尾为 None。"""
    events: list[MaaEvent] = field(default_factory=list)
    """等待窗口内观察到的全部事件。"""


def load_instance_config(path: str | Path) -> dict[str, Any]:
    """读取 MuMu 实例定义（configs/instances/*.yaml）。"""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"实例配置不是 mapping: {path}")
    for key in ("adb_path", "adb_host", "adb_port", "maa_dir", "resolution"):
        if key not in cfg:
            raise ValueError(f"实例配置缺少字段 {key}: {path}")
    return cfg


def _load_asst(maa_dir: str, user_dir: str) -> dict[str, Any]:
    """惰性加载 asst 接口；进程内只加载一次，重复调用直接复用。"""
    if _ASST_CTX:
        if _ASST_CTX["maa_dir"] != maa_dir:
            raise RuntimeError("同一进程不支持加载两个不同 MAA 目录")
        return _ASST_CTX
    os.add_dll_directory(maa_dir)
    py_dir = os.path.join(maa_dir, "Python")
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)
    from asst.asst import Asst  # noqa: PLC0415  # 惰性导入：依赖官方安装目录
    from asst.utils import Message  # noqa: PLC0415

    Path(user_dir).mkdir(parents=True, exist_ok=True)
    if not Asst.load(path=maa_dir, user_dir=user_dir):
        raise RuntimeError(f"Asst.load 失败: {maa_dir}")
    _ASST_CTX.update({"Asst": Asst, "Message": Message, "maa_dir": maa_dir})
    return _ASST_CTX


class MaaDriver:
    """单个 MAA 实例的驱动句柄。

    用法::

        driver = MaaDriver(load_instance_config(cfg_path), user_dir)
        driver.connect()
        tid = driver.append_copilot(job_path)
        driver.start()
        result = driver.wait_chain(timeout_s=600)
    """

    def __init__(self, instance_cfg: dict[str, Any], user_dir: str | Path) -> None:
        """以实例配置初始化；``user_dir`` 为 MAA 运行数据落盘目录。"""
        self._cfg = instance_cfg
        self._adb = str(instance_cfg["adb_path"])
        self._addr = f"{instance_cfg['adb_host']}:{instance_cfg['adb_port']}"
        ctx = _load_asst(str(instance_cfg["maa_dir"]), str(user_dir))
        self._asst_mod = ctx["Asst"]
        self._msg_mod = ctx["Message"]
        self._events: list[MaaEvent] = []
        self._queue: queue.Queue[MaaEvent] = queue.Queue()
        self._lock = threading.Lock()
        self._cap_lock = threading.Lock()
        self._t0_mono = time.monotonic()
        self._asst: Any = None

    # ---- 回调与事件流 ----

    def _on_callback(self, msg: int, details: bytes, arg: Any) -> None:
        """MAA 线程回调：只做解码与入队，严禁重活。"""
        del arg
        now_wall = int(time.time() * 1000)
        now_mono = int((time.monotonic() - self._t0_mono) * 1000)
        try:
            name = self._msg_mod(msg).name
        except Exception:  # noqa: BLE001  # 未知枚举值不能打断回调线程
            name = f"msg({msg})"
        try:
            body = json.loads(details.decode("utf-8"))
        except Exception:  # noqa: BLE001  # 保留原文便于排查
            body = {"raw": repr(details[:200])}
        evt = MaaEvent(now_wall, now_mono, name, body)
        with self._lock:
            self._events.append(evt)
        self._queue.put(evt)

    def mark(self) -> int:
        """返回当前事件日志游标（配合 :meth:`events_since` 使用）。"""
        with self._lock:
            return len(self._events)

    def events_since(self, mark: int) -> tuple[list[MaaEvent], int]:
        """返回 ``(mark 之后的新事件, 新游标)``。"""
        with self._lock:
            return list(self._events[mark:]), len(self._events)

    def wait_chain(self, timeout_s: float, mark: int | None = None) -> ChainResult:
        """等待当前任务链收尾，返回 ``ChainResult``。

        终态判定：见到 ``TaskChainCompleted`` 记为成功；
        ``TaskChainError`` 记为失败；``AllTasksCompleted`` 直接收尾。
        超时返回 ``finished_ok=False`` 且 ``terminal_msg=None``。
        """
        cursor = self.mark() if mark is None else mark
        seen: list[MaaEvent] = []
        deadline = time.monotonic() + timeout_s
        terminal: str | None = None
        while time.monotonic() < deadline:
            fresh, cursor = self.events_since(cursor)
            seen.extend(fresh)
            for evt in fresh:
                if evt.msg in _CHAIN_TERMINAL_MSGS:
                    terminal = evt.msg
            if terminal == "AllTasksCompleted" or terminal in (
                "TaskChainCompleted",
                "TaskChainError",
            ):
                break
            time.sleep(0.2)
        ok = terminal == "TaskChainCompleted"
        if terminal == "TaskChainCompleted":
            # AllTasksCompleted 紧随其后，补收一个拍
            time.sleep(0.5)
            fresh, cursor = self.events_since(cursor)
            seen.extend(fresh)
        return ChainResult(ok, terminal, seen)

    # ---- 连接与任务 ----

    def connect(self) -> bool:
        """创建 Asst 实例并连接模拟器，返回是否成功。"""
        cb_type = self._asst_mod.CallBackType

        @cb_type
        def _cb(msg: int, details: bytes, arg: Any) -> None:
            self._on_callback(msg, details, arg)

        self._cb_ref = _cb  # 持有引用，防止 CFUNCTYPE 被 GC
        self._asst = self._asst_mod(callback=_cb)
        ok = bool(self._asst.connect(self._adb, self._addr))
        if ok:
            time.sleep(1.0)  # 等待截图通道协商（ConnectionInfo）
        return ok

    def append_copilot(self, job_path: str | Path, formation: bool = True) -> int:
        """下发 Copilot 自动战斗任务，返回 task id（0 表示参数被拒）。"""
        if self._asst is None:
            raise RuntimeError("尚未 connect")
        return int(
            self._asst.append_task(
                "Copilot",
                {
                    "filename": str(job_path),
                    "formation": formation,
                    "use_sanity_potion": False,
                },
            )
        )

    def start(self) -> bool:
        """开始执行已下发的任务队列。"""
        return bool(self._asst.start())

    def running(self) -> bool:
        """任务是否仍在执行。"""
        return bool(self._asst and self._asst.running())

    def stop(self) -> bool:
        """请求停止当前任务（MAA 在管线步间检查停止标记）。"""
        if self._asst is None:
            return False
        return bool(self._asst.stop())

    def close(self) -> None:
        """停止并释放实例（幂等；关闭路径不容许抛出）。"""
        with contextlib.suppress(Exception):
            self.stop()
        self._asst = None

    # ---- ADB 通道（截图与模拟输入）----

    @property
    def adb_addr(self) -> str:
        """当前实例的 ADB 地址。"""
        return self._addr

    def screencap(self, timeout_s: float = 15.0) -> Image.Image:
        """经 adb exec-out 截一帧，返回 PIL Image（RGB）。

        串行化（内部锁）以避免与事件截图、状态采集并发打 adb。
        """
        with self._cap_lock:
            proc = subprocess.run(
                [self._adb, "-s", self._addr, "exec-out", "screencap", "-p"],
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError(f"adb screencap 失败 rc={proc.returncode}")
        return Image.open(io.BytesIO(proc.stdout)).convert("RGB")

    def _adb_shell_input(self, args: list[str]) -> None:
        subprocess.run(
            [self._adb, "-s", self._addr, "shell", "input", *args],
            capture_output=True,
            timeout=10,
            check=False,
        )

    def tap(self, x: int, y: int) -> None:
        """点触 (x, y)（1920x1080 基线像素坐标）。"""
        self._adb_shell_input(["tap", str(x), str(y)])

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 800) -> None:
        """匀速滑动（地图拖动 / 列表翻页用）。"""
        self._adb_shell_input(
            ["swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)]
        )

    def key_back(self) -> None:
        """发送安卓返回键（游戏内多数界面等效于左上角返回箭头）。"""
        self._adb_shell_input(["keyevent", "4"])


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    """冒烟入口：连接实例并打印 10 秒事件流。

    用法::

        python -m bridge.maa_driver [--instance <yaml>] [--user-dir <dir>]
            [--seconds 10]
    """
    import argparse

    parser = argparse.ArgumentParser(description="maa_driver 连接冒烟")
    parser.add_argument(
        "--instance",
        default=str(_repo_root() / "configs" / "instances" / "mumu_local.yaml"),
        help="实例定义 yaml（默认 configs/instances/mumu_local.yaml）",
    )
    parser.add_argument(
        "--user-dir",
        default=None,
        help="MAA 运行数据目录（默认 <cwd>/debug/maa_user）",
    )
    parser.add_argument("--seconds", type=float, default=10.0, help="监听时长（秒）")
    args = parser.parse_args()

    user_dir = args.user_dir or str(Path.cwd() / "debug" / "maa_user")
    driver = MaaDriver(load_instance_config(args.instance), user_dir)
    print(f"[smoke] 连接 {driver.adb_addr} ...")
    if not driver.connect():
        print("[smoke] connect 失败")
        return 1
    img = driver.screencap()
    print(f"[smoke] screencap 正常: {img.size[0]}x{img.size[1]}")
    print(f"[smoke] 监听 {args.seconds}s 事件流：")
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        try:
            evt = driver._queue.get(timeout=0.5)
        except queue.Empty:
            continue
        print(f"  +{evt.t_mono_ms:>8}ms {evt.msg} {str(evt.details)[:160]}")
    driver.close()
    print("[smoke] 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
