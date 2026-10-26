"""森空岛扫码窗口：复用现有账号通道，本地绘制二维码并选择绑定角色。"""

from __future__ import annotations

import subprocess
import sys
import tkinter as tk
import uuid
from collections.abc import Callable
from pathlib import Path
from tkinter import ttk

from .discovery import application_base, find_installations
from .service import CollectorService, child_environment
from .storage import read_json, write_json


def scan_script() -> Path:
    """打包与源码共用同一份扫码入口。"""
    for root in (
        application_base(),
        Path(getattr(sys, "_MEIPASS", application_base())),
    ):
        path = root / "tools/skland/gui.mjs"
        if path.exists():
            return path
    return Path(__file__).resolve().parents[3] / "tools/skland/gui.mjs"


class ScanWindow(tk.Toplevel):
    """无需终端输入的扫码和选角界面。"""

    def __init__(
        self, master: tk.Misc, service: CollectorService, on_done: Callable[[], None]
    ) -> None:
        super().__init__(master)
        self.title("获取干员练度")
        self.geometry("480x590")
        self.resizable(False, False)
        self.service, self.on_done = service, on_done
        self.directory = service.home / ".secrets" / f"scan_{uuid.uuid4().hex}"
        self.directory.mkdir(parents=True)
        self.event_file = self.directory / "progress.json"
        self.selection_file = self.directory / "selected_role.json"
        self.out = service.home / "roster_skland.local.json"
        ttk.Label(
            self, text="森空岛扫码获取练度", font=("Microsoft YaHei UI", 16, "bold")
        ).pack(pady=(22, 8))
        self.message = tk.StringVar(value="正在准备二维码…")
        ttk.Label(
            self, textvariable=self.message, wraplength=420, justify="center"
        ).pack(pady=8)
        self.canvas = tk.Canvas(
            self, width=290, height=290, bg="white", highlightthickness=0
        )
        self.canvas.pack(pady=12)
        self.choices = ttk.Frame(self)
        self.choices.pack(fill="x", padx=30)
        ttk.Label(
            self,
            text="登录凭据只保留在本机，不会进入导出包。",
            foreground="#64748b",
            wraplength=420,
        ).pack(pady=10)
        self.close_button = ttk.Button(self, text="取消", command=self.close)
        self.close_button.pack(pady=10)
        node = find_installations(service.settings)["node_path"]
        self.process: subprocess.Popen | None = None
        self.latest = ""
        if not node or not scan_script().exists():
            self.message.set("扫码组件未找到。请使用完整发行包，或先导入已有名册。")
        else:
            self.log = (self.directory / "scan.log").open("wb")
            self.process = subprocess.Popen(
                [
                    node,
                    str(scan_script()),
                    "--out",
                    str(self.out),
                    "--cred",
                    str(service.home / ".secrets/skland_cred.json"),
                    "--events",
                    str(self.event_file),
                    "--selection",
                    str(self.selection_file),
                    "--login",
                    "true",
                ],
                stdout=self.log,
                stderr=self.log,
                env=child_environment(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.after(400, self.poll)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.transient(master)
        self.wait_visibility()
        self.grab_set()

    def poll(self) -> None:
        """二维码与角色列表只在本地文件中传递，不经过第三方二维码服务。"""
        if not self.winfo_exists():
            return
        value = read_json(self.event_file, {}) or {}
        updated = value.get("updated_at")
        if updated and updated != self.latest:
            self.latest = updated
            self.message.set(value.get("message", "正在获取"))
            if value.get("matrix"):
                self.draw_matrix(value["matrix"])
            elif value.get("status") == "choose_role":
                self.canvas.delete("all")
                for child in self.choices.winfo_children():
                    child.destroy()
                for role in value["roles"]:
                    ttk.Button(
                        self.choices,
                        text=f"{role['server']}  {role['name']}",
                        command=lambda uid=role["uid"]: write_json(
                            self.selection_file, {"uid": uid}
                        ),
                    ).pack(fill="x", pady=4)
            elif value.get("status") == "done":
                player = value.get("player") or {}
                self.service.settings.bind_account(
                    str(player.get("uid") or uuid.uuid4().hex)
                )
                self.service.settings.roster_file = str(self.out)
                if player.get("nickname"):
                    self.service.settings.nickname = player["nickname"]
                self.service.save()
                self.on_done()
                self.close_button.configure(text="完成")
        if self.process and self.process.poll() is not None:
            if value.get("status") not in ("done", "error"):
                self.message.set("扫码组件已退出，可关闭窗口后重新扫码。")
            return
        self.after(400, self.poll)

    def draw_matrix(self, matrix: list[list[bool]]) -> None:
        """以Canvas格子显示二维码，不需要额外图形或二维码依赖。"""
        self.canvas.delete("all")
        cell = max(1, 280 // (len(matrix) + 8))
        size = cell * (len(matrix) + 8)
        margin = (290 - size) // 2 + cell * 4
        for row, line in enumerate(matrix):
            for col, dark in enumerate(line):
                if dark:
                    x, y = margin + col * cell, margin + row * cell
                    self.canvas.create_rectangle(
                        x, y, x + cell, y + cell, fill="black", outline="black"
                    )

    def close(self) -> None:
        """取消只停止本窗口创建的扫码请求，不触碰游戏或其他采集进程。"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if hasattr(self, "log"):
            self.log.close()
        self.destroy()
