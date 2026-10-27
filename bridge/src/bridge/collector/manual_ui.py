"""手动操作录制窗口：玩家直接在MuMu操作，助手仅显示记录状态与导出入口。"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .exporting import export_data
from .service import CollectorService


class ManualWindow(tk.Toplevel):
    """不需要作业队列的一局录制，结算自动结束，也可人工保留半局。"""

    def __init__(self, master: Any, service: CollectorService) -> None:
        super().__init__(master)
        self._poll_task = None
        self.service = service
        self.title("采集玩家手动操作")
        scale = getattr(master, "ui_scale", 1.0)
        self.geometry(f"{int(700 * scale)}x{int(500 * scale)}")
        self.stage = tk.StringVar()
        self.progress = tk.StringVar(value="尚未开始录制")
        self.detail = tk.StringVar(
            value="请先在主窗口设置中选择非C盘数据目录和所选MuMu。"
        )
        ttk.Label(
            self,
            text="玩家直接在 MuMu 窗口操作",
            font=("Microsoft YaHei UI", 16, "bold"),
        ).pack(pady=(22, 12))
        ttk.Label(
            self,
            text="无需作业文件或预定打法。请先停止所选设备上的自动作业和代理作战。\n"
            "录制游戏画面与Android原始触控/按键，不监听电脑其他窗口，助手不会替你点击。",
            wraplength=580,
            justify="left",
        ).pack(padx=25, anchor="w")
        row = ttk.Frame(self)
        row.pack(fill="x", padx=25, pady=18)
        ttk.Label(row, text="关卡备注（可留空）").pack(side="left")
        ttk.Entry(row, textvariable=self.stage).pack(
            side="left", fill="x", expand=True, padx=10
        )
        ttk.Label(
            self, textvariable=self.progress, font=("Microsoft YaHei UI", 12, "bold")
        ).pack(pady=8)
        ttk.Label(self, textvariable=self.detail, wraplength=580, justify="left").pack(
            padx=25, pady=8
        )
        ttk.Label(
            self,
            text="从编队或进关时开始；识别到真实结算后自动结束。\n"
            "未识别的干员/部署/技能标签保留缺失，原始画面与输入一起保存供复核。",
            wraplength=580,
            justify="left",
        ).pack(padx=25, pady=8, anchor="w")
        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=25, pady=14)
        self.start_button = ttk.Button(
            buttons, text="开始录制这一局", command=self.start
        )
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(
            buttons, text="停止录制（保留半局）", command=self.stop
        )
        self.stop_button.pack(side="left", padx=8)
        ttk.Button(buttons, text="导出手动操作包（含画面）", command=self.export).pack(
            side="left"
        )
        self._poll_task = self.after(300, self.poll)

    def destroy(self) -> None:
        """关闭小窗只撤销其刷新回调，后台录制仍由显式停止按钮控制。"""
        if self._poll_task is not None:
            self.after_cancel(self._poll_task)
        super().destroy()

    def start(self) -> None:
        """启动旁路读取，不消耗理智，不请求游戏输入。"""
        try:
            self.service.start_manual(self.stage.get().strip())
        except Exception as error:
            messagebox.showerror("录制未开始", str(error), parent=self)

    def stop(self) -> None:
        """仅结束录制，游戏继续由玩家控制。"""
        if self.service.status().get("manual") and self.service.is_running():
            self.service.send("stop_now")

    def export(self) -> None:
        """明确导出原图与原始输入，不将其冒充现有战术动作监督格式。"""
        if self.service.is_running():
            messagebox.showinfo(
                "请先结束录制", "等待结算或停止录制后再导出。", parent=self
            )
            return
        filename = filedialog.asksaveasfilename(
            parent=self,
            title="保存手动操作与画面",
            defaultextension=".zip",
            initialfile="玩家手动操作数据.zip",
            filetypes=[("ZIP", "*.zip")],
        )
        if filename:
            self.master.background(
                lambda: export_data(
                    self.service.data_root(),
                    Path(filename),
                    include_images=True,
                    manual_only=True,
                ),
                lambda value: messagebox.showinfo(
                    "已导出", f"已保存{value['files']}个文件，原件保留。", parent=self
                ),
                exclusive=True,
            )

    def poll(self) -> None:
        """显示真实记录数量与结算状态，不拿未解析的输入算成成功战术动作。"""
        if not self.winfo_exists():
            return
        try:
            state = self.service.status()
            manual = bool(state.get("manual"))
            active = self.service.is_running()
            self.start_button.configure(state="disabled" if active else "normal")
            self.stop_button.configure(
                state="normal" if active and manual else "disabled"
            )
            if manual:
                self.progress.set(
                    f"截图 {state.get('manual_frames', 0)} · "
                    f"原始事件 {state.get('manual_events', 0)} · "
                    f"触点轨迹 {state.get('manual_gestures', 0)}"
                )
                result = (state.get("manual_result") or {}).get("battle_result")
                result = {"win": "胜利", "lost": "失败", "unknown": "未确认"}.get(
                    result
                )
                self.detail.set(
                    state.get("message", "")
                    + (f"；观测结算：{result}" if result else "")
                )
        except (OSError, ValueError):
            self.detail.set("正在重新读取录制状态")
        self._poll_task = self.after(500, self.poll)
