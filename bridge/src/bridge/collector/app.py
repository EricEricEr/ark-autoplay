"""方舟数据采集助手窗口：普通玩家通过选项和按钮完成采集。"""

from __future__ import annotations

import ctypes
import errno
import os
import queue
import subprocess
import threading
import tkinter as tk
import uuid
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from . import VERSION
from .discovery import Device, collect_diagnostics, instance_config
from .exporting import export_data
from .roster_ui import ScanWindow
from .service import CollectorService, child_environment
from .storage import command_for, now, read_json, write_json

BLUE = "#2563eb"
INK = "#16243b"
MUTED = "#64748b"
BG = "#f3f6fb"
GREEN = "#15803d"


class CollectorApp(tk.Tk):
    """不显示终端的采集窗口；耗时任务均在后台执行。"""

    def __init__(self, home: Path | None = None, *, demo: bool = False) -> None:
        if os.name == "nt":
            try:
                user32 = ctypes.WinDLL("user32", use_last_error=True)
                user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
                user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            except (AttributeError, OSError):
                pass
        super().__init__()
        self.title("方舟数据采集助手")
        self.configure(bg=BG)
        self.ui_scale = max(1.0, self.winfo_fpixels("1i") / 96)
        width = int(min(1120 * self.ui_scale, self.winfo_screenwidth() - 70))
        height = int(min(800 * self.ui_scale, self.winfo_screenheight() - 100))
        self.geometry(f"{width}x{height}")
        self.minsize(900, 650)
        self.service = CollectorService(home, demo=demo)
        self.events: queue.Queue = queue.Queue()
        self.devices: list[Device] = []
        self.running_action = False
        self.exit_after_stop = False
        self.pages: dict[str, tk.Frame] = {}
        self.nav_buttons: dict[str, tk.Button] = {}
        self.last_status: dict[str, Any] = {}
        self.environment_text = tk.StringVar(value="首次使用，先检查采集环境")
        self.status_text = tk.StringVar(value="准备开始")
        self.detail_text = tk.StringVar(value="选择模拟器并导入任务后，点击开始采集。")
        self.task_text = tk.StringVar(value="还没有导入采集任务")
        self.roster_text = tk.StringVar(value="尚未获取干员练度")
        self.profile_text = tk.StringVar(value=self.service.settings.nickname)
        self.current_stage = tk.StringVar(value="—")
        self.count_total, self.count_done, self.count_valid, self.count_review = (
            tk.StringVar(value="0") for _ in range(4)
        )
        self.device_var = tk.StringVar()
        self.allow_medicine = tk.BooleanVar(value=self.service.settings.allow_medicine)
        self.allow_originite = tk.BooleanVar(
            value=self.service.settings.allow_originite
        )
        self.automatic = tk.BooleanVar(value=self.service.settings.automatic_recovery)
        self.convert = tk.BooleanVar(value=self.service.settings.convert_after)
        self.support = tk.BooleanVar(
            value=bool(self.service.settings.support_unit_usage)
        )
        self.include_images = tk.BooleanVar(value=False)
        self.include_raw_records = tk.BooleanVar(value=False)
        self.paths: dict[str, tk.StringVar] = {}
        self._style()
        self._layout()
        self.refresh_roster()
        self.refresh_tasks()
        self.show_page("采集面板")
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self.report_callback_exception = self.callback_error
        self.after(150, self.poll)
        if self.service.startup_notice:
            self.after(
                500,
                lambda: messagebox.showinfo(
                    "已恢复设置", self.service.startup_notice, parent=self
                ),
            )
        if not demo:
            self.after(400, lambda: self.auto_detect(silent=True))
        else:
            self.environment_text.set(
                "演示模式已就绪 · 不连接模拟器，不产生真实训练数据"
            )
            self.device_var.set("演示设备（不会控制游戏）")

    def _style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(12, 8))
        style.configure(
            "Primary.TButton",
            foreground="white",
            background=BLUE,
            borderwidth=0,
            padding=(18, 11),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#1d4ed8"), ("disabled", "#a7b7d2")],
        )
        style.configure(
            "TLabel",
            font=("Microsoft YaHei UI", 10),
            background="white",
            foreground=INK,
        )
        style.configure("TFrame", background="white")
        style.configure(
            "TCheckbutton",
            background="white",
            font=("Microsoft YaHei UI", 10),
            padding=5,
        )
        style.configure("TCombobox", padding=6, font=("Microsoft YaHei UI", 10))
        style.configure(
            "Horizontal.TProgressbar",
            background=BLUE,
            troughcolor="#e2e8f0",
            borderwidth=0,
        )
        self.option_add("*Font", ("Microsoft YaHei UI", 10))

    def _label(
        self,
        parent: tk.Misc,
        text: str,
        *,
        size: int = 10,
        color: str = INK,
        bold: bool = False,
        **kwargs: Any,
    ) -> tk.Label:
        return tk.Label(
            parent,
            text=text,
            bg="white",
            fg=color,
            font=("Microsoft YaHei UI", size, "bold" if bold else "normal"),
            **kwargs,
        )

    def _card(self, parent: tk.Misc, title: str, subtitle: str = "") -> tk.Frame:
        card = tk.Frame(
            parent,
            bg="white",
            highlightbackground="#e0e7ef",
            highlightthickness=1,
            padx=20,
            pady=16,
        )
        card.pack(fill="x", pady=(0, 12))
        self._label(card, title, size=12, bold=True).pack(anchor="w")
        if subtitle:
            self._label(
                card, subtitle, color=MUTED, wraplength=750, justify="left"
            ).pack(anchor="w", pady=(5, 10))
        return card

    def _layout(self) -> None:
        sidebar = tk.Frame(self, bg=INK, width=int(180 * self.ui_scale))
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        logo = tk.Canvas(sidebar, width=48, height=48, bg=INK, highlightthickness=0)
        logo.pack(anchor="w", padx=22, pady=(26, 10))
        logo.create_rectangle(2, 2, 46, 46, fill=BLUE, outline=BLUE)
        logo.create_text(
            24, 24, text="采", fill="white", font=("Microsoft YaHei UI", 22, "bold")
        )
        tk.Label(
            sidebar,
            text="方舟数据采集",
            bg=INK,
            fg="white",
            font=("Microsoft YaHei UI", 14, "bold"),
        ).pack(anchor="w", padx=22)
        tk.Label(
            sidebar,
            text="让每一局都有记录",
            bg=INK,
            fg="#9fb1ce",
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", padx=22, pady=(7, 28))
        for label in ("采集面板", "账号与练度", "任务与设置", "导出与帮助"):
            button = tk.Button(
                sidebar,
                text=label,
                anchor="w",
                relief="flat",
                bd=0,
                bg=INK,
                fg="#bdcbe0",
                activebackground="#253b5a",
                activeforeground="white",
                font=("Microsoft YaHei UI", 11),
                padx=22,
                pady=13,
                command=lambda value=label: self.show_page(value),
            )
            button.pack(fill="x", padx=8, pady=3)
            self.nav_buttons[label] = button
        tk.Label(
            sidebar,
            text=f"采集助手  {VERSION}\n数据保存在您的电脑",
            justify="left",
            bg=INK,
            fg="#8fa4c2",
            font=("Microsoft YaHei UI", 9),
        ).pack(side="bottom", anchor="w", padx=22, pady=24)
        body = tk.Frame(self, bg=BG)
        body.pack(side="left", fill="both", expand=True)
        header = tk.Frame(body, bg=BG)
        header.pack(fill="x", padx=26, pady=(22, 14))
        self.page_title = tk.Label(
            header,
            text="采集面板",
            bg=BG,
            fg=INK,
            font=("Microsoft YaHei UI", 21, "bold"),
        )
        self.page_title.pack(side="left")
        self.state_badge = tk.Label(
            header,
            textvariable=self.status_text,
            bg="#e8eef8",
            fg=BLUE,
            font=("Microsoft YaHei UI", 10),
            padx=13,
            pady=7,
        )
        self.state_badge.pack(side="right")
        if self.service.settings.demo:
            tk.Label(
                body,
                text="演示模式：只使用合成任务，不连接游戏、不产生真实训练数据",
                bg="#fff4d6",
                fg="#854d0e",
                pady=7,
            ).pack(fill="x", padx=26, pady=(0, 10))
        scrolling = tk.Frame(body, bg=BG)
        scrolling.pack(fill="both", expand=True, padx=26, pady=(0, 12))
        self.page_canvas = tk.Canvas(scrolling, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            scrolling, orient="vertical", command=self.page_canvas.yview
        )
        scrollbar.pack(side="right", fill="y")
        self.page_canvas.pack(side="left", fill="both", expand=True)
        self.page_canvas.configure(yscrollcommand=scrollbar.set)
        container = tk.Frame(self.page_canvas, bg=BG)
        window = self.page_canvas.create_window(0, 0, anchor="nw", window=container)
        container.bind(
            "<Configure>",
            lambda event: self.page_canvas.configure(
                scrollregion=self.page_canvas.bbox("all")
            ),
        )
        self.page_canvas.bind(
            "<Configure>",
            lambda event: self.page_canvas.itemconfigure(window, width=event.width),
        )
        self.bind(
            "<MouseWheel>",
            lambda event: self.page_canvas.yview_scroll(
                int(-event.delta / 120), "units"
            ),
        )
        for name in self.nav_buttons:
            page = tk.Frame(container, bg=BG)
            self.pages[name] = page
        self._dashboard(self.pages["采集面板"])
        self._account_page(self.pages["账号与练度"])
        self._settings_page(self.pages["任务与设置"])
        self._export_page(self.pages["导出与帮助"])

    def _dashboard(self, page: tk.Frame) -> None:
        self._label_banner(
            page, "① 连接模拟器    →    ② 获取练度、导入任务    →    ③ 开始采集"
        )
        card = self._card(
            page, "采集准备", "首次使用点一下自动检测，选择您准备采集的模拟器。"
        )
        row = ttk.Frame(card)
        row.pack(fill="x")
        self.device_combo = ttk.Combobox(
            row, textvariable=self.device_var, state="readonly", width=35
        )
        self.device_combo.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.device_combo.bind("<<ComboboxSelected>>", self.select_device)
        ttk.Button(row, text="自动检测", command=self.auto_detect).pack(side="right")
        tk.Label(
            card,
            textvariable=self.environment_text,
            bg="white",
            fg=MUTED,
            anchor="w",
            wraplength=740,
            justify="left",
        ).pack(fill="x", pady=(10, 0))
        row = ttk.Frame(card)
        row.pack(fill="x", pady=(9, 0))
        ttk.Button(
            row, text="选择采集运行库", command=lambda: self.choose_folder("maa_dir")
        ).pack(side="left")
        ttk.Button(
            row, text="获取干员练度", command=lambda: self.show_page("账号与练度")
        ).pack(side="left", padx=8)
        ttk.Button(row, text="调整模拟器画面", command=self.fix_display).pack(
            side="left", padx=8
        )
        task = self._card(page, "本次任务")
        row = ttk.Frame(task)
        row.pack(fill="x", pady=(8, 0))
        tk.Label(
            row,
            textvariable=self.task_text,
            bg="white",
            fg=INK,
            anchor="w",
            wraplength=550,
            justify="left",
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="导入采集任务", command=self.import_tasks).pack(
            side="right"
        )
        progress = self._card(page, "实时进度")
        numbers = tk.Frame(progress, bg="white")
        numbers.pack(fill="x", pady=(12, 12))
        for col, (caption, variable) in enumerate(
            (
                ("本轮作业", self.count_total),
                ("已尝试", self.count_done),
                (
                    "模拟成功" if self.service.settings.demo else "有效通关",
                    self.count_valid,
                ),
                ("待复核", self.count_review),
            )
        ):
            numbers.columnconfigure(col, weight=1)
            box = tk.Frame(numbers, bg="white")
            box.grid(row=0, column=col, sticky="ew")
            tk.Label(
                box,
                textvariable=variable,
                bg="white",
                fg=BLUE if col != 3 else "#b45309",
                font=("Microsoft YaHei UI", 24, "bold"),
            ).pack(anchor="w")
            self._label(box, caption, color=MUTED).pack(anchor="w")
        self.progress = ttk.Progressbar(progress, mode="determinate", maximum=1)
        self.progress.pack(fill="x", pady=(0, 9))
        row = ttk.Frame(progress)
        row.pack(fill="x")
        self._label(row, "当前关卡：", color=MUTED).pack(side="left")
        tk.Label(row, textvariable=self.current_stage, bg="white", fg=INK).pack(
            side="left"
        )
        tk.Label(
            progress,
            textvariable=self.detail_text,
            bg="white",
            fg=MUTED,
            anchor="w",
            wraplength=740,
            justify="left",
        ).pack(fill="x", pady=(5, 0))
        actions = tk.Frame(page, bg=BG)
        actions.pack(fill="x", pady=(2, 0))
        self.start_button = ttk.Button(
            actions,
            text="开始 / 继续未完成任务",
            style="Primary.TButton",
            command=self.start,
        )
        self.start_button.pack(side="left", padx=(0, 8))
        self.pause_button = ttk.Button(
            actions, text="当前局后暂停", command=self.pause_resume
        )
        self.pause_button.pack(side="left", padx=(0, 8))
        self.recover_button = ttk.Button(
            page, text="已处理，检查并继续", command=self.recover
        )
        self.recover_button.pack(anchor="w", pady=8)
        ttk.Button(
            actions, text="当前局后停止", command=lambda: self.send("stop_after")
        ).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="立即停止", command=self.stop_now).pack(side="right")

    def _label_banner(self, page: tk.Frame, text: str) -> None:
        tk.Label(
            page, text=text, bg="#e8eefc", fg="#345486", anchor="w", padx=13, pady=11
        ).pack(fill="x", pady=(0, 13))

    def _account_page(self, page: tk.Frame) -> None:
        card = self._card(
            page,
            "当前采集账号",
            "选择的练度资料应与模拟器当前登录账号一致。软件不需要游戏账号密码。",
        )
        tk.Label(
            card,
            textvariable=self.profile_text,
            bg="white",
            fg=INK,
            font=("Microsoft YaHei UI", 15, "bold"),
        ).pack(anchor="w", pady=7)
        tk.Label(
            card,
            textvariable=self.roster_text,
            bg="white",
            fg=MUTED,
            wraplength=750,
            justify="left",
        ).pack(anchor="w", pady=7)
        row = ttk.Frame(card)
        row.pack(fill="x", pady=(9, 0))
        ttk.Button(
            row, text="森空岛扫码获取", style="Primary.TButton", command=self.open_scan
        ).pack(side="left", padx=(0, 9))
        ttk.Button(row, text="游戏内扫描", command=self.operbox).pack(
            side="left", padx=(0, 9)
        )
        ttk.Button(row, text="导入已有名册", command=self.import_roster).pack(
            side="left"
        )
        self._label(
            card,
            "扫码可获取技能和模组；游戏内扫描缺少的字段会明确标记，不会猜填。",
            color=MUTED,
            wraplength=750,
            justify="left",
        ).pack(anchor="w", pady=(14, 0))
        supply = self._card(
            page,
            "理智补给",
            "默认不使用任何补给。只有当前账号持有者明确同意后，才开启相应选项。",
        )
        ttk.Checkbutton(
            supply,
            text="允许自动使用当前账号已有理智药（可能用尽）",
            variable=self.allow_medicine,
            command=lambda: self.authorize("medicine"),
        ).pack(anchor="w")
        ttk.Checkbutton(
            supply,
            text="允许自动使用已有源石补理智（可能用尽）",
            variable=self.allow_originite,
            command=lambda: self.authorize("originite"),
        ).pack(anchor="w")
        self._label(
            supply, "不购买源石、不充值。更换账号后，补给授权自动清除。", color=MUTED
        ).pack(anchor="w", pady=(10, 0))
        ttk.Button(page, text="更换采集账号", command=self.change_account).pack(
            anchor="w", pady=8
        )

    def _settings_page(self, page: tk.Frame) -> None:
        card = self._card(page, "采集选项")
        ttk.Checkbutton(
            card,
            text="自动恢复模拟器连接，并在游戏退出后尝试重新打开",
            variable=self.automatic,
            command=self.save_options,
        ).pack(anchor="w")
        ttk.Checkbutton(
            card,
            text="允许按作业需要借用一名助战干员",
            variable=self.support,
            command=self.save_options,
        ).pack(anchor="w")
        ttk.Checkbutton(
            card,
            text="采集完成后自动转换结构化训练数据",
            variable=self.convert,
            command=self.save_options,
        ).pack(anchor="w")
        self._label(
            card,
            "无法自动处理的登录、更新或弹窗会提示您处理；失败作业记账后继续。",
            color=MUTED,
            wraplength=750,
            justify="left",
        ).pack(anchor="w", pady=(10, 0))
        card = self._card(
            page,
            "文件位置",
            "采集数据须选择非C盘目录；软件设置和系统缓存保持默认位置。",
        )
        for name, caption in (
            ("data_dir", "数据保存位置"),
            ("maa_dir", "采集运行库"),
            ("mumu_manager", "MuMu安装位置"),
        ):
            row = ttk.Frame(card)
            row.pack(fill="x", pady=6)
            self._label(row, caption, color=MUTED, width=14, anchor="w").pack(
                side="left"
            )
            variable = tk.StringVar(value=getattr(self.service.settings, name))
            self.paths[name] = variable
            ttk.Entry(row, textvariable=variable, state="readonly").pack(
                side="left", fill="x", expand=True, padx=(0, 8)
            )
            ttk.Button(
                row, text="选择…", command=lambda key=name: self.choose_folder(key)
            ).pack(side="right")
        row = ttk.Frame(card)
        row.pack(fill="x", pady=(12, 0))
        ttk.Button(row, text="打开数据文件夹", command=self.open_data).pack(
            side="left", padx=(0, 9)
        )
        ttk.Button(row, text="以管理员权限重新打开", command=self.restart_admin).pack(
            side="left"
        )
        self._label(
            card,
            "普通权限能够采集时，无需提权。管理员权限不代表已授权使用源石。",
            color=MUTED,
            wraplength=750,
            justify="left",
        ).pack(anchor="w", pady=(12, 0))

    def _export_page(self, page: tk.Frame) -> None:
        card = self._card(
            page,
            "一键导出采集包",
            "导出ZIP后交给维护者即可。原始数据继续保存在本机，软件不会自动上传。",
        )
        ttk.Checkbutton(
            card,
            text="包含原始截图（文件更大，画面可能包含游戏内昵称）",
            variable=self.include_images,
        ).pack(anchor="w", pady=9)
        ttk.Checkbutton(
            card,
            text="保留原始记录（含未完成数据和本机路径，仅交可信维护者）",
            variable=self.include_raw_records,
        ).pack(anchor="w")
        self._label(
            card,
            "默认导出结构化对局、作业与干员练度，并移除登录凭据、本机配置和个人路径。",
            color=MUTED,
            wraplength=750,
            justify="left",
        ).pack(anchor="w", pady=8)
        ttk.Button(
            card, text="导出采集包…", style="Primary.TButton", command=self.export
        ).pack(anchor="w", pady=10)
        card = self._card(page, "异常与帮助")
        help_text = (
            "• 设备断开：软件保留队列并尝试恢复连接。\n"
            "• 作业失败：保留原始记录，继续下一份，不当作成功。\n"
            "• 登录过期、维护更新或未知弹窗：请在模拟器中处理。\n"
            "• 软件异常退出：重新打开后，可继续未完成任务。\n"
            "• 需要退出时，可选择完成当前局后停止，减少半局数据。"
        )
        self._label(
            card, help_text, color=MUTED, justify="left", anchor="w", wraplength=750
        ).pack(fill="x", pady=9)
        row = ttk.Frame(card)
        row.pack(fill="x", pady=8)
        ttk.Button(row, text="查看本轮日志", command=self.view_log).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(
            row, text="重试未通过项", command=lambda: self.start(retry_failed=True)
        ).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="打开数据文件夹", command=self.open_data).pack(side="left")
        ttk.Button(
            card,
            text="重新采集本任务全部作业",
            command=lambda: self.start(restart_all=True),
        ).pack(anchor="w", pady=(8, 0))
        ttk.Button(
            card, text="重新转换已有数据（不打关）", command=self.convert_existing
        ).pack(anchor="w", pady=8)
        ttk.Button(card, text="导出诊断包…", command=self.export_diagnostics).pack(
            anchor="w", pady=8
        )
        ttk.Button(card, text="停止无响应任务", command=self.stop_unresponsive).pack(
            anchor="w", pady=8
        )
        ttk.Button(
            card, text="体验演示模式（不会连接游戏）", command=self.open_demo
        ).pack(anchor="w", pady=(8, 0))

    def show_page(self, name: str) -> None:
        for page in self.pages.values():
            page.pack_forget()
        self.pages[name].pack(fill="both", expand=True)
        self.page_canvas.yview_moveto(0)
        self.page_title.configure(text=name)
        for key, button in self.nav_buttons.items():
            button.configure(
                bg="#253b5a" if key == name else INK,
                fg="white" if key == name else "#bdcbe0",
            )

    def error(self, error: Exception) -> None:
        self.detail_text.set("操作没有完成，可按提示处理后重试。")
        try:
            self.service.home.mkdir(parents=True, exist_ok=True)
            with (self.service.home / "app_errors.log").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(f"{now()} {type(error).__name__}: {error}\n")
        except OSError:
            pass
        if isinstance(error, PermissionError):
            text = (
                "没有权限访问所选位置。请换一个数据目录，"
                "或在设置中以管理员权限重新打开。"
            )
        elif isinstance(error, OSError) and error.errno == errno.ENOSPC:
            text = "保存磁盘已满。已有数据保留，请释放空间或选择新的数据位置。"
        elif isinstance(error, FileNotFoundError):
            text = "没有找到所需文件，请重新选择任务或运行库。"
        elif isinstance(error, subprocess.TimeoutExpired):
            text = "设备暂时没有响应，请确认模拟器已正常打开后重试。"
        else:
            text = str(error)
        messagebox.showerror("操作未完成", text, parent=self)

    def callback_error(self, kind, error, traceback) -> None:
        """按钮或扫码回调出错时保留窗口并给出普通中文提示。"""
        self.error(error)

    def recover(self) -> None:
        """玩家处理登录/磁盘问题后，检查实际条件再恢复原任务。"""
        if (
            not self.service.is_running()
            and not self.service.settings.accepted_notice
            and not self.service.settings.demo
        ):
            self.start()
            return
        self.background(
            self.service.recover,
            lambda _: self.detail_text.set("检查完成，正在恢复任务"),
            exclusive=True,
        )

    def convert_existing(self) -> None:
        """独立转换已有原件，不要求队列、设备或补给权限。"""
        try:
            self.service.start(conversion_only=True)
            self.show_page("采集面板")
        except Exception as error:
            self.error(error)

    def stop_unresponsive(self) -> None:
        if messagebox.askyesno(
            "停止本软件任务",
            "将结束本采集档案的工作进程，保留原始数据。当前局可能需重跑。是否继续？",
            parent=self,
        ):
            self.background(
                self.service.stop_unresponsive,
                lambda _: self.detail_text.set("已结束任务，可检查后续跑"),
                exclusive=True,
            )

    def export_diagnostics(self) -> None:
        filename = filedialog.asksaveasfilename(
            parent=self,
            title="保存诊断包",
            defaultextension=".zip",
            initialfile="采集诊断.zip",
            filetypes=[("诊断包", "*.zip")],
        )
        if filename:
            self.background(
                lambda: export_data(
                    Path(self.service.settings.data_dir or self.service.home),
                    Path(filename),
                    diagnostics_home=self.service.home,
                    diagnostics_only=True,
                ),
                lambda _: messagebox.showinfo(
                    "诊断包已保存",
                    "已导出脱敏诊断，不含登录凭据和配置原文。",
                    parent=self,
                ),
                exclusive=True,
            )

    def open_demo(self) -> None:
        """从按钮打开独立演示资料，不使用真实任务或真实模拟器。"""
        subprocess.Popen(
            command_for("--demo", "--home", str(self.service.home / "demo-preview")),
            env=child_environment(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def fix_display(self) -> None:
        if self.service.settings.demo:
            messagebox.showinfo(
                "演示模式",
                "演示不会修改模拟器。正式使用时，可在这里一键调整采集画面规格。",
                parent=self,
            )
            return
        if self.service.is_running() or self.running_action:
            messagebox.showinfo("正在采集", "请先停止当前任务再调整显示。", parent=self)
            return
        if not messagebox.askyesno(
            "调整所选模拟器的画面",
            "将所选MuMu设备调整为1920×1080、DPI 320。\n"
            "只修改该模拟器，游戏画面可能重排或需要重新打开。\n\n是否调整？",
            parent=self,
        ):
            return

        def work() -> None:
            from .discovery import list_devices, run_command, validate_address

            settings = self.service.settings
            if settings.mumu_manager:
                selected = next(
                    (
                        device
                        for device in list_devices(settings.mumu_manager)
                        if device.index == settings.instance_index
                    ),
                    None,
                )
                if selected and selected.address:
                    settings.device_addr = selected.address
            if not settings.device_addr:
                raise ValueError("请先打开所选模拟器，再点击调整画面")
            validate_address(settings.device_addr)
            run_command([settings.adb_path, "connect", settings.device_addr])
            for operation in (("size", "1920x1080"), ("density", "320")):
                result = run_command(
                    [
                        settings.adb_path,
                        "-s",
                        settings.device_addr,
                        "shell",
                        "wm",
                        *operation,
                    ]
                )
                if result.returncode:
                    raise RuntimeError("模拟器没有接受显示调整，请确认所选设备已启动")
            self.service.save()

        self.background(
            work,
            lambda _: messagebox.showinfo(
                "调整完成",
                "画面规格已调整，请确认游戏显示正常后继续采集。",
                parent=self,
            ),
            exclusive=True,
        )

    def background(
        self,
        work: Callable[[], Any],
        finished: Callable[[Any], None],
        *,
        exclusive: bool = False,
    ) -> None:
        if exclusive and self.running_action:
            messagebox.showinfo("正在处理", "当前操作还没有完成，请稍候。", parent=self)
            return
        if exclusive:
            self.running_action = True

        def complete(value: Any) -> None:
            if exclusive:
                self.running_action = False
            finished(value)

        def failed(error: Exception) -> None:
            if exclusive:
                self.running_action = False
            self.error(error)

        def execute() -> None:
            try:
                self.events.put((complete, work(), None))
            except Exception as error:
                self.events.put((failed, error, None))

        threading.Thread(target=execute, daemon=True).start()

    def auto_detect(self, silent: bool = False) -> None:
        if self.service.settings.demo:
            self.environment_text.set("演示模式无需连接设备")
            return
        self.environment_text.set("正在检查模拟器和运行库…")

        def finished(result: dict[str, Any]) -> None:
            settings = self.service.settings
            for key in ("mumu_manager", "adb_path", "maa_dir"):
                if result["paths"].get(key):
                    setattr(settings, key, result["paths"][key])
            self.devices = [Device(**row) for row in result["devices"]]
            self.device_combo["values"] = [device.label for device in self.devices]
            selected = next(
                (
                    device
                    for device in self.devices
                    if device.index == settings.instance_index
                ),
                None,
            )
            selected = (
                selected
                or next((device for device in self.devices if device.running), None)
                or (self.devices[0] if self.devices else None)
            )
            if selected:
                self.device_var.set(selected.label)
                settings.instance_index, settings.device_addr = (
                    selected.index,
                    selected.address,
                )
            self.environment_text.set(
                "   ·   ".join(row["detail"] for row in result["checks"])
            )
            self.sync_options()
            self.service.save()

        self.background(lambda: collect_diagnostics(self.service.settings), finished)

    def select_device(self, _: Any = None) -> None:
        if self.service.is_running():
            messagebox.showinfo(
                "正在采集", "请先停止当前任务再更换模拟器。", parent=self
            )
            return
        selected = next(
            (
                device
                for device in self.devices
                if device.label == self.device_var.get()
            ),
            None,
        )
        if selected:
            self.service.settings.instance_index = selected.index
            self.service.settings.device_addr = selected.address
            self.service.save()

    def sync_options(self) -> None:
        for name, variable in self.paths.items():
            variable.set(getattr(self.service.settings, name))
        self.allow_medicine.set(self.service.settings.allow_medicine)
        self.allow_originite.set(self.service.settings.allow_originite)
        self.profile_text.set(self.service.settings.nickname)

    def choose_folder(self, name: str) -> None:
        if self.service.is_running():
            messagebox.showinfo(
                "正在采集", "请先停止当前任务再修改文件位置。", parent=self
            )
            return
        if name == "mumu_manager":
            folder = filedialog.askdirectory(parent=self, title="选择MuMu安装文件夹")
            value = ""
            if folder:
                selected = next(
                    (
                        path
                        for path in (
                            Path(folder) / "MuMuManager.exe",
                            Path(folder) / "nx_main/MuMuManager.exe",
                            Path(folder) / "shell/MuMuManager.exe",
                        )
                        if path.is_file()
                    ),
                    None,
                )
                if selected is None:
                    messagebox.showinfo(
                        "未找到模拟器",
                        "这个文件夹中没有找到MuMu，请重新选择它的安装文件夹。",
                        parent=self,
                    )
                    return
                value = str(selected)
        else:
            value = filedialog.askdirectory(
                parent=self,
                title="选择数据保存位置"
                if name == "data_dir"
                else "选择采集运行库文件夹",
            )
        if value:
            try:
                if name == "data_dir":
                    self.service.set_data_directory(value)
                    self.refresh_tasks()
                else:
                    setattr(self.service.settings, name, value)
                    self.service.save()
            except ValueError as error:
                self.error(error)
                return
            self.sync_options()
            if name != "data_dir":
                self.auto_detect()

    def import_tasks(self) -> None:
        source = filedialog.askopenfilename(
            parent=self,
            title="选择维护者提供的采集任务",
            filetypes=[("采集任务", "*.zip *.json *.arktask"), ("所有文件", "*.*")],
        )
        if source:
            try:
                self.service.import_tasks(Path(source))
                self.refresh_tasks()
                for variable in (self.count_done, self.count_valid, self.count_review):
                    variable.set("0")
                self.detail_text.set("任务已导入，准备好账号和模拟器后即可开始。")
            except Exception as error:
                self.error(error)

    def refresh_tasks(self) -> None:
        try:
            summary = self.service.task_summary()
            self.task_text.set(
                f"{summary['title']}\n共 {summary['jobs']} 份作业"
                f" · {summary['stages']} 个关卡"
            )
            if not self.service.is_running():
                self.count_total.set(str(summary["jobs"]))
        except Exception:
            self.task_text.set("任务文件暂不可读取，请重新导入")

    def refresh_roster(self) -> None:
        settings = self.service.settings
        if settings.roster_file and Path(settings.roster_file).exists():
            document = read_json(Path(settings.roster_file), {}) or {}
            count = len(document.get("ops") or document.get("own_opers") or [])
            kind = (
                "含技能和模组"
                if document.get("source") == "skland" or document.get("ops")
                else "部分技能、模组信息需要补充"
            )
            self.roster_text.set(f"已读取 {count} 名干员 · {kind}")
        else:
            self.roster_text.set("尚未获取练度，可扫码、游戏内扫描或导入已有名册。")
        self.sync_options()

    def import_roster(self) -> None:
        if self.service.is_running():
            messagebox.showinfo(
                "正在采集", "请先停止当前任务再更换练度资料。", parent=self
            )
            return
        filename = filedialog.askopenfilename(
            parent=self, title="选择干员名册", filetypes=[("干员名册", "*.json")]
        )
        if not filename:
            return
        try:
            document = read_json(Path(filename))
            if not isinstance(document, dict) or not (
                document.get("ops") or document.get("own_opers")
            ):
                raise ValueError("没有找到可识别的干员资料")
            target = self.service.data_root() / "roster/imported_roster.json"
            write_json(target, document)
            player = document.get("player") or {}
            self.service.settings.bind_account(
                str(player.get("uid") or f"local-{uuid.uuid4().hex}")
            )
            self.service.settings.roster_file = str(target)
            self.service.save()
            self.refresh_roster()
        except Exception as error:
            self.error(error)

    def open_scan(self) -> None:
        if self.service.settings.demo:
            messagebox.showinfo(
                "演示模式",
                "演示不会进行真实账号登录。正式使用时，此按钮会显示森空岛扫码窗口。",
                parent=self,
            )
            return
        if self.service.is_running():
            messagebox.showinfo(
                "正在采集", "请先停止当前任务再切换账号或更新练度。", parent=self
            )
            return
        try:
            ScanWindow(self, self.service, self.refresh_roster)
        except ValueError as error:
            self.error(error)

    def operbox(self) -> None:
        if self.running_action:
            messagebox.showinfo("正在处理", "请等待当前读取操作完成。", parent=self)
            return
        if self.service.is_running():
            messagebox.showinfo("正在采集", "请先停止采集再扫描干员。", parent=self)
            return
        if self.service.settings.demo:
            messagebox.showinfo(
                "演示模式",
                "演示模式不会连接游戏。真实使用时可点击此按钮自动扫描干员。",
                parent=self,
            )
            return
        self.detail_text.set("正在扫描干员，请保持游戏已登录，不要同时操作模拟器。")
        self.show_page("采集面板")

        def work() -> str:
            settings = self.service.settings
            root = self.service.data_root()
            instance = instance_config(settings, root)
            out = root / "roster/operbox.json"
            spec = self.service.home / "operbox_request.local.json"
            write_json(
                spec,
                {
                    "instance": str(instance),
                    "out": str(out),
                    "user_dir": str(root / "debug/operbox"),
                },
            )
            log_path = self.service.home / "operbox.log"
            with log_path.open("wb") as log:
                result = subprocess.run(
                    command_for("--operbox", str(spec)),
                    env=child_environment(),
                    stdout=log,
                    stderr=log,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            if result.returncode:
                raise RuntimeError(
                    "干员扫描没有完成，请确认游戏已登录并处于主界面后重试。"
                )
            return str(out)

        def finished(path: str) -> None:
            self.service.settings.roster_file = path
            self.service.save()
            self.refresh_roster()
            self.detail_text.set("干员扫描已完成。技能和模组缺失项会保留为待补充。")

        self.background(work, finished, exclusive=True)

    def authorize(self, kind: str) -> None:
        if self.service.is_running():
            self.sync_options()
            messagebox.showinfo("正在采集", "请先停止采集再修改补给权限。", parent=self)
            return
        variable = self.allow_medicine if kind == "medicine" else self.allow_originite
        name = "理智药" if kind == "medicine" else "源石"
        if variable.get() and not messagebox.askyesno(
            "当前账号的补给授权",
            f"您是否为当前账号持有者，或已获得明确同意，允许采集时使用已有{name}补理智？\n\n仅作用于当前账号，不购买、不充值。",
            parent=self,
        ):
            variable.set(False)
        self.service.settings.allow_medicine = self.allow_medicine.get()
        self.service.settings.allow_originite = self.allow_originite.get()
        self.service.settings.resource_authorized_at = (
            now() if self.allow_medicine.get() or self.allow_originite.get() else ""
        )
        self.service.save()

    def save_options(self) -> None:
        if self.service.is_running():
            self.automatic.set(self.service.settings.automatic_recovery)
            self.convert.set(self.service.settings.convert_after)
            self.support.set(bool(self.service.settings.support_unit_usage))
            messagebox.showinfo("正在采集", "请先停止采集再修改本轮选项。", parent=self)
            return
        self.service.settings.automatic_recovery = self.automatic.get()
        self.service.settings.convert_after = self.convert.get()
        self.service.settings.support_unit_usage = int(self.support.get())
        self.service.save()

    def change_account(self) -> None:
        if self.service.is_running():
            messagebox.showinfo("正在采集", "请先停止采集再更换账号。", parent=self)
            return
        if not messagebox.askyesno(
            "更换采集账号",
            "更换后将清除当前练度关联和补给授权，原有采集数据保留。是否继续？",
            parent=self,
        ):
            return
        settings = self.service.settings
        settings.bind_account(f"local-{uuid.uuid4().hex}")
        settings.nickname, settings.roster_file, settings.last_run_dir = (
            "我的采集账号",
            "",
            "",
        )
        settings.allow_medicine = settings.allow_originite = False
        settings.resource_authorized_at = ""
        self.service.save()
        self.refresh_roster()

    def start(self, *, retry_failed: bool = False, restart_all: bool = False) -> None:
        if self.running_action:
            messagebox.showinfo(
                "正在处理", "请先等待当前操作完成，再开始采集。", parent=self
            )
            return
        if self.service.is_running():
            if self.last_status.get("status") == "paused":
                self.send("continue")
            return
        settings = self.service.settings
        if not settings.demo and not settings.accepted_notice:
            text = (
                "本工具为非官方采集工具，将通过截图和模拟点击操作所选模拟器，使用存在账号风险。\n\n"
                "请确认您有权使用当前账号采集，并已在模拟器内登录。采集期间请勿同时运行其他自动操作。\n\n是否已阅读并同意开始准备？"
            )
            if not messagebox.askyesno("首次采集说明", text, parent=self):
                return
            settings.accepted_notice = True
            settings.extras["notice"] = {"version": 1, "accepted_at": now()}
            self.service.save()
        if (
            not settings.demo
            and not settings.roster_file
            and not messagebox.askyesno(
                "尚未获取干员练度",
                "还没有获取干员练度。可以先到“账号与练度”扫码或扫描。\n\n是否继续采集，并将练度明确标记为缺失？",
                parent=self,
            )
        ):
            self.show_page("账号与练度")
            return
        try:
            if restart_all and not messagebox.askyesno(
                "重新采集全部作业",
                "已采过的作业也会重新执行，并消耗相应理智。原有数据保留。\n\n"
                "是否重新采集全部？",
                parent=self,
            ):
                return
            self.service.start(retry_failed=retry_failed, restart_all=restart_all)
            self.show_page("采集面板")
            self.detail_text.set("采集正在启动，关闭时可选择最小化继续。")
        except Exception as error:
            self.error(error)

    def send(self, command: str) -> None:
        if self.running_action:
            return
        try:
            self.service.send(command)
        except Exception as error:
            self.error(error)

    def pause_resume(self) -> None:
        self.send(
            "continue"
            if self.service.status().get("status") == "paused"
            else "pause_after"
        )

    def stop_now(self) -> None:
        if self.service.is_running() and messagebox.askyesno(
            "立即停止",
            "将中止当前局。已采数据保留，当前局可能不完整，需要重新采集。\n\n是否立即停止？",
            parent=self,
        ):
            self.send("stop_now")

    def open_data(self) -> None:
        try:
            folder = self.service.data_root()
        except ValueError as error:
            self.error(error)
            return
        folder.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(folder)

    def export(self) -> None:
        state = self.service.status()
        if self.service.is_running() and state.get("status") != "paused":
            messagebox.showinfo(
                "请先暂停采集",
                "请点击“当前局后暂停”，待当前局保存完成后再导出。",
                parent=self,
            )
            return
        if self.service.settings.demo:
            messagebox.showinfo(
                "演示模式",
                "演示不生成真实训练数据。正式采集后，可在这里导出交接包。",
                parent=self,
            )
            return
        filename = filedialog.asksaveasfilename(
            parent=self,
            title="保存采集包",
            defaultextension=".zip",
            initialfile="方舟采集数据.zip",
            filetypes=[("采集压缩包", "*.zip")],
        )
        if filename:
            self.detail_text.set("正在整理导出包，原始文件保留在本机…")
            images = self.include_images.get()
            redact_personal = not self.include_raw_records.get()
            self.background(
                lambda: export_data(
                    self.service.data_root(),
                    Path(filename),
                    include_images=images,
                    redact_personal=redact_personal,
                    diagnostics_home=self.service.home,
                ),
                lambda result: messagebox.showinfo(
                    "导出完成",
                    f"已导出 {result['files']} 个文件。\n"
                    "可将ZIP交给维护者；登录凭据未包含。",
                    parent=self,
                ),
            )

    def view_log(self) -> None:
        path = (
            Path(self.service.settings.last_run_dir) / "worker.log"
            if self.service.settings.last_run_dir
            else self.service.home / "app_errors.log"
        )
        window = tk.Toplevel(self)
        window.title("本轮采集日志")
        window.geometry("880x530")
        text = tk.Text(
            window, wrap="word", bg="#101c2f", fg="#d5e1f5", font=("Consolas", 10)
        )
        text.pack(fill="both", expand=True)
        text.insert(
            "1.0",
            path.read_text(encoding="utf-8", errors="replace")[-100_000:]
            if path.exists()
            else "暂无日志",
        )
        text.configure(state="disabled")
        ttk.Label(
            window, text="日志可能包含本机路径，请勿直接发布到公开平台。", padding=10
        ).pack(fill="x")

    def restart_admin(self) -> None:
        if self.service.settings.demo:
            messagebox.showinfo("演示模式", "演示模式不会请求管理员权限。", parent=self)
            return
        if self.service.is_running():
            messagebox.showinfo("正在采集", "请先停止采集再重新打开。", parent=self)
            return
        if os.name != "nt":
            return
        self.service.save()
        arguments = command_for("--home", str(self.service.home))
        if self.service.settings.demo:
            arguments += ["--demo"]
        shell = ctypes.WinDLL("shell32", use_last_error=True)
        shell.ShellExecuteW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_int,
        ]
        shell.ShellExecuteW.restype = ctypes.c_void_p
        result = shell.ShellExecuteW(
            None, "runas", arguments[0], subprocess.list2cmdline(arguments[1:]), None, 1
        )
        if result and result > 32:
            self.destroy()
        else:
            messagebox.showinfo(
                "保持当前窗口", "管理员请求未完成，当前窗口仍可继续使用。", parent=self
            )

    def close_app(self) -> None:
        if self.running_action:
            messagebox.showinfo(
                "正在处理", "正在读取干员或导出数据，请等待完成后退出。", parent=self
            )
            return
        if self.service.is_running():
            choice = messagebox.askyesnocancel(
                "采集正在进行",
                "是否最小化窗口并继续采集？\n\n选择“是”：最小化继续。\n选择“否”：当前局结束后停止。",
                parent=self,
            )
            if choice is True:
                self.iconify()
            elif choice is False:
                self.send("stop_after")
                self.exit_after_stop = True
                self.detail_text.set("当前局完成后停止；保存完成后即可退出。")
            return
        self.service.save()
        self.destroy()

    def poll(self) -> None:
        while True:
            try:
                callback, value, error = self.events.get_nowait()
            except queue.Empty:
                break
            if error:
                self.error(error)
            elif callback:
                try:
                    callback(value)
                except Exception as callback_error:
                    self.error(callback_error)
        try:
            state = self.service.status()
            self.last_status = state
            labels = {
                "idle": "待开始",
                "starting": "准备中",
                "running": "采集中",
                "paused": "已暂停",
                "waiting_device": "等待设备",
                "recovering": "正在恢复",
                "converting": "正在转换",
                "finished": "已完成",
                "stopped": "已停止",
                "error": "需要处理",
                "interrupted": "可恢复任务",
                "needs_user": "等待您处理",
                "storage_error": "保存失败，已停止",
                "conversion_error": "转换可重试",
                "unresponsive": "任务无响应",
            }
            self.status_text.set(labels.get(state["status"], "准备中"))
            self.state_badge.configure(
                bg="#dcfce7" if state["status"] == "running" else "#e8eef8",
                fg=GREEN if state["status"] == "running" else BLUE,
            )
            self.recover_button.configure(
                state="normal"
                if state["status"]
                in ("needs_user", "storage_error", "interrupted", "error", "stopped")
                and not self.running_action
                else "disabled"
            )
            if state.get("message"):
                self.detail_text.set(state["message"])
            latest = state.get("last_result") or {}
            if (
                latest
                and not latest.get("valid_collected_win")
                and not latest.get("demo_passed")
                and state.get("status") not in ("error", "interrupted")
            ):
                reason = latest.get("fail_stage")
                if reason == "formation":
                    self.detail_text.set(
                        "上一份作业编队未完成，可能缺少干员或助战。已记账，后续作业继续。"
                    )
                elif latest.get("prts_detected"):
                    self.detail_text.set(
                        "上一局发现代理相关标记，数据已保留待复核；后续作业继续。"
                    )
                elif reason == "process_exit":
                    self.detail_text.set(
                        "上一份采集进程意外结束，原始数据保留，后续作业继续。"
                    )
                elif latest.get("battle_result") == "unknown":
                    self.detail_text.set(
                        "上一局结果未能确认，数据已保留待复核；后续作业继续。"
                    )
            for key, variable in (
                ("total", self.count_total),
                ("attempted", self.count_done),
                ("valid", self.count_valid),
                ("review", self.count_review),
            ):
                if key in state:
                    variable.set(str(state[key]))
            self.current_stage.set(state.get("current_stage") or "—")
            self.progress.configure(
                maximum=max(1, int(state.get("total", self.count_total.get()) or 1)),
                value=int(state.get("attempted", 0)),
            )
            self.pause_button.configure(
                text="继续采集" if state["status"] == "paused" else "当前局后暂停"
            )
            active = self.service.is_running() or self.running_action
            self.start_button.configure(
                state="normal"
                if not active or state["status"] == "paused"
                else "disabled"
            )
            self.pause_button.configure(
                state="normal" if self.service.is_running() else "disabled"
            )
        except (OSError, ValueError) as error:
            self.detail_text.set("进度暂时不可读，正在重新检查。")
        if self.exit_after_stop and not self.service.is_running():
            self.service.save()
            self.destroy()
            return
        self.after(500, self.poll)
