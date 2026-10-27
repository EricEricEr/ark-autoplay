"""Windows打包入口；开发者构建，普通玩家只运行生成的exe。"""

import contextlib
import json
import sys
from pathlib import Path

from bridge.collector.__main__ import main


def entry() -> int:
    """工作进程发生启动错误时写本地诊断并退出，不能弹窗卡住整批任务。"""
    try:
        return main()
    except Exception as error:
        modes = (
            "--run",
            "--manual",
            "--episode",
            "--operbox",
            "--convert",
            "--self-test",
            "--ui-test",
        )
        mode = next((value for value in modes if value in sys.argv), None)
        if mode:
            with contextlib.suppress(OSError, ValueError, IndexError):
                request = Path(sys.argv[sys.argv.index(mode) + 1])
                request.with_suffix(".fatal.json").write_text(
                    json.dumps(
                        {"error_type": type(error).__name__, "error": str(error)},
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
        else:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "采集助手暂时无法打开",
                "本地数据目录无法访问或设置未能读取。\n请检查权限，或尝试以管理员权限重新打开。",
                parent=root,
            )
            root.destroy()
        return 1


if __name__ == "__main__":
    raise SystemExit(entry())
