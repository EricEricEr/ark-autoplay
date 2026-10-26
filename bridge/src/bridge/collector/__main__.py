"""采集软件统一入口；普通玩家运行窗口，后台工作模式由程序自行调用。"""

from __future__ import annotations

import argparse
import contextlib
import multiprocessing
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """按模式启动界面或隔离执行器，不在导入时连接游戏。"""
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="方舟数据采集助手")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--home", type=Path)
    parser.add_argument("--operbox", type=Path)
    parser.add_argument("--convert", type=Path)
    parser.add_argument("--self-test", type=Path)
    parser.add_argument("--ui-test", type=Path)
    args = parser.parse_args(argv)
    if args.self_test:
        import ctypes.util
        import json
        import tkinter

        from .discovery import find_installations
        from .roster_ui import scan_script
        from .storage import Settings, write_json
        from .worker import nav_file

        root = Path(__file__).parent
        resources = [
            root / "resources/runtime_compatibility.json",
            root.parent / "resources/battle_lifecycle.json",
            root.parent / "resources/post_battle/resource/tasks/ark_post_battle.json",
            nav_file(),
            scan_script(),
        ]
        checks = {path.name: path.is_file() for path in resources}
        for path in resources:
            if path.suffix == ".json" and path.exists():
                json.loads(path.read_text(encoding="utf-8"))
        checks["node_runtime"] = bool(find_installations(Settings())["node_path"])
        checks["tk_runtime"] = bool(tkinter.Tcl().eval("info patchlevel"))
        checks["native_library_loader"] = callable(ctypes.util.find_library)
        write_json(args.self_test, {"passed": all(checks.values()), "checks": checks})
        return 0 if all(checks.values()) else 2
    if args.convert:
        from bridge.convert_episodes import convert

        from .storage import read_json

        specification = read_json(args.convert)
        convert(Path(specification["data_root"]), Path(specification["output"]))
        return 0
    if args.run or args.episode:
        from .worker import episode, run_session

        return episode(args.episode) if args.episode else run_session(args.run)
    if args.operbox:
        from bridge.collect_operbox import main as collect_operbox

        from .storage import read_json

        spec = read_json(args.operbox)
        return collect_operbox(
            [
                "--instance",
                spec["instance"],
                "--out",
                spec["out"],
                "--user-dir",
                spec["user_dir"],
            ]
        )
    from .app import CollectorApp

    if args.ui_test and not args.home:
        parser.error("界面验收必须使用独立资料目录")
    application = CollectorApp(args.home, demo=args.demo or bool(args.ui_test))
    if args.ui_test:
        from .qa import attach

        attach(application, args.ui_test)
    application.mainloop()
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError):
            stream.reconfigure(errors="replace")
    raise SystemExit(main())
