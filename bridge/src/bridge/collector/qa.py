"""发行包自动验收，强制使用独立合成任务，不触碰模拟器或真实账号。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .storage import write_json


def attach(app: Any, proof: Path) -> None:
    """用实际按钮验证开始、暂停、继续；测试证据写入调用者指定目录。"""
    started = time.monotonic()
    proof.parent.mkdir(parents=True, exist_ok=True)
    stage = "initial"
    errors: list[str] = []
    app.report_callback_exception = lambda kind, error, tb: errors.append(
        f"{kind.__name__}: {error}"
    )

    def fail(error: Exception) -> None:
        write_json(
            proof,
            {
                "passed": False,
                "phase": stage,
                "error": str(error),
                "errors": errors,
                "state": app.service.status(),
            },
        )
        app.service.send("stop_now")
        app.destroy()

    def capture(name: str) -> None:
        from PIL import ImageGrab

        app.update_idletasks()
        ImageGrab.grab(window=app.winfo_id()).save(proof.parent / name)

    def tick() -> None:
        nonlocal stage
        try:
            if errors:
                raise RuntimeError(errors[-1])
            if time.monotonic() - started > 100:
                raise TimeoutError("打包界面测试超时，未影响任何真实任务")
            current = app.service.status()
            shown = app.last_status.get("status")
            if stage == "initial":
                capture("collector-dashboard.png")
                app.start_button.invoke()
                stage = "started"
            elif stage == "started" and current.get("status") == shown == "running":
                app.pause_button.invoke()
                stage = "pausing"
            elif stage == "pausing" and current.get("status") == shown == "paused":
                if current["attempted"] < 1:
                    raise AssertionError("暂停前未保存当前局")
                capture("collector-paused.png")
                app.pause_button.invoke()
                stage = "resumed"
            elif stage == "resumed" and current.get("status") == shown == "finished":
                if (current["attempted"], current["valid"], current["review"]) != (
                    6,
                    4,
                    2,
                ):
                    raise AssertionError(current)
                capture("collector-finished.png")
                app.show_page("账号与练度")
                capture("collector-account.png")
                if (
                    app.service.settings.allow_originite
                    or app.service.settings.allow_medicine
                ):
                    raise AssertionError("默认补给授权被打开")
                write_json(
                    proof,
                    {
                        "passed": True,
                        "demo_only": True,
                        "state": current,
                        "checks": [
                            "packaged window",
                            "start button",
                            "pause after episode",
                            "resume button",
                            "all tasks retained",
                            "failed task isolation",
                            "resource defaults off",
                        ],
                        "errors": errors,
                    },
                )
                if app.service.process:
                    app.service.process.wait(timeout=5)
                app.destroy()
                return
            app.after(150, tick)
        except Exception as error:
            fail(error)

    app.after(700, tick)
