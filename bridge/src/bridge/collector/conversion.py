"""可独立重试的本地转换，不重新进入战斗，不覆盖未完成输出。"""

from __future__ import annotations

import subprocess
import time
import uuid

from .resilience import writable_probe
from .storage import write_json


def run_conversion(data, run_dir, update, stopping, group, runtime):
    """每次写入新目录，失败保留原件与中间输出，按钮可独立重试。"""
    from .service import child_environment

    output = data / "exports" / f"converted-{uuid.uuid4().hex[:16]}"
    request = run_dir / "conversion.json"
    write_json(request, {"data_root": str(data), "output": str(output)})
    update(status="converting", message="正在本地转换，不会重新开打关卡")
    child = None
    try:
        with (run_dir / "conversion.log").open("ab") as log:
            child = runtime.subprocess.Popen(
                runtime.command_for("--convert", str(request)),
                cwd=data,
                env=child_environment(),
                stdout=log,
                stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            group.assign(child)
            while child.poll() is None:
                if stopping():
                    runtime.stop_owned_process(child)
                    update(
                        status="stopped", message="转换已中止，可单独重新转换；原件保留"
                    )
                    return
                writable_probe(data)
                time.sleep(0.5)
        if child.returncode:
            update(
                status="conversion_error",
                message="转换未完成，原件保留。请点击‘重新转换已有数据’。",
            )
            return
        write_json(output / "conversion_complete.json", {"complete": True})
        update(
            status="finished",
            converted_dir=str(output),
            message="转换已完成，可导出数据",
        )
    finally:
        if child is not None and child.poll() is None:
            runtime.stop_owned_process(child)
