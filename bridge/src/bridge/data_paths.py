"""项目数据与导出路径校验；系统设置、登录态和临时缓存不使用此限制。"""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath


def data_path(value: str | Path, *, label: str = "数据保存位置") -> Path:
    """要求明确指定非C盘位置，同时核对原始盘符和联接解析后的实际盘符。"""
    if not isinstance(value, (str, Path)):
        raise ValueError(f"请先选择非C盘的{label}")
    raw = os.path.expandvars(os.path.expanduser(str(value).strip()))
    if not raw:
        raise ValueError(f"请先选择非C盘的{label}")
    result = Path(raw).resolve()
    for candidate in (raw, str(result)):
        drive = PureWindowsPath(candidate).drive.casefold()
        if drive.removeprefix("\\\\?\\").removeprefix("\\\\.\\") == "c:":
            raise ValueError(
                f"{label}不能位于C盘，请选择其他磁盘；系统默认缓存不受此限制"
            )
    return result
