"""maa_driver 窄路径收敛单测（2026-10-21 乱码目录事故修复）。

真实截图不入库同规：这里只建临时目录（含中文与全角问号），
验证传给 MaaCore 窄字符 API 的路径形态，字节级断言。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bridge.maa_driver import resolve_narrow_path


def test_ascii_path_passthrough(tmp_path: Path) -> None:
    """纯 ASCII 路径原样返回（规范化后），不做任何改写。"""
    d = tmp_path / "maa_user_ascii"
    d.mkdir()
    out = resolve_narrow_path(d)
    assert out.isascii()
    assert os.path.samefile(out, d)


def test_cjk_fullwidth_path_shortened(tmp_path: Path) -> None:
    """含中文与全角问号的路径：返回纯 ASCII 形态（8.3 或 Junction 别名），
    且与原路径指向同一物理目录。

    本机已禁用 8.3（实测），走 Junction 分支；若临时目录联接创建被拒
    （环境策略），按 skip 标注机器能力，不掩盖语义。
    """
    d = tmp_path / "工作区？全角"
    d.mkdir(parents=True)
    try:
        out = resolve_narrow_path(d)
    except RuntimeError:
        pytest.skip("本机 8.3 短名与 Junction 别名均不可用（非 Windows 或策略拒绝）")
        return  # pragma: no cover
    # 字节级正确性：全是 ASCII，DLL 按 ANSI 解释也不会变形
    assert out.isascii(), f"窄路径应纯 ASCII，实得 {out!r}"
    assert Path(out).exists()
    # 与原路径指向同一物理目录（文件互见是硬证据）
    probe = d / "probe.txt"
    probe.write_text("x", encoding="utf-8")
    assert (Path(out) / "probe.txt").is_file()
    assert os.path.samefile(out, d)
    # 别名可复用：同一目标两次收敛得同一路径，不重复建联接
    assert resolve_narrow_path(d) == out


def test_nonexistent_path_raises(tmp_path: Path) -> None:
    """不存在且含非 ASCII 的路径：拿不到短名，明确报错拒绝静默写乱码。"""
    with pytest.raises(RuntimeError):
        resolve_narrow_path(tmp_path / "不存在的目录？")
