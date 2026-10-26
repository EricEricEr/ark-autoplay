"""maa_driver 窄路径收敛单测（2026-10-21 乱码目录事故修复）。

真实截图不入库同规：这里只建临时目录（含中文与全角问号），
验证传给 MaaCore 窄字符 API 的路径形态，字节级断言。

另含 wait_chain 终态优先级单测（2026-10-24 SingleStep 快链竞态修复）：
``TaskChainCompleted`` 与 ``AllTasksCompleted`` 同批到达时，后者不得
覆盖链自身终态——否则一条成功链会被误报为未收尾（loop_runner 路径 B
的心跳链因此被打成 channel_dead 的实证）。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from bridge.maa_driver import MaaDriver, MaaEvent, resolve_narrow_path


def _bare_driver() -> MaaDriver:
    """不 connect 的裸 driver：只够驱动 wait_chain 的事件游标逻辑。"""
    drv = MaaDriver.__new__(MaaDriver)
    drv._events = []
    drv._lock = threading.Lock()
    return drv


def test_wait_chain_all_tasks_does_not_override_completed() -> None:
    """快链同批到达：AllTasksCompleted 不覆盖 TaskChainCompleted。"""
    drv = _bare_driver()
    drv._events.extend(
        [
            MaaEvent(0, 0, "TaskChainStart", {"taskid": 1}),
            MaaEvent(0, 0, "TaskChainCompleted", {"taskid": 1}),
            MaaEvent(0, 0, "AllTasksCompleted", {}),
        ]
    )
    res = drv.wait_chain(timeout_s=2.0, mark=0)
    assert res.finished_ok
    assert res.terminal_msg == "TaskChainCompleted"


def test_wait_chain_error_terminal_is_not_ok() -> None:
    drv = _bare_driver()
    drv._events.extend(
        [
            MaaEvent(0, 0, "TaskChainError", {"taskid": 1}),
            MaaEvent(0, 0, "AllTasksCompleted", {}),
        ]
    )
    res = drv.wait_chain(timeout_s=2.0, mark=0)
    assert not res.finished_ok
    assert res.terminal_msg == "TaskChainError"


def test_wait_chain_timeout_returns_none_terminal() -> None:
    drv = _bare_driver()
    res = drv.wait_chain(timeout_s=0.3, mark=0)
    assert not res.finished_ok
    assert res.terminal_msg is None


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


@pytest.mark.skipif(os.name != "nt", reason="Windows路径兼容")
def test_unicode_temp_uses_ascii_cache_without_deleting_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """中文临时目录不返回乱码路径，兼容别名可复用且不覆盖已有目录。"""
    import hashlib
    import tempfile

    target = tmp_path / "中文数据？"
    target.mkdir()
    temporary = tmp_path / "中文临时"
    temporary.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temporary))
    monkeypatch.setattr("bridge.maa_driver._win32_short_name", lambda _: None)
    cache = Path(os.environ["PROGRAMDATA"]) / "synthetic-path-test"
    if not str(cache).isascii():
        pytest.skip("本次测试没有可写的ASCII兼容缓存")
    monkeypatch.setenv("PROGRAMDATA", str(cache))
    alias = resolve_narrow_path(target)
    assert alias.isascii() and os.path.samefile(alias, target)
    probe = target / "probe.txt"
    probe.write_text("synthetic", encoding="utf-8")
    assert (Path(alias) / probe.name).read_text(encoding="utf-8") == "synthetic"
    assert resolve_narrow_path(target) == alias

    occupied_target = tmp_path / "另一目录"
    occupied_target.mkdir()
    digest = hashlib.sha1(str(occupied_target.resolve()).encode("utf-8")).hexdigest()[
        :10
    ]
    occupied = Path(alias).with_name(f"arkmaa_{digest}")
    occupied.mkdir()
    marker = occupied / "keep.txt"
    marker.write_text("untouched", encoding="utf-8")
    with pytest.raises(RuntimeError, match="拒绝覆盖"):
        resolve_narrow_path(occupied_target)
    assert marker.read_text(encoding="utf-8") == "untouched"
