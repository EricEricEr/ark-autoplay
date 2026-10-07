"""冒烟测试：骨架可导入、关键占位结构存在。

后续真实测试（落盘完整性 / 断点续跑 / 协议兼容）在 M0 补。
"""

from bridge import __version__


def test_version_placeholder() -> None:
    """包可导入且带版本号。"""
    assert __version__ == "0.0.1"
