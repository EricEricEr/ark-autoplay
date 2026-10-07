"""立项骨架阶段测试占位：仅校验包结构可导入，保证 CI 全绿。

实现阶段由真实用例替换（见设计文档 §13 测试策略：
纯逻辑单测为主，感知与执行用 golden 文件 + 回放测试，不依赖真机）。
"""

import importlib

import pytest

_MODULES = [
    "client",
    "client.capture",
    "client.runtime",
    "client.input",
    "client.ui",
    "client.scheduler",
    "client.risk_notice",
]


@pytest.mark.parametrize("module", _MODULES)
def test_module_importable(module: str) -> None:
    """骨架冒烟测试：每个包/模块均可导入。"""
    importlib.import_module(module)
