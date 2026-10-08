"""划分泄漏检查占位测试。

TODO(M1): evaluation/splits.py 的三档关卡划分产出后，实现真实泄漏检查——
检测到同一关卡家族（章节 / 活动家族）同时出现在训练集与测试集即判失败，
CI 相应转红（设计文档 §12 最高准则）。
"""


def test_leak_placeholder() -> None:
    """占位用例：恒真，仅为 tests/leak/ 目录与 CI 入口占位。"""
    assert True
