"""离散语义动作（Action）的 schema 定义。

动作空间：deploy(干员,格子,朝向) / skill / retreat / wait；
非法动作（费用不足 / 格子占用 / 冷却中）由动作掩码在模型输出侧屏蔽。
完整说明见 docs/rfc/0001-state-action-protocol.md（RFC-0001）。
"""

from typing import Literal

# TODO(M0): 定义 pydantic Action 模型（type / op / r / c / dir / op_uid 字段）
#           与动作掩码接口（返回每类动作的合法布尔向量）。

ActionType = Literal["deploy", "skill", "retreat", "wait"]
"""动作类型枚举（占位 stub）。"""
