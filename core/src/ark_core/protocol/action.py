"""离散语义动作（Action）的 schema 定义与训练用编码表。

动作空间（方案 §6.2）：``deploy(干员,格子,朝向) / skill / retreat / wait``。
非法动作（费用不足 / 格子占用 / 冷却中）由动作掩码在模型输出侧屏蔽。
完整说明见 docs/rfc/0001-state-action-protocol.md（RFC-0001）。

编码表为何放这里
----------------
``ACTION_TYPES`` / ``DIRECTIONS`` 是**协议层概念**（模型、训练、数据管线都要用），
放在 protocol 层可避免 models → training 的反向依赖（分层：protocol ← models ← training）。
"""

from typing import Literal

# TODO(M0): 定义 pydantic Action 模型（type / op / r / c / dir / op_uid 字段）
#           与动作掩码接口（返回每类动作的合法布尔向量）。

ActionType = Literal["deploy", "skill", "retreat", "wait"]
"""协议层动作类型（RFC-0001 定义的四类语义动作）。

注意与 :data:`ACTION_TYPES` 的区别：后者是**训练用**的编码表，额外包含数据集里
出现的 ``speed`` / ``bullet_time``（它们是 MAA 作业的执行控制动作，语义上不改变
战场状态，但确实出现在轨迹里，必须能表达）。
"""

ACTION_TYPES: tuple[str, ...] = ("deploy", "skill", "retreat", "speed", "bullet_time")
"""训练用动作类型编码表（固定顺序，模型输出维度即其长度）。

顺序一旦发布**不得变更**（会让已训练的 checkpoint 权重错位）；新增类型只能追加。
"""

DIRECTIONS: tuple[str, ...] = ("Right", "Down", "Left", "Up", "None")
"""部署朝向编码表（``None`` 固定在末位 = 无朝向语义的动作/缺省）。

顺序同样不得变更，理由同上。
"""

DIR_ALIASES: dict[str, str] = {
    "Right": "Right", "right": "Right", "右": "Right",
    "Down": "Down", "down": "Down", "下": "Down",
    "Left": "Left", "left": "Left", "左": "Left",
    "Up": "Up", "up": "Up", "上": "Up",
    "None": "None", "无": "None", "": "None",
}
"""朝向写法归一表。实测作业里中英文混用（Left 44229 / 左 1211 等），
不归一会把同一方向拆成两个类别。"""

GRID_CHANNELS = 5
"""关卡格子通道数（与 datapipe.stagefeat 的 grid 编码严格一致）：
高度 / 可部署类型 / 是否可部署 / 起点 / 终点。"""
