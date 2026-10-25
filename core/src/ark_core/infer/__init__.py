"""在线推理胶水层：把已训练的行为克隆 checkpoint 接到真机 battle_state 数据流上。

「模型闭环打关」里程碑第一环；当前只有 track_a_v1（v1 冒烟模型 → 合法作业动作，
**不是自主作战能力证明**）。

依赖训练组（torch + numpy，``uv sync --group train``）：CI / 数据管线侧不要 import
本包，其单测已按 tests/unit/test_imitation.py 的 importorskip 模式在缺 torch 时
整文件 skip。
"""

from .track_a_v1 import (
    InferConfig,
    LoadedModel,
    StageContext,
    cell_to_xy,
    decode_action,
    encode_action_for_history,
    greedy_step,
    load_track_a_v1,
    xy_to_cell,
)

__all__ = [
    "InferConfig",
    "LoadedModel",
    "StageContext",
    "cell_to_xy",
    "decode_action",
    "encode_action_for_history",
    "greedy_step",
    "load_track_a_v1",
    "xy_to_cell",
]
