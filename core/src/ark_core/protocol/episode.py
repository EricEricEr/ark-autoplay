"""通关轨迹（Episode）的 schema 定义。

Parquet 一行一局：episode_id / protocol_version / stage / squad_hash /
source（含 prts.plus 作业署名）/ result / ticks（逐 tick 状态 + 动作 + 截图哈希）。
红线：只存截图哈希、不存截图本体（见 docs/rfc/0001 与 data/README.md）。
"""

# TODO(M0): 定义 pydantic Episode / Tick 模型与 Parquet 读写封装（占位 stub）。
