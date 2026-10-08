"""state_builder：bridge 原始输出 → 标准状态 JSON（protocol.State）。

输入为 bridge 经 IPC / 落盘产出的原始状态流（含解包数值），
输出为协议化的 State，附协议版本戳；识别置信度逐字段透传。
"""

# TODO(M0): 定义原始状态流的读取接口与 State 构建函数，
#           含置信度透传与协议版本戳写入（占位 stub）。
