"""战场状态落盘模块。

职责（对应设计文档 §5.4）：在作战过程中按 tick 实时采集并落盘战场状态，
包括费用、击杀数、剩余生命、待部署干员卡（费用/冷却/充能）、技力、
格子占用情况及截图校验哈希。数据来源是 MaaCore 的识别回调结果，
输出为"原始状态流"（落盘 JSON 行），供 core 仓库的 state_builder
经 IPC / 落盘文件消费，转换为带 protocol_version 的标准状态。

红线：截图本体不出本机，落盘只存哈希（见设计文档 §6.3）。
"""

# TODO(M0): 挂接 MaaCore 识别回调，定义逐 tick 状态采集与落盘循环。
# TODO(M0): 定义原始状态流的落盘格式（JSONL）与文件轮转策略。


class StateLogger:
    """战场状态落盘器（占位 stub，M0 实现）。"""

    def start(self, stage_code: str) -> None:
        """开始一局的逐 tick 状态落盘。"""
        raise NotImplementedError
