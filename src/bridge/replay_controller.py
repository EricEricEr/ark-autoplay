"""作业重放调度模块。

职责（对应设计文档 §5.4 / §10 数据工厂）：全自动重放 prts.plus 社区作业，
流水线为 领作业 → 编队（可含阵容随机化变体）→ 进图 → 重放执行
（联动 state_logger 逐 tick 落盘）→ 判胜负 → 存盘。只保留通关局轨迹；
掉帧 / 识别漂移的局剔除并记录。支持断点续跑（中断后从未完成作业继续）。

输入为 configs/jobs/ 下的重放任务定义（作业 id、阵容、变体数），
输出交给 episode_writer 写为协议化轨迹。
"""

# TODO(M0): 实现作业领取与任务队列（configs/jobs/ 驱动）。
# TODO(M0): 实现重放主循环与胜负判定、失败局剔除策略。
# TODO(M0): 实现断点续跑：任务进度持久化与恢复。


class ReplayController:
    """作业重放调度器（占位 stub，M0 实现）。"""

    def run(self, job_path: str) -> None:
        """执行一份重放任务定义，直至全部作业完成或中断。"""
        raise NotImplementedError
