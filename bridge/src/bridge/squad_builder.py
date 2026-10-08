"""编队与阵容随机化模块。

职责（对应设计文档 §5.4 / §10）：按 prts.plus 作业要求的阵容自动完成
游戏内编队；并生成阵容随机化变体（同关换队，作为数据增强，防止模型
"认关背答案"）。变体生成受 configs/jobs/ 中每关任务的变体数约束，
生成结果交给 replay_controller 逐变体重放。

干员以社区通用代号引用（与 prts.wiki 对齐），不涉及任何游戏素材本体。
"""

# TODO(M0): 实现按作业阵容自动编队（联动 MaaCore 编队界面操作）。
# TODO(M1): 实现阵容随机化变体生成（同定位干员替换 + 阵容哈希去重）。


class SquadBuilder:
    """编队构造器（占位 stub，M0 实现）。"""

    def build(self, squad: list[str]) -> None:
        """按给定干员代号列表完成游戏内编队。"""
        raise NotImplementedError

    def variants(self, squad: list[str], n: int) -> list[list[str]]:
        """生成 n 个阵容随机化变体（占位）。"""
        raise NotImplementedError
