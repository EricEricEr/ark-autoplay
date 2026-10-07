"""MuMu 多开实例池模块。

职责（对应设计文档 §5.4 / §16）：管理 2-3 个并发 MuMu 模拟器实例，
供数据工厂并行重放。实例定义来自 configs/instances/（端口 / 分辨率 / DPI），
通过 MuMu 多开器与 ADB 直连管理生命周期（启动 / 就绪等待 / 释放 / 异常重启）。

基线约束：仅支持 MuMu 模拟器，分辨率固定 1920x1080、DPI 320；
启动时校验实例参数，不符基线则拒绝分配并报错（设计文档 §9）。
"""

# TODO(M0): 实现实例定义加载（configs/instances/*.yaml）与基线校验。
# TODO(M0): 实现实例获取 / 释放的池化接口与 ADB 连接管理。
# TODO(M1): 实现实例异常检测与自动重启、并发上限保护。

BASELINE_WIDTH: int = 1920
BASELINE_HEIGHT: int = 1080
BASELINE_DPI: int = 320
"""识别基线：固定分辨率与 DPI，模板与格子坐标均按此标定。"""


class InstancePool:
    """MuMu 多开实例池（占位 stub，M0 实现）。"""

    def acquire(self) -> str:
        """获取一个就绪实例的 ADB 连接地址（如 127.0.0.1:16384）。"""
        raise NotImplementedError

    def release(self, adb_addr: str) -> None:
        """释放实例回池。"""
        raise NotImplementedError
