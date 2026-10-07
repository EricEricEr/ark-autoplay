"""通关轨迹写出模块。

职责（对应设计文档 §5.4 / §6.3）：将一局的逐 tick 原始状态与动作
整理为协议化轨迹（episode）写出，每行 / 每局携带：

- protocol_version：协议版本戳（与 core 仓库 protocol 库对齐，semver）；
- squad_hash：阵容哈希；
- source：作业来源署名（prts.plus 作业 id、作者、许可说明）；
- ticks 内截图只存 sha256 校验哈希，不存截图本体（素材红线，§3.2/§6.3）。

写出格式为 Parquet（一行一局）或 JSONL 原始档，供数据集发布前校验。
"""

# TODO(M0): 定义 episode 写出结构（字段对齐设计文档 §6.3）。
# TODO(M0): 写入时强制戳 protocol_version，并对写出结果做自检（哈希/必填字段）。
# TODO(M1): 补 Parquet 落盘与旧版本协议迁移兼容读取。

PROTOCOL_VERSION: str = "0.3"
"""最近一次对齐的协议版本（与 ark-autoplay-core 的 protocol 库保持一致）。"""


class EpisodeWriter:
    """轨迹写出器（占位 stub，M0 实现）。"""

    def write(self, episode: dict) -> str:
        """写出一局轨迹并返回落盘路径。

        写入前必须戳 ``PROTOCOL_VERSION`` 并校验必填字段与截图哈希齐全。
        """
        raise NotImplementedError
