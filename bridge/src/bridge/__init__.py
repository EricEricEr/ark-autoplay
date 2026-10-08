"""ark-bridge：Arknights Autoplay 数据工厂与战场状态供给包。

本包是 MaaAssistantArknights（MaaCore）的衍生隔离仓，仅负责：

- 战场状态落盘（state_logger）
- prts.plus 作业自动重放（replay_controller）
- 通关轨迹写出（episode_writer）
- 编队与阵容随机化（squad_builder）
- MuMu 多开实例池管理（instance_pool）

铁律：本包代码绝不被 ark-autoplay-core / ark-autoplay-client import；
对外只通过 IPC（本地 socket / 落盘 JSON）+ 版本化协议通信。
"""

__version__ = "0.0.1"
