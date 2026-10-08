"""ark-client：Arknights Autoplay 玩家端客户端顶级包。

职责：
    - 面向玩家本机的可安装程序：屏幕抓取（capture）→ 本地 ONNX CPU 推理（runtime）
      → 子弹时间时序调度（scheduler）→ MuMu ADB 模拟输入（input）；
    - 首次启动风险告知（risk_notice）与最小界面（ui）。
    - 仅支持 MuMu 模拟器（Windows 10/11 x64）。

铁律：本包不得 import 任何 bridge 仓库（ark-autoplay-maa-bridge，AGPL-3.0）的代码；
感知状态与决策权重一律通过版本化协议与发布 artifact 消费。

TODO(M0): 立项骨架阶段仅提供包结构，无实现逻辑。
"""
