# ADR-0001：三仓库划分与许可证隔离

- 状态：已接受
- 日期：2026-10-07（立项决策）
- 关联：设计文档 v2.1 §3.1；本仓库 `license-guard` 工作流

## 背景

项目的自动化底座复用 MAA / MaaFramework，其许可证为 **AGPL-3.0 + 附加条款**。AGPL 具有强传染性：衍生作品必须以同一许可证开源，且"通过网络提供服务"同样触发开源义务。若自有代码与 MAA 衍生代码同仓或发生代码级链接，整个项目将被传染为 AGPL，违背"核心代码 Apache-2.0、尽量宽松"的开源策略。

## 决策

拆分为**三个独立仓库**，按衍生关系确定许可证：

| 仓库 | 内容 | 许可证 |
|---|---|---|
| [ark-autoplay-core](https://github.com/EricEricEr/ark-autoplay-core) | 协议、感知后处理、决策模型、训练、评测、导出（全部自有代码） | Apache-2.0 |
| [ark-autoplay-maa-bridge](https://github.com/EricEricEr/ark-autoplay-maa-bridge) | MAA/MaaCore fork、状态落盘、重放控制器（衍生自上游） | AGPL-3.0（强制，不可选） |
| [ark-autoplay-client](https://github.com/EricEricEr/ark-autoplay-client) | 玩家端：录屏 + 本地推理 + 模拟输入 | Apache-2.0 |

隔离机制：

1. bridge 与 core / client 之间**只通过 IPC（本地 socket / 文件落盘）+ 版本化 JSON 协议**通信，不做任何代码级链接；
2. bridge 的任何代码不得被 core / client 引用（`license-guard` CI 扫描违禁 import，命中即红）；
3. 协议版本独立于各仓库代码演进（见 RFC-0001），跨仓库升级时先升协议、再升实现。

## 后果

- 正向：core / client 保持 Apache-2.0 宽松许可，可自由商用 / 二次开发；AGPL 传染范围被锁死在 bridge 一仓。
- 代价：跨仓库功能需经协议演进，联调成本上升；协议兼容测试成为 CI 刚需。
- 风险：若未来 MaaFramework 附加条款对发布方式有新约束，需重审本 ADR（见设计文档 §17 待核实清单第 5 条）。
