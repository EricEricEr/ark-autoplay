# ark-autoplay-client

> 让任何一台能运行 MuMu 模拟器的普通 PC，都能通过 ONNX CPU 本地推理驱动 AI 自主作战。
> The player-side client of Arknights Autoplay: local ONNX CPU inference that plays Arknights on any ordinary PC running the MuMu emulator.

Arknights Autoplay 是一个完全开源、社区共建、非官方、非商业的项目（完全开源、代码/权重/配方/评测全部公开），目标是训练一个不依赖逐关人工脚本、能自主通关明日方舟任意关卡的 AI。本仓库是它的**玩家端客户端**。

Arknights Autoplay is a fully open-source, community-driven, unofficial, non-commercial project aiming to train an AI that clears any Arknights stage without per-stage hand-written scripts. This repository is its **player-side client**.

## 本仓库在架构中的位置 / Position in the Architecture

```
┌────────────┐  截图   ┌────────────────┐ 结构化状态JSON ┌───────────────┐ 离散动作 ┌───────────────┐
│ 游戏客户端   │ ─────► │ 感知层(bridge)  │ ────────────► │ 决策层(core)   │ ───────► │ 执行层          │
│ MuMu模拟器  │ ◄───── │ MAA识别         │  协议/artifact │ 小网络/VLM     │  协议     │ 本仓库: scheduler│
└────────────┘ ADB输入 └────────────────┘               └───────────────┘          │ +capture/input  │
                                                                                  └───────────────┘
```

- **中文**：本仓库承接 core 仓库发布的 ONNX 决策模型与协议，在本机完成「MuMu 屏幕抓取 → ONNX CPU 推理 → 模拟输入」的闭环，是链路末端的执行侧与玩家侧。
- **English**: Consuming the ONNX decision model and versioned protocol published by `ark-autoplay-core`, this client closes the on-device loop of "MuMu screen capture → ONNX CPU inference → simulated input". It is the execution and player end of the pipeline.

## 支持范围 / Supported Environment

- **仅支持 MuMu 模拟器**（基线：MuMu 12，Windows 10/11 x64，分辨率 1920×1080，DPI 320）。不支持官方 PC 客户端、其他模拟器与手机真机。
  MuMu emulator only (baseline: MuMu 12, Windows 10/11 x64, 1920×1080, DPI 320). The official PC client, other emulators and real phones are NOT supported.
- **ONNX CPU 推理**：无需独立显卡，普通 PC（含核显/无独显机器）即可游玩使用。
  ONNX Runtime CPU inference — no discrete GPU required; any ordinary PC that can run MuMu can use it.

## 当前状态 / Current Status

- **中文**：立项骨架（roadmap M0）。目录结构、协议约定、工程质量体系已就位，无实现逻辑。路线图见各仓库设计文档，M3（第 5-6 个月）交付可安装客户端 + 公测。
- **English**: Project skeleton at milestone M0 — directory layout, protocol conventions and the quality toolchain are in place, with no implementation logic yet. A distributable client plus public beta is targeted at M3.

## 兄弟仓库 / Sibling Repositories

| 仓库 | 角色 | 许可证 |
|---|---|---|
| [ark-autoplay-core](https://github.com/EricEricEr/ark-autoplay-core) | 协议、感知后处理、决策模型、训练、评测、导出 | Apache-2.0 |
| [ark-autoplay-maa-bridge](https://github.com/EricEricEr/ark-autoplay-maa-bridge) | MAA fork、状态落盘、重放控制器（数据工厂） | AGPL-3.0 |
| [ark-autoplay-client](https://github.com/EricEricEr/ark-autoplay-client) | 玩家端：录屏 + 本地推理 + 模拟输入（本仓库） | Apache-2.0 |

本仓库与 bridge 之间**无代码级依赖**：感知状态与决策权重一律通过版本化协议与发布 artifact 消费。

## 许可证 / License

Apache-2.0，见 [LICENSE](LICENSE)。

## 素材红线声明 / Game Assets Policy

- **中文**：本项目与鹰角网络（Hypergryph）无任何关系，属研究性质的非官方项目。游戏截图、立绘、地图素材、解包数据版权归鹰角网络所有，**一律不进仓库、不随安装包分发**。仓库只发布代码与自产的结构化数据。若权利方对任何内容提出异议，维护者承诺在收到通知后尽快配合处理（下架、整改或删除），联系渠道为本仓库 Issue。
- **English**: This project is an unofficial, research-oriented work with no affiliation to Hypergryph. All game screenshots, character art, stage maps and unpacked game data are copyrighted by Hypergryph and are NEVER committed to this repository or bundled with any installer. Only code and self-produced structured data are published. If any rights holder objects to any content, maintainers will cooperate promptly upon notice (takedown, remediation or removal). Contact: GitHub Issues.

## 风险告知 / Risk Notice

- **中文**：使用本工具存在账号风险，后果由使用者自负。客户端首次启动会展示**不可跳过**的风险告知，确认记录落盘保存（见 `src/client/risk_notice.py`）。
- **English**: Use this tool at your own account risk. A non-skippable risk notice is shown on first launch, and the acknowledgement is recorded on disk (see `src/client/risk_notice.py`).

## 致谢 / Acknowledgements

MAA、MaaFramework、ArknightsGameData、prts.wiki、prts.plus 社区作业作者们。
