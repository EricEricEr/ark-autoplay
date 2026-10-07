# Arknights Autoplay — Core（项目大脑）

> 明日方舟自主作战 AI · 核心仓库：协议 / 感知 / 决策模型 / 训练 / 评测

## 项目定位｜What is this?

**中文**：本项目尝试训练一个不依赖逐关人工脚本、能自主通关明日方舟任意关卡的 AI——看到一张没见过的图，自己读地形、读波次、选干员、决定落子与开技能时机。本仓库是项目的**大脑**：定义状态 / 动作 / 轨迹协议（semver 版本化），实现感知后处理与双轨决策模型（10-50M 实体 Transformer 主线 + Qwen3-VL-4B 像素对照），并负责训练、三档零样本评测与 ONNX 导出。数据工厂（MAA 重放 prts.plus 作业产通关轨迹）与玩家端执行分别由 bridge / client 两个仓库承担。

**English**: An open attempt to train an AI that clears arbitrary Arknights stages without per-stage human scripts. This repository is the project's **brain**: it owns the versioned state/action/episode protocol, perception post-processing, the dual-track decision models (a 10-50M entity Transformer vs a Qwen3-VL-4B pixel baseline), training, zero-shot evaluation (split by stage family), and ONNX export. Data production (replaying prts.plus community clears) and the player-side executor live in the sibling `bridge` / `client` repositories.

## 架构中的位置｜Position in the architecture

```text
游戏客户端(MuMu 模拟器) ──截图──► bridge(AGPL, 数据工厂) ──IPC / 落盘 JSON──► 【本仓库 core】 ──离散动作──► client(玩家端)
```

- core 只消费 bridge 产出的**版本化 JSON 协议**，两者之间无代码级链接；`license-guard` CI 强制检查本仓库不得 import 任何 bridge/maa 相关模块。
- 决策输出为离散语义动作（`deploy / skill / retreat / wait`）+ 动作掩码，由 client 经 MuMu ADB 模拟输入执行。

## 当前状态｜Status

**立项骨架（Roadmap M0 基建期）**：仓库结构、协议 stub、CI 门禁与 agent 协作文件已就位；所有代码均为带 TODO 标注的占位 stub，尚无真实实现。里程碑见设计文档 §14（M0 基建 → M1 双轨 → M2 裁决 → M3 玩家端 → M4 后置）。

## 许可证｜License

代码采用 **Apache-2.0**（见 [LICENSE](LICENSE)）。三仓库按许可证隔离：bridge 因衍生自 MAA/MaaFramework 强制 AGPL-3.0，core 与 client 均为 Apache-2.0（决策记录见 [docs/adr/0001](docs/adr/0001-repo-split-license-isolation.md)）。

## 素材红线声明｜Asset red line

- 本项目与鹰角网络（Hypergryph）**无任何关系**，为研究性质的**非官方**社区项目。
- 本仓库**不分发**任何游戏截图、立绘、地图素材或解包数据（版权归鹰角网络）；`data/` 目录只放 schema 说明与合成样例，真实轨迹与截图**永不入库**（详见 [data/README.md](data/README.md)）。
- 只做"截图 + 模拟输入"，不读内存、不修改游戏客户端。

## 相关仓库｜Sibling repositories

| 仓库 | 角色 | 许可证 |
|---|---|---|
| [ark-autoplay-core](https://github.com/EricEricEr/ark-autoplay-core)（本仓库） | 大脑：协议 / 感知 / 模型 / 训练 / 评测 | Apache-2.0 |
| [ark-autoplay-maa-bridge](https://github.com/EricEricEr/ark-autoplay-maa-bridge) | 数据工厂：MAA fork、状态落盘、作业重放 | AGPL-3.0 |
| [ark-autoplay-client](https://github.com/EricEricEr/ark-autoplay-client) | 玩家端：录屏 + 本地推理 + 模拟输入 | Apache-2.0 |

## 贡献｜Contributing

贡献前请先读 [AGENTS.md](AGENTS.md)（环境与命令、代码风格、PR 约定、安全红线）。所有 PR 按 ReviewBench 九类缺陷 rubric 打标评审（模板已内置清单）。
