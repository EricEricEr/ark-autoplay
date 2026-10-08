# Arknights Autoplay — Core（项目大脑）

> 明日方舟自主作战 AI · 核心仓库：协议 / 感知 / 决策模型 / 训练 / 评测

## 项目定位｜What is this?

**中文**：本项目尝试训练一个不依赖逐关人工脚本、能自主通关明日方舟任意关卡的 AI——看到一张没见过的图，自己读地形、读波次、选干员、决定落子与开技能时机。本仓库是项目的**大脑**：定义状态 / 动作 / 轨迹协议（semver 版本化），实现感知后处理与双轨决策模型（10-50M 实体 Transformer 主线 + Qwen3-VL-4B 像素对照），并负责训练、三档零样本评测与 ONNX 导出。数据工厂（MAA 重放 prts.plus 作业产通关轨迹）与玩家端执行分别由 bridge / client 两个仓库承担。

**English**: An open attempt to train an AI that clears arbitrary Arknights stages without per-stage human scripts. This repository is the project's **brain**: it owns the versioned state/action/episode protocol, perception post-processing, the dual-track decision models (a 10-50M entity Transformer vs a Qwen3-VL-4B pixel baseline), training, zero-shot evaluation (split by stage family), and ONNX export. Data production (replaying prts.plus community clears) and the player-side executor live in the sibling `bridge` / `client` repositories.

## 架构中的位置｜Position in the architecture

```text
游戏客户端(MuMu 模拟器) ──截图──► bridge(数据工厂) ──版本化 JSON 协议──► 【本仓 core/】 ──离散动作──► client(玩家端)
```

- core 消费 bridge 产出的**版本化 JSON 协议**（历史沿革：三仓时代该协议同时承担许可证隔离职责；全项目统一 AGPL-3.0 后隔离已非必需，协议保留是因为它对跨进程解耦与可复现性依然有价值——见 [ADR-0004](docs/adr/0004-unify-agpl-monorepo.md)）。
- 决策输出为离散语义动作（`deploy / skill / retreat / wait`）+ 动作掩码，由 client 经 MuMu ADB 模拟输入执行。

## 当前状态｜Status

**立项骨架（Roadmap M0 基建期）**：仓库结构、协议 stub、CI 门禁与 agent 协作文件已就位；决策 / 训练 / 评测 / 感知层代码均为带 TODO 标注的占位 stub，尚无真实实现。已落地的真实实现集中在数据管线（见下方 quickstart）。里程碑见设计文档 §14（M0 基建 → M1 双轨 → M2 裁决 → M3 玩家端 → M4 后置）。

## 许可证｜License

本项目**整体采用 AGPL-3.0**（见根目录与各子目录的 `LICENSE`，四处文本完全一致）。

选择 AGPL 而非宽松许可证是**项目方的主动决定**：允许任何人自由使用、修改、分发，但**禁止将本项目（或其衍生作品）做成闭源商用产品**——只要分发或提供网络服务，就必须以 AGPL 公开全部源码。决策记录见 [ADR-0004](docs/adr/0004-unify-agpl-monorepo.md)（该 ADR 取代了早期三仓许可证隔离方案 [ADR-0001](docs/adr/0001-repo-split-license-isolation.md)）。

## 素材红线声明｜Asset red line

- 本项目与鹰角网络（Hypergryph）**无任何关系**，为研究性质的**非官方**社区项目。
- 本仓库**不分发**任何游戏截图、立绘、地图素材或解包数据（版权归鹰角网络）；`data/` 目录只放 schema 说明与合成样例，真实轨迹与截图**永不入库**（详见 [data/README.md](data/README.md)）。
- 只做"截图 + 模拟输入"，不读内存、不修改游戏客户端。
- **权利方异议响应｜Takedown commitment**：若鹰角网络或相关权利方对本仓库任何内容提出异议，维护者承诺在收到通知后尽快配合处理（下架、整改或删除），联系渠道为本仓库 Issue。If Hypergryph or any rights holder objects to any content here, maintainers will cooperate promptly upon notice (takedown, remediation or removal). Contact: GitHub Issues.

## 数据管线（datapipe v1）｜Data pipeline quickstart

把本机静态数据源（prts.plus 作业镜像 + 官方数值表，**均不入库**）转换为 featvec / 关卡注册表 / **关卡静态特征** / episode 轨迹 / 练度分桶产物（决策记录见 [docs/adr/0002](docs/adr/0002-datapipe-v1.md)、[docs/adr/0003](docs/adr/0003-stagefeat-v1.md)）：

```bash
uv sync
# 组织数据根（布局见 configs/data.yaml 头部注释），然后指定数据根并构建全部产物：
export ARK_DATA_ROOT=/path/to/ark_data_root
uv run python -m ark_core.datapipe.build all
# 也可分步：featvec | stages | stagefeat | episodes | tiers；--config / --out 可覆盖配置
```

- 源路径全部走 `configs/data.yaml` + `ARK_DATA_ROOT`（代码内零机器路径）；缺源时报错会列出缺失清单与修复指引。
- Deploy 非干员名三类 subtype（device / category / unknown_name）的词表在 `configs/deploy_vocab.yaml`；地形 / 路线编码表在 `configs/tile_vocab.yaml`（数据即配置）。
- 产物默认输出到 `<数据根>/processed/`；真实数据产物永不入库（见下方红线）。

### 关卡静态特征（stagefeat）

`stages.jsonl` 的嵌套 JSON（格子 token / 路线 checkpoint / 出怪三元组）转成模型可直接张量化的整型列：`grid`（每格 5 通道：高度 / 可部署类型 / 是否可部署 / 起点 / 终点）、`routes`（路线几何 + 等待时长）、`spawns`（出怪时刻表）。

**坐标序注意**：`grid` / `routes` 用 `[row, col]`，而 MAA 作业 deploy 的 `location` 是 `[x, y] = [col, row]`（两者均已用真实数据交叉验证，见 [ADR-0003](docs/adr/0003-stagefeat-v1.md) §3）。消费 episodes 中 deploy 位置的代码**必须**先过 `stagefeat.normalize_deploy_location()`。

## 相关目录｜Sibling directories

本仓为**合并仓 `ark-autoplay` 的 `core/` 子目录**（2026-10-09 由三独立仓库合并，见 [ADR-0004](docs/adr/0004-unify-agpl-monorepo.md)）：

| 目录 | 角色 | 许可证 |
|---|---|---|
| `core/`（本目录） | 大脑：协议 / 感知 / 模型 / 训练 / 评测 | AGPL-3.0 |
| `bridge/` | 数据工厂：MAA 驱动、状态落盘、作业重放 | AGPL-3.0 |
| `client/` | 玩家端：录屏 + 本地推理 + 模拟输入 | AGPL-3.0 |

## 贡献｜Contributing

贡献前请先读 [AGENTS.md](AGENTS.md)（环境与命令、代码风格、PR 约定、安全红线）。所有 PR 按 ReviewBench 九类缺陷 rubric 打标评审（模板已内置清单）。
