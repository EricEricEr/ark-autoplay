# Arknights Autoplay

> 训练一个**不依赖逐关人工脚本、能自主通关明日方舟任意关卡**的 AI。完全开源、非官方、非商业。
> An open attempt to train an AI that clears arbitrary Arknights stages without per-stage human scripts.

**许可证：AGPL-3.0**（见 [LICENSE](LICENSE)）— 任何人可自由使用、修改、分发，但**不得做成闭源商用产品**；只要分发或提供网络服务，就必须以 AGPL 公开全部源码。

---

## 这是什么

市面上的明日方舟自动化（以 [MAA](https://github.com/MaaAssistantArknights/MaaAssistantArknights) 为代表）是"识别 + 逐关人工配置脚本"：每张图都要有人写好作业才能自动打。本项目要回答一个不同的问题：

**能不能让 AI 像玩家一样，看到一张没见过的图，自己读地形、读波次、选干员、决定落子和开技能时机，最终通关？**

- **决策层**：双轨实验对打——10-50M 实体 Transformer 小网络（结构化状态输入）vs Qwen3-VL-4B 像素版 QLoRA。
- **数据来源**：MAA 重放 prts.plus 社区作业当数据工厂；自监督预训练 + 行为克隆 + DAgger。
- **裁决标准**：按关卡家族（章节/活动）划分留出集的**零样本通关率**，严禁按对局随机划分。
- **发行形态**：小网络导出 ONNX、CPU 实时推理，让能跑 MuMu 模拟器的普通 PC（含核显机）本地可用。

## 目录结构

本仓由三个独立仓库于 2026-10-09 合并而成（决策记录见 [ADR-0004](core/docs/adr/0004-unify-agpl-monorepo.md)），**各仓完整 git 历史均保留**。

| 目录 | 角色 | 说明 |
|---|---|---|
| [`core/`](core/) | **大脑** | 协议（状态/动作/轨迹，semver 版本化）、感知后处理、双轨决策模型、训练、评测、ONNX 导出 |
| [`bridge/`](bridge/) | **数据工厂** | 补丁 MaaCore 驱动、作业重放、状态/动作/截图落盘；含实验性模型闭环运行器 |
| [`client/`](client/) | **玩家端** | 屏幕抓取 + 本地 ONNX CPU 推理 + ADB 模拟输入 |

```text
MuMu模拟器 ──截图──► bridge(数据工厂) ──版本化协议──► core(决策) ──离散动作──► client(玩家端)
```

## 当前进度

项目处于 **M0 基建期**（2026-10-26 更新）。已实现真机状态采集、行为克隆训练、关卡级 K 折评测和实验性闭环；仍处于小数据链路验证阶段。完整口径见仓内 [项目状态](docs/STATUS.md) 和 [离线评测记录](core/docs/reports/2026-10-24-offline-v1-1.md)。

| 部分 | 状态 |
|---|---|
| 数据管线（featvec / 关卡注册表 / **关卡静态特征** / 作业转换 / 练度分桶） | ✅ 已实现并验证 |
| 重放链路（MAA 驱动 + 导航 + 轨迹落盘） | ✅ 已实现；依赖配套补丁运行时，长批次仍有导航/回主页失败 |
| 战场状态采集 | ✅ 动作前采集费用、击杀、手牌、已部署信息；无敌人实时位置/完整技力 |
| 模型 / 训练 / 离线评测 | ✅ v0/v1、SSL 对照、关卡级 K 折；当前状态监督为 176 局 / 1,161 对 |
| 实验性闭环 | ✅ 0-1 / 0-2 / 0-3 两轮共 6 次记录为通关；仅作回路验证 |
| 格子/朝向、通用自主通关 | ⚠️ 格子命中率低；尚无未见章节自主通关证据 |
| 玩家端 / ONNX 发行 | ⬜ 尚未形成可交付的完整玩家产品 |

## 快速开始

**协助者可按分工选择 [指南 A：仅采集](bridge/docs/collect-only.md) 或 [指南 B：采集并本地转换](bridge/docs/collect-and-convert.md)**，总览见 [采集者指南入口](bridge/docs/collecting.md)。当前结构化策略训练不要求 MP4，B 可直接整理出不含图片的状态—动作包。原始画面在采集者本地保留以便复核。

只采集需要 Windows 10/11 x64、MuMu（1920×1080 / DPI 320）、Python 3.12 + uv，以及 [配套补丁 MAA 运行时](bridge/maacore/README.md)。本地转换也无需训练用 torch/GPU 或模型权重。

```powershell
git clone https://github.com/EricEricEr/ark-autoplay.git
cd ark-autoplay/bridge
uv sync --frozen
uv run python -m bridge.replay_controller --help
```

`--help` 不连接游戏。正式采集前按指南填写本机实例配置，并取得与账号匹配的作业队列。普通官方 MAA 不导出本项目使用的 `BattleState`；克隆仓库不会自动获得补丁二进制或游戏数据。

练度采集优先用 [森空岛一键扫码](bridge/tools/skland/README.md)（现有 Node.js 18+、无 npm 依赖）：一次扫码后自动获取全干员、全部技能/专精和模组记录。只有无法扫码的账号才使用 OperBox，并补充其缺失字段；好友助战的练度单独记录。

各子目录独立管理依赖；开发检查为 `uv run pytest -q`、`uv run ruff check`，core/client 另运行 `uv run pyright`。core 训练依赖通过 `uv sync --group train` 安装。数据管线见 [core README](core/README.md)。

## 素材红线

- 本项目与鹰角网络（Hypergryph）**无任何关系**，为研究性质的**非官方**项目。
- **不分发任何游戏素材**：截图、立绘、地图贴图、解包数据一律不入库；轨迹只存截图哈希，由使用者本机重采重建。
- 只做"截图 + 模拟输入"，**不读内存、不修改游戏客户端**。
- **权利方异议响应**：若鹰角网络或相关权利方对任何内容提出异议，维护者承诺在收到通知后尽快配合处理（下架、整改或删除），联系渠道为 Issue。

## 贡献

各子目录的 `AGENTS.md` 载有该目录的环境命令与代码规范。所有 PR 按 **ReviewBench 九类缺陷 × 三级严重度** rubric 打标评审（模板已内置）。
