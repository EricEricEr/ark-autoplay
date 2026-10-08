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
| [`bridge/`](bridge/) | **数据工厂** | MAA 驱动、作业重放、轨迹落盘（当前经公开 `asst` API 调用官方 MAA 发行版） |
| [`client/`](client/) | **玩家端** | 屏幕抓取 + 本地 ONNX CPU 推理 + ADB 模拟输入 |

```text
MuMu模拟器 ──截图──► bridge(数据工厂) ──版本化协议──► core(决策) ──离散动作──► client(玩家端)
```

## 当前进度

项目处于 **M0 基建期**。**完整状态、待办清单与环境恢复步骤见 [STATUS.md](../STATUS.md)**（与本仓同级的交接文档）。

一句话概括现状：**数据管线和重放链路已通，模型与训练全部是占位 stub；最大卡点是重放时未记录战场数值状态（费用/击杀/技力），导致行为克隆缺少输入。**

| 部分 | 状态 |
|---|---|
| 数据管线（featvec / 关卡注册表 / **关卡静态特征** / 作业转换 / 练度分桶） | ✅ 已实现并验证 |
| 重放链路（MAA 驱动 + 导航 + 轨迹落盘） | ✅ 真机跑通（5 条轨迹） |
| 战场状态感知（费用/击杀/技力） | ❌ **卡点** |
| 决策模型 / 训练 / 评测 / 导出 | ⬜ 全为占位 stub |

## 快速开始

```bash
# 各子目录独立管理依赖（三套 venv）
cd core   && uv sync && uv run pytest -q    # 48 passed
cd bridge && uv sync && uv run pytest -q    # 15 passed
cd client && uv sync && uv run pytest -q    # 7 passed

# 数据管线（需先准备数据根，布局见 core/configs/data.yaml 头部注释）
export ARK_DATA_ROOT=/path/to/ark_data_root
cd core && uv run python -m ark_core.datapipe.build all
```

完整环境要求（MuMu 12 / MAA 发行版 / Python 3.12 + uv）与数据资产清单见 [STATUS.md](../STATUS.md) §1–§2。

## 素材红线

- 本项目与鹰角网络（Hypergryph）**无任何关系**，为研究性质的**非官方**项目。
- **不分发任何游戏素材**：截图、立绘、地图贴图、解包数据一律不入库；轨迹只存截图哈希，由使用者本机重采重建。
- 只做"截图 + 模拟输入"，**不读内存、不修改游戏客户端**。
- **权利方异议响应**：若鹰角网络或相关权利方对任何内容提出异议，维护者承诺在收到通知后尽快配合处理（下架、整改或删除），联系渠道为 Issue。

## 贡献

各子目录的 `AGENTS.md` 载有该目录的环境命令与代码规范。所有 PR 按 **ReviewBench 九类缺陷 × 三级严重度** rubric 打标评审（模板已内置）。
