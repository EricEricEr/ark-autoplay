# NOTICE —— 上游关系声明

本目录（`bridge/`）与 **[MaaAssistantArknights](https://github.com/MaaAssistantArknights/MaaAssistantArknights)** 的关系。

- 上游项目：MaaAssistantArknights / MaaCore
- 上游仓库：https://github.com/MaaAssistantArknights/MaaAssistantArknights
- 上游分支 / pinned commit：`dev-v2` / `a82a0e13c68d48cc21ec40380ac08c95f46e9802`（见 [`maacore/UPSTREAM.json`](maacore/UPSTREAM.json)）
- 上游许可证：**AGPL-3.0-only**

## 当前状态（2026-10-09）：**是衍生作品**

本目录现已包含上游源码的使用与修改：

- `maacore/_src/` 下为 MaaCore 源码（稀疏克隆，**不入库**，按 pinned commit 重建）；
- [`patches/0001-battle-state-callback.patch`](patches/0001-battle-state-callback.patch)
  是对上游源码的本地修改（新增战场状态回调，4 文件 +90 行 0 删除）。

因此**本目录是 MaaCore 的衍生作品（derivative work）**，依 AGPL-3.0 整体以
**AGPL-3.0** 授权——与本仓既有许可一致，**无需变更**。

> 说明：本文件早期版本（在三仓时代）曾自称"衍生作品"，而当时 `maacore/`、`patches/`
> 均为空、并未包含任何上游源码，属**过度自认**；随后一度改为"当前非衍生"。现随源码与
> 补丁的实际导入，该声明**正式生效**。

## 许可证核实结论（**修正项目原有认知**）

立项文档 §17 与旧版本文件一直写「MAA 为 **AGPL-3.0 + 附加条款（Additional Terms）**」，
并把"附加条款原文待粘贴"列为待办。**2026-10-09 从上游源码逐字核实：该"附加条款"不存在**。

| 核实项 | 结果 |
|---|---|
| 上游是否有 `LICENSE.md` | ❌ 无（只有 `LICENSE` 与 `LICENSE.spdx`） |
| `LICENSE` 与标准 AGPL-3.0 的差异 | **仅 -661 字节**，尾部即标准 AGPL 收尾 |
| 是否含真正的附加条款条文 | ❌ 无（`Additional Terms` 的命中是 **AGPL 第 7 条正文的通用措辞**） |
| GitHub API 报告 | `AGPL-3.0` |

**故无须粘贴任何附加条款**；立项文档 §17 待核实清单第 5 条**可关闭**。

## 真正有约束力的文件：上游《用户协议》

`terms-of-service.md`（**独立于 AGPL**，管**使用行为**而非代码分发）。与本项目相关：

| 条款 | 内容 | 影响 |
|---|---|---|
| §2.1 | 可在遵守 AGPL 前提下使用、修改、分发 | ✅ fork 合法 |
| §2.2 | **MAA Logo 不适用 AGPL**，未经许可不得商用 | ⚠️ 不得使用 MAA Logo |
| §2.3 | 未经许可**严禁宣称与 MAA Team 存在合作关系** | ⚠️ 只可写"使用/调用 MAA" |
| §2.4 | 最终用户不修改 MAA 时，使用与分发不受限 | ✅ 玩家端不受限 |
| §3.1 | 禁止在**森空岛**与**鹰角官方账号交互区**讨论/传播 MAA | ⚠️ 宣传须避开 |
| §3.4 | 禁止用于工作室批量刷号等非法活动 | ⚠️ 产品风险告知需提示 |

## 重建方式

源码树不入库（1.2 GB+）。按 [`maacore/UPSTREAM.json`](maacore/UPSTREAM.json) 记录的
pinned commit 与步骤精确重建，再 `git apply patches/*.patch`。

## 致谢

感谢 MaaAssistantArknights 社区、prts.wiki、prts.plus 与 ArknightsGameData 维护者的工作。本项目与鹰角网络（Hypergryph）无任何关系，为非官方研究性质项目，不分发任何游戏素材。
