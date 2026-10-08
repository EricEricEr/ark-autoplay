# NOTICE —— 上游关系声明

本目录（`bridge/`）与 **[MaaAssistantArknights](https://github.com/MaaAssistantArknights/MaaAssistantArknights)** 的关系，**取决于是否已导入上游源码**，两种状态必须分清。

## 当前状态：非衍生作品（2026-10-09）

- 上游项目：MaaAssistantArknights / MaaCore
- 上游仓库：https://github.com/MaaAssistantArknights/MaaAssistantArknights
- 上游许可证：**AGPL-3.0**

**本目录当前不包含任何上游源码**（`maacore/`、`patches/` 均为空）：

- 运行期经 `Asst.load()` **动态加载官方 MAA 发行版**、调用其公开 `asst` Python 接口；
- **不复制、不修改**上游源码，与上游之间无代码级链接。

因此**本目录当前不是** MAA 的衍生作品（derivative work）——它是独立实现、调用上游公开 API 的独立程序。本文件早期版本曾自称"衍生作品"，属**过度自认**，与仓库实际内容不符，已于 2026-10-09 更正。

## 未来状态：导入源码后即为衍生作品

一旦向 `maacore/` 导入 MaaCore 源码（目的例如：导出 `BattlefieldMatcher` 已识别、但未通过 `asst` 接口暴露的战场状态量），本目录**即成为上游的衍生作品（derivative work）**，此时：

1. 须将上游 AGPL-3.0 的**附加条款（Additional Terms）原文逐字粘贴于本文件**，并随 `patches/` 中的本地改动一并维护；
2. 本目录整体保持 **AGPL-3.0**（与当前一致，无需变更）；
3. 上游提交哈希须可追溯（`maacore/` 保留完整 git 历史，改动以 `patches/NNNN-*.patch` 维护）。

上述流程详见 [`maacore/README.md`](maacore/README.md)。

> **待核实**：上游附加条款的具体约束内容**至今未核实**（立项文档 §17 待核实清单第 5 条）。在导入源码之前必须完成，因为它可能影响发布方式。另需注意：MAA 除 AGPL 许可证外还有一份独立的**《用户协议》**（见其官网），约束的是"使用行为"而非"代码分发"，与本目录的许可证义务是两回事，发行产品前需单独评估。

## 致谢

感谢 MaaAssistantArknights 社区、prts.wiki、prts.plus 与 ArknightsGameData 维护者的工作。本项目与鹰角网络（Hypergryph）无任何关系，为非官方研究性质项目，不分发任何游戏素材。
