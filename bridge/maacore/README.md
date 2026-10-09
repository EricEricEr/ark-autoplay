# maacore/ —— MaaCore fork 与状态导出

本目录存放 **MaaAssistantArknights / MaaCore 的 fork 锚点**。

**源码树不入库**（1.2 GB+）：由 [`UPSTREAM.json`](UPSTREAM.json) 记录的 pinned commit
精确重建，仓库只保存「上游 commit pin + `../patches/` 中的本地改动」。
重建步骤见 [`UPSTREAM.json`](UPSTREAM.json) 同目录的说明（或 `UPSTREAM.json` 顶部注释与
`../NOTICE.md`）。这样既满足 AGPL 的源码可得性（公开上游按 commit 可取回），
又避免把上游整棵树塞进本仓。

## 当前状态（2026-10-09 实测）

| 项 | 状态 |
|---|---|
| 稀疏克隆 | ✅ 6.5 MB（仅 C++ 所需路径） |
| MaaUtils submodule | ✅ 已拉（提供 maadeps 框架 + meojson） |
| 预编译依赖 | ✅ MaaDeps v2.14.1 x64-windows（313 MB） |
| 编译 | ✅ **exit 0**，产出 `MaaCore.dll` 5.97 MB |
| 本地补丁 | ✅ `0001-battle-state-callback.patch`（4 文件 +90 行 0 删除） |

## fork 原则

1. **不做无谓的源码 fork**：上游提交哈希可追溯；**本地改动一律以 `patches/` 维护**，
   按序编号（`0001-xxx.patch`…），每个 patch 配一份 `.md` 说明文档。
   patch 必须能对 pristine 源码 `git apply --check` 通过（已实测）。
2. **改动最小化**：优先"只读取上游已有成员并序列化"，**不修改上游的识别/决策逻辑**。
   patch 0001 是范例：MaaCore 内部本就在维护费用/击杀/部署状态，我们只是把它导出来。
3. **许可证**：上游为 **AGPL-3.0**（**已核实无附加条款**，见 `UPSTREAM.json`）；
   本目录作为衍生作品同样以 AGPL-3.0 授权。仓库整体已是 AGPL-3.0，无需变更。
   ⚠️ **不要使用 MAA Logo，也不要宣称与 MAA Team 有合作关系**（用户协议 §2.2 / §2.3）。
4. **调用边界**：`maacore/` 只被本仓（`src/bridge/`）调用；跨进程边界仍走版本化协议
   （这是为解耦与可复现性，**不再**是许可证隔离要求——全项目已统一 AGPL-3.0，见
   `../core/docs/adr/0004-unify-agpl-monorepo.md`）。

## 更新上游的流程

```bash
cd bridge/maacore/_src
git fetch origin dev-v2
git rebase origin/dev-v2          # 或 checkout 新 commit
git apply ../../patches/*.patch   # 逐一 reapply，冲突按 patch 说明解决
# 然后重新配置与编译（见 UPSTREAM.json 的 build 小节）
```

## 历史待核实项（**已结案**）

> 原文（立项文档 §17 / 旧版 README）：
> 「MaaCore 现有回调是否足以支撑状态落盘，还是必须 fork 改造——第一个开发周给结论」

**结论（2026-10-09）**：回调**不足**，必须改造。证据：

- MaaCore 导出的 C API 仅 31 个 `Asst*` 函数，**无 `GetCost`/`GetKills`**；
- 回调消息类型仅 13 种（任务链生命周期），**不含战场状态字段**；
- 但 `BattleHelper` 内部**已维护** `m_cost` / `m_kills` / `m_total_kills` /
  `m_cur_deployment_opers` / `m_battlefield_opers`——**识别早就做了，只是不外吐**。

因此改造方式不是"新写识别"，而是"把已有值导出"——即 patch 0001。
