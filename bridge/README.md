# bridge —— 数据工厂 + 战场状态供给

> 合并仓 `ark-autoplay` 的 `bridge/` 子目录｜Data factory & battlefield-state supply
> Data factory & battlefield-state supply for Arknights Autoplay.

---

## 中文

### 一句话定位

**用 MAA 自动重放 prts.plus 社区作业当数据工厂：逐 tick 落盘"结构化战场状态 + 动作"，产出带协议版本戳的通关轨迹。**（总项目一句话方案见设计文档 §18，本目录为其中的"数据工厂"角色。）

### 在架构中的位置

```
MuMu模拟器 ──► bridge（本目录，数据工厂）──► 落盘 JSON / 版本化协议 ──► core（感知后处理/决策/训练）
              MAA 驱动 + 重放工厂
```

- 只认 prts.plus 作业 JSON 与游戏画面；输出即协议。
- **调用方式**：经 `Asst.load()` 动态加载**官方 MAA 发行版**并调用其公开 `asst` 接口，**不复制、不修改**上游源码（决策记录见 [docs/adr/0001](docs/adr/0001-pure-python-driver.md)）。`maacore/` 与 `patches/` 当前为空，为将来可能的状态导出改造预留。
- **历史沿革**：三仓时代本目录是 AGPL 隔离仓，其代码禁止被 core/client import（只能经 IPC 协议通信）。2026-10-09 全项目统一 AGPL-3.0 后该隔离已非必需，三个子目录可自由互相 import（见 [ADR-0004](../core/docs/adr/0004-unify-agpl-monorepo.md)）。

### 当前状态

**v1WIP（roadmap M0 进行中）**：重放数据工厂链路已实测跑通：固定点位导航
进 briefing → MAA Copilot 接管作战 → 节拍截图 + 回调事件流 → episode bundle
落盘。squad_builder / instance_pool 仍为占位 stub。

**已知关键限制（实测）**：MAA 内部**有**战场状态识别器（`BattlefieldMatcher::kills_analyze` /
`oper_cooling_analyze` 等，配套 `BattleCostFlag` / `BattleKillsFlag` 资源），但
**不通过 Python `asst` 接口暴露**（MaaCore 导出的 C API 无 `GetCost`/`GetKills`，回调
消息类型亦不含战场状态）。故当前落盘的战场状态**只有截图本体**，数值状态（费用/击杀/
技力）需自行 OCR 或改造 MaaCore 导出——这是训练数据链路的**当前卡点**，详见
[STATUS.md](../STATUS.md)。

### 执行环境

**仅限 MuMu 模拟器**（Windows 10/11 x64），识别基线固定 **1920×1080 / DPI 320**；官方 PC 客户端、其他模拟器、手机真机均不支持。实例启动时校验参数，不符基线拒绝运行。

### 许可证与衍生声明

- 本项目**整体采用 AGPL-3.0**（见根目录与本目录 `LICENSE`）。
- **衍生关系现状**：本目录**当前不是** MaaAssistantArknights / MaaCore 的衍生作品——它经公开 API 调用官方发行版，未包含、未修改上游源码（`maacore/`、`patches/` 均为空）。
  若将来为导出战场状态而导入 MaaCore 源码，本目录**即成为上游衍生作品**，届时须按 [NOTICE.md](NOTICE.md) 粘贴上游附加条款原文并保持 AGPL-3.0。

### 素材红线声明

- 本项目与鹰角网络（Hypergryph）无任何关系，为**非官方**研究性质项目。
- **不分发任何游戏素材**：游戏截图、立绘、地图素材、解包数据一律不进仓库；轨迹只存截图哈希，由使用者本机重采重建。
- **权利方异议响应**：若鹰角网络或相关权利方对本目录任何内容提出异议，维护者承诺在收到通知后尽快配合处理（下架、整改或删除），联系渠道为本合并仓 Issue。

### 相关目录

本目录为**合并仓 `ark-autoplay` 的 `bridge/` 子目录**（2026-10-09 由三独立仓库合并，见 [ADR-0004](../core/docs/adr/0004-unify-agpl-monorepo.md)）：

| 目录 | 角色 | 许可证 |
|---|---|---|
| `core/` | 协议 / 感知后处理 / 决策模型 / 训练 / 评测 | AGPL-3.0 |
| `bridge/`（本目录） | 数据工厂 | AGPL-3.0 |
| `client/` | 玩家端：录屏 + 本地推理 + 模拟输入 | AGPL-3.0 |

### 快速上手

```bash
uv sync --dev        # 安装依赖（uv）
uv run pytest        # 纯逻辑单测
uv run ruff check    # lint
```

本机实机跑工厂（开发机环境）：

1. 前置：MuMu 12（1920×1080 / DPI 320，游戏已登录停在任意常规界面）；
   官方 MAA 发行版解压安装（本目录只读调用其 `asst` Python 接口，
   **MAA GUI 保持关闭**）；参照 `configs/instances/example.yaml` 写好
   自己的实例 YAML（adb 路径、MAA 目录、实例端口均为机器本地配置）。
2. 冒烟连接：`python -m bridge.maa_driver --user-dir <本地工作目录>/debug/maa_user`
3. 批量重放：`python -m bridge.replay_controller --work <本地数据目录>`
   （作业队列为 `configs/jobs_main_v1.json`；episode 束落盘到
   `<数据目录>/episodes/`；导航点位标定见 `configs/nav_main.yaml` 注释）。

贡献规范见 `AGENTS.md`；PR 请按模板完成 ReviewBench 九类打标。

---

## English

### What this is

**The data factory of Arknights Autoplay: MAA replays prts.plus community stage guides automatically, logging structured battlefield state + actions tick-by-tick into versioned, protocol-stamped episodes.** It lives in `bridge/` of the unified `ark-autoplay` repository.

### Position in the architecture

- Sits between the MuMu emulator and `core/`.
- **How it uses MAA**: dynamically loads the *official MAA release* via `Asst.load()` and calls its public `asst` API — upstream source is neither copied nor modified. This directory is currently **not** a derivative work of MaaCore (`maacore/` and `patches/` are empty). Importing MaaCore source in the future would make it one, and `NOTICE.md` governs that case.
- **Historical note**: in the three-repo era this was an AGPL-quarantined repo whose code could not be imported by core/client (IPC protocol only). After the whole project moved to AGPL-3.0 on 2026-10-09 the quarantine is no longer required; subdirectories may import each other freely (see [ADR-0004](../core/docs/adr/0004-unify-agpl-monorepo.md)).

### Status

**v1WIP (roadmap M0).** The replay chain is verified end-to-end on real hardware; `squad_builder` / `instance_pool` remain stubs.

### Runtime environment

**MuMu emulator only** (Windows 10/11 x64), fixed baseline **1920×1080 / DPI 320**. No official PC client, other emulators, or physical phones.

### License & asset policy

- **AGPL-3.0** for the whole project (see root and local `LICENSE`).
- Unofficial, research-oriented project, **not affiliated with Hypergryph**; **no game assets** (screenshots, artworks, maps, unpacked data) are distributed — only hashes and re-capture scripts.

### Sibling directories

- `core/` — protocol / perception post-processing / models / training / evaluation
- `client/` — player-side runtime: capture + local inference + simulated input
