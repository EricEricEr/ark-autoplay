# ark-autoplay-maa-bridge

> 数据工厂 + 战场状态供给（MAA/MaaCore 衍生，AGPL-3.0 隔离仓）
> Data factory & battlefield-state supply for Arknights Autoplay (MaaCore derivative, AGPL-quarantined).

---

## 中文

### 一句话定位

**用 MAA 自动重放 prts.plus 社区作业当数据工厂：逐 tick 落盘"结构化战场状态 + 动作"，产出带协议版本戳的通关轨迹——这是 Arknights Autoplay 三仓库中唯一的 AGPL 隔离仓。**（总项目一句话方案见设计文档 §18，本仓为其中的"数据工厂"角色。）

### 在架构中的位置

```
MuMu模拟器 ──► bridge（本仓，AGPL-3.0）──► 落盘 JSON / IPC ──► core（Apache-2.0）
              MaaCore fork + 重放工厂        版本化协议          感知后处理/决策/训练
```

- 只认 prts.plus 作业 JSON 与游戏画面；**不 import core 的任何代码，输出即协议**。
- **铁律：本仓库代码绝不被 [ark-autoplay-core](https://github.com/EricEricEr/ark-autoplay-core) / [ark-autoplay-client](https://github.com/EricEricEr/ark-autoplay-client) import。对外只通过 IPC（本地 socket / 落盘 JSON）+ 版本化协议通信。**（core 侧 CI 有 license-guard 依赖方向检查。）

### 当前状态

**立项骨架（roadmap M0，第 1 个月）**：仅有带 docstring 的占位 stub，无实现逻辑。M0 交付目标：bridge 状态落盘、重放工厂、首批 ≥500 条过协议校验的通关轨迹。

### 执行环境

**仅限 MuMu 模拟器**（Windows 10/11 x64），识别基线固定 **1920×1080 / DPI 320**；官方 PC 客户端、其他模拟器、手机真机均不支持。实例启动时校验参数，不符基线拒绝运行。

### 许可证与衍生声明

- 本仓库是 [MaaAssistantArknights](https://github.com/MaaAssistantArknights/MaaAssistantArknights) 的衍生作品，采用 **AGPL-3.0**（见 `LICENSE`；衍生声明与上游附加条款见 `NOTICE.md`）。

### 素材红线声明

- 本项目与鹰角网络（Hypergryph）无任何关系，为**非官方**研究性质项目。
- **不分发任何游戏素材**：游戏截图、立绘、地图素材、解包数据一律不进仓库；轨迹只存截图哈希，由使用者本机重采重建。

### 姊妹仓库

| 仓库 | 角色 | 许可证 |
|---|---|---|
| [ark-autoplay-core](https://github.com/EricEricEr/ark-autoplay-core) | 协议 / 感知后处理 / 决策模型 / 训练 / 评测 | Apache-2.0 |
| [ark-autoplay-maa-bridge](https://github.com/EricEricEr/ark-autoplay-maa-bridge) | 数据工厂（本仓） | AGPL-3.0 |
| [ark-autoplay-client](https://github.com/EricEricEr/ark-autoplay-client) | 玩家端：录屏 + 本地推理 + 模拟输入 | Apache-2.0 |

### 快速上手

```bash
uv sync --dev        # 安装依赖（uv）
uv run pytest        # 测试（当前仅冒烟）
uv run ruff check    # lint
```

贡献规范见 `AGENTS.md`；PR 请按模板完成 ReviewBench 九类打标。

---

## English

### What this is

**The data factory of Arknights Autoplay: MAA replays prts.plus community stage guides automatically, logging structured battlefield state + actions tick-by-tick into versioned, protocol-stamped episodes.** This is the only AGPL-quarantined repo of the three-repo project.

### Position in the architecture

- Sits between the MuMu emulator and `ark-autoplay-core`; derived from MaaCore (see `maacore/` and `NOTICE.md`).
- **Iron rule: code in this repo must never be imported by `ark-autoplay-core` or `ark-autoplay-client`. All cross-repo communication goes through IPC (local socket / on-disk JSON) with a versioned protocol.**

### Status

**Project skeleton (roadmap M0).** Docstring-only stubs; no implementation yet.

### Runtime environment

**MuMu emulator only** (Windows 10/11 x64), fixed baseline **1920×1080 / DPI 320**. No official PC client, other emulators, or physical phones.

### License & asset policy

- **AGPL-3.0** — derivative work of MaaAssistantArknights (see `LICENSE`, `NOTICE.md`).
- Unofficial, research-oriented project, **not affiliated with Hypergryph**; **no game assets** (screenshots, artworks, maps, unpacked data) are distributed — only hashes and re-capture scripts.

### Sibling repos

- [ark-autoplay-core](https://github.com/EricEricEr/ark-autoplay-core) (Apache-2.0)
- [ark-autoplay-client](https://github.com/EricEricEr/ark-autoplay-client) (Apache-2.0)
