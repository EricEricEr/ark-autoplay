# AGENTS.md —— ark-autoplay-maa-bridge

> 本文件是所有 AI 协作者（Copilot / Codex / Cursor / Gemini CLI / Kimi Code 等）的总入口（AAIF 约定）。
> **CI 命令变更时必须同步更新本文件**（CI 含本文件与工作流一致性检查的前瞻要求，见设计文档 §13）。

## 项目概述

Arknights Autoplay 三仓库之一：**数据工厂 + 战场状态供给**。衍生自 MaaAssistantArknights / MaaCore，AGPL-3.0 隔离仓。职责：MAA 自动重放 prts.plus 作业，逐 tick 落盘结构化战场状态与动作，产出带协议版本戳的通关轨迹。当前处于**立项骨架阶段（M0）**，只有占位 stub。

### 铁律（最高优先级）

1. **本仓库代码绝不被 ark-autoplay-core / ark-autoplay-client import。** 对外通信只走 IPC（本地 socket / 落盘 JSON）+ 版本化协议；不暴露任何代码级接口。
2. 本仓库也**不 import core 的任何代码**；只认 prts.plus 作业 JSON 与游戏画面，**输出即协议**。
3. 执行环境**仅限 MuMu 模拟器**（1920×1080 / DPI 320 基线）。

## 目录布局

```
├── src/bridge/              # Python 包 ark-bridge（占位 stub）
│   ├── state_logger.py      # 战场状态落盘：每 tick 费用/击杀/干员卡/技力/格子占用
│   ├── replay_controller.py # 作业重放调度：领作业→建队→进图→执行→判胜负→存盘
│   ├── episode_writer.py    # 轨迹写出（protocol_version 戳 + 截图校验哈希）
│   ├── squad_builder.py     # 按作业阵容自动编队；阵容随机化变体生成
│   └── instance_pool.py     # MuMu 多开实例池（2-3 并发，多开器管理）
├── maacore/                 # MaaCore fork（保留上游 git 历史；当前为空）
├── patches/                 # 对 fork 的本地改动补丁（按序编号，不直接改 fork 树）
├── configs/
│   ├── jobs/                # 每关重放任务（作业 id、阵容、变体数）
│   └── instances/           # MuMu 实例定义（端口/分辨率/DPI），example.yaml 为模板
├── tests/                   # 落盘完整性、断点续跑、协议兼容测试（当前仅冒烟）
├── .github/                 # CI、Copilot 指令、PR/Issue 模板
├── LICENSE                  # AGPL-3.0
└── NOTICE.md                # 衍生作品声明 + 上游附加条款（首次导入源码时粘贴原文）
```

## 环境命令

依赖管理用 [uv](https://docs.astral.sh/uv/)（Python ≥ 3.12）：

```bash
uv sync --dev        # 建/同步虚拟环境并安装 dev 依赖
uv run pytest        # 运行测试
uv run ruff check    # lint
uv run ruff format --check   # 格式检查（如启用）
uv run pyright       # 类型检查（引入 pyright 后启用；当前未配置，属 TODO）
```

以上命令与 `.github/workflows/ci.yml` 保持一致；**改 CI 必改本节**。

## 代码风格

- ruff（规则集见 `pyproject.toml`）；类型注解全覆盖（向 pyright strict 靠拢）；
- **docstring 一律用中文**，每个模块 docstring 写清职责（可引用设计文档章节号）；
- 注释/文档中禁止出现不合规表述（见下方红线）；
- 一切配置入 `configs/`，禁止代码内魔法数字；协议相关常量（如 `PROTOCOL_VERSION`）集中定义并注明与 core 仓库协议库的对齐关系。

## 测试约定

- 纯逻辑单测逐步补到高覆盖；执行层不依赖真机（golden 文件 + tick 流回放）；
- 测试代码放 `tests/`，pytest 发现路径已在 `pyproject.toml` 配好。

## PR 约定

- 标题格式：`<type>(<scope>): <摘要>`，type ∈ `feat / fix / chore / docs / test / refactor / ci / protocol`，scope 用模块名（如 `state_logger`、`instance_pool`）。
- **必须按 ReviewBench 九类打标**（Correctness / Security / Reliability / Maintainability / Testing / Performance / API architecture / Accessibility / Documentation × High / Medium / Low），勾选表已在 `.github/PULL_REQUEST_TEMPLATE.md` 内置，逐项自评后提交。
- 协议字段改动：必须升 `PROTOCOL_VERSION` 并在 PR 描述附迁移说明。

## 安全红线（违反即打回）

1. **禁止提交任何游戏素材**：游戏截图、立绘、地图素材、解包数据（ArknightsGameData）版权归鹰角网络，一律不进仓库（落盘数据目录已在 `.gitignore` 兜底，但仍须自查）。
2. **隔离红线**：不得新增任何让 core / client 可以代码级引用本仓的入口（打包发布、公共符号导出到仓外接口等）。
3. 自动化方式只做"截图 + 模拟输入"（ADB），不读内存、不改客户端。
4. 文档与代码中禁止出现任何暗示使用本工具可规避游戏运营方检测或处罚的表述。
5. 密钥、账号、cookie 等任何凭据不入库。

## 参考

- 设计文档（v2.1 立项方案）：§3 许可证隔离、§5.2 本仓目录树、§6 数据协议、§10 数据工厂、§13 工程质量体系。
- 姐妹仓库：ark-autoplay-core（Apache-2.0）、ark-autoplay-client（Apache-2.0）。
