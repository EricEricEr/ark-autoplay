# AGENTS.md —— ark-autoplay-maa-bridge

> 本文件是所有 AI 协作者（Copilot / Codex / Cursor / Gemini CLI / Kimi Code 等）的总入口（AAIF 约定）。
> **CI 命令变更时必须同步更新本文件**（CI 含本文件与工作流一致性检查的前瞻要求，见设计文档 §13）。

## 项目概述

Arknights Autoplay 合并仓的 `bridge/` 子目录：**数据工厂 + 战场状态供给**。职责：MAA 自动重放 prts.plus 作业，逐 tick 落盘结构化战场状态与动作，产出带协议版本戳的通关轨迹。当前为 **v1WIP**：重放工厂链路已实测跑通（导航/接管/落盘），squad_builder 与 instance_pool 仍为占位 stub。

### 铁律（最高优先级）

1. **执行环境仅限 MuMu 模拟器**（1920×1080 / DPI 320 基线）；其他端一律不支持。
2. **状态采集使用配套补丁 MaaCore**：上游 pin、完整补丁、重建方法见 `maacore/README.md` 与 `UPSTREAM.json`。已确认属于上游衍生作品、采用 AGPL-3.0；历史“附加条款待粘贴”结论已由 `NOTICE.md` 更正。普通官方 DLL 不能替代。
3. 跨进程边界走版本化协议 / 落盘 JSON（为解耦与可复现性，**不再**是许可证隔离要求——全项目已统一 AGPL-3.0，见 `../core/docs/adr/0004-unify-agpl-monorepo.md`）。
4. **采集者默认禁止源石补理智**：`resource_policy.allow_originite` 缺项等同 false。只有该账号所有者明确同意才能在本地配置开放；维护者本机自抽号的既有授权单独保留，不推广到他人。共享模板保持禁止，不购买新源石或充值。
5. **个人目录路径不得入库**：代码/示例用占位符或环境变量，真实机器配置放入已忽略的 `*.local.yaml`。提交前在仓库根运行 `python scripts/check_private_paths.py --worktree`；根 `.github/workflows/privacy.yml` 检查提交快照。`--history` 可另外检查本地引用可达的历史 blob，快照检查不等于历史清理。
6. **后续更新必须不含隐私**：遵守根 `AGENTS.md` 和 `docs/PRIVACY.md`。暂存区与待推送新增提交均须通过 `scripts/check_privacy.py`，并人工复核；本地发布仓库使用 pre-commit/pre-push 门禁。旧历史按用户裁决保留，不强推。

## 目录布局

```
├── src/bridge/              # Python 包 ark-bridge
│   ├── maa_driver.py        # MAA 薄封装：连接/回调事件流/Copilot 下发/adb 截图点触
│   ├── navigator.py         # 主线导航：固定点位（钳位锚定）+ 蓝钮像素校验进 briefing
│   ├── state_logger.py      # 作战期节拍截图（1.5s tick + 事件补拍，PNG 落盘）
│   ├── replay_controller.py # 作业重放调度：领作业→进图→执行→结算推进→存盘
│   ├── blocker_watchdog.py  # 卡死自愈看门狗：阻断画面像素检出（判负结算/等级过低横幅/
│   │                        #   同步弹窗）+ 战斗中剧情冻结停滞检测（story_freeze，文本区
│   │                        #   连点）+ 点屏消解 + episode 记账（blockers_dismissed /
│   │                        #   battle_result；单测全合成图，真实截图不入库）
│   ├── episode_writer.py    # 轨迹写出（protocol_version 戳 + 截图校验哈希）
│   ├── convert_episodes.py  # 采集端离线配对/结构验收；输出新目录，不动原件，无视频依赖
│   ├── squad_builder.py     # 按作业阵容自动编队；阵容随机化变体生成（占位 stub）
│   └── instance_pool.py     # MuMu 多开实例池（2-3 并发，多开器管理；占位 stub）
├── maacore/                 # 上游 pin 与重建说明；本地 _src/ 不入库
├── patches/                 # 对 fork 的本地改动补丁（按序编号，不直接改 fork 树）
├── configs/
│   ├── jobs_main_v1.json    # v1 主线低星作业队列（内嵌 MAA 形态作业 + prts.plus 署名）
│   ├── nav_main.yaml        # 主线导航固定点位表（钳位锚定坐标 + 蓝钮校验参数）
│   ├── jobs/                # 后续每关重放任务定义（作业 id、阵容、变体数）
│   └── instances/           # MuMu 实例定义（端口/分辨率/DPI；example.yaml 为模板，
│                            #   mumu_local.yaml 为开发机实例，路径属机器本地配置）
├── docs/adr/                # 决策记录（0001 = 纯 Python 驱动官方 MAA，不 fork）
├── tests/                   # 纯逻辑单测（转换器/写出器/几何表/像素判定）
├── .github/                 # CI、Copilot 指令、PR/Issue 模板
├── LICENSE                  # AGPL-3.0
└── NOTICE.md                # 衍生作品声明与已核实的上游许可/使用约束
```

### 运行时依赖（本机执行，非 CI）

- **配套补丁 MAA 运行时**：`maa_driver` 经 `Asst.load` 调用；DLL、resource、Python/asst 必须配套，见 `maacore/README.md`。驱动期间 MAA GUI 保持关闭。采集者以 `example.yaml` 创建 `*.local.yaml` 本地配置。
- **MuMu 12 模拟器**（1920×1080 / DPI 320）+ 其自带 adb；实例地址入同上配置。
- 数据落盘根（episodes/tmp/debug）由运行参数 `--work` 指定，属机器本地目录。
- 采集分工见 `docs/collect-only.md` 与 `docs/collect-and-convert.md`；当前结构化训练不需要 MP4/PNG 输入。`python -m bridge.convert_episodes --data-root <根> --out <新目录>` 仅用标准库做离线转换，不连接游戏、不建词表或分配训练划分。
- 练度采集优先 `tools/skland/collect.mjs` 一键扫码导出（现有 Node.js 18+，无 npm 依赖）；不能扫码才用 OperBox。全部技能/模组和原始响应保留在名册中，登录态只留本地 `.secrets/`。扫码工具离线检查：`node --test tools/skland/collect.test.mjs`。

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
2. 全项目已统一 AGPL-3.0，允许 core/bridge/client 相互 import；跨进程协议仍版本化。实验性模型闭环与专家采集使用不同数据根。
3. 自动化方式只做"截图 + 模拟输入"（ADB），不读内存、不改客户端。
4. 文档与代码中禁止出现任何暗示使用本工具可规避游戏运营方检测或处罚的表述。
5. 密钥、账号、cookie 等任何凭据不入库。

## 参考

- 设计文档（v2.1 立项方案）：§3 许可证策略、§5.2 本目录树、§6 数据协议、§10 数据工厂、§13 工程质量体系。
- 相关目录：`core/`（协议 / 感知 / 模型 / 训练 / 评测）、`client/`（玩家端）；三者同为 AGPL-3.0。
