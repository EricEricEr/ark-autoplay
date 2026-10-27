# AGENTS.md —— ark-autoplay-core

> AI 协作 agent 的总入口（AAIF 开放标准，Copilot / Codex / Cursor / Gemini CLI 通用）。
> **本文件中的命令必须与 CI 实际命令一致；CI 命令变更时必须同步更新本文件。**

## 项目概述

明日方舟自主作战 AI 合并仓的 core 子目录（AGPL-3.0）：协议、数据管线、实体 Transformer、状态行为克隆、SSL/K 折评测与实验性推理。像素轨道、DAgger、完整 ONNX 玩家端仍未交付。与 bridge/client 同属单仓，可相互 import；跨进程使用版本化协议。

## 目录布局

```text
src/ark_core/
├── protocol/      # 状态/动作/轨迹 pydantic schema（改动必须升版本号）+ migrations/ 迁移器
├── perception/    # state_builder（bridge 原始输出→标准状态）、tile_map（格子坐标系）、enemy_det/
├── models/        # track_a/（实体 Transformer+指针网络+掩码头）、track_b/（Qwen3-VL-4B QLoRA 配方）
├── training/      # pretrain_ssl / imitate / augment / dagger
├── infer/         # 在线推理胶水（track_a_v1：battle_state → 合法作业动作，链路冒烟层，非能力证据）
├── evaluation/    # splits（三档划分+泄漏检查）、runner、report、dispute/（争议重标模板）
├── export/        # to_onnx / quantize
├── datapipe/      # 数据管线 v1（featvec/关卡元数据/作业转换/装置类别词表/低练分桶，ADR-0002；
                   # stagefeat 关卡静态特征 v1（ADR-0003）/ routefeat 路线几何→每格通道（v1.1））
└── baselines/     # 随机合法动作等基线
configs/           # 实验配置（yaml，一实验一配置，禁止代码内超参）
tests/             # unit / integration / golden（合成样例）/ leak（划分泄漏检查）
data/              # 仅 schema 说明与合成样例，真实数据永不入库
docs/              # rfc/（协议）、adr/（决策记录）、评测报告
```

## 环境命令（与 CI 一致）

| 命令 | 用途 |
|---|---|
| `uv sync` | 安装全部依赖（含 dev 组，生成 .venv） |
| `uv run pytest` | 运行全部测试（unit + leak + 占位） |
| `uv run ruff check` | lint 检查 |
| `uv run pyright` | 类型检查 |

**训练依赖是独立可选的 `train` 组**（torch + numpy，约 2.5GB）：

| 命令 | 用途 |
|---|---|
| `uv sync --group train` | 额外安装训练依赖（CI 与数据管线**不需要**，保持门禁轻量） |
| `$env:ARK_ML_ROOT="<数据根>"` | 训练管线所有输入/输出路径的根（机器路径不入库） |
| `uv run python -m ark_core.training.run build` | 构建模仿学习数据集（episodes → npz） |
| `uv run python -m ark_core.training.run train` | 训练 track_a（配置见 `configs/imitation_v0.yaml`） |
| `uv run python -m ark_core.training.run_v1 build` | 构建 **v1** 数据集（真机 battle_states ↔ 作业动作对 → npz，配置见 `configs/imitation_v1.yaml`） |
| `uv run python -m ark_core.training.run_v1 train` | 训练 v1（状态注入版 track_a；指标含 macro-F1 与同口径常数基线，链路冒烟不作能力证据） |

torch 缺失时 `tests/unit/test_imitation.py` 自动 skip，故未装 train 组也能跑全部门禁。
训练任务定义、指标口径与**能力边界**见 `docs/adr/0005-imitation-v0.md`（v0 无战场状态，
不是自主作战，不得如此表述）。

## 代码风格

用户已确认（2026-10-27）的当前监督训练流程：轮换验证后最终全量训练，各份有标签数据均参与训练和折外评测，不再永久固定四关只做验证。流程说明见 `docs/rotation-full.md`；折外成绩与最终全量模型的已见数据成绩分别报告，预训练接触过的验证数据不能冒称完全未见。

用户已确认的全量数据规则：默认不按动作、格子、出怪、编队、词表、网格边长或路线点数截断数据；预训练和评测不抽取前若干局/批。无标签观察和未配对原始流保留并注明用途，监督训练只使用可配对标签。新模型采用变长编队指针；旧固定头权重按原结构兼容加载。计算批大小不代表数据集条数上限。

- lint 强制走 ruff（配置见 `pyproject.toml`），提交前本地先跑 `uv run ruff check`。
- 所有公共 API 必须有类型注解；pyright（basic 模式）不得报错。
- docstring 与注释一律使用中文；每个模块首行 docstring 写清职责。
- 立项骨架期：占位 stub 用 `# TODO(Mx)` 标注所属里程碑，不写真实实现逻辑。
- 一切超参入 `configs/`（yaml），禁止代码内魔法数字。

## PR 约定

- 标题格式：`type(scope): 摘要`，`type ∈ {feat, fix, chore, docs, test, refactor, perf}`，scope 取模块名（如 `protocol`、`track_a`）。
- 所有 PR（人审或 agent 审）必须按 **ReviewBench 九类缺陷 × 三级严重度**打标，清单已内置在 `.github/PULL_REQUEST_TEMPLATE.md`：Correctness / Security / Reliability / Maintainability / Testing / Performance / API architecture / Accessibility / Documentation × High / Medium / Low。
- CI 全绿（ci）才可合入；模型类改动须附评测报告（见 `.github/instructions/models.instructions.md`）。
- 协议改动走 RFC（`docs/rfc/`），必须升 `protocol/version.py` 版本号并在 `protocol/migrations/` 提供迁移器。

## 安全红线

- **禁止提交任何游戏素材**：截图、立绘、地图素材、解包数据（ArknightsGameData）版权归鹰角网络，一律不入库；`data/` 只放 schema 与合成样例。
- **跨目录解耦（历史沿革）**：三仓时代 core/client 禁止 import bridge（AGPL 隔离，由已删除的 `license-guard` 工作流强制）。2026-10-09 全项目统一 AGPL-3.0 后该隔离**已非必需**，`core/`、`bridge/`、`client/` 之间可自由 import（见 `docs/adr/0004-unify-agpl-monorepo.md`）。跨进程边界仍走版本化协议 / 落盘 JSON，理由是解耦与可复现性，不再是许可证。
- **禁止提交密钥与凭据**；权重与数据集只发布到 Hugging Face，不入 git。
- 评测划分最高准则：按关卡家族（章节 / 活动）划分训练 / 测试，严禁按对局随机划分；泄漏检查用例位于 `tests/leak/`。
