# AGENTS.md —— ark-autoplay-core

> AI 协作 agent 的总入口（AAIF 开放标准，Copilot / Codex / Cursor / Gemini CLI 通用）。
> **本文件中的命令必须与 CI 实际命令一致；CI 命令变更时必须同步更新本文件。**

## 项目概述

明日方舟自主作战 AI 的核心仓库（Apache-2.0）：协议库（状态 / 动作 / 轨迹，semver 版本化）、感知后处理、双轨决策模型（实体 Transformer + Qwen3-VL-4B 对照）、训练（自监督 + 行为克隆 + DAgger）、三档零样本评测与 ONNX 导出。与 bridge（AGPL，数据工厂）、client（玩家端）两个仓库仅通过版本化 JSON 协议通信。

## 目录布局

```text
src/ark_core/
├── protocol/      # 状态/动作/轨迹 pydantic schema（改动必须升版本号）+ migrations/ 迁移器
├── perception/    # state_builder（bridge 原始输出→标准状态）、tile_map（格子坐标系）、enemy_det/
├── models/        # track_a/（实体 Transformer+指针网络+掩码头）、track_b/（Qwen3-VL-4B QLoRA 配方）
├── training/      # pretrain_ssl / imitate / augment / dagger
├── evaluation/    # splits（三档划分+泄漏检查）、runner、report、dispute/（争议重标模板）
├── export/        # to_onnx / quantize
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

## 代码风格

- lint 强制走 ruff（配置见 `pyproject.toml`），提交前本地先跑 `uv run ruff check`。
- 所有公共 API 必须有类型注解；pyright（basic 模式）不得报错。
- docstring 与注释一律使用中文；每个模块首行 docstring 写清职责。
- 立项骨架期：占位 stub 用 `# TODO(Mx)` 标注所属里程碑，不写真实实现逻辑。
- 一切超参入 `configs/`（yaml），禁止代码内魔法数字。

## PR 约定

- 标题格式：`type(scope): 摘要`，`type ∈ {feat, fix, chore, docs, test, refactor, perf}`，scope 取模块名（如 `protocol`、`track_a`）。
- 所有 PR（人审或 agent 审）必须按 **ReviewBench 九类缺陷 × 三级严重度**打标，清单已内置在 `.github/PULL_REQUEST_TEMPLATE.md`：Correctness / Security / Reliability / Maintainability / Testing / Performance / API architecture / Accessibility / Documentation × High / Medium / Low。
- CI 全绿（ci / license-guard）才可合入；模型类改动须附评测报告（见 `.github/instructions/models.instructions.md`）。
- 协议改动走 RFC（`docs/rfc/`），必须升 `protocol/version.py` 版本号并在 `protocol/migrations/` 提供迁移器。

## 安全红线

- **禁止提交任何游戏素材**：截图、立绘、地图素材、解包数据（ArknightsGameData）版权归鹰角网络，一律不入库；`data/` 只放 schema 与合成样例。
- **禁止 import bridge/maa**：core 与 client 仓库禁止 import bridge 仓库及任何 maa 相关模块的代码（AGPL 隔离，`license-guard` 工作流强制检查，命中即红）；跨仓库只走 IPC / 落盘 JSON。
- **禁止提交密钥与凭据**；权重与数据集只发布到 Hugging Face，不入 git。
- 评测划分最高准则：按关卡家族（章节 / 活动）划分训练 / 测试，严禁按对局随机划分；泄漏检查用例位于 `tests/leak/`。
