# Copilot 快速上手 —— ark-autoplay-core

明日方舟自主作战 AI 的核心仓库（Apache-2.0）：协议 / 感知 / 决策模型 / 训练 / 评测。更完整的约定见根目录 [AGENTS.md](../AGENTS.md)，两者冲突时以 AGENTS.md 为准。

## 构建与测试

```bash
uv sync               # 安装全部依赖（含 dev 组）
uv run pytest         # 全部测试
uv run ruff check     # lint
uv run pyright        # 类型检查
```

提交前本地至少跑 `ruff check` + `pytest`；CI（ubuntu + windows 矩阵）四道门：以上三个命令 + license-guard（违禁 import 扫描）。

## 布局速览

| 路径 | 内容 |
|---|---|
| `src/ark_core/protocol/` | 状态 / 动作 / 轨迹 pydantic schema，semver 版本化（`version.py` 为唯一版本来源），改动必须升版本号 + 写迁移器（`migrations/`） |
| `src/ark_core/perception/` | bridge 原始输出 → 标准状态；格子坐标系；enemy_det/ |
| `src/ark_core/models/` | track_a/（实体 Transformer + 指针网络 + 掩码头）、track_b/（Qwen3-VL-4B QLoRA） |
| `src/ark_core/training/` | 自监督 / 行为克隆 / 增强 / DAgger |
| `src/ark_core/evaluation/` | 三档划分（含泄漏检查）、runner、report、dispute/ |
| `configs/` | 实验配置（yaml，一实验一配置，禁止代码内超参） |
| `tests/` | unit / integration / golden（合成样例）/ leak（划分泄漏用例） |
| `docs/` | rfc/（协议 RFC）、adr/（决策记录） |
| `data/` | 只放 schema 与合成样例，**真实轨迹与截图永不入库** |

## 硬性约定

- docstring / 注释用中文；公共 API 必须有类型注解；骨架期占位逻辑用 `# TODO(Mx)` 标注里程碑。
- 禁止 import 任何 bridge / maa 相关模块（AGPL 隔离，CI 强制）。
- 禁止提交游戏截图 / 立绘 / 解包数据与任何密钥。
- 协议字段变更走 RFC（`docs/rfc/`），模型改动须附评测报告。
- PR 标题 `type(scope): 摘要`；PR 描述按 ReviewBench 九类 rubric 打标（模板已内置）。
