<!--
PR 标题格式：type(scope): 摘要
type ∈ {feat, fix, chore, docs, test, refactor, perf}；scope 取模块名（protocol / perception / track_a / track_b / training / evaluation / export / baselines / ci / docs …）。
协议（protocol/）改动必须升版本号 + 附迁移器 + 更新 docs/rfc/；模型（models/）改动必须附评测报告。
-->

## 变更说明

<!-- 做了什么、为什么 -->

## ReviewBench 九类缺陷打标（必填）

<!-- 对照评审 rubric，自查本 PR 涉及哪些类别的风险，勾选并注明严重度；无风险可全选"无"。
     类别含义举例见 AGENTS.md 与设计文档 §2.2。 -->

| 缺陷类别 | 无风险 | High | Medium | Low | 备注 |
|---|---|---|---|---|---|
| Correctness（正确性） | ☐ | ☐ | ☐ | ☐ | |
| Security（安全） | ☐ | ☐ | ☐ | ☐ | |
| Reliability（可靠性） | ☐ | ☐ | ☐ | ☐ | |
| Maintainability（可维护性） | ☐ | ☐ | ☐ | ☐ | |
| Testing（测试） | ☐ | ☐ | ☐ | ☐ | |
| Performance（性能） | ☐ | ☐ | ☐ | ☐ | |
| API architecture（接口架构） | ☐ | ☐ | ☐ | ☐ | |
| Accessibility（可用性） | ☐ | ☐ | ☐ | ☐ | |
| Documentation（文档） | ☐ | ☐ | ☐ | ☐ | |

## 自查清单

- [ ] 本地通过 `uv run ruff check` / `uv run pyright` / `uv run pytest`
- [ ] 未 import 任何 bridge / maa 相关模块（license-guard 红线）
- [ ] 未提交游戏截图 / 立绘 / 解包数据 / 密钥（素材与安全红线）
- [ ] 新增逻辑附测试；超参入 `configs/` 而非代码内
- [ ] 若改了 CI 命令：已同步更新 AGENTS.md

## 关联 issue / 评测报告

<!-- Closes #xxx；模型改动贴评测报告链接 -->
