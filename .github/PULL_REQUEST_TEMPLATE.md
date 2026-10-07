<!-- PR 标题格式：type(scope): 中文摘要（type ∈ feat/fix/docs/test/ci/chore/refactor） -->

## 变更说明

<!-- 做了什么、为什么做；关联 issue 用 #编号 -->

## 验证方式

<!-- 勾中项前把 [ ] 改成 [x]，并补充说明 -->

- [ ] 本地 `uv sync && uv run pytest && uv run ruff check && uv run pyright` 全绿
- [ ] 不涉及游戏素材（截图/立绘/地图/解包数据）入库
- [ ] 未 import bridge 仓库代码（ark-autoplay-maa-bridge）
- [ ] 若变更了 CI 命令，已同步更新 `AGENTS.md`

## ReviewBench 打标清单（九类缺陷 × 三级严重度）

按本 PR 实际触及/可能引入的问题，在每行勾选一个最高严重度；确认未涉及则勾选"无此项"。

| 缺陷类别 | 本项目含义举例 | 🟥 High | 🟨 Medium | 🟩 Low | ⬜ 无此项 |
|---|---|---|---|---|---|
| Correctness | 状态映射错位、动作掩码漏判、坐标系混用 | ☐ | ☐ | ☐ | ☐ |
| Security | 输入注入越权、依赖包漏洞、密钥入库 | ☐ | ☐ | ☐ | ☐ |
| Reliability | 识别超时未兜底、截图/输入通道断连无重试 | ☐ | ☐ | ☐ | ☐ |
| Maintainability | 协议字段无版本号、模块循环依赖 | ☐ | ☐ | ☐ | ☐ |
| Testing | 新模块无冒烟/回放测试、评测无复现脚本 | ☐ | ☐ | ☐ | ☐ |
| Performance | 推理超时预算（子弹时间 ≤2s）、ONNX 精度回退 | ☐ | ☐ | ☐ | ☐ |
| API architecture | 与 core 的协议/artifact 接口破坏 | ☐ | ☐ | ☐ | ☐ |
| Accessibility | 风险告知可跳过、日志不可读 | ☐ | ☐ | ☐ | ☐ |
| Documentation | 协议变更未更新文档、AGENTS.md 命令失效 | ☐ | ☐ | ☐ | ☐ |

## 打标说明（必填）

<!-- 对勾选 High/Medium 的每一项：说明位置、影响面、缓解措施 -->
