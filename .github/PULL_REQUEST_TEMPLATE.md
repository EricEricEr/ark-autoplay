<!--
PR 标题格式：<type>(<scope>): <摘要>
type ∈ feat / fix / chore / docs / test / refactor / ci / protocol
-->

## 变更说明

<!-- 做了什么、为什么；协议字段改动必须升 PROTOCOL_VERSION 并在此附迁移说明 -->

## 关联 issue

<!-- Closes #xxx -->

## ReviewBench 九类自评打标（必填）

> 按 ReviewBench rubric：9 类缺陷 × 3 级严重度。请逐项勾选本 PR 涉及的风险等级（无风险勾"无"），并在高风险项后附一句说明。

| 类别 | High | Medium | Low | 无 |
|---|---|---|---|---|
| Correctness（状态映射/动作/坐标正确性） | [ ] | [ ] | [ ] | [ ] |
| Security（注入/依赖漏洞/凭据入库） | [ ] | [ ] | [ ] | [ ] |
| Reliability（超时兜底/断点续跑） | [ ] | [ ] | [ ] | [ ] |
| Maintainability（协议版本/模块依赖） | [ ] | [ ] | [ ] | [ ] |
| Testing（新逻辑测试/golden） | [ ] | [ ] | [ ] | [ ] |
| Performance（落盘 IO/并发开销） | [ ] | [ ] | [ ] | [ ] |
| API architecture（bridge↔core IPC 协议兼容） | [ ] | [ ] | [ ] | [ ] |
| Accessibility（日志可读性/错误提示） | [ ] | [ ] | [ ] | [ ] |
| Documentation（协议 RFC/AGENTS.md/本仓文档同步） | [ ] | [ ] | [ ] | [ ] |

## 红线自查（全部勾选才能合入）

- [ ] 未提交任何游戏素材（截图/立绘/地图/解包数据）
- [ ] 未新增任何供 core / client 代码级引用本仓的入口（隔离红线）
- [ ] 仅"截图 + 模拟输入"，无内存读取 / 客户端修改
- [ ] CI 变更已同步更新 `AGENTS.md`（如适用）
- [ ] `uv run ruff check` 与 `uv run pytest` 本地通过

## 测试证据

<!-- 贴本地 / CI 运行结果摘要 -->
