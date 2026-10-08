# maacore/ —— MaaCore fork 策略

本目录将存放 **MaaAssistantArknights / MaaCore 的 fork**（当前为空，首次导入在 M0 内进行）。

## fork 原则

1. **保留上游 git 历史**：fork 以完整历史导入（不 squash、不重建提交），上游提交哈希可追溯，便于定期 rebase 跟进上游更新。
2. **本地改动以 `patches/` 维护**：我们对 MaaCore 的一切修改（战场状态落盘回调、重放控制钩子、日志增强等）**不直接混入 fork 树**，而是以独立 patch 文件存放在仓库根目录的 `patches/` 下，按序编号（`0001-xxx.patch`、`0002-yyy.patch`……），并在 patch 头部注明对应的上游 commit。
3. **导入前先看 NOTICE**：首次导入源码前，必须先将上游 AGPL-3.0 附加条款原文粘贴至根目录 `NOTICE.md`。
4. **构建方式**：`maacore/` 仅被本仓（`src/bridge/`）在同进程内调用；对 core / client 仓库不暴露任何代码级接口（只走 IPC / 落盘 JSON）。

## 更新上游的流程（规划）

```bash
# 在 maacore 内拉回上游，随后逐一 reapply patches/
git -C maacore fetch upstream
git -C maacore rebase upstream/master   # 冲突按 patch 对应关系解决
```

> 待核实（设计文档 §17）：MaaCore 现有回调是否足以支撑状态落盘，还是必须 fork 改造——第一个开发周给结论，直接转 issue 跟踪。
