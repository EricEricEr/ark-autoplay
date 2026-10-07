# Copilot 快速上手 —— ark-autoplay-maa-bridge

## 这是什么

Arknights Autoplay（明日方舟自主作战 AI，完全开源）三仓库之一：**数据工厂 + 战场状态供给**。MAA/MaaCore 衍生，AGPL-3.0 隔离仓。当前为立项骨架（M0），均为 docstring 占位 stub。

## 铁律（先读这个）

- 本仓代码**绝不被 ark-autoplay-core / ark-autoplay-client import**；对外只走 IPC（本地 socket / 落盘 JSON）+ 版本化协议。反之亦不 import core。
- 执行环境**仅限 MuMu 模拟器**，基线 1920×1080 / DPI 320。
- **禁止提交任何游戏素材**（截图 / 立绘 / 地图 / 解包数据）；轨迹只存截图哈希。
- 完整规范以根目录 `AGENTS.md` 为准，本文件只是速览。

## 构建 / 测试 / lint（与 CI 一致）

```bash
uv sync --dev
uv run ruff check
uv run pytest
```

CI 只有一个工作流 `.github/workflows/ci.yml`（ruff + pytest）。改 CI 命令时同步更新 `AGENTS.md`。

## 布局速览

| 路径 | 作用 |
|---|---|
| `src/bridge/state_logger.py` | 战场状态逐 tick 落盘 |
| `src/bridge/replay_controller.py` | prts.plus 作业重放调度（领作业→进图→执行→判胜负） |
| `src/bridge/episode_writer.py` | 轨迹写出（协议版本戳 + 校验哈希） |
| `src/bridge/squad_builder.py` | 自动编队 + 阵容随机化变体 |
| `src/bridge/instance_pool.py` | MuMu 多开实例池 |
| `maacore/` `patches/` | MaaCore fork（留上游历史）+ 补丁式改动 |
| `configs/jobs/` `configs/instances/` | 重放任务 / 实例定义（yaml） |
| `tests/` | 落盘完整性、断点续跑、协议兼容测试 |

## 风格

ruff + 类型注解；docstring 用中文；PR 标题 `<type>(<scope>): <摘要>`，并按 PR 模板完成 ReviewBench 九类打标。
