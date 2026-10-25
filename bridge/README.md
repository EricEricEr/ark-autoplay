# bridge —— 数据采集与实验性闭环

合并仓 `ark-autoplay` 的数据工厂：通过补丁 MaaCore 重放 prts.plus 作业，将动作前战场状态、执行事件和节拍截图保存成每局一个 bundle。

**协助采集请读 [采集者指南](docs/collecting.md)。** 运行时来源、版本和重建见 [maacore/README.md](maacore/README.md)；上游关系见 [NOTICE.md](NOTICE.md)。

## 当前实现（2026-10-26）

| 模块 | 能力与边界 |
|---|---|
| `maa_driver` | MAA 连接、回调、截图/输入、SingleStep 参数下发 |
| `replay_controller` | 回主页、Fight `nav_only` 导航、Copilot 编队重放、结算、落盘 |
| `battle_state` / `state_logger` | 动作前状态；1.5 秒节拍截图和事件补拍，并非连续全帧采样 |
| `blocker_watchdog` | 部分弹窗、判负屏、剧情冻结的有界消解；长批次仍会失败 |
| `roster_state` / `gen_queue` | 双通道名册合并、账号可达/练度/进度过滤 |
| `collect_operbox` | 已有 OperBox 采集流程的可配置入口，不含完整技能/模组等级 |
| `convert_episodes` | 采集端离线配对/校验、标准 JSONL 和轻量 bundle 输出，无需视频/GPU |
| `tools/skland/collect.mjs` | 可扫码账号的首选，一次扫码自动导出全干员、全部技能和模组明细 |
| `loop_runner` | 实验性模型闭环，需 core 训练环境与本地模型产物；普通采集不运行它 |

MAA 原生编队支持选助战，2026-10-25 已在好友维什戴尔的 1-8 作业中验证。允许缺一名不代表列表一定有该干员，也不保证助战技能/练度达标。

## 环境与入口

验证环境为 Windows 10/11 x64 + MuMu，基线固定 1920×1080、DPI 320。ADB 端口与安装路径按采集者机器配置，同一实例不同时由 MAA GUI 和脚本控制。

```powershell
uv sync --frozen
uv run python -m bridge.replay_controller --help
uv run python -m bridge.collect_operbox --help
uv run python -m bridge.convert_episodes --help
node tools/skland/collect.mjs --help
uv run pytest -q
uv run ruff check
```

纯采集依赖 numpy、Pillow、PyYAML，无需 torch/GPU。按 [完整实例模板](configs/instances/example.yaml) 创建本地副本，再提供 `--instance`、`--jobs`、`--nav`、`--work`。

练度优先使用 [森空岛扫码入口](tools/skland/README.md)；Node.js 18+，无 npm 依赖。不能扫码的账号才使用 OperBox 并补充技能/模组资料。

## 数据与许可

原始采集每局保存 `episode.json`、`battle_states.jsonl`、`action_events.jsonl`、`shots/`。原始作业在数据根的 `tmp/maa_jobs/`，交接须一并带回；本地转换包将作业内置为 `job.json`。当前结构化训练不读取视频/截图，画面原件留在采集者本地，按需补交复核帧。`ok=true` / `fail_stage=done` 表示流程完成，不能单独作为通关证明。

真实数据、截图、权重、个人配置和凭据不进入 Git；采集者保留原件，由维护者验收。交付不强制压缩格式或分卷大小。

全项目 AGPL-3.0，`patches/` 保存对 MaaCore 的修改，上游 checkout 和运行时二进制单独准备。本项目与鹰角网络、MAA Team 无合作关系，不使用 MAA Logo。

## English

The data factory replays Copilot jobs through a patched MaaCore runtime and records pre-action state, execution events, and local screenshots. See the [collector guide](docs/collecting.md) and [runtime instructions](maacore/README.md). Stock MAA lacks this project's state callback. Collection needs no training GPU or checkpoint; the experimental model loop is separate.
