# RFC-0001：状态 / 动作 / 轨迹协议（草案占位）

- 状态：草案（占位，M0 期间迭代）
- 作者：项目维护组
- 创建：2026-10-08
- 实现库：`src/ark_core/protocol/`（pydantic v2 schema，schema 即文档）
- 版本来源：`src/ark_core/protocol/version.py`（当前 `0.1.0`）

## 1. 版本化承诺

协议遵循 **semver**：

- 破坏性变更（删字段 / 改语义）→ 升主或次版本；
- 纯增量（新增可选字段）→ 升修订版本；
- 任何变更必须在 `protocol/migrations/` 提供旧版迁移器，并更新本 RFC；
- bridge / client 与本仓库之间仅以本协议的版本化 JSON 通信（IPC / 落盘），不做代码级链接。

## 2. 状态（state）字段清单草案

| 字段 | 类型 | 说明 |
|---|---|---|
| `protocol_version` | str | 协议版本号（semver） |
| `t_ms` | int | 距开局的毫秒时间戳 |
| `stage` | object | 关卡：`code`（如 `1-7`）/ `family`（章节活动家族，如 `main_00_03`） |
| `cost` | int | 当前费用 |
| `life` | int | 剩余生命值 |
| `deploy_limit` | int | 可部署人数上限 |
| `kill_count` | int | 击杀数 |
| `tiles` | array | 格子：`r` / `c` / `kind`（melee/high/ranged/forbidden）/ `device` |
| `hand` | array | 待部署卡：`op`（featvec 特征向量标识）/ `cost` / `cooldown_ms` / `charges` |
| `deployed` | array | 已部署：`op` / `r` / `c` / `dir` / `hp_ratio` / `sp_ratio` / `skill_ready` |
| `enemies` | array | 敌人：`kind` / `pos`（`route` + `progress`，不用像素）/ `hp_ratio` / `speed` |

要点：干员 / 敌人以**数值特征向量**（featvec，由解包数值生成）标识，禁用身份 embedding；敌人位置用"路线 + 进度"。

## 3. 动作（action）字段清单草案

| 动作 | 字段 | 说明 |
|---|---|---|
| `deploy` | `op` / `r` / `c` / `dir` | 部署干员到格子并指定朝向 |
| `skill` | `op_uid` | 开启指定干员技能 |
| `retreat` | `op_uid` | 撤退指定干员 |
| `wait` | — | 等待（含低置信度保守决策） |

非法动作（费用不足 / 格子占用 / 冷却中）由动作掩码在模型输出侧屏蔽。

## 4. 轨迹（episode）草案

Parquet 一行一局：`episode_id` / `protocol_version` / `stage` / `squad_hash` / `source`（prts.plus 作业作者署名等）/ `result` / `ticks[]`（`t_ms` + `shot_sha256` + `state` + `action`）。

**红线**：轨迹内只存截图哈希、不存截图本体；截图由重采脚本在使用者本机重建（素材红线，见 `data/README.md`）。

## 5. 待办

- [ ] 状态子结构（Tile / HandCard / DeployedOperator / Enemy）拆为强类型（M1）
- [ ] 动作掩码的接口定义（M1）
- [ ] 迁移器框架与首个迁移样例（M0 末）
