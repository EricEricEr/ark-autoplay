# data/ —— 素材红线与数据说明

## 素材红线（最高优先级）

游戏截图、立绘、地图素材、解包数据（ArknightsGameData）的版权归**鹰角网络（Hypergryph）**所有，**一律不进本仓库**。本项目与鹰角网络无任何关系，为研究性质的非官方项目。

- ✅ 本目录**只允许**：schema 说明文档、字段示例、**合成样例**（程序随机生成、不含任何真实游戏画面或数值拷贝）。
- ❌ **永不入库**：真实对局截图、真实通关轨迹（Parquet）、干员立绘 / 图标、地图贴图、解包 JSON、模型权重、作业原始数据。
- 大体积数据一律放 `data/large/`（已在 .gitignore 中忽略）或本机任意目录，**不进 git**。

## 真实数据如何获取

| 数据 | 获取方式 |
|---|---|
| 通关轨迹（Parquet，CC-BY-4.0） | Hugging Face Datasets 发布（版本 tag），仓库内不放数据本体 |
| 对局截图 | 仓库只提供"重采脚本 + 校验哈希"，使用者在自己机器上玩 / 重放时本机重建 |
| 模型权重 | Hugging Face（独立使用条款），不入 git |
| 静态关卡 / 数值数据 | 用户本机脚本从解包数据生成，不入库 |

## 数据管线产物 schema（datapipe v1，ADR-0002）

以下产物由 `ark_core.datapipe` 在本机静态数据源上构建，**只存在于各机器本地**（默认 `<数据根>/processed/`，见 configs/data.yaml），发布只走 Hugging Face 版本 tag：

| 产物 | 说明 |
|---|---|
| `featvec_operators.parquet` / `featvec_enemies.parquet` | 干员 / 敌人数值特征向量；列清单与归一化尺度见随行的 `featvec_schema.json`（`featvec_version` 版本化，列变更必须升版） |
| `stages_registry.parquet` | 关卡注册表：`stage_id/code/name/type/difficulty/family/rows/cols/格子统计/options 摘要` |
| `episodes_v0.jsonl` | copilot 作业转换的 episode（`protocol_version: "0.1.0"`）：`stage`（含 family）、`source`（prts.plus 署名：job_id/title/upload_time/views/like/rating_ratio）、`roster_tier`、`squad`（char_id+skill+req）、`groups`、`actions`（deploy/skill/retreat/speed/bullet_time，timing + target subtype）、`stats`（aux 跳过计数） |
| `jobs_roster_tiers.json` | 练度 5 桶 counts + per_family + by_job（桶语义见 docs/adr/0002 §6） |
| `build_report.json` | 构建全量计数报告（数据源版本、转换率、deploy subtype 分布、unknown_name 热榜） |

deploy target 的 subtype 定义（`device` / `category` / `unknown_name`）与词表见 `configs/deploy_vocab.yaml` 与 ADR-0002 §4。

## 合成样例约定

- 样例用于单测 / golden 测试 / 协议演示，必须由代码随机生成或手工编造，与真实游戏数值无对应关系。
- 提交样例前自查：不包含可辨识的真实游戏素材片段；干员 / 敌人以 `featvec:*` 代号引用（与 prts.wiki 通用代号对齐），不附图像。

红线检查由 ReviewBench rubric 的 Documentation / Security 类覆盖，PR 模板要求逐项确认。
