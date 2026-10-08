# ADR-0002：数据管线 v1（datapipe v1）

- 状态：已接受（2026-10-08）
- 范围：`src/ark_core/datapipe/`、`configs/data.yaml`、`configs/deploy_vocab.yaml`、`tests/unit/test_datapipe_*`、`tests/leak/test_family_grouped_split.py`
- 数据版本：`data_version: "77.6.0+a550f5e+ak_dataset_v0.1"`（游戏数据 77.6.0；敌人库 commit 指纹 a550f5e；prts.plus 作业镜像 ak_dataset_v0.1）

## 背景

项目立项骨架（M0）需要一条可复现的离线数据管线，把本机静态数据源（prts.plus 爬取的
MAA copilot 作业镜像 + 官方数值表，**均不入库**，见 data/README.md 红线）转换成训练 /
评测可消费的版本化产物。快速试验已在 ak_dataset_v0.1 上验证关键口径，本 ADR 把
口径固化进生产实现，并记录与验证基线对齐过程中发现的语义细节。

## 决策

### 1. 产物与 CLI

`uv run python -m ark_core.datapipe.build featvec|stages|episodes|tiers|all` 产出：

| 产物 | 内容 | 实测行数 |
|---|---|---|
| featvec_operators.parquet | 干员 featvec（one-hot + 归一化白值 + 技能关键词标签 + has_module） | 458 |
| featvec_enemies.parquet | 敌人 featvec（每 (enemy_id, level) 变体一行） | 2426 |
| featvec_schema.json | 版本化列清单（featvec_version "0.1.0"，列变更必须升版） | — |
| stages_registry.parquet | 关卡注册表（含 family / 格子统计 / options） | 3279 |
| episodes_v0.jsonl | copilot 作业 → episode（protocol_version "0.1.0"，保留署名） | 40269 |
| jobs_roster_tiers.json | 作业练度 5 桶（counts + per_family + by_job） | 40299 |
| build_report.json | 全量计数报告（转换率、subtype 分布、unknown_name 热榜） | — |

### 2. 路径解析（红线：代码内零机器路径）

所有源路径由 `configs/data.yaml` + 环境变量 `ARK_DATA_ROOT` 解析：相对路径接数据根，
绝对路径原样采用；9 类源逐个预检，缺失时报错并列出完整缺失清单与修复指引
（`sources.check_sources`）。`enemy_handbook_table` 标记为可选源。

### 3. 关卡家族 family（stages_meta.family_of）

规则按序：① `^(main|tough|hard)_(\d+)` → `main_<章节>`（tough/hard 并入章节）；
② `^act[0-9a-z]+` → 活动前缀；③ `^(camp|wk|sub|pro|tr|a|rogue)` → token；
④ 其余取第一个 `_` 分段。**关键语义**：a-token 是前缀正则而非分段等值——
a001/a003 剿灭轮换合并为家族 `a`，且 act 规则必须先于 a-token（否则 act23side
被 `a` 吞掉）。此口径在 copilot 作业上复现验证基线：**102 个家族、main_00..main_17**
（按分段等值实现会得到 103，差在 a001/a003 的合并）。

### 4. Deploy 非干员名的三类 subtype（deploy_vocab）

作业 Deploy target 是自由文本，非干员名形式化为（不再笼统计查表失败）：
`device`（关卡装置 / 召唤物 token，原文保留）、`category`（类别通配占位符 →
{profession, sub_hint, position_hint?} 规范映射）、`unknown_name`（计数、保留原文）。
词表即数据：`configs/deploy_vocab.yaml`（77 装置 token / 158 类别词形 / 1 别名）。
匹配管线先规范化（去装饰引号、分隔符截头、练度标注截尾、尾数字截除），干员解析按
“别名改写 → 作业阵容槽位 → 全局唯一名”顺序（9 组全局重名靠阵容上下文消歧）。

实测（v1 真实构建）：deploy 动作 277815 = operator 193752 + device 44189 +
category 31023 + unknown_name 8851。unknown 长尾（“书刀”“赛柯”等约 2.6k 个低频名）
保留计数；**TODO(M1)**：从本机 level_*.json token 列表自动扩充 device 词表吸收长尾。

### 5. 动作映射与丢弃口径

Deploy→deploy、Skill→skill、Retreat→retreat、SpeedUp→speed、BulletTime→bullet_time；
MoveCamera / Output / Click / Swipe / SkillUsage / SkillDaemon 为辅助类（aux），
跳过并按类型计数（episode.stats 与 build_report 双侧）。全部动作均为 aux 的作业整局
丢弃。实测：40299 作业 → 40269 episode、丢弃 30（all_actions_skipped），五类保留动作
计数与源镜像 stats.json 完全一致。

### 6. 练度分桶（jobs_tiers）——与字面描述的偏差（重要）

5 桶：全4星以下 / 单核平民 / 单核但标精二 / 多核高配 / 无阵容信息。
任务书的字面口径（“单核 = ≤1 个 TIER_6 且其余 ≤ 4 星”）实测得 5137/2904/30239，
**与验证基线计数（726/5581/2959/29740/1293）不符**。逐一排除后确认：复现基线的
唯一语义是 **核心干员 = 稀有度 ≥ TIER_5（5/6 星均计核心），单核 = 至多 1 名核心**
（其余自然 ≤ 4 星）；“未标精二”= 所有槽位 req.elite < 2 且 module == 0；lineup 空或
全部槽位无法解析 → 无阵容信息；占位槽不参与稀有度判断、不计核心。
以此口径真实构建逐桶精确命中验证计数。**指定桶名与验证计数优先于字面公式**；
若后续确认原口径有意只按 TIER_6，需要重跑验证改基线（另立 ADR）。

### 7. episode 协议字段

protocol_version 固定 "0.1.0"（与 protocol/version.py 对齐，改动走 RFC 升版）。
source 小节保留署名链：type/job_id/title（作者标题）/upload_time/views/like/
rating_ratio/url；squad 槽位带 char_id+skill+req 原文（无 id 的占位槽附
placeholder_subtype）；target 带 subtype 载荷；timing 保留 MAA 触发条件
（kills/costs/cost_changes/pre_delay/post_delay，供重放 tick 对齐）。

### 8. 本机输出限制（machine-local quirk）

本机 python 无法在 agent 工作区外创建文件（疑为杀软 / 受控文件夹访问），故真实构建
产物输出到 agent 工作区目录，build_report.json 记录了绝对路径。**不为此修改任何
系统安全设置**。TODO(infra)：排查 Defender 受控文件夹访问排除项后，产物回归仓库
`data/large/`（.gitignore 已忽略，永不入库）。

## 后果

- 正：口径全部锚定验证基线（计数逐项可复现）；机器路径零入库；词表 / 归一化尺度 /
  开关全部入 configs（AGENTS 超参纪律）；unknown_name 诚实计数，长尾治理有 M1 抓手。
- 负：episode 体积较大（40269 行 ≈ 155MB JSONL，含 actions 全文）；v1 尚未做
  关卡 × 作业 × 阵容三维覆盖看板（设计文档 §10，后续里程碑）；技能标签为启发式
  子串（TODO(M1) 换人工标签表）。
- 验证基线对照：干员 458 / 敌人 2426 / 关卡 3279 / episode 40269 / 桶
  726-5581-2959-29740-1293 / 家族 102 —— 全部一致（build_report.json 可见）。
