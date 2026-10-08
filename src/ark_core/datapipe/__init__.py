"""datapipe：数据管线 v1（featurize / 关卡元数据 / 作业转换 / 词表 / 低练分桶）。

把本机静态数据源（prts.plus 作业镜像 + 官方数值表，均不入库）转换为下游训练 / 评测
可直接消费的版本化产物：

- featvec.parquet：干员 / 敌人数值特征向量（profession·sub one-hot + 归一化数值 + 技能关键词标签）
- stages_registry.parquet：关卡注册表（含关卡家族 family）
- episodes_v0.jsonl：copilot 作业 → episode 轨迹（协议版本 protocol_version）
- jobs_roster_tiers.json：作业阵容练度分桶（5 桶）

输入路径全部由 configs/data.yaml + 环境变量 ARK_DATA_ROOT 解析，代码内不出现机器路径；
Deploy 动作的非干员名分类词表在 configs/deploy_vocab.yaml（数据即配置）。

CLI 入口：``uv run python -m ark_core.datapipe.build featvec|stages|episodes|tiers|all``
（见 build.py 与 README 数据管线 quickstart；设计与计数验证见 docs/adr/0002）。
"""
