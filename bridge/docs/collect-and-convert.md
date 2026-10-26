# 指南 B：采集并在本地转换

适合能运行 Python 命令、愿意协助检查和整理数据的协助者。按 [指南 A](collect-only.md) 完成环境、扫码练度和专家重放，再用本指南离线转换。**无需视频、GPU、torch 或官方静态图鉴，也不需要自己训练模型。**

**采集阶段默认禁止源石及理智药补理智**：`resource_policy.allow_originite: false`、`allow_medicine: false`，未配置也禁止。维护者本机自抽号的单独授权不适用于采集者；只有账号所有者同意后才能单独开放本地配置。理智不足造成的失败记账后继续队列。转换阶段只读写本地文件，不连接游戏、不消耗游戏资源。

## 1. 运行本地转换

在仓库 `bridge/` 目录执行，输入为本批原始数据根（含 `episodes/` 与 `tmp/maa_jobs/`）。输出必须是尚不存在的新目录：

```powershell
uv sync --frozen
$env:PYTHONUTF8 = '1'
uv run python -m bridge.convert_episodes --data-root C:/ark-data --out C:/ark-export/batch_20261026_001
```

命令不连接游戏、不访问网络、不读取画面。原件不修改、不删除；已存在的输出目录会报错，防止旧 ETL 那样追加出重复样本。重跑请用新的批次目录。

## 2. 这一步转换什么

- 读取原始 episode、状态流、事件流和原始作业，保留来源、关卡、实际动作和时间戳。
- 仅将明确标记 `win` 的专家局送入结构验收；`lost`、`unknown`、缺胜负字段、模型闭环局、空壳局进入报告。
- `SkillDaemon` 控制回调不作为训练动作；支持“全部动作均有状态”及“仅真实动作有状态”两种完整序列，通过明确来源索引配对，不能仅删动作却保留其控制帧。严格核对数量、类型、目标、时间顺序、部署坐标/朝向。
- 不能对齐的整局记入待复核，不静默截断或猜动作。`CopilotAction` 回调可能发生在等待条件之前，两个时间戳分别保留，不把回调当成物理执行时刻。
- 输出完整状态—动作对 JSONL，并附可直接交给现有 core 构建器的轻量 bundle；每局内置 `job.json`，不再依赖另一台机器的绝对路径。
- 生成输入/输出 SHA-256、逐局索引和质量报告。转换不建全局词表，不分 train/val/test。

结构校验不是人工通关认证：`win` 仍来自采集器元数据。本工具不读取结算图片，缺少结算图片引用时会提示警告。保留本地证据，供维护者抽查。

## 3. 输出与交付

```text
batch_20261026_001/
  MANIFEST.json                 协议版本、文件哈希与来源
  quality_report.json           扫描/保留/拒收数，逐局原因
  episodes_index.jsonl          逐局索引和原始输入哈希
  samples.jsonl                 完整状态、动作、回调及时间戳
  episodes/<stage_id>/<局号>/
    episode.json                派生元数据，注明来源及未含图片
    job.json                    本局作业，不依赖原机器路径
    battle_states.jsonl         原状态文件字节不变
    action_events.jsonl         原事件文件字节不变
```

交付转换目录，**另外附上**批次说明、扫码/OperBox 名册及助战补充、原始队列、运行日志。转换器不自动复制这些文件，避免误把登录态混入包。补充资料可作为并列目录交接；`MANIFEST.json` 的文件哈希只覆盖转换器输出本身。

不必交 MP4，不必上传 PNG 全集；原始 episode 和截图仍保存在采集者本地。对判定存疑的局，补交相应的结算帧/动作帧即可。转换目录不包含被拒局的训练输入，拒收原因在报告中，原始局没有被丢弃或删除。

## 4. 看懂质量报告

| 原因 | 含义 |
|---|---|
| `result_not_explicit_win` | 缺少明确 win；交维护者复核，不能直接改原件为 win |
| `job_json_missing_inside_data_root` | 缺少随包作业；检查 `tmp/maa_jobs/job_<id>.json` |
| `state_action_event_count_mismatch` | 状态、有效动作、回调数量不一致 |
| `action_type_alignment_mismatch` / `action_target_alignment_mismatch` | 可能用了不同作业、未执行到该步或候选组需解析 |
| `unsupported_job_action_type` | 作业含当前转换器未支持的动作，交维护者处理 |
| `*_timestamp_not_monotonic` | 对局时间轴异常 |
| `model_loop_not_expert_data` | 模型生成数据，不能直接混入专家模仿训练池 |

旧存量局即使曾通关，也可能没有 `battle_result`，因此会被严格转换器拒收；这是可见的待复核项，不能把拒收数当成游戏失败数。若保留数为 0，命令仍会生成报告，但没有可训练样本。

## 5. 维护者如何使用

轻量 `episodes/` 已兼容 core 的全量/增量构建器，支持相对 `job_file`；维护者用统一的关卡特征、干员特征、词表与划分构建 npz。`samples.jsonl` 是可复用的标准化配对数据，也可用于独立分析。

不同词表或划分构建的 npz 不能直接拼接。本转换器不生成独立词表/划分；本指南的交付终点为上述可合并结构化包，最终 npz 构建仍由维护者统一完成。

如果后续训练像素模型或敌人检测，还需要图像与视觉标注；现有结构化 JSON 无法恢复画面。这里的“不必交视频”适用于 MAA 已同步采到状态/动作的任务，来源只有视频的少人 WIKI 仍需从视频提取画面和动作。
