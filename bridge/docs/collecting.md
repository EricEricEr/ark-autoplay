# 采集者指南（Windows / MuMu）

本流程用于重放专家作业、采集训练轨迹。只采集不需要 GPU、torch 或模型权重。游戏由采集者在本机登录；交接数据无需提供账号密码。

## 1. 准备环境

| 项 | 要求 |
|---|---|
| 系统 | Windows 10/11 x64 |
| 模拟器 | MuMu，1920×1080、DPI 320；游戏已登录，先处理公告/首次教学 |
| Python | 3.12 + uv；版本由 `uv.lock` 固定 |
| MAA | [配套补丁运行时](../maacore/README.md)，保持 MAA GUI 关闭 |
| 作业 | 维护者分配的队列，匹配账号练度、进度和助战供给 |
| 存储 | 为截图预留数十 GB；目录使用英文路径可减少 Windows 编码问题 |

仓库包含源码与补丁，不包含补丁 DLL、模型权重、静态数据集或真实截图。普通官方 MAA 无本项目的状态回调。首次部署可使用维护者提供的配套运行时，也可按运行时文档自行构建。

```powershell
git clone https://github.com/EricEricEr/ark-autoplay.git
cd ark-autoplay/bridge
uv sync --frozen
Copy-Item configs/instances/example.yaml configs/instances/collector.local.yaml
```

修改 `collector.local.yaml` 中的 `adb_path`、`adb_port`、`maa_dir`。模板端口只是示例，以本机实例为准。当前工厂有目录名护栏：`maa_dir` 路径中必须包含 `maa_runtime_patched`；这个名称本身不能证明文件正确。

本地配置使用 `.local.yaml` 后缀，已被 Git 忽略。原有 `mumu_local*.yaml` 是开发机历史配置，不要直接照抄其中的磁盘路径。

## 2. 先检查连接与名册

以下命令均在 `bridge/` 执行；将示例数据目录改为自己的路径。`maa_driver` 会连接游戏并截图，先结束其他自动化任务。

```powershell
$env:PYTHONUTF8 = '1'
$dataRoot = 'C:/ark-data'
New-Item -ItemType Directory -Force "$dataRoot/debug" | Out-Null
uv run python -m bridge.maa_driver --instance configs/instances/collector.local.yaml --user-dir "$dataRoot/debug/maa_user"
uv run python -m bridge.collect_operbox --instance configs/instances/collector.local.yaml --out "$dataRoot/roster/operbox.json" --user-dir "$dataRoot/debug/maa_user"
```

OperBox 导出当前账号的干员、精英化、等级、潜能等。**它不提供完整技能/模组等级**；对本批使用的干员（含助战）另保留游戏详情截图，注明：

- 名称、精英化、等级、潜能；
- 实际所选技能、普通技能等级、该技能专精等级；
- 实际装备的模组名称/ID、模组等级，未装备明确记“未装备”；
- 是否助战、采集时间。升级、换技能或换模组后保存新快照。

未知值留空并说明，不能猜测或填成 0。项目技能编码为普通 1～7、专一/二/三为 8/9/10；保留原始截图便于复核。若使用已有森空岛导出，仅交接名册结果，不交接登录凭据。

## 3. 取得队列并先跑小样

维护者分配队列，采集者将其保存为 `$dataRoot/configs/queue.json`。仓内 `configs/jobs_main_v1.json` 只是历史小队列，不代表适合所有账号。

```powershell
uv run python -u -m bridge.replay_controller --jobs "$dataRoot/configs/queue.json" --instance configs/instances/collector.local.yaml --nav configs/nav_main.yaml --work $dataRoot --limit 3
```

首批完成后，按第 5 节检查；维护者确认有状态、动作、编队和通关证据后，再去掉 `--limit 3` 批量跑：

```powershell
uv run python -u -m bridge.replay_controller --jobs "$dataRoot/configs/queue.json" --instance configs/instances/collector.local.yaml --nav configs/nav_main.yaml --work $dataRoot --skip-done
```

脚本运行时保持终端/进程存活。连续出现回主页 StartUp 失败时停止本批、保留日志与画面再排查，不要把整批失败当成完成。`--skip-done` 当前以 `episode.json` 存在为依据，**不判断胜负或数据质量**；失败/未知局由维护者验收后制定补跑队列，不能盲目跳过。

队列筛选的已确认口径为 `gen_queue --max-missing 1 --require-req`；当前导航还需 `--require-fight`，账号进度可通过 `--account-progress` 限制。队列不是通关承诺：助战列表仍须真的有目标、技能可用，练度未知项仍需复核。`--require-ops-known` 会额外要求持有，不适合直接用于需要借战的队列。

当前重放入口没有“允许碎石”的采集者开关；普通采集者不应将维护者自抽号的碎石授权当作自己的授权。理智不足时先停止，按账号所有者明确约定处理。

## 4. 保留原始产物

```text
ark-data/
  episodes/<stage_id>/job-<id>-<时间>/
    episode.json
    battle_states.jsonl
    action_events.jsonl
    shots/
  tmp/maa_jobs/job_<id>.json
  configs/queue.json
  roster/                       名册、补充练度及证明截图
  debug/                        批次汇总、MAA 日志
```

`tmp/maa_jobs/` **必须带回**：episode 的 `job_file` 可能是采集者机器的绝对路径，接收端需要靠这些原始作业恢复编队、动作与溯源。原始文件不手改；异常写单独说明。

每批附一份文本或 JSON，记录贡献者代号、开始/结束时间及时区、游戏版本/服务器、运行 `git rev-parse HEAD` 的结果、MAA/补丁版本、使用的实例配置、队列名称与异常。采集者自行保留原件，交接配置时移除任何自行添加的凭据。

每局状态和动作都有 `t_ms`；动作 `location=[x,y]=[col,row]`，状态 `deployed` 则直接标 `row/col`，不要手工统一或交换坐标。程序的状态采样发生在作业动作前，截图默认 1.5 秒一张并有事件补拍；它不是连续录像。

若同时录制连续视频，用现有录屏工具保存清晰、原速、未剪辑画面，从编队录到结算，并记录与本局的对应关系；录屏不是结构化状态的替代品。技能/模组等详情可另行截图。

关卡需准确注明 ID、显示关码、难度；敌人种类/出怪数量由维护者关联同版本关卡数据。实时敌人位置仍需画面标注，MAA 状态文件目前不提供。

## 5. 接收与验收

| 检查 | 口径 |
|---|---|
| 原始文件 | JSON/JSONL 可读，作业 JSON、队列、名册、版本信息齐全 |
| 状态 | `battle_states.jsonl` 存在且非空，有真实变化而非全零/冻结 |
| 动作 | 有真实 CopilotAction；状态与动作顺序/时间可对应，异常记账 |
| 编队/练度 | 能恢复该局实际使用的干员、技能、助战和练度时间口径 |
| 通关 | 结合 `battle_result` 和结算证据复核；`ok`/`done` 单独不足以判胜 |
| 分池 | 胜局、败局、unknown、中断局分别验收；后几类不直接混入通关专家训练池 |

原始事件流还包含截图/任务事件，**总事件行数不等于动作数**。当前特定非交互动作也可能不产生状态，不能机械要求所有行 1:1。

数据交付不强制压缩格式或分卷大小。真实采集包不通过 GitHub/Git 提交，双方约定传输方式即可；仓库只收代码、补丁、文档和合成测试。
