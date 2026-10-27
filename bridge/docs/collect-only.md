# 指南 A：仅采集数据（普通玩家）

使用 Windows 采集助手，**无需安装 Python、Node.js 或输入命令**。不需要 GPU、模型权重或 MP4。界面与异常处理说明见 [采集助手使用说明](collector-app.md)。

## 点击流程

1. 完整解压维护者提供的软件 ZIP，双击 `ArkCollector.exe`。不要只复制 exe。MuMu 和游戏需已安装，账号由玩家本人登录。到“任务与设置”将采集数据保存位置选为自己的非C盘目录；软件设置、登录态及系统缓存可沿用系统默认位置。
2. 打开 MuMu 和游戏，进入主页；关闭其它正在控制同一模拟器的 MAA 或脚本。在采集面板检测并选择设备，再选择维护者提供的配套采集运行库。普通官方 MAA 缺少状态回调，不能替代。
3. 在“账号与练度”扫码获取练度，按提示选择当前游戏账号。不能扫码时可用游戏内扫描或导入名册；技能、模组缺项另留详情截图，助战练度单独记录，不猜填。
4. 点击“导入采集任务”，选择维护者按账号练度与进度分配的任务包，再点击“开始 / 继续未完成任务”。仅采集时，设置页的自动转换保持关闭。
5. 需要暂停时点击“当前局后暂停”；继续时再点击继续。退出前可选择“当前局后停止”。
6. 暂停或停止后，到“导出与帮助”点击“导出采集包”，将 ZIP 交给维护者；原始数据和截图继续留在本机。导出可按需包含截图或原始记录，软件不自动上传。

源石和理智药默认禁止，管理员权限不会开放补给。只有当前账号持有者主动授权才能单独启用已有补给；软件不购买源石、不充值。更换账号会清除此前授权与练度关联。

作业失败后记录并继续下一份，不因连续失败停队列。设备掉线会尝试恢复；登录、维护更新和未知弹窗需玩家处理。程序退出后重新打开可继续未完成任务；强制结束或断电时当前局可能不完整，不会冒充成功。截图持续占用磁盘空间。

<details>
<summary>维护者：源码部署、命令和完整验收口径</summary>

# 指南 A：仅采集数据（Windows / MuMu）

本流程用于重放专家作业、采集训练轨迹。只采集不需要 GPU、torch 或模型权重。游戏由采集者在本机登录；交接数据无需提供账号密码。

**默认禁止源石及理智药补理智。** `resource_policy.allow_originite` 和 `allow_medicine` 缺项均为 false。维护者本机自抽号的单独授权只适用于该账号，采集者不能继承。只有该账号所有者明确同意后，才能在该账号的本地配置单独开放。理智不足导致本份失败时记账继续后续清单，不因连续失败停整批。

## 1. 准备环境

| 项 | 要求 |
|---|---|
| 系统 | Windows 10/11 x64 |
| 模拟器 | MuMu，1920×1080、DPI 320；游戏已登录，先处理公告/首次教学 |
| Python | 3.12 + uv；版本由 `uv.lock` 固定 |
| 扫码练度 | 能扫码的账号使用现有 Node.js 18+ 运行森空岛入口，无 npm 依赖 |
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

修改 `collector.local.yaml` 中的 `adb_path`、`adb_port`、`maa_dir`，安装路径按本机填写。模板端口只是示例，以本机实例为准。当前工厂有目录名护栏：`maa_dir` 路径中必须包含 `maa_runtime_patched`；这个名称本身不能证明文件正确。AppData、TEMP、ProgramData及uv等工具的常规默认位置允许在系统盘；采集资料目录由用户选择非C盘位置，不把它写成固定的C盘根目录。

本地配置使用 `.local.yaml` 后缀，已被 Git 忽略。原有 `mumu_local*.yaml` 只保留模板，不含可直接运行的安装路径或设备端口。工厂、模型闭环与连接冒烟入口都须显式传入 `--instance`，不会自动使用开发机实例。

## 2. 练度优先扫码，OperBox 为不能扫码时的备选

**账号可以森空岛扫码时，一次命令完成扫码和练度获取，无需逐名手抄自有干员练度。** 这也是项目已确认的双通道优先顺序。以下命令在 `bridge/` 执行：

```powershell
$env:PYTHONUTF8 = '1'
$dataRoot = Read-Host '输入非C盘的数据保存目录完整路径（由你自行选择）'
if ([string]::IsNullOrWhiteSpace($dataRoot)) { throw '必须选择数据目录' }
$dataRoot = [IO.Path]::GetFullPath($dataRoot)
New-Item -ItemType Directory -Force "$dataRoot/debug" | Out-Null
node tools/skland/collect.mjs --data-root "$dataRoot"
uv run python -m bridge.roster_state --skland "$dataRoot/roster/roster_skland.json" --out "$dataRoot/roster/roster_merged.json"
```

磁盘不设默认值，C盘数据根或名册输出会在登录和写文件前被拒绝。名册输出到 `$dataRoot/roster/`，登录态默认输出到 `$dataRoot/.secrets/`；`--cred` 可单独指定本地凭据文件，系统默认凭据位置允许在C盘。终端显示二维码，用森空岛 APP 扫码并确认后自动导出；已有有效登录态时直接同步。多绑定角色用 `--uid <游戏UID>` 指定，须与录制账号一致。详细参数见 [扫码工具说明](../tools/skland/README.md)。

名册保留干员精英化/等级/潜能、**全部技能**的等级/专精、全部模组 ID/等级及完整原始响应。记录采集时间；练度改变后重新同步。每局实际技能/模组由作业/画面对齐，不能把已拥有或其它技能的专精当作本局配置。

森空岛名册只覆盖扫码账号自有干员，**助战干员**的实际技能、模组和练度需另行记录。

只有无法扫码时才使用以下备选：

```powershell
uv run python -m bridge.collect_operbox --instance configs/instances/collector.local.yaml --out "$dataRoot/roster/operbox.json" --user-dir "$dataRoot/debug/maa_user"
uv run python -m bridge.roster_state --operbox "$dataRoot/roster/operbox.json" --out "$dataRoot/roster/roster_merged.json"
```

运行库或资料路径含中文且没有8.3短名时，MAA会使用系统临时目录或ProgramData中的兼容别名；这是允许保留的系统默认缓存，实际采集资料仍写入所选数据目录。

OperBox 提供精英化/等级/潜能，缺技能/模组等级；此时对本批干员另保留详情截图，注明：

- 名称、精英化、等级、潜能；
- 实际所选技能、普通技能等级、该技能专精等级；
- 实际装备的模组名称/ID、模组等级，未装备明确记“未装备”；
- 是否助战、采集时间。升级、换技能或换模组后保存新快照。

未知值留空并说明，不能猜测或填成 0。项目技能编码为普通 1～7、专一/二/三为 8/9/10。森空岛路径保留原始导出；OperBox 备选保留补充截图。登录态仅在自选数据根的 `.secrets/`，不交接凭证。

正式跑关前确认连接（该命令会连接模拟器并截图，先结束其它控制任务）：

```powershell
uv run python -m bridge.maa_driver --instance configs/instances/collector.local.yaml --user-dir "$dataRoot/debug/maa_user"
```

## 3. 取得队列并先跑小样

维护者分配队列，采集者将其保存为 `$dataRoot/configs/queue.json`。仓内 `configs/jobs_main_v1.json` 只是历史小队列，不代表适合所有账号。

```powershell
uv run python -u -m bridge.replay_controller --jobs "$dataRoot/configs/queue.json" --instance configs/instances/collector.local.yaml --nav configs/nav_main.yaml --work $dataRoot --limit 3
```

首批完成后，按第 5 节检查；维护者确认有状态、动作、编队和通关证据后，再去掉 `--limit 3` 批量跑：

```powershell
uv run python -u -m bridge.replay_controller --jobs "$dataRoot/configs/queue.json" --instance configs/instances/collector.local.yaml --nav configs/nav_main.yaml --work $dataRoot --skip-done
```

脚本运行时保持终端/进程存活。单份导航、编队或执行失败，以及单局执行中抛出的异常，会记账后继续下一份；连续失败不会自动停止整个队列。摘要逐份写入`debug/factory_summary_*.jsonl`，正常结束或人工中断时另写汇总JSON。连接失败、写盘失败等进程级故障仍需人工恢复，不能把退出当成整批完成。`--skip-done` 当前以`episode.json`存在为依据，**不判断胜负或数据质量**；失败/未知局由维护者验收后制定补跑队列，不能盲目跳过。

### 异常恢复的实际边界

当前标准入口已有已知阻断画面处理、剧情/结算推进、回主页重试、单局异常隔离和增量事件/状态落盘。单张截图失败会记账并继续采样，**不代表ADB已重连**。异常退出时会回收该局截图线程，避免下一局画面继续写进上一局。

尚无通用的ADB掉线重连、模拟器/游戏崩溃后自动重启、采集进程异常退出后自动拉起，以及断电后自动恢复流程。磁盘写满、更新维护、登录失效和未识别弹窗也没有完整自动处理闭环。进程被强杀或机器掉电时，当前局仍可能不完整；已写出的旁路数据保留，不能据此认定该局完成或胜利。

以上是已实现能力的说明，不是“所有意外都会自动恢复”的保证。`--skip-done`只处理已有完成文件的跳过，失败补跑仍需按质量报告另行安排。

队列筛选的已确认口径为 `gen_queue --max-missing 1 --require-req`；当前导航还需 `--require-fight`，账号进度可通过 `--account-progress` 限制。队列不是通关承诺：助战列表仍须真的有目标、技能可用，练度未知项仍需复核。`--require-ops-known` 会额外要求持有，不适合直接用于需要借战的队列。

实例模板显式设置：

```yaml
resource_policy:
  allow_originite: false
```

缺少该项也按禁止处理。派发或更新任务时，非零 `stone` 请求会在到达 MAA 前被拒绝；导航显式使用 `stone: 0`，自动吃药保持原有关闭默认值。

只有该账号所有者明确同意后，才可修改其本地实例副本；禁止把允许配置作为共享模板。`allow_originite: true` 只放行调用者显式请求的源石次数，不自动增加补给，不包含购买或充值；Copilot 本身不提供 `stone` 补给次数接口。

## 4. 本地保留原件，交付结构化数据

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

**MP4 非当前结构化训练的必交项，也无需把节拍 PNG 合成视频。** 默认交付 JSON/JSONL、原始作业、名册、队列和日志；完整 PNG 留在采集者本地，复核时按局补取结算帧/动作帧。技能、模组缺口及助战练度按前述规则补充。

若同时录制视频，保留清晰、原速、未剪辑画面和本局对应关系。图像用于审计和视觉任务，当前结构化训练不读取它们；完成转换或交接不意味着删除本地原件。

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

</details>
