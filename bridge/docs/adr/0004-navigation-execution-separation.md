# ADR-0004：导航与执行分离——Copilot 不导航，必须另有前置导航

- 状态：已接受（2026-10-09）
- 修正：ADR-0003（"只用 Copilot，一步到位"）中"Copilot 自带导航"的**部分错误**
- 关联：`nav_maa.py`、`replay_controller.py`

## 背景：ADR-0003 的乐观假设被实测否证

ADR-0003 判定"Copilot 一步到位（自带导航 + 执行作业）"，据此**删掉了预导航阶段**。
端到端实测**失败**：

```
[job 48628] 任务链收尾=TaskChainError 动作事件=0
```

逐事件诊断（`data-validation/diag_copilot.py`）：

```
[ 3.42s] SubTaskError   subtask=ProcessTask first=['BattleStartPre']   ← 找不到"开始行动"
[ 3.94s] SubTaskStart   subtask=BattleFormationTask                    ← 编队却成功了（27s）
[27.86s] SubTaskStart   task=BattleStartAll (JustReturn)
[32.67s] SubTaskError   subtask=ProcessTask first=['BattleStartAll']   ← 找不到开战按钮
[32.78s] TaskChainError
```

**编队成功但找不到"开始行动"** → 说明游戏**不在关卡 briefing 界面**。

## 根因（源码依据）

`CopilotTask::set_params` 有**两条互斥分支**：

| 分支 | 触发条件 | 是否导航 |
|---|---|---|
| `filename_opt` | 传 `filename`（**我们用的**） | ❌ **完全不导航**，只设 `m_stage_name` |
| `multi_tasks_opt` | 传 `copilot_list` | ✅ 调 `navigate_to_stage` |

而 `navigate_to_stage` 位于 `MultiCopilotTaskPlugin::_run`，**只在 multi 分支被启用**
（`m_multi_copilot_plugin_ptr->set_enable(true)`）。普通 Copilot 分支明确
`set_enable(false)`。

即：**普通 `Copilot` 任务假设游戏已在关卡 briefing 界面**——
`BattleStartPre` 是 OCR 找"开始行动/开始作战"按钮，不在该界面必然失败。

ADR-0003 的推断错误在于：看到 `MultiCopilotTaskPlugin` 里有 `navigate_to_stage`
就认为"Copilot 自带导航"，**未核对该插件在普通分支下是 `set_enable(false)`**。

## 那为什么当初要加"Fight 预导航"？——它本来是对的

回溯 ADR-0002：原设计"Fight 预导航到 briefing → 抢停 → Copilot 接管"**逻辑成立**，
因为 Copilot 确实需要前置导航。问题只出在**载体选错**：`Fight` 会走代理作战。

### Fight 必走代理（源码依据）

```json
"FightBegin": {
  "algorithm": "JustReturn",
  "next": ["Fight@UsePrts-Annihilation", "Fight@UsePrts", "Fight@UsePrts-StageSN",
           "Fight@StartButton1", "Fight@PRTS#next"]
}
```

`UsePrts` **排在 `StartButton1` 之前**；`UsePrts` 的 ROI `[1000,550,280,90]`
正是关卡准备界面的"代理指挥"按钮。故**只要该关存在代理记录，Fight 必然先点代理**。
实测 `Fight@PRTS1` 出现 13 次、`PrtsErrorConfirm` 4 次、`FightMissionFailed` 73 次
（号商代理记录连自己都跑不通）。

## 决策

**导航与执行必须分离**，且导航**不得使用会触发代理作战的任务**：

1. **执行**：`Copilot(filename=...)` —— 忠实回放 prts.plus 玩家作业，
   显式 `NotUsePrts`，无代理风险；
2. **导航**：需要一个**纯导航、不含作战**的手段把游戏带到 briefing。
   候选按优先级：
   - `multi_copilot`（`copilot_list`）分支，它内部走 `navigate_to_stage` +
     `NotUsePrts`，**理论上最干净**（待验证）；
   - 若不可用，则用 `Fight` 导航但在**检测到 `UsePrts` 时立即停止**，
     并依赖 `prts_detected` 标记剔除被污染样本；
   - 自建几何导航器（ADR-0002 已废弃）**不恢复**——它逐关手工标定，不可扩展。

3. **护栏保留**：`PRTS_TASKS` 运行时检测 + episode 的 `prts_detected` 字段
   （正常为空），作为"导航手段是否干净"的持续验证，不因方案调整而移除。

## 未决问题（下一步要验证）

- `multi_copilot` 分支能否作为纯导航使用？它的 `navigate_to_stage` 是否会在
  到达 briefing 后**自动开始作战**（`_run` 里在导航后直接进入执行流程）？
  若会，则需在其后接 `AsstStop` —— 又回到抢停模式。
- 是否存在更干净的官方入口（如单独的 `StageNavigation` 任务）？
  初步查证：`AsstAppendTask` 支持的类型仅
  `Fight/StartUp/Infrast/Recruit/Roguelike/Copilot/SSSCopilot/ParadoxCopilot`，
  **没有独立的导航任务类型**。

## 后果

- **正**：明确了 Copilot 不导航这一事实，避免继续在错误假设上浪费时间；
  代理护栏与 `prts_detected` 标记已落地，可持续验证数据纯净度。
- **负**：数据工厂流程**当前不可用**（Copilot 无法自己到位），
  必须先把导航问题解决才能批量采数据。
- **状态**：代码已按 ADR-0003 改为 Copilot-only；导航缺口待补（见"未决问题"）。
