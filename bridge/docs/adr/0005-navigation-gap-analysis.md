# ADR-0005：导航缺口的完整分析——Copilot 只做地图内导航，缺"章节寻路"

- 状态：**已解决**（2026-10-09）——采用方案 A（patch 0002），实测通过
- 修正：ADR-0004 中"multi_copilot 分支理论最干净"的推测**部分成立但不充分**
- 关联：`docs/adr/0003`、`docs/adr/0004`、`nav_maa.py`、`patches/0002-fight-nav-only.patch`

## 结论（2026-10-09 实测通过）

实施了**方案 A**：给 `FightTask` 增加 `nav_only` 参数（patch 0002），
导航到关卡准备界面后禁用作战子任务、干净收尾，再交给 `Copilot` 执行玩家作业。

实测（`data-validation/verify_nav_only.py --stage 0-2`）：

```
asst::Assistant::append_task Fight {"stage": "0-2", "nav_only": true}
asst::FightTask::set_params nav_only enabled: navigate only, no battle

导航链路（全部命中）：
  Episode0 (章节寻路) → ClickChapterNew → EnterEpisodeNew
  → ClickStageName → ClickedCorrectStage
终态 TaskChainCompleted（24.2s）
```

三条判据：

| 判据 | 结果 | 证据 |
|---|---|---|
| 到达关卡准备界面 | ✅ | **蓝钮像素占比 0.520**（已知 baseline：briefing=0.52，其余界面=0.0） |
| 零代理痕迹 | ✅ | 本轮 1418 行日志内 `UsePrts`/`PRTS1`/`PrtsErrorConfirm` **均为 0** |
| 零作战动作 | ✅ | 无 `BattleProcessTask`，24.2s 正常 `TaskChainCompleted` |

完整流程（导航 + 执行）实测：`nav_only` 导航到位后交 `Copilot`，
**`CopilotAction` 出现 11 次**（作业动作确实在执行），
且走的是 Copilot 的 `BattleStartAll`/`BattleStartNormal` 链（非 `FightBegin`），
故不触碰 `UsePrts`。

以下保留原分析与候选方案的评估记录。

## 实测：`copilot_list`（multi 分支）确实会导航，但卡在主界面

用正确字段名（源码依据 `CopilotTask.h:19-26` 的
`MEO_JSONIZATION(MEO_OPT id, filename, MEO_OPT nav_name_override, MEO_OPT is_raid)`）
下发：

```python
copilot_list = [{"filename": <作业绝对路径>, "nav_name_override": "0-2"}]
```

⚠️ 此前用**数组** `[id, filename, nav_name, is_raid]` 会触发 C++ 反序列化异常
（`WinError 0xe06d7363`）——**字段名是 `nav_name_override` 而非 `nav_name`，
且必须是 JSON 对象**。

结果：

- ✅ **导航被触发**：日志 `No stage template available, using image-based OCR for 0-2`
- ✅ **零代理痕迹**：无 `UsePrts`/`PRTS1`/`PrtsErrorConfirm`（护栏确认干净）
- ✅ 走的是 `Copilot@FullStageNavigation`（Copilot 自己的导航任务，非 Fight）
- ❌ **卡死**：`FullStageNavigation` 反复重试，OCR 读到的全是主界面文本
  （`2026/10/091721` 时间、`431` 理智、`437790` 龙门币、`868/199`、`ID434541074`）

## 根因：Copilot 的导航**只管地图内滑动，没有"从主页进入章节地图"**

`MultiCopilotTaskPlugin::navigate_to_stage` 的 OCR 分支（源码）：

1. `find_stage()` —— OCR 找关名；
2. `Copilot@FullStageNavigation` —— **右滑 10 次**（`ChapterSwipeToTheRight`）；
3. `Copilot@StageNavigationSlowlySwipeLeft` —— **左滑**（反复 `m_max_retry` 次）；
4. 找到即 `enter_stage()`。

**全部是"在关卡地图上滑动找关卡"，不含任何"进入终端→曲谱→选章节"的步骤。**
故在主页时必然卡死。

对比 `Fight` 的 `StageNavigationTask`：

| 能力 | `Fight`（StageNavigationTask） | `Copilot`（navigate_to_stage） |
|---|---|---|
| **章节寻路**（主页→章节地图） | ✅ `m_chapter_task = Episode{N}`（`chapter_wayfinding()`） | ❌ **无** |
| 地图内滑动找关卡 | ✅ `swipe_and_find_stage()` | ✅ 同样机制 |
| 快路径 | ✅ `LastBattleStageName`（上次作战） | ❌ 无 |

即：**`Fight` = 章节寻路 + 地图内找关卡；`Copilot` = 仅地图内找关卡**。
两者互补，不可互相替代。

## 为什么 `Fight` 不能直接用（回顾 ADR-0003）

`Fight` 虽能完成完整导航，但 `FightBegin` 的 next 顺序是：

```json
"FightBegin": {"next": ["Fight@UsePrts-Annihilation", "Fight@UsePrts",
                        "Fight@UsePrts-StageSN", "Fight@StartButton1", "Fight@PRTS#next"]}
```

**`UsePrts` 在 `StartButton1` 之前** → 只要该关有代理记录，必然先点代理。
实测 `Fight@PRTS1` 出现 13 次并全部失败（号商代理为异常数据）。

## 关键新发现：`StartButton1` 的次数受 `times` 参数控制

```cpp
m_fight_task_ptr->set_times_limit("StartButton1", times)
    .set_times_limit("StartButton2", times);
```

但 **`times=0` 的语义是"禁用整个 Fight 任务"**（含导航）：

```cpp
if (times == 0) {
    m_start_up_task_ptr->set_enable(false);
    m_stage_navigation_task_ptr->set_enable(false);   // ← 导航也被禁
    m_fight_task_ptr->set_enable(false);
    m_sidestory_reopen_task_ptr->set_enable(false);
}
```

故 `times=0` **不能**实现"只导航不开打"。
（`UsePrts` 不受 `times` 约束，它只受 `PRTS1/2/3` 的 limit 影响，
而 `FightTask` 已把它们设为 0 —— 但 `UsePrts` 本身仍可点击，这就是漏洞所在。）

## 候选方案（待用户裁决）

### 方案 A：给 MaaCore 加"纯导航"能力（补丁 0002）

仿照现有 `patches/0001`（导出战场状态）的做法，在 `FightTask::set_params` 里
增加一个参数 `nav_only: true`：导航完成后**不执行** `m_fight_task_ptr`，
直接以成功结束。

- **优**：最干净——用 `Fight` 完整的章节寻路能力，且绝不触碰代理/开战；
  与现有补丁机制一致（已验证可行）；对上游是纯新增、可回滚。
- **劣**：需重新编译 MaaCore（约 5-10 分钟，流程已验证）；
  需同步维护 patch 与 UPSTREAM.json 说明。
- **风险**：低。补丁只加参数判断，不改现有行为（默认 false 时完全等价）。

### 方案 B：`Fight` 导航 + 检测 `UsePrts` 立即停

保留两段式，靠 `prts_detected` 护栏兜底剔除被污染样本。

- **优**：无需改 MAA。
- **劣**：**必然**产生代理污染窗口（`UsePrts` 一定被点到）；
  代理作战一旦开始，该局数据作废，浪费机器时间；
  且"点代理"本身对账号可能有副作用。
- **风险**：中。数据纯净度无法保证，只靠事后剔除。

### 方案 C：让 Copilot 先到章节地图，再交给它的导航

用 `StartUp` 任务链推进到"终端→曲谱→章节地图"，然后交给 `copilot_list`。

- **优**：无需改 MAA。
- **劣**：`StartUp` 任务不含章节选择逻辑（它只到主页）；
  需要自建"点终端→点曲谱→选章节"的点位操作——**正是 ADR-0002 废弃的
  自建几何导航**，会重新引入逐关标定问题。
- **风险**：高。回到已被证否的路子。

### 方案 D：改用 `Fight` 但接受代理（用户 2026-10-09 第 3 点授权）

用户明确"现在可以大方用 MAA，等前后端连起来再考虑停止"。

- **优**：立刻可用；符合用户当前授权。
- **劣**：与用户第 1 点"代理数据无价值"冲突——需要区分：
  用户授权的是"大方用 MAA"，但代理作战数据**仍是污染的**。
- **待澄清**：用户是否接受"Fight 走代理但把该局标记剔除"？

## 建议

**方案 A** 最符合所有约束（干净、可扩展、与既有补丁机制一致）。
其次若不想动 MAA，**方案 D + 严格剔除**可作为过渡（需用户确认接受剔除损耗）。

方案 B 不建议（明知会污染还去做）。
方案 C 不建议（回到已废弃的自建导航）。
