# Patch 0002：FightTask 支持 `nav_only`（只导航、不开打）

## 解决什么问题

数据工厂需要把游戏**从主页带到目标关卡的准备界面**，但 MAA 现有能力无法
干净地做到：

| 路径 | 能力 | 问题 |
|---|---|---|
| `Copilot` 的 `navigate_to_stage` | 仅**地图内**滑动找关卡 | ❌ 缺"主页→章节地图"的章节寻路，在主页会卡死 |
| `Fight` 的 `StageNavigationTask` | ✅ 完整章节寻路（`Episode{N}`）+ 地图内找关卡 | ❌ 导航后会**点代理作战**（`UsePrts`）并开打 |

关键卡点：`FightBegin` 的 next 顺序是

```json
["Fight@UsePrts-Annihilation", "Fight@UsePrts", "Fight@UsePrts-StageSN",
 "Fight@StartButton1", "Fight@PRTS#next"]
```

**`UsePrts` 排在 `StartButton1` 之前** → 只要该关有游戏内代理记录，`Fight`
必然先点代理。而自抽号的代理记录是号商机械刷出的**异常数值**（用户明确：
对正常账号毫无价值），实测其连自己都跑不通
（`Fight@PRTS1` 13 次 → `PrtsErrorConfirm` 4 次 → `FightMissionFailed` 73 次）。

## 补丁做什么

给 `FightTask::set_params` 增加一个**可选参数** `nav_only`（默认 `false`）：

```cpp
const bool nav_only = params.get("nav_only", false);   // 读参数

// ... 在 return true 之前 ...
if (nav_only) {
    m_fight_task_ptr->set_enable(false);            // 不作战
    m_sidestory_reopen_task_ptr->set_enable(false); // 不走复刻流程
    m_stage_drops_plugin_ptr->set_enable(false);    // 不上报关卡掉落
    m_fight_times_prt->set_enable(false);           // 不做连战计数
}
```

下发时：

```python
append_task("Fight", {"stage": "0-2", "nav_only": True})
```

导航子任务（`m_start_up_task_ptr` + `m_stage_navigation_task_ptr`）照常运行到
准备界面，然后**干净收尾**——之后交给 `Copilot` 执行玩家作业。

## 为什么可行（源码依据）

`PackageTask::run` 对 **disabled 的子任务是 `continue` 跳过**，整体仍返回 true：

```cpp
auto task_ptr = m_subtasks.at(i);
if (!task_ptr->get_enable()) {
    continue;          // ← 跳过，不影响整体成败
}
```

`FightTask` 的子任务装配（`FightTask.cpp:59-62`，顺序即执行顺序）：

```cpp
m_subtasks.emplace_back(m_start_up_task_ptr);         // 1. 启动到主页
m_subtasks.emplace_back(m_stage_navigation_task_ptr); // 2. 章节寻路 + 找关卡 ← 保留
m_subtasks.emplace_back(m_fight_task_ptr);            // 3. 作战 ← 禁用
m_subtasks.emplace_back(m_sidestory_reopen_task_ptr); // 4. 复刻 ← 禁用
```

## 为什么安全

1. **不改任何既有行为**：`nav_only` 默认 `false`，不传时与原版**完全等价**；
2. `m_fight_task_ptr` 在 `StageNavigationTask` 中只被一处引用
   （`StageNavigationTask.cpp:157`，剿灭 `Annihilation@UnableToAgent2` 分支），
   禁用**不影响**普通关卡的章节寻路与地图内找关卡；
3. 纯新增判断，无删除、无改动既有分支。

## 应用方式

```bash
cd bridge/maacore/_src
git apply ../../patches/0002-fight-nav-only.patch
```

已验证：`git apply --check` 对 pristine 源码 **exit 0**；
补丁 1616 B、32 行、**纯 LF 无 BOM**（PowerShell 的 `Set-Content`/`Out-File`
会引入 BOM 导致 `git apply` 失败，故由 `make_patch_0002.py` 用 Python 写出）。

## 与 patch 0001 的关系

两个补丁**互相独立**，可单独或同时应用：

| 补丁 | 作用 | 文件 |
|---|---|---|
| 0001 | 导出战场状态（`BattleState` 回调） | `BattleTask`/`BattleHelper` 等 4 文件 |
| 0002 | `Fight` 支持只导航不开打（`nav_only`） | `FightTask.cpp` 1 文件 |

同时应用时，`nav_only=True` 的导航**不会**产生 `BattleState`（因为不进入作战），
数据工厂的实际状态采集由随后的 `Copilot` 任务负责。

## 许可证

本补丁是对 MaaAssistantArknights（AGPL-3.0）的修改，同样以 **AGPL-3.0** 发布。
上游 commit 见 `maacore/UPSTREAM.json`。
