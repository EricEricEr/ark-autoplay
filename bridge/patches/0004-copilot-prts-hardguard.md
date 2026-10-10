# Patch 0004：Copilot 开打前「代理指挥」硬护栏（nav_only times_limit 拦截 + 详情页勾选复核）

上游 pin：`a82a0e13c68d48cc21ec40380ac08c95f46e9802`（dev-v2，见 `maacore/UPSTREAM.json`；许可证 AGPL-3.0-only）。

> **非独立可应用**：本补丁依赖 patch 0002 引入的 `nav_only` 代码块（A 段改的正是那个分支）。
> 对 pristine 单独 `git apply` **不适用**；已实测：对 pristine+0001+0002+0003 叠加时
> `git apply` exit 0，且应用结果与工作树**逐字节一致**（源：`t15_patcher_worklog.md` §3）。

## 解决什么问题

导航到关卡详情页并交 Copilot 执行作业时，游戏自带的「代理指挥」会顶替玩家作业开打
（数据工厂拿到的就不是玩家作业的轨迹）。补丁 0003 已在简报页加"判定 → 点掉 → 复验"的护栏，
但仍有两个缺口，本补丁补齐为**开打前硬护栏**：

### (1) nav_only 导航链仍会点到 UsePrts（上游漏配次数限制）

上游只对 `StartButton1` / `PRTS*` 设了 0 次限制（`FightTask.cpp:29-37`），**漏了 `UsePrts`**。
实测 asst.log：nav_only 期间 `Fight@UsePrts` 仍被选中 **7 次**、`UsePrts.png` 命中 **21 次（score 0.967）**
——点击来自 `StageBegin→Fight→FightBegin` 这条 next 链（即 `m_start_up_task_ptr`）,
即"导航"过程本身就可能在详情页把代理开关点开。`times_limit` 按"最长后缀"匹配
（`ProcessTask.cpp:488-490`），所以 `"UsePrts"` 同时覆盖 `Fight@UsePrts`。

### (2) 详情页勾选框：账号自身状态就可能是 ON

即使导航没点，账号本来的代理开关可能就是开的。且该勾选框**只在点「开始行动」之前可见**，
一旦开打就再也够不到——所以复核必须放在 Copilot 的**第一个**子任务。
本次真机实测（`t15_result.md` ①）：A 局详情页 `NotUsePrts` = **0.915**（勾选框 ON，白色实心框），
说明**账号原状态就是代理 ON**，护栏确实拦下了。

### (3) 判定不出时不能静默

页面异常（`UsePrts` 与 `NotUsePrts` 都不命中）时保留继续语义但必须**可见**（回调 + LogWarn）；
点了仍开启则中止本局，不能静默带着代理开打。

## 改动

4 个文件，共 **+29 行**（`CopilotTask.cpp` +8、`FightTask.cpp` +11、`MultiCopilotTaskPlugin.cpp` +6、`.h` +4）。

### A. `FightTask.cpp`：nav_only 分支禁用 UsePrts

在 `set_params()` 的 `if (nav_only)` 分支内插入：

```cpp
        // [PATCH 0004] 导航链绝不点开"代理指挥"勾选框。
        // 实测（asst.log）：nav_only 下 Fight@UsePrts 仍被选中 7 次、UsePrts.png
        // 命中 21 次（score 0.967）——点击来自 StageBegin→Fight→FightBegin 这条
        // next 链（m_start_up_task_ptr）；上游只对 StartButton1/PRTS* 设了 0 次限制，
        // **漏了 UsePrts**。times_limit 按"最长后缀"匹配（ProcessTask.cpp:488），
        // 故 "UsePrts" 同时覆盖 Fight@UsePrts；"-StageSN"/"-Annihilation" 是独立后缀。
        for (const auto& chain : { m_start_up_task_ptr, m_fight_task_ptr }) {
            chain->set_times_limit("UsePrts", 0)
                .set_times_limit("UsePrts-StageSN", 0)
                .set_times_limit("UsePrts-Annihilation", 0);
        }
```

### B. `CopilotTask.cpp`：护栏作为第一个子任务

```cpp
    // [PATCH 0004] 红线护栏：本局开打前确认"代理指挥"为 OFF。
    // 必须排在**所有**子任务之前——关卡详情页的勾选框只在点「开始行动」之前可见，
    // 一旦开打就再也够不到。
    auto prts_guard_ptr = std::make_shared<MultiCopilotTaskPlugin>(callback, inst, TaskType);
    prts_guard_ptr->set_prts_guard_only(true);
    prts_guard_ptr->set_retry_times(0);
    m_subtasks.emplace_back(prts_guard_ptr);
```

### C. `MultiCopilotTaskPlugin`：护栏专用模式

`_run()` 开头提前返回，复用 patch 0003 的 `ensure_prts_disabled()`（判定 / 最多点 3 次 / 复验）：

```cpp
    // [PATCH 0004] 护栏专用模式：只做"代理指挥必须为 OFF"的判定/取消，返回失败即中止本局。
    if (m_prts_guard_only) {
        return ensure_prts_disabled();
    }
```

`.h` 中新增开关与成员：

```cpp
    /// [PATCH 0004] 护栏专用模式：_run() 只做代理指挥开关的判定/取消，不做导航与配置读取。
    void set_prts_guard_only(bool value) { m_prts_guard_only = value; }
```

```cpp
    bool m_prts_guard_only = false; // [PATCH 0004]
```

**红线事件语义**（复用 0003 的 `ensure_prts_disabled()`，见 `0003-nav-prts-guard-and-probe.md`）：
两者都不命中 → `CopilotPrtsState(unknown)` + `LogWarn`，保留继续执行；
点了仍开启 → `SubTaskError/CopilotPrtsToggleFailed` 并**中止本局**。

## 为什么安全

1. **A 的作用域仅限 nav_only**：只动 `if (nav_only)` 分支里的两个任务链，正常刷图路径不受影响；
   `"UsePrts"` 系列本来就是"不许点代理"的语义，设 0 次限制只可能更保守。
2. **B 复用 0003 已验证的逻辑**：未开启时**不点击任何东西**（与上游 `NotUsePrts` 不命中等价）；
   已开启时"点掉并复验"，严格优于上游"点一次不管结果"；并 `set_retry_times(0)`，不反复骚扰页面。
3. **C 隔离**：guard-only 实例 `_run()` 直接返回，不读配置、不导航、不开打，失败即中止本局——
   最坏情况是"本局不做作业"，而不是"带着代理开打产生错数据"。
4. **可回滚**：DLL 与源码均可一键回退到 0003 版本（见验收表"回滚"行）。

**已知风险**：A 会改变导航链的推进方式，理论上可能影响导航成功与否（见下"早期失败"一节的历史记录）。

## 验收（本次实测）

### 构建 / 部署 / 补丁

| 项 | 结果 |
|---|---|
| MSBuild `MaaCore.vcxproj` Release | exit 0（`t15_patcher_worklog.md` §3） |
| 产物 | `MaaCore.dll` 6268416 B @ 2026-10-10 10:38:10，sha256 `DDE794DC441801DC5BD7AAF7CCE7D20BE9EDB0C39AF8B19EA8CEB521BD77350B` |
| 部署 | `E:\dev\maa_runtime_patched\MaaCore.dll` |
| 回滚点 | `MaaCore.dll.bak-20261010-patch0003`（上一版 = 0003 的 DLL） |
| 导出面 | `Asst*` 28 = 28，无增无减 |
| 补丁可应用性 | 对 pristine+0001+0002+0003：`git apply` exit 0 且与工作树逐字节一致；**单独对 pristine 不适用**（依赖 0002 的 nav_only 块） |
| 本文件对应的 .patch | `0004-copilot-prts-hardguard.patch`，当前 4247 B（重生成后 29 insertions） |

### 真机（`--instance configs/instances/mumu_local_patched.yaml`）

| run | episode | copilot_actions | battle_states | 链 | 护栏回调 |
|---|---|---|---|---|---|
| A | `job-22235-20261010-105743` | 5 | 4 | Copilot TaskChainCompleted | `state_before:"on"→"off"`（`turned_off`） |
| B（复现） | `job-22235-20261010-111009` | 5 | 4 | Copilot TaskChainCompleted，无 TaskChainError | `state_before:"off"→"off"`（`use_prts_button_present`） |

① **详情页勾选框**：A 局简报帧 `NotUsePrts` = **0.915**（勾选框 ON，见 `t15_checkbox_crop.png` 白色实心框）
   → 护栏点掉并复验 OFF；B 局已是 OFF（`UsePrts` 按钮形态）→ 未点击。**账号原状态就是代理 ON，护栏确实拦下了。**
② **战斗中底部**：`BattleFlag/PRTS.png` NCC = **0.408 / 0.405 / 0.403**（3 帧战斗/结算，阈值 0.9；
   verifier"无代理"基线 0.378–0.466）+ 底部 150px 条带目视无「接管作战 / 代理指挥作战正常进行中」
   （`t15_bottom_strip_battle.png`）。
③ **任务链**：CopilotAction = 5 > 0（两局）。A 生效：nav_only 下无任何**新的** `UsePrts` 任务点击，
   而 `FightBegin` 候选链仍被遍历。

证据帧：`frames_keep/t15_1_detailpage.png`、`frames_keep/t15_2_battle_bottom.png`、
`frames_keep/t15_3_settle.png`、`frames_keep/t15_repeat_1..3_*.png`、`t15_checkbox_crop.png`、
`t15_bottom_strip_battle.png`。

## 早期失败（真机 D，历史）与根因更正

- **§4 真机 D：未通过**（`t15_patcher_worklog.md` 第 4 节）。命令
  `python -m bridge.replay_controller --work <ark-data> --jobs <ark-data>\configs\queue_t8_pilot.json --only 22235`（cwd=bridge，PYTHONPATH=src）。
  结果：`[job 22235] 导航未完成（4-1）` ok 0/1；episode=`job-22235-20261010-102810`，
  `copilot_actions=0`、`battle_states=0`、`prts_detected=[]`，未进入战斗 → 验收①②③均无法判定。
  仅证实一点：10:40 那次运行里**没有新的 `UsePrts.png` 命中**（累计仍是 21，最后一次在 10:08），
  而 `"cur_task":"FightBegin"` 候选链仍被遍历（asst.log:165736）→ **A 的拦截生效**。
  当时**无法判定**导航失败是否由 A 引起（也可能是设备停在非主页/非地图态）。
- **T15 根因（决定性，`t15_result.md`）**：红线来自**桥侧配置**——
  `bridge/configs/instances/mumu_local.yaml` 的 `maa_dir` 指向**官方未打补丁**的
  `MAA-v6.18.0-win-x64`，而 `replay_controller.py:585` 的 `--instance` 默认就是它
  → 从 ~09:45 起所有跑批都在用官方 MaaCore，**补丁 0001/0002/0003/0004 全部未被加载**
  （asst.log 10:40/10:49 有 `append_task Fight {"nav_only":true}` 但无 `nav_only enabled` 日志），
  Fight 走完整刷图链直接开打。改用 `mumu_local_patched.yaml` 后 A/B 两局通过。
- **控制组结论**：不再需要；先前"导航未完成"是**官方 DLL**造成，与 patch 0004 的 A 无关。
  **A（times_limit 拦 UsePrts）保留；B（开打前护栏）已实证有效。**

## 已知局限 / 未验证

- **样本量**：A/B 仅 2 局、同一 queue 作业，未做多关卡/多实例回归；A 对导航链的长期影响无大样本证据。
- **B 局设备收尾未闭环**：B 局 `settled=false`——11:14:27 `screencap` 报
  `device '127.0.0.1:16416' not found`，MAA 以 `TaskChainStopped` 收尾、未回简报页；
  跑完后 adb 连接被拒（MuMu 端口不可达/疑似模拟器已关），**主页状态需人工确认**。
- **costs/kills 仍为 0**：本批 `battle_states` 的 `has_costs/has_kills` 仍为 False，
  即该问题在真实局里仍然成立 → 由 patch 0005 处理（见 `0005-battlestate-export-refresh.md`）。
- **页面异常分支**：两者都不命中时仅 `CopilotPrtsState(unknown)` + `LogWarn`（保留继续语义，同 0003），
  即"无法判定"仍然不会阻止开打——这是刻意的兼容选择，不是已验证的安全保证。

## 构建要点

- 本次构建记录：MSBuild `MaaCore.vcxproj` Release → exit 0，产物 6268416 B
  sha256 `DDE794DC…BD77350B`，随后部署到 `maa_runtime_patched`（源：`t15_patcher_worklog.md` §3）。
- 回滚（一句命令）：`Copy-Item maa_runtime_patched\MaaCore.dll.bak-20261010-patch0003 maa_runtime_patched\MaaCore.dll`。
- 后续 patch 0005 的构建踩坑（vcvars64 环境、`TrackFileAccess=false`、清理损坏 obj 解 LNK1136）
  记录在 `0005-battlestate-export-refresh.md`，本补丁的构建日志未留参数明细，故此处不臆测。

## 许可证

对 MaaAssistantArknights（AGPL-3.0-only）的修改，同样以 AGPL-3.0 发布。上游 commit 见 `maacore/UPSTREAM.json`。
