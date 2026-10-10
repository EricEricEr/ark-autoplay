# Patch 0005：BattleState 导出前刷新 costs/kills + 等待失败分支补快照

上游 pin：`a82a0e13c68d48cc21ec40380ac08c95f46e9802`（dev-v2，见 `maacore/UPSTREAM.json`；许可证 AGPL-3.0-only）。

> **非独立可应用**：配套的 `0005-battlestate-export-refresh.patch` 是用
> `git -C bridge/maacore/_src diff -- src/MaaCore/Task/Miscellaneous/BattleProcessTask.cpp`
> 生成的**工作树累积 diff**，相对 HEAD 已包含 patch 0001 在同文件内的改动
> （`notify_battle_state()` 的定义与 `do_action()` 中间那处调用）。
> 因此它**不能**单独 `git apply` 到 pristine 上游（缺 0001 上下文）；正确用法是
> 与 0001→0002→0003 按序叠加，或直接用工作树。

## 解决什么问题

patch 0001 已经把战场状态以 `BattleState` 回调导出，但实测 `costs`/`kills` 恒 0，
且一局只拿到最早几条。两个独立缺陷：

### (1) 导出点不刷新成员 → costs 恒 0、kills 停留在旧值

`notify_battle_state()` 只读成员变量（`m_cost` / `m_kills` / `m_cur_deployment_opers`），
自己不做识别；而刷新这些成员的 `update_kills()`/`update_cost()` 只在
`wait_condition()` 内部**特定条件分支**里被调用（作业动作没有声明 kills/costs 条件时不会走到）。
于是导出的是陈旧值——costs 恒 0。

**探针证据（T11）**：`bridge/debug/probe_battlestate.jsonl` 共 **407 行**，
其中 `"fn":"update_kills"` **407 行**，`"fn":"update_cost"` **0 行**——
即本局 `update_cost()` 一次都没被调用过。这同时排除了"调用了但识别失败"：
它根本没被调用（探针实现见 patch 0003）。

### (2) 等待失败/战斗结束分支不 notify → 战果永远缺席

`do_action()` 里 `if (!wait_condition(action)) { ... return false; }` 直接返回，
不发送快照。最后一次"长时间等待"（战斗结束前的那段）正是击杀累计完成、最能反映战果的时刻，
它**永不 notify**。

**证据（T11）**：同一批实测（`docs/adr/0007-fight-rejects-activity-stages.md` 复验段）
一局 PR-A-1 只产出 **4 条 BattleState**（与 4 个 CopilotAction 一一对应），
且这 4 条**全部早于首次击杀识别**——结尾状态没有任何一条被导出。

## 改动

唯一改动文件：`src/MaaCore/Task/Miscellaneous/BattleProcessTask.cpp`（其余 6 个 modified 文件属 0001/0002/0003，不在本补丁范围）。
累计 diff：`1 file changed, 30 insertions(+), 1 deletion(-)`。

### 1) `notify_battle_state()` 开头无条件刷新（`[PATCH 0005]`）

```cpp
void asst::BattleProcessTask::notify_battle_state()
{
    // [PATCH 0005] 导出前无条件刷新一次：catch 行动作未声明 kills/costs 条件时，
    // wait_condition() 不会调用 update_kills()/update_cost()，导致导出恒 0。
    if (const cv::Mat image = ctrler()->get_image(); !image.empty()) {
        update_kills(image);
        update_cost(image);
    }

    json::value info = basic_info_with_what("BattleState");
    info["details"] = battle_state_json();
    callback(AsstMsg::SubTaskExtraInfo, info);
}
```

### 2) `do_action()` 的等待失败分支补一条快照（`[PATCH 0005]`）

**原文（改后）**：

```cpp
    if (!wait_condition(action)) {
        // [PATCH 0005] 战斗结束/等待失败时原逻辑直接 return，最后一个长时间等待动作
        // 的战场状态（kills 已累计）永不导出，故此处补一条快照。
        notify_battle_state();
        return false;
    }
```

（patch 0001 的原文注释保留在函数定义处与动作执行前调用处，本补丁未改动它们；
上文两段中文注释均自工作树逐字复制。）

## 为什么安全

1. `update_kills()`/`update_cost()` 只读**一帧**图像并更新成员 + 日志，不参与任何识别/决策分支；
   `get_image()` 返回空则整块跳过，此时行为与旧版逐字节等价。
2. 失败分支仍 `return false`，返回值语义未变，只多发一次 `SubTaskExtraInfo` 回调；
   回调消费者（bridge 的 battle_state/episode_writer）对额外一条快照无副作用。
3. 开销：每次 `do_action()` 多一次取帧 + 两次识别。`wait_condition()` 在同一路径本来就要取帧，
   相对整局（分钟级）耗时增量可忽略；也不会改变作业时间轴（不新增 sleep）。

## 验收（本次实测）

| 项 | 结果 |
|---|---|
| Release 构建（VS2022，`MaaCore.vcxproj`，vcvars64 环境） | exit 0 |
| 部署 | `E:\dev\maa_runtime_patched\MaaCore.dll`，6,268,928 B，2026-10-10 12:51:37，SHA256 `F3FA4856EF4B7544ACC69445DFADC735A6E19D15AA9C843D947B35D2DA707F20` |
| 部署顺序一致性 | 该 DLL 时间戳（12:51:37）晚于被改源码 mtime（12:48:20），与"改动后重编并部署"一致 |
| 回滚点 | `MaaCore.dll.bak-20261010-pre0005`、`MaaCore.dll.bak-20261010-patch0005`（均 2026-10-10 10:38:10） |
| 真机复验（BattleState 是否不再恒 0、结尾快照是否出现） | **通过（3 局 3/3）**——4-1 连跑 3 局，每局 5 条 BattleState 全部非零、末条即 0005 新增的战斗结束快照 |
| 探针（改动前） | 407 行中 `update_cost` **0 行** |
| 探针（改动后，3 局） | run1 167 行、run2 210 行、run3 269 行（`probe_battlestate_0005.jsonl` / `_run2` / `_run3`）；**每局 `update_cost` 均 5 行**（0005 前恒 0） |

真机复验：4-1 **连跑 3 局**（同一 queue、同一账号）。episode 根目录：
`E:\Users\35151\Desktop\zcode工作区\有可能相关的模型？\ark-data\episodes\main_04-01\`
（结论由 device-runner 提供；patch-docs 已独立逐行复核 3 局的 `battle_states.jsonl` 与 3 个探针文件计数，数值与结论一致。）

`battle_states.jsonl` 逐行 `(t_ms\|costs\|kills)`，后缀 T/F = `in_battle`：

| run | episode（目录名） | 5 条快照逐行 | probe 行数 / `update_cost` | 结果 |
|---|---|---|---|---|
| 1 | `job-22235-20261010-181632` | 78427\|11\|0(T)、82657\|21\|0(T)、89978\|13\|1(T)、94334\|8\|3(T)、164608\|8\|50(F) | 167 / 5 | 通过 |
| 2 | `job-22235-20261010-184707` | 48152\|10\|0(T)、52551\|21\|0(T)、59719\|14\|1(T)、63992\|8\|3(T)、149632\|8\|57(F) | 210 / 5 | 通过 |
| 3 | `job-22235-20261010-185113` | 51634\|10\|0(T)、55724\|20\|0(T)、62805\|13\|1(T)、67041\|7\|3(T)、151702\|7\|56(F) | 269 / 5 | 通过 |

- **3/3 通过**，且三局末条（第 5 条）都是 0005 新增的"战斗结束补快照"（`in_battle=false`）：`kills` **末值非零且各不相同**
  （**50 / 57 / 56**，非固定值 → 不是常量兜底）；`costs` **全程非零**（7–21）；`kills` 每局单调不减（如 run2：0→0→1→3→57）；
- 每局 `update_cost` 均为 **5 行**（0005 前为 0 行），`update_kills` 分别为 162 / 205 / 264 行；
- 这条结束快照正是改动前**永不导出**的那条（此前同批实测只有 4 条，且全部早于首次击杀识别）。

**未验证 / 边界**：
1. **仅 4-1 单关卡**：3 局是同一 queue 的同一关卡连跑，未做更长时长、更多关卡/多实例的稳定性复验；
2. **未逐条核对每个 `update_cost` 事件**（只做了行数与导出数值层面的复核，本文亦然）；
3. 本任务未重新编译（构建结论引自构建线记录，部署 DLL 时间戳只作一致性旁证）；
4. `update_cost` 的 5 次调用是否覆盖全部该刷新的动作时机，未做对照实验。

## 构建要点

- 必须在 **vcvars64**（VS2022 x64）环境中调用 msbuild，否则基座工具集/Windows SDK 不可见；
- 传 `TrackFileAccess=false`（本次可用的构建参数组合）；
- 若链接报 **LNK1136（无效或损坏的文件）**，清理对应中间 `.obj` 后重建（本次为 `AsstCaller.obj`）。

## 许可证

对 MaaAssistantArknights（AGPL-3.0-only）的修改，同样以 AGPL-3.0 发布。上游 commit 见 `maacore/UPSTREAM.json`。
