# Patch 0003：简报页代理作战开关护栏 + BattleState costs/kills 探针

上游 pin：`a82a0e13c68d48cc21ec40380ac08c95f46e9802`（dev-v2，见 `maacore/UPSTREAM.json`）。
本补丁**独立可应用**（已对 pristine 源码 `git apply --check` exit 0），也与 0001/0002 叠加应用（已验证）。

## 解决什么问题

### (1) 代理作战开关被"静默跳过"（nav_only 链路的关键缺口）

上游 `MultiCopilotTaskPlugin::_run()` 在导航到简报页后执行：

```cpp
ProcessTask(*this, { "NotUsePrts" }).set_ignore_error(true).set_retry_times(0).run();
```

三种截然不同的局面在这里**都是静默通过**：

| 局面 | 视觉（resource/tasks/tasks.json, roi [1000,550,280,90]） | 上游结果 |
|---|---|---|
| 代理未开启 | `UsePrts` 形态命中（实测 0.966），`NotUsePrts` 0.23 不命中 | 跳过（预期） |
| 代理**已开启**、点击未生效 | `NotUsePrts` 命中但状态未变 | **静默通过 → Copilot 在代理模式下开打**，游戏用内置代理顶替玩家作业（数据工厂拿到错轨迹） |
| 页面异常/模板不匹配 | 两者都不命中 | **静默通过** |

实测 asst.log 里 `NotUsePrts` 出现 0 次：既可能是"代理本就没开"（良性），也可能是"点了没生效"
（恶性）——上游代码无法区分。本补丁把该步骤改为**先判状态再动作、每次判定都上报**：

- 代理未开启（`UsePrts` 命中）→ 不点击，回调 `CopilotPrtsState`（off/off）；
- 两者都不命中 → 回调 `CopilotPrtsState`（unknown/unknown）+ `LogWarn`，**可见**但保留继续执行的语义
  （避免误伤没有该按钮的关卡）；
- 代理已开启（`NotUsePrts` 命中）→ 最多点 3 次并逐次复验；仍开启则回调
  `SubTaskError`/`CopilotPrtsToggleFailed` 并让本次 Copilot 失败（**失败可见、不静默**）。

改动：`MultiCopilotTaskPlugin.h` 增 1 行声明；`.cpp` 替换 1 行为 `ensure_prts_disabled()` 调用 + 新增实现。

### (2) costs/kills 恒 0 的归因探针

patch 0001 导出的 `BattleState(costs,kills)` 实测恒 0，需区分"没被调用"与"调用了但识别失败"。
本补丁在 `BattleHelper::update_cost()/update_kills()` 内部打点，逐行追加 JSON 到
**`debug/probe_battlestate.jsonl`**（相对进程 cwd；可用环境变量 `ARK_BATTLESTATE_PROBE` 覆盖为绝对路径，
`ARK_JOB_ID` 会写进每行 `job_id`）：

```json
{"t_ms":...,"fn":"update_cost","n_cost":1,"n_kills":0,"has_result":true,"status_id":0,
 "success":false,"costs":0,"kills":0,"total_kills":0,"has_prev":false,"job_id":""}
```

- `n_cost` / `n_kills`：两个函数各自的累计调用次数（含本次）——若 `n_cost` 始终为 0，
  说明本局根本没走到 `update_cost`（未进入作战/无 Copilot 战斗轮）；
- `has_result` / `status_id` / `success`：识别是否返回、MatchStatus 原始值、是否成功匹配；
- `costs`/`kills`/`total_kills`：函数出口时的成员值（即 patch 0001 会导出的值）。

只读既有成员与识别结果，不改动任何识别/决策逻辑；上限 20 万行，超出后只计数不再写盘。

## 为什么安全

1. 未开启代理时**不点击任何东西** → 与上游等价（上游的 `NotUsePrts` 也不会命中）；
2. 已开启时"点掉并复验"严格优于上游的"点一次不管结果"；
3. 状态未知时保留上游继续语义，只是把静默变成回调 + 日志；
4. 探针纯读，失败只写 `LogError` 一次。

## 验收（本次实测）

| 项 | 结果 |
|---|---|
| `git apply --check` 对 pristine | exit 0 |
| `git apply` 叠加在 0001 之后 | exit 0，且应用结果与工作树**逐字节一致**（3 文件） |
| Release 构建（VS2022, MaaCore.vcxproj） | exit 0 |
| `Asst*` 导出 | 28 个，与上一版已部署 patched DLL 完全一致（无增无减） |
| 部署 | `E:\dev\maa_runtime_patched\MaaCore.dll`（备份 `MaaCore.dll.bak-20261010-094538`） |
| 资源一致性 | runtime 与 _src 的 `version.json`/`tasks/tasks.json`/`config.json` sha256 全同 |

**未验证**：真机联调（按"设备单写者"纪律，本任务未跑模拟器）；`probe_battlestate.jsonl` 尚无真实生产数据。

## 许可证

对 MaaAssistantArknights（AGPL-3.0）的修改，同样以 AGPL-3.0 发布。上游 commit 见 `maacore/UPSTREAM.json`。
