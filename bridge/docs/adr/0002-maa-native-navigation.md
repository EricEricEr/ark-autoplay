# ADR-0002：用 MAA 原生任务做导航，废弃自建几何导航器

- 状态：**部分被取代**（2026-10-09）——"导航交给 MAA"的方向**正确**且已采纳；
  但本文选定的载体 `Fight` **已被证否**（它会走游戏内代理作战），现改用 `Copilot`。
  见 **ADR-0003**。
- 范围：`bridge/src/bridge/navigator.py`（已废弃）、`nav_maa.py`（已重构为 CopilotRunner）
- 关联：ADR-0001（纯 Python 驱动）、**ADR-0003（Copilot-only，取代本文的 Fight 方案）**

> ⚠️ **阅读提示**：以下内容保留作为决策过程记录。其中关于 `Fight` 的实测数据
> （24.7s 到位、每步 score 达 0.999~1.000 等）**仍然成立**，但结论
> "用 Fight 做导航"**已作废**——后续发现 `Fight` 的 `UsePrts` 分支会触发
> 游戏内代理作战（实测 `Fight@PRTS1` 出现 13 次），而自抽号的代理记录是号商
> 机械刷出的异常数值、对训练无价值。最终方案见 ADR-0003。

## 背景

### 问题

数据工厂原用**自建几何导航器**（`navigator.py`：固定点位 + 钳位滑动 + 蓝钮像素校验）
把游戏导航到目标关卡的 briefing。该项目在真机联调中**反复失败**：

- 0-2 实测报 `节点点击未打开 briefing: 0-2`（本轮复现）；
- 交接文档记录其为 bridge v1 的**卡点**，用户当时明确指示：
  **"路径导航不是必需品；禁止再碰自建几何导航"**。

根因：地图中端滚动有 ±140px 惯性漂移，固定点位不可靠；而两端钳位死端只覆盖
部分关卡，扩展需逐关标定。

### 转机

用户在本次会话中指出：**"可以先用 MAA 的识别能力"**。实测证实这是对的。

## 实测证据（2026-10-09，真机 MuMu 12 + patched MaaCore）

### 1. MAA 的 `StartUp` 任务能自己识别界面并导航

```
[0.31s] TaskChainStart
[2.72s] CloseAnno            score=0.954   ← 识别到公告弹窗并关闭
[5.42s] StartUp              score=0.993   ← 识别已到主界面
[12.05s] TaskChainCompleted
```

第二次运行（界面已稳定）**仅 2.6 秒**：

```
[0.391s] StartAtHome   score=0.992   ← 单次识别 813ms
[2.610s] Home (action=Stop)
[2.610s] TaskChainCompleted           ← 总耗时 2.6s
```

→ **单次界面识别 0.8s，符合 MAA 正常水平（<3s）。**
（此前一次耗时 36s，原因是**游戏自身仍在加载**、画面反复变化，MAA 在等界面稳定——
不是 MAA 慢。）

### 2. MAA 的 `Fight` 任务**自带完整导航**

下发 `append_task("Fight", {"stage": "0-2", ...})` 后，MAA 自主完成：

| 时间 | 任务 | score |
|---|---|---|
| 0.16s | StageBegin | |
| 0.91s | Fight | 0.992 |
| 1.66s | TerminalDefault | 0.998 |
| 6.62s | GoLastBattle | |
| 9.94s | Episode0 | |
| 12.05s | ClickChapterNew | 1.000 |
| 16.11s | EnterEpisodeNew | 0.996 |
| **19.42s** | **ClickStageName** | **0.999** |
| **20.17s** | **ClickedCorrectStageOrSwipe** | **1.000** |
| 22.58s | FightBegin | |
| **24.69s** | **StartButton1** | **1.000** ← 准备界面"开始行动" |

**24.7 秒导航到位，且 `StartButton1` 命中 score=1.000**；自建导航器在同一关卡
直接失败。

### 3. 可在准备界面精确抢停

在检测到 `StartButton1` 时调用 `AsstStop` → **抢停成功**（事后 `StartUp` 5.5s
完成，证明未进入作战、理智未消耗）。

> 注：抢停后 `Fight` 任务链不会正常收尾（挂到超时），因为 `AsstStop` 是粗粒度
> 中断。这不影响"导航 + 截停"的用途，但**不能沿用 `wait_chain` 的终态判定**，
> 需按 `StartButton1` 出现即视为到位。

## 决策

**导航一律交给 MAA 原生任务，删除自建几何导航。**

采用"**Fight 导航 → 到准备界面截停 → Copilot 接管**"两段式（即 ADR-0001 中
记录的"备选方案 B"，当时因"抢停时点竞争"未采用；现实测抢停可靠，故启用）：

```
1. append_task("Fight", {"stage": code, "times": 1, "medicine": 0, "stone": 0})
2. 监听事件流，出现 StartButton1（或 BattleQuickFormation）→ AsstStop
3. append_task("Copilot", {...作业...}) → start
```

配套改动：

- `maa_driver` 新增通用 `append_task(task_type, params)`（原先只有 `append_copilot`）；
- `navigator.py` 的几何导航逻辑废弃；`nav_main.yaml` 的关卡标定表不再需要；
- 抢停判定按**任务名**（`StartButton1`）而非像素，因为 MAA 已给出 score 与任务语义。

## 后果

- **正**：
  - 导航鲁棒性由 MAA 社区保障（模板库覆盖全部关卡，无需逐关标定）；
  - 删除数百行易碎的自建几何代码与手工标定表；
  - 每个识别步骤都带 `score`，**可量化、可监控**，远优于像素阈值；
  - 用户此前的指示（"禁止再碰自建几何导航"）与"MAA 识别能力优先"同时满足。
- **负 / 已知边界**：
  - 抢停是粗粒度中断，`Fight` 链不保证正常收尾 → 到位判定改用任务名；
  - `Fight` 会先尝试"上次作战"（`GoLastBattle`）等路径，比专用导航多几步，
    但总耗时仍 <30s；
  - 若 `Fight` 的默认行为（自动编队、理智管理等）造成副作用，须用参数约束
    （`times: 1`、`medicine: 0`、`stone: 0`）——本 ADR 的实测即用此约束。
- **待办**：
  - 把两段式导航封装为 `bridge` 的正式流程并替换 `navigator`；
  - 用该流程跑通一局并验证 `BattleState` 落盘（P0-2 的下一步）。
