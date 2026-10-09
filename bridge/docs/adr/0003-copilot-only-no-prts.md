# ADR-0003：数据工厂只用 Copilot 路径，禁用 Fight（代理作战污染）

- 状态：已接受（2026-10-09）
- 取代：ADR-0002（"用 MAA 原生任务做导航"中采用 `Fight` 的部分）
- 关联：`nav_maa.py`、`replay_controller.py`、`gen_queue.py`

## 背景与纠错

ADR-0002 决定"导航交给 MAA 的 `Fight` 任务"，理由是它自带关卡导航、每步带
template score。**该决策在 2026-10-09 被用户抽查推翻**：

> 用户抽查模拟器运行时，发现 MAA 正在对 0-1 执行**游戏内代理作战**。

核查日志确认属实：

| 观察到的任务 | 次数 | 含义 |
|---|---|---|
| `Fight@PRTS1` | **13** | **确实进入了代理作战** |
| `Fight@UsePrts` | 68 | 反复尝试点"代理指挥"按钮 |
| `Fight@PrtsErrorConfirm` | 4 | 代理报错弹窗 |
| `Fight@ClickCornerAfterPRTS` | 2 | 代理后点角落 |
| `Fight@FightMissionFailed` | 73 | 作战失败 |

### 机制（源码依据）

`resource/tasks/tasks.json` 中：

```json
"UsePrts": {
  "action": "ClickSelf",
  "roi": [1000, 550, 280, 90],
  "maxTimes": 10,
  "next": ["#self", "UsePrts-Annihilation", "StartButton1"]
}
```

`next` 含 `StartButton1` —— 即**有可用代理就点代理直接开打，没有才走正常流程**。
而 `FightTask.cpp` 只把 `PRTS1/2/3` 的 `times_limit` 设为 0，**并未禁掉 `UsePrts`**。

### 为什么代理数据必须排除（用户说明）

> 自抽号的游戏内代理记录是号商机械刷出来的，数值是非正常数值，
> 对于正常账号没有任何价值。

实测佐证：`PrtsErrorConfirm` 4 次 + `FightMissionFailed` 73 次——
**号商的代理记录连自己都跑不通**。

## 决策

**数据工厂一律走 `Copilot` 路径，不再使用 `Fight` 做导航或作战。**

两条路径的关键差异（源码对照）：

| | `Fight` | `Copilot` |
|---|---|---|
| 起手 task | `FightBegin` | **`BattleStartPre`** |
| 代理作战 | ❌ 会走 `UsePrts` | ✅ 显式 `NotUsePrts`（`MultiCopilotTaskPlugin.cpp:59`） |
| 关卡导航 | `StageNavigationTask` | **自带**：`navigate_to_stage()` 模板匹配，无模板则图像 OCR |
| 动作来源 | MAA 自身逻辑 | **prts.plus 玩家作业** |

因此 **Copilot 一步到位**（导航 + 执行作业），
原先"Fight 导航到 briefing → 抢停 → Copilot 接管"的两段式**整体取消**——
它存在的唯一理由是"Fight 不会自己打"，而事实相反。

## 配套改动

- `replay_controller`：删除预导航阶段与抢停逻辑，直接 `append_copilot` → 执行；
- `nav_maa`：原 `MaaNativeNavigator`（Fight 导航）替换为 `CopilotRunner`；
- **代理护栏**：新增 `PRTS_TASKS` 常量与运行时检测，一旦观察到
  `UsePrts`/`PRTS1`/`PrtsErrorConfirm` 等即写入 episode 的 `prts_detected` 字段
  （正常为空列表），**下游可按此字段直接剔除被污染样本**，无需重跑或翻日志；
- `gen_queue` 增加 MAA 可导航性过滤（见 ADR-0004）。

## 数据来源定位（用户 2026-10-09 澄清，勿再误解）

曾错误表述"prts.plus 数据是 MAA 策略数据、无价值"。**正确理解**：

- prts.plus 作业是**玩家先手动打通、再转成作业分享**的，有真实价值，
  其中不乏质量很高的作业；
- 本项目用 MAA 只是**忠实回放**这些作业，MAA 自身不产出策略数据；
- 主线关卡 MAA **不会**用自有策略去打。

即：**代理作战（游戏内 PRTS）要排除，玩家作业（prts.plus）要珍视**——
两者是完全不同的东西。

## 后果

- **正**：数据来源纯净（玩家作业）；彻底避开代理污染；少一段易碎的抢停逻辑。
- **负 / 边界**：Copilot 导航依赖 MAA 的模板/OCR，未收录的关卡可能导航失败
  （如实记失败，不静默跳过）。
- **待办**：端到端验证（模拟器当时已关闭，未完成）。
