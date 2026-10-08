# ADR 0001：纯 Python 驱动官方 MAA 发行版，不 fork MaaCore

- 状态：已采纳（v1WIP 实测验证）
- 日期：2026-10-08/09（开发机通宵验证）
- 决策人：数据工厂 bridge v1 实施

## 背景

bridge 需要让 MAA 自动重放 prts.plus 社区作业（Copilot 任务）并全程采集
轨迹。立项方案原设计为 fork MaaCore 到 `maacore/` 加导出补丁（识别结果
逐 tick 回调）。v1 立项验证时先实测了**官方 MAA 发行版自带的 Python
接口（`asst`）**能否直接满足数据工厂需求。

## 实测结论（开发机，MAA v6.18.0 + MuMu 12 @127.0.0.1:16416）

1. **asst 接口完整可用**：`Asst.load(path, user_dir)` → `Asst(callback)`
   → `connect(adb路径, 地址)` → `append_task("Copilot", {...})` → `start()`，
   连接、截图协商、任务链生命周期回调全部正常。
2. **Copilot 作业回放成立**：作业 JSON（prts.plus copilot 格式）经
   `append_task` 下发即自动编队、进战、按子弹时间循环执行动作。
   `stage_name` 用短码（`0-1`），不用数据集长码（`main_00-01`）。
3. **回调不含战场状态数值**：`SubTaskExtraInfo` 的 `CopilotAction` 给
   动作事件（Deploy/Skill/Retreat + target + 时点），但费用/击杀/技力
   不出现 → 状态采集必须自截屏（`state_logger` 节拍截图，OCR 标注 TODO）。
4. **任务链收尾≠作战结束**：`TaskChainCompleted` 在作业动作全部执行完
   即触发，作战继续空跑到敌方波次结束 → 结算推进与胜负判定需自己做
   （v1WIP：轮询"开始行动"蓝钮像素判回 briefing；结算帧一并落盘）。
5. **导航属游戏内 UI 自动化，与 MAA 无关**：Copilot 要求游戏停在目标关
   briefing 起跑（首步 OCR"开始行动"）。自研固定点位导航（地图钳位端
   锚定 + 节点坐标 + 蓝钮像素校验）实测稳定够用（v1 仅 0/1 章钳位关卡）。
   备选方案 B（MAA Fight 任务代导航 + 回调抢停 AsstStop）因抢停时点竞争
   未采用，留作 TODO 兜底。

## 决策

1. **不 fork MaaCore**：官方发行版的 `asst` 接口已覆盖数据工厂 v1 全部
   需求；`maacore/` 保持空目录，`patches/` 不动。仅当后续需要 asst 未暴露
   的识别数据（如逐 tick 费用/格子占用）时才重新评估 fork。
2. **协议级复用**：运行时 `add_dll_directory` + `sys.path.insert` 动态加载
   MAA 安装目录内容，**不复制、不修改**；目录路径入机器本地实例配置。
   由此 NOTICE.md 的"附加条款原文待粘贴"一节维持现状：无源码导入，无可
   粘贴对象；分发的只有调用约定与本仓自产代码（AGPL-3.0）。
3. **CLI 与配置分离**：一切机器相关路径（MAA 目录、adb、实例地址、数据
   落盘根）只出现在 `configs/` 与运行参数，代码零机器路径常量。

## 影响

- bridge 代码量小、无 C++ 构建链；CI 不需要 MAA（单测纯逻辑）。
- 运行时依赖官方 MAA 发行版（见 AGENTS.md「运行时依赖」）。
- 风险：asst 接口行为随 MAA 版本漂移 → 实例配置记录已验证版本
  （v6.18.0），升级 MAA 需重新冒烟验证。
