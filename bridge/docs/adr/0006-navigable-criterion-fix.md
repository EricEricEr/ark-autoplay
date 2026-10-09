# ADR-0006：关卡可导航判据纠错——用地图数据（Tile-Pos）而非 Fight 的章节规则

- 状态：已接受（2026-10-09）
- 触发：用户质疑"全量 39,010 为什么 MAA 只能用 8,440？这都是从 MAA 作业网站下载的"
- 影响：`bridge/src/bridge/maa_stages.py`（重写）、`gen_queue.py`（换代过滤）、
  队列规模 **1,012 → 3,000 条**、覆盖关卡 **537 → 1,624**

## 错误是什么

`maa_stages.maa_navigable()` 用 **`Fight` 任务**的规则判"可导航"：

```cpp
// StageNavigationTask::set_stage_name
static const boost::regex stage_regex(R"(^([A-Za-z]{0,3})(\d{1,2})-(\d{1,2})(?:-?(\w+))*$)");
...
if (!Task.get(m_chapter_task)) {      // m_chapter_task = "Episode" + chapter
    Log.error("chapter task not exists", m_chapter_task);
    return false;
}
```

即要求**关卡名匹配正则**且**存在 `Episode{N}` 章节任务**。
用它过滤的结果是：活动关（占数据集 71%）几乎全被判死——
`GT-1` 不匹配 `\d-\d`，`SR-5` 虽在 task 表里但绝大多数活动关不在。

**结果：40,299 条作业里只剩 8,440 条（21%），活动关几乎全军覆没。**

## 为什么错

**两条路径的导航机制完全不同，我把 `Fight` 的规则套到了 `Copilot` 上。**

`Copilot` 的导航（`MultiCopilotTaskPlugin::navigate_to_stage`）：

1. `find_stage()` —— 全屏 OCR 找关名（用 `ClickStageName` 基础配置，
   其 `text` 是**空数组**，关名运行时才注入）；
2. 找不到 → 右滑 10 次再找（`Copilot@FullStageNavigation`）；
3. 再找不到 → 左滑反复扫最多 20 次；
4. 再不行 → 右滑检查初见剧情。

**无关卡白名单、无章节要求。** 而能否处理该关的真正判据在
`CopilotTask.cpp:109`（multi 分支）：

```cpp
const auto& map_data = Tile.find(stage_name);
if (!map_data || !json::open(map_data->second)) {
    return false;              // ← 关键是"有没有地图数据"
}
```

`Tile` 即 `TilePack`（`resource/Arknights-Tile-Pos/`）。

## 实测证据（Tile-Pos 覆盖率）

| 类型 | 覆盖 | |
|---|---|---|
| MAIN | 564/564 | 100% |
| SUB | 86/86 | 100% |
| CAMPAIGN | 36/36 | 100% |
| DAILY | 35/35 | 100% |
| GUIDE | 2/2 | 100% |
| **ACTIVITY** | **2152/2317** | **92.9%** |
| CLIMB_TOWER | 137/239 | 57.3% |
| **作业维度合计** | **40295/40299** | **100.0%** |

**几乎所有作业都有地图数据** —— 先前 21% 的估计是纯粹的误判。

## 决策

1. `maa_stages` 重写为两个**分开**的判据，并在 docstring 写明不可混用：
   - `has_map_data(stage_id, code, tile_keys)` —— **Copilot 判据（推荐）**，
     读 `Arknights-Tile-Pos`；
   - `fight_accepts(stage_code, tasks)` —— `Fight` 判据（保守），
     仅在用 `Fight` 导航时用。
2. `gen_queue` 的 `--nav-filter` 改用 `has_map_data`；
   实测过滤后 `no_map_data` 仅 **4 条**（对比旧规则误杀 23,149 条）。
3. 新增**干员可达过滤**（`--operbox` + `--max-missing`，默认 1 = 可借 1 助战）：
   按账号实采干员盒剔除凑不齐阵容的作业。

## 修正后的队列规模

```
读取 40299 条：干员不足 27526、可达 11333、超长 1436、无地图数据 4
→ 3000 条 / 1624 关 / 541 家族
类型：ACTIVITY 2062、MAIN 689、SUB 107、CAMPAIGN 72、DAILY 70
```

对比修正前：**1,012 条 / 537 关（几乎只有主线与少量活动）**。

## 教训

- **别把一条代码路径的约束套到另一条**。`Fight` 与 `Copilot` 名字都像"关卡导航"，
  实际机制（章节寻路 vs 地图内 OCR 扫描）与判据（`Episode{N}` vs 地图数据）
  完全不同。判定前必须找到**该路径**的源码依据。
- **用户的领域直觉值得当证据对待**。用户一句"这些作业都是从 MAA 作业网站
  下载的"直接指出了矛盾——社区作业若 79% 不可用，作业网站就没法运作。
  遇到反直觉结论时，先怀疑自己的判据，而不是先解释现象。
- 先前把"8,440 可导航"写进报告与文档，**本 ADR 予以更正**；
  相关结论（如"∩可导航 32%"）应基于新判据重算。
