"""MAA 关卡可导航性判定（依据 MaaCore 源码，非猜测）。

**两条路径的判据完全不同，勿混用（重要纠错）**
================================================

2026-10-09 曾用 ``Fight`` 的规则判"可导航"，**导致大批活动关被误判为不可用**
（用户指出"这些作业都是从 MAA 作业网站下载的"，质疑成立）。原因是把两条
独立路径的判据搞混了：

``Fight``（我们用它做 nav_only 导航）
-------------------------------------
``StageNavigationTask::set_stage_name``：

1. task 表里有同名 task → 直接可用；
2. 否则须匹配 ``^([A-Za-z]{0,3})(\\d{1,2})-(\\d{1,2})(?:-?(\\w+))*$``
   **且**存在 ``Episode{章节}`` task。
   → 这是**章节寻路**的要求，``GT-1`` 因不匹配该正则而被拒。

``Copilot``（执行作业，自带的 navigate_to_stage）
--------------------------------------------------
``MultiCopilotTaskPlugin::navigate_to_stage``：**无章节要求、无关卡白名单**，
纯靠 OCR 在地图上滑动找关名（``find_stage`` → 右滑 10 次 → 左滑反复扫）。

而 ``CopilotTask.cpp`` 的 multi 分支用

    const auto& map_data = Tile.find(stage_name);
    if (!map_data || !json::open(map_data->second)) return false;

即 **MAA 靠"有无地图数据（Arknights-Tile-Pos）"判断能否处理该关**。
实测该库覆盖：

| 类型 | 覆盖 |
|---|---|
| MAIN | 564/564 = 100% |
| SUB | 86/86 = 100% |
| CAMPAIGN / DAILY / GUIDE | 100% |
| **ACTIVITY** | **2152/2317 = 92.9%** |
| CLIMB_TOWER | 137/239 = 57.3% |
| 合计（作业维度） | **40295/40299 = 100.0%** |

结论：**绝大多数作业都有地图数据**。先前 8,440/39,010 的估计是错的。

本模块据此提供正确判据
----------------------
- :func:`load_tile_keys` —— 解析 Tile-Pos 的关卡 key 集合（权威判据）；
- :func:`has_map_data` —— Copilot 能否处理该关（**推荐用它做队列过滤**）；
- :func:`fight_accepts` —— ``Fight`` 能否接受该关名（导航用，保守判据）。

⚠️ 两条提醒：
1. "有地图数据 / Fight 接受"**不代表账号已解锁该关**——后者只能在运行时
   由导航结果体现；
2. ``Fight`` 的章节约束仍然真实存在：即使关卡有地图数据，
   用 ``Fight(nav_only)`` 导航时若关卡名推不出 ``Episode{N}``，
   **导航会失败**（但可改用 Copilot 自带导航 + 手工进入章节地图）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ---- Fight 路径（StageNavigationTask::set_stage_name）----
STAGE_RE = re.compile(r"^([A-Za-z]{0,3})(\d{1,2})-(\d{1,2})(?:-?(\w+))*$")

# Tile-Pos 文件名里的分类段（用于从 `{key}-{category}-{parent}-level_{name}` 提取 key）
_TILE_CATS = (
    "activities",
    "obt",
    "weekly",
    "campaign",
    "guide",
    "tutorial",
    "sandbox",
)


def load_maa_tasks(maa_dir: str | Path) -> set[str]:
    """合并 MAA 运行时下**所有** ``tasks.json`` 的 task 名。"""
    names: set[str] = set()
    for p in Path(maa_dir).rglob("tasks.json"):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(j, dict):
            names |= set(j.keys())
    return names


def _tile_key_of(stem: str) -> str:
    """从 Tile-Pos 文件名提取关卡 key。

    文件名结构 ``{key}-{category}-{parent}-level_{levelname}``；
    key 自身可能含连字符（如 ``main_00-01``），故按 category 标记切分。
    """
    for cat in _TILE_CATS:
        marker = f"-{cat}-"
        i = stem.find(marker)
        if i > 0:
            return stem[:i]
    j = stem.rfind("-level_")
    return stem[:j] if j > 0 else stem


def load_tile_keys(maa_dir: str | Path) -> set[str]:
    """加载 Arknights-Tile-Pos 的关卡 key 集合（**Copilot 可处理性的权威判据**）。"""
    root = Path(maa_dir) / "resource" / "Arknights-Tile-Pos"
    keys: set[str] = set()
    if not root.is_dir():
        return keys
    for f in root.glob("*.json"):
        keys.add(_tile_key_of(f.stem))
    return keys


def has_map_data(stage_id: str, code: str, tile_keys: set[str]) -> bool:
    """MAA 是否有该关的地图数据（= Copilot 能否处理）。

    ``tile_keys`` 为空时视为"未加载"，返回 True（不限制），便于纯逻辑测试。
    """
    if not tile_keys:
        return True
    sid = re.sub(r"#.*$", "", stage_id)
    if stage_id in tile_keys or sid in tile_keys:
        return True
    return bool(code) and code in tile_keys


def fight_accepts(stage_code: str, tasks: set[str]) -> bool:
    """``Fight`` 能否接受该关名（导航路径的**保守**判据）。

    规则见 ``StageNavigationTask::set_stage_name``。注意这比
    :func:`has_map_data` 严格得多——它额外要求 ``Episode{章节}`` 存在。
    """
    if not tasks:
        return True
    if stage_code in tasks:
        return True
    m = STAGE_RE.match(stage_code)
    if not m:
        return False
    return f"Episode{m.group(2)}" in tasks


def navigable_stage_types() -> tuple[str, ...]:
    """Tile-Pos 覆盖率较高的关卡类型（供文档/报告引用，非硬编码过滤）。"""
    return ("MAIN", "SUB", "ACTIVITY", "CAMPAIGN", "DAILY", "GUIDE")
