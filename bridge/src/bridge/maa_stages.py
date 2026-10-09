"""MAA 关卡可导航性判定（依据 MaaCore 源码，非猜测）。

为什么需要它
------------
`Fight` 任务**不是任意关名都接受**。实测（2026-10-09）下发 `GT-1` 被直接拒收：

    Unknown task: GT-1
    The stage name is not in invalid, or is not main line stage GT-1
    Cannot set stage GT-1

根因在 `StageNavigationTask::set_stage_name`（源码见 ``maacore/_src``）：

1. 若 task 表里有同名 task → 直接可用（``m_is_directly``）；
2. 否则须匹配正则 ``^([A-Za-z]{0,3})(\\d{1,2})-(\\d{1,2})(?:-?(\\w+))*$``，
   **且**必须存在名为 ``Episode{章节}`` 的 task。

即：`GT-1` 里 `GT` 是前缀、其后没有 ``数字-数字``，故不匹配 → 拒收。
这解释了为什么活动关（占数据集 71%）绝大多数无法用 `Fight` 直接导航。

本模块把该规则固化为可复用判定，供队列生成与运行时预检使用。

⚠️ 判定的是"**MAA 会不会接受这个关名**"，**不代表账号已解锁该关**——
后者取决于玩家进度，只能在运行时由导航结果体现。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# 与 MaaCore `StageNavigationTask::set_stage_name` 中的 boost::regex 严格一致
STAGE_RE = re.compile(r"^([A-Za-z]{0,3})(\d{1,2})-(\d{1,2})(?:-?(\w+))*$")


def load_maa_tasks(maa_dir: str | Path) -> set[str]:
    """合并 MAA 运行时下**所有** ``tasks.json`` 的 task 名。

    MAA 可能有主资源与增量资源多份 tasks.json（``resource/`` 与 ``cache/``），
    ``Task.get`` 在同一张表里查，故此处取并集。
    """
    names: set[str] = set()
    for p in Path(maa_dir).rglob("tasks.json"):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(j, dict):
            names |= set(j.keys())
    return names


def navigable_stages(tasks: set[str]) -> set[str]:
    """可导航的章节号集合（即存在 ``Episode{N}`` 的 N）。"""
    out: set[str] = set()
    for t in tasks:
        m = re.fullmatch(r"Episode(\d+)", t)
        if m:
            out.add(m.group(1))
    return out


def maa_navigable(stage: str, tasks: set[str]) -> bool:
    """该关名能否被 ``Fight`` 接受（规则见模块 docstring）。

    ``tasks`` 为空时视为"未加载规则表"，一律返回 True（不做限制），
    便于在没装 MAA 的环境里跑纯逻辑测试。
    """
    if not tasks:
        return True
    if stage in tasks:
        return True
    m = STAGE_RE.match(stage)
    if not m:
        return False
    return f"Episode{m.group(2)}" in tasks


def explain(stage: str, tasks: set[str]) -> str:
    """给出判定原因（用于日志/报告，便于排障）。"""
    if not tasks:
        return "no-task-table"
    if stage in tasks:
        return "direct-task"
    m = STAGE_RE.match(stage)
    if not m:
        return "shape-mismatch(需 前缀?+数字-数字，如 1-7 / JT8-2 / H10-1-Hard)"
    ep = f"Episode{m.group(2)}"
    if ep not in tasks:
        return f"missing-{ep}"
    return "ok"
