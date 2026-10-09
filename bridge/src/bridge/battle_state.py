"""战场状态转换：MaaCore 的 ``BattleState`` 回调 → 协议化状态序列。

背景
----
MaaCore 内部**本就维护**战场数值状态（费用/击杀/待部署栏/已部署干员），但官方
发行版**不通过 asst 接口暴露**。本项目以
``patches/0001-battle-state-callback.patch`` 打补丁后，它会在每个作业动作执行前
发出 ``SubTaskExtraInfo`` + ``what="BattleState"`` 的回调。

本模块把该回调体转换成**稳定、可版本化**的落盘格式（``battle_states.jsonl``），
供 core 侧训练消费。转换职责有三：

1. **字段白名单与改名**：只保留训练需要的字段，剔除上游内部细节；
2. **坐标系归一**：上游 ``deployed[].row/col`` 已是**格子坐标**（MaaCore 的 Point
   即格子），与 core 的 ``protocol.State`` 一致，**不做翻转**——这一点极易搞错，
   详见下方"坐标序"；
3. **精度标注**：击杀数在上游**识别不稳**（其源码自注"击杀数经常识别不准，
   所以依赖外部传入作为参考"），故本模块**原样透传**并不做任何平滑/插值，
   由下游自行决定如何使用。

坐标序（**关键，勿改**）
------------------------
- MaaCore 的 ``Point`` 在战场语境下是 **(col, row)**：上游 ``BattleHelper`` 的
  ``m_battlefield_opers`` 注释为 "已部署的干员, <实际职业,名称> -> <坐标>"，其
  ``Point.x`` 对应列、``Point.y`` 对应行。
- 本模块把上游的 ``loc.y`` → ``row``、``loc.x`` → ``col``（patch 里已如此命名），
  故落盘字段名自解释，**下游直接按名取用即可**。
- 注意这与 **episodes 里 deploy 动作**的坐标口径不同：后者是 MAA 作业的
  ``location=[x,y]=[col,row]``，core 侧消费时须经
  ``stagefeat.normalize_deploy_location`` 转成 ``[row,col]``。
  两者是**不同来源的两种数据**，不要混用同一条转换规则。

版本
----
``STATE_PROTOCOL_VERSION`` 与 core 的 ``protocol.version.PROTOCOL_VERSION`` 语义对齐，
但序列结构尚未纳入 RFC（当前为 draft）；字段变更必须升版本。
"""

from __future__ import annotations

from typing import Any

STATE_PROTOCOL_VERSION = "0.1.0-battlestate-draft"
"""battle_states.jsonl 的结构版本（字段变更必须升版）。"""

# 落盘字段白名单：上游 BattleState.details 里保留的键
_KEEP_SCALARS = (
    "in_battle",
    "in_speedup",
    "costs",
    "kills",
    "total_kills",
    "camera_count",
    "stage_name",
    "camera_shift",
)
_HAND_FIELDS = (
    "index",
    "name",
    "role",
    "role_id",
    "cost",
    "available",
    "cooling",
    "rect",
    "is_usual_location",
)
_DEPLOYED_FIELDS = ("name", "role", "role_id", "row", "col")


def _pick(src: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """按白名单取字段（缺失即忽略，不补 None——避免下游把缺字段当成 0）。"""
    return {k: src[k] for k in fields if k in src}


def battle_state_to_record(
    details: dict[str, Any],
    t_ms: int,
    *,
    job_id: int | str | None = None,
    stage_code: str | None = None,
) -> dict[str, Any]:
    """``BattleState`` 回调体 → 一条可落盘的状态记录。

    ``t_ms`` 为相对本局开始的毫秒数（由调用方提供，保证与动作事件同一时间轴）；
    ``job_id`` / ``stage_code`` 用于把状态与其来源作业绑定（便于多局混合分析）。
    """
    rec: dict[str, Any] = {
        "protocol_version": STATE_PROTOCOL_VERSION,
        "t_ms": int(t_ms),
    }
    if job_id is not None:
        rec["job_id"] = job_id
    if stage_code is not None:
        rec["stage_code"] = stage_code

    for k in _KEEP_SCALARS:
        if k in details:
            rec[k] = details[k]

    hand = details.get("hand")
    if isinstance(hand, list):
        rec["hand"] = [
            _pick(h, _HAND_FIELDS) for h in hand if isinstance(h, dict)
        ]
    deployed = details.get("deployed")
    if isinstance(deployed, list):
        rec["deployed"] = [
            _pick(d, _DEPLOYED_FIELDS) for d in deployed if isinstance(d, dict)
        ]
    return rec


def summarize_states(records: list[dict[str, Any]]) -> dict[str, Any]:
    """统计一批状态记录（写进 episode.json，便于快速判断这局有没有采到状态）。

    返回：条数、费用/击杀的取值范围、待部署栏规模、覆盖的时间跨度。
    空列表时返回 ``{"n": 0}``——**不伪造 0 值**，让"没采到"与"采到 0"可区分。
    """
    if not records:
        return {"n": 0}
    costs = [r["costs"] for r in records if isinstance(r.get("costs"), int)]
    kills = [r["kills"] for r in records if isinstance(r.get("kills"), int)]
    hand_sizes = [len(r.get("hand") or []) for r in records]
    ts = [r["t_ms"] for r in records if isinstance(r.get("t_ms"), int)]
    out: dict[str, Any] = {
        "n": len(records),
        "protocol_version": STATE_PROTOCOL_VERSION,
        "has_costs": bool(costs),
        "has_kills": bool(kills),
        "hand_size_max": max(hand_sizes) if hand_sizes else 0,
    }
    if costs:
        out["costs_range"] = [min(costs), max(costs)]
    if kills:
        out["kills_range"] = [min(kills), max(kills)]
    if ts:
        out["t_ms_range"] = [min(ts), max(ts)]
    return out
