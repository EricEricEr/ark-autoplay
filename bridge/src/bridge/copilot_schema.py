"""MAA 作业动作语义：回放保留全部受支持动作和字段，未知动作显式标记。"""

from __future__ import annotations

MAA_ACTION_TYPES = (
    "Deploy",
    "Skill",
    "Retreat",
    "SpeedUp",
    "BulletTime",
    "SkillUsage",
    "Output",
    "SkillDaemon",
    "MoveCamera",
    "DrawCard",
    "CheckIfStartOver",
    "ResetStopwatch",
    "Click",
    "Swipe",
    "SetUnitLocation",
)
ACTION_EVENT = {name: name for name in MAA_ACTION_TYPES}
ACTION_EVENT.update(Skill="UseSkill", SpeedUp="SwitchSpeed")
_ALIASES = {name.casefold(): name for name in MAA_ACTION_TYPES}
_ALIASES.update(
    {
        "部署": "Deploy",
        "技能": "Skill",
        "撤退": "Retreat",
        "二倍速": "SpeedUp",
        "子弹时间": "BulletTime",
        "技能用法": "SkillUsage",
        "输出": "Output",
        "打印": "Output",
        "donothing": "SkillDaemon",
        "摆完挂机": "SkillDaemon",
        "开摆": "SkillDaemon",
        "移动镜头": "MoveCamera",
        "抽牌": "DrawCard",
        "抽卡": "DrawCard",
        "调配": "DrawCard",
        "调配干员": "DrawCard",
        "检查重开": "CheckIfStartOver",
        "重置全局计时器": "ResetStopwatch",
        "点击": "Click",
        "滑动": "Swipe",
        "设置单位坐标": "SetUnitLocation",
    }
)


def canonical_action_type(value: str) -> str:
    """按 MAA 的别名归一，未知名称原样保留供拒收说明和后续升级。"""
    return _ALIASES.get(value.casefold(), value)
