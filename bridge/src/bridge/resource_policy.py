"""账号资源权限：采集者默认禁止源石，授权必须来自本次账号的本地配置。"""

from __future__ import annotations

from typing import Any


def originite_allowed(instance: dict[str, Any]) -> bool:
    """缺字段即禁止；字符串 false/true 不按 Python 真值自动转换。"""
    policy = instance.get("resource_policy", {})
    if not isinstance(policy, dict):
        raise ValueError("resource_policy 必须是 mapping")
    allowed = policy.get("allow_originite", False)
    if not isinstance(allowed, bool):
        raise ValueError("resource_policy.allow_originite 必须是布尔值")
    return allowed


def medicine_allowed(instance: dict[str, Any]) -> bool:
    """理智药权限按账号保存；共享/缺项配置默认不吃，本机授权单独启用。"""
    policy = instance.get("resource_policy", {})
    if not isinstance(policy, dict):
        raise ValueError("resource_policy 必须是 mapping")
    allowed = policy.get("allow_medicine", False)
    if not isinstance(allowed, bool):
        raise ValueError("resource_policy.allow_medicine 必须是布尔值")
    return allowed


def protect_task_params(
    task_type: str,
    params: dict[str, Any] | None,
    *,
    allow_originite: bool = False,
    allow_medicine: bool = False,
) -> dict[str, Any]:
    """派发/更新前检查资源参数；不原地修改调用者的字典。

    stone 是上游 Fight 的次数参数。Copilot 不支持用该字段补源石；关闭
    use_sanity_potion 是其现有默认行为。授权只准许显式请求，不自动加次数。
    """
    safe = dict(params or {})
    for field in ("medicine", "expiring_medicine"):
        if field in safe:
            count = safe[field]
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f"{field} 必须是非负整数次数")
            if count > 0 and not allow_medicine:
                raise PermissionError(
                    "当前账号默认不使用理智药；请先明确启用本地账号配置"
                )
    if "use_sanity_potion" in safe:
        if not isinstance(safe["use_sanity_potion"], bool):
            raise ValueError("use_sanity_potion 必须是布尔值")
        if safe["use_sanity_potion"] and not allow_medicine:
            raise PermissionError("当前账号默认不使用理智药")
    if "stone" in safe:
        count = safe["stone"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("stone 必须是非负整数次数")
        if count > 0 and not allow_originite:
            raise PermissionError(
                "当前账号禁止源石补理智；stone 必须为 0。"
                "仅账号所有者明确授权后可在本地实例配置开放权限。"
            )
    if task_type == "Fight":
        safe.setdefault("stone", 0)
        safe.setdefault("medicine", 0)
        safe.setdefault("expiring_medicine", 0)
        safe.setdefault("medicine_expire_days", 0)
    elif task_type == "Copilot":
        safe.setdefault("use_sanity_potion", allow_medicine)
    return safe
