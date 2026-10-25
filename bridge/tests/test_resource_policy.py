"""资源权限回归：默认无源石授权，错误请求不能到达 MAA，更新参数也不能绕过。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from bridge.maa_driver import MaaDriver
from bridge.nav_maa import safe_copilot_params
from bridge.resource_policy import originite_allowed, protect_task_params


def test_missing_permission_is_denied_and_defaults_do_not_use_supplies() -> None:
    assert not originite_allowed({})
    safe = protect_task_params("Fight", {"stage": "SYN-1"})
    assert safe["stone"] == safe["medicine"] == safe["expiring_medicine"] == 0
    assert safe["medicine_expire_days"] == 0
    assert safe_copilot_params()["use_sanity_potion"] is False


@pytest.mark.parametrize("value", ["true", "false", 1, None])
def test_permission_must_be_explicit_boolean(value: Any) -> None:
    with pytest.raises(ValueError):
        originite_allowed({"resource_policy": {"allow_originite": value}})


@pytest.mark.parametrize("count", [-1, True, "1", None])
def test_bad_stone_count_is_not_forwarded(count: Any) -> None:
    with pytest.raises(ValueError):
        protect_task_params("Fight", {"stone": count})


def test_opt_in_only_preserves_an_explicit_request_and_does_not_add_one() -> None:
    assert originite_allowed({"resource_policy": {"allow_originite": True}})
    request = {"stage": "SYN-1", "stone": 2}
    assert protect_task_params("Fight", request, allow_originite=True)["stone"] == 2
    assert request == {"stage": "SYN-1", "stone": 2}
    assert protect_task_params("Fight", {}, allow_originite=True)["stone"] == 0


def test_driver_blocks_both_initial_and_updated_paid_requests_before_sdk() -> None:
    class SDK:
        calls: list[Any] = []

        def append_task(self, *args: Any) -> int:
            self.calls.append(args)
            return 1

        def set_task_params(self, *args: Any) -> bool:
            self.calls.append(args)
            return True

    driver = MaaDriver.__new__(MaaDriver)
    sdk = SDK()
    driver._asst = sdk
    with pytest.raises(PermissionError):
        driver.append_task("Fight", {"stone": 1})
    with pytest.raises(PermissionError):
        driver.set_task_params(1, {"stone": 1})
    with pytest.raises(PermissionError):
        safe_copilot_params({"stone": 1})
    assert sdk.calls == []
    assert driver.append_task("Fight", {"stage": "SYN-1"}) == 1
    assert sdk.calls[0][1]["stone"] == 0


def test_copilot_entry_uses_default_resource_protection() -> None:
    class SDK:
        def append_task(self, kind: str, params: dict) -> int:
            assert kind == "Copilot"
            assert params["use_sanity_potion"] is False
            assert params["support_unit_usage"] == 1
            return 7

    driver = MaaDriver.__new__(MaaDriver)
    driver._asst = SDK()
    assert driver.append_copilot("synthetic_job.json", support_unit_usage=1) == 7


def test_shared_instance_configs_never_enable_originite() -> None:
    root = Path(__file__).resolve().parents[1] / "configs/instances"
    for path in root.glob("*.yaml"):
        if ".local." in path.name:
            continue
        assert not originite_allowed(yaml.safe_load(path.read_text(encoding="utf-8")))
