"""协议版本号的基础校验（骨架期样例测试，保证 CI 有可跑的通过用例）。"""

import re

from ark_core.protocol.version import PROTOCOL_VERSION


def test_protocol_version_semver_format() -> None:
    """PROTOCOL_VERSION 必须符合 semver 的 x.y.z 数字形式。"""
    assert re.fullmatch(r"\d+\.\d+\.\d+", PROTOCOL_VERSION), (
        f"协议版本号格式非法: {PROTOCOL_VERSION}"
    )


def test_state_default_matches_protocol_version() -> None:
    """State 默认 protocol_version 应与全局版本号一致。"""
    from ark_core.protocol.state import State

    assert State().protocol_version == PROTOCOL_VERSION
