"""协议库：状态 / 动作 / 轨迹的 pydantic schema，semver 版本化。

bridge 与 core/client 之间仅靠本库的版本化 JSON 协议通信。
任何协议变更必须升级 version.PROTOCOL_VERSION 并在 migrations/ 提供迁移器。
"""
