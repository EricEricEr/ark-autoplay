"""协议版本号（全库唯一来源，遵循 semver）。

协议库所有 schema 共用此版本号：
- 破坏性变更（字段删除 / 语义改变）升主版本或次版本；
- 纯增量（新增可选字段）升修订版本；
- 任何变更必须同时在 migrations/ 提供旧版迁移器，并更新 docs/rfc/0001。

见 docs/rfc/0001-state-action-protocol.md（RFC-0001）。
"""

PROTOCOL_VERSION = "0.1.0"
