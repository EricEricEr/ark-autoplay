---
applyTo: "src/ark_core/protocol/**"
---

# 协议目录改动约定

- **任何协议字段变更必须升级 `src/ark_core/protocol/version.py` 中的版本号**（semver：破坏性升主 / 次版本，纯增量升修订版本）。
- 必须在 `src/ark_core/protocol/migrations/` 提供旧版本 → 新版本的迁移器，保证历史轨迹可读；CI 含协议迁移测试。
- 必须同步更新 `docs/rfc/0001-state-action-protocol.md`（重大变更另立新 RFC）。
- schema 使用 pydantic v2；docstring 用中文；字段语义变更需在迁移器的变更记录中写清。
