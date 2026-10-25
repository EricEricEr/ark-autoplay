# 合并仓协作规则

本仓含 core、bridge、client；各子目录的环境、类型检查和测试规则仍适用。

## 发布隐私规则（用户 2026-10-26 明确）

- 后续提交与 GitHub 更新不得包含隐私内容。个人路径、账号标识、手机号、凭据、Cookie、token、私钥、真实名册、截图和本机私有配置不得入库。
- 示例使用占位符或环境变量；真实配置保留在已忽略的本地文件中。不得把含机器/账号上下文的工作区交接文档直接复制进公开仓库。
- 提交前检查实际暂存区：`python scripts/check_privacy.py --staged`，并人工复核 diff 和提交说明。
- 推送前更新远端基线，检查待推送的每个新增提交：`python scripts/check_privacy.py --outgoing origin/main`。只看最终 HEAD 不足以排除中间提交泄露。
- 本地 hook 可用 `python scripts/check_privacy.py --install-hooks` 安装；已有用户 hook 不覆盖。CI 是补充，不能代替上传前检查。
- 任何检查失败或内容是否可公开不能确认时，停止提交/推送，不自动放行或忽略。
- 既有低敏路径历史已由用户决定保留；不改写历史、不强推，不发布本地历史备份或清理副本。

详细检查范围、命令和人工复核要求见 [隐私发布规范](docs/PRIVACY.md)。
