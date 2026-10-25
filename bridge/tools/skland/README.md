# 森空岛一键练度采集

能扫码的账号优先走本通道；不能扫码才用 MAA OperBox。此入口复用项目已有扫码/签名客户端，二维码在本机生成，不依赖第三方二维码服务。

在 `bridge/` 执行（使用现有 Node.js 18+，无 npm 依赖）：

```powershell
node tools/skland/collect.mjs --out C:/ark-data/roster/roster_skland.json
```

首次运行显示二维码，用森空岛 APP 扫码确认后，程序自动选取绑定默认角色、拉取并导出练度。多角色时用 `--uid <游戏UID>` 指定；输出的 UID 必须与实际录制账号一致。用 `--login` 可重新扫码换账号；已有登录态时直接同步，过期尝试刷新。

导出包含：干员 ID/名称、精英化、等级、潜能；**每个技能**的 ID、普通等级和专精（统一等级 1～7/8～10）；全部 `equip` ID/等级/锁定标志、`defaultEquipId`；完整 `raw_player_info`。没有用旧静态名录过滤新角色，也不把第三技能的专精当作其他技能的练度。

模组保留接口原始 ID，不通过可能过时的图鉴强猜 X/Y 槽位。API 的占位装备也原样保留，由同版本官方表在加工阶段区分。缺字段为 null 或在 `unresolved` 记账。

```powershell
uv run python -m bridge.roster_state --skland C:/ark-data/roster/roster_skland.json --out C:/ark-data/roster/roster_merged.json
```

合并结果用于统一名册/筛选，`extra` 保留技能和模组明细；交接时仍须带上原始导出。每局实际选哪个技能/模组由该局作业和画面对应，账号拥有某模组不等于当局装备它。好友助战不属于本账号名册，需另行记录助战实际练度。

本地登录态默认写入仓库根 `.secrets/`（已忽略），不包含在名册输出中；交接只发名册，不发凭证。真实名册也不提交 Git。

离线检查：`node --test tools/skland/collect.test.mjs`，只使用合成数据和假请求，不读取真实凭证、不访问网络。`--help` 不发请求；`--fixture <文件>` 可离线转换已有 player/info，保留原始响应。

该入口本次完成离线验证；扫码接口沿用已有实现，未在此次更新中重新登录真实账号。登录接口若发生变化，会明确报错，不将失败写成空名册。

第三方二维码库：qrcode-generator，Copyright (c) 2009 Kazuhiko Arase，MIT；保留原文件声明与 [MIT 许可](lib/vendor/LICENSE)。
