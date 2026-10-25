# 补丁 MaaCore 运行时

状态采集需要本项目配套的 MaaCore；普通官方 MAA 没有 `BattleState` 回调。运行时二进制和上游源码 checkout 不随本 Git 仓库分发，采集者可以使用维护者提供的配套目录，或自行按以下记录重建。

## 固定版本与完整补丁

- 上游：[MaaAssistantArknights](https://github.com/MaaAssistantArknights/MaaAssistantArknights)，`dev-v2`，pin `a82a0e13c68d48cc21ec40380ac08c95f46e9802`。
- **重建入口**：[runtime-a82a0e13.patch](../patches/runtime-a82a0e13.patch)，对干净 pin 一次应用。
- 该补丁覆盖当前部署源码的 8 个文件，包括状态导出、`nav_only`、代理指挥护栏、费用/击杀刷新和本地探针。`UPSTREAM.json` 记录补丁 SHA-256、目标 tree 和已部署 DLL 的参考哈希。
- `0001`～`0005` 保留历史说明；部分文件是累积 diff，**不是可直接逐个应用的独立系列**。不能先应用完整补丁再重复应用历史补丁。
- 2026-10-26 已用独立临时 Git index 验证：干净 pin → 完整补丁 → tree 与实际运行源码 tree 一致。未为此次文档更新重新编译或真机测试。

## 运行时目录

```text
maa_runtime_patched/
  MaaCore.dll                 本项目补丁构建
  resource/                   与 pin 同一提交的完整资源及依赖数据
  Python/asst/                Python 调用封装
  其他运行依赖 DLL            参考 UPSTREAM.json，开发机由官方 v6.18.0 配套
```

**DLL 与 resource 必须配套。** 将新 dev-v2 DLL 与旧官方资源混用曾因 schema_version 不匹配导致 `AsstLoadResource` 失败。仅有源码稀疏 checkout 不保证带全 resource/地图子模块；准备运行目录前须取齐同一提交的资源与子模块内容。

当前 Python 工厂检查 `maa_dir` 路径包含 `maa_runtime_patched`；这只是避免误用的名称护栏，不能证明 DLL 的能力或资源版本。协助者需从提供方取得对应源码/版本清单；名称正确仍要通过首次状态采集验收。

## 开发者重建

重建需要已有 C++ 工具链；普通采集者拿到配套运行目录后无需安装编译工具。开发机验证过 VS 2022 Build Tools、Windows SDK、CMake >=3.28、MaaDeps v2.14.1 x64-windows；准确记录见 [UPSTREAM.json](UPSTREAM.json)。

在本仓根目录执行以下取源码/补丁步骤；不要在已有修改的 checkout 上强行覆盖：

```powershell
git clone --no-checkout https://github.com/MaaAssistantArknights/MaaAssistantArknights.git bridge/maacore/_src
git -C bridge/maacore/_src checkout a82a0e13c68d48cc21ec40380ac08c95f46e9802
git -C bridge/maacore/_src submodule update --init --recursive
git -C bridge/maacore/_src apply --check ../../patches/runtime-a82a0e13.patch
git -C bridge/maacore/_src apply ../../patches/runtime-a82a0e13.patch
```

然后按该 pin 的 `src/MaaUtils`/MaaDeps 说明准备 `src/MaaUtils/MaaDeps` 依赖；版本与 triplet 见清单。以下配置与开发机构建参数一致，并不自动下载依赖：

```powershell
cmake -S bridge/maacore/_src -B bridge/maacore/_src/build -G "Visual Studio 17 2022" -A x64 -DBUILD_WPF_GUI=OFF -DBUILD_DEBUG_DEMO=OFF -DBUILD_RESOURCE_UPDATER=OFF -DINSTALL_RESOURCE=OFF -DINSTALL_PYTHON=OFF -DWITH_EMULATOR_EXTRAS=OFF -DMAADEPS_TRIPLET=maa-x64-windows
cmake --build bridge/maacore/_src/build --config Release --target MaaCore
```

从 `build/bin/Release/MaaCore.dll` 取构建产物，与同 pin 完整资源、Python/asst、运行依赖组装成独立 `maa_runtime_patched` 目录，不覆盖普通 MAA 安装。不同工具链重建的 DLL 哈希可以不同；参考哈希用于识别维护者当前实测二进制，不能要求所有重建字节相同。

## 已知边界

普通 Copilot 采集在动作前导出状态；SingleStep 的 Output 心跳只刷新费用/击杀，手牌/部署字段可能陈旧，实验性模型闭环以影子台账补足。没有敌人实时位置/完整技力。探针可在进程工作目录产生 `debug/`；该目录被根 `.gitignore` 忽略，不能提交其中的截图。

许可证为 AGPL-3.0，上游关系见 [NOTICE.md](../NOTICE.md)。
