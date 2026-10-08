# AGENTS.md —— ark-autoplay-client

> agent 协作总入口（AAIF 约定，Copilot / Codex / Cursor / Gemini CLI 通用）。
> **本文件中的命令与 CI 保持一致；CI 命令变更时必须同步更新本文件。**

## 项目概述

玩家端客户端：MuMu 模拟器屏幕抓取 + 本地 ONNX CPU 推理 + ADB 模拟输入。AGPL-3.0。
发行目标：安装包 ≤100MB 含模型，任意可运行 MuMu 模拟器的 Windows 10/11 x64 PC（含核显机器）可用。
仅支持 MuMu 模拟器，不支持官方 PC 客户端 / 其他模拟器 / 手机真机。

## 目录布局

```
ark-autoplay-client/
├── src/client/            # Python 包（src 布局，包名 client）
│   ├── capture/           # 屏幕抓取（MuMu ADB screencap / 渲染管道截图）
│   ├── runtime/           # ONNX Runtime CPU 推理 + 动作掩码应用
│   ├── input/             # MuMu ADB 模拟输入（tap/swipe）
│   ├── scheduler.py       # 子弹时间时序器：点卡→减速窗口→推理→落格→超时兜底
│   ├── risk_notice.py     # 首次启动风险告知（不可跳过，确认记录落盘）
│   └── ui/                # 最小界面：选关/启停/日志
├── installer/             # 打包（目标：安装包 ≤100MB 含模型），见其 README
├── tests/                 # pytest（骨架阶段为导入冒烟测试）
└── .github/               # copilot 指令、PR/issue 模板、CI 工作流
```

## 环境命令（与 CI 一致）

```bash
uv sync              # 安装依赖（含 dev 组，生成/使用 .venv）
uv run pytest        # 测试
uv run ruff check    # lint
uv run pyright       # 类型检查
```

要求 Python ≥3.12（uv 会自动下载管理）。锁文件 `uv.lock` 已提交，CI 使用 `uv sync --frozen`。

## 代码风格

- ruff（规则集 `E, F, I, UP`，行长 100，`target-version = py312`）；提交前跑 `uv run ruff check`。
- 全部公开函数/模块带类型注解；pyright 必须通过。
- **docstring 一律用中文**，写清模块职责；注释同样中文。
- 文案与代码中禁止出现承诺或暗示可规避账号处罚的表述（合规红线见设计文档 §3.3）。
- 立项骨架阶段：占位 stub 只含模块 docstring + TODO 注释，不写实现逻辑。

## PR 约定

- 标题格式：`type(scope): 中文摘要`，type 取 `feat / fix / docs / test / ci / chore / refactor`（如 `feat(capture): 实现 ADB screencap 通道`）。
- 所有 PR（人审或 agent 审）按 **ReviewBench 九类缺陷 × 高/中/低三级严重度**打标，清单已内置在 `.github/PULL_REQUEST_TEMPLATE.md`：
  Correctness / Security / Reliability / Maintainability / Testing / Performance / API architecture / Accessibility / Documentation。
- 涉及协议字段或模型接口的变更，先确认与 core 仓库的版本兼容性。

## 安全红线（铁律）

- **不得 import 任何 bridge 仓库（ark-autoplay-maa-bridge，AGPL-3.0）代码**；感知状态与决策权重一律通过版本化协议与发布 artifact（ONNX 权重）消费。违反即 PR 打回。
- **禁止提交游戏素材**：截图、立绘、地图、解包数据（版权归鹰角网络）一律不进仓库；测试样本只用合成数据。
- 不提交密钥/凭据/个人路径；禁止内存读取、客户端修改类实现（本项目只做截图 + 模拟输入）。
- 风险告知逻辑（risk_notice）不得改为可跳过；其变更从严评审（ReviewBench Accessibility 类）。
