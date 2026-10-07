# Copilot 快速上手 —— ark-autoplay-client

玩家端客户端：MuMu 模拟器录屏 + 本地 ONNX CPU 推理 + ADB 模拟输入（Apache-2.0）。
详情与红线一律以根目录 `AGENTS.md` 为准，本文件只做速览。

## 构建与测试（命令与 CI 一致）

```bash
uv sync              # 安装依赖（uv 管理，Python ≥3.12）
uv run pytest        # 测试
uv run ruff check    # lint
uv run pyright       # 类型检查
```

## 布局速览

- `src/client/capture/` 屏幕抓取（MuMu ADB screencap / 渲染管道截图）
- `src/client/runtime/` ONNX Runtime CPU 推理 + 动作掩码
- `src/client/input/` MuMu ADB 模拟输入（tap/swipe）
- `src/client/scheduler.py` 子弹时间时序器（点卡→减速→推理≤2s→落格→超时取最高合法动作）
- `src/client/risk_notice.py` 首次启动不可跳过风险告知（确认记录落盘）
- `src/client/ui/` 最小界面 stub
- `installer/` 打包说明（目标：安装包 ≤100MB 含模型）
- `tests/` pytest（当前为导入冒烟占位）

## 约定速览

- docstring/注释中文；类型注解齐全；ruff + pyright 全绿才提交。
- PR 标题 `type(scope): 中文摘要`；按 PR 模板完成 ReviewBench 九类打标。
- 骨架阶段：占位 stub 只含 docstring + TODO，不写实现逻辑。

## 红线（违反即打回）

- 禁止 import bridge 仓库（ark-autoplay-maa-bridge）代码；状态/权重走协议与 artifact。
- 禁止提交游戏截图/立绘/地图/解包数据；测试只用合成样本。
- 只做"截图 + 模拟输入"，不做内存读取/客户端修改。
