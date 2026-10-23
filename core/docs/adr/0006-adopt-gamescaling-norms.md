# ADR-0006：采纳 GameScaling 的工程优点（非引入模型本体）

- 状态：**已采纳**（2026-10-23，用户指示「他的优点我们采纳」）
- 触发：米哈游崩坏 AI+R&D × 清华等 10-22 开源 GameScaling（Qwen3.5 键鼠游戏模型，Apache-2.0，0.8B/2B/9B/27B）
- 关联：方案 §8 轨道 B；本仓 v1-v4 训练评测线；`data-validation/gamescaling/` 三份独立评估

## 背景

我们已对 GameScaling 完成三级独立评估（证据落盘点见文末），确认它的工程做法有值得收编的部分，同时明确它不是明日方舟的即插即用解（方舟属它论文自证的「远世界」，其距离泛化被前沿模型两倍杀）。本 ADR 划分「采纳 / 存档候选 / 明确不采纳」三层边界。

## 采纳清单（工程规范，即刻生效的评审口径）

1. **输出词表法纪 = 评估硬闸**：VL 动作输出必须过「词表围栏 + `parsed` 率 + `dropped` 计数」三度量（零非法 token 才准谈能力）。来源：其 `gamescaling.action.KEY_VOCAB` 与我们的方舟契约 10/10 合规实测。
2. **系统提示模板化 + 逐字节契约**：未来任一 VL 动作实验的 system prompt 走「名称/规则/键位」三槽模板化；chat 模板与 history 渲染**逐字节固定**（含 `**Game Rules**` 这类伪装成标题的实义头，属 prompt 本体）；无思考轮也必须走训练同款渲染（其 `<think>\n\n</think>` 先例）。
3. **安全默认 No-op**：动作解析失败一律执行空操作，**绝不**用猜测键位代替危险键（适用将来的 client/executor 安全需求；其客户端就是这么默认的）。
4. **动作分块节奏的参照系**：其 200ms=6×33ms 打包模式是子弹时间粒度的参照样例；我们 ARK 子弹时间（§9）语义动作＋节奏可借鉴，但**块内键位语义不兼容**（其键鼠 74 键词表 vs 我方塔防触控格点），不得照搬。
5. **「zero-shot 对未知游戏保守 no-op」作为微调起点判别**：基底做 zero-shot 契约探针时，能保守输出合法空操作者才是干净微调基座（2B-S3 实测如此），乱按键者一票否决。

## 存档候选（不预决，评审时启用）

- **基底候选**：`GameScaling-2B-S3`（GGUF Q8_0，权重+serve 已在本机就绪；烟雾/预评/2B-9B 对照三判决书在案）——与 Qwen3-VL-4B 同场进 M1/M2 双轨评审；**是否启用看彼时数据证据说话**。
- 9B-S3 技备（local 推理贴 8G 顶 7.67G、决策 ~4.9s，性价比劣于 2B）。

## 明确不采纳（防走歪的负数清单）

1. **即插即用打方舟**：被拒绝于源头——方舟是它的远世界；其键鼠 74 键 / 鼠标千分位相对位移的动作空间与方舟触控格点不兼容，未经重新映射不可用。
2. **立即动工 `track_b_mapper`**：该映射器仅是「若启用它当基底」下的必要前提——**启动与否由基底评审结论反推**，当前冻结。
3. **为它租卡**：只让它正经上场评估时才算（¥2/次的对照器材逻辑；默认本机 4060 8G 起手）。

## 后果

- 三份判决书归档为评审附件：`data-validation/gamescaling/report.md`（冒烟）/ `report_ak_preeval.md`（方舟契约 GO）/ `report_ak_9b_compare.md`（2B vs 9B 推举 2B）。
- `models/track_b/__init__.py` stub TODO 注记本 ADR 的评审口径要点。
- 将来到 client/executor 设计时，「安全默认 No-op + 词表法纪」必须写进需求清单。

## 证据

- 公开：[HF 卡片](https://huggingface.co/THUSI-Lab/GameScaling)（权重/客户端/license）与 [arXiv:2610.28698](https://arxiv.org/abs/2610.28698)（缩放律、远世界降级、GUI 规模律）
- 本机：`data-validation/gamescaling/`（report.md / report_ak_preeval.md / report_ak_9b_compare.md / prompt_arknights_battle.txt / 原始 json）。
