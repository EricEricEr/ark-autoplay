"""track_b：Qwen3-VL-4B 像素路线（对照组）。

QLoRA 微调配方 + 数据转换器（轨迹 → 多模态对话格式）；
发行形态为 int4 GGUF 量化（见 export/quantize.py）。
"""

# TODO(M1): QLoRA 训练配方与轨迹→对话格式转换器（占位 stub）。
# 评审口径锚定 ADR-0006：输出词表法纪（parsed/dropped）、prompt 三槽+逐字节契约、
# 安全默认 No-op。基底候选 = Qwen3-VL-4B 对照 GameScaling-2B-S3（备档启用，见 ADR-0006）。
