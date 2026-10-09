# ============================================================================
# PATCH 0001 —— 战场状态回调（BattleState）
# ============================================================================
#
# 上游：MaaAssistantArknights / MaaCore（分支 dev-v2）
# 上游 commit：见 ../maacore/UPSTREAM.json 的 pinned_commit
# 改动规模：4 文件，+90 行，**0 删除**（纯新增，不修改任何识别/决策逻辑）
#
# ── 为什么需要这个补丁 ────────────────────────────────────────────────────
# MaaCore 内部**本就在维护**战场数值状态（BattleHelper 的 m_cost / m_kills /
# m_total_kills / m_cur_deployment_opers / m_battlefield_opers），但它们
# **不通过 asst 接口暴露**：导出的 C API 无 GetCost/GetKills，回调消息类型
# 也只有任务链生命周期（无战场状态）。数据工厂因此只能落盘截图、拿不到数值
# 状态，导致行为克隆缺少输入（见 STATUS.md §5）。
#
# 本补丁把这些**已经算出来的值**序列化为回调，不新增任何识别逻辑。
#
# ── 回调格式 ──────────────────────────────────────────────────────────────
# 消息类型：SubTaskExtraInfo        what 字段："BattleState"
# details：
# {
#   "in_battle": bool, "in_speedup": bool, "camera_count": int,
#   "costs": int,                     // 当前费用
#   "kills": int, "total_kills": int, // 已击杀 / 总击杀
#   "hand": [                         // 待部署栏（手牌）
#     {"index":int, "name":str, "role":str, "role_id":int, "cost":int,
#      "available":bool, "cooling":bool, "rect":[x,y,w,h],
#      "is_usual_location":bool}
#   ],
#   "deployed": [                     // 已部署干员（格子坐标，非像素）
#     {"name":str, "role":str, "role_id":int, "row":int, "col":int}
#   ],
#   "stage_name": str, "camera_shift": [double, double]
# }
#
# ── 已知精度限制（上游自注，必须如实传递给下游）──────────────────────────
# BattlefieldMatcher.h 原文：
#   "m_total_kills_prompt = 0; // 之前的击杀总数，因为击杀数经常识别不准
#    所以依赖外部传入作为参考"
# 即**击杀数 OCR 不稳**，MAA 用上一帧值做先验。消费方不应把 kills 当精确真值。
#
# ── 触发时机 ──────────────────────────────────────────────────────────────
# BattleProcessTask::do_action() 中 wait_condition() 之后、动作执行之前——
# 即"该作业动作即将执行时的战场状态"，正是行为克隆需要的 (state, action) 配对点。
#
# ── 许可 ──────────────────────────────────────────────────────────────────
# 上游为 AGPL-3.0（**已核实无附加条款**，见 UPSTREAM.json）。本补丁作为衍生
# 作品同样以 AGPL-3.0 授权；bridge 目录整体已是 AGPL-3.0，无需变更。
#
# ── 应用方式 ──────────────────────────────────────────────────────────────
#   git -C maacore/_src apply ../../patches/0001-battle-state-callback.patch
# ============================================================================
