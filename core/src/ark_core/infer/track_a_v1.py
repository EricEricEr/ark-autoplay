"""v1 行为克隆冒烟模型的在线推理胶水：真机 battle_state → MAA 作业形态动作。

诚实口径（**不是自主作战能力证明**）
-------------------------------------
配套的 ``track_a_v1.pt`` 是约 1k 状态-动作对上的链路冒烟模型（训练侧已在
``imitate_v1`` / ADR-0005 声明不作能力证据）。本模块是「模型闭环打关」里程碑的
第一环，只证明「battle_state → 模型四头 logits → 合法性过滤 → 合法作业动作」
这条在线链路可走通；任何对外表述不得夸大为自主作战能力。

在线对齐与三处诚实简化
------------------------
1. ``hist_state``：训练时第 i 位是第 i 个动作**之前**的快照（见 policy_v1 模块
   docstring 的对齐语义）；在线只保留最新一份 battle_state，故把当前状态向量
   **平铺到全部历史位**。状态由 cost/kills/手牌等缓变量构成，作为链路冒烟可接受，
   后续若在线留存状态历史可直接替换。
2. state_feat 的 ``step`` 维需要序列总长 ``total``，在线不可知，用
   ``total_proxy`` 代理（取训练分布周边的常数，配置可调，不许调参过拟合）。
3. 动作历史编码（``encode_action_for_history``）与训练编码
   （``state_action_data.encode_episode`` 的动作段）严格同口径——在线自回归的
   历史 token 若与训练分布不一致，历史条件就全在分布外。

合法性过滤（**迭代式受限 argmax**）
-------------------------------------
不是「argmax 后丢弃」：按 type logits 从高到低逐类尝试，每类先把非法候选的
logit 屏蔽到 -inf 再 argmax；该类没有任何合法候选就换下一类，全部判死返回
``None``。规则：

- **deploy**：目标槽位干员须在 ``hand`` 中、``available``、非 ``cooling``、且
  费用 ≤ ``costs + cost_margin``（``cost`` 字段缺失按可负担处理，与
  ``state_vector`` 的缺失即 0 口径一致）；目标格须可部署（grid 编码第
  :data:`DEPLOYABLE_CHANNEL` 通道）且未被 ``deployed`` 占用；UNK 目标非法；
  全局词表目标（支援装置等，无法从 battle_state 核验）由 ``allow_vocab_targets``
  开关整体放行；
- **skill / retreat**：目标槽位干员须在 ``deployed`` 里；全局词表与 UNK 非法；
- **speed / bullet_time**：仅受 ``InferConfig`` 开关管制（target 无目标语义，
  不约束）。

推理噪声闸门（v1.1，``InferConfig`` 默认开启，可整体回退 v1 行为）
----------------------------------------------------------------------
针对真机闭环实测的两类无效动作噪声（倍速反复切换、误撤退）：

- **倍速语义闸门**（``speed_semantic_gate``）：次数不限、只在必要时切换——
  本拍无任何合法有意义动作（纯等待死亡时间）才允许升 2x；模型正要下发真实
  动作（deploy/skill/retreat）且当前 2x 时，本拍直接把 SpeedUp 顶到输出
  降回 1x（``slow_for_action``）；两次换速之间加 ``speed_min_interval_s``
  迟滞（按步序号 × 决策节拍折算，历史只记动作不记时间戳，属已知简化）。
  速度状态只依据 battle_state 快照的 ``in_speedup`` 字段——bridge loop 侧
  曾出现该字段为脏值的已知坑，推理层按输入快照语义工作、不试图修正上游；
- **撤退双闸门**：``retreat_min_logit_margin``（retreat logit 须压过其余类型
  最高 logit 一个余量，余量从 yaml）+ ``retreat_min_interval_steps``
  （两次撤退最小间隔）；
- **重复护栏**（``repeat_guard``）：同一 (type, 目标名) 连续出现 N 次即从
  掩码剔除该名字候选（speed/bullet_time 按 type 整类屏蔽）；
- **重名部署屏蔽**：同名干员已在 ``deployed`` 时屏蔽该名 deploy 候选。

坐标序（ADR-0003 在案）
------------------------
作业 deploy ``location = [x, y] = [col, row]``，grid 是 ``[row, col]``。编码走
``xy_to_cell``、解码走 ``cell_to_xy``，两处都经
:func:`ark_core.datapipe.stagefeat.normalize_deploy_location` 的互逆口径，
往返恒等（单测在案）。

featvec 行序纪律
----------------
编队 featvec 必须以训练用数据集 npz 里**随数据集冻结**的矩阵为准（parquet 重建
会行序漂移，``build_dataset_delta`` 的 docstring 在案）。parquet 只用于
「中文名 → char_id → 行号」映射，并与 npz 冻结矩阵做全等校验，不一致直接报错。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..datapipe.routefeat import route_feature_channels
from ..datapipe.stagefeat import normalize_deploy_location
from ..models.track_a.policy_v1 import TrackAPolicyV1
from ..protocol.action import (
    ACTION_TYPES,
    DIR_ALIASES,
    DIRECTIONS,
    GRID_CHANNELS,
)
from ..training.imitation_data import (
    DIR_TO_IDX,
    MAX_CELLS,
    MAX_SQUAD,
    TYPE_TO_IDX,
    UNK_TARGET,
    Vocab,
    _load_featvec,
    _load_stagefeat,
)
from ..training.state_action_data import (
    JOB_TYPE_TO_ACTION,
    _load_name_map,
    base_stage_id,
    state_vector,
)

DEPLOYABLE_CHANNEL = 2
"""grid 编码中「该格是否可部署」的通道下标（布局契约见 protocol.action 的
``GRID_CHANNELS`` docstring：高度 / 可部署类型 / 是否可部署 / 起点 / 终点）。"""

_ACTION_TO_JOB_TYPE: dict[str, str] = {v: k for k, v in JOB_TYPE_TO_ACTION.items()}
"""协议动作类型 → 作业动作 type（``JOB_TYPE_TO_ACTION`` 的逆映射）。"""


@dataclass
class InferConfig:
    """在线推理配置（默认值即 ``configs/infer_v1.yaml`` 的同名字段）。

    噪声闸门（speed_semantic_gate / retreat_* / repeat_guard）针对的是真机闭环
    实测的两类无效动作噪声：倍速反复切换与误撤退。均默认开启；设回关闭值
    （注释里标注）即退化为 v1 行为，丧失噪声削减但不影响合法性层的其它规则。
    """

    decision_interval_s: float = 1.0
    """决策节拍（秒）：两次 greedy_step 最小间隔；本模块只携带逻辑值（调用方执行）。"""
    total_proxy: int = 30
    """state_feat ``step`` 维的序列总长代理（在线不知作业总长，取训练分布周边常数）。"""
    cost_margin: float = 0.0
    """费用合法性余量：``hand.cost <= costs + cost_margin`` 视为可部署。"""
    allow_speedup: bool = True
    """speed（倍速切换）动作开关。"""
    allow_bullet_time: bool = True
    """bullet_time（子弹时间）动作开关。"""
    allow_vocab_targets: bool = True
    """全局词表目标（支援装置等无法从 battle_state 核验的部署）整体放行开关。"""
    speed_semantic_gate: bool = True
    """倍速语义闸门（次数不限、只在必要时切换，替代按次数限流）：

    - **升 2x**：当且仅当本拍无任何合法的有意义动作（deploy / skill / retreat /
      bullet_time 全部候选非法或推断为 None）——「纯等待的死亡时间」才允许
      SpeedUp 升速；
    - **降 1x**：当且仅当本拍模型正要下发真实动作（deploy / skill / retreat）
      且当前 ``in_speedup`` 为真、且 ``slow_for_action`` 开启——此时直接把
      SpeedUp（降速）顶到输出，动作留到 1x 下再发；
    - 两次换速之间受 ``speed_min_interval_s`` 迟滞保护；迟滞期内换速被屏蔽，
      解码回落到其余合法类型（如实声明：迟滞优先于慢速契约）。

    关闭（False）即回到 `allow_speedup` 的纯开关语义。"""
    slow_for_action: bool = True
    """下发真实动作前优先降到 1x（reason：MAA 拖拽部署在 2x 下时序更紧张）。"""
    speed_min_interval_s: float = 5.0
    """两次倍速切换的最小间隔（迟滞防抖），按步序号 × ``decision_interval_s``
    折算（历史只记动作不记时间戳，在线快照取到 t_ms 时以快照为准——本模块
    决定用步序近似，属已知简化、如实记录）。"""
    retreat_min_logit_margin: float = 1.0
    """撤退余量闸门：retreat 仅在 ``retreat logit − 其余类型最高 logit ≥ 本值``
    时允许；默认 1.0（撤退占训练帧 <4%，真机实测误撤退是主要噪声之一，
    让「模型只是微微偏向撤退」的帧自然回落到次优动作）。``<= 0`` 即关闭。"""
    retreat_min_interval_steps: int = 3
    """两次撤退的最小步数间隔（防抖：模型在同一局势下连撤是典型自回归失稳）。
    ``<= 1`` 即关闭。"""
    repeat_guard: int = 3
    """重复护栏：同一 (type, 目标名) 在已执行历史末尾连续出现 N 次，屏蔽该类
    重复（(speed 等无目标语义的动作按 type 单独计）。``<= 0`` 即关闭。"""
    device: str = "cpu"
    """推理设备（冒烟模型参数仅数 M，真机在线 CPU 足够）。"""

    @classmethod
    def from_yaml(cls, path: Path | str) -> InferConfig:
        """从 yaml 加载（嵌套布局见 ``configs/infer_v1.yaml``；缺省键取默认值）。"""
        default = cls()
        with Path(path).open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        dec: dict[str, Any] = raw.get("decision") or {}
        af: dict[str, Any] = raw.get("action_filter") or {}
        ng: dict[str, Any] = raw.get("noise_gate") or {}
        return cls(
            decision_interval_s=float(dec.get("interval_s", default.decision_interval_s)),
            total_proxy=int(dec.get("total_proxy", default.total_proxy)),
            cost_margin=float(dec.get("cost_margin", default.cost_margin)),
            allow_speedup=bool(af.get("allow_speedup", default.allow_speedup)),
            allow_bullet_time=bool(af.get("allow_bullet_time", default.allow_bullet_time)),
            allow_vocab_targets=bool(af.get("allow_vocab_targets", default.allow_vocab_targets)),
            speed_semantic_gate=bool(ng.get("speed_semantic_gate", default.speed_semantic_gate)),
            slow_for_action=bool(ng.get("slow_for_action", default.slow_for_action)),
            speed_min_interval_s=float(
                ng.get("speed_min_interval_s", default.speed_min_interval_s)
            ),
            retreat_min_logit_margin=float(
                ng.get("retreat_min_logit_margin", default.retreat_min_logit_margin)
            ),
            retreat_min_interval_steps=int(
                ng.get("retreat_min_interval_steps", default.retreat_min_interval_steps)
            ),
            repeat_guard=int(ng.get("repeat_guard", default.repeat_guard)),
            device=str(raw.get("device", default.device)),
        )


@dataclass
class LoadedModel:
    """:func:`load_track_a_v1` 的产物：eval 模式模型 + 冻结词表 + ckpt 元信息。"""

    model: TrackAPolicyV1
    vocab: Vocab
    meta: dict[str, Any]
    """ckpt 附带的训练配置与维度信息（d_model / featvec_dim / state_dim / n_params 等）。"""


def load_track_a_v1(ckpt_path: Path | str, *, device: str = "cpu") -> LoadedModel:
    """加载 track_a_v1 checkpoint（加载范式照抄 ``imitate_v1`` 的续训段）。

    词表与 featvec/state 维度随 checkpoint 冻结；结构超参缺失项回落到
    ``TrackAPolicyV1`` 自身默认值；严格加载（``strict=True``，形状不符即报错）。
    """
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: dict[str, Any] = blob.get("config") or {}
    vocab = Vocab(extra_tokens=[str(x) for x in blob["vocab"]])
    model_kwargs = {
        k: cfg[k]
        for k in ("d_model", "n_heads", "n_layers", "n_dec_layers", "dropout")
        if k in cfg
    }
    grid_channels = int(blob.get("grid_channels", GRID_CHANNELS))
    model = TrackAPolicyV1(
        featvec_dim=int(blob["featvec_dim"]),
        vocab_size=vocab.size,
        max_squad=MAX_SQUAD,
        state_dim=int(blob["state_dim"]),
        grid_channels=grid_channels,
        **model_kwargs,  # type: ignore[arg-type]
    ).to(torch.device(device))
    model.load_state_dict(blob["state_dict"], strict=True)
    model.eval()
    meta = {
        k: blob[k]
        for k in (
            "config", "featvec_dim", "state_dim", "n_params",
            "best_epoch", "best_holdout", "holdout_stages", "grid_channels",
        )
        if k in blob
    }
    meta["grid_channels"] = grid_channels  # 老 ckpt 无键时补回落值，供调用方渲染 ctx
    return LoadedModel(model=model, vocab=vocab, meta=meta)


# ---------------------------------------------------------------- 关卡静态实体


@dataclass
class StageContext:
    """一关 + 一支编队的静态实体（构造一次，整局复用）。

    张量均为 numpy（greedy_step 内再转 torch 并加 batch 维）。
    """

    stage_id: str
    rows: int
    cols: int
    vocab: Vocab
    """非干员目标词表（必须与 ckpt 冻结词表一致：两侧都以训练 npz 为准）。"""
    squad_names: list[str]
    """编队槽位的作业中文名（槽位序即指针序），用于合法性核验与动作回写。"""
    squad_tokens: list[str]
    """与训练编码同口径的槽位 token（char_id 或 ``name:<中文名>`` 占位）。"""
    name2cid: dict[str, str]
    """featvec 的中文名 → char_id 映射（历史动作编码用）。"""
    cells: np.ndarray
    """格子特征 [n_cells, C] float32（grid row-major 展平；C = ckpt meta 的
    ``grid_channels``——5 = 纯地形口径，8 = 追加 routefeat 路线几何通道）。"""
    cell_pos: np.ndarray
    """归一化 (r, c) 坐标 [n_cells, 2] float32。"""
    spawns: np.ndarray
    """出怪特征 [n_spawns, 4] float32（与训练编码同口径归一化）。"""
    squad_feat: np.ndarray
    """编队 featvec [n_squad, featvec_dim] float32（未登录槽位零向量）。"""
    squad_pos: dict[str, int] = field(init=False)
    """槽位 token → 槽位序号（历史动作目标编码用）。"""

    def __post_init__(self) -> None:
        self.squad_pos = {tok: i for i, tok in enumerate(self.squad_tokens)}

    @classmethod
    def from_stage_id(
        cls,
        stage_id: str,
        squad_names: list[str],
        *,
        npz_path: Path | None = None,
        featvec_override: np.ndarray | None = None,
        processed_dir: Path | None = None,
        vocab: Vocab | None = None,
        grid_channels: int = GRID_CHANNELS,
    ) -> StageContext:
        """一次性构造关卡静态实体（cells / spawns / squad 等；整局可缓存复用）。

        - ``npz_path``：训练用数据集 npz，featvec 矩阵与词表随其冻结
          （**行序纪律见模块 docstring**）；
        - ``featvec_override``：显式给的 featvec 矩阵（此时必须显式给 ``vocab``）；
        - ``processed_dir``：stagefeat / featvec parquet 所在目录
          （默认取 ``npz_path.parent``，生产布局两者同目录）；
        - ``vocab``：显式词表；npz 路径下默认取 npz 冻结词表；
        - ``grid_channels``：cells 渲染通道数，**必须与 ckpt meta 的
          ``grid_channels`` 一致**（8 时追加 routefeat 路线几何通道，与训练
          ``state_action_data.encode_episode`` 同函数派生）；npz 自带
          ``grid_channels`` 元数据且与本参数不符时 fail fast。

        parquet 只用于「中文名 → char_id → 行号」映射：npz 路径下与其做矩阵全等
        校验，不一致直接报错（不做静默降级，同 ``build_dataset_delta`` 口径）。
        entity encoder 输出未做缓存——缓存要求复制 forward 尾部逻辑，与 policy_v1
        漂移的风险大于收益；静态 numpy 张量已随本对象复用，冒烟层到此为止。
        """
        if npz_path is None and featvec_override is None:
            raise ValueError("npz_path 与 featvec_override 至少给一个")
        if processed_dir is None:
            if npz_path is None:
                raise ValueError("无 npz_path 时必须显式给 processed_dir")
            processed_dir = npz_path.parent
        stagefeat = _load_stagefeat(processed_dir / "stagefeat.parquet")
        sf = stagefeat.get(stage_id) or stagefeat.get(base_stage_id(stage_id))
        if sf is None:
            raise ValueError(f"stagefeat 中找不到关卡 {stage_id}")
        name2cid = _load_name_map(processed_dir / "featvec_operators.parquet")
        feat_ids, featvec_now = _load_featvec(processed_dir / "featvec_operators.parquet")
        featvec_idx = {cid: i for i, cid in enumerate(feat_ids)}

        if npz_path is not None:
            z = np.load(npz_path, allow_pickle=True)
            featvec_mat = np.asarray(z["featvec"], dtype=np.float32)
            if vocab is None:
                vocab = Vocab(extra_tokens=[str(x) for x in z["vocab"]])
            if featvec_now.shape != featvec_mat.shape or not np.array_equal(
                featvec_now, featvec_mat
            ):
                raise ValueError("npz 冻结 featvec 与当前 parquet 不一致，请全量重建数据集")
            if "grid_channels" in z and int(z["grid_channels"]) != grid_channels:
                raise ValueError(
                    f"npz 元数据 grid_channels={int(z['grid_channels'])} 与请求 "
                    f"{grid_channels} 不符（ckpt 与 npz 混套了，请检查配套产物）"
                )
        else:
            if vocab is None:
                raise ValueError("featvec_override 路径必须显式给 vocab")
            featvec_mat = np.asarray(featvec_override, dtype=np.float32)

        rows, cols = int(sf["rows"]), int(sf["cols"])
        n_cells = rows * cols
        if n_cells > MAX_CELLS:
            raise ValueError(f"格数 {n_cells} 超训练上限 MAX_CELLS={MAX_CELLS}，本关卡不支持")
        cells = np.asarray(sf["grid"], dtype=np.float32).reshape(n_cells, GRID_CHANNELS)
        if grid_channels > GRID_CHANNELS:
            # 无可用路线几何的关 routefeat 内部退化为全零（与训练侧降级口径一致）
            route_feats, _ = route_feature_channels(sf)
            cells = np.concatenate([cells, route_feats], axis=1).astype(np.float32)
        rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        cell_pos = np.stack(
            [rr.reshape(-1) / max(1, rows - 1), cc.reshape(-1) / max(1, cols - 1)],
            axis=1,
        ).astype(np.float32)

        spawns_raw = sf.get("spawns") or []
        if spawns_raw:
            sp = np.asarray(
                [
                    [float(s.get("t_ms") or 0) / 1000.0, float(s.get("wave") or 0),
                     float(s.get("route") or 0), 1.0 if s.get("enemy_matched") else 0.0]
                    for s in spawns_raw
                ],
                dtype=np.float32,
            )
            sp[:, 0] = sp[:, 0] / max(1.0, float(sp[:, 0].max()))
            sp[:, 1] = sp[:, 1] / max(1.0, float(sp[:, 1].max()))
            sp[:, 2] = sp[:, 2] / max(1.0, float(sp[:, 2].max()))
        else:
            sp = np.zeros((0, 4), dtype=np.float32)

        names = list(squad_names)[:MAX_SQUAD]
        if not names:
            raise ValueError("编队为空")
        squad_tokens: list[str] = []
        squad_feat = np.zeros((len(names), featvec_mat.shape[1]), dtype=np.float32)
        for j, nm in enumerate(names):
            cid = name2cid.get(nm)
            if cid is None:
                squad_tokens.append(f"name:{nm}")  # 未登录保留槽位（同训练口径）
                continue
            squad_tokens.append(cid)
            row_idx = featvec_idx.get(cid, -1)
            if row_idx >= 0:
                squad_feat[j] = featvec_mat[row_idx]

        return cls(
            stage_id=stage_id, rows=rows, cols=cols, vocab=vocab,
            squad_names=names, squad_tokens=squad_tokens, name2cid=name2cid,
            cells=cells, cell_pos=cell_pos, spawns=sp, squad_feat=squad_feat,
        )

    @classmethod
    def from_record(
        cls,
        record: dict[str, Any],
        featvec: np.ndarray,
        vocab: Vocab,
        *,
        squad_names: list[str] | None = None,
    ) -> StageContext:
        """从训练 npz 的一条 record 构造（回放/评测冒烟用）。

        record 不含干员中文名：``squad_names`` 缺省时以 ``slot<i>`` 占位；
        占位名匹配不到实名 hand，deploy 多被过滤——仅作链路冒烟用途。
        """
        featvec_mat = np.asarray(featvec, dtype=np.float32)
        squad_idx = np.asarray(record["squad_idx"], dtype=np.int64)
        n = int(len(squad_idx))
        names = (
            list(squad_names)[:n]
            if squad_names is not None
            else [f"slot{i}" for i in range(n)]
        )
        squad_feat = np.zeros((n, featvec_mat.shape[1]), dtype=np.float32)
        for j, idx in enumerate(squad_idx):
            if idx >= 0:
                squad_feat[j] = featvec_mat[idx]
        return cls(
            stage_id=str(record["stage_id"]),
            rows=int(record["rows"]),
            cols=int(record["cols"]),
            vocab=vocab,
            squad_names=names,
            squad_tokens=[f"slot:{i}" for i in range(n)],
            name2cid={},
            cells=np.asarray(record["cells"], dtype=np.float32),
            cell_pos=np.asarray(record["cell_pos"], dtype=np.float32),
            spawns=np.asarray(record["spawns"], dtype=np.float32),
            squad_feat=squad_feat,
        )


# ---------------------------------------------------------------- 坐标与动作编码


def xy_to_cell(location: list[int] | tuple[int, int], rows: int, cols: int) -> int:
    """作业 deploy ``[x, y]`` → 扁平格子 idx（越界返回 0，与训练编码同口径）。"""
    r, c = normalize_deploy_location(location)
    if 0 <= r < rows and 0 <= c < cols:
        return r * cols + c
    return 0


def cell_to_xy(cell_idx: int, cols: int) -> list[int]:
    """扁平格子 idx → 作业 ``[x, y]``（``x=col, y=row``，与 :func:`xy_to_cell` 互逆）。"""
    r, c = divmod(int(cell_idx), int(cols))
    return [c, r]


def encode_action_for_history(
    action: dict[str, Any], ctx: StageContext
) -> tuple[int, int, int, int]:
    """MAA 作业形态动作 → (type_id, target_id, cell_id, dir_id)，历史编码入口。

    与训练编码（``state_action_data.encode_episode`` 的动作段）**严格同口径**；
    未登录目标落 UNK（在线无统计侧，静默同训练外推口径）。
    """
    a_type = JOB_TYPE_TO_ACTION[str(action.get("type"))]
    type_id = TYPE_TO_IDX[a_type]

    target_id = UNK_TARGET
    name = action.get("name")
    if name:
        nm = str(name)
        cid = ctx.name2cid.get(nm)
        slot = ctx.squad_pos.get(cid) if cid is not None else None
        if slot is None:
            slot = ctx.squad_pos.get(f"name:{nm}")
        if slot is not None:
            target_id = 1 + ctx.vocab.size + slot
        else:
            vi = ctx.vocab.index_of(f"name:{nm}")
            if vi is not None:
                target_id = vi

    loc = action.get("location")
    cell_id = xy_to_cell(loc, ctx.rows, ctx.cols) if loc else 0

    d_raw = action.get("direction")
    d = DIR_ALIASES.get(str(d_raw) if d_raw is not None else "None")
    dir_id = DIR_TO_IDX[d if d is not None else "None"]
    return type_id, target_id, cell_id, dir_id


# ---------------------------------------------------------------- 合法性解码


def _decode_target_name(target_idx: int, ctx: StageContext) -> str:
    """目标指针 → 作业目标名字（槽位取编队中文名；词表 token 去 ``name:`` 前缀）。"""
    if target_idx >= 1 + ctx.vocab.size:
        slot = target_idx - 1 - ctx.vocab.size
        return ctx.squad_names[slot] if slot < len(ctx.squad_names) else ""
    if 1 <= target_idx < ctx.vocab.size:
        token = ctx.vocab.extra_tokens[target_idx - 1]
        return token[5:] if token.startswith("name:") else token
    return ""


# 升速闸门的「有意义动作」集：本拍其中任一类可下发，就不是纯等待死亡时间
_MEANINGFUL_FOR_UP: tuple[str, ...] = ("deploy", "skill", "retreat", "bullet_time")
# 降速闸门的「真实动作」集：这些动作在 2x 下下发时序紧张，先降到 1x
# （bullet_time 本义就是放慢观察，不在其内）
_MEANINGFUL_FOR_DOWN: tuple[str, ...] = ("deploy", "skill", "retreat")


def _trailing_repeat(history: list[dict[str, Any]]) -> tuple[str | None, str, int]:
    """历史末尾同一 (type, 目标名) 的连续重复 → (协议类型, 名字, 连击长度)。

    空历史或未知作业类型返回 (None, "", 0)；speed/bullet_time 名目恒为 ""。
    """
    if not history:
        return None, "", 0
    last = history[-1]

    def _key(a: dict[str, Any]) -> tuple[str | None, str]:
        proto = JOB_TYPE_TO_ACTION.get(str(a.get("type")))
        return proto, str(a.get("name") or "")

    p_type, p_name = _key(last)
    if p_type is None:
        return None, "", 0
    n = 0
    for a in reversed(history):
        if _key(a) != (p_type, p_name):
            break
        n += 1
    return p_type, p_name, n


def decode_action(
    type_logits: np.ndarray,
    target_logits: np.ndarray,
    cell_logits: np.ndarray,
    dir_logits: np.ndarray,
    ctx: StageContext,
    battle_state: dict[str, Any],
    config: InferConfig | None = None,
    history: list[dict[str, Any]] | None = None,
    step_index: int = 0,
) -> dict[str, Any] | None:
    """末帧四头 logits → 合法性过滤后的作业形态动作（迭代式受限 argmax）。

    过滤规则见模块 docstring；全部候选类型判死时返回 ``None``（本步无可执行
    动作，由调用方按节拍等待下一次决策）。

    ``history`` / ``step_index``：噪声闸门（倍速迟滞、撤退间隔、重复护栏）需要
    已执行历史；不给（缺省）时各闸门按「无历史」处理（迟滞/间隔恒满足、
    连击长 0），完全回放 v1 既有行为。
    """
    cfg = config or InferConfig()
    type_logits = np.asarray(type_logits, dtype=np.float64)
    target_logits = np.asarray(target_logits, dtype=np.float64)
    cell_logits = np.asarray(cell_logits, dtype=np.float64)
    dir_logits = np.asarray(dir_logits, dtype=np.float64)

    hand = battle_state.get("hand") or []
    hand_by_name = {str(h.get("name")): h for h in hand if h.get("name")}
    deployed = battle_state.get("deployed") or []
    deployed_names = {str(d.get("name")) for d in deployed if d.get("name")}
    costs = float(battle_state.get("costs") or 0.0)

    n_cells = int(ctx.cells.shape[0])
    free_cells = np.asarray(ctx.cells[:, DEPLOYABLE_CHANNEL] > 0.5, dtype=np.bool_)
    for d in deployed:
        r, c = d.get("row"), d.get("col")
        if r is not None and c is not None:
            idx = int(r) * ctx.cols + int(c)
            if 0 <= idx < n_cells:
                free_cells[idx] = False

    squad_start = 1 + ctx.vocab.size
    n_slot_room = len(target_logits) - squad_start

    # ---- 历史侧事实（噪声闸门用；缺省 = 无历史，闸门条件恒松） ----
    hist = history or []
    last_speed_idx: int | None = None
    last_retreat_idx: int | None = None
    for i in range(len(hist) - 1, -1, -1):
        jt = str(hist[i].get("type"))
        if last_speed_idx is None and jt == "SpeedUp":
            last_speed_idx = i
        if last_retreat_idx is None and jt == "Retreat":
            last_retreat_idx = i
        if last_speed_idx is not None and last_retreat_idx is not None:
            break
    speed_elapsed_ok = True
    if last_speed_idx is not None:
        elapsed_s = (step_index - last_speed_idx) * cfg.decision_interval_s
        speed_elapsed_ok = elapsed_s >= cfg.speed_min_interval_s - 1e-9
    repeat_type, repeat_name, repeat_run = _trailing_repeat(hist)

    def slot_deployable(slot: int) -> bool:
        """编队槽位干员此刻是否可部署（在手、可用、非冷却、费用够含余量）。

        v1.1 噪声削减追加：同名干员已在 ``deployed`` 里时屏蔽该名部署
        （手牌理论上会消失，但心跳快照的 hand 滞后是真机实测场景）。"""
        if slot >= len(ctx.squad_names):
            return False
        if ctx.squad_names[slot] in deployed_names:
            return False
        h = hand_by_name.get(ctx.squad_names[slot])
        if h is None:
            return False
        if not h.get("available", False) or h.get("cooling", False):
            return False
        cst = h.get("cost")
        return cst is None or float(cst) <= costs + cfg.cost_margin

    def slot_deployed(slot: int) -> bool:
        return slot < len(ctx.squad_names) and ctx.squad_names[slot] in deployed_names

    def target_mask(act: str) -> np.ndarray:
        """按动作类型构造目标指针合法掩码（非法位 = False）。

        重复护栏：历史末尾 (act, 名字) 连击已达 ``repeat_guard`` 次时，本拍从
        掩码里剔除该名字对应的候选（其余候选不受影响，避免整类判死）。
        """
        mask = np.zeros(len(target_logits), dtype=np.bool_)
        if act == "deploy":
            if cfg.allow_vocab_targets:
                mask[1:ctx.vocab.size] = True
            for slot in range(min(len(ctx.squad_names), n_slot_room)):
                if slot_deployable(slot):
                    mask[squad_start + slot] = True
        elif act in ("skill", "retreat"):
            for slot in range(min(len(ctx.squad_names), n_slot_room)):
                if slot_deployed(slot):
                    mask[squad_start + slot] = True
        else:
            mask[:] = True  # speed / bullet_time：target 无目标语义，不约束
        if (
            cfg.repeat_guard > 0
            and repeat_run >= cfg.repeat_guard
            and repeat_type == act
            and repeat_name
        ):
            for slot in range(min(len(ctx.squad_names), n_slot_room)):
                if ctx.squad_names[slot] == repeat_name:
                    mask[squad_start + slot] = False
            for vi in range(1, ctx.vocab.size):
                if _decode_target_name(vi, ctx) == repeat_name:
                    mask[vi] = False
        return mask

    def type_has_target(act: str) -> bool:
        """该动作类型此刻是否有任何合法目标候选（目标层可行性）。"""
        return bool(target_mask(act).any())

    def type_base_allowed(i: int, act: str) -> bool:
        """合法性层（不含噪声闸门）的类型层可行性。"""
        if i >= len(type_logits):
            return False
        if act == "deploy":
            return bool(free_cells.any())
        if act in ("skill", "retreat"):
            return any(slot_deployed(s) for s in range(len(ctx.squad_names)))
        if act == "speed":
            return cfg.allow_speedup
        if act == "bullet_time":
            return cfg.allow_bullet_time
        return False

    # ---- 各类型的有效可行性（类型层 ∧ 目标层 ∧ 闸门） ----
    feasible: dict[str, bool] = {}
    for i, act in enumerate(ACTION_TYPES):
        ok = type_base_allowed(i, act) and type_has_target(act)
        if ok and act == "retreat":
            # 撤退余量闸门：retreat 的型 logit 必须压过其余类型最高 logit 一个余量
            if cfg.retreat_min_logit_margin > 0.0:
                others = max(
                    (float(type_logits[j]) for j, a2 in enumerate(ACTION_TYPES)
                     if a2 != "retreat" and j < len(type_logits)),
                    default=float("-inf"),
                )
                if float(type_logits[i]) - others < cfg.retreat_min_logit_margin:
                    ok = False
            # 撤退最小间隔（防抖）
            if (
                ok
                and cfg.retreat_min_interval_steps > 1
                and last_retreat_idx is not None
                and step_index - last_retreat_idx < cfg.retreat_min_interval_steps
            ):
                ok = False
        feasible[act] = ok

    # 无目标命名语义的重复护栏：speed/bullet_time 按 type 连击整类屏蔽
    for act in ("speed", "bullet_time"):
        if (
            cfg.repeat_guard > 0
            and repeat_run >= cfg.repeat_guard
            and repeat_type == act
        ):
            feasible[act] = False

    # ---- 倍速语义闸门（次数不限、只在必要时切换；关闭则回落 allow_speedup） ----
    in_2x = bool(battle_state.get("in_speedup", False))
    if cfg.speed_semantic_gate and cfg.allow_speedup:
        any_real_up = any(feasible.get(a) for a in _MEANINGFUL_FOR_UP)
        any_real_down = any(feasible.get(a) for a in _MEANINGFUL_FOR_DOWN)
        if in_2x:
            if cfg.slow_for_action and any_real_down:
                if speed_elapsed_ok:
                    # 正要下发真实动作且当前 2x：本拍只降速，动作留到 1x 下再发
                    return {"type": "SpeedUp"}
                # 迟滞期禁换速：如实回落普通解码（可能在 2x 下下发动作，
                # 迟滞防抖优先于慢速契约，见 InferConfig docstring）
            feasible["speed"] = False
        else:
            # 纯等待死亡时间（无任何合法有意义动作）才允许升 2x；否则无关换速一律屏蔽
            feasible["speed"] = (not any_real_up) and speed_elapsed_ok

    allowed = [
        (float(type_logits[i]), i)
        for i, act in enumerate(ACTION_TYPES)
        if feasible.get(act)
    ]
    for _score, t_idx in sorted(allowed, key=lambda x: -x[0]):
        act = ACTION_TYPES[t_idx]
        mask = target_mask(act)
        target_idx = int(np.argmax(np.where(mask, target_logits, -np.inf)))
        if act == "deploy":
            cell_idx = int(np.argmax(np.where(free_cells, cell_logits, -np.inf)))
            dir_idx = int(np.argmax(dir_logits)) if len(dir_logits) else 0
            return {
                "type": "Deploy",
                "name": _decode_target_name(target_idx, ctx),
                "location": cell_to_xy(cell_idx, ctx.cols),
                "direction": DIRECTIONS[dir_idx] if dir_idx < len(DIRECTIONS) else "None",
            }
        if act in ("skill", "retreat"):
            return {
                "type": _ACTION_TO_JOB_TYPE[act],
                "name": _decode_target_name(target_idx, ctx),
            }
        return {"type": _ACTION_TO_JOB_TYPE[act]}
    return None


# ---------------------------------------------------------------- 单步贪心推理


@torch.no_grad()
def greedy_step(
    model: TrackAPolicyV1,
    ctx: StageContext,
    battle_state: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    t: int,
    config: InferConfig | None = None,
    total_proxy: int | None = None,
) -> dict[str, Any] | None:
    """跑一步贪心解码：构造输入 → 前向取末帧 → 合法性过滤 → 作业形态动作。

    - ``history``：已执行动作（作业形态 dict，经 ``encode_action_for_history``
      编码后与训练历史右移口径一致：序列 = BOS + 全部已执行动作）；
    - ``t``：当前步序号（仅用于 state_feat 的 ``step`` 维；history 长度决定序列长）；
    - ``total_proxy``：``step`` 维的序列总长代理，缺省取 ``config.total_proxy``；
    - 返回 ``None``：不在战斗中（``in_battle`` 为假）或全部候选被合法性层判死。
    """
    cfg = config or InferConfig()
    if not battle_state.get("in_battle", True):
        return None
    model_gc = getattr(model, "grid_channels", GRID_CHANNELS)
    if int(ctx.cells.shape[1]) != int(model_gc):
        raise ValueError(
            f"ctx.cells 通道数 {int(ctx.cells.shape[1])} 与模型 grid_channels="
            f"{int(model_gc)} 不符：StageContext 须按 ckpt meta 的 grid_channels "
            "构造（fail fast，不做静默降级）"
        )
    total = total_proxy if total_proxy is not None else cfg.total_proxy

    enc = [encode_action_for_history(a, ctx) for a in history]
    sv = state_vector(battle_state, t, total, len(ctx.squad_names))
    device = next(model.parameters()).device
    seq = len(enc) + 1

    def _lt(ids: list[int]) -> torch.Tensor:
        return torch.as_tensor([ids], dtype=torch.long, device=device)

    # 历史右移 + 首列 BOS（与 imitate.shift_history 严格同口径：BOS 的
    # type 用 +len 哨兵槽，target/dir 用 0，cell 用 idx/MAX_CELLS 的 (0, 0)）
    h_type = _lt([len(ACTION_TYPES)] + [e[0] for e in enc])
    h_target = _lt([0] + [e[1] for e in enc])
    h_dir = _lt([0] + [e[3] for e in enc])
    cell_idx = torch.as_tensor(
        [[0] + [e[2] for e in enc]], dtype=torch.float32, device=device
    )
    h_cell = torch.stack([cell_idx / float(MAX_CELLS), torch.zeros_like(cell_idx)], dim=-1)
    # hist_state 平铺当前快照、不右移（诚实简化，见模块 docstring）
    hist_state = torch.as_tensor(
        np.tile(sv, (seq, 1))[None, :, :], dtype=torch.float32, device=device
    )

    def _ft(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a)[None], dtype=torch.float32, device=device)

    model.eval()
    out = model(
        _ft(ctx.cells),
        _ft(ctx.cell_pos),
        torch.ones((1, ctx.cells.shape[0]), dtype=torch.bool, device=device),
        _ft(ctx.spawns),
        torch.ones((1, ctx.spawns.shape[0]), dtype=torch.bool, device=device),
        _ft(ctx.squad_feat),
        torch.ones((1, len(ctx.squad_names)), dtype=torch.bool, device=device),
        h_type,
        h_target,
        h_cell,
        h_dir,
        hist_state,
    )
    last = {k: v[0, -1].detach().float().cpu().numpy() for k, v in out.items()}
    return decode_action(
        last["type"], last["target"], last["cell"], last["dir"], ctx, battle_state, cfg,
        history=history, step_index=t,
    )
