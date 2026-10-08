"""模仿学习数据集 v0：episodes + stagefeat + featvec → 可训练张量。

任务定义（v0，**无战场状态**）
------------------------------
给定（关卡静态特征，编队），自回归预测动作序列：``P(a_1..a_T | stage, squad)``。

这是行为克隆在"尚无战场状态"条件下的诚实形态。方案 §8 的目标形态是
``P(a_t | state_t)``，本 v0 用「关卡特征 + 编队 + 动作历史」替代 state_t；
当感知层就绪后，只需向实体集注入新 token 类型（敌人/干员实时状态），
模型结构无需重写。

输入实体（供实体 Transformer 消费）
-----------------------------------
- **格子 token**：关卡每格 5 通道（高度/可部署类型/可部署/起点/终点）+ 归一化 (r,c)；
- **出怪 token**：``spawns`` 的 (t, 敌人, 路线)，提供波次时序先验；
- **编队 token**：编队槽位（featvec 数值向量 + 费用），标识"手上有谁"；
- **动作历史 token**：已执行动作（类型/目标/格子/朝向），是当前 step 的"局势"代理。

输出头（每步）
--------------
- ``type``：deploy / skill / retreat / speed / bullet_time；
- ``target``：指针，指向「编队槽位 ∪ 全局装置与类别词表 ∪ UNK」；
- ``cell``：指针，指向「关卡格子」（仅 deploy 计算损失）；
- ``dir``：Right / Down / Left / Up / None（仅 deploy 计算损失）。

坐标系统一（**关键**）
----------------------
episodes 的 deploy ``location`` 是 ``[x, y] = [col, row]``，而 stagefeat 的 grid 是
``[row, col]``（实测依据见 ADR-0003 §3）。本模块**必须**经
:func:`ark_core.datapipe.stagefeat.normalize_deploy_location` 转换，否则格子指针全错。

目标词表的划分纪律
------------------
全局装置/类别词表**只从训练集构建**（:func:`build_vocab`），再套用到验证/测试集，
避免词表层面的泄漏。测试集里未登录的目标统一落到 ``UNK`` 槽并计数。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ..datapipe.stagefeat import normalize_deploy_location
from ..protocol.action import (
    ACTION_TYPES,
    DIRECTIONS,
    GRID_CHANNELS,
)
from ..protocol.action import (
    DIR_ALIASES as _DIR_ALIASES,
)

# 动作类型 / 朝向 / 格子通道的编码表统一由 protocol 层提供（单一事实来源，
# 并避免 models → training 的反向依赖）。此处仅做本模块内的短别名。
TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(ACTION_TYPES)}
DIR_TO_IDX: dict[str, int] = {d: i for i, d in enumerate(DIRECTIONS)}

UNK_TARGET = 0
"""目标指针的 UNK 槽（索引 0 保留）。"""
MAX_CELLS = 400
"""最大格数。实测 2459 关中 99.5% ≤400（最大 1600 = act2multi_tr02），超者多为
`*multi` 多连战与 `camp` 剿灭这类特殊玩法，与主线形态差异大。"""

MAX_SPAWNS = 400
"""最大出怪数。实测中位 38、最大 1393；>400 的仅 30 关（集中在 camp/multi 族）。
上限从 200 放宽到 400 可多保留约 2300 局（成本仅为注意力序列变长）。"""

MAX_SQUAD = 16
MAX_ACTIONS = 128


@dataclass
class Vocab:
    """全局目标词表（**仅由训练集构建**）。"""

    extra_tokens: list[str] = field(default_factory=list)
    """非干员目标（装置名 / 类别 id）的全局词表，索引从 1 开始。"""

    def index_of(self, token: str) -> int | None:
        """查词表索引；未登录返回 None。"""
        try:
            return self.extra_tokens.index(token) + 1
        except ValueError:
            return None

    @property
    def size(self) -> int:
        """词表槽数（含 UNK）。"""
        return len(self.extra_tokens) + 1


@dataclass
class BuildStats:
    """构建计数（进 build_report）。"""

    n_seen: int = 0
    n_kept: int = 0
    drop_no_squad: int = 0
    drop_no_stage: int = 0
    drop_too_many_cells: int = 0
    drop_too_many_spawns: int = 0
    drop_no_actions: int = 0
    drop_too_many_actions: int = 0
    n_operator_deploy: int = 0
    n_external_deploy: int = 0
    n_target_unk: int = 0
    n_cell_oob: int = 0
    n_dir_unknown: int = 0
    type_hist: Counter = field(default_factory=Counter)


def _load_stagefeat(path: Path) -> dict[str, dict[str, Any]]:
    """读 stagefeat.parquet → {stage_id: row}。"""
    t = pq.read_table(
        path,
        columns=["stage_id", "rows", "cols", "grid", "spawns", "opt_initial_cost"],
    )
    out: dict[str, dict[str, Any]] = {}
    for r in t.to_pylist():
        out[str(r["stage_id"])] = r
    return out


def _load_featvec(path: Path) -> tuple[list[str], np.ndarray]:
    """读干员 featvec → (char_id 列表, 数值矩阵 float32)。

    只取数值列（元信息列不参与），并做列内标准化用不上——featvec 已按配置归一化过。
    """
    t = pq.read_table(path)
    cols = t.column_names
    meta = {"char_id", "name", "rarity_tier", "sub_profession"}
    vec_cols = [c for c in cols if c not in meta]
    ids = [str(x) for x in t.column("char_id").to_pylist()]
    mat = np.zeros((len(ids), len(vec_cols)), dtype=np.float32)
    for j, c in enumerate(vec_cols):
        col = t.column(c).to_pylist()
        mat[:, j] = np.asarray([float(v or 0.0) for v in col], dtype=np.float32)
    return ids, mat


def _action_target_token(a: dict[str, Any]) -> tuple[str | None, bool]:
    """动作的目标 → (词表 token, 是否干员)。

    干员返回 (char_id, True)；装置返回 (名称, False)；类别返回 (category_id, False)；
    其余（unknown_name / 缺失）返回 (None, False) 表示走 UNK。
    """
    tgt = a.get("target") or {}
    kind = str(tgt.get("kind") or "")
    if kind == "operator":
        cid = tgt.get("char_id")
        return (str(cid), True) if cid else (None, False)
    if kind == "device":
        name = tgt.get("name")
        return (f"device:{name}", False) if name else (None, False)
    if kind == "category":
        cid = tgt.get("category_id")
        return (f"category:{cid}", False) if cid else (None, False)
    return (None, False)


def _squad_char_ids(ep: dict[str, Any]) -> list[str]:
    """编队槽位的 char_id（无 id 的占位槽跳过）。"""
    out: list[str] = []
    for sl in ep.get("squad") or []:
        cid = sl.get("char_id")
        if cid:
            out.append(str(cid))
    return out


def iter_episodes(path: Path, limit: int | None = None):
    """流式读 episodes JSONL。"""
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if limit is not None and n >= limit:
                return


def build_vocab(
    episodes_path: Path,
    keep_stage_ids: set[str],
    *,
    top_k: int = 256,
    limit: int | None = None,
) -> Vocab:
    """**只从训练集**构建非干员目标词表（避免词表泄漏）。"""
    counter: Counter = Counter()
    for ep in iter_episodes(episodes_path, limit):
        sid = str((ep.get("stage") or {}).get("stage_id") or "")
        if sid not in keep_stage_ids:
            continue
        for a in ep.get("actions") or []:
            if str(a.get("type")) != "deploy":
                continue
            token, is_op = _action_target_token(a)
            if token and not is_op:
                counter[token] += 1
    return Vocab(extra_tokens=[t for t, _ in counter.most_common(top_k)])


# ---------------------------------------------------------------- 单局编码


def encode_episode(
    ep: dict[str, Any],
    sf: dict[str, Any],
    featvec_idx: dict[str, int],
    vocab: Vocab,
    stats: BuildStats,
) -> dict[str, Any] | None:
    """单局 → 定长/变长张量字典；不合规返回 None（并已在 stats 计数）。"""
    stage = sf
    rows = int(stage["rows"])
    cols = int(stage["cols"])
    n_cells = rows * cols

    if n_cells > MAX_CELLS:
        stats.drop_too_many_cells += 1
        return None

    spawns_raw = stage.get("spawns") or []
    if len(spawns_raw) > MAX_SPAWNS:
        stats.drop_too_many_spawns += 1
        return None

    squad_ids = _squad_char_ids(ep)
    if not squad_ids:
        stats.drop_no_squad += 1
        return None
    if len(squad_ids) > MAX_SQUAD:
        squad_ids = squad_ids[:MAX_SQUAD]

    actions_in = ep.get("actions") or []
    if not actions_in:
        stats.drop_no_actions += 1
        return None
    if len(actions_in) > MAX_ACTIONS:
        stats.drop_too_many_actions += 1
        return None

    # ---- 格子张量：[n_cells, 5] + 归一化坐标 [n_cells, 2] ----
    grid_flat = np.asarray(stage["grid"], dtype=np.float32).reshape(n_cells, GRID_CHANNELS)
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    pos = np.stack(
        [
            rr.reshape(-1) / max(1, rows - 1),
            cc.reshape(-1) / max(1, cols - 1),
        ],
        axis=1,
    ).astype(np.float32)

    # ---- 出怪张量：[n_spawns, 4] = (t, wave, route, 归一化 t) ----
    if spawns_raw:
        sp = np.asarray(
            [
                [float(s.get("t_ms") or 0) / 1000.0, float(s.get("wave") or 0),
                 float(s.get("route") or 0), 1.0 if s.get("enemy_matched") else 0.0]
                for s in spawns_raw
            ],
            dtype=np.float32,
        )
        tmax = max(1.0, float(sp[:, 0].max()))
        sp[:, 0] = sp[:, 0] / tmax
        sp[:, 1] = sp[:, 1] / max(1.0, float(sp[:, 1].max()))
        sp[:, 2] = sp[:, 2] / max(1.0, float(sp[:, 2].max()))
    else:
        sp = np.zeros((0, 4), dtype=np.float32)

    # ---- 编队：featvec 行索引（未知干员 → -1，模型用零向量掩盖） ----
    squad_idx = np.asarray(
        [featvec_idx.get(cid, -1) for cid in squad_ids], dtype=np.int64
    )

    # ---- 动作序列 ----
    type_ids: list[int] = []
    target_ids: list[int] = []
    cell_ids: list[int] = []
    dir_ids: list[int] = []

    squad_pos = {cid: i for i, cid in enumerate(squad_ids)}
    for a in actions_in:
        t = str(a.get("type"))
        if t not in TYPE_TO_IDX:
            continue
        type_ids.append(TYPE_TO_IDX[t])
        stats.type_hist[t] += 1

        # 目标指针。索引布局：[0]=UNK ｜ [1, 1+|extra|)=全局装置/类别 ｜
        # 其后为编队槽位（每局长度不同，模型按 squad_mask 屏蔽）
        token, is_op = _action_target_token(a)
        tidx = UNK_TARGET
        if is_op:
            if token is not None:
                slot = squad_pos.get(token)
                if slot is not None:
                    tidx = 1 + vocab.size + slot
                else:
                    stats.n_target_unk += 1
            else:
                stats.n_target_unk += 1
        elif token is not None:
            vi = vocab.index_of(token)
            if vi is not None:
                tidx = vi
            else:
                stats.n_target_unk += 1
        else:
            stats.n_target_unk += 1
        if is_op and t == "deploy":
            stats.n_operator_deploy += 1
        elif t == "deploy":
            stats.n_external_deploy += 1
        target_ids.append(tidx)

        # 格子指针（deploy 必需；**坐标序转换在此**）
        cidx = 0
        loc = a.get("location")
        if loc and len(loc) >= 2:
            r_, c_ = normalize_deploy_location(loc)
            if 0 <= r_ < rows and 0 <= c_ < cols:
                cidx = r_ * cols + c_
            else:
                stats.n_cell_oob += 1
        cell_ids.append(cidx)

        # 朝向
        d_raw = a.get("direction")
        d = _DIR_ALIASES.get(str(d_raw) if d_raw is not None else "None")
        if d is None:
            stats.n_dir_unknown += 1
            d = "None"
        dir_ids.append(DIR_TO_IDX[d])

    if not type_ids:
        stats.drop_no_actions += 1
        return None

    return {
        "stage_id": str((ep.get("stage") or {}).get("stage_id") or ""),
        "family": str((ep.get("stage") or {}).get("family") or ""),
        "episode_id": str(ep.get("episode_id") or ""),
        "cells": grid_flat,
        "cell_pos": pos,
        "rows": rows,
        "cols": cols,
        "spawns": sp,
        "squad_idx": squad_idx,
        "squad_mask": np.ones(len(squad_ids), dtype=np.bool_),
        "type": np.asarray(type_ids, dtype=np.int64),
        "target": np.asarray(target_ids, dtype=np.int64),
        "cell": np.asarray(cell_ids, dtype=np.int64),
        "dir": np.asarray(dir_ids, dtype=np.int64),
    }


def build_dataset(
    processed_dir: Path,
    episodes_path: Path,
    train_families: set[str],
    *,
    out_path: Path,
    limit: int | None = None,
    vocab_top_k: int = 256,
) -> tuple[Vocab, BuildStats, int]:
    """构建模仿学习数据集并落盘为 ``.npz``（一局一条，变长用 object 数组存）。

    ``train_families`` 决定词表来源（**只看训练家族**），避免词表泄漏。
    """
    stagefeat = _load_stagefeat(processed_dir / "stagefeat.parquet")
    featvec_ids, featvec_mat = _load_featvec(processed_dir / "featvec_operators.parquet")
    featvec_idx = {cid: i for i, cid in enumerate(featvec_ids)}

    # 训练集家族对应的关卡集合（词表只从这里建）
    train_stage_ids = {
        sid for sid, r in stagefeat.items()
        if _family_of_stage(sid) in train_families
    }
    vocab = build_vocab(episodes_path, train_stage_ids, top_k=vocab_top_k, limit=limit)

    stats = BuildStats()
    records: list[dict[str, Any]] = []
    for ep in iter_episodes(episodes_path, limit):
        stats.n_seen += 1
        sid = str((ep.get("stage") or {}).get("stage_id") or "")
        sf = stagefeat.get(sid) or stagefeat.get(sid.split("#", 1)[0])
        if sf is None:
            stats.drop_no_stage += 1
            continue
        rec = encode_episode(ep, sf, featvec_idx, vocab, stats)
        if rec is None:
            continue
        records.append(rec)
        stats.n_kept += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        n=len(records),
        featvec=featvec_mat,
        vocab=np.asarray(vocab.extra_tokens, dtype=object),
        records=np.asarray(records, dtype=object),
    )
    return vocab, stats, len(records)


def _family_of_stage(stage_id: str) -> str:
    """关卡家族（复用 datapipe 口径）。"""
    from ..datapipe.stages_meta import family_of

    return family_of(stage_id)
