"""共享权重的多用途训练与全量评测；每份资料必须产生可核对的优化步骤。"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from ..models.track_a.all_fields import AllFieldsPolicy
from .imitate import TrainConfig, compute_loss, set_seed, to_device
from .imitate_v1 import MetricsV1, collate_v1, forward_batch_v1
from .pretrain_ssl import load_pretrained_into
from .rotation import write_report


def atomic_checkpoint(path: Path, value: dict) -> None:
    """写完并刷盘后替换，进程退出或磁盘异常保留上一轮完整检查点。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def select_targets(valid: Tensor, rate: float, gen: torch.Generator, evaluate: bool) -> Tensor:
    """训练只遮真实观测；评测覆盖所有有效位置，无条数截断。"""
    if evaluate:
        return valid.clone()
    selected = (torch.rand(valid.shape, generator=gen) < rate).to(valid.device) & valid
    if valid.ndim == 3:
        for row in range(valid.shape[1]):
            options = valid[0, row].nonzero(as_tuple=False).flatten()
            if len(options) and not bool(selected[0, row].any()):
                selected[0, row, options[int(torch.randint(len(options), (1,), generator=gen))]] = (
                    True
                )
    elif bool(valid.any()) and not bool(selected.any()):
        options = valid.nonzero(as_tuple=False)
        chosen = options[int(torch.randint(len(options), (1,), generator=gen))]
        selected[tuple(chosen)] = True
    return selected


def one_loss(
    model: AllFieldsPolicy,
    record: dict,
    batch: dict[str, Tensor],
    cfg: TrainConfig,
    auxiliary: dict,
    gen: torch.Generator,
    *,
    evaluate: bool = False,
) -> tuple[Tensor, dict, dict[str, Tensor] | None]:
    """有证据的字段才产生损失；计划重构与实际策略模仿分别计数。"""
    device = batch["cells"].device
    states = torch.as_tensor(record["observed_states"], device=device).unsqueeze(0)
    observed = torch.as_tensor(record["observed_mask"], device=device).unsqueeze(0)
    planned = torch.as_tensor(record["plan_type"], device=device).unsqueeze(0)
    plan_valid = torch.as_tensor(record["plan_mask"], device=device).unsqueeze(0)
    squad_valid = batch["squad_mask"] & torch.as_tensor(
        record["squad_idx"] >= 0, device=device
    ).unsqueeze(0)
    rate = float(auxiliary["mask_rate"])
    selections = {
        "cells": select_targets(batch["cell_mask"], rate, gen, evaluate),
        "spawns": select_targets(batch["spawn_mask"], rate, gen, evaluate),
        "squad": select_targets(squad_valid, rate, gen, evaluate),
        "states": select_targets(observed, rate, gen, evaluate),
        "plan": select_targets(plan_valid, rate, gen, evaluate),
    }
    out = model.auxiliary(batch, states, observed, selections, planned)
    total = out["cells"].sum() * 0
    metrics: dict[str, Any] = {}
    for key, gold in (
        ("cells", batch["cells"]),
        ("spawns", batch["spawns"]),
        ("squad", batch["squad_feat"]),
        ("states", states),
    ):
        valid = selections[key]
        if key != "states":
            valid = valid.unsqueeze(-1).expand_as(gold)
        count = int(valid.sum())
        value = (
            F.huber_loss(
                out[key][valid],
                gold[valid],
                reduction="mean",
                delta=float(auxiliary["huber_delta"]),
            )
            if count
            else total * 0
        )
        total = total + float(auxiliary["weights"][key]) * value
        metrics[key] = {"n": count, "loss_sum": float(value.detach()) * count}
    count = int(selections["plan"].sum())
    if count:
        value = F.cross_entropy(out["plan"][selections["plan"]], planned[selections["plan"]])
        total = total + float(auxiliary["weights"]["plan"]) * value
    metrics["plan"] = {
        "n": count,
        "correct": int(
            (out["plan"].argmax(-1)[selections["plan"]] == planned[selections["plan"]]).sum()
        ),
    }
    classification = out
    if evaluate:
        clean = {key: torch.zeros_like(value) for key, value in selections.items()}
        classification = model.auxiliary(batch, states, observed, clean, planned)
    for key in ("outcome", "marker", "failure"):
        target = int(record[key])
        if target >= 0:
            label = torch.tensor([target], device=device)
            value = F.cross_entropy(classification[key], label)
            total = total + float(auxiliary["weights"][key]) * value
            metrics[key] = {"n": 1, "correct": int(classification[key].argmax(-1).item() == target)}
        else:
            metrics[key] = {"n": 0, "correct": 0}
    policy = None
    if record["policy_allowed"]:
        policy = forward_batch_v1(model.policy, batch)
        value, _ = compute_loss(policy, batch, cfg)
        total = total + float(auxiliary["weights"]["policy"]) * value
    metrics["policy"] = {"n": len(record["type"]) if policy is not None else 0}
    return total, metrics, policy


@torch.no_grad()
def evaluate_all(
    model: AllFieldsPolicy,
    records: list[dict],
    batches: dict[str, dict],
    cfg: TrainConfig,
    auxiliary: dict,
    device: torch.device,
) -> dict:
    """逐份全量评测；无结局的记录不进入胜负分母，无状态的记录不进入状态分母。"""
    model.eval()
    counts: dict[str, Counter] = {}
    policy_metrics = MetricsV1()
    per_record = []
    gen = torch.Generator().manual_seed(cfg.seed)
    for record in records:
        batch = to_device(batches[record["coverage_id"]], device)
        loss, measured, policy = one_loss(model, record, batch, cfg, auxiliary, gen, evaluate=True)
        for key, values in measured.items():
            counts.setdefault(key, Counter()).update(values)
        if policy is not None:
            policy_metrics.add_batch(policy, batch, float(loss))
        per_record.append({"id": record["coverage_id"], "measured": measured})
    if len(per_record) != len(records):
        raise ValueError("全量评测漏记资料")
    tasks = {}
    for key, values in counts.items():
        tasks[key] = dict(values)
        if values.get("n"):
            if "loss_sum" in values:
                tasks[key]["mean_huber"] = values["loss_sum"] / values["n"]
            if "correct" in values:
                tasks[key]["accuracy"] = values["correct"] / values["n"]
    model.train()
    return {
        "records": len(records),
        "tasks": tasks,
        "policy": policy_metrics.summary(),
        "per_record": per_record,
    }


def train_all_fields(
    records: list[dict],
    featvec: np.ndarray,
    vocab: list[str],
    build_report: dict,
    cfg: TrainConfig,
    auxiliary: dict,
    *,
    train_ids: set[str],
    eval_ids: set[str],
    output: Path,
    device_name: str,
    log: Callable = print,
    initialize: Path | None = None,
) -> dict:
    """真实执行每份资料的优化步骤，轮末落检查点，异常退出后继续未完成轮次。"""
    output.mkdir(parents=True, exist_ok=True)
    if cfg.epochs < 1 or not (0 < float(auxiliary["mask_rate"]) <= 1):
        raise ValueError("训练轮数必须为正，掩码率必须在(0,1]之间")
    if float(auxiliary["weights"]["cells"]) <= 0:
        raise ValueError("所有资料共有的静态重构损失必须开启，不能仅登记空覆盖")
    if any(float(value) < 0 for value in auxiliary["weights"].values()):
        raise ValueError("损失权重不能为负")
    set_seed(cfg.seed)
    device = torch.device(device_name)
    model = AllFieldsPolicy(
        featvec_dim=featvec.shape[1],
        vocab_size=len(vocab) + 1,
        max_squad=0,
        state_dim=12,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        n_dec_layers=cfg.n_dec_layers,
        dropout=cfg.dropout,
        n_plan_types=len(build_report["plan_types"]) + 1,
        n_failure_types=len(build_report["failure_types"]) + 1,
    ).to(device)
    training = [r for r in records if r["coverage_id"] in train_ids]
    evaluation = [r for r in records if r["coverage_id"] in eval_ids]
    if len(training) != len(train_ids) or len(evaluation) != len(eval_ids):
        raise ValueError("资料划分丢失或身份重复")
    batches = {r["coverage_id"]: collate_v1([r], featvec, 12) for r in records}
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    generator = torch.Generator().manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    payload = {
        "config": asdict(cfg),
        "auxiliary": auxiliary,
        "train_ids": sorted(train_ids),
        "eval_ids": sorted(eval_ids),
        "vocab": vocab,
        "plan_types": build_report["plan_types"],
        "failure_types": build_report["failure_types"],
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    last = output / "last.pt"
    visits: Counter = Counter()
    applied: dict[str, Counter] = {}
    history = []
    begin = 0
    if last.exists():
        checkpoint = torch.load(last, map_location=device, weights_only=False)
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("已有检查点与本次数据或配置不同，拒绝混用")
        model.load_state_dict(checkpoint["multiuse_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        begin, history = checkpoint["epoch"], checkpoint["history"]
        visits, applied = (
            Counter(checkpoint["visits"]),
            {k: Counter(v) for k, v in checkpoint["applied"].items()},
        )
        rng.bit_generator.state = checkpoint["numpy_rng"]
        generator.set_state(checkpoint["mask_rng"].cpu())
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([row.cpu() for row in checkpoint["cuda_rng"]])
        log(f"[resume] {output.name} 从已完成的第{begin}轮继续")
    elif initialize:
        checkpoint = torch.load(initialize, map_location="cpu", weights_only=False)
        if "multiuse_model" in checkpoint:
            model.load_state_dict(checkpoint["multiuse_model"])
            log("[initialize] 全量模型继承已利用全部资料的辅助预热权重")
        else:
            loaded, skipped = load_pretrained_into(model.policy, checkpoint["state_dict"])
            log(f"[initialize] 既有预训练注入{loaded}个权重，跳过{len(skipped)}个不匹配项")
    for epoch in range(begin, cfg.epochs):
        started = time.monotonic()
        total = 0.0
        model.train()
        for position, index in enumerate(rng.permutation(len(training))):
            record = training[int(index)]
            identity = record["coverage_id"]
            batch = to_device(batches[identity], device)
            optimizer.zero_grad(set_to_none=True)
            loss, components, _ = one_loss(model, record, batch, cfg, auxiliary, generator)
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"非有限损失，资料保留并停止：{identity}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            visits[identity] += 1
            used = applied.setdefault(identity, Counter())
            used.update({key: int(value["n"]) for key, value in components.items()})
            total += float(loss.detach())
            if (position + 1) % int(auxiliary["progress_every_records"]) == 0:
                log(
                    f"[train] {output.name} 轮{epoch + 1}/{cfg.epochs} "
                    f"资料{position + 1}/{len(training)}"
                )
        history.append(
            {
                "epoch": epoch + 1,
                "mean_loss": total / len(training),
                "seconds": time.monotonic() - started,
            }
        )
        atomic_checkpoint(
            last,
            {
                "epoch": epoch + 1,
                "fingerprint": fingerprint,
                "multiuse_model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "visits": dict(visits),
                "applied": {key: dict(value) for key, value in applied.items()},
                "history": history,
                "numpy_rng": rng.bit_generator.state,
                "mask_rng": generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            },
        )
        write_report(
            output / "training_coverage.json",
            {
                "epoch": epoch + 1,
                "records": len(visits),
                "per_record": [
                    {
                        "id": key,
                        "optimizer_steps": visits[key],
                        "applied_targets": dict(applied[key]),
                    }
                    for key in sorted(visits)
                ],
            },
        )
        log(
            f"[epoch] {output.name} {epoch + 1}/{cfg.epochs}，实际使用{len(visits)}份资料，"
            f"loss={history[-1]['mean_loss']:.4f}"
        )
    if set(visits) != train_ids or any(visits[key] != cfg.epochs for key in train_ids):
        raise ValueError("优化步骤覆盖与资料清单不符")
    evaluated = evaluate_all(model, evaluation, batches, cfg, auxiliary, device)
    if {row["id"] for row in evaluated["per_record"]} != eval_ids:
        raise ValueError("评测没有覆盖整份评测资料")
    checkpoint_path = output / "policy.pt"
    atomic_checkpoint(
        checkpoint_path,
        {
            "state_dict": model.policy.state_dict(),
            "config": asdict(cfg),
            "vocab": vocab,
            "state_dim": 12,
            "grid_channels": 5,
            "featvec_dim": featvec.shape[1],
            "max_squad": 0,
            "training_mode": "all_available_fields",
            "training_records": len(training),
            "auxiliary_weights_in_last_checkpoint": True,
        },
    )
    result = {
        "training_records": len(training),
        "evaluation_records": len(evaluation),
        "epochs": cfg.epochs,
        "checkpoint": str(checkpoint_path),
        "history": history,
        "evaluation": evaluated,
        "train_ids": sorted(train_ids),
        "eval_ids": sorted(eval_ids),
        "initialization": str(initialize) if initialize else "fresh_no_data_exposure",
    }
    write_report(output / "report.json", result)
    return result
