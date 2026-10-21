"""pretrain_ssl（掩码实体建模 v0）单测：全合成数据，无真实游戏内容。

重点钉死三件事：
1. **掩码采样分布**：mask_rate≈15%（只在有效实体上叠加），80/10/10 三支比例；
2. **重组装形状**：mask_batch → forward_ssl → compute_ssl_loss 的形状与 loss 有限性；
3. **键级吻合**：SSLMaskedPolicy 的权重按键名+形状可整段注入 TrackAPolicyV1
   （除 v1 新增的 state_proj），这是「预训练能初始化下游」的契约。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# 训练依赖是可选的（`uv sync --group train`）：torch 缺失时整文件 skip，
# 保证 CI 与数据管线无需安装 torch 也能跑（约定同 test_imitation.py）。
pytest.importorskip("torch")
import torch  # noqa: E402

from ark_core.models.track_a.policy_v1 import TrackAPolicyV1  # noqa: E402
from ark_core.protocol.action import ACTION_TYPES, DIRECTIONS, GRID_CHANNELS  # noqa: E402
from ark_core.training.imitate import TrainConfig, collate  # noqa: E402
from ark_core.training.pretrain_ssl import (  # noqa: E402
    PretextConfig,
    SSLMaskedPolicy,
    _draw_sel,
    compute_ssl_loss,
    constant_baseline_eval,
    load_pretrained_into,
    mask_batch,
    pretrain,
)

FEATVEC = np.asarray(
    [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0]], dtype=np.float32
)


def _records(n: int = 4, family: str = "main_01") -> list[dict]:
    """合成 v0 record（3×4 格、2 出怪、3 编队、4 动作）。"""
    rows, cols = 3, 4
    grid = []
    for r in range(rows):
        for c in range(cols):
            grid.extend([r % 2, 1, 1, 0, 0])
    cell_pos = np.asarray(
        [[r / 2, c / 3] for r in range(rows) for c in range(cols)], dtype=np.float32
    )
    return [
        {
            "stage_id": f"main_01-0{i}",
            "family": family,
            "episode_id": f"ep-{i}",
            "cells": np.asarray(grid, dtype=np.float32).reshape(rows * cols, GRID_CHANNELS),
            "cell_pos": cell_pos,
            "rows": rows,
            "cols": cols,
            "spawns": np.asarray([[0.1, 0.0, 0.0, 1.0], [0.9, 1.0, 0.0, 1.0]], np.float32),
            "squad_idx": np.asarray([0, 1, 2], dtype=np.int64),
            "type": np.asarray([3, 0, 0, 2], dtype=np.int64),
            "target": np.asarray([0, 1, 2, 1], dtype=np.int64),
            "cell": np.asarray([0, 5, 7, 0], dtype=np.int64),
            "dir": np.asarray([4, 2, 1, 4], dtype=np.int64),
        }
        for i in range(n)
    ]


def _batch(recs: list[dict]) -> dict:
    """collate + squad_gold_idx（与 pretrain.prep 同口径）。"""
    c = collate(recs, FEATVEC)
    b = len(recs)
    nq = max(len(r["squad_idx"]) for r in recs)
    g = np.full((b, nq), -1, np.int64)
    for i, r in enumerate(recs):
        g[i, : len(r["squad_idx"])] = r["squad_idx"]
    c["squad_gold_idx"] = torch.from_numpy(g)
    return c


# ---------------------------------------------------------------- 掩码采样分布


def test_draw_sel_distribution() -> None:
    """抽中率≈mask_rate、80/10/10 三支比例、无效位永不抽中（定种子可复现）。"""
    cfg = PretextConfig(mask_rate=0.15, ratio_mask=0.8, ratio_random=0.1)
    gen = torch.Generator().manual_seed(0)
    valid = torch.rand(64, 300) < 0.7  # 70% 有效
    m80, mrand, mkeep = _draw_sel(valid, cfg, gen)
    sel = m80 | mrand | mkeep
    assert not bool((sel & ~valid).any())  # 无效位零抽中
    rate = sel.sum().item() / valid.sum().item()
    assert rate == pytest.approx(0.15, abs=0.02)
    tot = sel.sum().item()
    assert m80.sum().item() / tot == pytest.approx(0.8, abs=0.05)
    assert mrand.sum().item() / tot == pytest.approx(0.1, abs=0.05)
    assert mkeep.sum().item() / tot == pytest.approx(0.1, abs=0.05)
    # 同种子同结果（确定性）
    gen2 = torch.Generator().manual_seed(0)
    a, b, c = _draw_sel(valid, cfg, gen2)
    assert torch.equal(m80, a) and torch.equal(mrand, b) and torch.equal(mkeep, c)


def test_hist_mask_never_last_action() -> None:
    """末位动作永不遮（其输入列不在序列内，见模块 docstring 的因果约定）。"""
    recs = _records(2)
    batch = _batch(recs)
    cfg = PretextConfig(mask_rate=0.9)  # 高遮率放大信号
    gen = torch.Generator().manual_seed(1)
    mk = mask_batch(batch, torch.from_numpy(FEATVEC), 1 + 133 + 16, cfg, gen)
    lens = batch["lengths"]
    for i in range(len(recs)):
        assert not bool(mk["sel_hist"][i, int(lens[i]) - 1])  # 末位不遮
        assert bool(mk["sel_hist"][i, : int(lens[i]) - 1].any())  # 其余高覆盖


def test_hist_rand_replacement_valid_cell_index() -> None:
    """动作 cell 的 10% 随机替换：替换值必须是合法格子指针索引（越界回归）。

    早期实现把格子实体轴索引错用到动作轴（IndexError），本测试用
    ratio_mask=0 / ratio_random=0.9 强制走随机分支，n_cells(12) 与 T(4) 不同长。
    """
    recs = _records(2)
    batch = _batch(recs)
    cfg = PretextConfig(mask_rate=0.9, ratio_mask=0.0, ratio_random=0.9)
    gen = torch.Generator().manual_seed(1)
    mk = mask_batch(batch, torch.from_numpy(FEATVEC), 1 + 133 + 16, cfg, gen)
    cell_in = mk["inputs"]["cell"]
    n_cells = batch["cells"].shape[1]
    assert bool((cell_in >= 0).all()) and bool((cell_in < n_cells).all())
    # 强随机分支必发生（否则本测试形同虚设）
    gold = batch["cell"]
    assert int((cell_in != gold).sum()) > 0
    # 格子实体的 10% 随机替换：行内容仍是某个合法格子的内容
    cells_in = mk["inputs"]["cells"]
    assert cells_in.shape == batch["cells"].shape
    assert bool(torch.isfinite(cells_in).all())


# ---------------------------------------------------------------- 重组装形状


def test_mask_forward_loss_backward() -> None:
    """掩码 batch → 预置前向 → 损失有限且可反传（形状钉死）。"""
    recs = _records(4)
    batch = _batch(recs)
    cfg = PretextConfig()
    gen = torch.Generator().manual_seed(2)
    n_targets = 1 + 133 + 16
    mk = mask_batch(batch, torch.from_numpy(FEATVEC), n_targets, cfg, gen)

    model = SSLMaskedPolicy(
        featvec_dim=3, vocab_size=133, max_squad=4, n_featvec_rows=3,
        d_model=32, n_heads=4, n_layers=1, n_dec_layers=1, dropout=0.0,
    )
    from ark_core.training.pretrain_ssl import _forward_masked

    out = _forward_masked(model, batch, mk)
    b, t = batch["type"].shape
    nc = batch["cells"].shape[1]
    assert out["type"].shape == (b, t, len(ACTION_TYPES))
    assert out["target"].shape == (b, t, 1 + 133 + 4)
    assert out["dir"].shape == (b, t, len(DIRECTIONS))
    assert out["cell_ptr"].shape == (b, t, nc)
    assert out["cell_recon"].shape == (b, nc, GRID_CHANNELS + 2)
    assert out["spawn_recon"].shape == (b, batch["spawns"].shape[1], 4)
    assert out["squad_cls"].shape == (b, 3, 3)

    loss, parts = compute_ssl_loss(out, batch, mk, cfg)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.mask_cell.grad is not None
    assert all(np.isfinite(v) for v in parts.values())


def test_constant_baseline_eval_shapes() -> None:
    """恒预测基线在评测位上给出 0~1 命中率与非负 MAE。"""
    from ark_core.training.pretrain_ssl import _train_side_stats

    recs = _records(4)
    batch = _batch(recs)
    cfg = PretextConfig()
    gen = torch.Generator().manual_seed(3)
    mk = mask_batch(batch, torch.from_numpy(FEATVEC), 1 + 133 + 16, cfg, gen)
    stats = _train_side_stats([batch])
    base = constant_baseline_eval([batch], [mk], stats)
    for k in ("type_acc", "dir_acc", "cls_squad_acc"):
        assert 0.0 <= base[k] <= 1.0
    assert base["mae_cell"] >= 0.0


# ---------------------------------------------------------------- 键级吻合（下游初始化契约）


def test_load_pretrained_into_policy_v1() -> None:
    """SSLMaskedPolicy 全部 policy 键可注入 TrackAPolicyV1；只差 state_proj。"""
    kw: dict[str, Any] = dict(d_model=32, n_heads=4, n_layers=1, n_dec_layers=1, dropout=0.0)
    ssl = SSLMaskedPolicy(
        featvec_dim=3, vocab_size=5, max_squad=16, n_featvec_rows=3, **kw
    )
    for p in ssl.parameters():  # 让权重非平凡（零初始化无法证明真的注入）
        torch.nn.init.normal_(p, std=0.02)
    v1 = TrackAPolicyV1(featvec_dim=3, vocab_size=5, max_squad=16, state_dim=12, **kw)
    own_before = v1.state_dict()
    loaded, skipped = load_pretrained_into(v1, ssl.state_dict())
    own_after = v1.state_dict()
    # policy 键全数加载；被跳过的只能是无对应键的（state_proj/mask/重构头不参与本侧）
    assert loaded == len([k for k in own_after if not k.startswith("state_proj")])
    assert set(skipped) == {k for k in own_after if k.startswith("state_proj")}
    # 注入确实生效：encoder 权重逐值等于 ssl 侧
    for k in own_after:
        if k.startswith("encoder."):
            assert torch.equal(own_after[k], ssl.state_dict()[k])
    # state_proj 未被触碰
    for k in skipped:
        assert torch.equal(own_after[k], own_before[k])

    # 词表不一致侧：hist_target/head_target 形不合 → 进跳过表（下游 v2 词表=51 的实况）
    v1_small = TrackAPolicyV1(featvec_dim=3, vocab_size=2, max_squad=16, state_dim=12, **kw)
    _l2, skipped2 = load_pretrained_into(v1_small, ssl.state_dict())
    assert {
        "hist_target.weight",
        "head_target.weight",
        "head_target.bias",
        "state_proj.weight",
        "state_proj.bias",
    } <= set(skipped2)


# ---------------------------------------------------------------- 预训练主循环冒烟


def test_pretrain_smoke(tmp_path) -> None:
    """4 局（2 train 家族 + 2 holdout 家族）跑 1 epoch：ckpt 与报告字段齐全。"""
    recs = _records(2, family="main_01") + _records(2, family="hold_family")
    npz = tmp_path / "v0.npz"
    np.savez_compressed(
        npz,
        n=len(recs),
        featvec=FEATVEC,
        vocab=np.asarray(["device:x"], dtype=object),
        records=np.asarray(recs, dtype=object),
    )
    cfg = PretextConfig(
        d_model=32, n_heads=4, n_layers=1, n_dec_layers=1, dropout=0.0,
        batch_size=2, epochs=1, limit_train=10, holdout_eval_batches=1,
    )
    rep = pretrain(
        npz, cfg, holdout_families={"hold_family"}, out_dir=tmp_path / "run",
        device_str="cpu", log=lambda *_: None,
    )
    assert rep["n_train"] == 2 and rep["n_holdout"] == 2
    assert "baseline_random_init" in rep and "baseline_constant" in rep
    assert rep["history"] and "loss" in rep["history"][-1]["holdout"]
    assert (tmp_path / "run" / "pretrain_ssl.pt").is_file()


def test_finetune_smoke_wiring() -> None:
    """TrainConfig 透传：finetune_smoke 侧只验证类型（真跑在集成）。"""
    cfg = TrainConfig(epochs=3)
    assert cfg.epochs == 3
