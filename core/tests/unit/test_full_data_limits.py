"""全量容量回归：超过旧阈值仍保留完整记录，动态目标能反传到末槽。"""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from ark_core.models.track_a.policy import TrackAPolicy  # noqa: E402
from ark_core.training.imitate import (  # noqa: E402
    TrainConfig,
    collate,
    compute_loss,
    forward_batch,
)
from ark_core.training.imitation_data import (  # noqa: E402
    BuildStats,
    Vocab,
    build_vocab,
    encode_episode,
)
from ark_core.training.state_action_data import BuildStatsV1  # noqa: E402
from ark_core.training.state_action_data import encode_episode as encode_v1  # noqa: E402


def _large_example():
    ids = [f"char_synthetic_{i}" for i in range(24)]
    actions = [
        {
            "type": "deploy",
            "target": {"kind": "operator", "char_id": ids[-1]},
            "location": [7, 64],
            "direction": "Right",
        }
        for _ in range(129)
    ]
    episode = {
        "episode_id": "synthetic",
        "stage": {"stage_id": "synthetic", "family": "test"},
        "squad": [{"char_id": cid} for cid in ids],
        "actions": actions,
    }
    stage = {
        "rows": 65,
        "cols": 8,
        "grid": [0, 1, 1, 0, 0] * 520,
        "spawns": [{"t_ms": i, "enemy_matched": True} for i in range(401)],
    }
    return ids, episode, stage


def test_encoding_and_dynamic_target_keep_all_large_axes_and_backpropagate():
    ids, episode, stage = _large_example()
    vocab = Vocab(extra_tokens=[])
    stats = BuildStats()
    record = encode_episode(episode, stage, {cid: i for i, cid in enumerate(ids)}, vocab, stats)
    assert record is not None
    assert (
        len(record["cells"]),
        len(record["spawns"]),
        len(record["squad_idx"]),
        len(record["type"]),
    ) == (520, 401, 24, 129)
    assert record["cell"][-1] == 519
    batch = collate([record], np.ones((24, 4), dtype=np.float32))
    model = TrackAPolicy(
        featvec_dim=4,
        vocab_size=vocab.size,
        max_squad=0,
        d_model=16,
        n_heads=2,
        n_layers=1,
        n_dec_layers=1,
        dropout=0,
    )
    out = forward_batch(model, batch)
    assert out["target"].shape == (1, 129, 1 + vocab.size + 24)
    assert torch.isfinite(out["target"][0, -1, record["target"][-1]])
    loss, _ = compute_loss(out, batch, TrainConfig())
    assert torch.isfinite(loss)
    loss.backward()
    assert model.squad_query is not None
    assert model.squad_query.weight.grad is not None
    assert torch.isfinite(model.squad_query.weight.grad).all()


def test_full_vocab_keeps_low_frequency_training_targets_only(tmp_path):
    path = tmp_path / "episodes.jsonl"
    actions = [
        {"type": "deploy", "target": {"kind": "device", "name": f"d{i}"}} for i in range(300)
    ]
    rows = [
        {"stage": {"stage_id": "train"}, "actions": actions},
        {
            "stage": {"stage_id": "holdout"},
            "actions": [{"type": "deploy", "target": {"kind": "device", "name": "held"}}],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    vocab = build_vocab(path, {"train"})
    assert len(vocab.extra_tokens) == 300
    assert vocab.index_of("device:d299") is not None
    assert vocab.index_of("device:held") is None


def test_state_encoding_retains_unpaired_data_and_observation_without_labels():
    _, _, stage = _large_example()
    stats = BuildStatsV1()
    raw = [{"hand": [{"name": "A"}], "costs": i} for i in range(410)]
    rec = encode_v1(
        episode_id="observation",
        stage_id="synthetic",
        job={"actions": [], "opers": []},
        battle_states=raw,
        action_events=[],
        sf=stage,
        name2cid={"A": "char_A"},
        featvec_idx={"char_A": 0},
        vocab=Vocab(extra_tokens=[]),
        stats=stats,
    )
    assert rec is not None
    assert rec["entity_only"] and len(rec["type"]) == 0
    assert rec["state_feat"].shape[0] == 0
    assert rec["battle_states_raw"] == raw and len(rec["unpaired_states"]) == 410
    assert len(rec["squad_idx"]) == 1
    assert stats.n_unpaired_states == 410
    assert stats.n_extra_states_dropped == stats.drop_no_squad == stats.drop_no_actions == 0


@pytest.mark.parametrize("slots", [0, 16])
def test_loader_keeps_legacy_fixed_head_and_new_dynamic_head(tmp_path, slots):
    """旧权重没有 max_squad 元数据时，仍按原头维度严格恢复。"""
    from ark_core.infer.track_a_v1 import load_track_a_v1
    from ark_core.models.track_a.policy_v1 import TrackAPolicyV1

    config = dict(d_model=16, n_heads=2, n_layers=1, n_dec_layers=1, dropout=0)
    model = TrackAPolicyV1(featvec_dim=4, vocab_size=1, max_squad=slots, state_dim=12, **config)
    path = tmp_path / "synthetic.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "featvec_dim": 4,
            "state_dim": 12,
            "vocab": [],
        },
        path,
    )
    loaded = load_track_a_v1(path)
    assert loaded.model.max_squad == slots
    assert loaded.model.head_target.weight.shape == model.head_target.weight.shape
