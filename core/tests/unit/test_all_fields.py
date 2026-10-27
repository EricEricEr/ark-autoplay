"""异构资料真实训练回归：缺失字段不造标、失败资料共享梯度、全量覆盖及断点恢复。"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from ark_core.models.track_a.all_fields import AllFieldsPolicy  # noqa: E402
from ark_core.training.all_fields_data import (  # noqa: E402
    encode_sources,
    load_sources,
    observed_state,
)
from ark_core.training.all_fields_run import run_all_fields  # noqa: E402
from ark_core.training.all_fields_train import one_loss, train_all_fields  # noqa: E402
from ark_core.training.imitate import TrainConfig  # noqa: E402
from ark_core.training.imitate_v1 import collate_v1, load_records_v1  # noqa: E402


def fixture(tmp_path: Path) -> tuple[dict, list]:
    processed = tmp_path / "processed"
    processed.mkdir()
    stages = ["main_98-01", "main_99-01"]
    pq.write_table(
        pa.table(
            {
                "stage_id": stages,
                "rows": [2, 2],
                "cols": [2, 2],
                "grid": [[0, 1, 1, 0, 0] * 4] * 2,
                "spawns": [[], []],
                "opt_initial_cost": [10, 10],
            }
        ),
        processed / "stagefeat.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "char_id": ["char_synthetic"],
                "name": ["SYN_OPERATOR"],
                "rarity_tier": [3],
                "sub_profession": ["SYN"],
                "v0": [1.0],
                "v1": [0.5],
            }
        ),
        processed / "featvec_operators.parquet",
    )
    manifest = []
    for stage in stages:
        for kind in ("win", "lost", "unknown"):
            directory = tmp_path / "episodes" / stage / kind
            directory.mkdir(parents=True)
            meta = {
                "episode_id": "SYN_REPEATED",
                "stage_id": stage,
                "job_file": "job.json",
                "battle_result": kind,
                "prts_detected": ["SYN_MARKER"] if kind == "lost" else [],
            }
            if kind != "unknown":
                meta["fail_stage"] = "done" if kind == "win" else "battle"
            (directory / "episode.json").write_text(json.dumps(meta), encoding="utf-8")
            job = {
                "stage_name": "SYN",
                "opers": [{"name": "SYN_OPERATOR", "skill": 1}],
                "actions": [
                    {
                        "type": "Deploy",
                        "name": "SYN_OPERATOR",
                        "location": [0, 1],
                        "direction": "Up",
                    },
                    {"type": "SkillDaemon"},
                ],
            }
            (directory / "job.json").write_text(json.dumps(job), encoding="utf-8")
            if kind != "unknown":
                state = {
                    "costs": 20,
                    "kills": 0,
                    "total_kills": 2,
                    "in_speedup": False,
                    "hand": [
                        {"name": "SYN_OPERATOR", "cost": 10, "available": True, "cooling": False}
                    ],
                    "deployed": [],
                }
                (directory / "battle_states.jsonl").write_text(
                    json.dumps(state) + "\n", encoding="utf-8"
                )
            manifest.append(
                {"snapshot_bundle": str(directory), "eligible": kind == "win", "legacy": False}
            )
    manifest_path = tmp_path / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "experiment": {"device": "cpu", "seed": 42},
        "data": {
            "manifest": str(manifest_path),
            "processed_dir": str(processed),
            "expected_records": 6,
        },
        "rotation": {"k": 2},
        "model": {
            "hidden_dim": 16,
            "num_heads": 2,
            "num_layers": 1,
            "num_dec_layers": 1,
            "dropout": 0.0,
        },
        "training": {"epochs": 1, "batch_size": 1, "eval_train_batches": 0},
        "auxiliary": {
            "mask_rate": 0.25,
            "huber_delta": 1.0,
            "progress_every_records": 2,
            "weights": {
                key: 1.0
                for key in (
                    "cells",
                    "spawns",
                    "squad",
                    "states",
                    "plan",
                    "outcome",
                    "marker",
                    "failure",
                    "policy",
                )
            },
        },
        "evaluation": {"runs_dir": str(tmp_path / "runs")},
    }
    return config, load_sources(manifest_path)


@pytest.fixture(autouse=True)
def small_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_unknown_state_fields_do_not_become_zero_labels():
    values, mask = observed_state({}, 0, 1, 0)
    assert not mask.any()
    assert np.isfinite(values).all()
    values, mask = observed_state({"costs": 0, "hand": [], "deployed": []}, 0, 1, 1)
    assert mask[0] and values[0] == 0
    assert not mask[1] and not mask[2] and not mask[9] and not mask[11]


def test_failure_and_no_state_records_update_shared_weights_without_policy_labels(tmp_path):
    config, sources = fixture(tmp_path)
    records, featvec, vocab, report = encode_sources(
        sources, Path(config["data"]["processed_dir"]), {row["id"] for row in sources}
    )
    model = AllFieldsPolicy(
        featvec_dim=2,
        vocab_size=len(vocab) + 1,
        max_squad=0,
        state_dim=12,
        d_model=16,
        n_heads=2,
        n_layers=1,
        n_dec_layers=1,
        n_plan_types=len(report["plan_types"]) + 1,
        n_failure_types=len(report["failure_types"]) + 1,
    )
    for record in [r for r in records if not r["policy_allowed"]]:
        model.zero_grad(set_to_none=True)
        batch = collate_v1([record], featvec, 12)
        loss, counts, policy = one_loss(
            model,
            record,
            batch,
            TrainConfig(),
            config["auxiliary"],
            torch.Generator().manual_seed(42),
        )
        loss.backward()
        assert policy is None and counts["policy"]["n"] == 0
        assert model.policy.head_type.weight.grad is None
        cell_gradient = model.policy.encoder.cell_proj.weight.grad
        assert cell_gradient is not None and float(cell_gradient.abs().sum()) > 0
        if not len(record["observed_states"]):
            assert counts["states"]["n"] == counts["outcome"]["n"] == 0
        else:
            assert counts["states"]["n"] > 0 and counts["outcome"]["n"] == 1
            state_gradient = model.policy.state_proj.weight.grad
            assert state_gradient is not None and float(state_gradient.abs().sum()) > 0


def test_all_records_really_train_evaluate_and_transfer_to_final_policy(tmp_path):
    config, _ = fixture(tmp_path)
    result = run_all_fields(config, log=lambda _: None)
    assert result["records"] == 6
    assert result["availability"]["static_only_records"] == 2
    assert result["availability"]["state_frames"] == 4
    assert result["availability"]["policy_records"] == 2
    assert result["availability"]["outcome_records"] == 4
    assert result["out_of_fold"]["records"] == 6
    assert result["out_of_fold"]["policy"]["n_episodes"] == 2
    assert result["out_of_fold"]["policy"]["n_tokens"] == 2
    assert all(
        row["training_folds"] == row["evaluation_folds"] == 1
        for row in result["out_of_fold"]["per_record"]
    )
    final = result["final_in_sample"]
    assert final["records"] == 6 and final["tasks"]["outcome"]["n"] == 4
    assert final["tasks"]["plan"]["n"] == 12
    coverage = json.loads(
        (tmp_path / "runs/final_full/training_coverage.json").read_text(encoding="utf-8")
    )
    assert len(coverage["per_record"]) == 6
    assert all(
        row["optimizer_steps"] == 1 and row["applied_targets"]["cells"] > 0
        for row in coverage["per_record"]
    )
    payload = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert "encoder.cell_proj.weight" in payload["state_dict"]
    assert payload["training_records"] == 6
    restored, _, _, dimension, _ = load_records_v1(tmp_path / "runs/all_records.npz")
    assert len(restored) == 6 and dimension == 12
    assert sum(len(row["type"]) for row in restored) == 2


def test_checkpoint_resumes_after_saved_epoch_with_all_record_counts(tmp_path):
    config, sources = fixture(tmp_path)
    identities = {row["id"] for row in sources}
    records, featvec, vocab, report = encode_sources(
        sources, Path(config["data"]["processed_dir"]), identities
    )
    cfg = TrainConfig(
        d_model=16, n_heads=2, n_layers=1, n_dec_layers=1, dropout=0, epochs=2, batch_size=1
    )
    directory = tmp_path / "resume"

    def interrupt(message):
        if message.startswith("[epoch]"):
            raise RuntimeError("synthetic process interruption after durable checkpoint")

    with pytest.raises(RuntimeError, match="synthetic"):
        train_all_fields(
            records,
            featvec,
            vocab,
            report,
            cfg,
            config["auxiliary"],
            train_ids=identities,
            eval_ids=identities,
            output=directory,
            device_name="cpu",
            log=interrupt,
        )
    result = train_all_fields(
        records,
        featvec,
        vocab,
        report,
        cfg,
        config["auxiliary"],
        train_ids=identities,
        eval_ids=identities,
        output=directory,
        device_name="cpu",
        log=lambda _: None,
    )
    assert result["epochs"] == 2 and len(result["history"]) == 2
    coverage = json.loads((directory / "training_coverage.json").read_text(encoding="utf-8"))
    assert all(row["optimizer_steps"] == 2 for row in coverage["per_record"])
