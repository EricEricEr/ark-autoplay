"""轮换覆盖与全量拟合回归：实际优化器训练、折间隔离及全量成绩口径。"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from ark_core.training.rotation import grouped_folds, run_rotation  # noqa: E402


def test_groups_keep_chapter_variants_together_and_cover_all():
    records = [
        {"stage_id": stage, "type": [0]}
        for stage in ("main_98-01", "tough_98-01", "main_98-02#f#x", "main_99-01")
    ]
    folds = grouped_folds(records, 2, 42)
    assert len(folds) == 2 and not folds[0] & folds[1]
    assert any({"main_98-01", "tough_98-01", "main_98-02"} <= fold for fold in folds)
    assert set.union(*folds) == {"main_98-01", "tough_98-01", "main_98-02", "main_99-01"}


def make_inputs(tmp_path: Path) -> dict:
    processed, episodes = tmp_path / "processed", tmp_path / "episodes"
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
    for stage in stages:
        for i in range(2):
            directory = episodes / stage / f"job-{i}"
            directory.mkdir(parents=True)
            job = {
                "stage_name": "SYN",
                "opers": [{"name": "SYN_OPERATOR", "skill": 1}],
                "actions": [
                    {
                        "type": "Deploy",
                        "name": "SYN_OPERATOR",
                        "location": [0, 1],
                        "direction": "Up",
                    }
                ],
            }
            (directory / "job.json").write_text(json.dumps(job), encoding="utf-8")
            (directory / "episode.json").write_text(
                json.dumps(
                    {
                        "episode_id": "SYN_DUPLICATE_ID",
                        "stage_id": stage,
                        "job_file": "job.json",
                    }
                ),
                encoding="utf-8",
            )
            (directory / "battle_states.jsonl").write_text(
                json.dumps(
                    {
                        "t_ms": 10,
                        "costs": 20,
                        "kills": 0,
                        "total_kills": 2,
                        "hand": [{"name": "SYN_OPERATOR", "cost": 10, "available": True}],
                        "deployed": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
    return {
        "experiment": {"device": "cpu", "seed": 42},
        "data": {"processed_dir": str(processed), "episodes_root": str(episodes)},
        "rotation": {"k": 2},
        "evaluation": {"runs_dir": str(tmp_path / "runs")},
        "model": {"hidden_dim": 16, "num_heads": 2, "num_layers": 1, "num_dec_layers": 1},
        "training": {"epochs": 2, "batch_size": 2, "eval_train_batches": 0},
    }


def test_real_training_rotates_every_record_and_final_model_uses_all(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = make_inputs(tmp_path)
        result = run_rotation(config, log=lambda _: None)
    finally:
        torch.set_num_threads(previous)
    root = tmp_path / "runs"
    coverage = json.loads((root / "out_of_fold_report.json").read_text(encoding="utf-8"))
    assert result["labeled_records"] == 4 and result["out_of_fold"]["n_tokens"] == 4
    assert len(coverage["per_record"]) == 4  # 相同原始episode_id仍分别覆盖。
    assert all(r["train_folds"] == r["eval_folds"] == 1 for r in coverage["per_record"])
    final = json.loads((root / "final_full/train_report.json").read_text(encoding="utf-8"))
    assert final["n_train"] == 4 and final["n_holdout"] == 0
    assert final["full_data"] and final["checkpoint_selection"] == "last"
    assert final["best_epoch"] == 2 and final["best_holdout"] == {}
    assert Path(result["final_full_checkpoint"]).is_file()
    assert result["final_fit_in_sample"]["n_tokens"] == 4
    # 所有输入动作均保留，没有以折数或训练批大小截断。
    with np.load(root / "all_records.npz", allow_pickle=True) as archive:
        assert len(archive["records"]) == 4
