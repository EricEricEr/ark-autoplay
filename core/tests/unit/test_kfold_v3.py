"""v3 三连单测：K 折划分 / dir_loss 旋钮 / DIR 标签审计（全合成，无真实素材）。

重点钉死：
1. 折划分的**确定性**与**不重叠全覆盖**（按基础关，#f# 变体随基础关同折）；
2. ``dir_loss`` 四档到损失权重的映射（masked/eval_only=0，未知值报错）；
3. 审计的 direction 归一分类与「同 job 同格同干员」不一致率统计口径。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# 训练依赖是可选的（`uv sync --group train`）：kfold 模块经 .imitate 传递依赖
# torch，缺失时整文件 skip（约定同 test_imitation.py）。
pytest.importorskip("torch")
import torch  # noqa: E402,F401

from ark_core.training.dir_label_audit import _norm_dir, audit_dir_labels  # noqa: E402
from ark_core.training.kfold import (  # noqa: E402
    make_folds,
    resolve_w_dir,
    summarize_fold_metrics,
)

# ---------------------------------------------------------------- 折划分


def test_make_folds_deterministic_and_disjoint() -> None:
    """同种子同输入必同输出；折间不重叠；并集覆盖全部基础关。"""
    stages = (
        [f"main_0{i}-01" for i in range(6) for _ in range(5)]
        + ["main_01-05", "main_01-05#f#", "camp_02"]
    )
    f1 = make_folds(stages, 3, 42)
    f2 = make_folds(stages, 3, 42)
    assert f1 == f2  # 确定性
    flat = [g for fold in f1 for g in fold]
    assert len(flat) == len(set(flat))  # 不重叠
    assert set(flat) == {f"main_0{i}-01" for i in range(6)} | {"main_01-05", "camp_02"}
    # #f# 变体与基础关同折
    fold_of_main_0105 = next(i for i, fold in enumerate(f1) if "main_01-05" in fold)
    assert all("#f#" not in g for g in f1[fold_of_main_0105])  # 变体已并入基础关组
    with pytest.raises(ValueError):
        make_folds([], 3, 42)
    with pytest.raises(ValueError):
        make_folds(stages, 1, 42)


def test_make_folds_balance() -> None:
    """大组优先贪心装入：各折局数差不超过最大单组局数。"""
    stages = ["a"] * 10 + ["b"] * 10 + ["c"] * 2 + ["d"] * 2 + ["e"] * 1
    folds = make_folds(stages, 2, 42)
    sizes = [
        sum(stages.count(g) for g in fold) for fold in folds
    ]
    assert abs(sizes[0] - sizes[1]) <= 10


# ---------------------------------------------------------------- dir_loss 旋钮


def test_resolve_w_dir_modes() -> None:
    """四档映射与未知值报错（masked/eval_only 损失权重为 0）。"""
    assert resolve_w_dir("full", 0.5, 0.1) == 0.5
    assert resolve_w_dir("weighted", 0.5, 0.1) == 0.1
    assert resolve_w_dir("masked", 0.5, 0.1) == 0.0
    assert resolve_w_dir("eval_only", 0.5, 0.1) == 0.0
    with pytest.raises(ValueError, match="dir_loss"):
        resolve_w_dir("bogus", 0.5, 0.1)


def test_summarize_fold_metrics_mean_std() -> None:
    """均值±样本标准差（ddof=1）。"""
    summaries = [
        {"type_acc": 0.8, "macro_f1": 0.5, "cell_acc": 0.1, "dir_acc": 0.4, "target_acc": 0.6},
        {"type_acc": 0.9, "macro_f1": 0.7, "cell_acc": 0.1, "dir_acc": 0.4, "target_acc": 0.6},
    ]
    ms = summarize_fold_metrics(summaries)
    assert ms["type_acc"]["mean"] == pytest.approx(0.85)
    assert ms["type_acc"]["std"] == pytest.approx(0.05 * (2**0.5), rel=1e-6)
    assert ms["macro_f1"]["mean"] == pytest.approx(0.6)


# ---------------------------------------------------------------- DIR 审计


def test_norm_dir_kinds() -> None:
    """direction 归一五分类：missing/empty/none/known/unknown。"""
    assert _norm_dir(None)[0] == "missing"
    assert _norm_dir("")[0] == "empty"
    assert _norm_dir("None") == ("none", "None")
    assert _norm_dir("左") == ("known", "Left")
    assert _norm_dir("随便写") == ("unknown", "随便写")


def _audit_tree(tmp_path: Path, *, inconsistent: bool) -> tuple[Path, Path]:
    """造审计用合成目录：processed parquet + 两局同 job 的 episodes。

    ``inconsistent=True`` 时两局同关同格同干员方向相反（应被判不一致）。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    grid = []
    rows, cols = 8, 12
    for r in range(rows):
        for c in range(cols):
            grid.extend([r % 2, 1, 1, 0, 0])
    pq.write_table(
        pa.table(
            {
                "stage_id": ["main_99-01"],
                "rows": [rows],
                "cols": [cols],
                "grid": [grid],
                "spawns": [[]],
                "opt_initial_cost": [10],
            }
        ),
        processed / "stagefeat.parquet",
    )
    root = tmp_path / "episodes"
    dirs = ["Up", "Left"] if inconsistent else ["Up", "Up"]
    for i, d in enumerate(dirs):
        name = f"job-777-20261021-00000{i}"
        ed = root / "main_99-01" / name
        ed.mkdir(parents=True)
        job_path = tmp_path / f"job_{i}.json"
        job_path.write_text(
            json.dumps(
                {
                    "stage_name": "9-1",
                    "opers": [{"name": "能天使", "skill": 1}],
                    "actions": [
                        {"type": "Deploy", "name": "能天使", "location": [3, 3],
                         "direction": d},
                        {"type": "Deploy", "name": "无名", "direction": "随便写"},
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (ed / "episode.json").write_text(
            json.dumps(
                {"episode_id": name, "stage_id": "main_99-01", "job_file": str(job_path)}
            ),
            encoding="utf-8",
        )
        (ed / "battle_states.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    return root, processed


def test_audit_dir_inconsistency_flag(tmp_path: Path) -> None:
    """同 job 重放同格同干员方向相反 → 不一致率 1.0 → dirty；一致 → 不 dirty。"""
    root, processed = _audit_tree(tmp_path, inconsistent=True)
    rep = audit_dir_labels(
        processed,
        root,
        holdout_stages=set(),
        same_job_inconsistency_dirty=0.05,
        unknown_alias_dirty=0.99,  # 隔离变量：只看不一致率
    )
    sj = rep["b_inconsistency"]["same_job"]
    assert sj["n_groups_multi_obs"] == 1
    assert sj["n_inconsistent"] == 1
    assert sj["rate"] == 1.0
    assert rep["dirty"] is True
    # 未知别名被单独统计（"随便写" 每局各一次）
    assert rep["a_dir_kind"]["unknown_aliases"] == {"随便写": 2}

    root2, processed2 = _audit_tree(tmp_path / "clean", inconsistent=False)
    rep2 = audit_dir_labels(
        processed2,
        root2,
        holdout_stages=set(),
        same_job_inconsistency_dirty=0.05,
        unknown_alias_dirty=0.99,
    )
    assert rep2["b_inconsistency"]["same_job"]["rate"] == 0.0
    assert rep2["dirty"] is False
    assert rep2["d_prior"]["train"]["majority"] == "Up"
    assert rep2["d_prior"]["train"]["majority_acc"] == 1.0
