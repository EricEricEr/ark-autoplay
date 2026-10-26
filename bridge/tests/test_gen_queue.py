"""gen_queue 新增过滤单测：账号进度边界 + 干员已知名（groups 可达性）。

数据集用 tmp 目录里的最小 jsonl fixture（纯文本构造，无真实素材）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bridge.gen_queue import build_queue, groups_reachable, main, main_code_key


def _mk_dataset(tmp_path: Path, stages: list[dict], recs: list[dict]) -> Path:
    ds = tmp_path / "ds"
    ds.mkdir()
    with (ds / "stages.jsonl").open("w", encoding="utf-8") as f:
        for s in stages:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with (ds / "copilot.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return ds


def _mk_operbox(tmp_path: Path, names: list[str]) -> Path:
    ob = tmp_path / "operbox.json"
    ob.write_text(
        json.dumps({"own_opers": [{"name": n} for n in names]}, ensure_ascii=False),
        encoding="utf-8",
    )
    return ob


def _stage(sid: str, code: str, typ: str) -> dict:
    return {"stage_id": sid, "code": code, "type": typ}


def _rec(jid: int, sid: str, ops=(), groups=()) -> dict:
    return {
        "id": jid,
        "stage_id": sid,
        "lineup": [{"name": n, "skill": 1} for n in ops],
        "actions": [{"type": "Deploy"}],
        "groups": list(groups),
        "title": f"t{jid}",
        "like": 0,
        "views": 0,
        "rating_ratio": 0.9,
        "upload_time": "",
    }


def _build(ds: Path, **kw) -> tuple[list, dict]:
    defaults = dict(
        tiers_path=None,
        maa_dir=None,
        per_stage=5,
        max_actions=10,
        limit=None,
        types=None,
        nav_filter=False,
    )
    defaults.update(kw)
    return build_queue(ds, **defaults)


STAGES = [
    _stage("main_06-09", "6-9", "MAIN"),
    _stage("main_14-21", "14-21", "MAIN"),
    _stage("main_15-03", "15-3", "MAIN"),
    _stage("sub_03-1-1", "sub_03-1-1", "SUB"),
    _stage("tough_14-01", "TOUGH1", "TOUGH"),
]


def _uncapped_dataset(tmp_path: Path) -> Path:
    records = [_rec(jid, "main_00-01") for jid in range(1, 5)]
    for record, count in zip(records, (1, 31, 80, 2), strict=True):
        record["actions"] = [{"type": "SpeedUp"} for _ in range(count)]
    return _mk_dataset(tmp_path, [_stage("main_00-01", "0-1", "MAIN")], records)


def test_unlimited_selection_keeps_all_jobs_and_long_actions(tmp_path: Path) -> None:
    dataset = _uncapped_dataset(tmp_path)
    jobs, stats = _build(dataset, per_stage=0, max_actions=0)
    assert {job["source"]["job_id"] for job in jobs} == {1, 2, 3, 4}
    assert max(len(job["maa_job"]["actions"]) for job in jobs) == 80
    assert stats["per_stage_cap"] is None
    assert stats["max_actions"] is None
    assert "too_long" not in stats["filter"]


def test_cli_defaults_do_not_truncate_jobs(tmp_path: Path) -> None:
    dataset = _uncapped_dataset(tmp_path)
    out = tmp_path / "queue.json"
    assert main(["--dataset", str(dataset), "--out", str(out), "--no-nav-filter"]) == 0
    document = json.loads(out.read_text(encoding="utf-8"))
    assert len(document["jobs"]) == 4
    assert document["stats"]["per_stage_cap"] is None
    assert document["stats"]["max_actions"] is None


def test_unlimited_per_stage_still_honors_explicit_total_limit(tmp_path: Path) -> None:
    dataset = _uncapped_dataset(tmp_path)
    jobs, _ = _build(dataset, per_stage=0, max_actions=0, limit=3)
    assert len(jobs) == 3


def test_explicit_positive_caps_still_work(tmp_path: Path) -> None:
    dataset = _uncapped_dataset(tmp_path)
    jobs, _ = _build(dataset, per_stage=2, max_actions=30)
    assert {job["source"]["job_id"] for job in jobs} == {1, 4}


@pytest.mark.parametrize("settings", [{"per_stage": -1}, {"max_actions": -1}])
def test_negative_caps_rejected(tmp_path: Path, settings: dict) -> None:
    dataset = _uncapped_dataset(tmp_path)
    with pytest.raises(ValueError, match="0 表示不限"):
        _build(dataset, **settings)


def test_main_code_key() -> None:
    """只认主线数字码；S/R/活动码返回 None（保守放行依据）。"""
    assert main_code_key("14-21") == (14, 21)
    assert main_code_key("6-9") == (6, 9)
    assert main_code_key("S9-2") is None
    assert main_code_key("R8-2") is None
    assert main_code_key("act33side_01") is None


def test_account_progress_filter(tmp_path: Path) -> None:
    """main 系 <= progress 保留、超进度丢弃计数；非 main 系保守放行。"""
    recs = [
        _rec(1, "main_06-09"),
        _rec(2, "main_14-21"),
        _rec(3, "main_15-03"),
        _rec(4, "sub_03-1-1"),
        _rec(5, "tough_14-01"),
    ]
    ds = _mk_dataset(tmp_path, STAGES, recs)
    jobs, stats = _build(ds, account_progress="14-21")
    kept = {j["source"]["job_id"] for j in jobs}
    assert kept == {1, 2, 4, 5}
    assert stats["filter"].get("beyond_progress") == 1
    assert stats["account_progress"] == "14-21"


def test_account_progress_off_keeps_old_behavior(tmp_path: Path) -> None:
    """不传 --account-progress：超进度关照收（零回归）。"""
    recs = [_rec(3, "main_15-03")]
    ds = _mk_dataset(tmp_path, STAGES, recs)
    jobs, stats = _build(ds)
    assert {j["source"]["job_id"] for j in jobs} == {3}
    assert "beyond_progress" not in stats["filter"]
    assert stats["account_progress"] is None


def test_account_progress_bad_code_rejected(tmp_path: Path) -> None:
    """progress 参数非法（非主线数字码）直接 ValueError。"""
    ds = _mk_dataset(tmp_path, STAGES, [])
    with pytest.raises(ValueError):
        _build(ds, account_progress="S9-2")


def test_require_ops_known_drops_unknown_and_unreachable_groups(tmp_path: Path) -> None:
    """opers 未知名丢弃 + 候选组整组无持证丢弃；组内有 1 名持证则放行。"""
    ds = _mk_dataset(
        tmp_path,
        STAGES[:2],
        [
            _rec(1, "main_06-09", ops=("芬", "克洛丝")),  # 全持证
            _rec(2, "main_06-09", ops=("芬", "黍")),  # 黍 未持证
            _rec(
                3,
                "main_06-09",
                ops=("芬",),
                groups=(
                    {"name": "先锋", "options": [{"name": "德克萨斯"}, {"name": "山"}]},
                ),
            ),  # 组内无人持证
            _rec(
                4,
                "main_14-21",
                ops=("芬",),
                groups=(
                    {
                        "name": "先锋",
                        "options": [{"name": "德克萨斯"}, {"name": "克洛丝"}],
                    },
                ),
            ),  # 组内有持证
        ],
    )
    ob = _mk_operbox(tmp_path, ["芬", "克洛丝"])
    jobs, stats = _build(ds, operbox_path=ob, max_missing=1, require_ops_known=True)
    kept = {j["source"]["job_id"] for j in jobs}
    assert kept == {1, 4}
    assert stats["filter"].get("opers_unknown") == 1
    assert stats["filter"].get("groups_unreachable") == 1
    assert stats["require_ops_known"] is True


def test_require_ops_known_off_is_zero_regression(tmp_path: Path) -> None:
    """开关关闭时：同批数据按旧口径放行（max-missing 1 语义不变）。"""
    ds = _mk_dataset(
        tmp_path,
        STAGES[:1],
        [
            _rec(2, "main_06-09", ops=("芬", "黍")),  # 缺 1 ≤ 1：旧口径放行
            _rec(
                3,
                "main_06-09",
                ops=("芬",),
                groups=({"name": "先锋", "options": [{"name": "德克萨斯"}]},),
            ),  # groups 旧口径不看：放行
        ],
    )
    ob = _mk_operbox(tmp_path, ["芬", "克洛丝"])
    jobs, stats = _build(ds, operbox_path=ob, max_missing=1, require_ops_known=False)
    assert {j["source"]["job_id"] for j in jobs} == {2, 3}
    assert "opers_unknown" not in stats["filter"]
    assert "groups_unreachable" not in stats["filter"]


def test_groups_reachable_unit() -> None:
    """options/opers 两种 schema 都认；空组与无约束组放行。"""
    owned = {"芬"}
    big = [{"name": "g", "options": [{"name": "德克萨斯"}, {"name": "芬"}]}]
    assert groups_reachable(big, owned)
    assert groups_reachable([{"name": "g", "opers": [{"name": "芬"}]}], owned)
    assert groups_reachable([], owned)
    assert groups_reachable([{"name": "g", "options": []}], owned)
    assert groups_reachable(None, owned)
    assert not groups_reachable([{"name": "g", "options": [{"name": "山"}]}], owned)
