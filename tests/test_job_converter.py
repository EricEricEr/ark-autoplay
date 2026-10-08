"""作业转换器纯逻辑单测：数据集记录 → 内嵌作业条目。"""

import pytest

from bridge.replay_controller import dataset_record_to_job_entry, stage_id_to_code


def test_stage_id_to_code_main_series() -> None:
    """主线 stage_id 正确映射到短码。"""
    assert stage_id_to_code("main_00-01") == "0-1"
    assert stage_id_to_code("main_01-12") == "1-12"
    assert stage_id_to_code("main_08-10") == "8-10"


def test_stage_id_to_code_rejects_non_main() -> None:
    """非主线 main_XX-YY 一律拒绝（v1 范围）。"""
    for bad in ("tough_10-11", "main_00-TR01", "act23side_06", "main_1-1", ""):
        with pytest.raises(ValueError):
            stage_id_to_code(bad)


def _dataset_record() -> dict:
    return {
        "id": 48628,
        "stage_id": "main_00-02",
        "title": "序章 黑暗时代·上 - 0-2 - 守卫",
        "like": 61,
        "views": 737,
        "rating_ratio": 1.0,
        "upload_time": "2023-02-01T00:00:00",
        "lineup": [
            {
                "name": "黑角",
                "id": "char_502_nblade",
                "skill": 1,
                "skill_usage": 0,
                "req": {"elite": 1, "level": 30, "skill_level": 4, "module": 0,
                        "potentiality": 1},
            }
        ],
        "groups": [],
        "actions": [
            {"type": "Deploy", "name": "黑角", "location": [3, 2],
             "direction": "Right", "kills": 0, "costs": 0,
             "pre_delay": 0, "post_delay": 0},
            {"type": "SpeedUp"},
            {"type": "Output"},  # 白名单外，应丢弃
            {"type": "SkillDaemon"},
        ],
    }


def test_converter_shapes() -> None:
    """转换结果结构与关键约束符合预期。"""
    entry = dataset_record_to_job_entry(_dataset_record())
    assert set(entry) == {"source", "maa_job"}
    src, job = entry["source"], entry["maa_job"]

    assert src["type"] == "prts.plus" and src["job_id"] == 48628
    assert src["stage_id_raw"] == "main_00-02"
    assert "48628" in src["comment"]
    assert src["lineup_raw"][0]["req"]["elite"] == 1  # 原始 req 保留溯源

    assert job["stage_name"] == "0-2"
    assert job["minimum_required"] == "v4.0.0"
    assert len(job["opers"]) == 1
    # 编队要求整体省略（module:0 会被 MAA 当硬要求误判 OperatorMissing）
    assert "requirements" not in job["opers"][0]

    types = [a["type"] for a in job["actions"]]
    assert types == ["Deploy", "SpeedUp", "SkillDaemon"]  # Output 被丢弃
    deploy = job["actions"][0]
    assert deploy["name"] == "黑角"
    assert deploy["location"] == [3, 2]
    assert deploy["direction"] == "Right"
    assert "kills" not in deploy and "costs" not in deploy  # 0 值条件不下发


def test_converter_defaults_skill_fields() -> None:
    """缺失 skill/skill_usage 时填默认 1/0。"""
    rec = _dataset_record()
    rec["lineup"][0]["skill"] = None
    rec["lineup"][0]["skill_usage"] = None
    job = dataset_record_to_job_entry(rec)["maa_job"]
    assert job["opers"][0]["skill"] == 1
    assert job["opers"][0]["skill_usage"] == 0
