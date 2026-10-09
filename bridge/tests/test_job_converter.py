"""作业转换器纯逻辑单测：数据集记录 → 内嵌作业条目。"""

from bridge.replay_controller import dataset_record_to_job_entry, stage_id_to_code


def test_stage_id_to_code_main_series() -> None:
    """主线 stage_id 正确映射到短码（去前导零）。"""
    assert stage_id_to_code("main_00-01") == "0-1"
    assert stage_id_to_code("main_01-12") == "1-12"
    assert stage_id_to_code("main_08-10") == "8-10"


def test_stage_id_to_code_accepts_non_main_and_strips_modifier() -> None:
    """非 main_ 前缀的关卡不再被拒——活动关占数据集 71%，拒掉它们曾使队列只剩 6 条。

    历史：本函数原只接受 ``main_XX-YY`` 且对其它一律抛 ``ValueError``，
    是"队列只有 6 条"的代码级根因（2026-10-09 查明并修复，见 ADR-0006 附注）。
    """
    # 活动/分支/剿灭等：原样返回，交给 MAA 的 Fight 去导航
    assert stage_id_to_code("act23side_06") == "act23side_06"
    assert stage_id_to_code("sub_03-1-1") == "sub_03-1-1"
    assert stage_id_to_code("camp_01") == "camp_01"
    assert stage_id_to_code("wk_melee_1") == "wk_melee_1"
    # #f# 修饰（复刻关）应剥掉，否则不是有效关名
    assert stage_id_to_code("main_00-01#f#") == "0-1"
    assert stage_id_to_code("act16mini_08#f#") == "act16mini_08"


def test_stage_id_to_code_single_digit_main() -> None:
    """一位数章节也要支持（原正则要求两位，'main_1-1' 会被误拒）。"""
    assert stage_id_to_code("main_1-1") == "1-1"


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


def test_converter_prefers_dataset_stage_code() -> None:
    """数据集自带 stage_code 时优先采用（权威，覆盖全部关卡类型）。

    活动关的 stage_id 与游戏关名常不一致，靠 stage_id 猜会错；stages.jsonl 的
    code 字段是权威来源。
    """
    rec = _dataset_record()
    rec["stage_code"] = "0-2"  # 与推断结果一致时无差别
    assert dataset_record_to_job_entry(rec)["maa_job"]["stage_name"] == "0-2"

    rec2 = _dataset_record()
    rec2["stage_id"] = "act33side_07"
    rec2["stage_code"] = "IW-7"  # 权威关名与 stage_id 完全不同
    job = dataset_record_to_job_entry(rec2)["maa_job"]
    assert job["stage_name"] == "IW-7", "必须用数据集的 code，而非从 stage_id 猜"


def test_converter_defaults_skill_fields() -> None:
    """缺失 skill/skill_usage 时填默认 1/0。"""
    rec = _dataset_record()
    rec["lineup"][0]["skill"] = None
    rec["lineup"][0]["skill_usage"] = None
    job = dataset_record_to_job_entry(rec)["maa_job"]
    assert job["opers"][0]["skill"] == 1
    assert job["opers"][0]["skill_usage"] == 0

