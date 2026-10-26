"""公开原文还原回归：不拼错地图，不猜填缺失的操作参数。"""

import copy

from bridge.replay_controller import dataset_record_to_job_entry
from bridge.source_originals import hydrate_record, missing_runtime_fields


def test_unresolved_revision_preserves_old_map_and_old_actions():
    record = {
        "id": 1,
        "stage_id": "main_99-01",
        "lineup": [{"name": "A"}],
        "groups": [],
        "actions": [{"type": "Deploy", "name": "A"}],
    }
    original = copy.deepcopy(record)
    script = {
        "stage_name": "UNRESOLVED",
        "opers": [{"name": "B"}],
        "actions": [{"type": "Retreat", "name": "B"}],
    }
    hydrated = hydrate_record(record, script, {"main_99-01": {"code": "99-1"}})
    assert record == original
    assert hydrated["stage_id"] == record["stage_id"]
    assert hydrated["actions"] == record["actions"]
    assert hydrated["lineup"] == record["lineup"]
    assert hydrated["native_script_raw"] == script
    assert hydrated["source_quality"] == "original_stage_unresolved"


def test_original_level_path_restores_native_parameters_and_zero_values():
    record = {"id": 1, "stage_id": "main_99-01", "lineup": [], "actions": []}
    script = {
        "stage_name": "obt/main/level_main_99-02",
        "opers": [],
        "actions": [
            {"type": "MoveCamera", "distance": [4.5, 0], "cost_changes": 0},
            {"type": "Swipe", "begin": [0, 0], "end": [9, 9], "duration": 0},
            {"type": "Click", "location": [0, 0], "flag": False},
        ],
    }
    stages = {"main_99-02": {"level_id": "obt/main/level_main_99-02", "code": "99-2"}}
    hydrated = hydrate_record(record, script, stages)
    entry = dataset_record_to_job_entry(hydrated)
    assert hydrated["stage_id"] == "main_99-02"
    assert entry["maa_job"]["stage_name"] == "99-2"
    assert entry["maa_job"]["actions"] == script["actions"]
    assert not entry["source"]["incomplete_action_fields"]


def test_ambiguous_map_is_not_guessed_and_missing_fields_not_fabricated():
    hydrated = hydrate_record(
        {"stage_id": "old", "actions": []},
        {"stage_name": "X", "actions": [{"type": "MoveCamera"}]},
        {"a": {"code": "X"}, "b": {"code": "X"}},
    )
    assert hydrated["stage_id"] == "old"
    assert hydrated["source_quality"] == "original_stage_unresolved"
    assert missing_runtime_fields(hydrated["native_script_raw"]["actions"]) == [
        {"action_index": 0, "type": "MoveCamera", "fields": ["distance"]}
    ]
