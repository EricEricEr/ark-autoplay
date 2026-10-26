"""候选覆盖回归：覆盖所有合格人选，保持合法阵容和独立数据身份。"""

import copy

from bridge.candidate_coverage import expand_candidate_coverage


def _entry():
    return {
        "source": {"job_id": 42},
        "maa_job": {
            "stage_name": "SYN-1",
            "opers": [],
            "groups": [
                {"name": "front", "opers": [{"name": "A"}, {"name": "B"}]},
                {"name": "back", "opers": [{"name": "B"}, {"name": "C"}]},
            ],
            "actions": [
                {"type": "Deploy", "name": "front", "location": [0, 0]},
                {"type": "Deploy", "name": "back", "location": [1, 0]},
            ],
        },
    }


def _roster():
    return {
        name: {"name": name, "elite": 2, "level": 90, "skill_count": 3}
        for name in ("A", "B", "C")
    }


def test_every_candidate_is_covered_without_duplicate_members():
    original = _entry()
    before = copy.deepcopy(original)
    variants, report = expand_candidate_coverage(original, _roster())
    assert original == before
    assert set(report["covered_names"]) == {"A", "B", "C"}
    assert len({v["source"]["execution_key"] for v in variants}) == len(variants)
    for variant in variants:
        names = [op["name"] for op in variant["maa_job"]["opers"]]
        assert len(names) == len(set(names)) == 2
        assert not variant["maa_job"]["groups"]
        assert {a["name"] for a in variant["maa_job"]["actions"]} == set(names)
        assert variant["source"]["job_id"] == 42


def test_candidates_must_have_unlocked_requested_skill():
    original = _entry()
    original["maa_job"]["groups"] = [
        {
            "name": "front",
            "opers": [{"name": "A", "skill": 3}, {"name": "B", "skill": 3}],
        }
    ]
    roster = _roster()
    roster["B"]["elite"] = 1
    variants, report = expand_candidate_coverage(original, roster)
    assert report["covered_names"] == ["A"]
    assert report["excluded"][0]["name"] == "B"
    assert len(variants) == 1


def test_generic_slot_respects_deployment_position():
    original = _entry()
    original["maa_job"]["groups"] = []
    original["maa_job"]["actions"] = [
        {"type": "Deploy", "name": "slot", "location": [0, 0]}
    ]
    roster = _roster()
    roster["A"]["position"] = "MELEE"
    roster["B"]["position"] = roster["C"]["position"] = "RANGED"
    variants, report = expand_candidate_coverage(
        original,
        roster,
        stage={"grid": [[{"build": "RANGED"}]]},
        known_operator_names=set(roster),
    )
    assert set(report["covered_names"]) == {"B", "C"}
    assert len(variants) == 2


def test_missing_requirements_are_explicit_not_fabricated():
    original = _entry()
    original["maa_job"]["groups"][0]["opers"][0]["req"] = {"skill_level": 10}
    variants, report = expand_candidate_coverage(original, _roster())
    assert variants
    assert report["unverified_requirements"] == [
        {"group": "front", "name": "A", "fields": ["skill_level"]},
    ]


def test_free_squad_covers_every_actor_without_inventing_deploy_actions():
    original = _entry()
    original["maa_job"].update(groups=[], actions=[])
    variants, report = expand_candidate_coverage(original, _roster())
    assert set(report["covered_names"]) == {"A", "B", "C"}
    assert len(variants) == 3
    assert all(not variant["maa_job"]["actions"] for variant in variants)


def test_partially_fixed_squad_still_covers_flexible_group_candidates():
    original = _entry()
    original["maa_job"]["opers"] = [{"name": "D"}]
    variants, report = expand_candidate_coverage(
        original, {**_roster(), "D": {"name": "D"}}
    )
    assert set(report["covered_names"]) == {"A", "B", "C"}
    assert all("D" in {op["name"] for op in v["maa_job"]["opers"]} for v in variants)
