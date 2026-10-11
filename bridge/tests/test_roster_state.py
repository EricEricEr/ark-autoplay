# -*- coding: utf-8 -*-
"""bridge.roster_state 单元测试：双通道装载、优先级、技能口径、部署掩码。"""
import json
from pathlib import Path

from bridge.roster_state import (RosterOp, RosterState, build_state, load_operbox,
                                 load_overrides, load_skland)


def _w(p: Path, obj) -> Path:
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return p


def test_precedence_user_over_skland_over_operbox(tmp_path: Path):
    ob = _w(tmp_path / "ob.json", {"own_opers": [{"name": "桃金娘", "elite": 1, "level": 30, "own": True}]})
    sk = _w(tmp_path / "sk.json", {"ops": [{"name": "桃金娘", "elite": 2, "level": 70,
                                            "potential": 6, "skillLvl": 8, "mainSkillLvl": 7}]})
    ov = _w(tmp_path / "ov.json", {"ops": {"桃金娘": {"elite": 2, "level": 80, "skill_level": 10}}})
    st = build_state(operbox=ob, skland=sk, overrides=ov)
    op = st.ops["桃金娘"]
    assert op.elite == 2 and op.level == 80 and op.skill_level == 10
    assert op.provenance == "user_confirmed"
    ok, reasons = op.meets(2, 70, 10)
    assert ok, reasons


def test_skland_skill_mapping(tmp_path: Path):
    sk = _w(tmp_path / "sk.json", {"ops": [
        {"name": "a", "elite": 2, "level": 60, "skillLvl": 7, "mainSkillLvl": 7},   # 专一 -> 8
        {"name": "b", "elite": 2, "level": 60, "skillLvl": 9, "mainSkillLvl": 7},   # 专三 -> 10
        {"name": "c", "elite": 0, "level": 55, "skillLvl": 7, "mainSkillLvl": 7},   # 普通7 边界 -> 8（已知可接受）
        {"name": "d", "elite": 0, "level": 55, "skillLvl": 5, "mainSkillLvl": 5},   # 普通5
    ]})
    ops = {o.name: o for o in load_skland(sk)}
    assert ops["a"].skill_level == 8
    assert ops["b"].skill_level == 10
    assert ops["c"].skill_level == 8
    assert ops["d"].skill_level == 5
    assert ops["a"].skill_verified


def test_operbox_has_no_skill(tmp_path: Path):
    ob = _w(tmp_path / "ob.json", {"own_opers": [{"name": "芬", "elite": 0, "level": 1, "own": True}]})
    st = build_state(operbox=ob)
    op = st.ops["芬"]
    assert op.skill_level is None and not op.skill_verified
    ok, reasons = op.meets(0, 1, 7)
    assert not ok
    assert "skill_unknown in reasons or reasons == ['skill_unknown']".startswith("x") or "skill_unknown" in reasons


def test_deployment_mask(tmp_path: Path):
    st = RosterState()
    st.merge_in(RosterOp(name="克洛丝", elite=0, level=55, skill_level=7, provenance="skland", skill_verified=True))
    st.merge_in(RosterOp(name="芬", elite=0, level=1, skill_level=None, provenance="operbox"))
    lineup = [
        {"name": "克洛丝", "req": {"elite": 0, "level": 55, "skill_level": 7}},
        {"name": "芬", "req": {"elite": 0, "level": 30, "skill_level": 0}},
        {"name": "不存在", "req": {"elite": 0, "level": 1, "skill_level": 0}},
    ]
    rows = st.deployment_mask(lineup, allow_support=1)
    by = {r["name"]: r for r in rows}
    assert by["克洛丝"]["deployable"]
    assert not by["芬"]["deployable"] and any(r.startswith("level") for r in by["芬"]["reasons"])
    assert not by["不存在"]["deployable"] and by["不存在"]["support_ok"]
    # 不支持助战时不存在应全数不 ok
    rows0 = st.deployment_mask(lineup, allow_support=0)
    assert all(not r["deployable"] for r in rows0 if r["name"] == "不存在")


def test_merge_into_state_scales():
    st = RosterState()
    st.merge_in(RosterOp(name="x", elite=1, level=1, provenance="operbox"))
    st.merge_in(RosterOp(name="x", elite=2, level=90, skill_level=10, provenance="user_confirmed", skill_verified=True))
    js = st.to_json()
    assert js["n_own"] == 1 and js["provenance_counts"]["user_confirmed"] == 1
