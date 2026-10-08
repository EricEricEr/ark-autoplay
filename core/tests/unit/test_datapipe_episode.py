"""copilot_to_episode 转换器测试（合成作业 / 合成干员表 / 合成词表，不含真实游戏内容）。

覆盖：MAA→协议动作映射、aux 动作跳过计数、deploy 三类 subtype、
episode schema 形状（protocol_version / source 署名 / squad / target / timing）、
全辅助类动作作业的整局丢弃。
"""

import json
from pathlib import Path
from typing import Any

import pytest

from ark_core.datapipe.copilot_to_episode import (
    ACTION_MAP,
    AUX_ACTION_TYPES,
    EPISODE_PROTOCOL_VERSION,
    ConvertContext,
    OperatorNameIndex,
    convert_copilot,
    convert_job,
)
from ark_core.datapipe.deploy_vocab import DeployVocab

_SYNTH_OPS_JSONL = "\n".join(
    [
        json.dumps(
            {
                "id": "char_syn_alpha",
                "name": "测试干员甲",
                "profession": "WARRIOR",
                "sub": "lord",
                "rarity": "TIER_6",
                "position": "MELEE",
                "stats": {},
                "range": [],
                "skills": [],
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "id": "char_syn_beta",
                "name": "测试干员乙",
                "profession": "MEDIC",
                "sub": "physician",
                "rarity": "TIER_4",
                "position": "RANGED",
                "stats": {},
                "range": [],
                "skills": [],
            },
            ensure_ascii=False,
        ),
    ]
)

_SYNTH_VOCAB_YAML = """
vocab_version: "test"
aliases:
  旧称乙: 测试干员乙
devices:
  - 测试夹子
categories:
  - id: test_shield
    terms: [合成奶盾]
    profession: TANK
    sub_hint: guardian
"""

_STAGE_INDEX = {
    "tough_05-11": {
        "stage_id": "tough_05-11",
        "code": "T5-11",
        "name": "合成关卡",
        "type": "MAIN",
        "difficulty": "NORMAL",
        "family": "main_05",
    }
}


def _full_actions() -> list[dict[str, Any]]:
    """覆盖全部映射 + 三类 deploy + aux 的合成动作序列。"""
    return [
        {
            "type": "Deploy",
            "name": "测试干员甲",
            "location": [1, 2],
            "direction": "Up",
            "kills": 0,
            "costs": 18,
            "cost_changes": 0,
            "pre_delay": 0,
            "post_delay": 1500,
        },
        {"type": "Deploy", "name": "测试夹子", "location": [2, 2], "direction": "None"},
        {"type": "Deploy", "name": "合成奶盾", "location": [3, 2], "direction": "Down"},
        {"type": "Deploy", "name": "未知合成物", "location": [4, 2], "direction": "Left"},
        {
            "type": "Skill",
            "name": "测试干员甲",
            "kills": 3,
            "location": [1, 2],
            "direction": "None",
        },
        {
            "type": "Retreat",
            "name": "测试干员甲",
            "kills": 4,
            "location": [1, 2],
            "direction": "None",
        },
        {"type": "SpeedUp"},
        {"type": "BulletTime"},
        {"type": "SkillDaemon"},
        {"type": "Output"},
        {"type": "MoveCamera"},
        {"type": "Click"},
        {"type": "Swipe"},
        {"type": "SkillUsage"},
    ]


def _synth_job(actions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """一份合成作业（字段对应 prts.plus copilot 行）。"""
    return {
        "id": 999001,
        "stage_id": "tough_05-11",
        "stage_name_raw": "tough_05-11",
        "title": "合成作业标题",
        "upload_time": "2026-01-01T00:00:00",
        "views": 123,
        "like": 45,
        "rating_ratio": 0.75,
        "available": True,
        "lineup": [
            {
                "name": "测试干员甲",
                "id": "char_syn_alpha",
                "skill": 2,
                "skill_usage": 1,
                "req": {"elite": 2, "level": 90, "skill_level": 10, "module": 1, "potentiality": 1},
            },
            {"name": "合成奶盾", "id": None, "skill": None, "skill_usage": None, "req": None},
        ],
        "groups": [{"name": "近卫", "options": [{"name": "测试干员乙", "id": "char_syn_beta"}]}],
        "actions": _full_actions() if actions is None else actions,
    }


@pytest.fixture()
def ctx(tmp_path: Path) -> ConvertContext:
    """合成源文件 + 转换上下文。"""
    ops_path = tmp_path / "operators.jsonl"
    ops_path.write_text(_SYNTH_OPS_JSONL + "\n", encoding="utf-8")
    vocab_path = tmp_path / "vocab.yaml"
    vocab_path.write_text(_SYNTH_VOCAB_YAML, encoding="utf-8")
    return ConvertContext(
        op_index=OperatorNameIndex(ops_path),
        vocab=DeployVocab.load(vocab_path),
        stage_index=dict(_STAGE_INDEX),
        data_version="test-data-v0",
    )


def test_action_map_exact() -> None:
    """动作映射表与已验证口径逐项一致。"""
    assert ACTION_MAP == {
        "Deploy": "deploy",
        "Skill": "skill",
        "Retreat": "retreat",
        "SpeedUp": "speed",
        "BulletTime": "bullet_time",
    }
    assert set(AUX_ACTION_TYPES) == {
        "MoveCamera",
        "Output",
        "Click",
        "Swipe",
        "SkillUsage",
        "SkillDaemon",
    }


def test_episode_schema_shape(ctx: ConvertContext) -> None:
    """episode 顶层结构与关键字段形状。"""
    ep = convert_job(_synth_job(), ctx)
    assert ep["protocol_version"] == EPISODE_PROTOCOL_VERSION
    assert ep["episode_id"] == "copilot_999001"
    assert ep["data_version"] == "test-data-v0"
    assert ep["stage"]["family"] == "main_05"
    src = ep["source"]
    assert src["type"] == "prts.plus_copilot"
    assert src["job_id"] == 999001
    assert src["title"] == "合成作业标题"
    assert src["views"] == 123
    assert src["like"] == 45
    assert isinstance(ep["roster_tier"], str)
    slot0 = ep["squad"][0]
    assert slot0["char_id"] == "char_syn_alpha"
    assert slot0["skill"] == 2
    assert slot0["req"]["elite"] == 2
    slot1 = ep["squad"][1]
    assert slot1["char_id"] is None
    assert slot1["placeholder_subtype"] == "category"
    assert ep["groups"][0]["name"] == "近卫"


def test_actions_mapping_and_aux_counts(ctx: ConvertContext) -> None:
    """动作映射顺序 / aux 跳过计数 / deploy subtype 分布。"""
    ep = convert_job(_synth_job(), ctx)
    types = [a["type"] for a in ep["actions"]]
    assert types == [
        "deploy",
        "deploy",
        "deploy",
        "deploy",
        "skill",
        "retreat",
        "speed",
        "bullet_time",
    ]
    assert [a["seq"] for a in ep["actions"]] == list(range(len(types)))
    aux = ep["stats"]["aux_skipped_by_type"]
    for aux_type in AUX_ACTION_TYPES:
        assert aux[aux_type] == 1
    assert set(aux) == set(AUX_ACTION_TYPES)
    sub = ep["stats"]["deploy_subtypes"]
    assert sub == {"operator": 1, "device": 1, "category": 1, "unknown_name": 1}
    assert ctx.action_hist_aux["SkillDaemon"] == 1
    assert ctx.deploy_subtypes["device"] == 1
    assert ctx.unknown_names["未知合成物"] == 1


def test_deploy_targets_detail(ctx: ConvertContext) -> None:
    """deploy target 的 subtype 载荷：char_id / 类别映射 / 原文保留。"""
    ep = convert_job(_synth_job(), ctx)
    t_op, t_dev, t_cat, t_unk = (a["target"] for a in ep["actions"][:4])
    assert t_op["kind"] == "operator" and t_op["char_id"] == "char_syn_alpha"
    assert t_dev["kind"] == "device" and t_dev["name_raw"] == "测试夹子"
    assert t_cat["kind"] == "category"
    assert t_cat["profession"] == "TANK" and t_cat["sub_hint"] == "guardian"
    assert t_unk["kind"] == "unknown_name" and t_unk["name_raw"] == "未知合成物"
    first = ep["actions"][0]
    assert first["location"] == [1, 2]
    assert first["direction"] == "Up"
    assert first["timing"] == {
        "kills": 0,
        "costs": 18,
        "cost_changes": 0,
        "pre_delay": 0,
        "post_delay": 1500,
    }
    # 'None' 方向落 null（不出现 direction 键）
    assert "direction" not in ep["actions"][1]


def test_all_aux_job_returns_empty_actions(ctx: ConvertContext) -> None:
    """全 aux 作业：convert_job 给空 actions，由上层按开关整局丢弃。"""
    job = _synth_job(actions=[{"type": "SkillDaemon"}, {"type": "Output"}])
    ep = convert_job(job, ctx)
    assert ep["actions"] == []
    assert ep["stats"]["n_actions_aux_skipped"] == 2


def test_convert_copilot_end_to_end(ctx: ConvertContext, tmp_path: Path) -> None:
    """流式转换：3 份作业（1 份全 aux 丢弃）→ 2 行 episodes。"""
    copilot_path = tmp_path / "copilot.jsonl"
    jobs = [
        _synth_job(),
        _synth_job(actions=[{"type": "SkillDaemon"}]),
        _synth_job(actions=[{"type": "SpeedUp"}, {"type": "MoveCamera"}]),
    ]
    jobs[2]["id"] = 999003
    copilot_path.write_text(
        "\n".join(json.dumps(j, ensure_ascii=False) for j in jobs) + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "episodes.jsonl"
    res = convert_copilot(copilot_path, ctx, out_path, drop_when_all_skipped=True)
    assert res.n_jobs == 3
    assert res.n_kept == 2
    assert res.drop_reasons == {"all_actions_skipped": 1}
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    ep = json.loads(lines[0])
    assert ep["protocol_version"] == EPISODE_PROTOCOL_VERSION
