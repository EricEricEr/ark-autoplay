"""deploy_vocab 分类测试（合成词表 + 合成干员名，不含真实游戏内容）。

覆盖三类 subtype：device（关卡装置 / 召唤物）、category（类别通配）、
unknown_name；外加 operator 解析与规范化变体（装饰引号 / 练度标注 / 尾数字）。
"""

from pathlib import Path

import pytest

from ark_core.datapipe.deploy_vocab import DeployKind, DeployVocab

_SYNTH_VOCAB_YAML = """
vocab_version: "test"
aliases:
  旧称甲: 测试干员甲
devices:
  - 测试夹子
  - 测试泵站
categories:
  - id: test_aoe_heal
    terms: [合成群奶, 群疗]
    profession: MEDIC
    sub_hint: ringhealer
    position_hint: RANGED
  - id: test_shield
    terms: [合成奶盾]
    profession: TANK
    sub_hint: guardian
"""


class _FakeOps:
    """合成干员名解析：只有一个全局唯一干员名。"""

    _MAP = {"测试干员甲": "char_syn_alpha"}

    def resolve(self, name: str) -> str | None:
        return self._MAP.get(name)


@pytest.fixture()
def vocab(tmp_path: Path) -> DeployVocab:
    """写入合成词表 yaml 并加载。"""
    p = tmp_path / "vocab.yaml"
    p.write_text(_SYNTH_VOCAB_YAML, encoding="utf-8")
    return DeployVocab.load(p)


def test_operator_exact(vocab: DeployVocab) -> None:
    """干员名命中 → operator。"""
    cls = vocab.classify("测试干员甲", _FakeOps())
    assert cls.kind is DeployKind.OPERATOR
    assert cls.matched_form == "测试干员甲"


def test_operator_alias(vocab: DeployVocab) -> None:
    """别名改写后命中干员 → operator。"""
    cls = vocab.classify("旧称甲", _FakeOps())
    assert cls.kind is DeployKind.OPERATOR


def test_device_token(vocab: DeployVocab) -> None:
    """装置 / 召唤物 token → device。"""
    cls = vocab.classify("测试泵站", _FakeOps())
    assert cls.kind is DeployKind.DEVICE


def test_device_decorated_quotes(vocab: DeployVocab) -> None:
    """带装饰引号的装置名（作业常见写法）→ device。"""
    cls = vocab.classify("“测试夹子”", _FakeOps())
    assert cls.kind is DeployKind.DEVICE
    assert cls.matched_form == "测试夹子"


def test_category_term(vocab: DeployVocab) -> None:
    """类别通配占位符 → category，且带 profession / sub_hint / position_hint。"""
    cls = vocab.classify("合成群奶", _FakeOps())
    assert cls.kind is DeployKind.CATEGORY
    assert cls.category is not None
    assert cls.category.category_id == "test_aoe_heal"
    assert cls.category.profession == "MEDIC"
    assert cls.category.sub_hint == "ringhealer"
    assert cls.category.position_hint == "RANGED"


def test_category_annotation_suffix(vocab: DeployVocab) -> None:
    """类别词 + 练度标注（合成群奶：精一满级及以上）→ category。"""
    cls = vocab.classify("群疗：精一满级及以上", _FakeOps())
    assert cls.kind is DeployKind.CATEGORY
    assert cls.category is not None
    assert cls.category.category_id == "test_aoe_heal"


def test_category_trailing_digits(vocab: DeployVocab) -> None:
    """类别词 + 槽位尾数字（群疗2）→ category。"""
    cls = vocab.classify("群疗2", _FakeOps())
    assert cls.kind is DeployKind.CATEGORY


def test_unknown_name(vocab: DeployVocab) -> None:
    """词表外名字 → unknown_name，保留原文。"""
    cls = vocab.classify("完全没见过的合成名", _FakeOps())
    assert cls.kind is DeployKind.UNKNOWN_NAME
    assert cls.raw == "完全没见过的合成名"


def test_duplicate_term_rejected(tmp_path: Path) -> None:
    """词表 category 词形重复 → ValueError（配置即数据，重复即错误）。"""
    p = tmp_path / "dup.yaml"
    p.write_text(
        "categories:\n"
        "  - {id: a, terms: [重复词], profession: TANK, sub_hint: any}\n"
        "  - {id: b, terms: [重复词], profession: MEDIC, sub_hint: any}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="重复"):
        DeployVocab.load(p)
