"""Deploy 动作非干员名的形式化分类（词表数据在 configs/deploy_vocab.yaml）。

背景：prts.plus（MAA copilot）作业的 Deploy target 是自由文本。除真实干员名外还稳定出现
三类名字，本模块把它们形式化为 action subtype（而不是笼统当作查表失败）：

1. ``operator``      —— 真实干员（含别名改写后命中，如 麒麟X夜刀 → 麒麟R夜刀）；
2. ``device``        —— 关卡装置 / 干员召唤物 token，原始名保留；
3. ``category``      —— 类别通配占位符（奶盾 / 单奶 / 速狙 …），映射到
   {profession, sub_hint, position_hint?} 的规范小词表；
4. ``unknown_name``  —— 词表之外的名字，保留原文并由调用方计数。

匹配管线：先规范化（去装饰引号 / 书名号 → 分隔符截头 → 练度标注尾去除 → 尾数字去除），
每个候选依次尝试 干员（含别名）→ device → category，首个命中生效；
TODO(M1): 从本机 level_*.json token 列表自动扩充 devices（长尾约 2.6k 个低频名）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import yaml

# 仓库内默认词表：configs/deploy_vocab.yaml
DEFAULT_VOCAB_REL = Path("configs") / "deploy_vocab.yaml"

# 装饰字符：书名号 / 各式引号，作业里常用于括住召唤物名（如“弦惊”、【奶盾】）
_DECOR_CHARS = "【】“”\"‘’「」『』"

# 分隔符：类别词 + 练度标注的常见分隔（如 医疗：精一满级及以上）
_SEPARATORS = (":", "：", " ", ",", "，")

# 练度标注后缀（从命中关键词起截尾，如 奶盾：精一满级及以上 → 奶盾）
_ANNOTATION_RE = re.compile(r"(精[一二0-9]|满级|满潜|及以上|专[一二三0-9]|模组|练度)")

# 尾数字（槽位编号，如 奶1 / 速狙2 / 单奶1）
_TRAILING_DIGITS_RE = re.compile(r"[0-9０-９]+$")


class DeployKind(StrEnum):
    """Deploy target 的 subtype（episode JSON 中落盘的字符串值）。"""

    OPERATOR = "operator"
    DEVICE = "device"
    CATEGORY = "category"
    UNKNOWN_NAME = "unknown_name"


@dataclass(frozen=True)
class CategoryEntry:
    """类别通配词条。profession 为 None 表示跨职业功能性站位描述。"""

    category_id: str
    profession: str | None
    sub_hint: str
    position_hint: str | None
    term: str  # 命中的词形（便于追踪，规范名见 category_id）


@dataclass(frozen=True)
class DeployClassification:
    """一个 Deploy 名字的分类结果。"""

    raw: str
    kind: DeployKind
    matched_form: str  # 实际命中的候选形（规范化后），未命中时等于 raw
    category: CategoryEntry | None = None


class OperatorLookup(Protocol):
    """干员名解析接口：输入（可能经别名改写的）名字，返回 char_id 或 None。"""

    def resolve(self, name: str) -> str | None: ...


def default_vocab_path() -> Path:
    """默认词表路径：<repo>/configs/deploy_vocab.yaml。"""
    return Path(__file__).resolve().parents[3] / DEFAULT_VOCAB_REL


class DeployVocab:
    """deploy_vocab.yaml 的加载与分类器。"""

    def __init__(
        self,
        devices: frozenset[str],
        terms: dict[str, CategoryEntry],
        aliases: dict[str, str],
        vocab_version: str,
    ) -> None:
        self._devices = devices
        self._terms = terms
        self._aliases = aliases
        self.vocab_version = vocab_version

    @classmethod
    def load(cls, path: Path | None = None) -> DeployVocab:
        """从 yaml 加载词表；结构错误时抛 ValueError（指明文件与键）。"""
        p = (path or default_vocab_path()).expanduser().resolve()
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"词表不是 yaml 对象：{p}")

        devices = frozenset(str(d).strip() for d in (raw.get("devices") or []) if str(d).strip())
        terms: dict[str, CategoryEntry] = {}
        for item in raw.get("categories") or []:
            cid = str(item["id"])
            profession = item.get("profession")
            position_hint = item.get("position_hint")
            for term in item.get("terms") or []:
                term = str(term).strip()
                if not term:
                    continue
                if term in terms:
                    raise ValueError(f"词表 category 词条重复（{p}）：{term}")
                terms[term] = CategoryEntry(
                    category_id=cid,
                    profession=str(profession) if profession else None,
                    sub_hint=str(item.get("sub_hint", "any")),
                    position_hint=str(position_hint) if position_hint else None,
                    term=term,
                )
        aliases = {str(k): str(v) for k, v in (raw.get("aliases") or {}).items()}
        return cls(
            devices=devices,
            terms=terms,
            aliases=aliases,
            vocab_version=str(raw.get("vocab_version", "")),
        )

    @property
    def device_count(self) -> int:
        """device token 数（报告用）。"""
        return len(self._devices)

    @property
    def category_term_count(self) -> int:
        """category 词形数（报告用）。"""
        return len(self._terms)

    def canonical_operator_name(self, name: str) -> str:
        """别名改写：命中 aliases 时返回官方名，否则原样返回。"""
        return self._aliases.get(name, name)

    def _candidates(self, name: str) -> list[str]:
        """生成匹配候选（保序去重）：原文 → 去装饰 → 分隔符截头 → 去标注 → 去尾数字。"""
        out: list[str] = []

        def add(v: str) -> None:
            v = v.strip()
            if v and v not in out:
                out.append(v)

        base = name.strip()
        add(base)
        undecorated = base
        for ch in _DECOR_CHARS:
            undecorated = undecorated.replace(ch, "")
        add(undecorated)
        heads = [undecorated]
        for sep in _SEPARATORS:
            if sep in undecorated:
                heads.append(undecorated.split(sep, 1)[0])
        for head in heads:
            add(head)
            no_anno = _ANNOTATION_RE.split(head, maxsplit=1)[0]
            add(no_anno)
            add(_TRAILING_DIGITS_RE.sub("", no_anno))
            add(_TRAILING_DIGITS_RE.sub("", head))
        return out

    def classify(self, name: str, ops: OperatorLookup) -> DeployClassification:
        """分类一个 Deploy/Skill/Retreat 目标名（规则顺序见模块 docstring）。"""
        for cand in self._candidates(name):
            char_id = ops.resolve(self.canonical_operator_name(cand))
            if char_id is not None:
                return DeployClassification(raw=name, kind=DeployKind.OPERATOR, matched_form=cand)
            if cand in self._devices:
                return DeployClassification(raw=name, kind=DeployKind.DEVICE, matched_form=cand)
            entry = self._terms.get(cand)
            if entry is not None:
                return DeployClassification(
                    raw=name, kind=DeployKind.CATEGORY, matched_form=cand, category=entry
                )
        return DeployClassification(raw=name, kind=DeployKind.UNKNOWN_NAME, matched_form=name)
