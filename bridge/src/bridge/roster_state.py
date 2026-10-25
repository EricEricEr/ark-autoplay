"""名册统一入口（RosterState）——练度数据双通道 + 部署掩码。

用户决策（2026-10-11）：练度读取进最终模型分两种情况——
  A. 可以登录森空岛扫码的账号：直接扫码批量获取（skland 通道，含技能/潜能/模组）
  B. 无法登录森空岛的账号（如自抽号）：暂用 MAA OperBox 实拍通道
覆盖层：user_overrides.json（用户亲口读数）——任何情况下都优先。
优先级：user_confirmed > skland > operbox。

技能口径统一为数据集/prts.plus 约定：普通技能 1-7；专精一/二/三 = 8/9/10。
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PRECEDENCE = {"operbox": 1, "skland": 2, "user_confirmed": 3}


@dataclass
class RosterOp:
    name: str
    elite: int = 0
    level: int = 0
    potential: int = 0
    skill_level: int | None = None      # 1-7/8-10 口径；None = 数据源拿不到（operbox）
    main_skill_lvl: int | None = None   # skland 原值（普通技能级，专精时恒 7）
    provenance: str = ""                # operbox / skland / user_confirmed
    skill_verified: bool = False        # skill_level 是否来自可信源（skland/user）
    extra: dict[str, Any] = field(default_factory=dict)

    def meets(
        self, elite: int = 0, level: int = 0, skill_level: int = 0
    ) -> tuple[bool, list[str]]:
        """返回 (是否满足, 不满足原因列表)。skill_level=0 表示作业无技能要求。"""
        reasons: list[str] = []
        if self.elite < elite:
            reasons.append(f"elite {self.elite}<{elite}")
        elif self.elite == elite and self.level < level:
            reasons.append(f"level {self.level}<{level}")
        if skill_level:
            if self.skill_level is None:
                reasons.append("skill_unknown")
            elif self.skill_level < skill_level:
                reasons.append(f"skill {self.skill_level}<{skill_level}")
        return (len([r for r in reasons if r != "skill_unknown"]) == 0 and
                "skill_unknown" not in reasons), reasons


@dataclass
class RosterState:
    ops: dict[str, RosterOp] = field(default_factory=dict)

    def merge_in(self, op: RosterOp) -> None:
        cur = self.ops.get(op.name)
        newer = PRECEDENCE.get(op.provenance, 0)
        if cur is None or newer >= PRECEDENCE.get(cur.provenance, 0):
            self.ops[op.name] = op

    def deployment_mask(
        self, lineup: Iterable[dict[str, Any]], allow_support: int = 0
    ) -> list[dict[str, Any]]:
        """对一组 lineup 条目逐个输出部署可用性。

        lineup 条目形如 {"name": ..., "req": {"elite","level","skill_level", ...}}
        返回每条的掩码记录；缺员 ≤ allow_support 时标记 support_ok=True。
        """
        out = []
        missing = 0
        rows = []
        for e in lineup:
            name = str(e.get("name") or "")
            rq = e.get("req") or {}
            re_ = rq.get("elite") or 0
            rl = rq.get("level") or 0
            sl = rq.get("skill_level") or 0
            op = self.ops.get(name)
            if op is None:
                missing += 1
                rows.append({"name": name, "own": False, "deployable": False,
                             "reasons": ["missing"], "own_state": None,
                             "required": {"elite": re_, "level": rl,
                                          "skill_level": sl}})
                continue
            ok, reasons = op.meets(re_, rl, sl)
            _skill_ver = op.skill_verified or (sl == 0) or (
                op.skill_level is not None
            )
            rows.append({"name": name, "own": True, "deployable": ok,
                         "reasons": reasons,
                         "skill_verified": bool(op.skill_level is not None),
                         "own_state": {"elite": op.elite, "level": op.level,
                                       "skill_level": op.skill_level,
                                       "provenance": op.provenance},
                         "required": {"elite": re_, "level": rl, "skill_level": sl}})
        for row in rows:
            if row.get("own") is False:
                row["support_ok"] = missing <= allow_support
        out = rows
        return out

    def to_json(self) -> dict[str, Any]:
        return {"schema": "roster_state/1.0",
                "n_own": len(self.ops),
                "provenance_counts": {
                    p: sum(1 for o in self.ops.values() if o.provenance == p)
                    for p in PRECEDENCE
                },
                "own_opers": [
                    asdict(o) for o in sorted(self.ops.values(), key=lambda x: x.name)
                ]}


# ---------- 通道装载 ----------

def load_operbox(path: Path) -> list[RosterOp]:
    """通道 B：MAA OperBox 实拍。无技能字段。"""
    j = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for o in (j.get("own_opers") or []):
        if not o.get("own", True):
            continue
        out.append(RosterOp(
            name=str(o["name"]), elite=int(o.get("elite") or 0),
            level=int(o.get("level") or 0),
            potential=int(o.get("potential") or 0), skill_level=None,
            main_skill_lvl=None,
            provenance="operbox", skill_verified=False))
    return out


def _skland_skill(o: dict[str, Any]) -> tuple[int | None, int | None]:
    """skland skillLvl(0..9) & mainSkillLvl -> 数据集口径 skill_level(None 若全未知)。
    规则（与 data-validation/roster_analysis_v2.py 一致）：
      raw<=6 -> raw；raw>=7 且 main>=7 -> raw+1（专一8/专二9/专三10）；
      raw==7 且 main<7 -> 7。
    诚实边界：普通7 与 专一7 无法二值化时按普通7。
    """
    raw = o.get("skillLvl")
    main = o.get("mainSkillLvl")
    if raw is None:
        return None, main
    if raw <= 6:
        return raw, main
    if (main or 0) >= 7:
        return raw + 1, main
    return 7, main


def load_skland(path: Path) -> list[RosterOp]:
    """通道 A：森空岛扫码批量名册（roster_skland.json）。"""
    j = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for o in (j.get("ops") or []):
        sk, main = _skland_skill(o)
        out.append(RosterOp(
            name=str(o["name"]), elite=int(o.get("elite") or 0),
            level=int(o.get("level") or 0),
            potential=int(o.get("potential") or 0), skill_level=sk,
            main_skill_lvl=main,
            provenance="skland", skill_verified=sk is not None))
    return out


def load_overrides(path: Path) -> list[RosterOp]:
    """覆盖层：用户亲口核对数据（最高优先级）。"""
    if not path.is_file():
        return []
    j = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for name, p in (j.get("ops") or {}).items():
        out.append(RosterOp(
            name=str(name), elite=int(p.get("elite") or 0),
            level=int(p.get("level") or 0),
            potential=int(p.get("potential") or 0),
            skill_level=p.get("skill_level"), main_skill_lvl=p.get("mainSkillLvl"),
            provenance="user_confirmed",
            skill_verified=p.get("skill_level") is not None,
            extra={"note": p.get("note", "")}))
    return out


def build_state(*, operbox: Path | None = None, skland: Path | None = None,
                overrides: Path | None = None) -> RosterState:
    st = RosterState()
    if operbox and operbox.is_file():
        for o in load_operbox(operbox):
            st.merge_in(o)
    if skland and skland.is_file():
        for o in load_skland(skland):
            st.merge_in(o)
    if overrides and overrides.is_file():
        for o in load_overrides(overrides):
            # 逐字段合并（2026-10-25 修）：覆盖层常为「只给技能字段」的部分覆盖
            # （user_overrides 的 note 明确「精级/等级请用 OperBox 新值」），
            # 整行替换会把没给的 elite/level/potential 清零。约定：
            # 覆盖层数值字段为 0 视为「未提供」（实读不为 0），技能字段 None 同理。
            cur = st.ops.get(o.name)
            if cur is None:
                st.ops[o.name] = o
                continue
            if o.elite:
                cur.elite = o.elite
            if o.level:
                cur.level = o.level
            if o.potential:
                cur.potential = o.potential
            if o.skill_level is not None:
                cur.skill_level = o.skill_level
            if o.main_skill_lvl is not None:
                cur.main_skill_lvl = o.main_skill_lvl
            if o.extra:
                cur.extra.update(o.extra)
            cur.provenance = f"user_confirmed+{cur.provenance}"
            cur.skill_verified = cur.skill_verified or (cur.skill_level is not None)
    return st


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="名册统一入口：双通道装载 + 合并 + （可选）部署掩码试算"
    )
    ap.add_argument("--operbox", type=Path, default=None,
                    help="MAA OperBox JSON（通道 B）")
    ap.add_argument("--skland", type=Path, default=None,
                    help="森空岛 roster_skland.json（通道 A）")
    ap.add_argument("--overrides", type=Path, default=None,
                    help="user_overrides.json（最高优先级）")
    ap.add_argument("--out", type=Path, default=None, help="合并名册输出 JSON")
    ap.add_argument("--mask-demo", type=Path, default=None,
                    help="（可选）对一个 copilot.jsonl 的前 N 条作业"
                         "计算部署掩码并打印摘要")
    ap.add_argument("--mask-n", type=int, default=5)
    args = ap.parse_args(argv)

    st = build_state(operbox=args.operbox, skland=args.skland, overrides=args.overrides)
    print(f"merged {len(st.ops)} ops: {st.to_json()['provenance_counts']}")

    if args.out:
        args.out.write_text(
            json.dumps(st.to_json(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"written -> {args.out}")

    if args.mask_demo and args.mask_demo.is_file():
        n = 0
        for line in args.mask_demo.open(encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows = st.deployment_mask(rec.get("lineup") or [], allow_support=1)
            ok = sum(1 for r in rows if r["deployable"] or r.get("support_ok"))
            n += 1
            if n <= args.mask_n:
                bad = [
                    f"{r['name']}<{','.join(r['reasons'])}>"
                    for r in rows
                    if not r["deployable"] and not r.get("support_ok")
                ]
                print(f"[{rec.get('id')}] {ok}/{len(rows)} deployable; "
                      f"blocks: {'; '.join(bad[:3]) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
