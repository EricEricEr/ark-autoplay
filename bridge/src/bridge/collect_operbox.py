"""采集账号 OperBox 名册：沿用已验证流程，实例/输出路径均由参数传入。

只保存正常结束且非空的最终快照；失败不覆盖已有名册。OperBox 不含完整技能和
模组等级，需游戏详情截图或已有可信导出补充。模块导入和 --help 不连接游戏。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from bridge.maa_driver import MaaDriver, MaaEvent, load_instance_config
from bridge.nav_maa import wait_settle


def select_snapshot(events: Iterable[MaaEvent]) -> dict[str, Any] | None:
    """选取非空、done=true 且拥有干员数最多的回调，忽略加载期空快照。"""
    best: dict[str, Any] | None = None
    for event in events:
        body = event.details
        if not isinstance(body, dict) or body.get("what") != "OperBoxInfo":
            continue
        details = body.get("details")
        if not isinstance(details, dict) or not details.get("done"):
            continue
        own = details.get("own_opers")
        if not isinstance(own, list) or not own:
            continue
        if best is None or len(own) >= len(best["own_opers"]):
            best = details
    return best


def main(argv: list[str] | None = None) -> int:
    """连接指定实例、回主页、扫描干员，输出与 roster_state 兼容的名册。"""
    parser = argparse.ArgumentParser(
        description="采集账号 OperBox 名册（不含完整技能/模组）"
    )
    parser.add_argument("--instance", type=Path, required=True, help="本机实例 YAML")
    parser.add_argument("--out", type=Path, required=True, help="输出名册 JSON")
    parser.add_argument("--user-dir", type=Path, required=True, help="MAA 本地日志目录")
    parser.add_argument("--timeout", type=float, default=420.0, help="扫描超时秒数")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    driver = MaaDriver(load_instance_config(args.instance), args.user_dir)
    try:
        if not driver.connect():
            print("[operbox] 连接失败")
            return 1
        if not wait_settle(driver, timeout_s=200.0):
            print("[operbox] 回主页失败，未开始扫描")
            return 2
        mark = driver.mark()
        task_id = driver.append_task("OperBox", {})
        if not task_id or not driver.start():
            print("[operbox] 任务提交/启动失败")
            return 3
        result = driver.wait_chain(timeout_s=args.timeout, mark=mark)
        snapshot = select_snapshot(result.events)
        if not result.finished_ok or snapshot is None:
            print("[operbox] 未取得正常结束的非空最终快照；已有名册保持原样")
            return 4
        own = snapshot["own_opers"]
        doc = {
            "collected_at": datetime.now().astimezone().isoformat(),
            "source": "MAA OperBox",
            "n_own": len(own),
            "own_opers": own,
            "owned_names": sorted({str(o["name"]) for o in own if o.get("name")}),
            "missing_fields": ["skill_level", "skill_mastery", "module_level"],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(args.out.name + ".tmp")
        temporary.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        temporary.replace(args.out)
        print(f"[operbox] 已保存 {len(own)} 名干员 -> {args.out}")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
