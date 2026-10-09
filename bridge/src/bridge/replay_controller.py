"""作业重放调度模块（数据工厂主循环）。

职责（设计文档 §5.4 / §10）：按 jobs 文件逐个重放 prts.plus 作业：
构建 MAA 作业 JSON（保留来源元数据）→ 固定点位导航到关卡 briefing
→ 下发 Copilot 接管作战 → 全程采集（回调事件流 + 战场状态 + 1.5s 节拍截图 +
CopilotAction 即时补拍 + 结算抓帧）→ 写 episode 束（bundle 结构沿用
``data-validation/capture_episode.py`` 已验证格式）。

输入为 ``configs/jobs_main_v1.json``（内嵌转换好的 MAA 形态作业 +
来源署名，控制器离线可用）；输出交给 ``episode_writer`` 落盘。

战场状态采集（2026-10-09 新增）
-------------------------------
使用**打过补丁的 MaaCore**（``patches/0001-battle-state-callback.patch``）时，
每个作业动作执行前会收到 ``what="BattleState"`` 回调，含费用/击杀/待部署栏/
已部署干员；本模块将其转为 ``battle_states.jsonl`` 并写进 episode 清单
（``battle_states_file`` / ``battle_states`` 摘要）。

用**官方发行版**时不发此回调 → ``battle_states.jsonl`` 不生成、
``battle_states_file`` 为 None、``battle_states.n = 0``。
**两种运行方式产出的 bundle 结构兼容**，可混存；靠 ``battle_states_file``
是否为 None 区分这批数据有没有状态。

已知边界（如实记录不求完美）：

- Copilot 任务链在作业动作执行完即收尾，作战自然结束靠
  ``navigator.wait_battle_settle`` 轮询蓝钮推进（期间桌面上会闪过
  结算画面并被抓帧）；胜负判定属 TODO(M0.x)；
- 断点续跑仅按"episode 目录已存在即跳过"（``--skip-done``）；
- 中途失败（导航/下发失败）记摘要后继续下一局。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from bridge.battle_state import battle_state_to_record, summarize_states
from bridge.episode_writer import EpisodeWriter
from bridge.maa_driver import MaaDriver, load_instance_config
from bridge.navigator import NavigationError, Navigator
from bridge.state_logger import StateLogger

# 数据集动作类型 → MAA 作业动作白名单（其余如 Output/MoveCamera 丢弃）
_MAA_ACTION_TYPES = ("Deploy", "Skill", "Retreat", "SpeedUp", "SkillDaemon")

_CHAIN_TIMEOUT_S = 480.0
"""任务链最长执行时间（防卡死保护）。"""

_SETTLE_TIMEOUT_S = 300.0
"""作战自然结束+结算推进的最长等待。"""


def stage_id_to_code(stage_id: str) -> str:
    """数据集 stage_id → 游戏短码：``main_00-01`` → ``0-1``（v1 仅主线）。"""
    m = re.fullmatch(r"main_(\d{2})-(\d{2})", stage_id)
    if not m:
        raise ValueError(f"v1 不支持的 stage_id: {stage_id}")
    return f"{int(m.group(1))}-{int(m.group(2))}"


def dataset_record_to_job_entry(rec: dict[str, Any]) -> dict[str, Any]:
    """数据集 copilot.jsonl 单条 → 内嵌作业条目（离线可回放）。

    返回 ``{"source": {...来源署名}, "maa_job": {...MAA 作业形态}}``；
    编队要求整体省略（实测 MAA 会把 ``module: 0`` 当作模组硬要求，
    对无模组低星干员直接判 OperatorMissing；原始 req 留在 source 溯源）。
    """
    stage_id = rec["stage_id"]
    code = stage_id_to_code(stage_id)
    opers = []
    for op in rec.get("lineup", []):
        opers.append(
            {
                "name": op["name"],
                "skill": op.get("skill") or 1,
                "skill_usage": op.get("skill_usage") or 0,
            }
        )
    actions = []
    for act in rec.get("actions", []):
        if act["type"] not in _MAA_ACTION_TYPES:
            continue
        out: dict[str, Any] = {"type": act["type"]}
        for key in ("name", "location", "direction", "pre_delay", "post_delay"):
            val = act.get(key)
            if val not in (None, "", "None"):
                out[key] = val
        if act.get("kills"):
            out["kills"] = act["kills"]
        if act.get("costs"):
            out["costs"] = act["costs"]
        actions.append(out)
    source = {
        "type": "prts.plus",
        "job_id": rec["id"],
        "title": rec.get("title", ""),
        "like": rec.get("like", 0),
        "views": rec.get("views", 0),
        "rating_ratio": rec.get("rating_ratio"),
        "upload_time": rec.get("upload_time", ""),
        "stage_id_raw": stage_id,
        "lineup_raw": [
            {"name": op["name"], "id": op.get("id"), "req": op.get("req")}
            for op in rec.get("lineup", [])
        ],
        "comment": (
            f"来源 prts.plus 作业 #{rec['id']}（ak_dataset_v0.1 copilot.jsonl），"
            "编队要求已置零回放，原始 req 见 lineup_raw"
        ),
    }
    maa_job = {
        "stage_name": code,
        "opers": opers,
        "groups": rec.get("groups", []),
        "actions": actions,
        "doc": {
            "title": rec.get("title", "") or f"prts.plus #{rec['id']}",
            "details": (
                f"来源: prts.plus 作业 #{rec['id']}（ark-autoplay 数据工厂重放）"
            ),
        },
        "minimum_required": "v4.0.0",
    }
    return {"source": source, "maa_job": maa_job}


def load_jobs(path: str | Path) -> list[dict[str, Any]]:
    """读取 jobs 定义文件（支持 {"jobs":[...]} 或裸 list）。"""
    with open(path, encoding="utf-8") as f:
        jobs = json.load(f)
    if isinstance(jobs, dict) and "jobs" in jobs:
        jobs = jobs["jobs"]
    if not isinstance(jobs, list):
        raise ValueError(f"jobs 文件结构非法: {path}")
    for j in jobs:
        if "source" not in j or "maa_job" not in j:
            raise ValueError("jobs 条目缺 source/maa_job")
    return jobs


class ReplayController:
    """作业重放调度器：队列消费 jobs 文件，逐局产出 episode。"""

    def __init__(
        self,
        instance_cfg_path: str | Path,
        nav_cfg_path: str | Path,
        work_root: str | Path,
    ) -> None:
        """装配驱动/导航/写出；``work_root`` 为数据落盘根（机器本地）。"""
        self._work = Path(work_root)
        (self._work / "tmp" / "maa_jobs").mkdir(parents=True, exist_ok=True)
        maa_user = self._work / "debug" / "maa_user"
        maa_user.mkdir(parents=True, exist_ok=True)
        self._driver = MaaDriver(load_instance_config(instance_cfg_path), maa_user)
        self._nav = Navigator(self._driver, Navigator.load_nav_config(nav_cfg_path))
        self._writer = EpisodeWriter(self._work)

    def _write_maa_job(self, entry: dict[str, Any]) -> Path:
        path = self._work / "tmp" / "maa_jobs" / f"job_{entry['source']['job_id']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry["maa_job"], f, ensure_ascii=False, indent=1)
        return path

    def _run_one(self, entry: dict[str, Any]) -> dict[str, Any]:
        """重放一个作业，返回结果摘要。"""
        src = entry["source"]
        stage_code = entry["maa_job"]["stage_name"]
        stage_id = src["stage_id_raw"]
        job_id = src["job_id"]
        print(f"[job {job_id}] {stage_id} ({stage_code}) 开始", flush=True)
        summary: dict[str, Any] = {"job_id": job_id, "stage_code": stage_code}

        t0 = time.time()
        t0_wall = int(t0 * 1000)
        records: list[dict[str, Any]] = []
        states: list[dict[str, Any]] = []

        def rec(msg: str, **kw: Any) -> None:
            records.append(
                {"t_ms": int(time.time() * 1000) - t0_wall, "msg": msg, **kw}
            )

        def rec_state(details: dict[str, Any]) -> None:
            """把 MaaCore 的 BattleState 回调转成协议的战场状态记录。

            需 patched MaaCore（``patches/0001-battle-state-callback.patch``）；
            官方发行版不会发此回调，届时本函数永不触发、``states`` 保持为空。
            """
            states.append(
                battle_state_to_record(
                    details,
                    int(time.time() * 1000) - t0_wall,
                    job_id=job_id,
                    stage_code=stage_code,
                )
            )

        try:
            self._nav.goto_stage_briefing(stage_code)
        except NavigationError as exc:
            print(f"[job {job_id}] 导航失败: {exc}", flush=True)
            summary.update(ok=False, error=f"nav: {exc}")
            return summary

        run_dir = self._writer.begin(stage_id, job_id)
        job_file = self._write_maa_job(entry)

        def on_shot(tag: str, rel: str) -> None:
            if tag == "tick":
                rec("ShotTick", file=rel)
            elif tag == "action":
                rec("ShotOnAction", file=rel)
            else:
                rec("Shot", tag=tag, file=rel)

        logger = StateLogger(self._driver, run_dir, on_shot=on_shot)
        logger.start()
        logger.snap("briefing")

        tid = self._driver.append_copilot(job_file, formation=True)
        if tid <= 0:
            logger.stop()
            print(f"[job {job_id}] append_copilot 被拒绝 (tid={tid})", flush=True)
            summary.update(ok=False, error=f"append tid={tid}")
            (run_dir / "error.txt").write_text(f"append tid={tid}", encoding="utf-8")
            return summary

        mark = self._driver.mark()
        if not self._driver.start():
            logger.stop()
            summary.update(ok=False, error="asst.start 失败")
            return summary

        # 作战过程：轮询事件流，CopilotAction 触发补拍
        cursor = mark
        terminal: str | None = None
        seen_actions = 0
        deadline = time.monotonic() + _CHAIN_TIMEOUT_S
        while self._driver.running() and terminal is None:
            fresh, cursor = self._driver.events_since(cursor)
            for evt in fresh:
                rec(evt.msg, details=evt.details)
                if evt.msg == "TaskChainStart":
                    logger.set_active(True)
                elif evt.msg == "SubTaskExtraInfo":
                    what = evt.details.get("what")
                    if what == "CopilotAction":
                        seen_actions += 1
                        logger.snap("action")
                    elif what == "BattleState":
                        # patched MaaCore 才有；官方发行版不会走到这里
                        rec_state(evt.details.get("details") or {})
                elif evt.msg in ("TaskChainCompleted", "TaskChainError"):
                    terminal = evt.msg
            if time.monotonic() > deadline:
                print(f"[job {job_id}] 链超时，stop", flush=True)
                self._driver.stop()
                time.sleep(2.0)
                terminal = terminal or "Timeout"
                break
            time.sleep(0.2)
        print(
            f"[job {job_id}] 任务链收尾={terminal} 动作事件={seen_actions}", flush=True
        )

        # 作战自然结束 + 结算推进：链收尾≠作战结束，轮询蓝钮回到 briefing
        logger.set_active(False)
        result_files: list[str] = []
        settle_deadline = time.monotonic() + _SETTLE_TIMEOUT_S
        settled = False
        while time.monotonic() < settle_deadline:
            time.sleep(3.0)
            rel = logger.snap("settle")
            result_files.append(rel)
            if self._nav.briefing_open():
                settled = True
                break
            x, y = self._nav._ui["settle_tap"]
            self._driver.tap(int(x), int(y))
        logger.stop()
        fresh, cursor = self._driver.events_since(cursor)
        for evt in fresh:
            rec(evt.msg, details=evt.details)
        print(
            f"[job {job_id}] 结算推进 settled={settled} snaps={len(result_files)}",
            flush=True,
        )

        self._writer.write_events(run_dir, records)
        states_path = self._writer.write_battle_states(run_dir, states)
        state_summary = summarize_states(states)
        result_first = result_files[0] if result_files else ""
        result_last = result_files[-1] if result_files else ""
        episode = {
            "episode_id": f"replay-{job_id}-{time.strftime('%Y%m%d-%H%M%S')}",
            "stage_id": stage_id,
            "stage_code": stage_code,
            "source": {
                "type": "prts.plus",
                "job_id": job_id,
                "note": src.get("comment", ""),
            },
            "job_file": str(job_file),
            "result_screen_files": [result_first, result_last],
            "action_events_file": "action_events.jsonl",
            # 战场状态（需 patched MaaCore）：无记录时为 None 且不落文件
            "battle_states_file": states_path.name if states_path else None,
            "battle_states": state_summary,
            "shot_files": None,
            "duration_ms": int((time.time() - t0) * 1000),
            "copilot_actions": seen_actions,
            "settled_back_to_briefing": settled,
        }
        ep_path = self._writer.write_episode(run_dir, episode)
        summary.update(
            ok=terminal == "TaskChainCompleted",
            episode=str(ep_path),
            copilot_actions=seen_actions,
            battle_states=state_summary["n"],
            settled=settled,
        )
        print(
            f"[job {job_id}] episode -> {ep_path}  "
            f"战场状态 {state_summary['n']} 条{state_summary}",
            flush=True,
        )
        return summary

    def run(
        self,
        jobs_path: str | Path,
        only: set[str] | None = None,
        limit: int | None = None,
        skip_done: bool = False,
    ) -> list[dict[str, Any]]:
        """消费 jobs 队列；``only`` 按 job_id 过滤，``skip_done`` 跳过已落盘局。"""
        jobs = load_jobs(jobs_path)
        if only:
            jobs = [j for j in jobs if str(j["source"]["job_id"]) in only]
        if skip_done:
            jobs = [j for j in jobs if not self._has_episode(j)]
        if limit:
            jobs = jobs[:limit]
        if not jobs:
            print("[factory] 无待跑作业")
            return []
        if not self._driver.connect():
            raise RuntimeError("MAA 连接失败")
        summaries = []
        try:
            for entry in jobs:
                summaries.append(self._run_one(entry))
        finally:
            self._driver.close()
        out = self._work / "debug" / f"factory_summary_{int(time.time())}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=1)
        print(f"[factory] 完成 {len(summaries)} 局 -> {out}")
        return summaries

    def _has_episode(self, entry: dict[str, Any]) -> bool:
        ep_dir = self._work / "episodes" / entry["source"]["stage_id_raw"]
        marker = f"job-{entry['source']['job_id']}-"
        return ep_dir.is_dir() and any(
            p.name.startswith(marker) for p in ep_dir.iterdir()
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    """工厂 CLI：``python -m bridge.replay_controller --work <数据根>``。"""
    root = _repo_root()
    parser = argparse.ArgumentParser(description="ark-autoplay 重放数据工厂 v1WIP")
    parser.add_argument(
        "--jobs", default=str(root / "configs" / "jobs_main_v1.json"), help="作业队列"
    )
    parser.add_argument(
        "--instance",
        default=str(root / "configs" / "instances" / "mumu_local.yaml"),
        help="实例定义",
    )
    parser.add_argument(
        "--nav", default=str(root / "configs" / "nav_main.yaml"), help="导航点位表"
    )
    parser.add_argument("--work", required=True, help="数据落盘根（本地目录）")
    parser.add_argument("--only", default=None, help="只跑指定 job_id（逗号分隔）")
    parser.add_argument("--limit", type=int, default=None, help="最多跑 N 个")
    parser.add_argument(
        "--skip-done", action="store_true", help="跳过已有 episode 目录的作业"
    )
    args = parser.parse_args()

    controller = ReplayController(args.instance, args.nav, args.work)
    only = set(args.only.split(",")) if args.only else None
    summaries = controller.run(
        args.jobs, only=only, limit=args.limit, skip_done=args.skip_done
    )
    ok = sum(1 for s in summaries if s.get("ok"))
    print(f"[factory] ok {ok}/{len(summaries)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
