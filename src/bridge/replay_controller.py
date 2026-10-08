"""作业重放调度模块（数据工厂主循环）。

职责（对应设计文档 §5.4 / §10）：按 jobs 文件逐个重放 prts.plus 作业：
构建 MAA 作业 JSON（保留来源元数据）→ 导航到关卡 briefing → 下发
Copilot 接管作战 → 全程采集（回调事件带墙钟时间戳、节拍截图、每次
CopilotAction 补拍、战后结算模板判胜负）→ 写 episode 束。
支持按 job_id 过滤与断点跳跑（``--skip-done``）。

输入为 ``configs/jobs_main_v1.json``（已内嵌转换好的 MAA 形态作业 +
来源署名，控制器离线可用）；输出交给 ``episode_writer`` 落盘。

已知边界（v1）：

- Copilot 任务链在作业动作执行完即收尾，**作战自然结束依赖
  ``navigator.detect_result`` 结算模板判定**，模板缺失时结果记 null；
- 失败局同样产出 episode（``result.win=false``），不剔除（剔除策略属
  core 侧语料治理，本仓忠实记录）；
- 断点续跑仅按"episode 目录已存在即跳过"，局级恢复属 TODO(M0)。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from bridge.episode_writer import EpisodeWriter
from bridge.maa_driver import MaaDriver, MaaEvent, load_instance_config
from bridge.navigator import NavigationError, Navigator, ScreenBank
from bridge.state_logger import StateLogger

# 数据集动作类型 → MAA 作业动作白名单（其余如 Output/MoveCamera 丢弃）
_MAA_ACTION_TYPES = ("Deploy", "Skill", "Retreat", "SpeedUp", "SkillDaemon")

_CHAIN_TIMEOUT_S = 480.0
"""Copilot 任务链最长执行时间（关卡超时保护）。"""


def stage_id_to_code(stage_id: str) -> str:
    """数据集 stage_id → 游戏短码：``main_00-01`` → ``0-1``。

    v1 仅支持主线 ``main_XX-YY``；其余（tough_/H 关、活动等）抛错。
    """
    import re

    m = re.fullmatch(r"main_(\d{2})-(\d{2})", stage_id)
    if not m:
        raise ValueError(f"v1 不支持的 stage_id: {stage_id}")
    return f"{int(m.group(1))}-{int(m.group(2))}"


def dataset_record_to_job_entry(rec: dict[str, Any]) -> dict[str, Any]:
    """数据集 copilot.jsonl 单条 → 内嵌作业条目（离线可回放）。

    返回 ``{"source": {...来源署名}, "maa_job": {...MAA 作业形态}}``；
    来源元数据（原始作业 id / 标题 / 评分等）原样保留，
    编队要求置零（不强制练度，原始 req 留在 source 供溯源）。
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
                "requirements": {
                    "elite": 0,
                    "level": 0,
                    "skill_level": 1,
                    "module": 0,
                    "potentiality": 0,
                },
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
    """读取 jobs 定义文件（list 条目需含 source + maa_job）。"""
    with open(path, encoding="utf-8") as f:
        jobs = json.load(f)
    if isinstance(jobs, dict) and "jobs" in jobs:
        jobs = jobs["jobs"]
    if not isinstance(jobs, list):
        raise ValueError(f"jobs 文件结构非法: {path}")
    for j in jobs:
        if "source" not in j or "maa_job" not in j:
            raise ValueError(f"jobs 条目缺 source/maa_job: {j.get('source', {})}")
    return jobs


def merge_executed_actions(
    job_actions: list[dict[str, Any]], action_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """作业动作与实机 CopilotAction 事件按 name+次序合并。

    每条作业动作附带 ``executed: {t_rel_ms, wall_ms}``（未执行到则为
    ``None``）；多余回调事件不丢（由调用方另存 extra）。
    """
    pool = list(action_events)  # 依时间顺序的回调动作事件
    merged: list[dict[str, Any]] = []
    for act in job_actions:
        item = dict(act)
        item["executed"] = None
        if act.get("type") in ("Deploy", "Skill", "Retreat") and act.get("name"):
            for i, evt in enumerate(pool):
                if evt.get("target") == act["name"]:
                    item["executed"] = {
                        "t_rel_ms": evt.get("t_rel_ms"),
                        "wall_ms": evt.get("wall_ms"),
                        "action": evt.get("action"),
                    }
                    pool.pop(i)
                    break
        merged.append(item)
    return merged


def _extract_copilot_actions(events: list[MaaEvent], chain_start_wall: int) -> list[dict[str, Any]]:
    """从事件流提取 CopilotAction 动作事件（带相对/墙钟时间戳）。"""
    out = []
    for evt in events:
        if evt.msg != "SubTaskExtraInfo":
            continue
        d = evt.details
        if d.get("what") != "CopilotAction":
            continue
        detail = d.get("details", {})
        out.append(
            {
                "t_rel_ms": evt.t_wall_ms - chain_start_wall,
                "wall_ms": evt.t_wall_ms,
                "action": detail.get("action"),
                "target": detail.get("target"),
                "elapsed_time": detail.get("elapsed_time"),
            }
        )
    return out


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
        nav_cfg = Navigator.load_nav_config(nav_cfg_path)
        templates_dir = Path(nav_cfg_path).parent / nav_cfg.get(
            "templates_dir", "nav_templates"
        )
        self._nav = Navigator(self._driver, nav_cfg, ScreenBank.load(templates_dir))
        self._writer = EpisodeWriter(self._work)
        self._crops: dict[str, list[int]] = nav_cfg.get("battle_crops", {})

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
        run_dir = self._writer.begin(stage_id, job_id)
        print(f"[job {job_id}] {stage_id} ({stage_code}) -> {run_dir}", flush=True)
        summary = {"job_id": job_id, "stage_code": stage_code, "ok": False}

        try:
            self._nav.goto_stage_briefing(stage_code)
        except NavigationError as exc:
            print(f"[job {job_id}] 导航失败: {exc}", flush=True)
            summary["error"] = f"nav: {exc}"
            (run_dir / "error.txt").write_text(str(exc), encoding="utf-8")
            return summary

        job_path = self._write_maa_job(entry)
        tid = self._driver.append_copilot(job_path, formation=True)
        if tid <= 0:
            print(f"[job {job_id}] append_copilot 被拒绝 (tid={tid})", flush=True)
            summary["error"] = f"append tid={tid}"
            (run_dir / "error.txt").write_text(
                f"append_copilot tid={tid}", encoding="utf-8"
            )
            return summary

        logger = StateLogger(self._driver, run_dir, interval_s=1.5, crops=self._crops)
        mark = self._driver.mark()
        logger.start()
        logger.extra_shot("briefing")
        if not self._driver.start():
            summary["error"] = "asst.start 失败"
            logger.stop()
            return summary

        # 作战过程：轮询事件流，CopilotAction 触发补拍
        chain_start_wall = int(time.time() * 1000)
        seen_copilot = 0
        deadline = time.monotonic() + _CHAIN_TIMEOUT_S
        cursor = mark
        terminal: str | None = None
        battle_started = False
        while time.monotonic() < deadline:
            fresh, cursor = self._driver.events_since(cursor)
            for evt in fresh:
                if evt.msg == "SubTaskStart" and "BattleProcess" in str(
                    evt.details.get("class", "")
                ):
                    if not battle_started:
                        battle_started = True
                        logger.extra_shot("battle_start")
                if (
                    evt.msg == "SubTaskExtraInfo"
                    and evt.details.get("what") == "CopilotAction"
                ):
                    seen_copilot += 1
                    target = evt.details.get("details", {}).get("target", "")
                    logger.extra_shot(f"action_{seen_copilot}_{target}")
                if evt.msg in ("TaskChainCompleted", "TaskChainError"):
                    terminal = evt.msg
            if terminal:
                break
            time.sleep(0.2)
        result_chain = self._driver.wait_chain(10.0, mark=mark)
        if terminal is None:
            terminal = result_chain.terminal_msg
        print(
            f"[job {job_id}] 任务链收尾={terminal} 动作事件={seen_copilot}",
            flush=True,
        )

        # 作战自然结束：结算画面判定（任务链收尾≠作战结束）
        win, settle_img = self._nav.detect_result()
        if settle_img is not None:
            settle_img.save(run_dir / "shots" / "settle.png")
        logger.extra_shot("battle_end")
        logger.stop()
        all_events, _ = self._driver.events_since(mark)
        self._nav.advance_after_battle()

        action_events = _extract_copilot_actions(result_chain.events or all_events, chain_start_wall)
        events_payload = [
            {"t_wall_ms": e.t_wall_ms, "t_mono_ms": e.t_mono_ms, "msg": e.msg,
             "details": e.details}
            for e in all_events
        ]
        self._writer.write_events(run_dir, events_payload)
        squad = [op["name"] for op in entry["maa_job"].get("opers", [])]
        episode = {
            "episode_id": run_dir.name,
            "stage_id": stage_id,
            "stage_code": stage_code,
            "source": src,
            "squad": squad,
            "actions": merge_executed_actions(
                entry["maa_job"].get("actions", []), action_events
            ),
            "result": {
                "win": win,
                "settled_screen": "shots/settle.png" if settle_img is not None else None,
            },
            "frames_index": "frames_index.jsonl",
            "shots_manifest": None,
        }
        ep_path = self._writer.write_episode(run_dir, episode)
        summary.update(
            {
                "ok": terminal == "TaskChainCompleted",
                "win": win,
                "episode": str(ep_path),
                "copilot_actions": seen_copilot,
            }
        )
        print(
            f"[job {job_id}] win={win} episode={ep_path.name}", flush=True
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
        marker = f"{entry['source']['job_id']}-"
        return ep_dir.is_dir() and any(
            p.name.startswith(marker) for p in ep_dir.iterdir()
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    """工厂 CLI：``python -m bridge.replay_controller --work <数据根>``。"""
    root = _repo_root()
    parser = argparse.ArgumentParser(description="ark-autoplay 重放数据工厂 v1")
    parser.add_argument(
        "--jobs", default=str(root / "configs" / "jobs_main_v1.json"), help="作业队列"
    )
    parser.add_argument(
        "--instance",
        default=str(root / "configs" / "instances" / "mumu_local.yaml"),
        help="实例定义",
    )
    parser.add_argument(
        "--nav", default=str(root / "configs" / "nav_main.yaml"), help="导航几何表"
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
