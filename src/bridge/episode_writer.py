"""通关轨迹写出模块。

职责（对应设计文档 §5.4 / §6.3）：一局作战结束后，把 episode 束写出到
``<out_root>/episodes/<stage_id>/<job_id>-<时间戳>/``：

- ``episode.json``：协议化轨迹头（schema ``0.1.0-draft``），含 episode_id、
  protocol_version、关卡、作业来源署名（prts.plus 作业 id / 标题 / 评分）、
  阵容、动作表（作业动作与实机 CopilotAction 事件按 name+次序合并）、
  结果与结算画面、frames_index 路径、截图 sha256 清单；
- ``action_events.jsonl``：整局回调事件流（墙钟时间戳）；
- ``shots/``：截图本体与费用/击杀裁剪（**本机留存，路径只进清单**，
  清单与轨迹允许入库，截图本体永不入仓——素材红线 §3.2/§6.3）。

写入前强制戳 ``PROTOCOL_VERSION`` 并校验必填字段与截图哈希齐全。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

PROTOCOL_VERSION: str = "0.1.0-draft"
"""episode schema 版本戳（与 data-validation 的 v0 轨迹对齐；待 core 协议库
落地后切换为 semver 对齐，见设计文档 §6.3）。"""

REQUIRED_EPISODE_FIELDS: tuple[str, ...] = (
    "episode_id",
    "stage_id",
    "stage_code",
    "source",
    "squad",
    "actions",
    "result",
    "frames_index",
    "shots_manifest",
)
"""episode.json 必填字段（写入自检）。"""


def sha256_file(path: str | Path) -> str:
    """计算文件 sha256（分块读取，适配大截图）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class EpisodeWriter:
    """轨迹写出器：每局一个 episode 目录。"""

    def __init__(self, out_root: str | Path) -> None:
        """``out_root`` 为数据落盘根（机器本地路径，由调用方从运行参数传入）。"""
        self._root = Path(out_root)

    def begin(self, stage_id: str, job_id: int | str) -> Path:
        """创建一局 episode 目录（含 shots/），返回路径。"""
        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = self._root / "episodes" / str(stage_id) / f"{job_id}-{ts}"
        (run_dir / "shots").mkdir(parents=True, exist_ok=False)
        return run_dir

    def write_events(self, run_dir: str | Path, events: list[dict[str, Any]]) -> Path:
        """把回调事件流写为 ``action_events.jsonl``，返回路径。"""
        path = Path(run_dir) / "action_events.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for evt in events:
                f.write(json.dumps(evt, ensure_ascii=False) + "\n")
        return path

    def build_shots_manifest(self, run_dir: str | Path) -> list[dict[str, Any]]:
        """遍历 shots/ 生成截图清单（相对 run_dir 路径 + sha256 + 字节数）。"""
        shots_dir = Path(run_dir) / "shots"
        manifest: list[dict[str, Any]] = []
        for p in sorted(shots_dir.rglob("*")):
            if p.is_file():
                manifest.append(
                    {
                        "file": str(p.relative_to(run_dir)).replace("\\", "/"),
                        "sha256": sha256_file(p),
                        "bytes": p.stat().st_size,
                    }
                )
        return manifest

    def validate(self, episode: dict[str, Any]) -> None:
        """校验必填字段与哈希齐全；不合法抛 ``ValueError``。"""
        missing = [k for k in REQUIRED_EPISODE_FIELDS if k not in episode]
        if missing:
            raise ValueError(f"episode 缺必填字段: {missing}")
        if episode.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("protocol_version 未戳或与当前版本不符")
        for item in episode.get("shots_manifest", []):
            if not item.get("sha256") or not item.get("file"):
                raise ValueError("shots_manifest 存在缺哈希/缺路径条目")

    def write_episode(self, run_dir: str | Path, episode: dict[str, Any]) -> Path:
        """戳协议版本、自检后写出 ``episode.json``，返回路径。"""
        run_dir = Path(run_dir)
        episode = dict(episode)
        episode["protocol_version"] = PROTOCOL_VERSION
        if "shots_manifest" not in episode or episode["shots_manifest"] is None:
            episode["shots_manifest"] = self.build_shots_manifest(run_dir)
        self.validate(episode)
        path = run_dir / "episode.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(episode, f, ensure_ascii=False, indent=1)
        return path
