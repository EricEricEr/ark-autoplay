"""通关轨迹写出模块。

职责（设计文档 §5.4 / §6.3）：一局作战结束后，把 episode 束写出到
``<out_root>/episodes/<stage_id>/job-<作业id>-<时间戳>/``，
bundle 结构与 ``data-validation/capture_episode.py`` 已验证格式一致：

- ``episode.json``：轨迹清单（``protocol_version=0.1.0-draft``），含
  episode_id / 关卡 / 作业来源 / 作业文件 / 结算帧列表 /
  action_events.jsonl 路径 / 截图 sha256 清单 / 时长；
- ``action_events.jsonl``：回调事件流 + 截图标记（相对毫秒时间戳）；
- ``shots/``：节拍帧 + 动作触发帧 + 结算帧（**本机留存**，
  入仓内容仅限 sha256 清单——素材红线 §3.2/§6.3）。

写入前强制戳 ``PROTOCOL_VERSION`` 并自检必填字段。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

PROTOCOL_VERSION: str = "0.1.0-draft"
"""episode schema 版本戳（与 capture_episode.py 产出对齐）。"""

REQUIRED_EPISODE_FIELDS: tuple[str, ...] = (
    "episode_id",
    "stage_id",
    "stage_code",
    "source",
    "job_file",
    "result_screen_files",
    "action_events_file",
    "shot_files",
    "duration_ms",
)
"""episode.json 必填字段（写入自检，与已验证 bundle 保持一致）。"""


def sha256_tag(path: str | Path, prefix: int = 16) -> str:
    """文件 sha256 短戳：``sha256:<前16位hex>``（与已验证清单格式一致）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()[:prefix]}"


class EpisodeWriter:
    """轨迹写出器：每局一个 episode 目录。"""

    def __init__(self, out_root: str | Path) -> None:
        """``out_root`` 为数据落盘根（机器本地路径，运行参数传入）。"""
        self._root = Path(out_root)

    def begin(self, stage_id: str, job_id: int | str) -> Path:
        """创建一局 episode 目录（含 shots/），返回路径。"""
        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = self._root / "episodes" / str(stage_id) / f"job-{job_id}-{ts}"
        (run_dir / "shots").mkdir(parents=True, exist_ok=False)
        return run_dir

    def write_events(self, run_dir: str | Path, records: list[dict[str, Any]]) -> Path:
        """写 ``action_events.jsonl``（每行一条带 t_ms 的记录）。"""
        path = Path(run_dir) / "action_events.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path

    def write_battle_states(
        self, run_dir: str | Path, records: list[dict[str, Any]]
    ) -> Path | None:
        """写 ``battle_states.jsonl``（每行一条战场状态，与动作同时间轴）。

        由 MaaCore 的 ``BattleState`` 回调转成（见 ``battle_state`` 模块）。
        **无记录时返回 None 且不创建文件**——让"这局没采到状态"与"采到空序列"
        在文件系统层面就可区分（前者查不到文件，后者是空文件）。
        """
        if not records:
            return None
        path = Path(run_dir) / "battle_states.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path

    def build_shot_manifest(self, run_dir: str | Path) -> dict[str, str]:
        """遍历 shots/ 生成 ``{相对路径: sha256短戳}`` 清单。"""
        shots_dir = Path(run_dir) / "shots"
        manifest: dict[str, str] = {}
        for p in sorted(shots_dir.iterdir()):
            if p.is_file():
                rel = f"shots/{p.name}"
                manifest[rel] = sha256_tag(p)
        return manifest

    def validate(self, episode: dict[str, Any]) -> None:
        """校验必填字段；不合法抛 ``ValueError``。"""
        missing = [k for k in REQUIRED_EPISODE_FIELDS if k not in episode]
        if missing:
            raise ValueError(f"episode 缺必填字段: {missing}")
        if episode.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("protocol_version 未戳或与当前版本不符")
        if not episode.get("shot_files"):
            raise ValueError("shot_files 为空（一局至少应有节拍/结算帧）")

    def write_episode(self, run_dir: str | Path, episode: dict[str, Any]) -> Path:
        """戳协议版本、补齐截图清单、自检后写 ``episode.json``。"""
        run_dir = Path(run_dir)
        episode = dict(episode)
        episode["protocol_version"] = PROTOCOL_VERSION
        if not episode.get("shot_files"):
            episode["shot_files"] = self.build_shot_manifest(run_dir)
        self.validate(episode)
        path = run_dir / "episode.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(episode, f, ensure_ascii=False, indent=2)
        return path
