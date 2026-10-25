"""采集端转换：不依赖视频/训练环境，严守原件、对齐与可搬家边界。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bridge.convert_episodes import convert, digest, read_json


def _rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _raw(root: Path, *, result: str = "win") -> Path:
    ep = root / "episodes/synthetic_01/job-11-first"
    ep.mkdir(parents=True)
    job = root / "tmp/maa_jobs/job_11.json"
    job.parent.mkdir(parents=True)
    job.write_text(
        json.dumps(
            {
                "stage_name": "SYN-1",
                "opers": [{"name": "synthetic_op", "skill": 1}],
                "actions": [
                    {"type": "SpeedUp"},
                    {"type": "SkillDaemon", "name": "synthetic_op"},
                    {
                        "type": "Deploy",
                        "name": "synthetic_op",
                        "location": [2, 1],
                        "direction": "Right",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (ep / "episode.json").write_text(
        json.dumps(
            {
                "episode_id": "synthetic-episode-11",
                "stage_id": "synthetic_01",
                "stage_code": "SYN-1",
                "battle_result": result,
                "source": {"type": "prts.plus", "job_id": 11, "author": "synthetic"},
                "job_file": "Z:/other-machine/job_11.json",
                "result_screen_files": ["shots/result.png"],
            }
        ),
        encoding="utf-8",
    )
    _rows(
        ep / "battle_states.jsonl",
        [
            {"t_ms": 10, "costs": 20},
            {"t_ms": 50, "costs": 21},
        ],
    )
    _rows(
        ep / "action_events.jsonl",
        [
            {
                "t_ms": 0,
                "details": {
                    "what": "CopilotAction",
                    "details": {"action": "SwitchSpeed", "target": ""},
                },
            },
            {
                "t_ms": 10,
                "details": {
                    "what": "CopilotAction",
                    "details": {"action": "Deploy", "target": "synthetic_op"},
                },
            },
        ],
    )
    return ep


def test_portable_export_without_images_keeps_originals_and_hashes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    _raw(root)
    original = {p.relative_to(root): digest(p) for p in root.rglob("*") if p.is_file()}
    out = tmp_path / "converted"
    report = convert(root, out)
    assert report["counts"] == {
        "scanned": 1,
        "accepted_episodes": 1,
        "rejected_episodes": 0,
        "samples": 2,
    }
    assert original == {p: digest(root / p) for p in original}
    samples = [json.loads(s) for s in (out / "samples.jsonl").read_text().splitlines()]
    assert samples[1]["action"]["location"] == [2, 1]  # 不交换 x/y。
    assert samples[0]["action_event_t_ms"] < samples[0]["state_t_ms"]
    bundle = next((out / "episodes").glob("*/*"))
    assert bundle.name == "job-11-first"
    assert read_json(bundle / "episode.json")["job_file"] == "job.json"
    assert (bundle / "job.json").is_file()
    manifest = read_json(out / "MANIFEST.json")
    assert not manifest["split_assigned"] and not manifest["vocabulary_built"]
    assert all(digest(out / name) == value for name, value in manifest["files"].items())
    assert not list(out.rglob("*.png")) and not list(out.rglob("*.mp4"))
    second = tmp_path / "converted-again"
    convert(root, second)
    assert (out / "samples.jsonl").read_bytes() == (
        second / "samples.jsonl"
    ).read_bytes()


@pytest.mark.parametrize("result", ["lost", "unknown", ""])
def test_non_wins_are_audited_not_promoted(tmp_path: Path, result: str) -> None:
    root = tmp_path / "raw"
    _raw(root, result=result)
    report = convert(root, tmp_path / "out")
    assert report["counts"]["samples"] == 0
    assert report["rejected"][0]["reason"] == "result_not_explicit_win"


@pytest.mark.parametrize("bad", ["count", "type", "target", "timestamp"])
def test_alignment_errors_do_not_make_silent_training_pairs(
    tmp_path: Path, bad: str
) -> None:
    root = tmp_path / "raw"
    ep = _raw(root)
    if bad == "count":
        _rows(ep / "battle_states.jsonl", [{"t_ms": 10}])
    elif bad == "timestamp":
        _rows(ep / "battle_states.jsonl", [{"t_ms": 50}, {"t_ms": 10}])
    else:
        events = [
            json.loads(s) for s in (ep / "action_events.jsonl").read_text().splitlines()
        ]
        events[1]["details"]["details"]["action" if bad == "type" else "target"] = (
            "wrong"
        )
        _rows(ep / "action_events.jsonl", events)
    report = convert(root, tmp_path / "out")
    assert report["counts"]["accepted_episodes"] == 0
    assert report["counts"]["rejected_episodes"] == 1


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        convert(tmp_path / "raw", out)
    assert marker.read_text() == "keep"


def test_declared_job_cannot_read_outside_the_data_root(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    ep = _raw(root)
    job = root / "tmp/maa_jobs/job_11.json"
    outside = tmp_path / "outside.json"
    job.rename(outside)
    meta = read_json(ep / "episode.json")
    meta["job_file"] = str(outside)
    (ep / "episode.json").write_text(json.dumps(meta), encoding="utf-8")
    report = convert(root, tmp_path / "out")
    assert report["rejected"][0]["reason"] == "job_json_missing_inside_data_root"


def test_write_error_never_publishes_a_partial_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bridge import convert_episodes

    root = tmp_path / "raw"
    _raw(root)

    def fail(*_: object) -> None:
        raise OSError("synthetic disk error")

    monkeypatch.setattr(convert_episodes.shutil, "copyfile", fail)
    out = tmp_path / "out"
    with pytest.raises(OSError):
        convert(root, out)
    assert not out.exists()
