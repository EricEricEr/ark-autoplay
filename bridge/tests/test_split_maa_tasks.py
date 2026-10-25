"""活动能力判断必须读取拆分任务表，不能把原生支持的活动误判为不可导航。"""

from pathlib import Path

from bridge.maa_stages import fight_accepts, load_maa_tasks


def test_split_activity_tasks_are_included(tmp_path: Path) -> None:
    tasks = tmp_path / "resource/tasks"
    (tasks / "Stages").mkdir(parents=True)
    (tasks / "tasks.json").write_text('{"Episode0": {}}', encoding="utf-8")
    (tasks / "Stages/GT.json").write_text('{"GT-5": {}}', encoding="utf-8")
    names = load_maa_tasks(tmp_path)
    assert "Episode0" in names
    assert fight_accepts("GT-5", names)
    assert not fight_accepts("GT-1", names)
