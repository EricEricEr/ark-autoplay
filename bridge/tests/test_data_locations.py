"""真实资料选择与非C盘校验回归；失败必须发生在写文件或连接游戏前。"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bridge import maa_driver, replay_controller
from bridge.collector import app, exporting
from bridge.collector.service import CollectorService
from bridge.collector.storage import app_home
from bridge.convert_episodes import convert
from bridge.data_paths import data_path
from bridge.nav_maa import _tap_camp_entry


@pytest.mark.parametrize(
    "value", ["", " ", "C:/synthetic-data", "c:\\synthetic-data", "C:relative",
              "\\\\?\\C:\\synthetic-data", "\\\\.\\C:\\synthetic-data"]
)
def test_invalid_data_locations_are_rejected_without_writing(value, tmp_path):
    before = list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="C盘"):
        data_path(value)
    assert list(tmp_path.iterdir()) == before


def test_resolved_junction_target_on_c_is_rejected(monkeypatch):
    monkeypatch.setattr(Path, "resolve", lambda self: Path("C:/synthetic-target"))
    with pytest.raises(ValueError, match="C盘"):
        data_path("Z:/synthetic-alias")


def test_metadata_defaults_remain_and_real_data_has_no_implicit_location(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LOCALAPPDATA", "C:/SyntheticAppData")
    assert app_home().as_posix() == "C:/SyntheticAppData/ArkCollector"
    service = CollectorService(tmp_path / "profile")
    assert service.settings.data_dir == ""
    with pytest.raises(ValueError, match="选择非C盘"):
        service.import_tasks(tmp_path / "not-read.json")
    with pytest.raises(ValueError, match="选择非C盘"):
        service.start(conversion_only=True)
    assert not (service.home / "data").exists()


def test_user_choice_is_persisted_and_switch_preserves_old_files(tmp_path):
    service = CollectorService(tmp_path / "profile")
    old = tmp_path / "old-data"
    old.mkdir()
    marker = old / "keep.json"
    marker.write_text("{}", encoding="utf-8")
    service.settings.data_dir = str(old)
    service.settings.queue_file = str(old / "queue.json")
    service.settings.last_run_dir = str(old / "run")
    selected = tmp_path / "自选 数据目录"
    service.set_data_directory(selected)
    assert service.data_root() == selected.resolve()
    assert not service.settings.queue_file and not service.settings.last_run_dir
    assert marker.read_text(encoding="utf-8") == "{}"
    restored = CollectorService(service.home)
    assert restored.data_root() == selected.resolve()
    assert not selected.exists()


def test_gui_rejects_c_choice_without_replacing_saved_location(tmp_path, monkeypatch):
    service = CollectorService(tmp_path / "profile")
    service.set_data_directory(tmp_path / "chosen")
    errors = []
    window = SimpleNamespace(service=service, error=errors.append)
    monkeypatch.setattr(
        app.filedialog, "askdirectory", lambda **kwargs: "C:/synthetic-data"
    )
    app.CollectorApp.choose_folder(window, "data_dir")
    assert len(errors) == 1 and "C盘" in str(errors[0])
    assert service.data_root() == (tmp_path / "chosen").resolve()


def test_reselecting_same_data_directory_keeps_current_tasks(tmp_path):
    service = CollectorService(tmp_path / "profile")
    chosen = tmp_path / "chosen"
    service.settings.data_dir = (chosen / "..").as_posix() + "/chosen"
    service.settings.queue_file = str(chosen / "queue.json")
    service.settings.last_run_dir = str(chosen / "run")
    service.set_data_directory(chosen)
    assert service.settings.queue_file == str(chosen / "queue.json")
    assert service.settings.last_run_dir == str(chosen / "run")


def test_conversion_and_export_reject_c_before_reading_sources(tmp_path):
    with pytest.raises(ValueError, match="C盘"):
        convert(tmp_path / "not-read", Path("C:/synthetic-output"))
    with pytest.raises(ValueError, match="C盘"):
        exporting.export_data(tmp_path / "not-read", Path("C:/synthetic.zip"))


def test_diagnostics_export_keeps_system_location_exception(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("诊断包不应套用真实数据的磁盘限制")

    monkeypatch.setattr(exporting, "data_path", forbidden)
    exporting.export_data(tmp_path, tmp_path / "diagnostics.zip", diagnostics_only=True)
    assert (tmp_path / "diagnostics.zip").is_file()


def test_annihilation_scroll_uses_current_driver(monkeypatch):
    calls = []
    driver = SimpleNamespace(
        swipe=lambda *args: calls.append(("swipe", args)),
        tap=lambda *args: calls.append(("tap", args)),
    )
    monkeypatch.setattr("bridge.nav_maa.time.sleep", lambda _: None)
    _tap_camp_entry(driver, "camp_01")
    assert calls[:3] == [("swipe", (1466, 380, 1466, 1000, 260))] * 3
    assert calls[3] == ("tap", (1466, 657))


@pytest.mark.parametrize("module", [maa_driver, replay_controller])
def test_cli_requires_explicit_instance_before_control(
    module, monkeypatch, tmp_path, capsys
):
    arguments = [] if module is maa_driver else ["--work", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", ["synthetic", *arguments])
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 2
    assert "--instance" in capsys.readouterr().err


def test_legacy_c_data_is_readable_as_settings_but_cannot_restart(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    original = {"data_dir": "C:/synthetic-legacy-data"}
    (home / "settings.local.json").write_text(json.dumps(original), encoding="utf-8")
    service = CollectorService(home)
    with pytest.raises(ValueError, match="C盘"):
        service.start(conversion_only=True)
    saved = json.loads((home / "settings.local.json").read_text(encoding="utf-8"))
    assert saved == original
