"""静态数据产物的显式位置、非C盘校验及逐文件绕过回归。"""

from dataclasses import replace
from pathlib import Path

import pytest

from ark_core.datapipe.sources import (
    SOURCE_KEYS,
    DataConfig,
    SourceError,
    out_path,
    resolve_sources,
)


def configuration(tmp_path: Path, *, root: Path | None = None) -> DataConfig:
    return DataConfig(
        config_path=tmp_path / "synthetic.yaml", data_version="synthetic", data_root=root,
        sources={key: str(tmp_path / (key + ".json")) for key in SOURCE_KEYS},
        optional_sources=frozenset(), outputs={"dir": "processed"}, featvec={}, episodes={}, raw={},
    )


def test_missing_output_location_cannot_fall_back_to_repository(tmp_path):
    with pytest.raises(SourceError, match="未指定数据输出位置"):
        resolve_sources(configuration(tmp_path))
    assert not (tmp_path / "processed").exists()


def test_explicit_root_or_output_is_respected_without_creating_files(tmp_path, monkeypatch):
    cfg = configuration(tmp_path)
    assert resolve_sources(cfg, tmp_path / "chosen").out_dir == tmp_path / "chosen"
    assert resolve_sources(replace(cfg, data_root=tmp_path)).out_dir == tmp_path / "processed"
    cfg_absolute = replace(cfg, outputs={"dir": str(tmp_path / "absolute-output")})
    assert resolve_sources(cfg_absolute).out_dir == tmp_path / "absolute-output"
    monkeypatch.chdir(tmp_path)
    assert resolve_sources(cfg, Path("chosen-relative")).out_dir == tmp_path / "chosen-relative"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("path", ["C:/synthetic", "c:\\synthetic", "\\\\?\\C:\\synthetic"])
def test_c_output_is_rejected_even_with_valid_input_root(tmp_path, path):
    with pytest.raises(SourceError, match="C盘"):
        resolve_sources(configuration(tmp_path, root=tmp_path), Path(path))


def test_per_file_absolute_c_override_is_rejected(tmp_path):
    cfg = configuration(tmp_path, root=tmp_path)
    cfg = replace(cfg, outputs={"dir": "processed", "one": "C:/synthetic-item.json"})
    resolved = resolve_sources(cfg)
    with pytest.raises(SourceError, match="C盘"):
        out_path(resolved, "one")


def test_old_c_input_root_is_allowed_with_explicit_non_c_output(tmp_path):
    cfg = configuration(tmp_path, root=Path("C:/synthetic-old-inputs"))
    assert resolve_sources(cfg, tmp_path / "chosen").out_dir == tmp_path / "chosen"
