"""数据源路径解析（configs/data.yaml + 环境变量 ARK_DATA_ROOT）。

职责：
- 加载 configs/data.yaml（或 CLI 指定的配置文件），解析 data_root；
- 把 sources.* 里的相对路径拼到数据根上，绝对路径原样采用（支持 ~ 与环境变量展开）；
- 源文件缺失时抛出带完整缺失清单与修复指引的 SourceError。

红线：代码内不得出现任何机器本地路径；真实路径只存在于配置文件与环境变量中。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# 环境变量名：数据根（相对路径解析的基准）
DATA_ROOT_ENV = "ARK_DATA_ROOT"

# 仓库内默认配置文件：configs/data.yaml（src/ark_core/datapipe/ → 上三级为仓库根）
DEFAULT_CONFIG_REL = Path("configs") / "data.yaml"

# 必须存在的源 key（与 configs/data.yaml 的 sources 小节一一对应）
SOURCE_KEYS = (
    "copilot_jsonl",
    "stages_jsonl",
    "operators_jsonl",
    "character_table",
    "skill_table",
    "battle_equip_table",
    "uniequip_table",
    "enemy_handbook_table",
    "enemy_database",
)


class SourceError(RuntimeError):
    """数据源配置或文件缺失错误（含修复指引）。"""


@dataclass(frozen=True)
class DataConfig:
    """configs/data.yaml 的解析结果。"""

    config_path: Path
    data_version: str
    data_root: Path | None
    sources: dict[str, str]
    optional_sources: frozenset[str]
    outputs: dict[str, str]
    featvec: dict[str, Any]
    episodes: dict[str, Any]
    raw: dict[str, Any] = field(repr=False)


def _repo_root() -> Path:
    """仓库根目录（src 布局：本文件上三级）。"""
    return Path(__file__).resolve().parents[3]


def default_config_path() -> Path:
    """默认配置路径：<repo>/configs/data.yaml。"""
    return _repo_root() / DEFAULT_CONFIG_REL


def load_config(config_path: Path | None = None) -> DataConfig:
    """读取并校验 data.yaml；文件不存在或关键字段缺失时报 SourceError。"""
    path = (config_path or default_config_path()).expanduser().resolve()
    if not path.is_file():
        raise SourceError(
            f"找不到数据源配置：{path}\n请检查仓库 configs/data.yaml 是否存在，或用 --config 指定。"
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise SourceError(f"数据源配置不是 yaml 对象：{path}")

    data_version = str(raw.get("data_version") or "").strip()
    if not data_version:
        raise SourceError(f"{path} 缺少 data_version。")

    root_raw = raw.get("data_root")
    if root_raw is None:
        env_root = os.environ.get(DATA_ROOT_ENV, "").strip()
        data_root = Path(_expand(env_root)).resolve() if env_root else None
    else:
        data_root = Path(_expand(str(root_raw))).resolve()

    sources_raw = raw.get("sources") or {}
    if not isinstance(sources_raw, dict):
        raise SourceError(f"{path} 的 sources 小节必须是 mapping。")
    sources = {str(k): str(v) for k, v in sources_raw.items()}
    missing_keys = [k for k in SOURCE_KEYS if k not in sources]
    if missing_keys:
        raise SourceError(f"{path} 的 sources 缺少条目：{', '.join(missing_keys)}")

    optional = frozenset(str(k) for k, v in (raw.get("optional_sources") or {}).items() if v)
    outputs = {str(k): str(v) for k, v in (raw.get("outputs") or {}).items()}
    featvec = dict(raw.get("featvec") or {})
    episodes = dict(raw.get("episodes") or {})
    return DataConfig(
        config_path=path,
        data_version=data_version,
        data_root=data_root,
        sources=sources,
        optional_sources=optional,
        outputs=outputs,
        featvec=featvec,
        episodes=episodes,
        raw=raw,
    )


def _expand(p: str) -> str:
    """展开 ~ 与环境变量（Windows 也支持 %VAR%）。"""
    return os.path.expanduser(os.path.expandvars(p))


def _resolve_one(rel_or_abs: str, root: Path | None) -> Path:
    """单条路径解析：绝对路径直接用，相对路径接到数据根（无根时接到仓库根）。"""
    p = Path(_expand(rel_or_abs))
    if p.is_absolute():
        return p.resolve()
    base = root if root is not None else _repo_root()
    return (base / p).resolve()


@dataclass(frozen=True)
class ResolvedSources:
    """解析后的源 / 输出路径集合（全为绝对路径）。"""

    config: DataConfig
    paths: dict[str, Path]
    missing: dict[str, Path]
    out_dir: Path

    def require(self, key: str) -> Path:
        """取必须存在的源路径；缺失时报 SourceError（build 时先整体预检，见 check()）。"""
        if key in self.missing:
            raise SourceError(f"必要数据源缺失：{key} -> {self.missing[key]}")
        return self.paths[key]

    def optional(self, key: str) -> Path | None:
        """取可选源路径；缺失时返回 None。"""
        return None if key in self.missing else self.paths.get(key)


def resolve_sources(
    cfg: DataConfig,
    out_override: Path | None = None,
) -> ResolvedSources:
    """解析全部 sources 与输出目录路径，统计缺失项（不写任何文件）。

    out_override：CLI --out 传入的输出目录（绝对优先），覆盖 outputs.dir。
    """
    paths: dict[str, Path] = {}
    missing: dict[str, Path] = {}
    for key in SOURCE_KEYS:
        p = _resolve_one(cfg.sources[key], cfg.data_root)
        paths[key] = p
        if not p.is_file():
            missing[key] = p

    out_dir_raw = (
        str(out_override) if out_override is not None else cfg.outputs.get("dir", "processed")
    )
    out_dir = _resolve_one(out_dir_raw, cfg.data_root)
    return ResolvedSources(config=cfg, paths=paths, missing=missing, out_dir=out_dir)


def check_sources(resolved: ResolvedSources) -> None:
    """预检必要数据源；任一缺失时抛出列出全部缺失与修复指引的 SourceError。"""
    fatal = {k: p for k, p in resolved.missing.items() if k not in resolved.config.optional_sources}
    if not fatal:
        return
    lines = [f"  - {k}: {p}" for k, p in sorted(fatal.items())]
    raise SourceError(
        "缺失必要数据源（真实数据本就不入库，见 data/README.md）：\n"
        + "\n".join(lines)
        + "\n修复方法：1) 设置环境变量 "
        + DATA_ROOT_ENV
        + " 指向数据根目录；或 2) 编辑 configs/data.yaml 的 data_root / sources.\n"
        + "数据根布局约定见 configs/data.yaml 头部注释。"
    )


def out_path(resolved: ResolvedSources, output_key: str) -> Path:
    """输出文件路径：outputs 小节给文件名，落到 out_dir 下。"""
    name = resolved.config.outputs.get(output_key, output_key)
    p = Path(name)
    return p if p.is_absolute() else resolved.out_dir / p
