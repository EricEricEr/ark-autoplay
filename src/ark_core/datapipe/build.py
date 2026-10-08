"""datapipe v1 构建 CLI。

用法：
    uv run python -m ark_core.datapipe.build featvec|stages|episodes|tiers|all \
        [--config PATH] [--out DIR]

- featvec：干员 / 敌人特征向量 parquet + featvec_schema.json
- stages：关卡注册表 parquet（含 family）
- episodes：copilot 作业 → episodes_v0.jsonl（含 deploy subtype / aux 计数）
- tiers：作业阵容练度 5 桶 jobs_roster_tiers.json
- all：以上全部 + 汇总 build_report.json（单独步骤也会写各自收到的部分）

路径解析全部走 configs/data.yaml + 环境变量 ARK_DATA_ROOT（见 sources.py）；
真实数据不入库（data/README.md 红线），本 CLI 只在本机静态数据齐备时可用。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .copilot_to_episode import ConvertContext, OperatorNameIndex, convert_copilot
from .deploy_vocab import DeployVocab
from .featvec import FEATVEC_VERSION, build_featvec
from .jobs_tiers import TIER_ORDER, build_tiers
from .sources import (
    DataConfig,
    ResolvedSources,
    SourceError,
    check_sources,
    load_config,
    out_path,
    resolve_sources,
)
from .stages_meta import build_registry, family_of, load_stage_index

_STEPS = ("featvec", "stages", "episodes", "tiers")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """命令行参数解析。"""
    parser = argparse.ArgumentParser(
        prog="python -m ark_core.datapipe.build",
        description="datapipe v1 构建（featvec / stages / episodes / tiers / all）",
    )
    parser.add_argument("step", choices=[*_STEPS, "all"], help="要执行的构建步骤")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="数据源配置（默认 <repo>/configs/data.yaml）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="产物输出目录（默认取配置的 outputs.dir，相对数据根）",
    )
    return parser.parse_args(argv)


def _run_featvec(cfg: DataConfig, resolved: ResolvedSources) -> dict[str, Any]:
    res = build_featvec(cfg, resolved, resolved.out_dir)
    return {
        "featvec_version": FEATVEC_VERSION,
        "n_operators": res.n_operators,
        "n_enemy_rows": res.n_enemy_rows,
        "n_op_vector_columns": len(res.op_vector_columns),
        "n_enemy_vector_columns": len(res.enemy_vector_columns),
        "outputs": [str(res.op_out), str(res.enemy_out), str(res.schema_out)],
    }


def _run_stages(resolved: ResolvedSources) -> dict[str, Any]:
    out = out_path(resolved, "stages_registry_parquet")
    res = build_registry(resolved.require("stages_jsonl"), out)
    return {"n_stages": res.n_stages, "n_families": res.n_families, "outputs": [str(res.out_path)]}


def _run_episodes(cfg: DataConfig, resolved: ResolvedSources) -> dict[str, Any]:
    ctx = ConvertContext(
        op_index=OperatorNameIndex(resolved.require("operators_jsonl")),
        vocab=DeployVocab.load(),
        stage_index=load_stage_index(resolved.require("stages_jsonl")),
        data_version=cfg.data_version,
    )
    out = out_path(resolved, "episodes_jsonl")
    drop = bool(cfg.episodes.get("drop_when_all_actions_skipped", True))
    res = convert_copilot(resolved.require("copilot_jsonl"), ctx, out, drop_when_all_skipped=drop)
    topk = int(cfg.episodes.get("unknown_names_topk_in_report", 100))
    return {
        "n_jobs": res.n_jobs,
        "n_kept": res.n_kept,
        "n_dropped": res.n_dropped,
        "drop_reasons": res.drop_reasons,
        "action_hist_kept": dict(ctx.action_hist_kept.most_common()),
        "action_hist_aux_skipped": dict(ctx.action_hist_aux.most_common()),
        "deploy_subtypes": dict(ctx.deploy_subtypes.most_common()),
        "unknown_names_top": ctx.unknown_names.most_common(topk),
        "stage_join_missing": ctx.stage_join_missing,
        "vocab": {
            "vocab_version": ctx.vocab.vocab_version,
            "n_device_tokens": ctx.vocab.device_count,
            "n_category_terms": ctx.vocab.category_term_count,
        },
        "outputs": [str(res.out_path)],
    }


def _run_tiers(cfg: DataConfig, resolved: ResolvedSources) -> dict[str, Any]:
    out = out_path(resolved, "jobs_roster_tiers_json")
    # tiers 与 episodes 共用同一家族口径：按作业 stage_id 直接取 family_of
    res = build_tiers(
        copilot_jsonl=resolved.require("copilot_jsonl"),
        operators_jsonl=resolved.require("operators_jsonl"),
        family_of_job=lambda job: family_of(str(job.get("stage_id") or "")),
        out=out,
        data_version=cfg.data_version,
    )
    return {
        "tier_order": list(TIER_ORDER),
        "counts": res.counts,
        "n_families": res.n_families,
        "outputs": [str(res.out_path)],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 入口；返回进程退出码。"""
    args = _parse_args(argv)
    t0 = time.monotonic()
    try:
        cfg = load_config(args.config)
        resolved = resolve_sources(cfg, out_override=args.out)
        check_sources(resolved)
    except SourceError as e:
        print(f"[datapipe] 数据源错误：\n{e}", file=sys.stderr)
        return 2

    resolved.out_dir.mkdir(parents=True, exist_ok=True)
    steps = list(_STEPS) if args.step == "all" else [args.step]
    report: dict[str, Any] = {
        "data_version": cfg.data_version,
        "config_path": str(cfg.config_path),
        "sources": {k: str(p) for k, p in resolved.paths.items()},
        "sources_missing_optional": [k for k in resolved.missing],
        "out_dir": str(resolved.out_dir),
        "sections": {},
    }
    runners = {
        "featvec": lambda: _run_featvec(cfg, resolved),
        "stages": lambda: _run_stages(resolved),
        "episodes": lambda: _run_episodes(cfg, resolved),
        "tiers": lambda: _run_tiers(cfg, resolved),
    }
    for step in steps:
        sec_t0 = time.monotonic()
        section = runners[step]()
        section["elapsed_sec"] = round(time.monotonic() - sec_t0, 3)
        report["sections"][step] = section
        print(f"[datapipe] {step}: 完成（{section['elapsed_sec']}s）")
    report["elapsed_sec"] = round(time.monotonic() - t0, 3)

    report_out = out_path(resolved, "build_report_json")
    with report_out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[datapipe] 报告：{report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
