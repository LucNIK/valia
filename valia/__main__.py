# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Command line: `python -m valia fetch|build|baseline|train|all`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from . import __version__
from .baseline import SEGMENTS, evaluate
from .clean import clean, read_raw
from .config import CLEAN_DIR, RAW_DIR, REPORTS_DIR, SOURCE_NAME
from .enrich import coverage, enrich, fetch_references
from .features import add_features
from .fetch import fetch, parse_departements, parse_years, raw_path
from .storage import load_table, save_table

SALES = CLEAN_DIR / "sales"


def cmd_fetch(args) -> None:
    fetch(parse_years(args.years), parse_departements(args.departements), Path(args.raw))


def cmd_build(args) -> None:
    years, deps = parse_years(args.years), parse_departements(args.departements)
    started = time.perf_counter()
    # Communes never span two départements, so cleaning and neighbourhood features are computed
    # one département at a time: memory stays bounded even for the whole of France.
    communes, stations = (None, None) if args.no_enrich else fetch_references()
    parts, funnel, files = [], {}, 0
    for dep in deps:
        paths = [p for y in years if (p := raw_path(y, dep, Path(args.raw))).exists()]
        if not paths:
            continue
        files += len(paths)
        sales, dep_funnel = clean(read_raw(paths))
        parts.append(enrich(add_features(sales), communes, stations))
        for step, n in dep_funnel.items():
            funnel[step] = funnel.get(step, 0) + n
    if not parts:
        sys.exit("no raw files found: run `python -m valia fetch` first")
    sales = pd.concat(parts, ignore_index=True).sort_values(["date", "id_mutation"], kind="stable")
    written = save_table(sales.reset_index(drop=True), Path(args.clean) / SALES.name)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "funnel.json").write_text(json.dumps(funnel, indent=2))
    (REPORTS_DIR / "enrichment.json").write_text(json.dumps(coverage(sales), indent=2))
    print(f"[build] {files} files -> {len(sales):,} sales in {time.perf_counter() - started:.1f}s -> {written}")
    for step, n in funnel.items():
        print(f"        {step:<24} {n:>12,}")


def cmd_baseline(args) -> None:
    sales = load_table(Path(args.clean) / SALES.name)
    report = evaluate(sales, args.test_year)
    report["source"] = SOURCE_NAME
    deps = sorted(sales["dep"].unique())
    listed = ", ".join(deps[:8]) + ("…" if len(deps) > 8 else "")
    report["scope"] = "France" if len(deps) > 90 else f"Ensemble ({listed})"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "baseline.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (REPORTS_DIR / "baseline.md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))


def cmd_train(args) -> None:
    from .model import train_and_evaluate

    sales = load_table(Path(args.clean) / SALES.name)
    started = time.perf_counter()
    model, intervals, report = train_and_evaluate(sales, args.test_year, fast=args.fast)
    report["source"] = SOURCE_NAME
    report["training_seconds"] = round(time.perf_counter() - started, 1)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, intervals, REPORTS_DIR)
    (REPORTS_DIR / "model.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (REPORTS_DIR / "model.md").write_text(model_markdown(report), encoding="utf-8")
    print(model_markdown(report))


def save_model(model, intervals, folder: Path) -> None:
    """The learned gap model, plus everything needed to rebuild a prediction: anchors and intervals."""
    if model.backend == "lightgbm":
        model.regressor.booster_.save_model(str(folder / "model.lgb.txt"))
    else:
        import joblib

        joblib.dump(model.regressor, folder / "model.joblib")
    anchors = {"commune": model.anchor.commune, "dep": model.anchor.dep, "france": model.anchor.france}
    (folder / "anchors.json").write_text(json.dumps(anchors, ensure_ascii=False))
    payload = {"global_half_width_log": intervals.global_q, "by_group": intervals.by_group}
    (folder / "intervals.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def model_markdown(report: dict) -> str:
    def verdict(ok: bool) -> str:
        return "✅" if ok else "❌"

    o = report["model"]["overall"]
    lines = [f"# Model — test year {report['test_year']} ({report['backend']})", "",
             f"Train {report['train_sales']:,} · calibration {report['calibration_sales']:,} · "
             f"test {report['test_sales']:,} sales.", "",
             f"- Accuracy: MAE(log) **{o['mae_log']}** vs target {report['target_mae_log']} "
             f"({report['improvement_vs_commune_pct']} % better than the commune median) "
             f"{verdict(report['passes']['accuracy'])}",
             f"- Intervals: **{o['coverage_pct']} %** of 2025 prices inside the 80 % range "
             f"(target 78-82 %) {verdict(report['passes']['coverage'])}", "",
             "| Scope | Sales | MdAPE % | Within 10 % | MAE € | Coverage % | Range width % |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    rows = [("France", o)] + [(s, report["model"]["by_segment"][s]) for s in SEGMENTS
                              if s in report["model"]["by_segment"]]
    rows += list(report["model"]["by_type"].items())
    for label, m in rows:
        lines.append(f"| {label} | {m['sales']:,} | {m['mdape_pct']} | {m['within_10pct']} | "
                     f"{m['mae_eur']:,.0f} | {m['coverage_pct']} | {m['median_width_pct']} |")
    lines += ["", "| Reference | MdAPE % | MAE(log) |", "| --- | ---: | ---: |"]
    for name, m in report["baselines"].items():
        lines.append(f"| {name} | {m['mdape_pct']} | {m['mae_log']} |")
    if "feature_importance_pct" in report:
        top = list(report["feature_importance_pct"].items())[:8]
        lines += ["", "Top features (gain %): " + ", ".join(f"{k} {v}" for k, v in top)]
    return "\n".join(lines) + "\n"


def to_markdown(report: dict) -> str:
    lines = [f"# Baseline — test year {report['test_year']}", "",
             (f"Trained on {report['train_years']} ({report['train_sales']:,} sales). "
              f"Model target: MAE(log) ≤ {report['target_mae_log']}."), ""]
    for name, entry in report["estimators"].items():
        lines += [f"## {name}", "", "| Scope | Sales | MdAPE % | Within 10 % | MAE € |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        rows = [(report.get("scope", "France"), entry["overall"])]
        rows += [(s, entry["by_segment"][s]) for s in SEGMENTS if s in entry["by_segment"]]
        rows += list(entry["by_type"].items())
        for label, m in rows:
            lines.append(f"| {label} | {m['sales']:,} | {m['mdape_pct']} | {m['within_10pct']} | {m['mae_eur']:,.0f} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="valia", description="French property price pipeline (DVF).")
    parser.add_argument("--version", action="version", version=f"valia {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "build", "all"):
        p = sub.add_parser(name)
        p.add_argument("--no-enrich", action="store_true", help="skip the open-data enrichment downloads")
        p.add_argument("--fast", action="store_true", help="small model, for smoke tests")
        p.add_argument("--years", default="2021-2025", help="e.g. 2021-2025 or 2023,2024")
        p.add_argument("--departements", default="all", help="'all' or a list such as 75,69,13")
        p.add_argument("--raw", default=str(RAW_DIR))
        p.add_argument("--clean", default=str(CLEAN_DIR))
        p.add_argument("--test-year", type=int, default=None)
    for name in ("baseline", "train"):
        p = sub.add_parser(name)
        p.add_argument("--clean", default=str(CLEAN_DIR))
        p.add_argument("--test-year", type=int, default=None)
        p.add_argument("--fast", action="store_true", help="small model, for smoke tests")
    args = parser.parse_args(argv)

    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "baseline":
        cmd_baseline(args)
    elif args.command == "train":
        cmd_train(args)
    else:
        cmd_fetch(args)
        cmd_build(args)
        cmd_baseline(args)
        cmd_train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
