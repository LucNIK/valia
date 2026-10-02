# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Command line: `python -m valia fetch|build|baseline|all`."""

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
    parts, funnel, files = [], {}, 0
    for dep in deps:
        paths = [p for y in years if (p := raw_path(y, dep, Path(args.raw))).exists()]
        if not paths:
            continue
        files += len(paths)
        sales, dep_funnel = clean(read_raw(paths))
        parts.append(add_features(sales))
        for step, n in dep_funnel.items():
            funnel[step] = funnel.get(step, 0) + n
    if not parts:
        sys.exit("no raw files found: run `python -m valia fetch` first")
    sales = pd.concat(parts, ignore_index=True).sort_values(["date", "id_mutation"], kind="stable")
    written = save_table(sales.reset_index(drop=True), Path(args.clean) / SALES.name)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "funnel.json").write_text(json.dumps(funnel, indent=2))
    print(f"[build] {files} files -> {len(sales):,} sales in {time.perf_counter() - started:.1f}s -> {written}")
    for step, n in funnel.items():
        print(f"        {step:<24} {n:>12,}")


def cmd_baseline(args) -> None:
    sales = load_table(Path(args.clean) / SALES.name)
    report = evaluate(sales, args.test_year)
    report["source"] = SOURCE_NAME
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "baseline.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (REPORTS_DIR / "baseline.md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))


def to_markdown(report: dict) -> str:
    lines = [f"# Baseline — test year {report['test_year']}", "",
             (f"Trained on {report['train_years']} ({report['train_sales']:,} sales). "
              f"Model target: MAE(log) ≤ {report['target_mae_log']}."), ""]
    for name, entry in report["estimators"].items():
        lines += [f"## {name}", "", "| Scope | Sales | MdAPE % | Within 10 % | MAE € |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        rows = [("France", entry["overall"])]
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
        p.add_argument("--years", default="2021-2025", help="e.g. 2021-2025 or 2023,2024")
        p.add_argument("--departements", default="all", help="'all' or a list such as 75,69,13")
        p.add_argument("--raw", default=str(RAW_DIR))
        p.add_argument("--clean", default=str(CLEAN_DIR))
        p.add_argument("--test-year", type=int, default=None)
    p = sub.add_parser("baseline")
    p.add_argument("--clean", default=str(CLEAN_DIR))
    p.add_argument("--test-year", type=int, default=None)
    args = parser.parse_args(argv)

    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "baseline":
        cmd_baseline(args)
    else:
        cmd_fetch(args)
        cmd_build(args)
        cmd_baseline(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
