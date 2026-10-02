# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Table storage: Parquet when pyarrow is available, gzipped CSV otherwise."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def save_table(df: pd.DataFrame, stem: Path) -> Path:
    """Write `df` next to `stem` (without extension) and return the file actually written."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    if has_parquet():
        path = stem.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    else:
        path = stem.with_suffix(".csv.gz")
        df.to_csv(path, index=False)
    return path


def load_table(stem: Path) -> pd.DataFrame:
    parquet, csv = stem.with_suffix(".parquet"), stem.with_suffix(".csv.gz")
    if parquet.exists() and has_parquet():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv, dtype={"code_commune": str, "dep": str}, parse_dates=["date"])
    raise FileNotFoundError(f"no table at {stem} (.parquet or .csv.gz)")
