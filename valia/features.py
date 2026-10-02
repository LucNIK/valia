# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Leakage-safe neighbourhood features.

For every sale, the median price per m² of the sales that happened *strictly before* it, over the
previous 365 days, in the same ~550 m grid cell and in the same commune. Sales on the same day never
see each other, so the features can be computed once and used for training and evaluation alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CELL_DEG_LAT, CELL_DEG_LON, PRIOR_MIN_SALES, PRIOR_WINDOW


def cell_id(lat: pd.Series, lon: pd.Series) -> pd.Series:
    row = np.floor(lat / CELL_DEG_LAT).astype("int64")
    col = np.floor(lon / CELL_DEG_LON).astype("int64")
    return row.astype(str) + ":" + col.astype(str)


def _prior(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Rolling median / count of log price per m² over earlier sales sharing `key`."""
    ordered = df[[key, "date", "log_ppm2"]].sort_values([key, "date"], kind="stable")
    rolling = (ordered.set_index("date")
               .groupby(key, sort=False)["log_ppm2"]
               .rolling(PRIOR_WINDOW, closed="left"))   # 'left' excludes the current date itself
    median = rolling.median().to_numpy()
    count = rolling.count().to_numpy()
    result = pd.DataFrame({f"{prefix}_prior": median, f"{prefix}_n": count}, index=ordered.index)
    result.loc[result[f"{prefix}_n"] < PRIOR_MIN_SALES, f"{prefix}_prior"] = np.nan
    return result.reindex(df.index)


def add_features(sales: pd.DataFrame) -> pd.DataFrame:
    out = sales.copy()
    out["cell"] = cell_id(out["lat"], out["lon"])
    out = out.join(_prior(out, "cell", "cell")).join(_prior(out, "code_commune", "commune"))
    out["cell_n"] = out["cell_n"].fillna(0).astype(int)
    out["commune_n"] = out["commune_n"].fillna(0).astype(int)
    return out
