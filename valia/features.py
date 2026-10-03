# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Leakage-safe neighbourhood features.

Every feature describes a sale using only sales that happened *strictly before* it, so the features
can be computed once and used for training and evaluation alike:

* rolling priors — median log price per m² over the previous 365 days, in the ~550 m cell, the
  ~2.2 km cell and the commune; each both for all homes and for the same property type (a house
  is not priced like the flat next door);
* comparable sales — the k nearest earlier sales of the same type over the previous 24 months
  (excluding the sale's own month), with their median price, distance and age. This is how an
  appraiser works, and the strongest single signal where cells hold few sales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CELL_DEG_LAT, CELL_DEG_LON, PRIOR_MIN_SALES, PRIOR_WINDOW

COARSE_FACTOR = 4                 # ~2.2 km cells
KNN_K = 10
KNN_LOOKBACK_MONTHS = 24
EARTH_RADIUS_KM = 6_371.0


def cell_id(lat: pd.Series, lon: pd.Series, factor: int = 1) -> pd.Series:
    row = np.floor(lat / (CELL_DEG_LAT * factor)).astype("int64")
    col = np.floor(lon / (CELL_DEG_LON * factor)).astype("int64")
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


def knn_comparables(df: pd.DataFrame, k: int = KNN_K, lookback: int = KNN_LOOKBACK_MONTHS) -> pd.DataFrame:
    """Median log price per m², distance (km) and age (months) of the k nearest earlier sales of the
    same type. Only months strictly before the sale's month are searched."""
    from sklearn.neighbors import BallTree

    out = np.full((len(df), 3), np.nan)
    months = (df["date"].dt.year * 12 + df["date"].dt.month).to_numpy()
    coords = np.radians(df[["lat", "lon"]].to_numpy(dtype=float))
    target = df["log_ppm2"].to_numpy(dtype=float)
    kinds = df["type"].to_numpy()
    for kind in pd.unique(kinds):
        idx = np.flatnonzero(kinds == kind)
        kind_months = months[idx]
        for month in np.unique(kind_months):
            past = idx[(kind_months < month) & (kind_months >= month - lookback)]
            if len(past) < k:
                continue
            query = idx[kind_months == month]
            dist, pos = BallTree(coords[past], metric="haversine").query(coords[query], k=k)
            neighbours = past[pos]
            out[query, 0] = np.median(target[neighbours], axis=1)
            out[query, 1] = np.median(dist, axis=1) * EARTH_RADIUS_KM
            out[query, 2] = np.median(month - months[neighbours], axis=1)
    return pd.DataFrame(out, index=df.index, columns=["knn_prior", "knn_km", "knn_age"])


def add_features(sales: pd.DataFrame) -> pd.DataFrame:
    out = sales.copy()
    out["cell"] = cell_id(out["lat"], out["lon"])
    out["cell2"] = cell_id(out["lat"], out["lon"], COARSE_FACTOR)
    out["cell_type"] = out["cell"] + "|" + out["type"]
    out["cell2_type"] = out["cell2"] + "|" + out["type"]
    out["commune_type"] = out["code_commune"].astype(str) + "|" + out["type"]
    for key, prefix in (("cell", "cell"), ("code_commune", "commune"), ("cell_type", "cell_type"),
                        ("cell2_type", "cell2_type"), ("commune_type", "commune_type")):
        out = out.join(_prior(out, key, prefix))
        out[f"{prefix}_n"] = out[f"{prefix}_n"].fillna(0).astype(int)
    out = out.join(knn_comparables(out))
    return out.drop(columns=["cell2", "cell_type", "cell2_type", "commune_type"])
