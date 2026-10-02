# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Reference estimators every model must beat, measured on a chronological holdout.

* commune median — median price per m² of the commune over the training years
  (falls back to the département, then to France);
* neighbourhood prior — median of the previous 365 days in the ~550 m cell
  (falls back to the commune prior, then to the commune median).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LARGE_CITY_MIN_SALES, MID_CITY_MIN_SALES

SEGMENTS = ("Paris", "Grandes villes", "Villes moyennes", "Rural")


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    test_year: int


def chronological_split(sales: pd.DataFrame, test_year: int | None = None) -> Split:
    """Train on every year before `test_year`, test on `test_year` (default: the latest full year)."""
    if test_year is None:
        last_date = sales["date"].max()
        test_year = last_date.year if last_date.month == 12 and last_date.day >= 15 else last_date.year - 1
    train = sales[sales["year"] < test_year]
    test = sales[sales["year"] == test_year]
    if train.empty or test.empty:
        raise ValueError(f"need sales before and during {test_year} for a chronological holdout")
    return Split(train, test, test_year)


def segment_of(train: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    """Paris, then communes by number of sales during the training years."""
    volume = train.groupby("code_commune").size()
    n = frame["code_commune"].map(volume).fillna(0)
    seg = np.where(n >= LARGE_CITY_MIN_SALES, "Grandes villes",
                   np.where(n >= MID_CITY_MIN_SALES, "Villes moyennes", "Rural"))
    seg = np.where(frame["dep"].eq("75"), "Paris", seg)
    return pd.Series(seg, index=frame.index, name="segment")


def commune_median(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    by_commune = train.groupby("code_commune")["log_ppm2"].median()
    by_dep = train.groupby("dep")["log_ppm2"].median()
    pred = test["code_commune"].map(by_commune)
    pred = pred.fillna(test["dep"].map(by_dep)).fillna(train["log_ppm2"].median())
    return pred.rename("pred")


def neighbourhood_prior(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    pred = test["cell_prior"].fillna(test["commune_prior"])
    return pred.fillna(commune_median(train, test)).rename("pred")


def metrics(log_pred: pd.Series, frame: pd.DataFrame) -> dict:
    """Errors on the price itself (the predicted price per m² times the surface)."""
    pred_price = np.exp(log_pred) * frame["surface"]
    err = (pred_price - frame["price"]).abs()
    ape = err / frame["price"]
    return {
        "sales": len(frame),
        "mae_eur": round(float(err.mean()), 0),
        "mdape_pct": round(float(ape.median() * 100), 2),
        "mape_pct": round(float(ape.mean() * 100), 2),
        "within_10pct": round(float((ape <= 0.10).mean() * 100), 1),
        "mae_log": round(float((log_pred - frame["log_ppm2"]).abs().mean()), 4),
    }


def evaluate(sales: pd.DataFrame, test_year: int | None = None) -> dict:
    split = chronological_split(sales, test_year)
    test = split.test.assign(segment=segment_of(split.train, split.test))
    estimators = {"commune_median": commune_median, "neighbourhood_prior": neighbourhood_prior}
    report: dict = {
        "test_year": split.test_year,
        "train_years": sorted(int(y) for y in split.train["year"].unique()),
        "train_sales": len(split.train),
        "estimators": {},
    }
    for name, estimator in estimators.items():
        pred = estimator(split.train, test)
        entry = {"overall": metrics(pred, test), "by_segment": {}, "by_type": {}}
        for seg in SEGMENTS:
            mask = test["segment"].eq(seg)
            if mask.any():
                entry["by_segment"][seg] = metrics(pred[mask], test[mask])
        for code, label in (("A", "Appartements"), ("M", "Maisons")):
            mask = test["type"].eq(code)
            if mask.any():
                entry["by_type"][label] = metrics(pred[mask], test[mask])
        report["estimators"][name] = entry
    base = report["estimators"]["commune_median"]["overall"]["mae_log"]
    report["target_mae_log"] = round(base * 0.75, 4)  # the model must be 25% better than this
    return report
