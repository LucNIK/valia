# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Gradient-boosted price model with conformal prediction intervals.

Protocol, all chronological:
  * train proper  — every training year except its last three months;
  * calibration   — the last three months of the training years, never seen by the model;
                    the conformal quantiles are computed there, per segment and property type;
  * test          — the latest full year, untouched until the final evaluation.

The target is log(price per m²), so errors are relative and a 10 % miss weighs the same in Paris
and in a village. Trees cannot extrapolate in time, so the model does not predict the price level
itself: it predicts the gap to an *anchor*, the recent price of the neighbourhood (falling back to
the commune, then the département). The anchor follows the market without leakage; the model learns
what makes a home dearer or cheaper than its surroundings.

LightGBM is used when installed; scikit-learn's HistGradientBoosting otherwise.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baseline import SEGMENTS, chronological_split, commune_median, metrics, neighbourhood_prior, segment_of
from .enrich import ENRICH_COLUMNS

FEATURES = [
    "is_house", "surface_log", "rooms", "m2_per_room", "land_log", "lat", "lon", "months",
    "anchor", "cell_prior", "cell_n_log", "commune_prior", "commune_n_log", *ENRICH_COLUMNS,
]
CALIBRATION_MONTHS = 3
COVERAGE = 0.80
MIN_GROUP_SIZE = 200      # smaller calibration groups fall back to the global quantile
EPOCH = pd.Timestamp(2021, 1, 1)


@dataclass
class Anchor:
    """Reference level per sale: recent cell prior, else recent commune prior, else static medians."""
    commune: dict[str, float] = field(default_factory=dict)
    dep: dict[str, float] = field(default_factory=dict)
    france: float = float("nan")

    @classmethod
    def fit(cls, train: pd.DataFrame) -> Anchor:
        return cls(commune=train.groupby("code_commune")["log_ppm2"].median().to_dict(),
                   dep=train.groupby("dep")["log_ppm2"].median().to_dict(),
                   france=float(train["log_ppm2"].median()))

    def level(self, frame: pd.DataFrame) -> pd.Series:
        static = frame["code_commune"].map(self.commune).fillna(frame["dep"].map(self.dep)).fillna(self.france)
        return frame["cell_prior"].fillna(frame["commune_prior"]).fillna(static).astype(float)


class AnchoredModel:
    """prediction = anchor level + learned gap, all in log price per m²."""

    def __init__(self, backend: str, regressor, anchor: Anchor) -> None:
        self.backend, self.regressor, self.anchor = backend, regressor, anchor

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        level = self.anchor.level(frame)
        gap = self.regressor.predict(design(frame, level))
        return pd.Series(level.to_numpy() + gap, index=frame.index)


def design(frame: pd.DataFrame, level: pd.Series | None = None) -> pd.DataFrame:
    """Model inputs. Missing values stay NaN: the trees route them on their own."""
    X = pd.DataFrame(index=frame.index)
    X["is_house"] = frame["type"].eq("M").astype(float)
    X["surface_log"] = np.log(frame["surface"])
    X["rooms"] = frame["rooms"].astype(float)
    X["m2_per_room"] = frame["surface"] / frame["rooms"].where(frame["rooms"] > 0)
    X["land_log"] = np.log1p(frame["land"].fillna(0))
    X["lat"], X["lon"] = frame["lat"], frame["lon"]
    X["months"] = (frame["date"].dt.year - EPOCH.year) * 12 + frame["date"].dt.month - 1
    X["anchor"] = level if level is not None else np.nan
    X["cell_prior"], X["commune_prior"] = frame["cell_prior"], frame["commune_prior"]
    X["cell_n_log"] = np.log1p(frame["cell_n"])
    X["commune_n_log"] = np.log1p(frame["commune_n"])
    for col in ENRICH_COLUMNS:
        X[col] = frame.get(col, np.nan)
    return X[FEATURES].astype("float32")


class SklearnModel:
    """HistGradientBoosting cannot bin a feature that is entirely missing (e.g. an enrichment whose
    source was unavailable). Such columns are set to a constant, which the trees then ignore."""

    def __init__(self, estimator) -> None:
        self.estimator = estimator
        self.empty: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SklearnModel:
        self.empty = [c for c in X.columns if X[c].isna().all()]
        self.estimator.fit(self._prepare(X), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(self._prepare(X))

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.assign(**{c: 0.0 for c in self.empty}) if self.empty else X


def make_regressor(fast: bool = False):
    """LightGBM if available (production), scikit-learn otherwise. VALIA_BACKEND=sklearn forces it."""
    try:
        if os.environ.get("VALIA_BACKEND", "").lower() == "sklearn":
            raise ImportError("scikit-learn backend requested")
        import lightgbm as lgb
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return "sklearn", SklearnModel(HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.08, max_leaf_nodes=63 if fast else 255,
            max_iter=200 if fast else 1500, min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.05, n_iter_no_change=50, random_state=0))
    return "lightgbm", lgb.LGBMRegressor(
        objective="l1", learning_rate=0.05, num_leaves=63 if fast else 255,
        n_estimators=200 if fast else 3000, min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1)


def fit(regressor, backend: str, X: pd.DataFrame, y: pd.Series):
    if backend == "lightgbm":
        import lightgbm as lgb

        rng = np.random.default_rng(0)
        valid = rng.random(len(X)) < 0.05      # random 5 % for early stopping only
        regressor.fit(X[~valid], y[~valid], eval_set=[(X[valid], y[valid])], eval_metric="l1",
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        regressor.fit(X, y)
    return regressor


def group_key(segment: pd.Series, kind: pd.Series) -> pd.Series:
    return segment.astype(str) + " · " + kind.map({"A": "Appartements", "M": "Maisons"}).astype(str)


def conformal_quantile(scores: np.ndarray, coverage: float = COVERAGE) -> float:
    """Split-conformal quantile: the ceil((n+1)·coverage)-th smallest absolute residual."""
    n = len(scores)
    if n == 0:
        return float("nan")
    rank = min(math.ceil((n + 1) * coverage), n)
    return float(np.sort(scores)[rank - 1])


@dataclass
class Intervals:
    """Half-widths in log space, per calibration group, with a global fallback."""
    by_group: dict[str, float] = field(default_factory=dict)
    global_q: float = float("nan")

    def half_width(self, groups: pd.Series) -> np.ndarray:
        return groups.map(self.by_group).fillna(self.global_q).to_numpy()


def calibrate(residuals: pd.Series, groups: pd.Series) -> Intervals:
    scores = residuals.abs()
    intervals = Intervals(global_q=conformal_quantile(scores.to_numpy()))
    for name, part in scores.groupby(groups):
        if len(part) >= MIN_GROUP_SIZE:
            intervals.by_group[str(name)] = conformal_quantile(part.to_numpy())
    return intervals


def interval_report(frame: pd.DataFrame, log_pred: np.ndarray, half: np.ndarray) -> dict:
    lo, hi = log_pred - half, log_pred + half
    inside = (frame["log_ppm2"].to_numpy() >= lo) & (frame["log_ppm2"].to_numpy() <= hi)
    width = (np.exp(hi) - np.exp(lo)) / np.exp(log_pred)  # relative width around the estimate
    return {"coverage_pct": round(float(inside.mean() * 100), 1),
            "median_width_pct": round(float(np.median(width) * 100), 1)}


def train_and_evaluate(sales: pd.DataFrame, test_year: int | None = None, fast: bool = False):
    """Fit, calibrate and evaluate. Returns (model, intervals, report)."""
    split = chronological_split(sales, test_year)
    cutoff = pd.Timestamp(split.test_year, 1, 1) - pd.DateOffset(months=CALIBRATION_MONTHS)
    proper = split.train[split.train["date"] < cutoff]
    calib = split.train[split.train["date"] >= cutoff]
    if proper.empty or calib.empty:
        raise ValueError("not enough history to separate training and calibration periods")

    backend, regressor = make_regressor(fast)
    anchor = Anchor.fit(proper)
    level = anchor.level(proper)
    fit(regressor, backend, design(proper, level), proper["log_ppm2"] - level)
    model = AnchoredModel(backend, regressor, anchor)

    calib_groups = group_key(segment_of(split.train, calib), calib["type"])
    intervals = calibrate(calib["log_ppm2"] - model.predict(calib), calib_groups)

    test = split.test.assign(segment=segment_of(split.train, split.test))
    test_groups = group_key(test["segment"], test["type"])
    pred = model.predict(test)
    half = intervals.half_width(test_groups)

    report: dict = {
        "backend": backend,
        "test_year": split.test_year,
        "train_sales": len(proper), "calibration_sales": len(calib), "test_sales": len(test),
        "features": FEATURES,
        "coverage_target_pct": COVERAGE * 100,
        "model": {"overall": {**metrics(pred, test), **interval_report(test, pred.to_numpy(), half)},
                  "by_segment": {}, "by_type": {}},
        "baselines": {},
        "intervals": {"global_half_width_log": round(intervals.global_q, 4),
                      "by_group": {k: round(v, 4) for k, v in sorted(intervals.by_group.items())}},
    }
    for seg in SEGMENTS:
        mask = test["segment"].eq(seg)
        if mask.any():
            report["model"]["by_segment"][seg] = {**metrics(pred[mask], test[mask]),
                                                  **interval_report(test[mask], pred[mask].to_numpy(), half[mask])}
    for code, label in (("A", "Appartements"), ("M", "Maisons")):
        mask = test["type"].eq(code)
        if mask.any():
            report["model"]["by_type"][label] = {**metrics(pred[mask], test[mask]),
                                                 **interval_report(test[mask], pred[mask].to_numpy(), half[mask])}
    for name, estimator in (("commune_median", commune_median), ("neighbourhood_prior", neighbourhood_prior)):
        report["baselines"][name] = metrics(estimator(split.train, test), test)

    base = report["baselines"]["commune_median"]["mae_log"]
    got = report["model"]["overall"]["mae_log"]
    coverage = report["model"]["overall"]["coverage_pct"]
    report["target_mae_log"] = round(base * 0.75, 4)
    report["improvement_vs_commune_pct"] = round((1 - got / base) * 100, 1)
    report["passes"] = {"accuracy": got <= base * 0.75, "coverage": 78.0 <= coverage <= 82.0}
    if backend == "lightgbm":
        gains = regressor.booster_.feature_importance(importance_type="gain")
        total = gains.sum() or 1.0
        report["feature_importance_pct"] = {f: round(float(g / total * 100), 1)
                                            for f, g in sorted(zip(FEATURES, gains, strict=True), key=lambda t: -t[1])}
    return model, intervals, report
