# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Static assets for the browser app.

Everything the app needs to estimate a home is precomputed here and published as small static files,
so estimation runs in the browser with no server:

  data/meta.json                 versions, data date, published accuracy, fallbacks
  data/model.bin + model.json    the web model (see treemodel.py)
  data/intervals.json            conformal half-widths per segment and property type
  data/stations.json             railway stations, for the distance feature
  data/communes/<dep>.json       per commune: segment, static median, profile, recent priors
  data/parcels/<code>.json       per parcel holding dwellings: building data from the BDNB (ODbL)
  data/tiles/index.json          which tiles exist: {row: [col, ...]}
  data/tiles/<row>_<col>.json    per ~2.2 km tile: recent priors of its ~550 m cells and the sales of
                                 the last 24 months (comparables and nearest-neighbour features)

Privacy (DVF licence): sale positions are rounded to 3 decimals (~100 m) and dates to the month;
no address, parcel or mutation id is ever published.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .bdnb import SOURCE_NAME as BDNB_SOURCE
from .config import CELL_DEG_LAT, CELL_DEG_LON, PRIOR_MIN_SALES
from .enrich import parent_commune
from .features import COARSE_FACTOR, KNN_K, KNN_LOOKBACK_MONTHS, cell_id
from .model import EXPLAIN_GROUPS

FORMAT = "valia-web/2"
PARCEL_FIELDS = ["dpe_class", "year_built", "levels", "dwellings", "elevator", "social_share"]
COORD_DECIMALS = 3
TYPES = ("A", "M")


def month_index(dates: pd.Series) -> pd.Series:
    return dates.dt.year * 12 + dates.dt.month


def _num(x: float, digits: int = 6) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), digits)


def priors_at(sales: pd.DataFrame, key: str, as_of: pd.Timestamp) -> pd.DataFrame:
    """Median log price per m² and count over [as_of - 365 days, as_of), per key and type and overall."""
    window = sales[(sales["date"] >= as_of - pd.Timedelta(days=365)) & (sales["date"] < as_of)]
    rows = []
    for kind in (*TYPES, "all"):
        part = window if kind == "all" else window[window["type"] == kind]
        g = part.groupby(key)["log_ppm2"].agg(["median", "count"])
        g["kind"] = kind
        rows.append(g)
    out = pd.concat(rows).reset_index().rename(columns={key: "key"})
    out.loc[out["count"] < PRIOR_MIN_SALES, "median"] = np.nan
    return out


def _prior_dict(table: pd.DataFrame) -> dict[str, dict[str, list]]:
    """{key: {"A": [median, n], "M": [...], "all": [...]}}"""
    out: dict[str, dict[str, list]] = {}
    for key, kind, med, n in table[["key", "kind", "median", "count"]].itertuples(index=False):
        out.setdefault(str(key), {})[kind] = [_num(med), int(n)]
    return out


def export_assets(sales: pd.DataFrame, forest, anchor: dict, intervals: dict, segments: dict[str, str],
                  communes: pd.DataFrame | None, stations: np.ndarray | None, report: dict,
                  out_dir: Path, parcels: pd.DataFrame | None = None) -> dict:
    """Write every asset under out_dir/data. Returns a summary (counts and sizes)."""
    from .treemodel import write_binary

    data = out_dir / "data"
    (data / "communes").mkdir(parents=True, exist_ok=True)
    (data / "tiles").mkdir(parents=True, exist_ok=True)

    last = sales["date"].max()
    as_of = (last + pd.offsets.MonthBegin(1)).normalize()       # first day of the next month
    now_month = int(as_of.year * 12 + as_of.month)

    model_bytes = write_binary(forest, data / "model.bin")
    (data / "intervals.json").write_text(json.dumps(intervals, indent=2, ensure_ascii=False))
    station_list = [] if stations is None else [round(float(v), 4) for v in np.asarray(stations).ravel()]
    (data / "stations.json").write_text(json.dumps(station_list))

    frame = sales.copy()
    frame["cell"] = cell_id(frame["lat"], frame["lon"])
    frame["tile"] = cell_id(frame["lat"], frame["lon"], COARSE_FACTOR)
    frame["code_commune"] = frame["code_commune"].astype(str)

    # ---- communes: segment, static median, profile, recent priors
    commune_priors = _prior_dict(priors_at(frame, "code_commune", as_of))
    profile = {}
    if communes is not None and len(communes):
        for row in communes.itertuples(index=False):
            profile[str(row.code)] = row
    deps = frame.groupby("code_commune")["dep"].first()
    parents = dict(zip(deps.index, parent_commune(deps.index.to_series()), strict=True))
    by_dep: dict[str, dict] = {}
    for code, dep in deps.items():
        entry = {"seg": segments.get(code, "Rural"), "static": _num(anchor["commune"].get(code)),
                 "prior": commune_priors.get(code, {})}
        p = profile.get(code)
        if p is None:   # arrondissements of Paris, Lyon, Marseille share their city's profile
            p = profile.get(str(parents[code]))
        if p is not None:
            entry["pop_log"] = _num(np.log1p(getattr(p, "population", np.nan)))
            entry["density_log"] = _num(np.log1p(getattr(p, "density", np.nan)))
            entry["density_grid"] = _num(getattr(p, "density_grid", np.nan))
            entry["equipment_level"] = _num(getattr(p, "equipment_level", np.nan))
        by_dep.setdefault(str(dep), {})[code] = entry
    for dep, entries in by_dep.items():
        (data / "communes" / f"{dep}.json").write_text(json.dumps(entries, separators=(",", ":")))

    # ---- tiles: fine-cell priors, coarse priors, sales of the last 24 months
    cell_priors = _prior_dict(priors_at(frame, "cell", as_of))
    tile_priors = _prior_dict(priors_at(frame, "tile", as_of))
    recent = frame[month_index(frame["date"]) >= now_month - KNN_LOOKBACK_MONTHS]
    cells_by_tile: dict[str, list[str]] = {}
    for cell, tile in frame[["cell", "tile"]].drop_duplicates().itertuples(index=False):
        if cell in cell_priors:
            cells_by_tile.setdefault(tile, []).append(cell)
    tiles = set(cells_by_tile) | set(recent["tile"].unique())
    sizes = 0
    grouped = dict(tuple(recent.groupby("tile")))
    for tile in tiles:
        part = grouped.get(tile)
        payload: dict = {
            "cells": {c: cell_priors[c] for c in sorted(cells_by_tile.get(tile, []))},
            "coarse": tile_priors.get(tile, {}),
        }
        if part is not None and len(part):
            part = part.sort_values(["date", "lat", "lon"], kind="stable")
            codes = sorted(part["code_commune"].unique())
            index = {c: i for i, c in enumerate(codes)}
            payload["communes"] = codes
            payload["sales"] = {
                "m": month_index(part["date"]).astype(int).tolist(),
                "t": part["type"].tolist(),
                "la": part["lat"].round(COORD_DECIMALS).tolist(),
                "lo": part["lon"].round(COORD_DECIMALS).tolist(),
                "s": part["surface"].round(1).tolist(),
                "p": part["price"].round(0).astype(int).tolist(),
                "l": part["log_ppm2"].round(6).tolist(),
                "c": [index[c] for c in part["code_commune"]],
            }
        text = json.dumps(payload, separators=(",", ":"))
        sizes += len(text)
        (data / "tiles" / f"{tile.replace(':', '_')}.json").write_text(text)

    index: dict[str, list[int]] = {}
    for tile in tiles:
        r, c = tile.split(":")
        index.setdefault(r, []).append(int(c))
    (data / "tiles" / "index.json").write_text(
        json.dumps({r: sorted(cols) for r, cols in sorted(index.items())}, separators=(",", ":")))

    parcel_summary = export_parcels(parcels, data)

    overall = report.get("model", {}).get("overall", {})
    meta = {
        "format": FORMAT,
        "version": report.get("version", ""),
        "data_through": last.date().isoformat(),
        "as_of": as_of.date().isoformat(),
        "now_month": now_month,
        "epoch_year": 2021,
        "grid": {"lat": CELL_DEG_LAT, "lon": CELL_DEG_LON, "coarse": COARSE_FACTOR},
        "prior_min_sales": PRIOR_MIN_SALES,
        "knn": {"k": KNN_K, "lookback_months": KNN_LOOKBACK_MONTHS},
        "static": {"dep": {k: _num(v) for k, v in anchor["dep"].items()}, "france": _num(anchor["france"])},
        "accuracy": {
            "test_year": report.get("test_year"), "test_sales": report.get("test_sales"),
            "mdape_pct": overall.get("mdape_pct"), "within_10pct": overall.get("within_10pct"),
            "coverage_pct": overall.get("coverage_pct"),
            "improvement_vs_commune_pct": report.get("improvement_vs_commune_pct"),
            "by_segment": {k: {"mdape_pct": v.get("mdape_pct"), "coverage_pct": v.get("coverage_pct")}
                           for k, v in report.get("model", {}).get("by_segment", {}).items()},
        },
        "parcel_fields": PARCEL_FIELDS,
        "explain_groups": EXPLAIN_GROUPS,
        "source": "Demandes de valeurs foncières géolocalisées (Etalab) — Licence Ouverte 2.0",
        "buildings_source": BDNB_SOURCE if parcels is not None else None,
    }
    (data / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return {"tiles": len(tiles), "tile_bytes": sizes, "model_bytes": model_bytes,
            "communes": sum(len(v) for v in by_dep.values()), "recent_sales": len(recent), **parcel_summary}


def export_parcels(parcels: pd.DataFrame | None, data: Path) -> dict:
    """data/parcels/<5 first characters>.json: {rest of the parcel id: [PARCEL_FIELDS values]}.

    This is a database derived from the BDNB, published under the ODbL like its source."""
    folder = data / "parcels"
    folder.mkdir(parents=True, exist_ok=True)
    if parcels is None or parcels.empty:
        (folder / "index.json").write_text("[]")
        return {"parcels": 0, "parcel_bytes": 0}
    frame = parcels.drop_duplicates("parcel").copy()
    frame = frame[frame["parcel"].astype(str).str.len() == 14]
    frame["code"] = frame["parcel"].str.slice(0, 5)
    frame["key"] = frame["parcel"].str.slice(5)
    size = 0
    codes = []
    for code, part in frame.groupby("code", sort=True):
        rows = {}
        for row in part[["key", *PARCEL_FIELDS]].itertuples(index=False):
            values = [_num(v, 2) for v in row[1:]]
            rows[row[0]] = [int(v) if v is not None and float(v).is_integer() else v for v in values]
        text = json.dumps(rows, separators=(",", ":"))
        size += len(text)
        (folder / f"{code}.json").write_text(text)
        codes.append(str(code))
    (folder / "index.json").write_text(json.dumps(codes, separators=(",", ":")))
    (folder / "LICENCE.txt").write_text(
        "Données dérivées de la BDNB (CSTB), publiées sous licence ODbL 1.0 : "
        "https://opendatacommons.org/licenses/odbl/1-0/\n")
    return {"parcels": len(frame), "parcel_bytes": size}
