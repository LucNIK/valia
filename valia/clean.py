# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""From raw DVF rows to one clean row per single-dwelling sale.

A DVF mutation spans several rows (one per lot, parcel or land use). We keep a mutation only when
it is a plain sale of exactly one apartment or one house, optionally with ancillary lots (garage,
cellar), and nothing commercial. Its price then refers to that one dwelling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ANCILLARY_TYPE,
    COMMERCIAL_TYPE,
    DWELLING_TYPES,
    MAX_PRICE_M2,
    MAX_ROOMS,
    MAX_SURFACE_M2,
    MIN_PRICE_EUR,
    MIN_PRICE_M2,
    MIN_SURFACE_M2,
)

RAW_COLUMNS = [
    "id_mutation", "date_mutation", "nature_mutation", "valeur_fonciere", "code_commune",
    "code_departement", "id_parcelle", "type_local", "surface_reelle_bati",
    "nombre_pieces_principales", "surface_terrain", "longitude", "latitude",
]
RAW_DTYPES = {"id_mutation": str, "code_commune": str, "code_departement": str, "id_parcelle": str,
              "type_local": str, "nature_mutation": str}

CLEAN_COLUMNS = ["id_mutation", "date", "year", "month", "dep", "code_commune", "parcel", "type", "surface",
                 "rooms", "land", "lat", "lon", "price", "price_m2", "log_ppm2"]


def read_raw(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p, usecols=RAW_COLUMNS, dtype=RAW_DTYPES, low_memory=False) for p in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RAW_COLUMNS)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (sales, funnel) where funnel counts how many mutations each rule removed."""
    funnel: dict[str, int] = {}
    df = raw[raw["nature_mutation"] == "Vente"].copy()
    funnel["mutations_total"] = int(raw["id_mutation"].nunique())
    funnel["after_sale_only"] = int(df["id_mutation"].nunique())

    df["is_dwelling"] = df["type_local"].isin(DWELLING_TYPES)
    df["is_commercial"] = df["type_local"].eq(COMMERCIAL_TYPE)
    other_built = df["type_local"].notna() & ~df["is_dwelling"] & ~df["type_local"].eq(ANCILLARY_TYPE)

    per_mut = df.assign(other_built=other_built).groupby("id_mutation").agg(
        commercial=("is_commercial", "any"),
        other=("other_built", "any"),
        prices=("valeur_fonciere", "nunique"),
    )
    # A dwelling repeated on several land-use rows is still one dwelling: count distinct ones.
    dwellings = (df[df["is_dwelling"]]
                 .drop_duplicates(["id_mutation", "type_local", "surface_reelle_bati", "nombre_pieces_principales"])
                 .groupby("id_mutation").size())
    per_mut["dwellings"] = dwellings.reindex(per_mut.index).fillna(0).astype(int)
    keep = per_mut[(per_mut["dwellings"] == 1) & ~per_mut["commercial"] & ~per_mut["other"]
                   & (per_mut["prices"] == 1)].index
    funnel["after_single_dwelling"] = len(keep)
    df = df[df["id_mutation"].isin(keep)]

    land = (df.drop_duplicates(["id_mutation", "id_parcelle", "surface_terrain"])
              .groupby("id_mutation")["surface_terrain"].sum(min_count=1))
    sales = df[df["is_dwelling"]].drop_duplicates("id_mutation").set_index("id_mutation")
    sales["land"] = land.reindex(sales.index).fillna(0.0)

    out = pd.DataFrame({
        "id_mutation": sales.index,
        "date": pd.to_datetime(sales["date_mutation"].to_numpy()),
        "dep": sales["code_departement"].to_numpy(),
        "code_commune": sales["code_commune"].to_numpy(),
        "parcel": sales["id_parcelle"].to_numpy(),          # the dwelling's parcel, joins the BDNB
        "type": sales["type_local"].map(DWELLING_TYPES).to_numpy(),
        "surface": pd.to_numeric(sales["surface_reelle_bati"], errors="coerce").to_numpy(),
        "rooms": pd.to_numeric(sales["nombre_pieces_principales"], errors="coerce").to_numpy(),
        "land": sales["land"].to_numpy(),
        "lat": pd.to_numeric(sales["latitude"], errors="coerce").to_numpy(),
        "lon": pd.to_numeric(sales["longitude"], errors="coerce").to_numpy(),
        "price": pd.to_numeric(sales["valeur_fonciere"], errors="coerce").to_numpy(),
    })
    out.loc[out["type"] == "A", "land"] = 0.0  # land under a flat belongs to the co-ownership

    plausible = (
        out["surface"].between(MIN_SURFACE_M2, MAX_SURFACE_M2)
        & out["rooms"].between(0, MAX_ROOMS)
        & (out["price"] >= MIN_PRICE_EUR)
        & out["lat"].notna() & out["lon"].notna()
    )
    out = out[plausible].copy()
    out["price_m2"] = out["price"] / out["surface"]
    out = out[out["price_m2"].between(MIN_PRICE_M2, MAX_PRICE_M2)].copy()
    funnel["after_plausibility"] = len(out)

    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["log_ppm2"] = np.log(out["price_m2"])
    out = out.sort_values(["date", "id_mutation"]).reset_index(drop=True)
    return out[CLEAN_COLUMNS], funnel
