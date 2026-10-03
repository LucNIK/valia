# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Building data from the BDNB (Base de données nationale des bâtiments, CSTB), joined by parcel.

DVF gives each sale its cadastral parcel; the BDNB links parcels to building groups and describes
them. For every parcel holding dwellings we keep the building group with the most dwellings and:

  dpe_class     energy class of its representative dwelling, A=1 … G=7
  dpe_date      date that DPE was made (a DPE made after a sale is never used for that sale)
  year_built    construction year (land files, else the DPE's estimate)
  levels        number of storeys
  dwellings     number of dwellings
  elevator      1 if the building has a lift
  social_share  share of social housing among its dwellings

The BDNB is published under the ODbL: any database built from it and published (the app's parcel
files) is shared under the same licence, with attribution to the CSTB.

Downloads are per département (`open_data_millesime_<m>_dep<dd>_csv.zip`, a few hundred MB). Only
the four tables used are read from the archive, column by column, and the archive is deleted
afterwards unless asked otherwise. Any failure leaves the building columns empty: the model runs
without them, exactly as before v0.4.
"""

from __future__ import annotations

import io
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import DATA_DIR

MILLESIME = "2026-02-a"
SOURCE_NAME = f"BDNB millésime {MILLESIME} (CSTB) — ODbL"
URL = ("https://open-data.s3.fr-par.scw.cloud/bdnb_millesime_{m}/millesime_{m}_dep{dep}/"
       "open_data_millesime_{m}_dep{dep}_csv.zip")
BDNB_DIR = DATA_DIR / "ref" / "bdnb"
USER_AGENT = f"valia/{__version__} (+https://github.com/LucNIK/valia)"

DPE_CLASSES = {c: i for i, c in enumerate("ABCDEFG", start=1)}
TABLES = {
    "rel_batiment_groupe_parcelle": ["batiment_groupe_id", "parcelle_id"],
    "batiment_groupe_ffo_bat": ["batiment_groupe_id", "annee_construction", "nb_niveau", "nb_log",
                                "presence_ascenseur", "nb_log_soc"],
    "batiment_groupe_dpe_representatif_logement": ["batiment_groupe_id", "classe_bilan_dpe",
                                                   "date_etablissement_dpe", "annee_construction_dpe"],
}
PARCEL_COLUMNS = ["parcel", "dpe_class", "dpe_date", "year_built", "levels", "dwellings", "elevator",
                  "social_share"]
# Model inputs derived from them (see attach())
BUILDING_COLUMNS = ["dpe_class", "year_built", "levels", "dwellings_log", "elevator", "social_share"]
MIN_YEAR, MAX_YEAR = 1000, 2030


def _member(archive: zipfile.ZipFile, table: str) -> str | None:
    names = [n for n in archive.namelist() if n.rsplit("/", 1)[-1].lower() == f"{table}.csv"]
    return names[0] if names else None


def _read_table(archive: zipfile.ZipFile, table: str, wanted: list[str]) -> pd.DataFrame | None:
    """Only the wanted columns that exist; the separator is detected from the header."""
    name = _member(archive, table)
    if name is None:
        print(f"[bdnb] {table}.csv not in archive")
        return None
    with archive.open(name) as fh:
        header = fh.readline().decode("utf-8-sig")
    sep = ";" if header.count(";") > header.count(",") else ","
    present = [c.strip().strip('"') for c in header.strip().split(sep)]
    cols = [c for c in wanted if c in present]
    if "batiment_groupe_id" not in cols:
        print(f"[bdnb] {table}.csv has no batiment_groupe_id")
        return None
    with archive.open(name) as fh:
        return pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8-sig"), sep=sep, usecols=cols, dtype=str,
                           low_memory=False)


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map(
        {"t": 1.0, "true": 1.0, "1": 1.0, "oui": 1.0, "f": 0.0, "false": 0.0, "0": 0.0, "non": 0.0})


def _num(s: pd.Series | None, index: pd.Index) -> pd.Series:
    if s is None:
        return pd.Series(np.nan, index=index)
    return pd.to_numeric(s, errors="coerce")


def parcels_from_tables(rel: pd.DataFrame, ffo: pd.DataFrame | None, dpe: pd.DataFrame | None) -> pd.DataFrame:
    """One row per parcel: the building group with the most dwellings on it."""
    rel = rel.dropna(subset=["parcelle_id", "batiment_groupe_id"])
    groups = pd.DataFrame({"batiment_groupe_id": rel["batiment_groupe_id"].unique()})
    if ffo is not None:
        ffo = ffo.drop_duplicates("batiment_groupe_id")
        groups = groups.merge(ffo, on="batiment_groupe_id", how="left")
    if dpe is not None:
        dpe = dpe.drop_duplicates("batiment_groupe_id")
        groups = groups.merge(dpe, on="batiment_groupe_id", how="left")
    idx = groups.index
    g = pd.DataFrame({"batiment_groupe_id": groups["batiment_groupe_id"]})
    g["dwellings"] = _num(groups.get("nb_log"), idx)
    def valid_year(v: pd.Series) -> pd.Series:
        return v.where(v.between(MIN_YEAR, MAX_YEAR))

    g["year_built"] = valid_year(_num(groups.get("annee_construction"), idx)).fillna(
        valid_year(_num(groups.get("annee_construction_dpe"), idx)))
    g["levels"] = _num(groups.get("nb_niveau"), idx).where(lambda v: v.between(1, 60))
    g["elevator"] = _bool(groups["presence_ascenseur"]) if "presence_ascenseur" in groups else np.nan
    social = _num(groups.get("nb_log_soc"), idx)
    g["social_share"] = (social / g["dwellings"].where(g["dwellings"] > 0)).clip(0, 1).round(2)
    cls = groups["classe_bilan_dpe"].str.strip().str.upper() if "classe_bilan_dpe" in groups else None
    g["dpe_class"] = cls.map(DPE_CLASSES) if cls is not None else np.nan
    date = groups.get("date_etablissement_dpe")
    g["dpe_date"] = (pd.to_datetime(date.str.slice(0, 10), errors="coerce") if date is not None
                     else pd.Series(pd.NaT, index=idx))
    g.loc[g["dpe_class"].isna(), "dpe_date"] = pd.NaT

    joined = rel[["parcelle_id", "batiment_groupe_id"]].merge(g, on="batiment_groupe_id", how="left")
    joined = joined.assign(_rank=joined["dwellings"].fillna(-1))
    best = (joined.sort_values(["parcelle_id", "_rank", "batiment_groupe_id"], ascending=[True, False, True],
                               kind="stable")
                  .drop_duplicates("parcelle_id"))
    best = best[(best["dwellings"].fillna(0) > 0) | best["dpe_class"].notna()]
    out = best.rename(columns={"parcelle_id": "parcel"})[PARCEL_COLUMNS].reset_index(drop=True)
    return out


def read_archive(path: Path) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(path) as archive:
            rel = _read_table(archive, "rel_batiment_groupe_parcelle", TABLES["rel_batiment_groupe_parcelle"])
            if rel is None or "parcelle_id" not in rel:
                return None
            ffo = _read_table(archive, "batiment_groupe_ffo_bat", TABLES["batiment_groupe_ffo_bat"])
            dpe = _read_table(archive, "batiment_groupe_dpe_representatif_logement",
                              TABLES["batiment_groupe_dpe_representatif_logement"])
    except (OSError, zipfile.BadZipFile, ValueError, pd.errors.ParserError) as exc:
        print(f"[bdnb] {path.name} unreadable: {exc}")
        return None
    return parcels_from_tables(rel, ffo, dpe)


def _download(url: str, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=600) as resp, tmp.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        tmp.replace(path)
        return True
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        print(f"[bdnb] could not download {url}: {exc}")
        return False


def parcels_for(dep: str, root: Path = BDNB_DIR, keep_archive: bool = False) -> pd.DataFrame | None:
    """Parcel table for one département, cached as parquet."""
    cached = root / f"parcels_{dep}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    archive = root / f"dep{dep}.zip"
    if not archive.exists():
        candidates = [dep, dep.lower()] if dep[1:].isalpha() else [dep]
        if not any(_download(URL.format(m=MILLESIME, dep=d), archive) for d in candidates):
            return None
    table = read_archive(archive)
    if not keep_archive:
        archive.unlink(missing_ok=True)
    if table is None:
        return None
    table.to_parquet(cached, index=False)
    print(f"[bdnb] dep {dep}: {len(table):,} parcels with dwellings")
    return table


def attach(sales: pd.DataFrame, parcels: pd.DataFrame | None) -> pd.DataFrame:
    """Building columns for each sale. A DPE made after the sale is not used for it."""
    out = sales.copy()
    if parcels is None or "parcel" not in out:
        for col in BUILDING_COLUMNS:
            out[col] = np.nan
        return out
    info = parcels.drop_duplicates("parcel").set_index("parcel")
    keys = out["parcel"].astype(str)
    late = keys.map(info["dpe_date"]) > out["date"]
    out["dpe_class"] = keys.map(info["dpe_class"]).where(~late)
    out["year_built"] = keys.map(info["year_built"])
    out["levels"] = keys.map(info["levels"])
    out["dwellings_log"] = np.log1p(keys.map(info["dwellings"]))
    out["elevator"] = keys.map(info["elevator"])
    out["social_share"] = keys.map(info["social_share"])
    return out


def building_features(entry: dict | None) -> dict[str, float]:
    """Model inputs from a published parcel entry (or user overrides), for the reference inference."""
    entry = entry or {}

    def get(key: str) -> float:
        v = entry.get(key)
        return math.nan if v is None else float(v)

    dwellings = get("dwellings")
    return {"dpe_class": get("dpe_class"), "year_built": get("year_built"), "levels": get("levels"),
            "dwellings_log": math.log1p(dwellings) if not math.isnan(dwellings) else math.nan,
            "elevator": get("elevator"), "social_share": get("social_share")}


def coverage(frame: pd.DataFrame) -> dict:
    return {col: round(float(frame[col].notna().mean()) * 100, 1) for col in BUILDING_COLUMNS if col in frame}
