# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Open-data enrichment: commune profile and distance to the nearest railway station.

Both sources are optional by design. If a download fails or a file changes shape, the matching
columns are left empty (NaN): gradient-boosted trees handle missing values natively, so the model
keeps working and the pipeline never breaks because a third-party file moved.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import DATA_DIR

REF_DIR = DATA_DIR / "ref"
USER_AGENT = f"valia/{__version__} (+https://github.com/LucNIK/valia)"

# "Communes et villes de France" (data.gouv.fr, Licence Ouverte 2.0), csv.gz resource.
COMMUNES_URL = "https://www.data.gouv.fr/api/1/datasets/r/262afe2d-1c35-40da-ace0-a2eb595eaced"
# SNCF Gares & Connexions, "Gares de voyageurs" (ODbL), Opendatasoft CSV export.
STATIONS_URL = ("https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/"
                "gares-de-voyageurs/exports/csv?delimiter=%3B")

# DVF uses arrondissement codes in Paris, Lyon and Marseille; commune files use the city code.
ARRONDISSEMENT_PARENTS = (("751", "75056"), ("6938", "69123"), ("132", "13055"))
ENRICH_COLUMNS = ["pop_log", "density_log", "density_grid", "equipment_level", "station_km"]
EARTH_RADIUS_KM = 6_371.0


def download(url: str, path: Path) -> Path | None:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as resp:
            path.write_bytes(resp.read())
        return path
    except OSError as exc:
        print(f"[enrich] could not download {url}: {exc} — columns left empty")
        return None


def parent_commune(code: pd.Series) -> pd.Series:
    out = code.astype(str)
    for prefix, parent in ARRONDISSEMENT_PARENTS:
        out = out.where(~(out.str.startswith(prefix) & (out.str.len() == 5) & (out != parent)), parent)
    return out


def load_communes(path: Path | None) -> pd.DataFrame | None:
    """code_insee, population, density, density grid and equipment level, or None if unusable."""
    if path is None:
        return None
    try:
        raw = pd.read_csv(path, dtype={"code_insee": str}, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"[enrich] communes file unreadable: {exc}")
        return None
    wanted = {"code_insee": "code", "population": "population", "densite": "density",
              "grille_densite": "density_grid", "niveau_equipements_services": "equipment_level"}
    if "code_insee" not in raw.columns:
        print("[enrich] communes file has no code_insee column — columns left empty")
        return None
    frame = raw[[c for c in wanted if c in raw.columns]].rename(columns=wanted)
    frame["density_grid"] = pd.to_numeric(frame.get("density_grid", pd.Series(dtype=float))
                                           .astype(str).str.extract(r"(\d)")[0], errors="coerce")
    return frame.drop_duplicates("code")


def _coords_from(raw: pd.DataFrame) -> pd.DataFrame | None:
    cols = {c.lower(): c for c in raw.columns}
    lat = next((cols[c] for c in cols if c in ("latitude", "lat", "y_wgs84")), None)
    lon = next((cols[c] for c in cols if c in ("longitude", "lon", "lng", "x_wgs84")), None)
    if lat and lon:
        return pd.DataFrame({"lat": pd.to_numeric(raw[lat], errors="coerce"),
                             "lon": pd.to_numeric(raw[lon], errors="coerce")})
    pos = next((cols[c] for c in cols if "position" in c or "geo_point" in c or "coordon" in c), None)
    if pos:
        pairs = raw[pos].astype(str).str.extract(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")
        return pd.DataFrame({"lat": pd.to_numeric(pairs[0], errors="coerce"),
                             "lon": pd.to_numeric(pairs[1], errors="coerce")})
    return None


def load_stations(path: Path | None) -> np.ndarray | None:
    """Station coordinates as an (n, 2) array of [lat, lon] in degrees, or None if unusable."""
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8-sig")
        sep = ";" if text.count(";") > text.count(",") / 2 else ","
        raw = pd.read_csv(io.StringIO(text), sep=sep)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"[enrich] stations file unreadable: {exc}")
        return None
    coords = _coords_from(raw)
    if coords is None:
        print("[enrich] stations file has no coordinates — station_km left empty")
        return None
    coords = coords.dropna()
    coords = coords[coords["lat"].between(-90, 90) & coords["lon"].between(-180, 180)]
    return coords[["lat", "lon"]].to_numpy() if len(coords) else None


def fetch_references(root: Path = REF_DIR) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    communes = load_communes(download(COMMUNES_URL, root / "communes.csv.gz"))
    stations = load_stations(download(STATIONS_URL, root / "stations.csv"))
    return communes, stations


def nearest_km(points_deg: np.ndarray, targets_deg: np.ndarray) -> np.ndarray:
    """Great-circle distance in km from each point to the nearest target (BallTree, haversine)."""
    from sklearn.neighbors import BallTree

    tree = BallTree(np.radians(targets_deg), metric="haversine")
    dist, _ = tree.query(np.radians(points_deg), k=1)
    return dist[:, 0] * EARTH_RADIUS_KM


def enrich(sales: pd.DataFrame, communes: pd.DataFrame | None, stations: np.ndarray | None) -> pd.DataFrame:
    out = sales.copy()
    for col in ENRICH_COLUMNS:
        out[col] = np.nan
    if communes is not None and len(out):
        codes = out["code_commune"].astype(str)
        prof = communes.set_index("code")
        exact = codes.map(prof["population"]) if "population" in prof else pd.Series(np.nan, index=out.index)
        key = codes.where(exact.notna(), parent_commune(codes))
        if "population" in prof:
            out["pop_log"] = np.log1p(pd.to_numeric(key.map(prof["population"]), errors="coerce"))
        if "density" in prof:
            out["density_log"] = np.log1p(pd.to_numeric(key.map(prof["density"]), errors="coerce"))
        for col in ("density_grid", "equipment_level"):
            if col in prof:
                out[col] = pd.to_numeric(key.map(prof[col]), errors="coerce")
    if stations is not None and len(out):
        out["station_km"] = nearest_km(out[["lat", "lon"]].to_numpy(), stations)
    return out


def coverage(frame: pd.DataFrame) -> dict:
    """Share of sales with each enrichment filled, reported in the model card."""
    return {col: round(float(frame[col].notna().mean()) * 100, 1) for col in ENRICH_COLUMNS if col in frame}


__all__ = ["ENRICH_COLUMNS", "coverage", "enrich", "fetch_references", "nearest_km", "parent_commune"]
