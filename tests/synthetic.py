# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Synthetic raw DVF rows with the real column layout, including the awkward cases."""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = [
    "id_mutation", "date_mutation", "numero_disposition", "nature_mutation", "valeur_fonciere",
    "adresse_numero", "adresse_nom_voie", "code_postal", "code_commune", "nom_commune",
    "code_departement", "id_parcelle", "nombre_lots", "code_type_local", "type_local",
    "surface_reelle_bati", "nombre_pieces_principales", "code_nature_culture", "nature_culture",
    "surface_terrain", "longitude", "latitude",
]

COMMUNES = {  # code: (dep, lat, lon, base price per m²)
    "75056": ("75", 48.8566, 2.3522, 10_500),
    "69123": ("69", 45.7640, 4.8357, 5_000),
    "23096": ("23", 46.1700, 1.8700, 1_100),
}


def row(mid, date, value, commune, *, type_local=None, surface=None, rooms=None, land=None,
        parcel="p1", nature="Vente", lat=None, lon=None):
    dep, clat, clon, _ = COMMUNES[commune]
    return {
        "id_mutation": mid, "date_mutation": date, "numero_disposition": 1, "nature_mutation": nature,
        "valeur_fonciere": value, "adresse_numero": 1, "adresse_nom_voie": "RUE TEST",
        "code_postal": "00000", "code_commune": commune, "nom_commune": "Test", "code_departement": dep,
        "id_parcelle": f"{commune}{parcel}", "nombre_lots": 1,
        "code_type_local": None, "type_local": type_local, "surface_reelle_bati": surface,
        "nombre_pieces_principales": rooms, "code_nature_culture": "S" if land else None,
        "nature_culture": "sols" if land else None, "surface_terrain": land,
        "longitude": clon if lon is None else lon, "latitude": clat if lat is None else lat,
    }


def edge_cases() -> pd.DataFrame:
    """One mutation per cleaning rule, with the expected outcome in the id."""
    rows = [
        row("keep-flat", "2024-03-01", 300_000, "69123", type_local="Appartement", surface=60, rooms=3),
        row("keep-flat-garage", "2024-03-02", 320_000, "69123", type_local="Appartement", surface=60, rooms=3),
        row("keep-flat-garage", "2024-03-02", 320_000, "69123", type_local="Dépendance"),
        row("keep-house", "2024-03-03", 180_000, "23096", type_local="Maison", surface=110, rooms=5,
            land=800, parcel="a"),
        row("keep-house", "2024-03-03", 180_000, "23096", land=1_200, parcel="b"),
        row("drop-two-flats", "2024-03-04", 500_000, "69123", type_local="Appartement", surface=50, rooms=2),
        row("drop-two-flats", "2024-03-04", 500_000, "69123", type_local="Appartement", surface=40, rooms=2),
        row("drop-shop", "2024-03-05", 400_000, "69123", type_local="Appartement", surface=70, rooms=3),
        row("drop-shop", "2024-03-05", 400_000, "69123", type_local="Local industriel. commercial ou assimilé",
            surface=80),
        row("drop-exchange", "2024-03-06", 250_000, "69123", type_local="Appartement", surface=55, rooms=2,
            nature="Echange"),
        row("drop-land-only", "2024-03-07", 40_000, "23096", land=5_000),
        row("drop-tiny-price", "2024-03-08", 1_000, "69123", type_local="Appartement", surface=40, rooms=2),
        row("drop-no-coords", "2024-03-09", 200_000, "69123", type_local="Appartement", surface=40, rooms=2,
            lat=float("nan"), lon=float("nan")),
        row("drop-crazy-ppm2", "2024-03-10", 3_000_000, "23096", type_local="Appartement", surface=20, rooms=1),
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def market(years=(2021, 2022, 2023, 2024, 2025), per_year=400, seed=7) -> pd.DataFrame:
    """A small market with a yearly trend, a size effect and noise, for end-to-end tests."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for i in range(per_year):
            commune = rng.choice(list(COMMUNES))
            _dep, clat, clon, base = COMMUNES[commune]
            kind = "Maison" if commune == "23096" or rng.random() < 0.3 else "Appartement"
            surface = float(rng.integers(25, 160))
            ppm2 = base * (1.04 ** (year - 2021)) * (surface / 70) ** -0.12 * np.exp(rng.normal(0, 0.12))
            date = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=int(rng.integers(0, 365)))
            rows.append(row(f"m{year}-{i}", date.date().isoformat(), round(ppm2 * surface, -2), commune,
                            parcel=f"000AB{int(rng.integers(0, 80)):04d}",
                            type_local=kind, surface=surface, rooms=max(1, int(surface // 25)),
                            land=float(rng.integers(200, 2_000)) if kind == "Maison" else None,
                            lat=clat + rng.normal(0, 0.01), lon=clon + rng.normal(0, 0.01)))
    return pd.DataFrame(rows, columns=COLUMNS)


def random_forest(features: list[str], n_trees: int = 30, depth: int = 4, seed: int = 3,
                  centers: dict[str, tuple[float, float]] | None = None):
    """A random tree ensemble in the binary format, exercising every decision type.
    `centers` gives (mean, spread) per feature so thresholds fall where real values do."""
    from valia.treemodel import Forest

    rng = np.random.default_rng(seed)
    node_start, leaf_start, tree_nodes = [], [], []
    feature, decision, threshold, left, right, value = [], [], [], [], [], []
    node_count: list[float] = []
    leaf_count: list[float] = []
    for t in range(n_trees):
        node_start.append(len(feature))
        leaf_start.append(len(value))
        if t == 0:                                    # a single-leaf tree, as LightGBM can emit
            tree_nodes.append(0)
            value.append(0.01)
            leaf_count.append(100.0)
            continue
        n_internal = 2 ** depth - 1
        tree_nodes.append(n_internal)
        for i in range(n_internal):
            f = int(rng.integers(0, len(features)))
            mean, spread = (centers or {}).get(features[f], (0.0, 2.0))
            feature.append(f)
            decision.append(int(rng.choice([0, 2, 4, 6, 8, 10])))     # missing none/zero/NaN x default side
            threshold.append(float(mean + rng.normal(0, spread)))
            kids = [2 * i + 1, 2 * i + 2]
            # children past the internal nodes are leaves, encoded as ~leaf_index
            left.append(kids[0] if kids[0] < n_internal else ~(kids[0] - n_internal))
            right.append(kids[1] if kids[1] < n_internal else ~(kids[1] - n_internal))
        value.extend(rng.normal(0, 0.05, n_internal + 1).tolist())
        leaves = rng.integers(1, 500, n_internal + 1).astype(float)
        leaf_count.extend(leaves.tolist())
        counts = [0.0] * n_internal                  # each node: the rows of its subtree
        for i in range(n_internal - 1, -1, -1):
            kids = [2 * i + 1, 2 * i + 2]
            counts[i] = sum(counts[k] if k < n_internal else leaves[k - n_internal] for k in kids)
        node_count.extend(counts)
    return Forest(features=list(features),
                  node_start=np.array(node_start, dtype=np.int32), leaf_start=np.array(leaf_start, dtype=np.int32),
                  tree_nodes=np.array(tree_nodes, dtype=np.int32), feature=np.array(feature, dtype=np.uint8),
                  decision=np.array(decision, dtype=np.uint8), threshold=np.array(threshold, dtype=np.float64),
                  left=np.array(left, dtype=np.int32), right=np.array(right, dtype=np.int32),
                  value=np.array(value, dtype=np.float64), node_count=np.array(node_count, dtype=np.float64),
                  leaf_count=np.array(leaf_count, dtype=np.float64))


def bdnb_parcels(sales: pd.DataFrame, seed: int = 5) -> pd.DataFrame:
    """Building data for most parcels of a synthetic market, in the shape bdnb.parcels_from_tables returns."""
    rng = np.random.default_rng(seed)
    parcels = sorted(sales["parcel"].dropna().unique())
    parcels = [p for p in parcels if rng.random() < 0.85]
    n = len(parcels)
    return pd.DataFrame({
        "parcel": parcels,
        "dpe_class": np.where(rng.random(n) < 0.8, rng.integers(1, 8, n), np.nan),
        "dpe_date": pd.to_datetime("2021-07-01") + pd.to_timedelta(rng.integers(0, 1500, n), unit="D"),
        "year_built": np.where(rng.random(n) < 0.9, rng.integers(1850, 2024, n), np.nan),
        "levels": rng.integers(1, 9, n).astype(float),
        "dwellings": rng.integers(1, 80, n).astype(float),
        "elevator": np.where(rng.random(n) < 0.7, rng.integers(0, 2, n), np.nan),
        "social_share": np.round(rng.random(n) * 0.5, 2),
    })
