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
            dep, clat, clon, base = COMMUNES[commune]
            kind = "Maison" if commune == "23096" or rng.random() < 0.3 else "Appartement"
            surface = float(rng.integers(25, 160))
            ppm2 = base * (1.04 ** (year - 2021)) * (surface / 70) ** -0.12 * np.exp(rng.normal(0, 0.12))
            date = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=int(rng.integers(0, 365)))
            rows.append(row(f"m{year}-{i}", date.date().isoformat(), round(ppm2 * surface, -2), commune,
                            type_local=kind, surface=surface, rooms=max(1, int(surface // 25)),
                            land=float(rng.integers(200, 2_000)) if kind == "Maison" else None,
                            lat=clat + rng.normal(0, 0.01), lon=clon + rng.normal(0, 0.01)))
    return pd.DataFrame(rows, columns=COLUMNS)
