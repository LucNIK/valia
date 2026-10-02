# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Constants shared by the pipeline: data source, coverage and cleaning thresholds."""

from __future__ import annotations

from pathlib import Path

# Etalab's geolocated DVF ("DVF+ géolocalisées"), one gzipped CSV per year and département.
SOURCE_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{year}/departements/{dep}.csv.gz"
SOURCE_NAME = "Demandes de valeurs foncières géolocalisées (Etalab, data.gouv.fr) — Licence Ouverte 2.0"

# DVF does not cover Alsace (67, 68), Moselle (57) or Mayotte (976).
EXCLUDED_DEPARTEMENTS = frozenset({"57", "67", "68", "976"})
METRO = [f"{i:02d}" for i in range(1, 96) if i != 20] + ["2A", "2B"]
OVERSEAS = ["971", "972", "973", "974"]
DEPARTEMENTS = [d for d in METRO + OVERSEAS if d not in EXCLUDED_DEPARTEMENTS]

DWELLING_TYPES = {"Appartement": "A", "Maison": "M"}
ANCILLARY_TYPE = "Dépendance"          # garage, cellar: allowed alongside the dwelling
COMMERCIAL_TYPE = "Local industriel. commercial ou assimilé"

# Plausibility bounds for a single-dwelling sale.
MIN_SURFACE_M2, MAX_SURFACE_M2 = 9.0, 1_000.0
MIN_PRICE_EUR = 15_000.0
MIN_PRICE_M2, MAX_PRICE_M2 = 300.0, 30_000.0
MAX_ROOMS = 20

# Neighbourhood priors: ~550 m grid cells in mainland France; only strictly earlier sales count.
CELL_DEG_LAT, CELL_DEG_LON = 0.005, 0.007
PRIOR_WINDOW = "365D"
PRIOR_MIN_SALES = 3

# Segments used to report accuracy (number of sales in the commune during the training years).
LARGE_CITY_MIN_SALES = 1_000
MID_CITY_MIN_SALES = 200

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
REPORTS_DIR = Path("reports")
