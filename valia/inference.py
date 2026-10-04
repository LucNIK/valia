# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Reference inference over the exported web assets.

This is exactly what the browser does, written in Python: the JavaScript in `web/app/` must produce
the same features and the same price for the same query (`python -m valia parity` checks it).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bdnb import building_features
from .treemodel import Forest, explain_row, predict_row, read_binary

EARTH_RADIUS_KM = 6_371.0
MAX_RING = 4                      # up to 9x9 coarse tiles, ~20 km
TYPE_LABEL = {"A": "Appartements", "M": "Maisons"}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2) - math.radians(lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * math.asin(min(1.0, math.sqrt(h))) * EARTH_RADIUS_KM


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return math.nan
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def department_of(code: str) -> str:
    return code[:3] if code.startswith("97") else code[:2]


@dataclass
class Query:
    lat: float
    lon: float
    type: str              # "A" apartment, "M" house
    surface: float
    rooms: float
    land: float = 0.0
    citycode: str = ""
    parcel: str = ""                       # 14-character cadastral id, from the address lookup
    building: dict | None = None           # user corrections, e.g. {"dpe_class": 4, "year_built": None}


@dataclass
class Assets:
    root: Path
    meta: dict = field(init=False)
    forest: Forest = field(init=False)
    intervals: dict = field(init=False)
    stations: list[float] = field(init=False)
    tile_index: dict[str, set[str]] = field(init=False)
    trends: dict = field(init=False)
    parcel_codes: set[str] = field(init=False)
    _tiles: dict = field(default_factory=dict, init=False)
    _communes: dict = field(default_factory=dict, init=False)
    _parcels: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        data = self.root / "data"
        self.meta = json.loads((data / "meta.json").read_text())
        self.forest = read_binary(data / "model.bin")
        self.intervals = json.loads((data / "intervals.json").read_text())
        self.stations = json.loads((data / "stations.json").read_text())
        trends_file = data / "trends.json"
        self.trends = json.loads(trends_file.read_text()) if trends_file.exists() else {"deps": {}, "france": {}}
        index = json.loads((data / "tiles" / "index.json").read_text())
        self.tile_index = {r: {str(c) for c in cols} for r, cols in index.items()}
        parcel_index = data / "parcels" / "index.json"
        self.parcel_codes = set(json.loads(parcel_index.read_text())) if parcel_index.exists() else set()

    def tile(self, row: int, col: int) -> dict | None:
        key = f"{row}_{col}"
        if key not in self._tiles and str(col) not in self.tile_index.get(str(row), ()):
            return None
        if key not in self._tiles:
            path = self.root / "data" / "tiles" / f"{key}.json"
            self._tiles[key] = json.loads(path.read_text()) if path.exists() else None
        return self._tiles[key]

    def commune(self, code: str) -> dict | None:
        dep = department_of(code)
        if dep not in self._communes:
            path = self.root / "data" / "communes" / f"{dep}.json"
            self._communes[dep] = json.loads(path.read_text()) if path.exists() else {}
        return self._communes[dep].get(code)


    def parcel_file(self, code: str) -> dict:
        if code not in self.parcel_codes:
            return {}
        if code not in self._parcels:
            self._parcels[code] = json.loads((self.root / "data" / "parcels" / f"{code}.json").read_text())
        return self._parcels[code]

    def parcel(self, parcel_id: str) -> dict | None:
        """Building data of a parcel, as {field: value}, or None when unknown."""
        if len(parcel_id) != 14 or parcel_id[:5] not in self.parcel_codes:
            return None
        values = self.parcel_file(parcel_id[:5]).get(parcel_id[5:])
        return None if values is None else dict(zip(self.meta["parcel_fields"], values, strict=True))


def building_of(assets: Assets, q: Query) -> dict:
    """Published building data of the parcel, then the user's corrections on top."""
    entry = dict(assets.parcel(q.parcel) or {}) if q.parcel else {}
    for key, value in (q.building or {}).items():
        entry[key] = value
    return entry


def grid(meta: dict, lat: float, lon: float, factor: int = 1) -> tuple[int, int]:
    g = meta["grid"]
    return math.floor(lat / (g["lat"] * factor)), math.floor(lon / (g["lon"] * factor))


def _prior(entry: dict | None, kind: str) -> tuple[float, int]:
    if not entry or kind not in entry:
        return math.nan, 0
    med, n = entry[kind]
    return (math.nan if med is None else med), int(n)


def covered_km(meta: dict, lat: float, lon: float, row: int, col: int, ring: int) -> float:
    """Radius around the query fully inside the (2·ring+1)² block of coarse tiles: any sale closer
    than this lies in the block, so a k-th neighbour within it cannot be beaten by a farther tile."""
    g = meta["grid"]
    h, w = g["lat"] * g["coarse"], g["lon"] * g["coarse"]
    south, north = (row - ring) * h, (row + ring + 1) * h
    west, east = (col - ring) * w, (col + ring + 1) * w
    km_lat = math.pi * EARTH_RADIUS_KM / 180
    km_lon = km_lat * math.cos(math.radians(max(abs(south), abs(north))))
    return min((lat - south) * km_lat, (north - lat) * km_lat, (lon - west) * km_lon, (east - lon) * km_lon)


def neighbours(assets: Assets, q: Query) -> tuple[list[dict], dict | None, dict]:
    """Same-type sales around the query, nearest first.

    Rings of coarse tiles are added until the k-th nearest sale is provably the true k-th nearest
    (it lies within the covered radius), or MAX_RING is reached. Ties keep tile order, then file order.
    """
    k = assets.meta["knn"]["k"]
    factor = assets.meta["grid"]["coarse"]
    row, col = grid(assets.meta, q.lat, q.lon, factor)
    found: list[dict] = []
    nearest_any: dict | None = None
    for ring in range(0, MAX_RING + 1):
        cells = [(row + dr, col + dc) for dr in range(-ring, ring + 1) for dc in range(-ring, ring + 1)
                 if max(abs(dr), abs(dc)) == ring]
        for r, c in cells:
            tile = assets.tile(r, c)
            if not tile or "sales" not in tile:
                continue
            s = tile["sales"]
            for i in range(len(s["m"])):
                d = haversine_km(q.lat, q.lon, s["la"][i], s["lo"][i])
                sale = {"m": s["m"][i], "t": s["t"][i], "lat": s["la"][i], "lon": s["lo"][i],
                        "surface": s["s"][i], "price": s["p"][i], "l": s["l"][i],
                        "commune": tile["communes"][s["c"][i]], "km": d}
                if nearest_any is None or d < nearest_any["km"]:
                    nearest_any = sale
                if s["t"][i] == q.type:
                    found.append(sale)
        found.sort(key=lambda x: x["km"])
        if len(found) >= k and found[k - 1]["km"] <= covered_km(assets.meta, q.lat, q.lon, row, col, ring):
            break
    return found, nearest_any, assets.tile(row, col) or {}


def features(assets: Assets, q: Query) -> tuple[dict, dict]:
    """Model inputs for a query, plus context (commune entry, comparables, anchor source)."""
    meta = assets.meta
    found, nearest_any, own_tile = neighbours(assets, q)
    code = q.citycode
    entry = assets.commune(code) if code else None
    if entry is None and nearest_any is not None:          # e.g. a city code instead of an arrondissement
        code = nearest_any["commune"]
        entry = assets.commune(code)
    entry = entry or {}

    row, col = grid(meta, q.lat, q.lon)
    cell = own_tile.get("cells", {}).get(f"{row}:{col}")
    f: dict[str, float] = {}
    f["is_house"] = 1.0 if q.type == "M" else 0.0
    f["surface_log"] = math.log(q.surface)
    f["rooms"] = float(q.rooms)
    f["m2_per_room"] = q.surface / q.rooms if q.rooms > 0 else math.nan
    f["land_log"] = math.log1p(q.land if q.type == "M" else 0.0)
    f["lat"], f["lon"] = q.lat, q.lon
    as_of_month = meta["now_month"]
    f["months"] = float(as_of_month - 1 - meta["epoch_year"] * 12)   # same as design(): months since 2021-01
    for prefix, src, kind in (("cell", cell, "all"), ("commune", entry.get("prior"), "all"),
                              ("cell_type", cell, q.type), ("cell2_type", own_tile.get("coarse"), q.type),
                              ("commune_type", entry.get("prior"), q.type)):
        med, n = _prior(src, kind)
        f[f"{prefix}_prior"] = med
        f[f"{prefix}_n_log"] = math.log1p(n)
    k = meta["knn"]["k"]
    if len(found) >= k:
        top = found[:k]
        f["knn_prior"] = median([s["l"] for s in top])
        f["knn_km"] = median([s["km"] for s in top])
        f["knn_age"] = median([float(as_of_month - s["m"]) for s in top])
    else:
        f["knn_prior"] = f["knn_km"] = f["knn_age"] = math.nan
    for col_name in ("pop_log", "density_log", "density_grid", "equipment_level"):
        v = entry.get(col_name)
        f[col_name] = math.nan if v is None else float(v)
    building = building_of(assets, q)
    f.update(building_features(building))
    st = assets.stations
    f["station_km"] = (min(haversine_km(q.lat, q.lon, st[i], st[i + 1]) for i in range(0, len(st), 2))
                       if st else math.nan)

    anchor, source = math.nan, "france"
    for name in ("knn_prior", "cell_type_prior", "cell_prior", "cell2_type_prior",
                 "commune_type_prior", "commune_prior"):
        if not math.isnan(f[name]):
            anchor, source = f[name], name
            break
    if math.isnan(anchor):
        static = entry.get("static")
        dep_static = meta["static"]["dep"].get(department_of(code)) if code else None
        if static is not None:
            anchor, source = static, "commune_static"
        elif dep_static is not None:
            anchor, source = dep_static, "dep_static"
        else:
            anchor = meta["static"]["france"]
    f["anchor"] = anchor
    context = {"commune": code, "segment": entry.get("seg", "Rural"), "comparables": found[:5],
               "anchor_source": source, "building": building}
    return f, context


def explain(assets: Assets, row: np.ndarray) -> dict:
    """Why this price: base + per-feature contributions (log price per m², summing to the model's gap),
    and their sums per group of meta["explain_groups"]."""
    base, phi = explain_row(assets.forest, row)
    contributions = dict(zip(assets.forest.features, phi.tolist(), strict=True))
    groups = {name: sum(contributions[f] for f in members if f in contributions)
              for name, members in assets.meta["explain_groups"].items()}
    return {"base": base, "contributions": contributions, "groups": groups}


MIN_TREND_POINTS = 8


def _change(values: list, back: int) -> float | None:
    """Relative change from `back` quarters before the last point to the last point."""
    if len(values) <= back or values[-1] is None or values[-1 - back] is None:
        return None
    return values[-1] / values[-1 - back] - 1


def trend(assets: Assets, commune: str, kind: str) -> dict | None:
    """The most local price series with enough points: the commune for this type of home, the commune
    for all homes, then the département, then France."""
    entry = assets.commune(commune) if commune else None
    dep = assets.trends["deps"].get(department_of(commune), {}) if commune else {}
    candidates = [("commune", kind, (entry or {}).get("trend", {}).get(kind)),
                  ("commune", "all", (entry or {}).get("trend", {}).get("all")),
                  ("departement", kind, dep.get(kind)), ("france", kind, assets.trends["france"].get(kind))]
    for scope, k, values in candidates:
        if values and sum(v is not None for v in values) >= MIN_TREND_POINTS:
            first = next(v for v in values if v is not None)
            return {"scope": scope, "kind": k, "values": values, "year_change": _change(values, 4),
                    "total_change": values[-1] / first - 1 if values[-1] is not None else None}
    return None


def estimate(assets: Assets, q: Query, with_explanation: bool = False) -> dict:
    f, ctx = features(assets, q)
    row = np.array([f[name] for name in assets.forest.features], dtype=np.float64)
    gap = predict_row(assets.forest, row)
    log_ppm2 = f["anchor"] + gap
    group = f"{ctx['segment']} · {TYPE_LABEL[q.type]}"
    half = assets.intervals["by_group"].get(group, assets.intervals["global_half_width_log"])
    return {
        "log_ppm2": log_ppm2,
        "price": math.exp(log_ppm2) * q.surface,
        "low": math.exp(log_ppm2 - half) * q.surface,
        "high": math.exp(log_ppm2 + half) * q.surface,
        "price_m2": math.exp(log_ppm2),
        "features": f,
        "trend": trend(assets, ctx["commune"], q.type),
        **ctx,
        **({"explanation": explain(assets, row)} if with_explanation else {}),
    }


def _json_number(x: float) -> float | None:
    return None if isinstance(x, float) and math.isnan(x) else x


def sample_queries(assets: Assets, n: int, seed: int = 0) -> list[Query]:
    """Homes to check the browser against: around published sales (jittered ~300 m), with random
    characteristics, plus edge cases (no city code, unknown city code, far from any sale)."""
    rng = np.random.default_rng(seed)
    keys = [(int(r), int(c)) for r, cols in sorted(assets.tile_index.items()) for c in sorted(cols, key=int)]
    pool: list[tuple[float, float, str]] = []
    for r, c in (keys[i] for i in rng.permutation(len(keys))[:200]):
        tile = assets.tile(r, c) or {}
        s = tile.get("sales")
        if s:
            pool += [(s["la"][i], s["lo"][i], tile["communes"][s["c"][i]]) for i in range(len(s["m"]))]
    parcel_pool: list[str] = []
    for code in sorted(assets.parcel_codes)[:50]:
        parcel_pool += [code + key for key in assets.parcel_file(code)]
    queries: list[Query] = []
    for i in range(n):
        lat, lon, code = pool[int(rng.integers(len(pool)))] if pool else (46.5, 2.5, "")
        kind = "M" if rng.random() < 0.45 else "A"
        surface = float(rng.integers(15, 220) if kind == "M" else rng.integers(12, 160))
        rooms = float(max(1, min(12, round(surface / 22 + rng.normal(0, 0.8)))))
        land = float(rng.integers(0, 3000)) if kind == "M" else 0.0
        citycode = code if i % 10 else ("" if i % 20 else "00000")
        parcel = ""
        if parcel_pool and i % 3 == 0:
            parcel = parcel_pool[int(rng.integers(len(parcel_pool)))]
        elif i % 17 == 0:
            parcel = "99999000ZZ9999"                        # unknown parcel
        building = None
        if i % 7 == 0:
            building = {"dpe_class": None if rng.random() < 0.3 else int(rng.integers(1, 8)),
                        "year_built": None if rng.random() < 0.3 else int(rng.integers(1850, 2025))}
        queries.append(Query(round(lat + rng.normal(0, 0.003), 6), round(lon + rng.normal(0, 0.004), 6),
                             kind, surface, rooms, land, citycode, parcel, building))
    queries.append(Query(0.0, -30.0, "A", 50.0, 2.0, 0.0, ""))         # middle of the ocean
    return queries


def fixtures(assets: Assets, queries: list[Query], explain_every: int = 10) -> list[dict]:
    """Expected outputs, in the JSON shape the JavaScript tests read. Explanations are slow in
    Python (about a second for the web model), so only one query in `explain_every` gets one."""
    out = []
    for n, q in enumerate(queries):
        r = estimate(assets, q, with_explanation=n % explain_every == 0)
        out.append({
            "query": q.__dict__,
            "features": {k: _json_number(v) for k, v in r["features"].items()},
            "log_ppm2": r["log_ppm2"], "price": r["price"], "low": r["low"], "high": r["high"],
            "commune": r["commune"], "segment": r["segment"], "anchor_source": r["anchor_source"],
            "building": {k: _json_number(v) if isinstance(v, float) else v for k, v in r["building"].items()},
            "comparables": [[c["lat"], c["lon"], c["m"], c["price"]] for c in r["comparables"]],
            "trend": r["trend"],
            **({"explanation": r["explanation"]} if "explanation" in r else {}),
        })
    return out
