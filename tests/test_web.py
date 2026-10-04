# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Web export: binary trees, static assets, and the reference inference the browser mirrors.

The key property: a home estimated in the app gets exactly the features the model was trained
with, as if it had been sold on the first day after the data ends.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests import synthetic
from valia import bdnb
from valia.baseline import segment_of
from valia.clean import clean
from valia.enrich import enrich
from valia.export import export_assets
from valia.features import add_features
from valia.inference import Assets, Query, estimate, features, haversine_km, median, neighbours
from valia.model import FEATURES, Anchor, design
from valia.treemodel import explain_row, parse_lightgbm, predict, predict_row, read_binary, write_binary

STATIONS = np.array([[48.8443, 2.3744], [45.7605, 4.8596], [46.1720, 1.8690]])
COMMUNES = pd.DataFrame({"code": ["75056", "69123", "23096"], "population": [2.1e6, 5.2e5, 1.3e4],
                         "density": [2e4, 1.1e4, 90.0], "density_grid": [1, 1, 5],
                         "equipment_level": [4, 4, 2]})

LGB_TEXT = """tree
version=v4
num_class=1
max_feature_idx=1
feature_names=a b
objective=regression_l1

Tree=0
num_leaves=3
num_cat=0
split_feature=0 1
split_gain=1 1
threshold=1.5 0.000000000000000000000000000000000000001
decision_type=2 8
left_child=1 -2
right_child=-1 -3
leaf_value=10 20 30
leaf_weight=1 1 1
leaf_count=1 1 1
internal_value=0 0
internal_weight=0 0
internal_count=3 2
is_linear=0
shrinkage=1


Tree=1
num_leaves=1
num_cat=0
split_feature=
split_gain=
threshold=
decision_type=
left_child=
right_child=
leaf_value=0.5
leaf_weight=
leaf_count=
internal_value=
internal_weight=
internal_count=
is_linear=0
shrinkage=1


end of trees
"""


def shapley_by_enumeration(forest, row: np.ndarray) -> np.ndarray:
    """Exact Shapley values of the path-dependent value function, by brute force over feature subsets."""
    from itertools import combinations

    from valia.treemodel import _goes_left

    x = row.astype(np.float32).astype(np.float64)

    def value(subset: set[int]) -> float:
        total = 0.0
        for t in range(forest.n_trees):
            base, leaf = int(forest.node_start[t]), int(forest.leaf_start[t])

            def go(node: int, base: int = base, leaf: int = leaf) -> float:
                if node < 0:
                    return float(forest.value[leaf + ~node])
                i = base + node
                f, left, right = int(forest.feature[i]), int(forest.left[i]), int(forest.right[i])
                if f in subset:
                    return go(left if _goes_left(x[f], int(forest.decision[i]), forest.threshold[i]) else right)

                def n(c: int, base: int = base, leaf: int = leaf) -> float:
                    return float(forest.leaf_count[leaf + ~c] if c < 0 else forest.node_count[base + c])

                return (n(left) * go(left) + n(right) * go(right)) / float(forest.node_count[i])

            total += float(forest.value[leaf]) if forest.tree_nodes[t] == 0 else go(0)
        return total

    k = len(forest.features)
    phi = np.zeros(k)
    for j in range(k):
        others = [i for i in range(k) if i != j]
        for r in range(k):
            for subset in combinations(others, r):
                w = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
                phi[j] += w * (value(set(subset) | {j}) - value(set(subset)))
    return phi


class TreeModelTest(unittest.TestCase):
    def test_treeshap_is_exact(self) -> None:
        forest = synthetic.random_forest(["a", "b", "c", "d", "e"], n_trees=6, depth=4, seed=11)
        rows = np.random.default_rng(2).normal(0, 2, (6, 5))
        rows[1, 2] = np.nan
        rows[2, 0] = 0.0
        for row in rows:
            base, phi = explain_row(forest, row)
            self.assertAlmostEqual(base + phi.sum(), predict_row(forest, row), places=12)
            np.testing.assert_allclose(phi, shapley_by_enumeration(forest, row), atol=1e-12)
            with tempfile.TemporaryDirectory() as tmp:          # counts survive the binary format
                write_binary(forest, Path(tmp, "m.bin"))
                np.testing.assert_array_equal(explain_row(read_binary(Path(tmp, "m.bin")), row)[1], phi)

    def test_treeshap_matches_lightgbm(self) -> None:
        try:
            import lightgbm as lgb
        except ImportError:
            self.skipTest("LightGBM not installed")
        rng = np.random.default_rng(4)
        X = rng.normal(0, 1, (2000, 6))
        X[rng.random(X.shape) < 0.1] = np.nan
        X = X.astype(np.float32)                                   # as the pipeline feeds LightGBM
        y = np.nan_to_num(X[:, 0]) * 2 + np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2]) + rng.normal(0, 0.1, 2000)
        booster = lgb.train({"objective": "l1", "num_leaves": 15, "verbose": -1}, lgb.Dataset(X, y), 50)
        forest = parse_lightgbm(booster.model_to_string())
        want = booster.predict(X[:20], pred_contrib=True)
        for row, expected in zip(X[:20], want, strict=True):
            base, phi = explain_row(forest, row)
            np.testing.assert_allclose(phi, expected[:-1], atol=1e-6)
            self.assertAlmostEqual(base, expected[-1], places=6)

    def test_explanation_groups_cover_every_feature_once(self) -> None:
        from valia.model import EXPLAIN_GROUPS

        members = [f for group in EXPLAIN_GROUPS.values() for f in group]
        self.assertEqual(sorted(members), sorted(FEATURES))

    def test_lightgbm_text_and_decision_rules(self) -> None:
        forest = parse_lightgbm(LGB_TEXT)
        self.assertEqual(forest.features, ["a", "b"])
        self.assertEqual(forest.n_trees, 2)
        # decision_type bits: 1 categorical, 2 default left, 4-8 missing type (4: zero, 8: NaN)
        # node 0: a <= 1.5, missing none ; node 1: b <= ~0, missing NaN, default right
        self.assertAlmostEqual(predict_row(forest, np.array([2.0, 0.0])), 10.5)     # right of node 0 -> leaf 0
        self.assertAlmostEqual(predict_row(forest, np.array([1.0, -5.0])), 20.5)    # b <= ~0 -> leaf 1
        self.assertAlmostEqual(predict_row(forest, np.array([1.0, 5.0])), 30.5)     # b > ~0 -> leaf 2
        self.assertAlmostEqual(predict_row(forest, np.array([np.nan, 5.0])), 30.5)  # NaN a -> 0.0 -> left
        self.assertAlmostEqual(predict_row(forest, np.array([1.0, np.nan])), 30.5)

    def test_binary_round_trip(self) -> None:
        forest = synthetic.random_forest(["x", "y", "z"], n_trees=12, depth=3)
        X = np.random.default_rng(0).normal(0, 2, (200, 3))
        X[::7, 1] = np.nan
        X[::5, 2] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "model.bin")
            size = write_binary(forest, path)
            self.assertEqual(size, path.stat().st_size)
            back = read_binary(path)
            np.testing.assert_array_equal(predict(forest, X), predict(back, X))
            self.assertEqual(json.loads(path.with_suffix(".json").read_text())["format"], "valia-trees/2")


def build_market() -> tuple[pd.DataFrame, pd.Timestamp, pd.DataFrame]:
    """Synthetic sales, with positions already rounded as the export rounds them, so that the
    training pipeline and the browser see exactly the same neighbours."""
    raw = synthetic.market(per_year=500)
    raw["latitude"] = raw["latitude"].round(3)
    raw["longitude"] = raw["longitude"].round(3)
    sales, _ = clean(raw)
    parcels = synthetic.bdnb_parcels(sales)
    sales = bdnb.attach(enrich(add_features(sales), COMMUNES, STATIONS), parcels)
    as_of = (sales["date"].max() + pd.offsets.MonthBegin(1)).normalize()
    return sales, as_of, parcels


def export(sales: pd.DataFrame, root: Path, parcels: pd.DataFrame | None = None):
    forest = synthetic.random_forest(FEATURES, n_trees=40, depth=5, centers={
        "is_house": (0.5, 0.1), "surface_log": (4.2, 0.4), "rooms": (3, 1), "lat": (47, 1.5),
        "lon": (3, 1.5), "months": (40, 15), "anchor": (8, 0.8), "knn_prior": (8, 0.8),
        "knn_km": (1, 1), "knn_age": (8, 4), "station_km": (5, 4), "cell_prior": (8, 0.8)})
    train = sales[sales["year"] < 2025]
    anchor = Anchor.fit(train)
    per_commune = sales.groupby("code_commune", as_index=False)[["dep"]].first()
    segments = dict(zip(per_commune["code_commune"], segment_of(train, per_commune), strict=True))
    intervals = {"global_half_width_log": 0.3, "by_group": {"Paris · Appartements": 0.2}}
    report = {"version": "test", "test_year": 2025, "model": {"overall": {"mdape_pct": 15.0}}}
    anchors = {"commune": anchor.commune, "dep": anchor.dep, "france": anchor.france}
    summary = export_assets(sales, forest, anchors, intervals, segments, COMMUNES,
                            np.round(STATIONS, 4), report, root, parcels)
    return forest, anchor, summary


class ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.sales, cls.as_of, cls.parcels = build_market()
        cls.forest, cls.anchor, cls.summary = export(cls.sales, cls.root, cls.parcels)
        cls.assets = Assets(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_assets_are_written_and_private(self) -> None:
        data = self.root / "data"
        meta = json.loads((data / "meta.json").read_text())
        self.assertEqual(meta["as_of"], self.as_of.date().isoformat())
        self.assertGreater(self.summary["tiles"], 0)
        text = "".join(p.read_text() for p in (data / "tiles").glob("*.json"))
        for mid in self.sales["id_mutation"].head(50):
            self.assertNotIn(str(mid), text)
        for path in (data / "tiles").glob("*_*.json"):
            sales = json.loads(path.read_text()).get("sales")
            if sales:
                for v in sales["la"] + sales["lo"]:
                    self.assertEqual(round(v, 3), v)
                self.assertTrue(all(isinstance(m, int) for m in sales["m"]))
        communes = json.loads((data / "communes" / "75.json").read_text())
        self.assertEqual(next(iter(communes.values()))["seg"], "Paris")

    def virtual_sale(self, q: Query, code: str) -> pd.Series:
        """The query as a sale dated as_of, through the training pipeline."""
        dep = code[:2]
        row = {c: np.nan for c in self.sales.columns}
        row.update({"id_mutation": "virtual", "date": self.as_of, "year": self.as_of.year, "type": q.type,
                    "surface": q.surface, "rooms": q.rooms, "land": q.land if q.type == "M" else np.nan,
                    "lat": q.lat, "lon": q.lon, "code_commune": code, "dep": dep, "log_ppm2": 99.0,
                    "price": 1.0, "parcel": q.parcel or np.nan})
        base = self.sales.drop(columns=[c for c in self.sales.columns if c.endswith(("_prior", "_n"))
                                        or c.startswith("knn_") or c in ("cell",)], errors="ignore")
        frame = pd.concat([base, pd.DataFrame([row])[base.columns]], ignore_index=True)
        frame = bdnb.attach(enrich(add_features(frame), COMMUNES, np.round(STATIONS, 4)), self.parcels)
        virtual = frame[frame["id_mutation"].eq("virtual")]
        level = self.anchor.level(virtual)
        return design(virtual, level).iloc[0]

    def test_inference_features_match_training(self) -> None:
        first = self.parcels.groupby(self.parcels["parcel"].str.slice(0, 5))["parcel"].first()
        cases = [(Query(48.857, 2.352, "A", 62, 3, citycode="75056", parcel=first["75056"]), "75056"),
                 (Query(45.7641, 4.8361, "M", 120, 5, land=600, citycode="69123", parcel=first["69123"]), "69123"),
                 (Query(46.17, 1.87, "M", 95, 4, land=1500, citycode=""), "23096")]
        for q, code in cases:
            with self.subTest(code=code):
                got, ctx = features(self.assets, q)
                self.assertEqual(ctx["commune"], code)
                want = self.virtual_sale(q, code)
                # Positions are rounded to ~100 m, so several sales can sit at exactly the k-th distance:
                # which of them is taken is arbitrary (and differs between BallTree and the browser).
                found, _, _ = neighbours(self.assets, q)
                k = self.assets.meta["knn"]["k"]
                tie = len(found) > k and found[k - 1]["km"] == found[k]["km"]
                skip = {"knn_prior", "knn_age", "anchor"} if tie else set()
                for name in (n for n in FEATURES if n not in skip):
                    a, b = float(np.float32(got[name])), float(want[name])
                    if math.isnan(b):
                        self.assertTrue(math.isnan(a), name)
                    else:
                        self.assertAlmostEqual(a, b, delta=1e-4 * max(1.0, abs(b)), msg=name)

    def test_estimate(self) -> None:
        q = Query(48.857, 2.352, "A", 62, 3, citycode="75056")
        out = estimate(self.assets, q)
        f = out["features"]
        row = np.array([f[n] for n in FEATURES])
        self.assertAlmostEqual(out["log_ppm2"], f["anchor"] + predict_row(self.forest, row))
        self.assertLess(out["low"], out["price"])
        self.assertLess(out["price"], out["high"])
        self.assertAlmostEqual(math.log(out["high"] / out["price"]), 0.2)       # Paris · Appartements
        self.assertEqual(out["anchor_source"], "knn_prior")
        self.assertLessEqual(len(out["comparables"]), 5)

    def test_explanation_adds_up_to_the_estimate(self) -> None:
        q = Query(48.857, 2.352, "A", 62, 3, citycode="75056", parcel=self.parcels["parcel"].iloc[0])
        out = estimate(self.assets, q, with_explanation=True)
        e = out["explanation"]
        self.assertAlmostEqual(out["features"]["anchor"] + e["base"] + sum(e["groups"].values()), out["log_ppm2"],
                               places=9)
        self.assertEqual(set(e["groups"]), set(self.assets.meta["explain_groups"]))

    def test_price_trend(self) -> None:
        paris = estimate(self.assets, Query(48.857, 2.352, "A", 62, 3, citycode="75056"))["trend"]
        self.assertEqual((paris["scope"], paris["kind"]), ("commune", "A"))
        self.assertEqual(len(paris["values"]), len(self.assets.meta["trend"]["ends"]))
        self.assertAlmostEqual(paris["year_change"], 0.04, delta=0.03)      # the synthetic market: +4 % a year
        flats_in_gueret = estimate(self.assets, Query(46.17, 1.87, "A", 40, 2, citycode="23096"))["trend"]
        self.assertEqual(flats_in_gueret["kind"], "all")                    # no flat sold there: all homes
        nowhere = estimate(self.assets, Query(43.3, 5.4, "A", 50, 2, citycode="13201"))["trend"]
        self.assertEqual(nowhere["scope"], "france")
        # every published point aggregates at least TREND_MIN_SALES sales
        from valia.export import TREND_MIN_SALES, trends
        ends = [pd.Timestamp(e) for e in self.assets.meta["trend"]["ends"]]
        small = self.sales[self.sales["code_commune"].eq("23096")].head(TREND_MIN_SALES - 1)
        self.assertEqual(trends(small, "code_commune", ends), {})

    def test_user_corrections_replace_the_published_building(self) -> None:
        parcel = self.parcels["parcel"].iloc[0]
        q = Query(48.857, 2.352, "A", 62, 3, citycode="75056", parcel=parcel)
        published, ctx = features(self.assets, q)
        self.assertEqual(ctx["building"]["year_built"], self.parcels["year_built"].iloc[0])
        q.building = {"dpe_class": 2, "year_built": None}
        corrected, ctx = features(self.assets, q)
        self.assertEqual(corrected["dpe_class"], 2)
        self.assertTrue(math.isnan(corrected["year_built"]))
        self.assertEqual(corrected["levels"], published["levels"])
        unknown, _ = features(self.assets, Query(48.857, 2.352, "A", 62, 3, parcel="99999000ZZ9999"))
        self.assertTrue(all(math.isnan(unknown[c]) for c in bdnb.BUILDING_COLUMNS))

    def test_far_from_any_sale_falls_back_to_static_levels(self) -> None:
        q = Query(43.3, 5.4, "A", 50, 2, citycode="13201")                   # Marseille: no data here
        f, ctx = features(self.assets, q)
        self.assertTrue(math.isnan(f["knn_prior"]))
        self.assertEqual(ctx["anchor_source"], "france")
        self.assertAlmostEqual(f["anchor"], self.anchor.france, places=5)


class BuildingTest(unittest.TestCase):
    def test_parcel_table_picks_the_building_with_most_dwellings(self) -> None:
        rel = pd.DataFrame({"batiment_groupe_id": ["g1", "g2", "g3", "g4"],
                            "parcelle_id": ["75104000AB0001", "75104000AB0001", "75104000AB0002", "75104000AB0003"]})
        ffo = pd.DataFrame({"batiment_groupe_id": ["g1", "g2", "g3", "g4"],
                            "annee_construction": ["1880", "1975", "0", ""],
                            "nb_niveau": ["6", "12", "2", "1"], "nb_log": ["20", "3", "1", "0"],
                            "presence_ascenseur": ["f", "t", "f", "f"], "nb_log_soc": ["0", "3", "0", "0"]})
        dpe = pd.DataFrame({"batiment_groupe_id": ["g1", "g3"], "classe_bilan_dpe": ["D", "N"],
                            "date_etablissement_dpe": ["2023-05-02 00:00:00", "2022-01-01"],
                            "annee_construction_dpe": ["1890", "1960"]})
        out = bdnb.parcels_from_tables(rel, ffo, dpe).set_index("parcel")
        self.assertEqual(list(out.index), ["75104000AB0001", "75104000AB0002"])   # AB0003 has no dwelling
        first = out.loc["75104000AB0001"]
        got = (first["dpe_class"], first["year_built"], first["levels"], first["dwellings"])
        self.assertEqual(got, (4, 1880, 6, 20))
        self.assertEqual(first["elevator"], 0)
        second = out.loc["75104000AB0002"]
        self.assertTrue(math.isnan(second["dpe_class"]))                 # "N" is not a class
        self.assertEqual(second["year_built"], 1960)                     # 0 is no year: the DPE's estimate is used

    def test_archive_reading_is_tolerant(self) -> None:
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "dep75.zip")
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("csv/rel_batiment_groupe_parcelle.csv",
                           "batiment_groupe_id;parcelle_id;code_departement_insee\ng1;75104000AB0001;75\n")
                z.writestr("csv/batiment_groupe_ffo_bat.csv",
                           "batiment_groupe_id,nb_log,annee_construction\ng1,8,1930\n")
            out = bdnb.read_archive(path)
            self.assertEqual(out.iloc[0]["parcel"], "75104000AB0001")
            self.assertEqual(out.iloc[0]["dwellings"], 8)
            self.assertTrue(math.isnan(out.iloc[0]["dpe_class"]))          # no DPE table: empty, not a failure
            bad = Path(tmp, "bad.zip")
            bad.write_bytes(b"not a zip")
            self.assertIsNone(bdnb.read_archive(bad))

    def test_a_dpe_made_after_the_sale_is_not_used(self) -> None:
        sales = pd.DataFrame({"parcel": ["P1", "P1", "P2"],
                              "date": pd.to_datetime(["2022-01-01", "2024-01-01", "2024-01-01"])})
        parcels = pd.DataFrame({"parcel": ["P1"], "dpe_class": [6.0], "dpe_date": pd.to_datetime(["2023-03-01"]),
                                "year_built": [1950.0], "levels": [3.0], "dwellings": [9.0], "elevator": [1.0],
                                "social_share": [0.0]})
        out = bdnb.attach(sales, parcels)
        self.assertTrue(math.isnan(out["dpe_class"].iloc[0]))
        self.assertEqual(out["dpe_class"].iloc[1], 6)
        self.assertEqual(out["year_built"].iloc[0], 1950)                # static facts are always known
        self.assertTrue(out.iloc[2][bdnb.BUILDING_COLUMNS].isna().all())


class HelpersTest(unittest.TestCase):
    def test_median_and_distance(self) -> None:
        self.assertEqual(median([3, 1, 2]), 2)
        self.assertEqual(median([4, 1, 3, 2]), 2.5)
        self.assertTrue(math.isnan(median([])))
        self.assertAlmostEqual(haversine_km(48.8566, 2.3522, 45.7640, 4.8357), 392, delta=5)


if __name__ == "__main__":
    unittest.main()
