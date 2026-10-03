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
from valia.baseline import segment_of
from valia.clean import clean
from valia.enrich import enrich
from valia.export import export_assets
from valia.features import add_features
from valia.inference import Assets, Query, estimate, features, haversine_km, median
from valia.model import FEATURES, Anchor, design
from valia.treemodel import parse_lightgbm, predict, predict_row, read_binary, write_binary

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


class TreeModelTest(unittest.TestCase):
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
            self.assertEqual(json.loads(path.with_suffix(".json").read_text())["format"], "valia-trees/1")


def build_market() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Synthetic sales, with positions already rounded as the export rounds them, so that the
    training pipeline and the browser see exactly the same neighbours."""
    raw = synthetic.market(per_year=500)
    raw["latitude"] = raw["latitude"].round(3)
    raw["longitude"] = raw["longitude"].round(3)
    sales, _ = clean(raw)
    sales = enrich(add_features(sales), COMMUNES, STATIONS)
    as_of = (sales["date"].max() + pd.offsets.MonthBegin(1)).normalize()
    return sales, as_of


def export(sales: pd.DataFrame, root: Path):
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
                            np.round(STATIONS, 4), report, root)
    return forest, anchor, summary


class ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.sales, cls.as_of = build_market()
        cls.forest, cls.anchor, cls.summary = export(cls.sales, cls.root)
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
                    "price": 1.0})
        base = self.sales.drop(columns=[c for c in self.sales.columns if c.endswith(("_prior", "_n"))
                                        or c.startswith("knn_") or c in ("cell",)], errors="ignore")
        frame = pd.concat([base, pd.DataFrame([row])[base.columns]], ignore_index=True)
        frame = enrich(add_features(frame), COMMUNES, np.round(STATIONS, 4))
        virtual = frame[frame["id_mutation"].eq("virtual")]
        level = self.anchor.level(virtual)
        return design(virtual, level).iloc[0]

    def test_inference_features_match_training(self) -> None:
        cases = [(Query(48.857, 2.352, "A", 62, 3, citycode="75056"), "75056"),
                 (Query(45.7641, 4.8361, "M", 120, 5, land=600, citycode="69123"), "69123"),
                 (Query(46.17, 1.87, "M", 95, 4, land=1500, citycode=""), "23096")]
        for q, code in cases:
            with self.subTest(code=code):
                got, ctx = features(self.assets, q)
                self.assertEqual(ctx["commune"], code)
                want = self.virtual_sale(q, code)
                for name in FEATURES:
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

    def test_far_from_any_sale_falls_back_to_static_levels(self) -> None:
        q = Query(43.3, 5.4, "A", 50, 2, citycode="13201")                   # Marseille: no data here
        f, ctx = features(self.assets, q)
        self.assertTrue(math.isnan(f["knn_prior"]))
        self.assertEqual(ctx["anchor_source"], "france")
        self.assertAlmostEqual(f["anchor"], self.anchor.france, places=5)


class HelpersTest(unittest.TestCase):
    def test_median_and_distance(self) -> None:
        self.assertEqual(median([3, 1, 2]), 2)
        self.assertEqual(median([4, 1, 3, 2]), 2.5)
        self.assertTrue(math.isnan(median([])))
        self.assertAlmostEqual(haversine_km(48.8566, 2.3522, 45.7640, 4.8357), 392, delta=5)


if __name__ == "__main__":
    unittest.main()
