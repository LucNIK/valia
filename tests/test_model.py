# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Model tests: enrichment, conformal intervals, chronological training on a synthetic market."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests import synthetic
from valia.clean import clean
from valia.enrich import enrich, load_communes, load_stations, nearest_km, parent_commune
from valia.features import add_features
from valia.model import FEATURES, Intervals, calibrate, conformal_quantile, design, train_and_evaluate


def market_sales(per_year: int = 600) -> pd.DataFrame:
    sales, _ = clean(synthetic.market(per_year=per_year))
    return add_features(sales)


class EnrichmentTest(unittest.TestCase):
    def test_arrondissements_map_to_their_city(self) -> None:
        codes = pd.Series(["75112", "69383", "13208", "75056", "23096"])
        self.assertEqual(list(parent_commune(codes)), ["75056", "69123", "13055", "75056", "23096"])

    def test_nearest_station_distance(self) -> None:
        paris, lyon = [48.8566, 2.3522], [45.7640, 4.8357]
        km = nearest_km(np.array([paris, lyon]), np.array([paris]))
        self.assertAlmostEqual(km[0], 0, places=3)
        self.assertAlmostEqual(km[1], 392, delta=5)   # Paris-Lyon as the crow flies

    def test_reference_files_are_parsed_tolerantly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            communes = Path(tmp, "communes.csv")
            communes.write_text("code_insee,population,densite,grille_densite,niveau_equipements_services\n"
                                "75056,2100000,20000,1,4\n23096,13000,90,5 - Bourgs ruraux,2\n")
            frame = load_communes(communes)
            self.assertEqual(frame.set_index("code").loc["23096", "density_grid"], 5)
            stations = Path(tmp, "stations.csv")
            stations.write_text("nom;position_geographique\nParis Gare de Lyon;48.8443, 2.3744\nBroken;\n")
            self.assertEqual(load_stations(stations).shape, (1, 2))
            bad = Path(tmp, "bad.csv")
            bad.write_text("foo,bar\n1,2\n")
            self.assertIsNone(load_stations(bad))
            self.assertIsNone(load_communes(bad))

    def test_enrich_fills_columns_and_survives_missing_sources(self) -> None:
        sales = market_sales(50)
        communes = pd.DataFrame({"code": ["75056", "69123"], "population": [2.1e6, 5.2e5],
                                 "density": [2e4, 1.1e4], "density_grid": [1, 1], "equipment_level": [4, 4]})
        out = enrich(sales, communes, np.array([[48.8443, 2.3744]]))
        paris = out["code_commune"].eq("75056")
        self.assertTrue(out.loc[paris, "pop_log"].notna().all())
        self.assertTrue(out.loc[~paris & out["code_commune"].eq("23096"), "pop_log"].isna().all())
        self.assertTrue(out["station_km"].notna().all())
        bare = enrich(sales, None, None)
        self.assertTrue(bare["station_km"].isna().all())


class ConformalTest(unittest.TestCase):
    def test_quantile_rank(self) -> None:
        scores = np.arange(1, 10, dtype=float)              # n = 9
        self.assertEqual(conformal_quantile(scores, 0.8), 8.0)  # ceil(10 * 0.8) = 8th value

    def test_small_groups_fall_back_to_global(self) -> None:
        rng = np.random.default_rng(0)
        residuals = pd.Series(rng.normal(0, 0.1, 1_000))
        groups = pd.Series(["big"] * 950 + ["tiny"] * 50)
        intervals = calibrate(residuals, groups)
        self.assertIn("big", intervals.by_group)
        self.assertNotIn("tiny", intervals.by_group)
        widths = intervals.half_width(pd.Series(["big", "tiny", "unknown"]))
        self.assertEqual(widths[1], intervals.global_q)
        self.assertEqual(widths[2], intervals.global_q)

    def test_intervals_cover_about_80_percent_of_fresh_residuals(self) -> None:
        rng = np.random.default_rng(1)
        calib = pd.Series(rng.normal(0, 0.15, 5_000))
        q = Intervals(global_q=conformal_quantile(calib.abs().to_numpy())).global_q
        fresh = rng.normal(0, 0.15, 20_000)
        self.assertAlmostEqual(float((np.abs(fresh) <= q).mean()), 0.80, delta=0.02)


class TrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sales = market_sales()
        cls.model, cls.intervals, cls.report = train_and_evaluate(cls.sales, fast=True)

    def test_design_matrix_has_every_feature(self) -> None:
        X = design(self.sales)
        self.assertEqual(list(X.columns), FEATURES)
        self.assertEqual(X.dtypes.unique().tolist(), [np.dtype("float32")])

    def test_chronological_protocol(self) -> None:
        r = self.report
        self.assertEqual(r["test_year"], 2025)
        self.assertEqual(r["train_sales"] + r["calibration_sales"],
                         int((self.sales["year"] < 2025).sum()))

    def test_model_beats_both_references(self) -> None:
        model = self.report["model"]["overall"]["mae_log"]
        for reference in ("commune_median", "neighbourhood_prior"):
            self.assertLess(model, self.report["baselines"][reference]["mae_log"], reference)

    def test_prediction_is_anchor_plus_gap(self) -> None:
        sample = self.sales.tail(20)
        level = self.model.anchor.level(sample)
        gap = self.model.regressor.predict(design(sample, level))
        np.testing.assert_allclose(self.model.predict(sample).to_numpy(), level.to_numpy() + gap)

    def test_coverage_is_reported_and_plausible(self) -> None:
        coverage = self.report["model"]["overall"]["coverage_pct"]
        self.assertTrue(60 <= coverage <= 95, coverage)
        self.assertIn("accuracy", self.report["passes"])


if __name__ == "__main__":
    unittest.main()
