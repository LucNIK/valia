# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Pipeline tests: cleaning rules, leakage-free features, chronological evaluation, CLI end to end."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests import synthetic
from valia import baseline
from valia.__main__ import main
from valia.clean import clean
from valia.features import add_features
from valia.fetch import parse_departements, parse_years


class CleaningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sales, self.funnel = clean(synthetic.edge_cases())
        self.ids = set(self.sales["id_mutation"])

    def test_only_single_dwelling_sales_survive(self) -> None:
        self.assertEqual(self.ids, {"keep-flat", "keep-flat-garage", "keep-house"})

    def test_house_land_is_summed_over_parcels_and_flat_land_is_zero(self) -> None:
        house = self.sales.set_index("id_mutation").loc["keep-house"]
        self.assertEqual(house["land"], 2_000)
        self.assertEqual(self.sales.set_index("id_mutation").loc["keep-flat", "land"], 0)

    def test_price_per_m2_and_log_target(self) -> None:
        flat = self.sales.set_index("id_mutation").loc["keep-flat"]
        self.assertAlmostEqual(flat["price_m2"], 5_000)
        self.assertAlmostEqual(flat["log_ppm2"], np.log(5_000))
        self.assertEqual(flat["type"], "A")

    def test_funnel_is_monotonic(self) -> None:
        values = list(self.funnel.values())
        self.assertEqual(values, sorted(values, reverse=True))


class FeatureLeakageTest(unittest.TestCase):
    def _sales(self, dates_prices):
        raw = pd.DataFrame([synthetic.row(f"s{i}", d, p * 50, "69123", type_local="Appartement",
                                          surface=50, rooms=2)
                            for i, (d, p) in enumerate(dates_prices)], columns=synthetic.COLUMNS)
        sales, _ = clean(raw)
        return add_features(sales).set_index("id_mutation")

    def test_prior_uses_only_strictly_earlier_sales(self) -> None:
        feats = self._sales([("2024-01-01", 4_000), ("2024-01-02", 4_000), ("2024-01-03", 4_000),
                             ("2024-01-04", 9_000), ("2024-01-05", 5_000)])
        self.assertTrue(np.isnan(feats.loc["s2", "cell_prior"]))       # only 2 earlier sales
        self.assertAlmostEqual(np.exp(feats.loc["s3", "cell_prior"]), 4_000)  # its own 9 000 excluded
        self.assertEqual(feats.loc["s4", "cell_n"], 4)

    def test_same_day_sales_do_not_see_each_other(self) -> None:
        feats = self._sales([("2024-01-01", 4_000), ("2024-01-02", 4_000), ("2024-01-03", 4_000),
                             ("2024-02-01", 9_000), ("2024-02-01", 1_000)])
        self.assertEqual(feats.loc["s3", "cell_n"], 3)
        self.assertEqual(feats.loc["s4", "cell_n"], 3)

    def test_window_forgets_sales_older_than_a_year(self) -> None:
        feats = self._sales([("2022-01-01", 4_000), ("2022-01-02", 4_000), ("2022-01-03", 4_000),
                             ("2024-06-01", 5_000)])
        self.assertEqual(feats.loc["s3", "cell_n"], 0)


class EvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        sales, _ = clean(synthetic.market())
        self.sales = add_features(sales)

    def test_split_is_chronological(self) -> None:
        split = baseline.chronological_split(self.sales)
        self.assertEqual(split.test_year, 2025)
        self.assertTrue((split.train["date"] < pd.Timestamp(2025, 1, 1)).all())

    def test_report_has_both_estimators_segments_and_target(self) -> None:
        report = baseline.evaluate(self.sales)
        self.assertEqual(set(report["estimators"]), {"commune_median", "neighbourhood_prior"})
        segments = report["estimators"]["commune_median"]["by_segment"]
        self.assertIn("Paris", segments)
        self.assertLess(report["target_mae_log"], report["estimators"]["commune_median"]["overall"]["mae_log"])

    def test_neighbourhood_prior_tracks_the_trend_better(self) -> None:
        report = baseline.evaluate(self.sales)
        commune = report["estimators"]["commune_median"]["overall"]["mae_log"]
        prior = report["estimators"]["neighbourhood_prior"]["overall"]["mae_log"]
        self.assertLess(prior, commune)  # the prior sees last year's prices, the median sees 4 years


class CliTest(unittest.TestCase):
    def test_build_and_baseline_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp, "raw")
            market = synthetic.market(years=(2023, 2024, 2025), per_year=200)
            for (year, dep), part in market.groupby([market["date_mutation"].str[:4], "code_departement"]):
                path = raw / year / f"{dep}.csv.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                part.to_csv(path, index=False)
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                self.assertEqual(main(["build", "--years", "2023-2025", "--departements", "75,69,23",
                                       "--raw", str(raw), "--clean", "clean"]), 0)
                self.assertEqual(main(["baseline", "--clean", "clean"]), 0)
                report = json.loads(Path("reports/baseline.json").read_text())
                self.assertEqual(report["test_year"], 2025)
                self.assertTrue(Path("reports/baseline.md").read_text().startswith("# Baseline"))
            finally:
                os.chdir(cwd)


class ArgumentsTest(unittest.TestCase):
    def test_years_and_departements(self) -> None:
        self.assertEqual(parse_years("2021-2023,2025"), [2021, 2022, 2023, 2025])
        self.assertEqual(parse_departements("75, 2a,1"), ["75", "2A", "01"])
        with self.assertRaises(ValueError):
            parse_departements("67")  # Alsace is not covered by DVF
        self.assertNotIn("57", parse_departements("all"))


if __name__ == "__main__":
    unittest.main()
