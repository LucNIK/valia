<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 John Luke NIKABOU (LucNIK) -->

# Model card — Valia

*Filled in by each release. Latest: v0.4 — building data from the BDNB; the −25 % target is met.*

Last full run: DVF 2021–2025, all covered départements, test year **2025** (800,229 sales never seen in training).

## Intended use

Give a private person or a professional an order of magnitude for the price of an apartment or a house
in France, with an honest range. **Not** a substitute for a professional appraisal, a mortgage valuation
or a notary's opinion.

## Data

| Item | Value |
| --- | --- |
| Source | DVF géolocalisées (Etalab, data.gouv.fr), Licence Ouverte 2.0 |
| Coverage | Metropolitan France and overseas départements, except Alsace (67, 68), Moselle (57) and Mayotte (976) |
| Scope | Plain sales of exactly one apartment or one house (with optional garage or cellar) |
| Cleaning funnel | 7,319,609 mutations → 6,803,387 sales → 4,379,133 single-dwelling sales → **4,263,898** plausible sales (`reports/funnel.json`) |
| Split | Training 2021–2024: 3,463,669 sales · Test 2025: 800,229 sales |

## Evaluation protocol

- Chronological holdout: train on every year before the latest full year, test on that year.
- Errors measured on the price itself: MdAPE, share of estimates within 10 %, MAE in euros.
- Reported for France, Paris, large cities (≥ 1 000 sales in the training years), mid-size cities
  (≥ 200), rural communes (< 200), apartments and houses.

## Reference estimators (v0.1)

Median absolute percentage error on the sale price (lower is better), and share of estimates within ±10 %.

| Scope | Test sales | Commune median | Neighbourhood prior | Within 10 % (best) |
| --- | ---: | ---: | ---: | ---: |
| **France** | 800,229 | 20.70 % | **19.21 %** | 28.3 % |
| Paris | 28,884 | 14.87 % | **12.67 %** | 40.8 % |
| Large cities | 317,128 | 20.36 % | **17.25 %** | 31.1 % |
| Mid-size cities | 213,080 | 19.28 % | **18.36 %** | 29.6 % |
| Rural | 241,137 | **23.43 %** | 24.60 % | 22.7 % |
| Apartments | 352,727 | 18.90 % | **15.59 %** | 34.3 % |
| Houses | 447,502 | **22.22 %** | 22.78 % | 23.9 % |

**Target for v0.2:** MAE on log price per m² ≤ **0.2138**, i.e. 25 % better than the commune median.
Neither reference wins everywhere: recent neighbourhood prices are better in cities and for apartments,
the commune median is better in rural areas and for houses, where ~550 m cells hold too few sales.

## Model (v0.2)

- Target: log price per m²; prediction = recent anchor + learned gap (LightGBM, L1 loss).
- Intervals: split conformal at 80 %, calibrated on the last three months of the training years, one
  quantile per segment × property type (global quantile for groups under 200 sales).

### Run 1 — v0.2.0 (3 October 2026)

Train 3,271,761 · calibration 191,908 · test 800,229 sales (2025). Enrichment coverage ≥ 98.9 %.

| Scope | MAE(log) model | MAE(log) commune median | Gain | 80 % range coverage |
| --- | ---: | ---: | ---: | ---: |
| **France** | **0.2371** | 0.2851 | −16.8 % | **81.4 %** ✅ |
| Paris | 0.1950 | 0.2207 | −11.6 % | 81.5 % |
| Large cities | 0.2078 | 0.2722 | −23.7 % | 81.4 % |
| Mid-size cities | 0.2242 | 0.2675 | −16.2 % | 81.4 % |
| Rural | 0.2922 | 0.3252 | −10.1 % | 81.3 % |

Median absolute error on the price: **16.2 %** for France (commune median: 20.7 %).

**Verdict: intervals pass, accuracy does not** (target −25 %). The weakest segments are rural areas
and houses: the neighbourhood anchor mixed houses and flats and was missing where cells hold few
sales. Per the delivery plan, the web app waits until the accuracy target is met.

### Run 2 — v0.2.1 (3 October 2026)

Same data and split as run 1. Changes: same-type priors (a house is compared with houses) at ~550 m,
~2.2 km and commune level; comparable sales (median price, distance and age of the 10 nearest earlier
sales of the same type over 24 months, excluding the sale's own month), which now start the anchor.

| Scope | Test sales | MAE(log) model | MAE(log) commune median | Gain | MdAPE | 80 % range coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **France** | 800,229 | **0.2312** | 0.2851 | **−18.9 %** | **15.68 %** | **81.3 %** ✅ |
| Paris | 28,884 | 0.1942 | 0.2207 | −12.0 % | 12.37 % | 81.2 % |
| Large cities | 317,128 | 0.1986 | 0.2722 | −27.0 % | 13.95 % | 81.2 % |
| Mid-size cities | 213,080 | 0.2191 | 0.2675 | −18.1 % | 14.82 % | 81.4 % |
| Rural | 241,137 | 0.2892 | 0.3252 | −11.1 % | 20.15 % | 81.3 % |
| Apartments | 352,727 | 0.1889 | — | −26.0 % | 13.08 % | 81.3 % |
| Houses | 447,502 | 0.2645 | — | −14.3 % | 18.23 % | 81.4 % |

Better than both references in every segment, with calibrated intervals everywhere; 34.0 % of
estimates are within 10 % of the sale price. Top features by gain: surface, land, anchor, comparables.

**Decision (v0.3).** The −25 % target is not met (−18.9 %). The model still beats the commune median
and the neighbourhood prior in every segment, and its ranges are honest, so it ships with these
numbers published in the app; −25 % stays the stated target. The remaining gap is mostly what DVF
cannot see — condition, floor, energy rating — which v0.4 addresses with the BDNB building data.

## Web model (v0.3)

- The browser runs a compact LightGBM model (`--profile web`: 63 leaves, 800 trees) instead of the full
  one (255 leaves, 3,000 trees, ~70 MB). It is trained, calibrated and evaluated on its own with the same
  protocol; its accuracy is the one displayed in the app (`reports/web/model.md`).
- Trees are exported to a compact binary format (`valia/treemodel.py`) evaluated in JavaScript with
  LightGBM's exact decision rules (float32 inputs, missing-value routing).
- An estimate uses the same features as training, as if the home were sold on the first day after the
  data ends. `valia/inference.py` is the reference; the JavaScript mirrors it line by line.
- Published data: per-commune and per-~2.2 km tile priors, and the sales of the last 24 months with
  their position rounded to ~100 m and their date to the month; no address, parcel or identifier.

### Run 3 — v0.3.0 web model (3 October 2026)

Same data and split as runs 1 and 2. Training time: 2 min 46 s. Model file: 800 trees, 49,600 splits,
1.3 MB (0.68 MB compressed) instead of ~70 MB for the full model.

| Scope | Test sales | MAE(log) | MdAPE | Within 10 % | 80 % range coverage | Range width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **France** | 800,229 | **0.2325** | **15.79 %** | 33.7 % | **81.3 %** ✅ | 73.7 % |
| Paris | 28,884 | 0.1945 | 12.39 % | 41.8 % | 81.1 % | 55.0 % |
| Large cities | 317,128 | 0.1997 | 14.03 % | 37.4 % | 81.2 % | 59.2 % |
| Mid-size cities | 213,080 | 0.2204 | 14.94 % | 35.4 % | 81.4 % | 77.2 % |
| Rural | 241,137 | 0.2910 | 20.36 % | 26.4 % | 81.4 % | 100.2 % |
| Apartments | 352,727 | 0.1899 | 13.18 % | 39.6 % | 81.3 % | 59.2 % |
| Houses | 447,502 | 0.2661 | 18.38 % | 29.1 % | 81.3 % | 77.2 % |

−18.4 % error vs the commune median (full model: −18.9 %): the compact model costs 0.11 point of MdAPE
for a file 50 times smaller. Top features by gain: surface, land, anchor, comparables, months.

**Exit criterion met:** on the same run, the browser code reproduced the Python reference on 1,000 homes
across France (features, price and range to 10⁻⁹) before the site was published to GitHub Pages.
The same check runs on a synthetic market in CI on every push.

## Building data (v0.4)

- Source: BDNB millésime 2026-02.a (CSTB), ODbL. Tables used: `rel_batiment_groupe_parcelle`,
  `batiment_groupe_ffo_bat`, `batiment_groupe_dpe_representatif_logement`.
- Join: the sale's cadastral parcel (DVF `id_parcelle`) → building groups on that parcel → the one
  with the most dwellings.
- New inputs: DPE class (A=1 … G=7), construction year, storeys, log number of dwellings, lift,
  share of social housing.
- Leakage: a DPE dated after the sale is set to missing for that sale (a renovation after the sale
  must not explain its price). The other attributes describe the building itself and are static.
- In the app: the address is matched to its parcel (IGN reverse geocoding) and the building data
  pre-fills the energy class and construction year, which the user can correct.
- Limits: for an apartment the DPE is that of a representative dwelling of the building, not
  necessarily the one sold; the BDNB is a 2026 snapshot applied to 2021–2025 sales.

### Run 4 — v0.4.0 (3 October 2026)

Same data and split as runs 1–3, plus the BDNB (millésime 2026-02.a). Share of sales with each
building input: construction year 92.7 %, storeys and dwellings 93.6 %, DPE class 31.4 % (a DPE is
only used when it was made before the sale). Lift and social-housing share are not in the open
BDNB tables used (0 %): they stay empty and the trees ignore them.

| Scope | Test sales | MAE(log) full | Gain vs commune median | MdAPE full | MdAPE web | Within 10 % (full) | 80 % range coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **France** | 800,229 | **0.2090** | **−26.7 %** ✅ | **13.94 %** | **14.24 %** | 37.9 % | **81.5 %** ✅ |
| Paris | 28,884 | 0.1926 | −12.7 % | 12.18 % | 12.29 % | 42.2 % | 81.0 % |
| Large cities | 317,128 | 0.1855 | −31.9 % | 12.89 % | 13.15 % | 40.4 % | 81.4 % |
| Mid-size cities | 213,080 | 0.1973 | −26.2 % | 13.07 % | 13.34 % | 40.1 % | 81.5 % |
| Rural | 241,137 | 0.2522 | −22.4 % | 16.95 % | 17.35 % | 32.0 % | 81.6 % |
| Apartments | 352,727 | 0.1796 | — | 12.34 % | 12.59 % | 41.9 % | 81.4 % |
| Houses | 447,502 | 0.2322 | — | 15.45 % | 15.82 % | 34.6 % | 81.6 % |

The web model (the one in the app) reaches −25.6 % (MAE(log) 0.2121 ≤ target 0.2138).

**Verdict: target met.** Both models beat the commune median by more than 25 %, with calibrated
ranges in every segment. Construction year becomes the second most important input after surface;
the DPE class follows (5.4 % of gain in the web model). The largest gains are where v0.3 was weakest:
houses (MdAPE 18.4 % → 15.8 %) and rural communes (20.4 % → 17.4 %), and the ranges get narrower
(France: 73.7 % → 66.7 % width for the web model).

Run history (France, full model): MdAPE 16.2 % (run 1) → 15.7 % (run 2) → **13.9 %** (run 4);
gain vs commune median −16.8 % → −18.9 % → **−26.7 %**.

## Explanations (v1.0)

- Each estimate is decomposed with path-dependent TreeSHAP (Lundberg et al., 2020) on the web model:
  log price per m² = anchor + base + Σ contributions, where base is the average gap over the training
  rows. The tree file stores the training row counts per node (`valia-trees/2`) for this purpose.
- Exactness is tested three ways: against Shapley values computed by brute force over every subset of
  features, against LightGBM's own `pred_contrib` (CI), and browser against Python on the reference
  homes (to 10⁻⁹).
- The app groups contributions into surface and rooms, type, land, building, precise location, date,
  and area (neighbourhood prices and commune profile). They describe what the model learned from past
  sales, not causal effects.

## Known limits

- DVF has no floor, energy rating (DPE), view or condition: two identical flats in the same building
  can sell for very different prices, and the range must say so.
- Rural communes have few sales; the range is wider there.
- Prices are past transactions; the model estimates, it does not forecast.
- Household income is not used yet: the only simple open file per commune dates from 2014, too old to
  describe today's market.
