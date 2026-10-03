<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 John Luke NIKABOU (LucNIK) -->

# Model card — Valia

*Filled in by each release. v0.1 publishes the reference estimators only; the model arrives in v0.2.*

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

### v0.2.1 — changes under test

- Same-type priors (a house is compared with houses) at ~550 m, ~2.2 km and commune level.
- Comparable sales: median price, distance and age of the 10 nearest earlier sales of the same type
  over 24 months, excluding the sale's own month. The anchor now starts from these comparables.

## Known limits

- DVF has no floor, energy rating (DPE), view or condition: two identical flats in the same building
  can sell for very different prices, and the range must say so.
- Rural communes have few sales; the range is wider there.
- Prices are past transactions; the model estimates, it does not forecast.
- Household income is not used yet: the only simple open file per commune dates from 2014, too old to
  describe today's market.
