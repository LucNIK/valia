<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 John Luke NIKABOU (LucNIK) -->

<div align="center">

# Valia

**What is this home worth — and how sure are we?**

Property price estimation for the whole of France, built on the public DVF sales records.
A price, a calibrated range, the reasons behind it and the closest real sales — computed in the browser, with no server.

`v0.1 · 4.26 M sales · data pipeline` · [Model card](docs/model-card.md) · MIT

</div>

---

## Status

| Version | Scope | Exit criterion | State |
| --- | --- | --- | --- |
| **v0.1** | DVF pipeline: download, cleaning, leakage-free neighbourhood features, reference estimators | Reproducible dataset in one command, tests green | **Done** — 4.26 M sales, see the [model card](docs/model-card.md) |
| v0.2 | Open-data enrichment, gradient-boosted model anchored on recent prices, conformal intervals per segment | 25 % lower error than the commune median; 80 % intervals cover 78–82 % of prices | **In progress** — run 1: −16.8 % error vs commune median (target −25 %), 81.4 % interval coverage ✅ |
| v0.3 | Web MVP: address, estimate, range, 5 comparable sales, Method page | Browser and Python agree on 1 000 homes | Planned |
| v1.0 | SHAP explanations, local trend, comparison, offline app, automatic retraining | Public demo and model card | Planned |

## Why it is different

- **Measured honestly.** Every number is computed on sales that happened *after* the training data —
  never on a shuffled split. Results are reported separately for Paris, large cities, mid-size cities
  and rural France, because a national average hides where a model fails.
- **No leakage, by construction.** Neighbourhood features only use sales that happened strictly before
  the one being described, over the previous 365 days. Tests prove it.
- **Uncertainty you can see.** From v0.2, every estimate comes with a range calibrated by conformal
  prediction, and its real coverage is published.
- **Respectful of the data licence.** No exact address is ever displayed and no sale is indexable by
  search engines (see [SECURITY.md](SECURITY.md)).

## Quick start

```bash
python -m pip install -e ".[dev]"
python -m valia all --years 2023-2025 --departements 75   # Paris only, ~1 minute
cat reports/baseline.md
```

| Command | What it does |
| --- | --- |
| `python -m valia fetch` | Downloads the geolocated DVF files (one per year and département), with a local cache |
| `python -m valia build` | Keeps single-dwelling sales, adds leakage-free neighbourhood features, writes `data/clean/sales` |
| `python -m valia baseline` | Measures the reference estimators on the latest full year, writes `reports/baseline.{json,md}` |
| `python -m valia train` | Trains the model, calibrates the 80 % intervals, writes `reports/model.{json,md}` and the model files |
| `python -m valia all` | The four steps in a row |

Options: `--years 2021-2025`, `--departements all` or `75,69,13`, `--test-year 2025`.

## Pipeline

1. **Fetch** — [DVF géolocalisées](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees)
   (Etalab), five rolling years, all départements except Alsace-Moselle and Mayotte, which DVF does not cover.
2. **Clean** — a DVF sale spans several rows. A sale is kept only when it is a plain sale (`Vente`) of
   exactly one apartment or one house, possibly with a garage or cellar, and nothing commercial.
   Surfaces, prices and price per m² must be plausible. Every rule is counted in `reports/funnel.json`.
3. **Features** — for each sale, using earlier sales only: median price per m² over the previous 365 days
   in its ~550 m cell, ~2.2 km cell and commune, for all homes and for the same property type; and the
   10 nearest comparable sales of the same type over the previous 24 months.
4. **Enrichment** — commune population, density, INSEE density grid and level of services
   ("Communes et villes de France", data.gouv.fr), and the distance to the nearest railway station
   (SNCF). Optional by design: if a source is unavailable, its columns stay empty and the model still runs.
5. **Reference estimators** — the commune median and the neighbourhood prior, measured on the latest
   full year. The model must beat the commune median by 25 %.
6. **Model** — gradient boosting (LightGBM) predicting the *gap* to an anchor, the recent price of the
   neighbourhood: trees cannot extrapolate in time, the anchor follows the market without leakage.
   Trained on every year but its last three months, calibrated on those three months (split conformal,
   per segment and property type), evaluated on the latest full year.

## Data and licence

Data: *Demandes de valeurs foncières géolocalisées*, Etalab / data.gouv.fr, under the
[Licence Ouverte 2.0](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/).
DVF covers declared sale prices only: it contains neither the floor, the energy rating nor the
condition of a home, and the model never pretends otherwise.

Code: [MIT](LICENSE) © 2026 John Luke NIKABOU (LucNIK).
