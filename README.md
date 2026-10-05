<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 John Luke NIKABOU (LucNIK) -->

<div align="center">

# Valia

**What is this home worth — and how sure are we?**

Property price estimation for the whole of France, built on the public DVF sales records.
A price, a calibrated range, the reasons behind it and the closest real sales — computed in the browser, with no server.

`v0.4 · 4.26 M sales · web app + buildings` · [Live app](https://lucnik.github.io/valia/) · [Model card](docs/model-card.md) · MIT

</div>

---

## Status

| Version | Scope | Exit criterion | State |
| --- | --- | --- | --- |
| **v0.1** | DVF pipeline: download, cleaning, leakage-free neighbourhood features, reference estimators | Reproducible dataset in one command, tests green | **Done** — 4.26 M sales, see the [model card](docs/model-card.md) |
| **v0.2** | Open-data enrichment, gradient-boosted model anchored on comparable sales, conformal intervals per segment | Beat both references in every segment; 80 % intervals cover 78–82 % of prices (target −25 % vs commune median) | **Done** — −18.9 % error vs commune median, better in every segment, 81.3 % coverage ✅ |
| **v0.3** | Web app: address, estimate, range, 5 comparable sales, Method page, dark and light themes | Browser and Python agree on 1 000 homes | **Done** — browser = Python on 1,000 real homes |
| **v0.4** | Building data (BDNB): energy class, construction year, storeys, dwellings — pre-filled in the app and correctable | Close the gap to −25 % vs commune median | **Done** — −26.7 % error vs commune median (web model −25.6 %), 13.9 % median error, 81.5 % coverage ✅ |
| v1.0 | Why this price (TreeSHAP, in the browser) · local trend · comparison · neighbourhood map · immersive 3D map · installable offline app | Public demo and model card | **In progress** — explanations, local trend, comparison and neighbourhood map done |
| v1.x | Accounts, saved homes, history, price alerts and notifications | — | Later |

## Why it is different

- **Measured honestly.** Every number is computed on sales that happened *after* the training data —
  never on a shuffled split. Results are reported separately for Paris, large cities, mid-size cities
  and rural France, because a national average hides where a model fails.
- **No leakage, by construction.** Neighbourhood features only use sales that happened strictly before
  the one being described, over the previous 365 days. Tests prove it.
- **Uncertainty you can see.** Every estimate comes with a range calibrated by conformal
  prediction, and its real coverage is published.
- **Explained.** Every estimate is broken down with TreeSHAP — reference level of the area, then the
  exact effect of surface, building, land, location and date — computed in the browser.
- **No server.** The app is static: the model runs in the browser, and only the typed address leaves it,
  to the national address service (IGN). The JavaScript is checked against the Python reference.
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
| `python -m valia train --profile web` | Trains the compact model shipped in the browser, writes `reports/web/` |
| `python -m valia export` | Writes the app's static data to `web/data/` (priors, recent sales rounded to ~100 m, model) |
| `python -m valia parity` | Reference estimates on 1,000 homes, which the browser must reproduce (`cd web && npm test`) |
| `python -m valia all` | Fetch, build, baseline and train in a row |

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
5. **Buildings** — the [BDNB](https://bdnb.io) (CSTB) links each cadastral parcel to its buildings: energy
   class of a representative dwelling, construction year, storeys and number of dwellings.
   Joined to each sale by its parcel; a DPE made after a sale is never used for that sale.
6. **Reference estimators** — the commune median and the neighbourhood prior, measured on the latest
   full year. The model must beat the commune median by 25 %.
7. **Model** — gradient boosting (LightGBM) predicting the *gap* to an anchor, the recent price of the
   neighbourhood: trees cannot extrapolate in time, the anchor follows the market without leakage.
   Trained on every year but its last three months, calibrated on those three months (split conformal,
   per segment and property type), evaluated on the latest full year.

## Web app

`web/` is a static site with no build step: ES modules typed with JSDoc and checked by `tsc --strict`,
IBM Plex, gold by day and bordeaux by night. To run it locally after `train --profile web` and `export`:

```bash
cd web && python -m http.server 8000   # http://localhost:8000
```

The Data workflow rebuilds everything after each DVF release and publishes the site to GitHub Pages.
The Site workflow republishes the app alone in a couple of minutes, with the data of the last Data run.
Fonts and MapLibre GL JS are served by the site itself (`scripts/vendor-web.sh`, pinned versions).

## Data and licence

Data: *Demandes de valeurs foncières géolocalisées*, Etalab / data.gouv.fr, under the
[Licence Ouverte 2.0](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/).
DVF covers declared sale prices only: it contains neither the floor, the energy rating nor the
condition of a home, and the model never pretends otherwise.

Building data: *Base de données nationale des bâtiments* (BDNB), CSTB, under the
[ODbL](https://opendatacommons.org/licenses/odbl/1-0/). The building files the app publishes
(`data/parcels/`) are derived from it and shared under the same licence.

Code: [MIT](LICENSE) © 2026 John Luke NIKABOU (LucNIK).
