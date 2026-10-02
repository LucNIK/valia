<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 John Luke NIKABOU (LucNIK) -->

# Contributing

## Commit policy

| Rule | What it requires |
| --- | --- |
| R1 — Maintainer commits | Every commit on `main` is written and signed by the maintainer, `LucNIK <lukepulanka@gmail.com>`. |
| R2 — Single author | Commit messages carry no co-author or tool trailers. |
| R3 — Signed | Every commit and tag is signed (SSH) and shows **Verified**. |
| R4 — Tests first | `pytest` and `ruff` are green before any tag. |
| R5 — No leakage | A feature may only use information available before the sale it describes; tests prove it. |

R1 and R2 are enforced locally by the hooks in `.githooks/`. Enable them once per clone:

```bash
git config core.hooksPath .githooks
```

## Development

```bash
python -m pip install -e ".[dev]"
ruff check . && pytest -q
python -m valia all --years 2023-2025 --departements 75   # one département, three years
```
