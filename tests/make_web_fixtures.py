# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Synthetic web assets and reference estimates, for the JavaScript tests in CI.

    python -m tests.make_web_fixtures web/test/fixtures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tests.test_web import build_market, export
from valia.inference import Assets, fixtures, sample_queries


def main(out: str) -> None:
    root = Path(out)
    sales, _, parcels = build_market()
    export(sales, root, parcels)
    rows = fixtures(Assets(root), sample_queries(Assets(root), 300, seed=1))
    (root / "parity.json").write_text(json.dumps(rows, separators=(",", ":")))
    print(f"[fixtures] {len(rows)} reference estimates -> {root}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "web/test/fixtures")
