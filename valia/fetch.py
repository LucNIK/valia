# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""Download the geolocated DVF files, one per year and département, with a local cache."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import DEPARTEMENTS, EXCLUDED_DEPARTEMENTS, RAW_DIR, SOURCE_URL

USER_AGENT = "valia/0.1 (+https://github.com/LucNIK/valia)"


def parse_years(spec: str) -> list[int]:
    """'2021-2025' or '2023,2025' -> [2021, ..., 2025]."""
    years: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            years.update(range(start, end + 1))
        elif part:
            years.add(int(part))
    return sorted(years)


def parse_departements(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(DEPARTEMENTS)
    deps = [d.strip().upper() for d in spec.split(",") if d.strip()]
    deps = [d.zfill(2) if d.isdigit() and len(d) < 2 else d for d in deps]
    excluded = [d for d in deps if d in EXCLUDED_DEPARTEMENTS]
    if excluded:
        raise ValueError(f"DVF does not cover département(s) {', '.join(excluded)}")
    return deps


def raw_path(year: int, dep: str, root: Path = RAW_DIR) -> Path:
    return root / str(year) / f"{dep}.csv.gz"


def fetch(years: list[int], deps: list[str], root: Path = RAW_DIR, retries: int = 3) -> list[Path]:
    """Download missing files; already-cached files are kept. Returns the files available."""
    available = []
    for year in years:
        for dep in deps:
            path = raw_path(year, dep, root)
            if path.exists() and path.stat().st_size > 0:
                available.append(path)
                continue
            url = SOURCE_URL.format(year=year, dep=dep)
            path.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(retries):
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                    with urllib.request.urlopen(request, timeout=120) as resp:
                        tmp = path.with_suffix(".part")
                        tmp.write_bytes(resp.read())
                        tmp.replace(path)
                    available.append(path)
                    print(f"[fetch] {year} {dep:>3}  {path.stat().st_size / 1e6:6.1f} MB")
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code == 404:  # year not published yet for this département
                        print(f"[fetch] {year} {dep:>3}  not published")
                        break
                    if attempt == retries - 1:
                        raise
                except (urllib.error.URLError, TimeoutError):
                    if attempt == retries - 1:
                        raise
                time.sleep(2 * (attempt + 1))
    return available
