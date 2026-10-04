// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

/**
 * The browser must reproduce the Python reference (valia/inference.py) on every reference home.
 *
 *   python -m tests.make_web_fixtures web/test/fixtures      # synthetic assets (CI)
 *   python -m valia parity --assets web --out web/test/fixtures/parity.json   # real assets
 *   VALIA_ASSETS=web node --test web/test/
 */

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { Assets } from "../app/assets.js";
import { estimate } from "../app/infer.js";
import { fsLoader } from "./fs-loader.js";

const here = dirname(fileURLToPath(import.meta.url));
const fixtures = process.env.VALIA_FIXTURES || join(here, "fixtures");
const assetsRoot = process.env.VALIA_ASSETS || fixtures;
const parityFile = join(fixtures, "parity.json");
const REL = 1e-9;

/**
 * @param {number | null} want
 * @param {number} got
 * @param {string} what
 */
function close(want, got, what) {
  if (want === null) {
    assert.ok(Number.isNaN(got), `${what}: expected NaN, got ${got}`);
    return;
  }
  const tol = REL * Math.max(1, Math.abs(want));
  assert.ok(Math.abs(want - got) <= tol, `${what}: expected ${want}, got ${got}`);
}

test("browser estimates match the Python reference", { skip: !existsSync(parityFile) && "no fixtures" }, async () => {
  const rows = JSON.parse(readFileSync(parityFile, "utf8"));
  const assets = await Assets.open(fsLoader(assetsRoot));
  assert.ok(rows.length > 0);
  let checked = 0;
  let explained = 0;
  for (const row of rows) {
    const q = row.query;
    const got = await estimate(assets, { ...q, citycode: q.citycode });
    const where = `query ${JSON.stringify(q)}`;
    for (const [name, want] of Object.entries(row.features)) close(want, got.features[name], `${where} ${name}`);
    close(row.log_ppm2, got.logPpm2, `${where} log_ppm2`);
    close(row.low, got.low, `${where} low`);
    close(row.high, got.high, `${where} high`);
    assert.equal(got.commune, row.commune, `${where} commune`);
    assert.equal(got.segment, row.segment, `${where} segment`);
    assert.equal(got.anchorSource, row.anchor_source, `${where} anchor`);
    for (const [name, want] of Object.entries(row.building)) {
      if (want === null) assert.ok(got.building[name] === null || Number.isNaN(got.building[name]), `${where} ${name}`);
      else close(/** @type {number} */ (want), /** @type {number} */ (got.building[name]), `${where} building ${name}`);
    }
    if (row.trend === null) assert.equal(got.trend, null, `${where} trend`);
    else {
      const t = /** @type {NonNullable<typeof got.trend>} */ (got.trend);
      assert.equal(t.scope, row.trend.scope, `${where} trend scope`);
      assert.equal(t.kind, row.trend.kind, `${where} trend kind`);
      assert.deepEqual(t.values, row.trend.values, `${where} trend values`);
      for (const [a, b] of [[row.trend.year_change, t.yearChange], [row.trend.total_change, t.totalChange]]) {
        if (a === null) assert.equal(b, null, `${where} trend change`);
        else close(a, /** @type {number} */ (b), `${where} trend change`);
      }
    }
    if (row.explanation) {
      close(row.explanation.base, got.explanation.base, `${where} base`);
      for (const [name, want] of Object.entries(row.explanation.contributions)) {
        close(/** @type {number} */ (want), got.explanation.contributions[name], `${where} contribution ${name}`);
      }
      for (const [name, want] of Object.entries(row.explanation.groups)) {
        close(/** @type {number} */ (want), got.explanation.groups[name], `${where} group ${name}`);
      }
      explained++;
    }
    assert.deepEqual(got.comparables.map((c) => [c.lat, c.lon, c.m, c.price]), row.comparables, `${where} comparables`);
    checked++;
  }
  assert.equal(checked, rows.length);
  assert.ok(explained > 0, "no reference explanation in the fixtures");
});
