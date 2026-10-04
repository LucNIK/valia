// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { Assets } from "../app/assets.js";
import { comparablePoints, darken, priceCells, quantileBreaks } from "../app/mapdata.js";
import { fsLoader } from "./fs-loader.js";

const here = dirname(fileURLToPath(import.meta.url));
const root = process.env.VALIA_ASSETS || join(here, "fixtures");

test("quantile breaks", () => {
  assert.deepEqual(quantileBreaks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), [3, 5, 7, 9]);
  assert.deepEqual(quantileBreaks([5, 5, 5, 5]), [5]);
  assert.deepEqual(quantileBreaks([]), []);
});

test("dark map style", () => {
  const style = { version: 8, layers: [
    { id: "bg", paint: { "fill-color": "#FFFFFF" } },
    { id: "road", paint: { "line-color": ["case", true, "#888", "#000000"], "line-width": 2 } },
    { id: "label", paint: { "text-color": "#222222", "text-halo-color": "#fff" } }] };
  const dark = darken(style);
  assert.equal(dark.layers[0].paint["fill-color"], "#0d0c0f");             // white becomes the background
  const [bg, road, label] = dark.layers.map((/** @type {any} */ l) => l.paint);
  const L = (/** @type {string} */ h) => parseInt(h.slice(1, 3), 16);
  assert.ok(L(label["text-color"]) > L(label["text-halo-color"]) + 100, "labels stay readable");
  assert.equal(road["line-width"], 2);
  assert.equal(road["line-color"][0], "case");
  assert.equal(bg["fill-color"].length, 7);
  assert.equal(style.layers[0].paint["fill-color"], "#FFFFFF", "the input is not modified");
});

test("price squares and comparables around a home", { skip: !existsSync(join(root, "data", "meta.json")) && "no fixtures" },
  async () => {
    const assets = await Assets.open(fsLoader(root));
    const { cells, kind } = await priceCells(assets, 48.857, 2.352, "A");
    assert.ok(cells.features.length >= 5);
    assert.equal(kind, "A");
    for (const f of cells.features) {
      assert.equal(f.geometry.coordinates[0].length, 5);
      assert.ok(f.properties.ppm2 > 1000 && f.properties.sales >= assets.meta.prior_min_sales);
    }
    const ring = cells.features[0].geometry.coordinates[0];
    assert.ok(Math.abs(ring[2][1] - ring[0][1] - assets.meta.grid.lat) < 1e-9);
    const points = comparablePoints([{ lat: 48.86, lon: 2.35, price: 300000, surface: 30, m: 24300, km: 0.2 }]);
    assert.deepEqual(points.features[0].properties, { n: "1", price: 300000, ppm2: 10000, m: 24300, km: 0.2 });
  });
