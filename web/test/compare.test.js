// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { Assets } from "../app/assets.js";
import { compare } from "../app/compare.js";
import { estimate } from "../app/infer.js";
import { fsLoader } from "./fs-loader.js";

const here = dirname(fileURLToPath(import.meta.url));
const root = process.env.VALIA_ASSETS || join(here, "fixtures");

test("the per-group differences add up to the gap in price per m²",
  { skip: !existsSync(join(root, "data", "meta.json")) && "no fixtures" }, async () => {
    const assets = await Assets.open(fsLoader(root));
    const a = await estimate(assets, { lat: 48.857, lon: 2.352, type: "A", surface: 62, rooms: 3, citycode: "75056" });
    const b = await estimate(assets, { lat: 46.17, lon: 1.87, type: "M", surface: 110, rooms: 5, land: 900,
                                       citycode: "23096", building: { dpe_class: 6, year_built: 1955 } });
    const c = compare(a, b);
    const sum = c.rows.reduce((s, r) => s + r.diff, 0);
    assert.ok(Math.abs(sum - (b.logPpm2 - a.logPpm2)) < 1e-9);
    assert.ok(Math.abs(c.priceM2 - (b.priceM2 / a.priceM2 - 1)) < 1e-12);
    assert.ok(c.price < 0, "a house in Guéret costs less than a flat in Paris");
    for (let i = 1; i < c.rows.length; i++) assert.ok(Math.abs(c.rows[i - 1].diff) >= Math.abs(c.rows[i].diff));
    assert.deepEqual(compare(a, a).rows.map((r) => r.diff), c.rows.map(() => 0));
  });
