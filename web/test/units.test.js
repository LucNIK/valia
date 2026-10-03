// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

import assert from "node:assert/strict";
import { test } from "node:test";

import { cellOf, departmentOf, haversineKm, median } from "../app/geo.js";
import { predictRow, readForest } from "../app/trees.js";

test("median and distances", () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 3, 2]), 2.5);
  assert.ok(Number.isNaN(median([])));
  assert.ok(Math.abs(haversineKm(48.8566, 2.3522, 45.764, 4.8357) - 392) < 5);
  assert.equal(departmentOf("2A004"), "2A");
  assert.equal(departmentOf("97411"), "974");
  assert.deepEqual(cellOf({ lat: 0.005, lon: 0.007, coarse: 4 }, -21.1, 55.5), [-4220, 7928]);
});

/** One tree: x0 <= 1.5 ? (x1 <= 0, NaN goes right ? 20 : 30) : 10, plus a single-leaf tree of 0.5. */
function tinyForest() {
  const nTrees = 2, nNodes = 2, nLeaves = 4;
  const buf = new ArrayBuffer(24 + 12 * nTrees + nNodes * (1 + 1 + 8 + 4 + 4) + 8 * nLeaves);
  const v = new DataView(buf);
  "VTREES01".split("").forEach((ch, i) => v.setUint8(i, ch.charCodeAt(0)));
  [2, nTrees, nNodes, nLeaves].forEach((x, i) => v.setUint32(8 + 4 * i, x, true));
  let o = 24;
  /**
   * @param {(o: number, x: number) => void} put
   * @param {number} size
   * @param {number[]} xs
   */
  const w = (put, size, xs) => xs.forEach((x) => { put(o, x); o += size; });
  const i32 = (/** @type {number} */ p, /** @type {number} */ x) => v.setInt32(p, x, true);
  w(i32, 4, [0, 2]);            // node_start
  w(i32, 4, [0, 3]);            // leaf_start
  w(i32, 4, [2, 0]);            // nodes per tree
  w((p, x) => v.setUint8(p, x), 1, [0, 1]);       // feature
  w((p, x) => v.setUint8(p, x), 1, [2, 8]);       // decision: default left / NaN missing, default right
  w((p, x) => v.setFloat64(p, x, true), 8, [1.5, 1e-39]);
  w(i32, 4, [1, ~1]);           // left
  w(i32, 4, [~0, ~2]);          // right
  w((p, x) => v.setFloat64(p, x, true), 8, [10, 20, 30, 0.5]);
  return readForest(buf);
}

test("tree decisions follow LightGBM", () => {
  const f = tinyForest();
  assert.equal(predictRow(f, [2, 0]), 10.5);
  assert.equal(predictRow(f, [1, -5]), 20.5);
  assert.equal(predictRow(f, [1, 5]), 30.5);
  assert.equal(predictRow(f, [NaN, 5]), 30.5);      // NaN with missing type none -> 0.0 -> left
  assert.equal(predictRow(f, [1, NaN]), 30.5);      // NaN missing type -> default right
  assert.throws(() => predictRow(f, [1]));
});

test("parcel ids from the address service", async () => {
  const { parcelId } = await import("../app/geocode.js");
  assert.equal(parcelId({ id: "75104000AB0012" }), "75104000AB0012");
  assert.equal(parcelId({ departmentcode: "44", municipalitycode: "109", section: "EX", number: "8" }), "44109000EX0008");
  assert.equal(parcelId({ departmentcode: "01", municipalitycode: "4", oldmunicipalitycode: "12", section: "A", number: "123" }),
               "010040120A0123");
  assert.equal(parcelId({ section: "A" }), null);
});
