// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

import assert from "node:assert/strict";
import { test } from "node:test";

import { distance, month, perM2, price } from "../app/format.js";

test("formatting", () => {
  const nb = (/** @type {string} */ s) => s.replace(/\s/g, " ");
  assert.equal(nb(price(412_345)), "412 000 €");
  assert.equal(nb(price(87_654)), "87 700 €");
  assert.equal(nb(perM2(6_648)), "6 650 €");
  assert.equal(month(2025 * 12 + 3), "mars 2025");
  assert.equal(month(2024 * 12 + 12), "déc. 2024");
  assert.equal(distance(0.163), "160 m");
  assert.equal(distance(2.345), "2,3 km");
});
