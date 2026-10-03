// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

/** Runs theme.js in a fake page and returns its rule plus the resulting <html> attributes. */
function load(stored = /** @type {string | null} */ (null), prefersDark = false) {
  /** @type {Record<string, string>} */
  const attrs = {};
  /** @type {Record<string, any>} */
  const window = {
    matchMedia: () => ({ matches: prefersDark, addEventListener() {} }),
  };
  const context = {
    window,
    document: {
      documentElement: { setAttribute: (/** @type {string} */ k, /** @type {string} */ v) => { attrs[k] = v; } },
      querySelectorAll: () => [],
      getElementById: () => null,
      addEventListener() {},
      hidden: false,
    },
    localStorage: { getItem: () => stored, setItem() {} },
    setInterval() {},
    Date,
  };
  vm.runInNewContext(readFileSync(new URL("../app/theme.js", import.meta.url), "utf8"), context);
  return { resolve: window.valiaTheme.resolve, attrs };
}

test("auto: dark when the browser asks for it or at night", () => {
  const { resolve } = load();
  assert.equal(resolve("auto", false, 12), "light");
  assert.equal(resolve("auto", false, 6), "dark");
  assert.equal(resolve("auto", false, 7), "light");
  assert.equal(resolve("auto", false, 19), "light");
  assert.equal(resolve("auto", false, 20), "dark");
  assert.equal(resolve("auto", true, 12), "dark");
});

test("manual modes win over the browser and the clock", () => {
  const { resolve } = load();
  assert.equal(resolve("light", true, 23), "light");
  assert.equal(resolve("dark", false, 12), "dark");
});

test("stored mode is applied, and an unknown value falls back to auto", () => {
  assert.equal(load("dark").attrs["data-theme"], "dark");
  assert.equal(load("dark").attrs["data-theme-mode"], "dark");
  assert.equal(load("garbage").attrs["data-theme-mode"], "auto");
  assert.equal(load(null, true).attrs["data-theme"], "dark");
});
