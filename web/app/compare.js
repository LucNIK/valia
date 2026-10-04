// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

/**
 * Two estimates side by side: what makes their price per m² differ.
 *
 * Each estimate is log(€/m²) = anchor + base + sum of group contributions (see explain()). The base
 * is the same model constant for both, so the gap between two homes splits exactly into one
 * difference per group; the area's reference level (anchor) is counted with the "area" group.
 */

/**
 * @typedef {object} Explained
 * @property {number} price
 * @property {number} priceM2
 * @property {number} logPpm2
 * @property {Record<string, number>} features
 * @property {{base: number, groups: Record<string, number>}} explanation
 */

/**
 * @param {Explained} a
 * @param {Explained} b
 * @returns {{price: number, priceM2: number, rows: {group: string, diff: number}[]}}
 *   price and priceM2: relative differences of B over A; rows: log differences per group, largest
 *   first, summing to log(B €/m²) − log(A €/m²).
 */
export function compare(a, b) {
  const groups = new Set([...Object.keys(a.explanation.groups), ...Object.keys(b.explanation.groups)]);
  /** @type {{group: string, diff: number}[]} */
  const rows = [];
  for (const group of groups) {
    let diff = (b.explanation.groups[group] || 0) - (a.explanation.groups[group] || 0);
    if (group === "area") {
      diff += b.features.anchor - a.features.anchor + (b.explanation.base - a.explanation.base);
    }
    rows.push({ group, diff });
  }
  rows.sort((x, y) => Math.abs(y.diff) - Math.abs(x.diff));
  return { price: b.price / a.price - 1, priceM2: b.priceM2 / a.priceM2 - 1, rows };
}
