// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

/**
 * What the neighbourhood map draws, as plain GeoJSON: no map library here, so it is tested in Node.
 *
 *  - price squares: the ~550 m cells of the 3×3 tiles around the home, coloured by the median price
 *    per m² of the last 12 months (only cells with at least `prior_min_sales` sales are published);
 *  - the comparable sales, numbered like the table;
 *  - colours: a single-hue ramp, light to dark, with breaks at the quantiles of the visible cells.
 */

import { cellOf } from "./geo.js";

/** @typedef {{type: "Feature", geometry: any, properties: Record<string, any>}} Feature */
/** @typedef {{type: "FeatureCollection", features: Feature[]}} Collection */

export const MIN_CELLS_FOR_TYPE = 5;

/**
 * Price squares around a point.
 * @param {import("./assets.js").Assets} assets
 * @param {number} lat @param {number} lon
 * @param {"A" | "M"} kind
 * @returns {Promise<{cells: Collection, kind: "A" | "M" | "all"}>}
 */
export async function priceCells(assets, lat, lon, kind) {
  const g = assets.meta.grid;
  const [row, col] = cellOf(g, lat, lon, g.coarse);
  /** @type {Record<string, import("./assets.js").Priors>} */
  const cells = {};
  const tiles = [];
  for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) tiles.push(assets.tile(row + dr, col + dc));
  for (const tile of await Promise.all(tiles)) Object.assign(cells, tile?.cells || {});

  const priced = (/** @type {"A" | "M" | "all"} */ k) =>
    Object.entries(cells).filter(([, p]) => p[k] && p[k][0] !== null);
  /** @type {"A" | "M" | "all"} */
  const use = priced(kind).length >= MIN_CELLS_FOR_TYPE ? kind : "all";
  /** @type {Feature[]} */
  const features = priced(use).map(([key, p]) => {
    const [r, c] = key.split(":").map(Number);
    const [s, n, w, e] = [r * g.lat, (r + 1) * g.lat, c * g.lon, (c + 1) * g.lon];
    const [median, sales] = /** @type {[number, number]} */ (p[use]);
    return {
      type: "Feature",
      geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] },
      properties: { key, ppm2: Math.round(Math.exp(median)), sales },
    };
  });
  return { cells: { type: "FeatureCollection", features }, kind: use };
}

/**
 * Class breaks at the quantiles: `k` classes, `k − 1` increasing breaks.
 * @param {number[]} values
 * @param {number} [k]
 */
export function quantileBreaks(values, k = 5) {
  const s = [...values].sort((a, b) => a - b);
  if (s.length === 0) return [];
  const out = [];
  for (let i = 1; i < k; i++) {
    const v = s[Math.min(s.length - 1, Math.floor((i * s.length) / k))];
    if (out.length === 0 || v > out[out.length - 1]) out.push(v);
  }
  return out;
}

/**
 * @param {{lat: number, lon: number, price: number, surface: number, m: number, km: number}[]} comps
 * @returns {Collection}
 */
export function comparablePoints(comps) {
  return {
    type: "FeatureCollection",
    features: comps.map((c, i) => ({
      type: "Feature",
      geometry: { type: "Point", coordinates: [c.lon, c.lat] },
      properties: { n: String(i + 1), price: c.price, ppm2: Math.round(c.price / c.surface), m: c.m, km: c.km },
    })),
  };
}

/**
 * A light map style turned dark: every grey or colour keeps its hue and has its lightness flipped
 * and compressed towards the page's dark background, so roads and labels stay readable.
 * @param {any} style MapLibre style JSON (colours as #rgb / #rrggbb, as in the IGN styles)
 * @param {[number, number, number]} [background] darkest value (RGB)
 */
export function darken(style, background = [13, 12, 15]) {
  /** @param {string} hex */
  const flip = (hex) => {
    const h = hex.length === 4 ? hex.replace(/[0-9a-f]/gi, (d) => d + d) : hex;
    const rgb = [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
    const lum = (Math.max(...rgb) + Math.min(...rgb)) / 2 / 255;
    const target = 1 - lum;                                         // light becomes dark
    const out = rgb.map((v, i) => {
      const tint = (v / 255 - lum) * 0.6;                           // keep a little of the hue
      const value = background[i] / 255 + (target * 0.8 + tint) * (1 - background[i] / 255);
      return Math.round(Math.max(0, Math.min(1, value)) * 255);
    });
    return `#${out.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
  };
  /** @param {any} v @returns {any} */
  const walk = (v) => {
    if (typeof v === "string") return /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(v) ? flip(v) : v;
    if (Array.isArray(v)) return v.map(walk);
    if (v && typeof v === "object") return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, walk(x)]));
    return v;
  };
  return { ...style, layers: style.layers.map((/** @type {any} */ layer) => ({ ...layer, paint: walk(layer.paint) })) };
}
