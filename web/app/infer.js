// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

/**
 * Estimation in the browser: a line-by-line mirror of valia/inference.py. The two are checked
 * against each other on reference homes (test/parity.test.js), so any change here goes there too.
 */

import { parcelOf } from "./assets.js";
import { cellOf, departmentOf, EARTH_RADIUS_KM, haversineKm, median } from "./geo.js";
import { explainRow, predictRow } from "./trees.js";

export const MAX_RING = 4;
export const TYPE_LABEL = /** @type {const} */ ({ A: "Appartements", M: "Maisons" });

/**
 * @typedef {import("./assets.js").Assets} Assets
 * @typedef {import("./assets.js").Priors} Priors
 * @typedef {import("./assets.js").Meta} Meta
 * @typedef {object} Query
 * @property {number} lat
 * @property {number} lon
 * @property {"A" | "M"} type
 * @property {number} surface
 * @property {number} rooms
 * @property {number} [land]
 * @property {string} [citycode]
 * @property {string} [parcel]     14-character cadastral id, from the address lookup
 * @property {Record<string, number | null> | null} [building]  user corrections, e.g. {dpe_class: 4}
 * @typedef {object} Sale
 * @property {number} m
 * @property {string} t
 * @property {number} lat
 * @property {number} lon
 * @property {number} surface
 * @property {number} price
 * @property {number} l
 * @property {string} commune
 * @property {number} km
 */

/**
 * Radius around the query fully inside the (2·ring+1)² block of coarse tiles.
 * @param {Meta} meta
 * @param {number} lat
 * @param {number} lon
 * @param {number} row
 * @param {number} col
 * @param {number} ring
 */
export function coveredKm(meta, lat, lon, row, col, ring) {
  const g = meta.grid;
  const h = g.lat * g.coarse;
  const w = g.lon * g.coarse;
  const south = (row - ring) * h;
  const north = (row + ring + 1) * h;
  const west = (col - ring) * w;
  const east = (col + ring + 1) * w;
  const kmLat = (Math.PI * EARTH_RADIUS_KM) / 180;
  const kmLon = kmLat * Math.cos((Math.max(Math.abs(south), Math.abs(north)) * Math.PI) / 180);
  return Math.min((lat - south) * kmLat, (north - lat) * kmLat, (lon - west) * kmLon, (east - lon) * kmLon);
}

/**
 * Same-type sales around the query, nearest first (see neighbours() in inference.py).
 * @param {Assets} assets
 * @param {Query} q
 * @returns {Promise<{found: Sale[], nearestAny: Sale | null, own: import("./assets.js").Tile}>}
 */
export async function neighbours(assets, q) {
  const { k } = assets.meta.knn;
  const [row, col] = cellOf(assets.meta.grid, q.lat, q.lon, assets.meta.grid.coarse);
  /** @type {Sale[]} */
  const found = [];
  /** @type {Sale | null} */
  let nearestAny = null;
  for (let ring = 0; ring <= MAX_RING; ring++) {
    /** @type {[number, number][]} */
    const cells = [];
    for (let dr = -ring; dr <= ring; dr++) {
      for (let dc = -ring; dc <= ring; dc++) {
        if (Math.max(Math.abs(dr), Math.abs(dc)) === ring) cells.push([row + dr, col + dc]);
      }
    }
    const tiles = await Promise.all(cells.map(([r, c]) => assets.tile(r, c)));
    for (const tile of tiles) {
      if (!tile || !tile.sales || !tile.communes) continue;
      const s = tile.sales;
      for (let i = 0; i < s.m.length; i++) {
        const d = haversineKm(q.lat, q.lon, s.la[i], s.lo[i]);
        const sale = { m: s.m[i], t: s.t[i], lat: s.la[i], lon: s.lo[i], surface: s.s[i], price: s.p[i],
                       l: s.l[i], commune: tile.communes[s.c[i]], km: d };
        if (nearestAny === null || d < nearestAny.km) nearestAny = sale;
        if (s.t[i] === q.type) found.push(sale);
      }
    }
    found.sort((a, b) => a.km - b.km);
    if (found.length >= k && found[k - 1].km <= coveredKm(assets.meta, q.lat, q.lon, row, col, ring)) break;
  }
  const own = (await assets.tile(row, col)) || { cells: {}, coarse: {} };
  return { found, nearestAny, own };
}

/**
 * @param {Priors | undefined | null} entry
 * @param {"A" | "M" | "all"} kind
 * @returns {[number, number]}
 */
function prior(entry, kind) {
  const p = entry ? entry[kind] : undefined;
  if (!p) return [NaN, 0];
  return [p[0] === null ? NaN : p[0], p[1]];
}

/** @param {number | null | undefined} v */
const num = (v) => (v === null || v === undefined ? NaN : v);

/**
 * Published building data of the parcel, then the user's corrections on top (see building_of()).
 * @param {Assets} assets @param {Query} q
 */
export async function buildingOf(assets, q) {
  /** @type {Record<string, number | null>} */
  const entry = { ...((q.parcel ? await parcelOf(assets, q.parcel) : null) || {}) };
  for (const [key, value] of Object.entries(q.building || {})) entry[key] = value;
  return entry;
}

/**
 * Model inputs from building data (see building_features() in bdnb.py).
 * @param {Record<string, number | null>} entry
 */
export function buildingFeatures(entry) {
  /** @param {string} key */
  const get = (key) => num(entry[key]);
  const dwellings = get("dwellings");
  return { dpe_class: get("dpe_class"), year_built: get("year_built"), levels: get("levels"),
           dwellings_log: Number.isNaN(dwellings) ? NaN : Math.log1p(dwellings),
           elevator: get("elevator"), social_share: get("social_share") };
}

const ANCHOR_CHAIN = ["knn_prior", "cell_type_prior", "cell_prior", "cell2_type_prior",
                      "commune_type_prior", "commune_prior"];

/**
 * Model inputs for a query, plus context.
 * @param {Assets} assets
 * @param {Query} q
 */
export async function features(assets, q) {
  const meta = assets.meta;
  const { found, nearestAny, own } = await neighbours(assets, q);
  let code = q.citycode || "";
  let entry = code ? await assets.commune(code) : null;
  if (entry === null && nearestAny !== null) {        // e.g. a city code instead of an arrondissement
    code = nearestAny.commune;
    entry = await assets.commune(code);
  }
  const e = entry || /** @type {import("./assets.js").Commune} */ ({});

  const [row, col] = cellOf(meta.grid, q.lat, q.lon);
  const cell = own.cells?.[`${row}:${col}`];
  /** @type {Record<string, number>} */
  const f = {};
  f.is_house = q.type === "M" ? 1.0 : 0.0;
  f.surface_log = Math.log(q.surface);
  f.rooms = q.rooms;
  f.m2_per_room = q.rooms > 0 ? q.surface / q.rooms : NaN;
  f.land_log = Math.log1p(q.type === "M" ? q.land || 0 : 0);
  f.lat = q.lat;
  f.lon = q.lon;
  const asOfMonth = meta.now_month;
  f.months = asOfMonth - 1 - meta.epoch_year * 12;
  /** @type {[string, Priors | undefined, "A" | "M" | "all"][]} */
  const priors = [["cell", cell, "all"], ["commune", e.prior, "all"], ["cell_type", cell, q.type],
                  ["cell2_type", own.coarse, q.type], ["commune_type", e.prior, q.type]];
  for (const [prefix, src, kind] of priors) {
    const [med, n] = prior(src, kind);
    f[`${prefix}_prior`] = med;
    f[`${prefix}_n_log`] = Math.log1p(n);
  }
  const { k } = meta.knn;
  if (found.length >= k) {
    const top = found.slice(0, k);
    f.knn_prior = median(top.map((s) => s.l));
    f.knn_km = median(top.map((s) => s.km));
    f.knn_age = median(top.map((s) => asOfMonth - s.m));
  } else {
    f.knn_prior = f.knn_km = f.knn_age = NaN;
  }
  f.pop_log = num(e.pop_log);
  f.density_log = num(e.density_log);
  f.density_grid = num(e.density_grid);
  f.equipment_level = num(e.equipment_level);
  const building = await buildingOf(assets, q);
  Object.assign(f, buildingFeatures(building));
  const st = assets.stations;
  let station = Infinity;
  for (let i = 0; i < st.length; i += 2) station = Math.min(station, haversineKm(q.lat, q.lon, st[i], st[i + 1]));
  f.station_km = st.length ? station : NaN;

  let anchor = NaN;
  let source = "france";
  for (const name of ANCHOR_CHAIN) {
    if (!Number.isNaN(f[name])) {
      anchor = f[name];
      source = name;
      break;
    }
  }
  if (Number.isNaN(anchor)) {
    const stat = e.static;
    const depStatic = code ? meta.static.dep[departmentOf(code)] : undefined;
    if (stat !== null && stat !== undefined) {
      anchor = stat;
      source = "commune_static";
    } else if (depStatic !== null && depStatic !== undefined) {
      anchor = depStatic;
      source = "dep_static";
    } else {
      anchor = meta.static.france;
    }
  }
  f.anchor = anchor;
  return { features: f, commune: code, segment: e.seg || "Rural", comparables: found.slice(0, 5),
           anchorSource: source, building };
}

/**
 * Price estimate with its 80 % range.
 * @param {Assets} assets
 * @param {Query} q
 */
export async function estimate(assets, q) {
  const ctx = await features(assets, q);
  const f = ctx.features;
  const row = assets.features.map((name) => f[name]);
  const gap = predictRow(assets.forest, row);
  const logPpm2 = f.anchor + gap;
  const group = `${ctx.segment} · ${TYPE_LABEL[q.type]}`;
  const byGroup = assets.intervals.by_group;
  const half = Object.hasOwn(byGroup, group) ? byGroup[group] : assets.intervals.global_half_width_log;
  return {
    ...ctx,
    logPpm2,
    price: Math.exp(logPpm2) * q.surface,
    low: Math.exp(logPpm2 - half) * q.surface,
    high: Math.exp(logPpm2 + half) * q.surface,
    priceM2: Math.exp(logPpm2),
    halfWidth: half,
    explanation: explain(assets, row),
  };
}

/**
 * Why this price (see explain() in inference.py): TreeSHAP contributions of each input to the
 * model's gap, in log price per m², and their sums per group of meta.explain_groups.
 * anchor + base + sum(groups) = log price per m².
 * @param {Assets} assets
 * @param {number[]} row model inputs, in assets.features order
 */
export function explain(assets, row) {
  const { base, contributions: phi } = explainRow(assets.forest, row);
  /** @type {Record<string, number>} */
  const contributions = {};
  assets.features.forEach((name, i) => { contributions[name] = phi[i]; });
  /** @type {Record<string, number>} */
  const groups = {};
  for (const [name, members] of Object.entries(assets.meta.explain_groups)) {
    let sum = 0;
    for (const m of members) if (Object.hasOwn(contributions, m)) sum += contributions[m];
    groups[name] = sum;
  }
  return { base, contributions, groups };
}
