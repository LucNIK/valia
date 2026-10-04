// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

import { departmentOf } from "./geo.js";
import { readForest } from "./trees.js";

/**
 * Reads files under the assets root. The browser uses fetch(); the tests read from disk.
 * `json` resolves to null when the file does not exist.
 * @typedef {object} Loader
 * @property {(path: string) => Promise<any>} json
 * @property {(path: string) => Promise<ArrayBuffer>} binary
 */

/**
 * @typedef {[number | null, number]} Prior            [median log €/m² or null, sales]
 * @typedef {{A?: Prior, M?: Prior, all?: Prior}} Priors
 * @typedef {object} Commune
 * @property {string} seg
 * @property {number | null} static
 * @property {Priors} prior
 * @property {number | null} [pop_log]
 * @property {number | null} [density_log]
 * @property {number | null} [density_grid]
 * @property {number | null} [equipment_level]
 * @property {Record<string, (number | null)[]>} [trend]
 * @typedef {{deps: Record<string, Record<string, (number | null)[]>>, france: Record<string, (number | null)[]>}} Trends
 * @typedef {object} TileSales
 * @property {number[]} m   month index (year * 12 + month)
 * @property {string[]} t   "A" | "M"
 * @property {number[]} la
 * @property {number[]} lo
 * @property {number[]} s   surface m²
 * @property {number[]} p   price €
 * @property {number[]} l   log price per m²
 * @property {number[]} c   index into communes
 * @typedef {object} Tile
 * @property {Record<string, Priors>} cells
 * @property {Priors} coarse
 * @property {string[]} [communes]
 * @property {TileSales} [sales]
 * @typedef {object} Meta
 * @property {string} format
 * @property {string} version
 * @property {string} data_through
 * @property {string} as_of
 * @property {number} now_month
 * @property {number} epoch_year
 * @property {{lat: number, lon: number, coarse: number}} grid
 * @property {{k: number, lookback_months: number}} knn
 * @property {{dep: Record<string, number | null>, france: number}} static
 * @property {Record<string, any>} accuracy
 * @property {string} source
 * @property {string[]} parcel_fields
 * @property {Record<string, string[]>} explain_groups
 * @property {{ends: string[], window_months: number, min_sales: number}} trend
 * @property {string | null} [buildings_source]
 * @typedef {{global_half_width_log: number, by_group: Record<string, number>}} Intervals
 */

export class Assets {
  /**
   * Use `Assets.open(loader)`.
   * @param {Loader} loader
   * @param {Meta} meta
   * @param {Intervals} intervals
   * @param {number[]} stations flat [lat, lon, lat, lon, ...]
   * @param {Record<string, number[]>} index existing tiles, {row: [col, ...]}
   * @param {import("./trees.js").Forest} forest
   * @param {string[]} features model inputs, in order
   * @param {string[]} parcelCodes communes with published building data
   * @param {Trends} trends price series per département and for France
   */
  constructor(loader, meta, intervals, stations, index, forest, features, parcelCodes, trends) {
    this.loader = loader;
    this.meta = meta;
    this.intervals = intervals;
    this.stations = stations;
    this.index = new Map(Object.entries(index).map(([r, cols]) => [Number(r), new Set(cols)]));
    this.forest = forest;
    this.features = features;
    this.parcelCodes = new Set(parcelCodes);
    this.trends = trends;
    /** @type {Map<string, Promise<Record<string, (number | null)[]>>>} */
    this.parcels = new Map();
    /** @type {Map<string, Promise<Tile | null>>} */
    this.tiles = new Map();
    /** @type {Map<string, Promise<Record<string, Commune>>>} */
    this.communes = new Map();
  }

  /** @param {Loader} loader */
  static async open(loader) {
    const [meta, intervals, stations, index, model, header, parcelCodes, trends] = await Promise.all([
      loader.json("data/meta.json"), loader.json("data/intervals.json"), loader.json("data/stations.json"),
      loader.json("data/tiles/index.json"), loader.binary("data/model.bin"), loader.json("data/model.json"),
      loader.json("data/parcels/index.json"), loader.json("data/trends.json"),
    ]);
    if (!meta || meta.format !== "valia-web/2") throw new Error("unsupported asset format");
    const forest = readForest(model);
    if (header.features.length !== forest.nFeatures) throw new Error("model header does not match the trees");
    return new Assets(loader, meta, intervals, stations || [], index || {}, forest, header.features, parcelCodes || [],
                      trends || { deps: {}, france: {} });
  }

  /**
   * @param {number} row
   * @param {number} col
   * @returns {Promise<Tile | null>}
   */
  tile(row, col) {
    if (!this.index.get(row)?.has(col)) return Promise.resolve(null);
    const key = `${row}_${col}`;
    let p = this.tiles.get(key);
    if (!p) {
      p = this.loader.json(`data/tiles/${key}.json`);
      this.tiles.set(key, p);
    }
    return p;
  }

  /**
   * @param {string} code
   * @returns {Promise<Commune | null>}
   */
  async commune(code) {
    const dep = departmentOf(code);
    let p = this.communes.get(dep);
    if (!p) {
      p = this.loader.json(`data/communes/${dep}.json`).then((x) => x || {});
      this.communes.set(dep, p);
    }
    const all = await p;
    return Object.hasOwn(all, code) ? all[code] : null;
  }
}

/**
 * Building data of a cadastral parcel, as {field: value}, or null when unknown.
 * @param {Assets} assets
 * @param {string} parcelId 14 characters
 * @returns {Promise<Record<string, number | null> | null>}
 */
export async function parcelOf(assets, parcelId) {
  const code = parcelId.slice(0, 5);
  if (parcelId.length !== 14 || !assets.parcelCodes.has(code)) return null;
  let p = assets.parcels.get(code);
  if (!p) {
    p = assets.loader.json(`data/parcels/${code}.json`).then((x) => x || {});
    assets.parcels.set(code, p);
  }
  const all = await p;
  const key = parcelId.slice(5);
  if (!Object.hasOwn(all, key)) return null;
  const values = all[key];
  /** @type {Record<string, number | null>} */
  const entry = {};
  assets.meta.parcel_fields.forEach((name, i) => { entry[name] = values[i]; });
  return entry;
}

/**
 * Loader over HTTP, relative to `base` (the app's URL).
 * @param {string | URL} base
 * @returns {Loader}
 */
export function httpLoader(base) {
  return {
    async json(path) {
      const r = await fetch(new URL(path, base));
      if (r.status === 404) return null;
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return r.json();
    },
    async binary(path) {
      const r = await fetch(new URL(path, base));
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return r.arrayBuffer();
    },
  };
}
