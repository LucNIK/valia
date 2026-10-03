// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

/** Small numeric helpers, identical to valia/inference.py. */

export const EARTH_RADIUS_KM = 6371.0;

/** @param {number} deg */
const rad = (deg) => (deg * Math.PI) / 180;

/**
 * Great-circle distance in km.
 * @param {number} lat1
 * @param {number} lon1
 * @param {number} lat2
 * @param {number} lon2
 */
export function haversineKm(lat1, lon1, lat2, lon2) {
  const p1 = rad(lat1);
  const p2 = rad(lat2);
  const dp = p2 - p1;
  const dl = rad(lon2) - rad(lon1);
  const h = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2 * Math.asin(Math.min(1, Math.sqrt(h))) * EARTH_RADIUS_KM;
}

/**
 * Median; the mean of the two middle values for an even count, NaN when empty.
 * @param {number[]} values
 */
export function median(values) {
  const s = [...values].sort((a, b) => a - b);
  const n = s.length;
  if (n === 0) return NaN;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

/** @param {string} code INSEE commune code */
export function departmentOf(code) {
  return code.startsWith("97") ? code.slice(0, 3) : code.slice(0, 2);
}

/** @typedef {{lat: number, lon: number, coarse: number}} Grid */

/**
 * Grid cell of a position (factor 1: ~550 m cells; the coarse factor: ~2.2 km tiles).
 * @param {Grid} g
 * @param {number} lat
 * @param {number} lon
 * @param {number} [factor]
 * @returns {[number, number]}
 */
export function cellOf(g, lat, lon, factor = 1) {
  return [Math.floor(lat / (g.lat * factor)), Math.floor(lon / (g.lon * factor))];
}
