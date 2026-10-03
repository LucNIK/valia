// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

/**
 * Address search with the national address base, served by IGN's Géoplateforme
 * (the successor of api-adresse.data.gouv.fr). Only the typed text is sent.
 */

const ENDPOINT = "https://data.geopf.fr/geocodage/search";

/**
 * @typedef {object} Place
 * @property {string} label     "12 Rue de la Paix 75002 Paris"
 * @property {string} context   "75, Paris, Île-de-France"
 * @property {string} citycode  INSEE code (arrondissement for Paris, Lyon, Marseille)
 * @property {string} type      housenumber | street | locality | municipality
 * @property {number} lat
 * @property {number} lon
 */

/**
 * @param {string} text
 * @param {AbortSignal} [signal]
 * @returns {Promise<Place[]>}
 */
export async function searchAddress(text, signal) {
  const url = new URL(ENDPOINT);
  url.search = new URLSearchParams({ q: text, autocomplete: "1", limit: "6", index: "address" }).toString();
  const r = await fetch(url, { signal });
  if (!r.ok) throw new Error(`géocodage indisponible (HTTP ${r.status})`);
  const body = await r.json();
  /** @type {any[]} */
  const features = Array.isArray(body?.features) ? body.features : [];
  return features
    .filter((f) => Array.isArray(f?.geometry?.coordinates) && f.properties?.citycode)
    .map((f) => ({
      label: String(f.properties.label),
      context: String(f.properties.context || ""),
      citycode: String(f.properties.citycode),
      type: String(f.properties.type || ""),
      lon: Number(f.geometry.coordinates[0]),
      lat: Number(f.geometry.coordinates[1]),
    }));
}
