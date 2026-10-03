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

/**
 * Cadastral parcel at a point (reverse geocoding on the parcel index), as the 14-character id
 * used by DVF and the BDNB: département (2) + commune (3) + prefix (3) + section (2) + number (4).
 * @param {number} lat
 * @param {number} lon
 * @param {AbortSignal} [signal]
 * @returns {Promise<string | null>}
 */
export async function parcelAt(lat, lon, signal) {
  const url = new URL(ENDPOINT.replace(/search$/, "reverse"));
  url.search = new URLSearchParams({ lat: String(lat), lon: String(lon), index: "parcel", limit: "1" }).toString();
  const r = await fetch(url, { signal });
  if (!r.ok) return null;
  const body = await r.json();
  const p = body?.features?.[0]?.properties;
  return p ? parcelId(p) : null;
}

/**
 * @param {Record<string, any>} p properties of a parcel feature
 * @returns {string | null}
 */
export function parcelId(p) {
  if (typeof p.id === "string" && /^[0-9A-Z]{14}$/.test(p.id)) return p.id;
  const dep = String(p.departmentcode ?? "");
  const com = String(p.municipalitycode ?? "");
  const prefix = String(p.oldmunicipalitycode || p.districtcode || "000");
  const section = String(p.section ?? "");
  const number = String(p.number ?? "");
  if (!dep || !com || !section || !number) return null;
  const id = dep.padStart(2, "0") + com.padStart(3, "0") + prefix.padStart(3, "0") + section.padStart(2, "0") +
    number.padStart(4, "0");
  return id.length === 14 ? id.toUpperCase() : null;
}
