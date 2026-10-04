// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

/**
 * Neighbourhood map: IGN's grey Plan (vector tiles, no key), the price squares around the home, the
 * home and its comparable sales. MapLibre GL JS is served by the site itself (vendor/) and loaded
 * only when a map is first shown. Dark theme: the same style with its colours flipped (darken()).
 */

import * as fmt from "./format.js";
import { comparablePoints, darken, priceCells, quantileBreaks } from "./mapdata.js";

const STYLE_URL = "https://data.geopf.fr/annexes/ressources/vectorTiles/styles/PLAN.IGN/gris.json";
// Single-hue ramps, light to dark (gold by day, bordeaux by night), as the rest of the site.
const RAMP = {
  light: ["#f4e6b8", "#e6c868", "#c9a227", "#946f00", "#5c4500"],
  dark: ["#3d1a24", "#6b2a3c", "#9a3b53", "#cf5770", "#f293a8"],
};
const LOCALE = {
  "NavigationControl.ZoomIn": "Zoomer",
  "NavigationControl.ZoomOut": "Dézoomer",
  "NavigationControl.ResetBearing": "Réorienter vers le nord",
  "CooperativeGesturesHandler.WindowsHelpText": "Ctrl + molette pour zoomer sur la carte",
  "CooperativeGesturesHandler.MacHelpText": "⌘ + molette pour zoomer sur la carte",
  "CooperativeGesturesHandler.MobileHelpText": "Deux doigts pour déplacer la carte",
};

/** @type {Promise<any> | null} */
let library = null;
/** @type {any} */
let baseStyle = null;
/** @type {any} */
let map = null;
/** @type {{cells: any, comps: any, home: any, breaks: number[]} | null} */
let overlay = null;

function loadLibrary() {
  if (library) return library;
  library = new Promise((resolve, reject) => {
    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = "vendor/maplibre-gl.css";
    document.head.append(css);
    const script = document.createElement("script");
    script.src = "vendor/maplibre-gl.js";
    script.onload = () => resolve(/** @type {any} */ (window).maplibregl);
    script.onerror = () => reject(new Error("MapLibre could not be loaded"));
    document.head.append(script);
  });
  return library;
}

const theme = () => (document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light");
/** @param {string} name */
const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

async function styleFor(/** @type {"light" | "dark"} */ mode) {
  if (!baseStyle) {
    const r = await fetch(STYLE_URL);
    if (!r.ok) throw new Error(`map style: HTTP ${r.status}`);
    baseStyle = await r.json();
  }
  return mode === "dark" ? darken(baseStyle) : baseStyle;
}

/** The first text font of the base style, for the comparables' numbers (its glyphs are served). */
function labelFont() {
  for (const layer of baseStyle?.layers || []) {
    const font = layer.layout?.["text-font"];
    if (Array.isArray(font) && typeof font[0] === "string") return [font[0]];
  }
  return ["Source Sans Pro Regular"];
}

function addOverlay() {
  if (!map || !overlay) return;
  const ramp = RAMP[theme()];
  /** @type {any[]} */
  const fill = ["step", ["get", "ppm2"], ramp[0]];
  overlay.breaks.forEach((b, i) => fill.push(b, ramp[Math.min(i + 1, ramp.length - 1)]));
  const surface = token("--surface") || "#ffffff";
  const text = token("--text") || "#141417";
  const accent = token("--accent") || "#946f00";
  for (const [id, data] of [["cells", overlay.cells], ["comps", overlay.comps], ["home", overlay.home]]) {
    const source = map.getSource(id);
    if (source) source.setData(data);
    else map.addSource(id, { type: "geojson", data });
  }
  const layers = [
    { id: "cells-fill", type: "fill", source: "cells", paint: { "fill-color": fill, "fill-opacity": 0.55 } },
    { id: "cells-line", type: "line", source: "cells", paint: { "line-color": surface, "line-width": 0.6 } },
    { id: "comps-dot", type: "circle", source: "comps",
      paint: { "circle-radius": 9, "circle-color": text, "circle-stroke-color": surface, "circle-stroke-width": 2 } },
    { id: "comps-n", type: "symbol", source: "comps",
      layout: { "text-field": ["get", "n"], "text-font": labelFont(), "text-size": 11, "text-allow-overlap": true },
      paint: { "text-color": surface } },
    { id: "home", type: "circle", source: "home",
      paint: { "circle-radius": 8, "circle-color": accent, "circle-stroke-color": surface, "circle-stroke-width": 3 } },
  ];
  for (const layer of layers) {
    if (map.getLayer(layer.id)) map.removeLayer(layer.id);
    map.addLayer(layer);
  }
}

/**
 * @param {HTMLElement} legend
 * @param {number[]} breaks
 * @param {string} what
 */
function renderLegend(legend, breaks, what) {
  const ramp = RAMP[theme()];
  const items = [];
  for (let i = 0; i <= breaks.length; i++) {
    const swatch = document.createElement("i");
    swatch.style.background = ramp[Math.min(i, ramp.length - 1)];
    const label = document.createElement("span");
    label.textContent = i === 0 ? `< ${fmt.perM2(breaks[0])}`
      : i === breaks.length ? `≥ ${fmt.perM2(breaks[i - 1])}` : `${fmt.perM2(breaks[i - 1])} – ${fmt.perM2(breaks[i])}`;
    const item = document.createElement("div");
    item.append(swatch, label);
    items.push(item);
  }
  const title = document.createElement("div");
  title.className = "legend-title";
  title.textContent = what;
  legend.replaceChildren(title, ...items);
}

/**
 * Shows (or moves) the map for an estimate.
 * @param {HTMLElement} container
 * @param {HTMLElement} legend
 * @param {import("./assets.js").Assets} assets
 * @param {{lat: number, lon: number, type: "A" | "M"}} home
 * @param {import("./infer.js").Sale[]} comparables
 * @returns {Promise<"A" | "M" | "all">} the property type the squares describe
 */
export async function showNeighbourhood(container, legend, assets, home, comparables) {
  const { cells, kind } = await priceCells(assets, home.lat, home.lon, home.type);
  const breaks = quantileBreaks(cells.features.map((f) => f.properties.ppm2));
  overlay = {
    cells, breaks,
    comps: comparablePoints(comparables),
    home: { type: "FeatureCollection", features: [{ type: "Feature", properties: {},
      geometry: { type: "Point", coordinates: [home.lon, home.lat] } }] },
  };
  const what = kind === "M" ? "maisons" : kind === "A" ? "appartements" : "logements";
  if (breaks.length) renderLegend(legend, breaks, `Prix médian au m² des ${what}, 12 derniers mois`);
  else legend.replaceChildren();

  const maplibregl = await loadLibrary();
  if (!map) {
    map = new maplibregl.Map({
      container, style: await styleFor(theme()), center: [home.lon, home.lat], zoom: 14.2,
      cooperativeGestures: true, locale: LOCALE, attributionControl: false, maxZoom: 18,
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true,
      customAttribution: "Fond © IGN – Plan IGN · Prix : DVF" }));
    map.on("style.load", addOverlay);
    const popup = new maplibregl.Popup({ closeButton: false, offset: 10 });
    map.on("click", "cells-fill", (/** @type {any} */ e) => {
      const p = e.features[0].properties;
      popup.setLngLat(e.lngLat)
        .setText(`${fmt.perM2(p.ppm2)} le m² · ${p.sales} vente${p.sales > 1 ? "s" : ""} en 12 mois`).addTo(map);
    });
    map.on("click", "comps-dot", (/** @type {any} */ e) => {
      const p = e.features[0].properties;
      popup.setLngLat(e.lngLat).setText(`N° ${p.n} · ${fmt.price(p.price)} · ${fmt.perM2(p.ppm2)} le m² · ` +
        `${fmt.month(p.m)}`).addTo(map);
    });
    for (const id of ["cells-fill", "comps-dot"]) {
      map.on("mouseenter", id, () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", id, () => { map.getCanvas().style.cursor = ""; });
    }
    let shownTheme = theme();
    new MutationObserver(async () => {                      // theme switch: restyle, then redraw the overlay
      if (theme() === shownTheme) return;                   // (theme.js re-applies the same theme every minute)
      shownTheme = theme();
      map.setStyle(await styleFor(theme()), { diff: false });
      if (overlay?.breaks.length) renderLegend(legend, overlay.breaks, legend.firstChild?.textContent || "");
    }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  } else {
    map.jumpTo({ center: [home.lon, home.lat], zoom: 14.2 });
    if (map.isStyleLoaded()) addOverlay();
  }
  return kind;
}
