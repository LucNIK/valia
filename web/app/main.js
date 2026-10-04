// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

import { Assets, httpLoader, parcelOf } from "./assets.js";
import * as fmt from "./format.js";
import { parcelAt, searchAddress } from "./geocode.js";
import { departmentOf } from "./geo.js";
import { lastMonthOf, renderTrend } from "./chart.js";
import { compare } from "./compare.js";
import { estimate } from "./infer.js";
import { showNeighbourhood } from "./map.js";

/** Départements DVF does not cover (Alsace-Moselle land registry, Mayotte). */
const NOT_COVERED = new Set(["57", "67", "68", "976"]);
const KM_PER_DEG = 111.195;

/**
 * @template {HTMLElement} T
 * @param {string} id
 * @returns {T}
 */
function $(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} missing`);
  return /** @type {T} */ (el);
}

const form = /** @type {HTMLFormElement} */ ($("form"));
const address = /** @type {HTMLInputElement} */ ($("address"));
const list = /** @type {HTMLUListElement} */ ($("suggestions"));
const surface = /** @type {HTMLInputElement} */ ($("surface"));
const rooms = /** @type {HTMLInputElement} */ ($("rooms"));
const land = /** @type {HTMLInputElement} */ ($("land"));
const landField = $("land-field");
const dpe = /** @type {HTMLSelectElement} */ ($("dpe"));
const year = /** @type {HTMLInputElement} */ ($("year"));
const hint = $("building-hint");
const DPE_LETTERS = " ABCDEFG";
const error = $("error");
const submit = /** @type {HTMLButtonElement} */ ($("submit"));
const result = $("result");

const assetsPromise = Assets.open(httpLoader(new URL("./", location.href)));
assetsPromise.then(showAccuracy).catch(() => showError("Les données n'ont pas pu être chargées. Réessayez plus tard."));

/** @typedef {import("./geocode.js").Place & {parcel?: string}} Located */
/** @type {Located | null} */
let place = null;
/** @type {Promise<void>} */
let lookup = Promise.resolve();
/**
 * An estimate on screen, kept to be compared with the next one.
 * @typedef {{label: string, type: "A" | "M", out: Awaited<ReturnType<typeof estimate>>, state: string}} Shown
 */
/** @type {Shown | null} */
let current = null;
/** @type {Shown | null} */
let compared = null;
/** @type {import("./geocode.js").Place[]} */
let options = [];
let active = -1;
/** @type {AbortController | null} */
let pending = null;
/** @type {ReturnType<typeof setTimeout> | undefined} */
let timer;

// ---- address autocomplete (ARIA combobox)

address.addEventListener("input", () => {
  place = null;
  clearTimeout(timer);
  const text = address.value.trim();
  if (text.length < 3) return closeList();
  timer = setTimeout(async () => {
    pending?.abort();
    pending = new AbortController();
    try {
      options = await searchAddress(text, pending.signal);
      renderList();
    } catch (err) {
      if (/** @type {Error} */ (err).name !== "AbortError") closeList();
    }
  }, 180);
});

address.addEventListener("keydown", (e) => {
  if (list.hidden || !options.length) return;
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    active = (active + (e.key === "ArrowDown" ? 1 : -1) + options.length) % options.length;
    renderList();
  } else if (e.key === "Enter" && active >= 0) {
    e.preventDefault();
    choose(options[active]);
  } else if (e.key === "Escape") {
    closeList();
  }
});
address.addEventListener("blur", () => setTimeout(closeList, 120));

function renderList() {
  list.replaceChildren(...options.map((o, i) => {
    const li = document.createElement("li");
    li.id = `opt-${i}`;
    li.setAttribute("role", "option");
    li.setAttribute("aria-selected", String(i === active));
    li.textContent = o.label;
    li.addEventListener("mousedown", (e) => {
      e.preventDefault();
      choose(o);
    });
    return li;
  }));
  list.hidden = options.length === 0;
  address.setAttribute("aria-expanded", String(!list.hidden));
  if (active >= 0) address.setAttribute("aria-activedescendant", `opt-${active}`);
  else address.removeAttribute("aria-activedescendant");
}

function closeList() {
  options = [];
  active = -1;
  list.hidden = true;
  list.replaceChildren();
  address.setAttribute("aria-expanded", "false");
  address.removeAttribute("aria-activedescendant");
}

/** @param {import("./geocode.js").Place} p */
function choose(p) {
  place = { ...p };
  address.value = p.label;
  closeList();
  surface.focus();
  lookup = findBuilding(place);
}

/**
 * Finds the parcel under the address and pre-fills the building fields from the BDNB.
 * @param {Located} where
 */
async function findBuilding(where) {
  hint.textContent = "Recherche du bâtiment…";
  hint.className = "hint";
  try {
    const [assets, parcel] = await Promise.all([assetsPromise, parcelAt(where.lat, where.lon)]);
    if (place !== where) return;                       // another address was chosen meanwhile
    const entry = parcel ? await parcelOf(assets, parcel) : null;
    where.parcel = parcel || "";
    if (!entry) {
      hint.textContent = "Bâtiment non trouvé : renseignez ces champs si vous les connaissez.";
      return;
    }
    dpe.value = entry.dpe_class ? String(entry.dpe_class) : "";
    year.value = entry.year_built ? String(entry.year_built) : "";
    const facts = [entry.levels ? `${entry.levels} niveau${entry.levels > 1 ? "x" : ""}` : "",
                   entry.dwellings ? `${entry.dwellings} logement${entry.dwellings > 1 ? "s" : ""}` : "",
                   entry.elevator ? "ascenseur" : ""].filter(Boolean).join(", ");
    hint.textContent = `Trouvé dans la BDNB${facts ? ` (${facts})` : ""}. Corrigez si votre logement diffère, ` +
      "par exemple après une rénovation.";
    hint.className = "hint found";
  } catch {
    if (place === where) hint.textContent = "Bâtiment non trouvé : renseignez ces champs si vous les connaissez.";
  }
}

// ---- form

/** @returns {"A" | "M"} */
function selectedType() {
  const checked = /** @type {HTMLInputElement | null} */ (form.querySelector('input[name="type"]:checked'));
  return checked?.value === "M" ? "M" : "A";
}

for (const radio of form.querySelectorAll('input[name="type"]')) {
  radio.addEventListener("change", () => { landField.hidden = selectedType() !== "M"; });
}

/** @param {string} message */
function showError(message) {
  error.textContent = message;
  error.hidden = false;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  error.hidden = true;
  if (!place && options.length) choose(options[0]);
  if (!place) return showError("Choisissez une adresse dans la liste.");
  const s = Number(surface.value);
  const r = Number(rooms.value);
  const l = Number(land.value || 0);
  if (!(s >= 9 && s <= 1000)) return showError("La surface doit être comprise entre 9 et 1 000 m².");
  if (!(Number.isInteger(r) && r >= 1 && r <= 20)) return showError("Le nombre de pièces doit être compris entre 1 et 20.");
  if (!(l >= 0)) return showError("La surface du terrain ne peut pas être négative.");
  const y = year.value ? Number(year.value) : null;
  if (y !== null && !(Number.isInteger(y) && y >= 1000 && y <= 2030)) {
    return showError("L'année de construction doit être comprise entre 1000 et 2030.");
  }
  if (NOT_COVERED.has(departmentOf(place.citycode))) {
    return showError("Les ventes de ce département ne sont pas publiées dans DVF (Alsace, Moselle, Mayotte) : pas d'estimation possible.");
  }
  const where = place;
  const type = selectedType();
  submit.disabled = true;
  try {
    await lookup;
    const assets = await assetsPromise;
    const building = { dpe_class: dpe.value ? Number(dpe.value) : null, year_built: y };
    const out = await estimate(assets, { lat: where.lat, lon: where.lon, type, surface: s, rooms: r, land: l,
                                         citycode: where.citycode, parcel: where.parcel || "", building });
    render(assets, out, type, where);
    const params = new URLSearchParams({ a: where.label, c: where.citycode, lat: String(where.lat),
                                        lon: String(where.lon), t: type, s: String(s), r: String(r), l: String(l),
                                        p: where.parcel || "", d: dpe.value, y: year.value });
    current = { label: where.label, type, out, state: params.toString() };
    renderCompare(assets);
    if (compared) params.set("vs", compared.state);
    history.replaceState(null, "", `#${params.toString()}`);
  } catch (err) {
    console.error(err);
    showError("L'estimation a échoué. Réessayez dans un instant.");
  } finally {
    submit.disabled = false;
  }
});

// ---- result

/**
 * @param {Assets} assets
 * @param {Awaited<ReturnType<typeof estimate>>} out
 * @param {"A" | "M"} type
 * @param {import("./geocode.js").Place} where
 */
function render(assets, out, type, where) {
  $("price").textContent = fmt.price(out.price);
  $("ppm2").textContent = fmt.perM2(out.priceM2);
  $("low").textContent = fmt.price(out.low);
  $("high").textContent = fmt.price(out.high);
  const span = Math.log(out.high) - Math.log(out.low);
  $("marker").style.left = `${((Math.log(out.price) - Math.log(out.low)) / span) * 100}%`;

  const f = out.features;
  const kinds = type === "M" ? "de maisons" : "d'appartements";
  const explain = $("explain");
  /** @type {Record<string, string>} */
  const sources = {
    knn_prior: `la médiane des ${assets.meta.knn.k} ventes ${kinds} les plus proches, situées à ${fmt.distance(f.knn_km)} ` +
      `en médiane et vendues il y a ${Math.round(f.knn_age)} mois en médiane`,
    cell_type_prior: `les ventes ${kinds} des douze derniers mois dans un rayon d'environ 500 m`,
    cell_prior: "les ventes des douze derniers mois dans un rayon d'environ 500 m",
    cell2_type_prior: `les ventes ${kinds} des douze derniers mois dans un rayon d'environ 2 km`,
    commune_type_prior: `les ventes ${kinds} des douze derniers mois dans la commune`,
    commune_prior: "les ventes des douze derniers mois dans la commune",
  };
  const weak = !(out.anchorSource in sources);
  explain.className = weak ? "note warn" : "note";
  const accuracy = assets.meta.accuracy?.by_segment?.[out.segment]?.mdape_pct;
  explain.textContent = weak
    ? "Il y a trop peu de ventes récentes autour de cette adresse : l'estimation part d'un niveau moyen " +
      "(commune, département ou France) et reste peu fiable."
    : `Le niveau de référence part de ${sources[out.anchorSource]}. Chaque ligne indique de combien ce qui ` +
      "distingue votre logement fait monter ou baisser le prix au m², d'après les ventes passées." +
      (accuracy ? ` Dans ce type de zone (${out.segment.toLowerCase()}), l'écart médian avec le prix de vente ` +
                  `réel a été de ${String(accuracy).replace(".", ",")} % lors du test sur ${assets.meta.accuracy.test_year}.` : "");

  renderWhy(assets, out, type);
  renderTrendCard(assets, out, type, where);
  renderComparables(out.comparables, where);
  result.hidden = false;
  showNeighbourhood($("map"), $("map-legend"), assets, { lat: where.lat, lon: where.lon, type }, out.comparables)
    .then(() => { $("map-card").hidden = false; })
    .catch((err) => {                                      // no map (offline, blocked tiles): the rest stands
      console.warn(err);
      $("map-card").hidden = true;
    });
  result.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
}

/**
 * A log effect as a signed percentage: "+6,2 %", "−12 %".
 * @param {number} g
 */
function pct(g) {
  const v = (Math.exp(g) - 1) * 100;
  const r = Math.abs(v) < 10 ? Math.round(v * 10) / 10 : Math.round(v);
  return `${r > 0 ? "+" : r < 0 ? "−" : ""}${String(Math.abs(r)).replace(".", ",")} %`;
}

/**
 * One line of a breakdown: label and detail, a bar centred on zero, the value.
 * @param {string} label @param {string} detail @param {string} value
 * @param {number | null} g log effect (null: no bar) @param {number} scale log effect of a full half-bar
 * @param {string} [cls]
 */
function whyRow(label, detail, value, g, scale, cls) {
  const li = document.createElement("li");
  if (cls) li.className = cls;
  const what = document.createElement("div");
  what.className = "what";
  const strong = document.createElement("strong");
  strong.textContent = label;
  const span = document.createElement("span");
  span.textContent = detail;
  what.append(strong, span);
  const bar = document.createElement("div");
  bar.className = "bar";
  if (g !== null) {
    const i = document.createElement("i");
    const w = Math.min(50, (Math.abs(g) / scale) * 50);
    i.className = g < 0 ? "down" : "";
    i.style.left = `${g < 0 ? 50 - w : 50}%`;
    i.style.width = `${w}%`;
    bar.append(i);
  }
  bar.setAttribute("aria-hidden", "true");
  const val = document.createElement("div");
  val.className = "val";
  val.textContent = value;
  li.append(what, bar, val);
  return li;
}

/**
 * Short description of each group for one estimate ("62 m², 3 pièces", "classe E · construit en 1870"…).
 * @param {Assets} assets
 * @param {Awaited<ReturnType<typeof estimate>>} out
 * @param {"A" | "M"} type
 */
function groupDetails(assets, out, type) {
  const f = out.features;
  const b = out.building;
  const s = Math.exp(f.surface_log);
  return /** @type {Record<string, string>} */ ({
    size: `${Math.round(s)} m², ${f.rooms} pièce${f.rooms > 1 ? "s" : ""}`,
    type: type === "M" ? "Maison" : "Appartement",
    land: type === "M" && f.land_log > 0 ? `${Math.round(Math.expm1(f.land_log))} m²` : "Sans terrain",
    building: [b.dpe_class ? `classe ${DPE_LETTERS[Number(b.dpe_class)]}` : "",
               b.year_built ? `construit en ${b.year_built}` : "",
               b.levels ? `${b.levels} niveau${Number(b.levels) > 1 ? "x" : ""}` : ""].filter(Boolean).join(" · ")
              || "Non renseigné",
    location: Number.isNaN(f.station_km) ? "Position dans le quartier" : `Gare à ${fmt.distance(f.station_km)}`,
    date: `Marché de ${fmt.month(assets.meta.now_month - 1)}`,
    area: "Ventes voisines, dynamique et profil de la commune",
  });
}

/** Groups of meta.explain_groups, in plain words. */
const GROUP_LABELS = /** @type {Record<string, string>} */ ({
  size: "Surface et pièces", type: "Type de bien", land: "Terrain", building: "Bâtiment",
  location: "Emplacement précis", date: "Date", area: "Quartier et commune",
});
const MIN_SHOWN = Math.log(1.005);          // effects under 0.5 % are grouped as "Autres facteurs"

/**
 * "Pourquoi ce prix": reference level of the area, then each group's effect, then the estimate.
 * In log space the rows add up exactly; shown as percentages they multiply.
 * @param {Assets} assets
 * @param {Awaited<ReturnType<typeof estimate>>} out
 * @param {"A" | "M"} type
 */
function renderWhy(assets, out, type) {
  const f = out.features;
  const e = out.explanation;
  const details = groupDetails(assets, out, type);

  const shown = Object.entries(e.groups).filter(([, g]) => Math.abs(g) >= MIN_SHOWN)
    .sort((x, y) => Math.abs(y[1]) - Math.abs(x[1]));
  const rest = Object.values(e.groups).filter((g) => Math.abs(g) < MIN_SHOWN).reduce((a, g) => a + g, 0);
  const scale = Math.max(Math.log(1.1), ...shown.map(([, g]) => Math.abs(g)));

  /**
   * @param {string} label @param {string} detail @param {string} value @param {number | null} g
   * @param {string} [cls]
   */
  const row = (label, detail, value, g, cls) => whyRow(label, detail, value, g, scale, cls);

  const reference = Math.exp(f.anchor + e.base);
  const items = [row("Niveau de référence", "Prix au m² d'un logement type ici", fmt.perM2(reference), null, "ref")];
  for (const [name, g] of shown) items.push(row(GROUP_LABELS[name] || name, details[name] || "", pct(g), g));
  if (Math.abs(rest) >= Math.log(1.001)) items.push(row("Autres facteurs", "Effets de moins de 0,5 % chacun", pct(rest), rest));
  items.push(row("Votre estimation", "Prix au m²", fmt.perM2(out.priceM2), null, "total"));
  $("why").replaceChildren(...items);
}

/**
 * @param {Assets} assets
 * @param {Awaited<ReturnType<typeof estimate>>} out
 * @param {"A" | "M"} type
 * @param {{label: string}} where
 */
function renderTrendCard(assets, out, type, where) {
  const card = $("trend-card");
  const t = out.trend;
  card.hidden = !t;
  if (!t) return;
  /** @param {number | null} c @param {string} id */
  const stat = (c, id) => {
    const node = $(id);
    node.className = c !== null && c > 0 ? "up" : "";
    node.textContent = c === null ? "—"
      : `${c > 0 ? "+" : c < 0 ? "−" : ""}${String(Math.abs(Math.round(c * 1000) / 10)).replace(".", ",")} %`;
  };
  stat(t.yearChange, "trend-year");
  stat(t.totalChange, "trend-total");
  const ends = assets.meta.trend.ends;
  const firstIndex = t.values.findIndex((v) => v !== null);
  $("trend-since-label").textContent = `Depuis ${fmt.month(lastMonthOf(ends[firstIndex]))}`;
  const kinds = t.kind === "M" ? "maisons" : t.kind === "A" ? "appartements" : "logements";
  const city = (where.label.match(/\d{5}\s+(.+)$/) || [])[1];
  const scope = t.scope === "commune" ? (city ? `à ${city}` : "dans la commune")
    : t.scope === "departement" ? "dans le département" : "en France";
  const label = `Prix médian au m² des ${kinds} vendus ${scope}, sur 12 mois glissants`;
  renderTrend($("trend"), t.values, ends, label);
  $("trend-note").textContent = `${label}, trimestre par trimestre (DVF).` +
    (t.scope !== "commune" ? " Trop peu de ventes dans la commune pour une courbe locale fiable." : "") +
    (t.kind === "all" ? " Pas assez de ventes de ce type de bien : tous les logements sont comptés." : "");
}

/**
 * @param {import("./infer.js").Sale[]} comps
 * @param {{lat: number, lon: number}} at
 */
function renderComparables(comps, at) {
  $("comps-card").hidden = comps.length === 0;
  $("comps").replaceChildren(...comps.map((c, i) => {
    const tr = document.createElement("tr");
    for (const text of [String(i + 1), fmt.distance(c.km), fmt.month(c.m), `${Math.round(c.surface)} m²`, fmt.price(c.price),
                        fmt.perM2(c.price / c.surface)]) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.append(td);
    }
    return tr;
  }));

  const NS = "http://www.w3.org/2000/svg";
  /** @param {string} tag @param {Record<string, string | number>} attrs */
  const svg = (tag, attrs) => {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
    return el;
  };
  const cos = Math.cos((at.lat * Math.PI) / 180);
  const points = comps.map((c) => [(c.lon - at.lon) * cos * KM_PER_DEG, (c.lat - at.lat) * KM_PER_DEG]);
  const reach = Math.max(0.05, ...points.map(([x, y]) => Math.hypot(x, y)));
  const scale = 50 / reach;
  $("radar").replaceChildren(
    svg("circle", { class: "ring", cx: 0, cy: 0, r: 25 }),
    svg("circle", { class: "ring", cx: 0, cy: 0, r: 50 }),
    ...points.map(([x, y]) => svg("circle", { class: "sale", cx: (x * scale).toFixed(1), cy: (-y * scale).toFixed(1), r: 3 })),
    svg("circle", { class: "me", cx: 0, cy: 0, r: 5 }),
  );
}

/** @param {Assets} assets */
function showAccuracy(assets) {
  const a = assets.meta.accuracy || {};
  const [y, m] = assets.meta.data_through.split("-").map(Number);
  $("through").textContent = `, ventes jusqu'à ${fmt.month(y * 12 + m)}`;
  if (a.mdape_pct) {
    const n = a.test_sales ? `${Number(a.test_sales).toLocaleString("fr-FR")} ` : "";
    const cover = a.coverage_pct ? `, prix réel dans la fourchette ${String(a.coverage_pct).replace(".", ",")} % du temps` : "";
    $("accuracy").textContent = `Précision mesurée sur ${n}ventes de ${a.test_year} jamais vues par le modèle : ` +
      `écart médian de ${String(a.mdape_pct).replace(".", ",")} %${cover}.`;
  }
}

// ---- comparison of two homes

const compareBanner = $("compare-banner");

$("compare-btn").addEventListener("click", () => {
  if (!current) return;
  compared = current;
  $("compare-a").textContent = `${compared.label} · ${fmt.price(compared.out.price)}`;
  compareBanner.hidden = false;
  result.hidden = true;
  place = null;
  for (const input of [address, surface, rooms, land, year]) input.value = "";
  dpe.value = "";
  hint.textContent = "";
  window.scrollTo({ top: form.offsetTop - 80, behavior: "smooth" });
  address.focus({ preventScroll: true });
});

$("compare-cancel").addEventListener("click", () => {
  compared = null;
  compareBanner.hidden = true;
});

$("compare-clear").addEventListener("click", () => {
  compared = null;
  $("compare-card").hidden = true;
  compareBanner.hidden = true;
  if (current) history.replaceState(null, "", `#${current.state}`);
});

/**
 * Side by side: the home kept with "Comparer" (A) and the one just estimated (B).
 * @param {Assets} assets
 */
function renderCompare(assets) {
  const card = $("compare-card");
  card.hidden = !compared || !current;
  if (!compared || !current) return;
  compareBanner.hidden = true;
  const a = compared;
  const b = current;

  $("compare-grid").replaceChildren(...[["A", a], ["B", b]].map(([tag, h]) => {
    const shown = /** @type {Shown} */ (h);
    const box = document.createElement("div");
    const f = shown.out.features;
    const bld = shown.out.building;
    const t = document.createElement("div");
    t.className = "tag";
    t.textContent = `Bien ${tag}`;
    const where = document.createElement("div");
    where.className = "where";
    where.textContent = shown.label;
    const big = document.createElement("div");
    big.className = "big";
    big.textContent = fmt.price(shown.out.price);
    const dl = document.createElement("dl");
    const facts = [
      ["Prix au m²", fmt.perM2(shown.out.priceM2)],
      ["Fourchette", `${fmt.price(shown.out.low)} – ${fmt.price(shown.out.high)}`],
      ["Bien", `${shown.type === "M" ? "Maison" : "Appart."}, ${Math.round(Math.exp(f.surface_log))} m², ${f.rooms} p.`],
      ["Bâtiment", [bld.dpe_class ? DPE_LETTERS[Number(bld.dpe_class)] : "", bld.year_built || ""].filter(Boolean).join(" · ") || "—"],
      ["Tendance 1 an", shown.out.trend && shown.out.trend.yearChange !== null ? pct(Math.log1p(shown.out.trend.yearChange)) : "—"],
    ];
    for (const [k, v] of facts) {
      const dt = document.createElement("dt");
      dt.textContent = String(k);
      const dd = document.createElement("dd");
      dd.textContent = String(v);
      dl.append(dt, dd);
    }
    box.append(t, where, big, dl);
    return box;
  }));

  const c = compare(a.out, b.out);
  const euros = b.out.price - a.out.price;
  const gap = $("compare-gap");
  gap.replaceChildren();
  const lead = document.createTextNode(Math.abs(c.price) < 0.005 ? "Les deux biens valent à peu près le même prix "
    : `Le bien B est estimé ${c.price > 0 ? "plus cher" : "moins cher"} de `);
  const strong = document.createElement("strong");
  strong.textContent = Math.abs(c.price) < 0.005 ? "" : `${fmt.price(Math.abs(euros))} (${pct(Math.log1p(c.price))})`;
  gap.append(lead, strong, document.createTextNode(`, soit ${pct(Math.log1p(c.priceM2))} au m².`));

  const da = groupDetails(assets, a.out, a.type);
  const db = groupDetails(assets, b.out, b.type);
  // A group whose inputs are the same for both homes (the date, often) still differs a little: trees
  // interact. That share goes with the small effects rather than reading as a cause.
  const same = (/** @type {{group: string}} */ r) => r.group !== "area" && da[r.group] === db[r.group];
  const shownRows = c.rows.filter((r) => Math.abs(r.diff) >= MIN_SHOWN && !same(r));
  const rest = c.rows.filter((r) => Math.abs(r.diff) < MIN_SHOWN || same(r)).reduce((s, r) => s + r.diff, 0);
  const scale = Math.max(Math.log(1.1), ...shownRows.map((r) => Math.abs(r.diff)));
  const items = shownRows.map((r) => whyRow(GROUP_LABELS[r.group] || r.group,
    r.group === "area" ? "Niveau des prix autour de chaque adresse" : `${da[r.group]} → ${db[r.group]}`,
    pct(r.diff), r.diff, scale));
  if (Math.abs(rest) >= Math.log(1.001)) {
    items.push(whyRow("Autres facteurs", "Petits écarts et effets croisés des caractéristiques", pct(rest), rest, scale));
  }
  items.push(whyRow("Écart au m²", "Bien B par rapport au bien A", pct(Math.log1p(c.priceM2)), null, scale, "total"));
  $("compare-why").replaceChildren(...items);
}

/**
 * Re-estimates a home from a shareable state (the "vs" part of a link).
 * @param {Assets} assets
 * @param {string} state
 * @returns {Promise<Shown | null>}
 */
async function fromState(assets, state) {
  const p = new URLSearchParams(state);
  const lat = Number(p.get("lat"));
  const lon = Number(p.get("lon"));
  const s = Number(p.get("s"));
  const r = Number(p.get("r"));
  if (!p.get("a") || !Number.isFinite(lat) || !Number.isFinite(lon) || !(s > 0) || !(r > 0)) return null;
  const type = p.get("t") === "M" ? "M" : "A";
  const out = await estimate(assets, { lat, lon, type, surface: s, rooms: r, land: Number(p.get("l") || 0),
    citycode: p.get("c") || "", parcel: p.get("p") || "",
    building: { dpe_class: p.get("d") ? Number(p.get("d")) : null, year_built: p.get("y") ? Number(p.get("y")) : null } });
  return { label: String(p.get("a")), type, out, state };
}

// ---- shareable link: #a=label&c=citycode&lat=..&lon=..&t=A&s=65&r=3&l=0

(function restore() {
  if (!location.hash) return;
  const p = new URLSearchParams(location.hash.slice(1));
  const lat = Number(p.get("lat"));
  const lon = Number(p.get("lon"));
  if (!p.get("a") || !Number.isFinite(lat) || !Number.isFinite(lon)) return;
  place = { label: String(p.get("a")), context: "", citycode: String(p.get("c") || ""), type: "", lat, lon,
            parcel: String(p.get("p") || "") };
  dpe.value = p.get("d") || "";
  year.value = p.get("y") || "";
  address.value = place.label;
  if (p.get("t") === "M") {
    /** @type {HTMLInputElement} */ ($("type-m")).checked = true;
    landField.hidden = false;
  }
  surface.value = p.get("s") || "";
  rooms.value = p.get("r") || "";
  land.value = p.get("l") || "";
  const vs = p.get("vs");
  if (vs) {
    assetsPromise.then((assets) => fromState(assets, vs)).then((a) => {
      compared = a;
      if (surface.value && rooms.value) form.requestSubmit();
    }).catch(() => { if (surface.value && rooms.value) form.requestSubmit(); });
    return;
  }
  if (surface.value && rooms.value) form.requestSubmit();
})();
