// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

/** Fills the accuracy table of the method page from data/meta.json. */

const SEGMENTS = ["Paris", "Grandes villes", "Villes moyennes", "Rural"];

/** @param {number | null | undefined} v */
const pct = (v) => (v === null || v === undefined ? "—" : `${String(v).replace(".", ",")} %`);

async function main() {
  const r = await fetch(new URL("data/meta.json", location.href));
  if (!r.ok) return;
  const meta = await r.json();
  const a = meta.accuracy || {};
  const intro = document.getElementById("m-intro");
  if (intro && a.test_year) {
    const n = a.test_sales ? `${Number(a.test_sales).toLocaleString("fr-FR")} ventes de ` : "les ventes de ";
    intro.textContent = `Mesurée sur ${n}${a.test_year}, jamais vues par le modèle (données jusqu'au ${meta.data_through}).`;
  }
  /** @type {[string, any][]} */
  const rows = [["France", a], ...SEGMENTS.filter((s) => a.by_segment?.[s]).map((s) => /** @type {[string, any]} */ ([s, a.by_segment[s]]))];
  const body = document.getElementById("m-table");
  body?.replaceChildren(...rows.map(([name, m]) => {
    const tr = document.createElement("tr");
    for (const text of [name, pct(m.mdape_pct), pct(m.coverage_pct)]) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.append(td);
    }
    return tr;
  }));
}

main().catch(() => { /* the static text stays */ });
