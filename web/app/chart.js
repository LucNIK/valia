// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

/**
 * Price trend chart: one series, inline SVG, no library. Thin line, the last point labelled,
 * a crosshair that snaps to the nearest quarter with a tooltip, and a table for screen readers.
 */

import * as fmt from "./format.js";

const NS = "http://www.w3.org/2000/svg";
const H = 190;
const PAD = { top: 18, right: 64, bottom: 26, left: 8 };

/**
 * @param {string} tag @param {Record<string, string | number>} [attrs]
 * @returns {SVGElement}
 */
function el(tag, attrs = {}) {
  const node = /** @type {SVGElement} */ (document.createElementNS(NS, tag));
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

/**
 * Month index (year * 12 + month) of the last month of a window ending (exclusive) on `end`.
 * @param {string} end ISO date, first day of a month
 */
export function lastMonthOf(end) {
  const [y, m] = end.split("-").map(Number);
  return y * 12 + m - 1;
}

/** @param {number} lo @param {number} hi */
function niceTicks(lo, hi) {
  const span = hi - lo || hi * 0.1 || 1;
  const raw = span / 3;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw) || raw;
  const start = Math.floor(lo / step) * step;
  const ticks = [];
  for (let v = start; v <= hi + step * 0.001; v += step) ticks.push(v);
  if (ticks[ticks.length - 1] < hi) ticks.push(ticks[ticks.length - 1] + step);
  return ticks;
}

/**
 * @param {HTMLElement} host container: the SVG, tooltip and table are rendered inside
 * @param {(number | null)[]} values median price per m², one per window
 * @param {string[]} ends window ends (meta.trend.ends)
 * @param {string} label what the series is, for the accessible name
 */
export function renderTrend(host, values, ends, label) {
  const W = Math.max(280, Math.round(host.clientWidth || 600));   // drawn at its real width: text stays legible
  const points = values.map((v, i) => ({ v, i, m: lastMonthOf(ends[i]) })).filter((p) => p.v !== null);
  const vs = points.map((p) => /** @type {number} */ (p.v));
  const ticks = niceTicks(Math.min(...vs), Math.max(...vs));
  const lo = ticks[0];
  const hi = ticks[ticks.length - 1];
  const n = values.length;
  const x = (/** @type {number} */ i) => PAD.left + (i / Math.max(1, n - 1)) * (W - PAD.left - PAD.right);
  const y = (/** @type {number} */ v) => PAD.top + (1 - (v - lo) / (hi - lo || 1)) * (H - PAD.top - PAD.bottom);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "trend-svg", role: "img", "aria-label": label });
  for (const t of ticks) {
    svg.append(el("line", { class: "grid", x1: PAD.left, x2: W - PAD.right, y1: y(t), y2: y(t) }));
    const text = el("text", { class: "axis", x: W - PAD.right + 8, y: y(t) + 4 });
    text.textContent = fmt.perM2(t);
    svg.append(text);
  }
  values.forEach((v, i) => {                                   // a year label under each December
    if (lastMonthOf(ends[i]) % 12 !== 0) return;
    const text = el("text", { class: "axis", x: x(i), y: H - 6, "text-anchor": "middle" });
    text.textContent = String(Math.floor((lastMonthOf(ends[i]) - 1) / 12));
    svg.append(text);
  });

  let d = "";                                                  // gaps where a quarter has too few sales
  values.forEach((v, i) => {
    if (v === null) return;
    d += `${i > 0 && values[i - 1] !== null ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
  });
  svg.append(el("path", { class: "line", d }));
  const last = points[points.length - 1];
  svg.append(el("circle", { class: "dot", cx: x(last.i), cy: y(/** @type {number} */ (last.v)), r: 4 }));

  const cross = el("line", { class: "cross", y1: PAD.top, y2: H - PAD.bottom, visibility: "hidden" });
  const focus = el("circle", { class: "dot focus", r: 4, visibility: "hidden" });
  svg.append(cross, focus);
  const hit = el("rect", { x: 0, y: 0, width: W, height: H, fill: "transparent", tabindex: 0 });
  svg.append(hit);

  const tip = document.createElement("div");
  tip.className = "trend-tip";
  tip.hidden = true;
  const tipValue = document.createElement("strong");
  const tipWhen = document.createElement("span");
  tip.append(tipValue, tipWhen);

  /** @param {{v: number | null, i: number, m: number}} p */
  const show = (p) => {
    const px = x(p.i);
    const py = y(/** @type {number} */ (p.v));
    for (const [k, v] of [["x1", px], ["x2", px]]) cross.setAttribute(String(k), String(v));
    cross.setAttribute("visibility", "visible");
    focus.setAttribute("cx", String(px));
    focus.setAttribute("cy", String(py));
    focus.setAttribute("visibility", "visible");
    tipValue.textContent = `${fmt.perM2(/** @type {number} */ (p.v))} le m²`;
    tipWhen.textContent = `12 mois jusqu'à ${fmt.month(p.m)}`;
    tip.hidden = false;
    const box = svg.getBoundingClientRect();
    const left = (px / W) * box.width;
    tip.style.left = `${Math.min(Math.max(left, 70), box.width - 70)}px`;
    tip.style.top = `${(py / H) * box.height}px`;
  };
  const hide = () => {
    cross.setAttribute("visibility", "hidden");
    focus.setAttribute("visibility", "hidden");
    tip.hidden = true;
  };
  /** @param {number} clientX */
  const nearest = (clientX) => {
    const box = svg.getBoundingClientRect();
    const vx = ((clientX - box.left) / box.width) * W;
    return points.reduce((a, b) => (Math.abs(x(b.i) - vx) < Math.abs(x(a.i) - vx) ? b : a));
  };
  let keyIndex = points.length - 1;
  hit.addEventListener("pointermove", (e) => show(nearest(/** @type {PointerEvent} */ (e).clientX)));
  hit.addEventListener("pointerleave", hide);
  hit.addEventListener("focus", () => show(points[keyIndex]));
  hit.addEventListener("blur", hide);
  hit.addEventListener("keydown", (e) => {
    const k = /** @type {KeyboardEvent} */ (e).key;
    if (k !== "ArrowLeft" && k !== "ArrowRight") return;
    e.preventDefault();
    keyIndex = Math.max(0, Math.min(points.length - 1, keyIndex + (k === "ArrowRight" ? 1 : -1)));
    show(points[keyIndex]);
  });

  const table = document.createElement("table");
  table.className = "sr-only";
  const caption = document.createElement("caption");
  caption.textContent = label;
  table.append(caption);
  for (const p of points) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = `12 mois jusqu'à ${fmt.month(p.m)}`;
    const td = document.createElement("td");
    td.textContent = fmt.perM2(/** @type {number} */ (p.v));
    tr.append(th, td);
    table.append(tr);
  }

  const lastLabel = el("text", { class: "last", x: x(last.i) + 8, y: y(/** @type {number} */ (last.v)) - 8,
                                 "text-anchor": "end" });
  lastLabel.textContent = fmt.perM2(/** @type {number} */ (last.v));
  svg.insertBefore(lastLabel, cross);
  host.replaceChildren(svg, tip, table);
  let width = W;
  const observer = new ResizeObserver(() => {
    if (!host.isConnected || !host.contains(svg)) return observer.disconnect();
    if (Math.abs(host.clientWidth - width) > 4) {
      width = host.clientWidth;
      observer.disconnect();
      renderTrend(host, values, ends, label);
    }
  });
  observer.observe(host);
}
