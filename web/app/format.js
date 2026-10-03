// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

const MONTHS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."];
const euros = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 });

/**
 * Rounds a price the way an estimate should be read: to the thousand, or to the hundred below 100 k€.
 * @param {number} value
 */
export function price(value) {
  const step = value >= 100_000 ? 1000 : 100;
  return `${euros.format(Math.round(value / step) * step)} €`;
}

/** @param {number} value */
export function perM2(value) {
  return `${euros.format(Math.round(value / 10) * 10)} €`;
}

/**
 * @param {number} m month index, year * 12 + month (January = 1)
 */
export function month(m) {
  const year = Math.floor((m - 1) / 12);
  return `${MONTHS[(m - 1) % 12]} ${year}`;
}

/** @param {number} km */
export function distance(km) {
  return km < 1 ? `${Math.max(10, Math.round(km * 100) * 10)} m` : `${km.toFixed(1).replace(".", ",")} km`;
}
