// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

/**
 * Evaluator for the "valia-trees/1" binary format written by valia/treemodel.py.
 * Same decision rules as LightGBM: inputs are rounded to float32, a NaN becomes 0.0 unless the
 * split's missing type is NaN, and "zero" / "NaN" missing values follow the default side.
 */

const MAGIC = "VTREES01";
const ZERO_THRESHOLD = 1e-35;
const DEFAULT_LEFT = 2;
const MISSING_ZERO = 1;
const MISSING_NAN = 2;

/**
 * @typedef {object} Forest
 * @property {number} nFeatures
 * @property {number} nTrees
 * @property {Int32Array} nodeStart
 * @property {Int32Array} leafStart
 * @property {Int32Array} treeNodes
 * @property {Uint8Array} feature
 * @property {Uint8Array} decision
 * @property {Float64Array} threshold
 * @property {Int32Array} left
 * @property {Int32Array} right
 * @property {Float64Array} value
 */

/**
 * @param {ArrayBuffer} buffer
 * @returns {Forest}
 */
export function readForest(buffer) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 8));
  if (magic !== MAGIC) throw new Error("not a valia tree file");
  const nFeatures = view.getUint32(8, true);
  const nTrees = view.getUint32(12, true);
  const nNodes = view.getUint32(16, true);
  const nLeaves = view.getUint32(20, true);
  let offset = 24;

  // Copies keep typed arrays aligned whatever the offset; values are little-endian on every
  // platform that runs a browser, and DataView reads would be needlessly slow here.
  /**
   * @param {number} count
   * @param {number} size
   */
  const take = (count, size) => {
    const bytes = buffer.slice(offset, offset + count * size);
    offset += count * size;
    return bytes;
  };
  const nodeStart = new Int32Array(take(nTrees, 4));
  const leafStart = new Int32Array(take(nTrees, 4));
  const treeNodes = new Int32Array(take(nTrees, 4));
  const feature = new Uint8Array(take(nNodes, 1));
  const decision = new Uint8Array(take(nNodes, 1));
  const threshold = new Float64Array(take(nNodes, 8));
  const left = new Int32Array(take(nNodes, 4));
  const right = new Int32Array(take(nNodes, 4));
  const value = new Float64Array(take(nLeaves, 8));
  if (offset !== buffer.byteLength) throw new Error("tree file has an unexpected size");
  return { nFeatures, nTrees, nodeStart, leafStart, treeNodes, feature, decision, threshold, left, right, value };
}

/**
 * @param {number} x
 * @param {number} decision
 * @param {number} threshold
 */
function goesLeft(x, decision, threshold) {
  const missing = (decision >> 2) & 3;
  if (Number.isNaN(x) && missing !== MISSING_NAN) x = 0.0;
  if ((missing === MISSING_ZERO && x >= -ZERO_THRESHOLD && x <= ZERO_THRESHOLD) ||
      (missing === MISSING_NAN && Number.isNaN(x))) {
    return (decision & DEFAULT_LEFT) !== 0;
  }
  return x <= threshold;
}

/**
 * Raw score of one row of features.
 * @param {Forest} f
 * @param {ArrayLike<number>} row
 */
export function predictRow(f, row) {
  if (row.length !== f.nFeatures) throw new Error(`expected ${f.nFeatures} features, got ${row.length}`);
  const x = Float64Array.from(row, (v) => Math.fround(v));
  let total = 0.0;
  for (let t = 0; t < f.nTrees; t++) {
    const base = f.nodeStart[t];
    const leafBase = f.leafStart[t];
    if (f.treeNodes[t] === 0) {
      total += f.value[leafBase];
      continue;
    }
    let node = 0;
    while (node >= 0) {
      const i = base + node;
      node = goesLeft(x[f.feature[i]], f.decision[i], f.threshold[i]) ? f.left[i] : f.right[i];
    }
    total += f.value[leafBase + ~node];
  }
  return total;
}
