// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

/**
 * Evaluator for the "valia-trees" binary format written by valia/treemodel.py.
 * Same decision rules as LightGBM: inputs are rounded to float32, a NaN becomes 0.0 unless the
 * split's missing type is NaN, and "zero" / "NaN" missing values follow the default side.
 * Version 2 files carry the training row counts per node, used to explain predictions (TreeSHAP).
 */

const MAGIC = "VTREES02";
const MAGIC_V1 = "VTREES01";
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
 * @property {Uint32Array | null} nodeCount
 * @property {Uint32Array | null} leafCount
 */

/**
 * @param {ArrayBuffer} buffer
 * @returns {Forest}
 */
export function readForest(buffer) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 8));
  if (magic !== MAGIC && magic !== MAGIC_V1) throw new Error("not a valia tree file");
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
  const nodeCount = magic === MAGIC ? new Uint32Array(take(nNodes, 4)) : null;
  const leafCount = magic === MAGIC ? new Uint32Array(take(nLeaves, 4)) : null;
  if (offset !== buffer.byteLength) throw new Error("tree file has an unexpected size");
  return { nFeatures, nTrees, nodeStart, leafStart, treeNodes, feature, decision, threshold, left, right, value,
           nodeCount, leafCount };
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

// ---- TreeSHAP: a mirror of the Python reference in valia/treemodel.py ---------------------------
// Same arithmetic in the same order; the paths live in one preallocated buffer (each recursion level
// copies its parent's path into the next segment) instead of fresh arrays, as in LightGBM's C++.

/** Path buffers: feature index, share of rows when absent (zero), x follows (one), weight. */
class Paths {
  /** @param {number} size */
  constructor(size) {
    this.feature = new Int32Array(size);
    this.zero = new Float64Array(size);
    this.one = new Float64Array(size);
    this.weight = new Float64Array(size);
  }
}

/**
 * @param {Paths} p @param {number} o segment start @param {number} depth @param {number} zero
 * @param {number} one @param {number} feature
 */
function extend(p, o, depth, zero, one, feature) {
  p.feature[o + depth] = feature;
  p.zero[o + depth] = zero;
  p.one[o + depth] = one;
  p.weight[o + depth] = depth === 0 ? 1.0 : 0.0;
  for (let i = depth - 1; i >= 0; i--) {
    p.weight[o + i + 1] += one * p.weight[o + i] * (i + 1) / (depth + 1);
    p.weight[o + i] = zero * p.weight[o + i] * (depth - i) / (depth + 1);
  }
}

/** @param {Paths} p @param {number} o @param {number} depth @param {number} index */
function unwind(p, o, depth, index) {
  const one = p.one[o + index];
  const zero = p.zero[o + index];
  let nextOne = p.weight[o + depth];
  for (let i = depth - 1; i >= 0; i--) {
    if (one !== 0) {
      const tmp = p.weight[o + i];
      p.weight[o + i] = nextOne * (depth + 1) / ((i + 1) * one);
      nextOne = tmp - p.weight[o + i] * zero * (depth - i) / (depth + 1);
    } else {
      p.weight[o + i] = p.weight[o + i] * (depth + 1) / (zero * (depth - i));
    }
  }
  for (let i = index; i < depth; i++) {
    p.feature[o + i] = p.feature[o + i + 1];
    p.zero[o + i] = p.zero[o + i + 1];
    p.one[o + i] = p.one[o + i + 1];
  }
}

/** @param {Paths} p @param {number} o @param {number} depth @param {number} index */
function unwoundSum(p, o, depth, index) {
  const one = p.one[o + index];
  const zero = p.zero[o + index];
  let nextOne = p.weight[o + depth];
  let total = 0.0;
  for (let i = depth - 1; i >= 0; i--) {
    if (one !== 0) {
      const tmp = nextOne * (depth + 1) / ((i + 1) * one);
      total += tmp;
      nextOne = p.weight[o + i] - tmp * zero * (depth - i) / (depth + 1);
    } else {
      total += p.weight[o + i] / (zero * (depth - i) / (depth + 1));
    }
  }
  return total;
}

/**
 * @param {Forest} f @param {number} t @param {Float64Array} x @param {Float64Array} phi @param {Paths} p
 */
function treeShap(f, t, x, phi, p) {
  const base = f.nodeStart[t];
  const leafBase = f.leafStart[t];
  const nodeCount = /** @type {Uint32Array} */ (f.nodeCount);
  const leafCount = /** @type {Uint32Array} */ (f.leafCount);
  /** @param {number} child */
  const count = (child) => (child < 0 ? leafCount[leafBase + ~child] : nodeCount[base + child]);

  /**
   * @param {number} node @param {number} parent segment of the parent's path @param {number} depth
   * @param {number} zero @param {number} one @param {number} feature
   */
  const recurse = (node, parent, depth, zero, one, feature) => {
    const o = parent + depth + 1;
    for (let k = 0; k < depth; k++) {
      p.feature[o + k] = p.feature[parent + k];
      p.zero[o + k] = p.zero[parent + k];
      p.one[o + k] = p.one[parent + k];
      p.weight[o + k] = p.weight[parent + k];
    }
    extend(p, o, depth, zero, one, feature);
    if (node < 0) {
      const value = f.value[leafBase + ~node];
      for (let i = 1; i <= depth; i++) {
        const w = unwoundSum(p, o, depth, i);
        phi[p.feature[o + i]] += w * (p.one[o + i] - p.zero[o + i]) * value;
      }
      return;
    }
    const i = base + node;
    const split = f.feature[i];
    const toLeft = goesLeft(x[split], f.decision[i], f.threshold[i]);
    const hot = toLeft ? f.left[i] : f.right[i];
    const cold = toLeft ? f.right[i] : f.left[i];
    const total = count(node);
    const hotZero = count(hot) / total;
    const coldZero = count(cold) / total;
    let inZero = 1.0;
    let inOne = 1.0;
    for (let k = 1; k <= depth; k++) {
      if (p.feature[o + k] === split) {
        inZero = p.zero[o + k];
        inOne = p.one[o + k];
        unwind(p, o, depth, k);
        depth -= 1;
        break;
      }
    }
    recurse(hot, o, depth + 1, hotZero * inZero, inOne, split);
    recurse(cold, o, depth + 1, coldZero * inZero, 0.0, split);
  };

  if (f.treeNodes[t] === 0) return;
  recurse(0, 0, 0, 1.0, 1.0, -1);
}

/**
 * Average raw score over the training rows: the base of every explanation.
 * @param {Forest} f
 */
export function expectedValue(f) {
  if (!f.nodeCount || !f.leafCount) throw new Error("this tree file cannot explain predictions");
  let total = 0.0;
  for (let t = 0; t < f.nTrees; t++) {
    const leafBase = f.leafStart[t];
    const nLeaves = f.treeNodes[t] + 1;
    let dot = 0.0;
    let sum = 0.0;
    let plain = 0.0;
    for (let j = 0; j < nLeaves; j++) {
      dot += f.leafCount[leafBase + j] * f.value[leafBase + j];
      sum += f.leafCount[leafBase + j];
      plain += f.value[leafBase + j];
    }
    total += sum > 0 ? dot / sum : plain / nLeaves;
  }
  return total;
}

/**
 * Splits a raw score into a base plus one contribution per feature (TreeSHAP):
 * base + sum(contributions) equals predictRow(f, row).
 * @param {Forest} f
 * @param {ArrayLike<number>} row
 * @returns {{base: number, contributions: Float64Array}}
 */
export function explainRow(f, row) {
  if (!f.nodeCount || !f.leafCount) throw new Error("this tree file cannot explain predictions");
  if (row.length !== f.nFeatures) throw new Error(`expected ${f.nFeatures} features, got ${row.length}`);
  const x = Float64Array.from(row, (v) => Math.fround(v));
  const phi = new Float64Array(f.nFeatures);
  let deepest = 0;
  for (let t = 0; t < f.nTrees; t++) deepest = Math.max(deepest, f.treeNodes[t]);
  const levels = deepest + 3;
  const paths = new Paths((levels * (levels + 1)) / 2 + levels);
  for (let t = 0; t < f.nTrees; t++) treeShap(f, t, x, phi, paths);
  return { base: expectedValue(f), contributions: phi };
}
