# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""LightGBM tree ensembles: text-model parser, compact binary format, reference evaluator.

The browser cannot run LightGBM, so the trained trees are exported to a small binary file that a
few lines of JavaScript can walk. This module is the single source of truth for that format and
holds the Python evaluator the JavaScript one is tested against, decision rule for decision rule:

* a missing value (NaN) is replaced by 0.0 unless the split's missing type is "NaN";
* "Zero" missing type: values in [-1e-35, 1e-35] go to the default side; "NaN": NaN does;
* otherwise `value <= threshold` goes left. Inputs are float32, as LightGBM sees them.

It also explains a prediction with TreeSHAP (Lundberg et al., 2020, path-dependent algorithm): the
raw score is split exactly into a base value plus one contribution per feature, using how many
training rows went through each node. This is what LightGBM's `pred_contrib` returns.

Binary layout, little-endian, all arrays contiguous:
  header  : magic b"VTREES02", uint32 n_features, uint32 n_trees, uint32 n_nodes, uint32 n_leaves
  per tree: int32 node_start[n_trees], int32 leaf_start[n_trees], int32 n_tree_nodes[n_trees]
  per node: uint8 feature[n_nodes], uint8 decision[n_nodes], float64 threshold[n_nodes],
            int32 left[n_nodes], int32 right[n_nodes]     (child >= 0: node, child < 0: leaf ~child)
  per leaf: float64 value[n_leaves]
  counts  : uint32 node_count[n_nodes], uint32 leaf_count[n_leaves]   (training rows per node/leaf)
"VTREES01" files are the same without the counts: they predict but cannot explain.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"VTREES02"
MAGIC_V1 = b"VTREES01"
ZERO_THRESHOLD = 1e-35
DEFAULT_LEFT = 2
MISSING_NONE, MISSING_ZERO, MISSING_NAN = 0, 1, 2


@dataclass
class Forest:
    features: list[str]
    node_start: np.ndarray      # int32 per tree
    leaf_start: np.ndarray      # int32 per tree
    tree_nodes: np.ndarray      # int32 per tree (0 for a single-leaf tree)
    feature: np.ndarray         # uint8 per node
    decision: np.ndarray        # uint8 per node
    threshold: np.ndarray       # float64 per node
    left: np.ndarray            # int32 per node
    right: np.ndarray           # int32 per node
    value: np.ndarray           # float64 per leaf
    node_count: np.ndarray | None = None    # training rows through each node (for TreeSHAP)
    leaf_count: np.ndarray | None = None    # training rows in each leaf

    @property
    def n_trees(self) -> int:
        return len(self.node_start)


def _floats(line: str) -> list[float]:
    return [float(x) for x in line.split()]


def _ints(line: str) -> list[int]:
    return [int(x) for x in line.split()]


def parse_lightgbm(text: str) -> Forest:
    """Parse LightGBM's text model (`booster.save_model`)."""
    header, _, body = text.partition("\nTree=")
    features: list[str] = []
    for line in header.splitlines():
        if line.startswith("feature_names="):
            features = line.split("=", 1)[1].split()
    if not features:
        raise ValueError("not a LightGBM text model: no feature_names")
    if len(features) > 255:
        raise ValueError("the binary format stores feature indices on one byte")

    node_start, leaf_start, tree_nodes = [], [], []
    feature, decision, threshold, left, right, value = [], [], [], [], [], []
    node_count: list[float] = []
    leaf_count: list[float] = []
    for block in ("Tree=" + body).split("\nTree=")[0:]:
        if not block.strip() or block.startswith("end of trees"):
            continue
        fields = {}
        for line in block.splitlines():
            if "=" in line:
                key, val = line.split("=", 1)
                fields[key] = val
        if "num_leaves" not in fields:
            continue
        if fields.get("num_cat", "0") != "0":
            raise ValueError("categorical splits are not supported by the binary format")
        n_leaves = int(fields["num_leaves"])
        node_start.append(len(feature))
        leaf_start.append(len(value))
        tree_nodes.append(n_leaves - 1)
        value.extend(_floats(fields["leaf_value"]))
        leaf_count.extend(_floats(fields.get("leaf_count", "")) or [1.0] * n_leaves)
        if n_leaves > 1:
            counts = _floats(fields.get("internal_count", ""))
            node_count.extend(counts or [float(n_leaves)] * (n_leaves - 1))
            feature.extend(_ints(fields["split_feature"]))
            decision.extend(_ints(fields["decision_type"]))
            threshold.extend(_floats(fields["threshold"]))
            left.extend(_ints(fields["left_child"]))
            right.extend(_ints(fields["right_child"]))
    if not node_start:
        raise ValueError("no trees found in the model")
    return Forest(
        features=features,
        node_start=np.array(node_start, dtype=np.int32),
        leaf_start=np.array(leaf_start, dtype=np.int32),
        tree_nodes=np.array(tree_nodes, dtype=np.int32),
        feature=np.array(feature, dtype=np.uint8),
        decision=np.array(decision, dtype=np.uint8),
        threshold=np.array(threshold, dtype=np.float64),
        left=np.array(left, dtype=np.int32),
        right=np.array(right, dtype=np.int32),
        value=np.array(value, dtype=np.float64),
        node_count=np.array(node_count, dtype=np.float64),
        leaf_count=np.array(leaf_count, dtype=np.float64),
    )


def _goes_left(x: float, decision: int, threshold: float) -> bool:
    missing = (decision >> 2) & 3
    if np.isnan(x) and missing != MISSING_NAN:
        x = 0.0
    if (missing == MISSING_ZERO and -ZERO_THRESHOLD <= x <= ZERO_THRESHOLD) or \
            (missing == MISSING_NAN and np.isnan(x)):
        return bool(decision & DEFAULT_LEFT)
    return x <= threshold


def predict_row(forest: Forest, row: np.ndarray) -> float:
    """Raw score for one row of features (float32 semantics, as LightGBM)."""
    row = row.astype(np.float32).astype(np.float64)
    total = 0.0
    for t in range(forest.n_trees):
        base, leaf_base = int(forest.node_start[t]), int(forest.leaf_start[t])
        if forest.tree_nodes[t] == 0:
            total += forest.value[leaf_base]
            continue
        node = 0
        while node >= 0:
            i = base + node
            if _goes_left(row[forest.feature[i]], int(forest.decision[i]), forest.threshold[i]):
                node = int(forest.left[i])
            else:
                node = int(forest.right[i])
        total += forest.value[leaf_base + ~node]
    return total


def predict(forest: Forest, X: np.ndarray) -> np.ndarray:
    return np.array([predict_row(forest, row) for row in np.asarray(X, dtype=np.float64)])


def write_binary(forest: Forest, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        MAGIC,
        struct.pack("<4I", len(forest.features), forest.n_trees, len(forest.feature), len(forest.value)),
        forest.node_start.astype("<i4").tobytes(), forest.leaf_start.astype("<i4").tobytes(),
        forest.tree_nodes.astype("<i4").tobytes(),
        forest.feature.astype("u1").tobytes(), forest.decision.astype("u1").tobytes(),
        forest.threshold.astype("<f8").tobytes(),
        forest.left.astype("<i4").tobytes(), forest.right.astype("<i4").tobytes(),
        forest.value.astype("<f8").tobytes(),
    ]
    if forest.node_count is not None and forest.leaf_count is not None:
        parts += [forest.node_count.astype("<u4").tobytes(), forest.leaf_count.astype("<u4").tobytes()]
    else:
        parts[0] = MAGIC_V1
    data = b"".join(parts)
    path.write_bytes(data)
    path.with_suffix(".json").write_text(json.dumps({
        "format": "valia-trees/2" if data[:8] == MAGIC else "valia-trees/1",
        "features": forest.features, "trees": forest.n_trees,
        "nodes": len(forest.feature), "leaves": len(forest.value), "bytes": len(data),
    }, indent=2))
    return len(data)


def read_binary(path: Path) -> Forest:
    data = path.read_bytes()
    if data[:8] not in (MAGIC, MAGIC_V1):
        raise ValueError("not a valia tree file")
    n_features, n_trees, n_nodes, n_leaves = struct.unpack_from("<4I", data, 8)
    features = json.loads(path.with_suffix(".json").read_text())["features"]
    if len(features) != n_features:
        raise ValueError("feature list does not match the binary header")
    offset = 24

    def take(dtype: str, count: int) -> np.ndarray:
        nonlocal offset
        arr = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
        offset += arr.nbytes
        return arr.copy()

    forest = Forest(features, take("<i4", n_trees), take("<i4", n_trees), take("<i4", n_trees),
                    take("u1", n_nodes), take("u1", n_nodes), take("<f8", n_nodes),
                    take("<i4", n_nodes), take("<i4", n_nodes), take("<f8", n_leaves))
    if data[:8] == MAGIC:
        forest.node_count = take("<u4", n_nodes).astype(np.float64)
        forest.leaf_count = take("<u4", n_leaves).astype(np.float64)
    if offset != len(data):
        raise ValueError("tree file has an unexpected size")
    return forest


# ---- TreeSHAP ---------------------------------------------------------------------------------
# Path-dependent TreeSHAP, Algorithm 2 of Lundberg et al. (2020), as in shap and LightGBM.
# web/app/trees.js mirrors this code line by line.

class _Path:
    """The unique features met on the way to a node: index, share of rows when the feature is
    'absent' (zero), whether x follows that branch (one), and the permutation weight."""

    __slots__ = ("feature", "one", "weight", "zero")

    def __init__(self, n: int) -> None:
        self.feature = [-1] * n
        self.zero = [0.0] * n
        self.one = [0.0] * n
        self.weight = [0.0] * n

    def copy(self) -> _Path:
        out = _Path(0)
        out.feature, out.zero, out.one, out.weight = self.feature[:], self.zero[:], self.one[:], self.weight[:]
        return out


def _extend(p: _Path, depth: int, zero: float, one: float, feature: int) -> None:
    p.feature[depth], p.zero[depth], p.one[depth] = feature, zero, one
    p.weight[depth] = 1.0 if depth == 0 else 0.0
    for i in range(depth - 1, -1, -1):
        p.weight[i + 1] += one * p.weight[i] * (i + 1) / (depth + 1)
        p.weight[i] = zero * p.weight[i] * (depth - i) / (depth + 1)


def _unwind(p: _Path, depth: int, index: int) -> None:
    one, zero = p.one[index], p.zero[index]
    next_one = p.weight[depth]
    for i in range(depth - 1, -1, -1):
        if one != 0:
            tmp = p.weight[i]
            p.weight[i] = next_one * (depth + 1) / ((i + 1) * one)
            next_one = tmp - p.weight[i] * zero * (depth - i) / (depth + 1)
        else:
            p.weight[i] = p.weight[i] * (depth + 1) / (zero * (depth - i))
    for i in range(index, depth):
        p.feature[i], p.zero[i], p.one[i] = p.feature[i + 1], p.zero[i + 1], p.one[i + 1]


def _unwound_sum(p: _Path, depth: int, index: int) -> float:
    one, zero = p.one[index], p.zero[index]
    next_one = p.weight[depth]
    total = 0.0
    for i in range(depth - 1, -1, -1):
        if one != 0:
            tmp = next_one * (depth + 1) / ((i + 1) * one)
            total += tmp
            next_one = p.weight[i] - tmp * zero * (depth - i) / (depth + 1)
        else:
            total += p.weight[i] / (zero * (depth - i) / (depth + 1))
    return total


def _tree_shap(forest: Forest, t: int, x: np.ndarray, phi: np.ndarray) -> None:
    base, leaf_base = int(forest.node_start[t]), int(forest.leaf_start[t])
    n_nodes = int(forest.tree_nodes[t])

    def count(child: int) -> float:
        return float(forest.leaf_count[leaf_base + ~child] if child < 0 else forest.node_count[base + child])

    def recurse(node: int, p: _Path, depth: int, zero: float, one: float, feature: int) -> None:
        _extend(p, depth, zero, one, feature)
        if node < 0:
            value = float(forest.value[leaf_base + ~node])
            for i in range(1, depth + 1):
                w = _unwound_sum(p, depth, i)
                phi[p.feature[i]] += w * (p.one[i] - p.zero[i]) * value
            return
        i = base + node
        split = int(forest.feature[i])
        left, right = int(forest.left[i]), int(forest.right[i])
        goes_left = _goes_left(x[split], int(forest.decision[i]), forest.threshold[i])
        hot, cold = (left, right) if goes_left else (right, left)
        total = count(node)
        hot_zero, cold_zero = count(hot) / total, count(cold) / total
        in_zero, in_one = 1.0, 1.0
        for k in range(1, depth + 1):                       # the feature was already split on: undo it
            if p.feature[k] == split:
                in_zero, in_one = p.zero[k], p.one[k]
                _unwind(p, depth, k)
                depth -= 1
                break
        recurse(hot, p.copy(), depth + 1, hot_zero * in_zero, in_one, split)
        recurse(cold, p.copy(), depth + 1, cold_zero * in_zero, 0.0, split)

    if n_nodes == 0:
        return
    recurse(0, _Path(n_nodes + 2), 0, 1.0, 1.0, -1)


def expected_value(forest: Forest) -> float:
    """Average raw score over the training rows: the base of every explanation."""
    if forest.node_count is None or forest.leaf_count is None:
        raise ValueError("this tree file has no node counts: it cannot explain predictions")
    total = 0.0
    for t in range(forest.n_trees):
        leaf_base = int(forest.leaf_start[t])
        n_leaves = int(forest.tree_nodes[t]) + 1
        counts = forest.leaf_count[leaf_base:leaf_base + n_leaves]
        values = forest.value[leaf_base:leaf_base + n_leaves]
        total += float(np.dot(counts, values) / counts.sum()) if counts.sum() > 0 else float(values.mean())
    return total


def explain_row(forest: Forest, row: np.ndarray) -> tuple[float, np.ndarray]:
    """(base, contributions): base + contributions.sum() equals predict_row(forest, row)."""
    if forest.node_count is None or forest.leaf_count is None:
        raise ValueError("this tree file has no node counts: it cannot explain predictions")
    x = row.astype(np.float32).astype(np.float64)
    phi = np.zeros(len(forest.features))
    for t in range(forest.n_trees):
        _tree_shap(forest, t, x, phi)
    return expected_value(forest), phi
