# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)

"""LightGBM tree ensembles: text-model parser, compact binary format, reference evaluator.

The browser cannot run LightGBM, so the trained trees are exported to a small binary file that a
few lines of JavaScript can walk. This module is the single source of truth for that format and
holds the Python evaluator the JavaScript one is tested against, decision rule for decision rule:

* a missing value (NaN) is replaced by 0.0 unless the split's missing type is "NaN";
* "Zero" missing type: values in [-1e-35, 1e-35] go to the default side; "NaN": NaN does;
* otherwise `value <= threshold` goes left. Inputs are float32, as LightGBM sees them.

Binary layout, little-endian, all arrays contiguous:
  header  : magic b"VTREES01", uint32 n_features, uint32 n_trees, uint32 n_nodes, uint32 n_leaves
  per tree: int32 node_start[n_trees], int32 leaf_start[n_trees], int32 n_tree_nodes[n_trees]
  per node: uint8 feature[n_nodes], uint8 decision[n_nodes], float64 threshold[n_nodes],
            int32 left[n_nodes], int32 right[n_nodes]     (child >= 0: node, child < 0: leaf ~child)
  per leaf: float64 value[n_leaves]
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"VTREES01"
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
        if n_leaves > 1:
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
    data = b"".join(parts)
    path.write_bytes(data)
    path.with_suffix(".json").write_text(json.dumps({
        "format": "valia-trees/1", "features": forest.features, "trees": forest.n_trees,
        "nodes": len(forest.feature), "leaves": len(forest.value), "bytes": len(data),
    }, indent=2))
    return len(data)


def read_binary(path: Path) -> Forest:
    data = path.read_bytes()
    if data[:8] != MAGIC:
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

    return Forest(features, take("<i4", n_trees), take("<i4", n_trees), take("<i4", n_trees),
                  take("u1", n_nodes), take("u1", n_nodes), take("<f8", n_nodes),
                  take("<i4", n_nodes), take("<i4", n_nodes), take("<f8", n_leaves))
