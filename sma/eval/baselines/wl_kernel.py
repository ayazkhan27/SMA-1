"""Weisfeiler-Leman graph-kernel control over SMA's own Tier-0 cases.

The internal control for the ladder: does *generic* graph similarity computed
on the SAME deterministic extraction (sma.encoders get_encoder("logs")) match
structure-mapping retrieval? If yes, SMA's edge is the extraction; if no, the
edge is the mapping mathematics.

Graph construction (per case):
  - nodes  = unique expressions (keyed by canonical s-expression) plus unique
    entities (keyed by (name, type));
  - edges  = statement -> argument, position-annotated.

Node labels (iteration 0):
  - statements: the functor (exactly what SMA's own content vectors see);
  - entities whose names are arbitrary per-session identifiers (types
    "event", "session") are labeled by TYPE ONLY. This is not a kindness, it
    is a necessity: the session entity touches nearly every statement, so a
    case-unique session name would make every refined label case-unique after
    one iteration and the kernel would degenerate to zero similarity.
  - all other entities (components, event_type tokens, integer counts) keep
    "type:name" -- their names are shared vocabulary, i.e. real content.

Refinement: 2 WL iterations; new label = hash(own label, position-sorted child
labels, sorted parent labels). Similarity = cosine over the concatenated label
histograms of iterations 0..2 (Shervashidze et al. 2011, WL subtree kernel,
normalized).
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
from scipy import sparse

from sma.ir.schema import Case, Entity, Statement
from sma.ir.sexpr import dumps_statement

# Entity types whose names are arbitrary per-session identifiers.
_ID_LIKE_TYPES = frozenset({"event", "session"})

WL_ITERATIONS = 2


def _entity_label(ent: Entity) -> str:
    if ent.type in _ID_LIKE_TYPES:
        return f"ent:{ent.type}"
    return f"ent:{ent.type}:{ent.name}"


def _hash_label(payload: str, memo: dict[str, str]) -> str:
    cached = memo.get(payload)
    if cached is None:
        cached = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()
        memo[payload] = cached
    return cached


def wl_histogram(case: Case, iterations: int = WL_ITERATIONS) -> Counter[str]:
    """Concatenated WL label histogram (iterations 0..n) for one case."""
    # --- build the node/edge structure ---------------------------------
    node_labels: list[str] = []
    children: list[list[tuple[int, int]]] = []  # node -> [(arg_pos, child_node)]
    parents: list[list[int]] = []
    stmt_idx: dict[str, int] = {}
    ent_idx: dict[tuple[str, str], int] = {}

    def add_node(label: str) -> int:
        node_labels.append(label)
        children.append([])
        parents.append([])
        return len(node_labels) - 1

    def visit(node: Statement | Entity) -> int:
        if isinstance(node, Entity):
            key = (node.name, node.type)
            idx = ent_idx.get(key)
            if idx is None:
                idx = add_node(_entity_label(node))
                ent_idx[key] = idx
            return idx
        skey = dumps_statement(node)
        idx = stmt_idx.get(skey)
        if idx is not None:
            return idx
        idx = add_node(f"f:{node.functor}")
        stmt_idx[skey] = idx
        for pos, arg in enumerate(node.args):
            child = visit(arg)
            children[idx].append((pos, child))
            parents[child].append(idx)
        return idx

    for statement in case.statements:
        visit(statement)

    # --- WL refinement ---------------------------------------------------
    memo: dict[str, str] = {}
    histogram: Counter[str] = Counter()
    labels = list(node_labels)
    for node_label in labels:
        histogram[f"wl0:{node_label}"] += 1
    for it in range(1, iterations + 1):
        new_labels = []
        for idx, own in enumerate(labels):
            child_part = sorted(f"c{pos}:{labels[child]}" for pos, child in children[idx])
            parent_part = sorted(f"p:{labels[p]}" for p in parents[idx])
            payload = own + "|" + ",".join(child_part) + "|" + ",".join(parent_part)
            new_labels.append(_hash_label(payload, memo))
        labels = new_labels
        for node_label in labels:
            histogram[f"wl{it}:{node_label}"] += 1
    return histogram


class WLKernelRetriever:
    """Cosine-normalized WL subtree kernel retrieval over encoded cases."""

    def __init__(self, iterations: int = WL_ITERATIONS):
        self.iterations = iterations
        self.doc_ids: list[str] = []
        self.feature_index: dict[str, int] = {}
        self.doc_matrix: sparse.csr_matrix | None = None  # rows L2-normalized

    def build(self, cases: list[Case]) -> None:
        histograms = [wl_histogram(c, self.iterations) for c in cases]
        self.doc_ids = [c.case_id for c in cases]
        self.feature_index = {}
        for hist in histograms:
            for feat in hist:
                if feat not in self.feature_index:
                    self.feature_index[feat] = len(self.feature_index)
        rows, cols, vals = [], [], []
        for row, hist in enumerate(histograms):
            for feat, count in hist.items():
                rows.append(row)
                cols.append(self.feature_index[feat])
                vals.append(float(count))
        matrix = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(histograms), max(len(self.feature_index), 1))
        )
        norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1.0
        self.doc_matrix = sparse.diags(1.0 / norms) @ matrix

    def retrieve(self, query_case: Case, k: int = 10) -> list[tuple[str, float]]:
        if self.doc_matrix is None:
            return []
        hist = wl_histogram(query_case, self.iterations)
        # Query norm uses the FULL histogram (including features absent from
        # the index vocabulary) so cosine is honest, not inflated.
        q_norm = float(np.sqrt(sum(v * v for v in hist.values()))) or 1.0
        q = np.zeros(self.doc_matrix.shape[1])
        for feat, count in hist.items():
            col = self.feature_index.get(feat)
            if col is not None:
                q[col] = count / q_norm
        scores = self.doc_matrix @ q
        ranked = sorted(
            zip(self.doc_ids, map(float, scores)), key=lambda row: (-row[1], row[0])
        )
        return ranked[:k]
