"""Graph-neighbourhood encoder + licit/illicit typology lattice for Elliptic.

The Elliptic dataset ships three files:

* ``elliptic_txs_features.csv`` — no header; col 0 = txId, col 1 = time-step
  (1..49), cols 2..167 = 166 anonymized features (the first ~93 are local tx
  features, the rest are aggregations of one-hop neighbour features);
* ``elliptic_txs_classes.csv`` — ``txId,class`` with class in {1 (illicit),
  2 (licit), unknown};
* ``elliptic_txs_edgelist.csv`` — ``txId1,txId2`` directed bitcoin flows.

The encoder turns one transaction into a *case over a typology of
graph-neighbourhood descriptors*, NOT the 166 flat features:

  - ``fanIn_*`` / ``fanOut_*`` — predecessor / successor degree class;
  - ``inVal_*`` / ``outVal_*`` — value tier of incoming / outgoing flow,
    read from the local value feature aggregated over neighbours;
  - ``temp_*`` — temporal-step bucket;
  - ``nbrIllicit_*`` / ``nbrLicit_*`` — neighbour *label context*: how many
    predecessors / successors are known-illicit / known-licit. **Label-leak
    guard:** a node's OWN class is never emitted, and neighbour labels are
    only those visible in the indexed (train) split passed to the encoder.

These descriptor terms hang off an is-a typology lattice (e.g. ``fanOut_high``
is_a ``fanOut_any`` is_a ``flowTopology``; ``nbrIllicit_many`` is_a
``illicitContext`` is_a ``neighbourContext``), so SMA can ascend a too-specific
observation to a shared ancestor when structure-mapping. Higher-order
``flowsFrom`` / ``flowsTo`` relations wire the node's own topology descriptor to
its neighbour-context descriptor, giving the cross-record structure flat-tabular
encodings discard.
"""

from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass, field

from sma.ontology.graph import OntologyGraph, Term

# Elliptic class codes (string, as they appear in the CSV).
ILLICIT, LICIT, UNKNOWN = "1", "2", "unknown"

# Feature-column layout in elliptic_txs_features.csv (0-based over the CSV row).
COL_TXID = 0
COL_TIME = 1
# Local features start at col 2. Two anonymized local features used as proxy
# value channels (the dataset is anonymized; these are stable per-tx scalars).
COL_LOCAL_VALUE = 2  # first local feature — proxy for a transaction-value channel
COL_AGG_VALUE = 95   # first aggregated (neighbour) feature — neighbour-value proxy


def _tier(value: float, lo: float, hi: float) -> str:
    """Three-way tier label for a z-scored feature: low / mid / high."""
    if value <= lo:
        return "low"
    if value >= hi:
        return "high"
    return "mid"


def _degree_class(deg: int) -> str:
    """Bucket a degree into none / one / few / many."""
    if deg == 0:
        return "none"
    if deg == 1:
        return "one"
    if deg <= 4:
        return "few"
    return "many"


def _count_class(count: int) -> str:
    """Bucket a neighbour-label count into none / some / many."""
    if count == 0:
        return "none"
    if count <= 2:
        return "some"
    return "many"


@dataclass
class EllipticGraph:
    """In-memory Elliptic transaction graph."""

    time_step: dict[str, int]
    label: dict[str, str]  # txId -> {"1","2","unknown"}
    feats: dict[str, list[float]]  # txId -> full feature row (incl. time at idx 0)
    preds: dict[str, list[str]] = field(default_factory=dict)  # txId -> predecessors
    succs: dict[str, list[str]] = field(default_factory=dict)  # txId -> successors

    def labelled_ids(self) -> list[str]:
        """Sorted txIds with a known (non-unknown) label."""
        return sorted(t for t, c in self.label.items() if c in (ILLICIT, LICIT))


def load_elliptic(data_dir: str) -> EllipticGraph:
    """Load the three Elliptic CSVs from ``data_dir`` into an :class:`EllipticGraph`."""
    d = pathlib.Path(data_dir)
    feats_p = d / "elliptic_txs_features.csv"
    classes_p = d / "elliptic_txs_classes.csv"
    edges_p = d / "elliptic_txs_edgelist.csv"

    feats: dict[str, list[float]] = {}
    time_step: dict[str, int] = {}
    with feats_p.open(encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            txid = row[COL_TXID]
            vals = [float(x) for x in row[1:]]  # idx0 = time-step, then 166 feats
            feats[txid] = vals
            time_step[txid] = int(round(vals[0]))

    label: dict[str, str] = {}
    with classes_p.open(encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd, None)  # header: txId,class
        for row in rd:
            if len(row) >= 2:
                label[row[0]] = row[1]

    preds: dict[str, list[str]] = {t: [] for t in feats}
    succs: dict[str, list[str]] = {t: [] for t in feats}
    with edges_p.open(encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd, None)  # header: txId1,txId2
        for row in rd:
            if len(row) < 2:
                continue
            a, b = row[0], row[1]
            if a in succs and b in preds:
                succs[a].append(b)  # a -> b : a flows to b
                preds[b].append(a)  # b has predecessor a

    return EllipticGraph(time_step=time_step, label=label, feats=feats, preds=preds, succs=succs)


# Typology vocabulary -------------------------------------------------------
# Each tuple is (term_id, parent_id). Roots have parent "".
_DEGREE_BUCKETS = ("none", "one", "few", "many")
_TIERS = ("low", "mid", "high")
_TEMP_BUCKETS = ("early", "mid", "late")
_COUNT_BUCKETS = ("none", "some", "many")


def build_typology(name: str = "elliptic_typology") -> OntologyGraph:
    """Build the licit/illicit graph-neighbourhood typology lattice.

    Returns an :class:`OntologyGraph` whose is-a edges let a specific descriptor
    (e.g. ``fanOut_high``) ascend to a shared ancestor (``fanOut_any`` ->
    ``flowTopology``) during structure-mapping. Mounting this graph populates the
    predicate lattice; ``NeighbourhoodEncoder`` emits cases over these term ids.
    """
    terms: dict[str, Term] = {}

    def add(tid: str, parent: str, nm: str = "") -> None:
        terms[tid] = Term(id=tid, name=nm or tid.replace("_", " "),
                          parents=(parent,) if parent else ())

    # Roots of the typology.
    add("flowTopology", "")
    add("valueProfile", "")
    add("temporalProfile", "")
    add("neighbourContext", "")
    # Two licit/illicit typology poles: descriptor families subsume into these so
    # an "illicit-looking neighbourhood" can match across distinct surface forms.
    add("illicitTypology", "")
    add("licitTypology", "")

    # fan-in / fan-out degree classes.
    add("fanIn_any", "flowTopology")
    add("fanOut_any", "flowTopology")
    for b in _DEGREE_BUCKETS:
        add(f"fanIn_{b}", "fanIn_any")
        add(f"fanOut_{b}", "fanOut_any")
    # High fan-out / high fan-in are illicit-typology cues (layering / dispersal).
    terms["fanOut_many"].parents = ("fanOut_any", "illicitTypology")
    terms["fanIn_many"].parents = ("fanIn_any", "illicitTypology")
    terms["fanOut_one"].parents = ("fanOut_any", "licitTypology")

    # value tiers (incoming / outgoing).
    add("inVal_any", "valueProfile")
    add("outVal_any", "valueProfile")
    for t in _TIERS:
        add(f"inVal_{t}", "inVal_any")
        add(f"outVal_{t}", "outVal_any")

    # temporal buckets.
    add("temp_any", "temporalProfile")
    for b in _TEMP_BUCKETS:
        add(f"temp_{b}", "temp_any")

    # neighbour label context — known illicit / licit predecessors & successors.
    add("illicitContext", "neighbourContext")
    add("licitContext", "neighbourContext")
    # illicitContext rolls up into the illicit typology pole.
    terms["illicitContext"].parents = ("neighbourContext", "illicitTypology")
    terms["licitContext"].parents = ("neighbourContext", "licitTypology")
    for c in _COUNT_BUCKETS:
        add(f"nbrIllicit_{c}", "illicitContext")
        add(f"nbrLicit_{c}", "licitContext")
    # "no illicit neighbours" is not itself an illicit cue: re-parent to context.
    terms["nbrIllicit_none"].parents = ("neighbourContext",)
    terms["nbrLicit_none"].parents = ("neighbourContext",)

    # Typed higher-order relations: a node's own out-topology *flowsTo* its
    # successor/neighbour label context, and its in-topology *flowsFrom* its
    # predecessor context. When both endpoints co-occur in a case,
    # mount().build_case emits ``flowsTo(fanOut(subj), nbrIllicit(subj))`` — the
    # cross-record structure SMA maps and flat encodings discard. Relations are
    # declared on the degree-bucket terms (one endpoint) toward the context
    # buckets; build_case only materializes a relation when BOTH terms are
    # present on the same transaction.
    def wire(src: str, rel: str, dsts: tuple[str, ...]) -> None:
        terms[src].relations = terms[src].relations + tuple((rel, d) for d in dsts)

    illicit_ctx = tuple(f"nbrIllicit_{c}" for c in _COUNT_BUCKETS)
    licit_ctx = tuple(f"nbrLicit_{c}" for c in _COUNT_BUCKETS)
    for b in _DEGREE_BUCKETS:
        wire(f"fanOut_{b}", "flowsTo", illicit_ctx + licit_ctx)
        wire(f"fanIn_{b}", "flowsFrom", illicit_ctx + licit_ctx)

    return OntologyGraph(name=name, terms=terms)


@dataclass
class NeighbourhoodEncoder:
    """Encode a transaction's local graph neighbourhood into typology term ids.

    ``visible_labels`` is the label map the encoder is allowed to READ for
    neighbour context — pass the train/index split's labels so test-node
    neighbour context never peeks at held-out labels. A node's OWN class is
    never emitted regardless. ``lo``/``hi`` are the z-score tier cut points.
    """

    graph: EllipticGraph
    visible_labels: dict[str, str]
    lo: float = -0.3
    hi: float = 0.3

    def encode(self, txid: str) -> list[str]:
        """Return the sorted typology term ids describing ``txid``'s neighbourhood."""
        g = self.graph
        terms: list[str] = []

        preds = g.preds.get(txid, [])
        succs = g.succs.get(txid, [])
        terms.append(f"fanIn_{_degree_class(len(preds))}")
        terms.append(f"fanOut_{_degree_class(len(succs))}")

        feats = g.feats.get(txid, [])
        # feats[0] is the time-step; local & aggregated value channels follow.
        in_val = feats[COL_AGG_VALUE - 1] if len(feats) > COL_AGG_VALUE - 1 else 0.0
        out_val = feats[COL_LOCAL_VALUE - 1] if len(feats) > COL_LOCAL_VALUE - 1 else 0.0
        terms.append(f"inVal_{_tier(in_val, self.lo, self.hi)}")
        terms.append(f"outVal_{_tier(out_val, self.lo, self.hi)}")

        ts = g.time_step.get(txid, 1)
        if ts <= 16:
            terms.append("temp_early")
        elif ts <= 33:
            terms.append("temp_mid")
        else:
            terms.append("temp_late")

        # Neighbour LABEL context — leak-guarded: only visible_labels, never self.
        n_illicit = sum(
            1 for n in preds + succs
            if n != txid and self.visible_labels.get(n) == ILLICIT
        )
        n_licit = sum(
            1 for n in preds + succs
            if n != txid and self.visible_labels.get(n) == LICIT
        )
        terms.append(f"nbrIllicit_{_count_class(n_illicit)}")
        terms.append(f"nbrLicit_{_count_class(n_licit)}")

        return sorted(set(terms))
