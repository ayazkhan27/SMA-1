"""Structural-fraud arm: graph-neighbourhood encoding of the Elliptic Bitcoin
transaction graph for retrieval-by-analogy illicit detection.

The flat-tabular finance null (4b) showed SMA has no edge when each record is
encoded independently: there is no cross-record structure to map. Elliptic is a
*graph* (≈203k transaction nodes, 166 features, plus a directed bitcoin-flow
edgelist), so it carries the predecessor/successor topology a single flat row
lacks. This module encodes each transaction's local neighbourhood as a case of
**higher-order relations** over a licit/illicit *typology lattice* — fan-in/out
degree class, in/out value tier, temporal step, and (leak-guarded) neighbour
label context wired by ``flowsFrom``/``flowsTo`` — so SMA can structure-map an
illicit-pattern analog where flat/vector methods see only an isolated vector.
"""

from __future__ import annotations

from sma.eval.fraud_elliptic.encoder import (
    EllipticGraph,
    NeighbourhoodEncoder,
    build_typology,
    load_elliptic,
)

__all__ = [
    "EllipticGraph",
    "NeighbourhoodEncoder",
    "build_typology",
    "load_elliptic",
]
