"""Mount a normalized :class:`OntologyGraph` onto SMA's matching machinery.

Mounting lifts an ontology's is_a hierarchy into a :class:`Canonicalizer`
predicate lattice (so structurally-distinct-but-related terms can ascend to a
shared ancestor during matching) and provides the case/index builders that turn
a set of present terms into an SMA :class:`Case`.

A term ``T`` present on a subject becomes the statement ``fid(T)(subject)``;
each typed relation ``(s, rel, o)`` whose *both* endpoints are present becomes
the higher-order statement ``rel(fid(s)(subject), fid(o)(subject))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from sma.index.macfac import MacFacIndex
from sma.ir.canon import Canonicalizer
from sma.ir.schema import Case, entity, make_case, stmt
from sma.match.types import MatchConfig

from .graph import OntologyGraph
from .loader import fid


def _default_config() -> MatchConfig:
    """Ontology matching default: ascend up to two is_a hops (``delta=2``)."""
    return MatchConfig(delta=2, rho=0.95)


@dataclass
class MountedOntology:
    """An :class:`OntologyGraph` bound to a populated :class:`Canonicalizer`."""

    graph: OntologyGraph
    canon: Canonicalizer
    config: MatchConfig

    def build_case(
        self,
        term_ids: Iterable[str],
        subject: str = "subject",
        metadata: Mapping[str, Any] | None = None,
    ) -> Case:
        """Build a :class:`Case` for the given present ``term_ids``.

        Unknown term ids (not in ``graph.terms``) are dropped. Each present term
        contributes ``fid(term)(subject)``; each typed relation with both
        endpoints present contributes the higher-order
        ``rel(fid(s)(subject), fid(o)(subject))``.
        """
        present = [t for t in term_ids if t in self.graph.terms]
        present_set = set(present)
        subj = entity(subject, subject)

        statements = [stmt(fid(t), subj) for t in present]
        for s, rel, o in self.graph.typed_relations():
            if s in present_set and o in present_set:
                statements.append(stmt(rel, stmt(fid(s), subj), stmt(fid(o), subj)))

        return make_case(statements, metadata=metadata)

    def build_index(
        self,
        records: Iterable[tuple[str, Iterable[str], Mapping[str, Any] | None]],
        config: MatchConfig | None = None,
    ) -> MacFacIndex:
        """Build a :class:`MacFacIndex` over ``(key, term_ids, metadata)`` records.

        Each record's ``key`` is stored under ``metadata["key"]`` and the
        returned index carries a ``key_of`` attribute mapping ``case_id -> key``
        so callers can recover the original key from a retrieval result.
        """
        cases: list[Case] = []
        key_of: dict[str, str] = {}
        for key, term_ids, metadata in records:
            md = dict(metadata or {})
            md["key"] = key
            case = self.build_case(term_ids, metadata=md)
            cases.append(case)
            key_of[case.case_id] = key

        index = MacFacIndex(config=config or self.config, canon=self.canon)
        index.build(cases)
        index.key_of = key_of
        return index


def mount(graph: OntologyGraph, config: MatchConfig | None = None) -> MountedOntology:
    """Mount ``graph``: populate a predicate lattice from its is_a edges.

    Every ``(child, parent)`` in :meth:`OntologyGraph.is_a_edges` becomes a
    lattice edge ``fid(child) -> fid(parent)``. The default config ascends up to
    two hops (``MatchConfig(delta=2, rho=0.95)``).
    """
    cfg = config or _default_config()
    canon = Canonicalizer()
    for child, parent in graph.is_a_edges():
        canon.lattice.add(fid(child), fid(parent))
    return MountedOntology(graph=graph, canon=canon, config=cfg)
