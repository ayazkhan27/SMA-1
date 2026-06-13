"""Normalized ontology graph contract shared across the ontology package.

The graph is a small, serializable representation of an OBO/OWL ontology: a
flat map of term ids to :class:`Term` records, with helpers that yield the
is-a and typed-relation edges actually used to build the predicate lattice and
higher-order case statements. Obsolete terms (and any edge touching one) are
skipped by the edge iterators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Term:
    """A single ontology term.

    ``id`` is the canonical prefixed id (e.g. ``"HP:0001250"``); ``parents``
    are is_a parent ids; ``relations`` are typed ``(rel_type, target_id)`` pairs
    (e.g. ``("part_of", "GO:0005634")``).
    """

    id: str
    name: str = ""
    parents: tuple[str, ...] = ()
    relations: tuple[tuple[str, str], ...] = ()
    obsolete: bool = False


@dataclass
class OntologyGraph:
    """A flat, normalized view of an ontology."""

    name: str
    version: str = ""
    terms: dict[str, Term] = field(default_factory=dict)

    def active_terms(self) -> dict[str, Term]:
        """Return only the non-obsolete terms."""
        return {tid: term for tid, term in self.terms.items() if not term.obsolete}

    def is_a_edges(self) -> Iterator[tuple[str, str]]:
        """Yield ``(child_id, parent_id)`` is_a edges with both endpoints active."""
        active = self.active_terms()
        for child_id, term in active.items():
            for parent_id in term.parents:
                if parent_id in active:
                    yield child_id, parent_id

    def typed_relations(self) -> Iterator[tuple[str, str, str]]:
        """Yield ``(subj_id, rel_type, obj_id)`` typed relations, active only."""
        active = self.active_terms()
        for subj_id, term in active.items():
            for rel_type, obj_id in term.relations:
                if obj_id in active:
                    yield subj_id, rel_type, obj_id
