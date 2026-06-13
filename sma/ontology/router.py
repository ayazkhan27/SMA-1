"""Route term ids and domains to the ontologies that can resolve them.

The :class:`DomainRouter` maps two things onto ontology names: id *prefixes*
(``"HP:"`` -> ``"hpo"``) and human *domains* (``"medicine"`` -> ``"hpo"``).
:meth:`DomainRouter.route` resolves a batch of term ids and/or a domain into the
de-duplicated, order-stable list of ontology names that should be consulted.
"""

from __future__ import annotations

from typing import Iterable

from .registry import OntologyRegistry


class DomainRouter:
    """Maps id prefixes and domains to registered ontology names."""

    def __init__(self, registry: OntologyRegistry) -> None:
        self.registry = registry
        self._prefixes: dict[str, str] = {}
        self._domains: dict[str, str] = {}

    def register_prefix(self, prefix: str, ontology_name: str) -> None:
        """Bind an id prefix (e.g. ``"HP:"``) to an ontology name."""
        self._prefixes[prefix] = ontology_name

    def register_domain(self, domain: str, ontology_name: str) -> None:
        """Bind a domain label (e.g. ``"medicine"``) to an ontology name."""
        self._domains[domain] = ontology_name

    def _ontology_for_term(self, term_id: str) -> str | None:
        """Return the ontology bound to the longest matching prefix, if any."""
        best: str | None = None
        best_len = -1
        for prefix, name in self._prefixes.items():
            if term_id.startswith(prefix) and len(prefix) > best_len:
                best = name
                best_len = len(prefix)
        return best

    def route(
        self,
        term_ids: Iterable[str] | None = None,
        domain: str | None = None,
    ) -> list[str]:
        """Resolve ``term_ids`` and/or ``domain`` to ontology names.

        A mapped ``domain`` contributes its ontology first; then each term id
        contributes the ontology of its longest matching prefix. The result is
        de-duplicated while preserving first-seen order. Returns ``[]`` when
        nothing matches.
        """
        ordered: list[str] = []
        seen: set[str] = set()

        def add(name: str | None) -> None:
            if name is not None and name not in seen:
                seen.add(name)
                ordered.append(name)

        if domain is not None:
            add(self._domains.get(domain))

        if term_ids is not None:
            for term_id in term_ids:
                add(self._ontology_for_term(term_id))

        return ordered
