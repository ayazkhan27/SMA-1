"""A small registry of named ontologies that loads and mounts them on demand.

Each :class:`OntologyEntry` records where an ontology lives on disk and how to
parse it; :meth:`OntologyRegistry.get` lazily loads the file into an
:class:`~sma.ontology.graph.OntologyGraph`, mounts it (building the predicate
lattice + match index machinery), and caches the result so repeated lookups are
cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .loader import load_ontology

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .mount import MountedOntology
    from sma.match.types import MatchConfig


def _infer_format(path: str) -> str:
    """Infer the on-disk format from a path's extension.

    ``.obo`` -> ``"obo"``; ``.owl``/``.owl.xml``/``.rdf``/``.xml`` -> ``"owl"``.
    """
    lower = str(path).lower()
    if lower.endswith(".obo"):
        return "obo"
    if (
        lower.endswith(".owl")
        or lower.endswith(".owl.xml")
        or lower.endswith(".rdf")
        or lower.endswith(".xml")
    ):
        return "owl"
    raise ValueError(f"Cannot infer ontology format from extension: {path}")


@dataclass
class OntologyEntry:
    """A registered ontology: its name, source path, format, and version."""

    name: str
    path: str
    format: str
    version: str = ""


class OntologyRegistry:
    """A name-keyed collection of ontologies, loaded and mounted on demand."""

    def __init__(self) -> None:
        self._entries: dict[str, OntologyEntry] = {}
        self._order: list[str] = []
        self._mounted: dict[str, "MountedOntology"] = {}

    def register(
        self,
        name: str,
        path: str,
        fmt: str | None = None,
        version: str | None = None,
    ) -> OntologyEntry:
        """Register an ontology under ``name``.

        ``fmt`` is inferred from the file extension when omitted. Re-registering
        a name replaces its entry and invalidates any cached mounted ontology.
        """
        resolved_fmt = fmt if fmt is not None else _infer_format(path)
        entry = OntologyEntry(
            name=name,
            path=str(path),
            format=resolved_fmt,
            version=version or "",
        )
        if name not in self._entries:
            self._order.append(name)
        self._entries[name] = entry
        self._mounted.pop(name, None)
        return entry

    def get(self, name: str, config: "MatchConfig | None" = None) -> "MountedOntology":
        """Load + mount the named ontology, caching the result.

        The mounted ontology is cached on first access; subsequent calls return
        the same object (identity-stable) without re-reading the file.
        """
        cached = self._mounted.get(name)
        if cached is not None:
            return cached

        try:
            entry = self._entries[name]
        except KeyError:
            raise KeyError(f"No ontology registered under {name!r}") from None

        # Import lazily: mount.py is owned by a sibling agent and may bring in
        # heavier match machinery; keep registry import-time light and decoupled.
        from .mount import mount

        graph = load_ontology(entry.path, name=entry.name)
        if not entry.version and graph.version:
            entry.version = graph.version
        mounted = mount(graph, config=config)
        self._mounted[name] = mounted
        return mounted

    def list(self) -> list[OntologyEntry]:
        """Return registered entries in registration order."""
        return [self._entries[name] for name in self._order]

    def __contains__(self, name: object) -> bool:
        return name in self._entries
