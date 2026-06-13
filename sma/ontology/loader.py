"""Universal OBO/OWL ontology loaders into the normalized :class:`OntologyGraph`.

``load_obo`` parses the OBO flat-file ``[Term]`` blocks; ``load_owl`` parses the
common RDF/XML subset using the stdlib ``xml.etree`` only (rdflib is not a
dependency). ``load_ontology`` dispatches on the file extension.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .graph import OntologyGraph, Term

# Known prefixes whose underscore-encoded OWL ids should be restored to a colon
# form (e.g. ``HP_0001250`` -> ``HP:0001250``).
_KNOWN_PREFIXES = (
    "HP", "GO", "MONDO", "MP", "CHEBI", "UBERON", "CL", "DOID", "SO", "PATO",
    "NCBITaxon", "EFO", "ORDO", "OMIM", "TST",
)


def fid(term_id: str) -> str:
    """Functor-safe id: ``HP:0001250`` -> ``HP_0001250``."""
    return term_id.replace(":", "_")


def load_ontology(path: str, name: str = "") -> OntologyGraph:
    """Load an ontology, dispatching on the ``.obo``/``.owl``/``.owl.xml`` extension."""
    lower = str(path).lower()
    if lower.endswith(".obo"):
        return load_obo(path, name=name)
    if lower.endswith(".owl") or lower.endswith(".owl.xml") or lower.endswith(".rdf") or lower.endswith(".xml"):
        return load_owl(path, name=name)
    raise ValueError(f"Unrecognized ontology extension: {path}")


# --------------------------------------------------------------------------- #
# OBO
# --------------------------------------------------------------------------- #
def load_obo(path: str, name: str = "") -> OntologyGraph:
    """Parse an OBO flat file into an :class:`OntologyGraph`."""
    version = ""
    header_ontology = ""
    terms: dict[str, Term] = {}

    in_term = False
    cur_id = ""
    cur_name = ""
    cur_parents: list[str] = []
    cur_relations: list[tuple[str, str]] = []
    cur_obsolete = False

    def flush() -> None:
        nonlocal cur_id, cur_name, cur_parents, cur_relations, cur_obsolete
        if cur_id and ":" in cur_id:
            terms[cur_id] = Term(
                id=cur_id,
                name=cur_name,
                parents=tuple(cur_parents),
                relations=tuple(cur_relations),
                obsolete=cur_obsolete,
            )
        cur_id = ""
        cur_name = ""
        cur_parents = []
        cur_relations = []
        cur_obsolete = False

    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                # New stanza. Flush any pending term, then track whether we are
                # entering a [Term] stanza (others, like [Typedef], are ignored).
                if in_term:
                    flush()
                in_term = stripped == "[Term]"
                continue

            if not in_term:
                # Header region (before any stanza).
                if stripped.startswith("data-version:"):
                    version = stripped[len("data-version:"):].strip()
                elif stripped.startswith("ontology:") and not header_ontology:
                    header_ontology = stripped[len("ontology:"):].strip()
                continue

            if stripped.startswith("id:"):
                cur_id = stripped[len("id:"):].strip()
            elif stripped.startswith("name:"):
                cur_name = stripped[len("name:"):].strip()
            elif stripped.startswith("is_a:"):
                token = stripped[len("is_a:"):].strip().split("!", 1)[0].strip()
                token = token.split()[0] if token else ""
                if ":" in token:
                    cur_parents.append(token)
            elif stripped.startswith("relationship:"):
                rest = stripped[len("relationship:"):].strip().split("!", 1)[0].strip()
                parts = rest.split()
                if len(parts) >= 2 and ":" in parts[1]:
                    cur_relations.append((parts[0], parts[1]))
            elif stripped.startswith("is_obsolete:"):
                if stripped[len("is_obsolete:"):].strip().lower() == "true":
                    cur_obsolete = True

    if in_term:
        flush()

    if not name:
        name = header_ontology or Path(path).stem
    name = name.removesuffix(".obo")
    return OntologyGraph(name=name, version=version, terms=terms)


# --------------------------------------------------------------------------- #
# OWL / RDF-XML
# --------------------------------------------------------------------------- #
def _local(tag: str) -> str:
    """Strip the namespace from an etree tag: ``{uri}label`` -> ``label``."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _attr(elem: ET.Element, local_name: str) -> str | None:
    """Fetch an attribute by local name, ignoring namespace prefix."""
    for key, value in elem.attrib.items():
        if _local(key) == local_name:
            return value
    return None


def _term_id_from_iri(iri: str) -> str:
    """Derive a compact prefixed term id from an IRI.

    Takes the fragment after ``#`` or the final ``/`` and restores the colon for
    known prefixes (``HP_0001250`` -> ``HP:0001250``).
    """
    frag = iri.rsplit("#", 1)[-1]
    frag = frag.rsplit("/", 1)[-1]
    if "_" in frag:
        prefix, _, rest = frag.partition("_")
        if prefix in _KNOWN_PREFIXES and rest:
            return f"{prefix}:{rest}"
    return frag


def load_owl(path: str, name: str = "") -> OntologyGraph:
    """Parse the common RDF/XML subset of an OWL ontology (stdlib etree only)."""
    tree = ET.parse(path)
    root = tree.getroot()

    version = ""
    terms: dict[str, Term] = {}

    for elem in root.iter():
        local = _local(elem.tag)

        if local == "Ontology":
            for child in list(elem):
                clocal = _local(child.tag)
                if clocal == "versionIRI":
                    res = _attr(child, "resource")
                    if res:
                        version = version or res
                elif clocal == "versionInfo":
                    if child.text and child.text.strip():
                        version = version or child.text.strip()
            continue

        if local != "Class":
            continue

        about = _attr(elem, "about") or _attr(elem, "ID")
        if not about:
            continue
        term_id = _term_id_from_iri(about)

        term_name = ""
        parents: list[str] = []
        relations: list[tuple[str, str]] = []
        obsolete = False

        for child in list(elem):
            clocal = _local(child.tag)
            if clocal == "label":
                if child.text and not term_name:
                    term_name = child.text.strip()
            elif clocal == "deprecated":
                if child.text and child.text.strip().lower() == "true":
                    obsolete = True
            elif clocal == "subClassOf":
                resource = _attr(child, "resource")
                if resource:
                    parents.append(_term_id_from_iri(resource))
                    continue
                # Anonymous superclass: look for an owl:Restriction.
                for restr in child.iter():
                    if _local(restr.tag) != "Restriction":
                        continue
                    rel_type = ""
                    target = ""
                    for rchild in list(restr):
                        rlocal = _local(rchild.tag)
                        if rlocal == "onProperty":
                            res = _attr(rchild, "resource")
                            if res:
                                rel_type = _local(_term_id_from_iri(res))
                        elif rlocal == "someValuesFrom":
                            res = _attr(rchild, "resource")
                            if res:
                                target = _term_id_from_iri(res)
                    if rel_type and target:
                        relations.append((rel_type, target))

        terms[term_id] = Term(
            id=term_id,
            name=term_name,
            parents=tuple(parents),
            relations=tuple(relations),
            obsolete=obsolete,
        )

    if not name:
        name = Path(path).stem
    name = name.removesuffix(".owl")
    return OntologyGraph(name=name, version=version, terms=terms)
