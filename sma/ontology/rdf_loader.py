"""Complete RDF loader using rdflib (for ontologies our xml.etree subset misses).

Unlike :func:`sma.ontology.loader.load_owl` (a stdlib RDF/XML subset), this uses
rdflib to fully parse OWL — including Turtle, restrictions, and the complete class
graph — for ontologies like FIBO whose power lives in typed relations expressed
via ``owl:Restriction``. Falls back gracefully if rdflib is absent.
"""
from __future__ import annotations

from pathlib import Path

from .graph import OntologyGraph, Term
from .loader import _KNOWN_PREFIXES


def _compact(uri: str) -> str:
    frag = str(uri).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    if "_" in frag:
        prefix, _, rest = frag.partition("_")
        if prefix in _KNOWN_PREFIXES and rest:
            return f"{prefix}:{rest}"
    return frag


def load_rdflib(path: str, name: str = "", fmt: str | None = None,
                pattern: str = "*.rdf") -> OntologyGraph:
    """Load an OWL/RDF ontology (file, Turtle, or a directory of RDF files) via rdflib."""
    import logging
    import rdflib
    from rdflib import OWL, RDF, RDFS
    from rdflib.namespace import DC, DCTERMS

    # FIBO metadata carries malformed xsd:dateTime literals; rdflib logs a
    # traceback per occurrence but parses fine. Silence that noise.
    logging.getLogger("rdflib.term").setLevel(logging.CRITICAL)

    g = rdflib.Graph()
    root = Path(path)
    files = sorted(root.rglob(pattern)) if root.is_dir() else [root]
    for f in files:
        try:
            g.parse(str(f), format=fmt) if fmt else g.parse(str(f))
        except Exception:
            continue

    # Collect named owl:Class (and rdfs:Class) subjects with a real IRI.
    terms: dict[str, Term] = {}
    classes = set(g.subjects(RDF.type, OWL.Class)) | set(g.subjects(RDF.type, RDFS.Class))
    for cls in classes:
        if isinstance(cls, rdflib.BNode):
            continue
        tid = _compact(cls)
        label = g.value(cls, RDFS.label) or g.value(cls, DCTERMS.title) or g.value(cls, DC.title)
        parents: list[str] = []
        relations: list[tuple[str, str]] = []
        for sup in g.objects(cls, RDFS.subClassOf):
            if isinstance(sup, rdflib.BNode):
                # owl:Restriction -> typed relation (onProperty + someValuesFrom).
                prop = g.value(sup, OWL.onProperty)
                tgt = g.value(sup, OWL.someValuesFrom) or g.value(sup, OWL.allValuesFrom)
                if prop is not None and tgt is not None and not isinstance(tgt, rdflib.BNode):
                    relations.append((_compact(prop), _compact(tgt)))
            else:
                parents.append(_compact(sup))
        obsolete = bool(g.value(cls, OWL.deprecated))
        terms[tid] = Term(id=tid, name=str(label) if label else "",
                          parents=tuple(dict.fromkeys(parents)),
                          relations=tuple(dict.fromkeys(relations)), obsolete=obsolete)

    version = ""
    for o in g.objects(None, OWL.versionIRI):
        version = str(o); break
    if not name:
        name = root.stem
    return OntologyGraph(name=name, version=version, terms=terms)
