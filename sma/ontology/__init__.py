"""Universal OWL/OBO ontology loader, mounter, registry, and router for SMA-1.

This package generalizes the hand-rolled HPO mount in
``scripts/rare_disease_test.py`` into a reusable pipeline: parse any OBO/OWL
ontology into a normalized :class:`OntologyGraph`, mount it onto a
``Canonicalizer`` (is-a edges become the predicate lattice), build a
``MacFacIndex`` over cases, and retrieve by structural analogy. A
:class:`OntologyRegistry` caches mounted ontologies and a :class:`DomainRouter`
selects which ontology a query belongs to. See ``sma/ontology/README.md``.
"""

from __future__ import annotations

from .attack import load_attack_stix
from .cpc import load_cpc
from .mitre_xml import load_capec, load_cwe, load_mitre_xml
from .rdf_loader import load_rdflib
from .usgaap import load_usgaap
from .graph import OntologyGraph, Term
from .loader import fid, load_obo, load_ontology, load_owl, load_owl_dir
from .mount import MountedOntology, mount
from .registry import OntologyEntry, OntologyRegistry
from .router import DomainRouter

__all__ = [
    "OntologyGraph",
    "Term",
    "load_obo",
    "load_owl",
    "load_owl_dir",
    "load_ontology",
    "fid",
    "MountedOntology",
    "mount",
    "OntologyRegistry",
    "OntologyEntry",
    "DomainRouter",
    "load_attack_stix",
    "load_cpc",
    "load_capec",
    "load_cwe",
    "load_mitre_xml",
    "load_rdflib",
    "load_usgaap",
]
