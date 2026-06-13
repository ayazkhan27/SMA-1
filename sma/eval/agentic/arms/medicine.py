"""Medicine arm: HPO ontology + rare-disease phenotype records.

``load()`` mounts the Human Phenotype Ontology and parses ``phenotype.hpoa`` into
``disease_id -> {hpo_term_id}`` records, restricted to phenotypic-abnormality
(aspect ``P``) annotations of diseases carrying 7..30 phenotypes -- the same
record-construction used by ``scripts/bench_ontology_suite.load_hpo_records`` and
the 7..30 eligibility band of ``sma.eval.ontology_bench.run_arm``.
"""

from __future__ import annotations

import pathlib

from sma.ontology import MountedOntology, load_obo, mount

ROOT = pathlib.Path(__file__).resolve().parents[4]
HP_OBO = ROOT / "data/raw/hpo/hp.obo"
HPOA = ROOT / "data/raw/hpo/phenotype.hpoa"

MIN_TERMS = 7
MAX_TERMS = 30


def load_hpo_records(path: pathlib.Path = HPOA) -> dict[str, set[str]]:
    """Parse ``phenotype.hpoa`` into ``disease_id -> {hpo_term_id}`` (aspect P).

    Skips header/comment lines, keeps only phenotypic-abnormality annotations
    (column 10 == ``"P"``), and retains diseases with 7..30 phenotypes.
    """
    rec: dict[str, set[str]] = {}
    for line in path.open():
        if line.startswith(("#", "database_id")):
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 11 or p[10] != "P":
            continue
        rec.setdefault(p[0], set()).add(p[3])
    return {d: terms for d, terms in rec.items() if MIN_TERMS <= len(terms) <= MAX_TERMS}


def load() -> tuple[MountedOntology, dict[str, set[str]]]:
    """Return the mounted HPO ontology and its disease->phenotype records."""
    mounted = mount(load_obo(str(HP_OBO), name="hpo"))
    records = load_hpo_records()
    return mounted, records
