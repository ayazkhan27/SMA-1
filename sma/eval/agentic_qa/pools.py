"""Registered question pools for the Phase 5 LLM-QA "trustworthy specialist" phase.

This builds the three pre-registered question pools of
``configs/preregistration_v2_llmqa.md`` from the flagship medicine arm
(HPO + ``phenotype.hpoa``), holding the index fixed so attribution stays clean:

* **answerable** — cases whose true disease IS indexed (the agent should answer + cite);
* **out-of-knowledge / novel** — cases whose true disease is HELD OUT of the index
  (the agent should ABSTAIN *and* flag NOVEL). The held-out cases are, by
  construction, both unanswerable and novel, so ``ook`` and ``novel`` are the
  same list of :class:`QAItem`.

Each clinical case is a *hard* partial/imprecise observation generated exactly
like ``sma/eval/ontology_bench.run_arm``: sample a few of the disease's
phenotypes, climb 0-2 is-a levels (imprecision), and add a few noise terms, then
render the surviving terms as a natural-language presentation. Determinism: every
``set`` is sorted to a list before use and the single RNG is explicitly seeded,
so identical ``(seed, n_index, ...)`` yields identical pools.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from sma.ontology import MountedOntology


@dataclass
class QAItem:
    """One pre-registered LLM-QA case.

    ``case_text`` is the NL presentation shown to the agent; ``case_terms`` are
    the (possibly climbed/noised) ontology term ids backing it; ``gold_id`` /
    ``gold_name`` are the true disease. ``answerable`` is True iff the gold
    disease is indexed; ``novel`` is True iff the gold disease was held out.
    """

    case_text: str
    case_terms: frozenset[str]
    gold_id: str
    gold_name: str
    answerable: bool
    novel: bool


def _parse_hpoa(hpoa_path: str) -> dict[str, tuple[str, set[str]]]:
    """Parse ``phenotype.hpoa`` into ``disease_id -> (name, {hpo_term_id})``.

    Mirrors the flagship record construction (medicine arm /
    ``scripts/bench_ontology_suite.load_hpo_records``): skip header/comment
    lines, tab-split, keep only phenotypic-abnormality rows (column 10 == ``"P"``),
    and read column 0 as the disease id, column 1 as the disease name, column 3
    as the HPO term.
    """
    rec: dict[str, tuple[str, set[str]]] = {}
    with open(hpoa_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("#", "database_id")):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 11 or p[10] != "P":
                continue
            disease_id, disease_name, hpo_term = p[0], p[1], p[3]
            name, terms = rec.setdefault(disease_id, (disease_name, set()))
            terms.add(hpo_term)
    return rec


def build_pools(
    mounted: MountedOntology,
    hpoa_path: str,
    *,
    seed: int = 7,
    n_index: int = 1500,
    n_answerable: int = 120,
    n_held: int = 120,
    min_ph: int = 7,
    max_ph: int = 30,
) -> dict:
    """Build the registered LLM-QA pools over the medicine (HPO) arm.

    ``mounted`` is the mounted HPO ontology; ``hpoa_path`` points at
    ``phenotype.hpoa``. Returns a dict with keys:

    * ``"index_items"`` — ``list[IndexItem]`` for the INDEXED diseases (the
      shared knowledge every memory is built over);
    * ``"answerable"`` — ``n_answerable`` :class:`QAItem`\\ s drawn from INDEXED
      diseases (``answerable=True, novel=False``);
    * ``"ook"`` / ``"novel"`` — ``n_held`` :class:`QAItem`\\ s drawn from
      HELD-OUT diseases (``answerable=False, novel=True``); the same list is
      returned under both keys because held-out cases are both unanswerable and
      novel.

    Eligible diseases carry ``min_ph..max_ph`` phenotypes that are present in
    ``mounted.graph.terms``; their ids are sorted then shuffled under ``seed``.
    The first ``n_index`` are INDEXED; the remainder are HELD-OUT.
    """
    # Local import keeps this module importable even while the sibling
    # agentic_qa.metrics is mid-construction (the package __init__ imports it).
    from sma.eval.agentic import IndexItem

    graph = mounted.graph

    def term_text(t: str) -> str:
        nm = graph.terms[t].name if t in graph.terms else ""
        return nm or t

    parents = {tid: tuple(term.parents) for tid, term in graph.terms.items()}
    parsed = _parse_hpoa(hpoa_path)

    # Eligibility: known phenotypes only, count in [min_ph, max_ph]. SORTED ids.
    known: dict[str, tuple[str, list[str]]] = {}
    for did in sorted(parsed):
        name, terms = parsed[did]
        present = sorted(t for t in terms if t in graph.terms)
        if min_ph <= len(present) <= max_ph:
            known[did] = (name, present)

    eligible = sorted(known)
    rng = random.Random(seed)
    rng.shuffle(eligible)

    indexed_ids = eligible[:n_index]
    held_ids = eligible[n_index:]

    # Shared index: IndexItem(key=id, term_ids, text=space-joined term NAMES, meta).
    index_items = [
        IndexItem(
            key=did,
            term_ids=frozenset(known[did][1]),
            text=" ".join(term_text(t) for t in known[did][1]),
            meta={"name": known[did][0]},
        )
        for did in indexed_ids
    ]

    # Noise pool: every phenotype present across the INDEXED diseases (SORTED), so
    # injected distractors are in-vocabulary, matching the ontology_bench generator.
    noise_pool = sorted({t for did in indexed_ids for t in known[did][1]})

    def make_case(terms: list[str]) -> tuple[frozenset[str], str]:
        """Hard query: sample <=5 phenotypes, climb 0-2 is-a levels, +3 noise."""
        keep = rng.sample(terms, min(5, len(terms)))
        q: list[str] = []
        for t in keep:
            cur = t
            for _ in range(rng.choice([0, 0, 1, 1, 2])):
                ps = parents.get(cur)
                if ps:
                    cur = rng.choice(sorted(ps))
            q.append(cur)
        if noise_pool:
            q += rng.sample(noise_pool, min(3, len(noise_pool)))
        text = "Patient presents with: " + ", ".join(term_text(t) for t in q)
        return frozenset(q), text

    def qitems(ids: list[str], n: int, *, answerable: bool) -> list[QAItem]:
        out: list[QAItem] = []
        for did in ids[:n]:
            name, terms = known[did]
            case_terms, case_text = make_case(terms)
            out.append(
                QAItem(
                    case_text=case_text,
                    case_terms=case_terms,
                    gold_id=did,
                    gold_name=name,
                    answerable=answerable,
                    novel=not answerable,
                )
            )
        return out

    answerable = qitems(indexed_ids, n_answerable, answerable=True)
    novel = qitems(held_ids, n_held, answerable=False)

    return {
        "index_items": index_items,
        "answerable": answerable,
        "ook": novel,
        "novel": novel,
    }
