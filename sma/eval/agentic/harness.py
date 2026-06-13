"""One-shot agentic harness: swap the :class:`Memory`, hold everything else fixed.

``run_oneshot`` builds one benchmark from an arm ``(mounted ontology, entity ->
term-set records)`` and scores every memory identically:

* a deterministic ``holdout_frac`` of entities is reserved as NOVEL (their
  queries are unanswerable and feed the abstain/novelty metrics); only the rest
  is indexed in every memory;
* hard queries are generated for both answerable and novel entities (sample a
  few terms, climb 0-2 is-a levels for imprecision, add noise terms);
* each memory retrieves the true key's rank, a confidence, and a novelty signal;
* metrics: tail top-k (answerable; all + rare slice), risk-coverage AURC
  (answerable), novelty F1 over ALL queries (novelty > 0.5 vs truly novel);
* primary stat: SMA vs the best enterprise-RAG memory (best by tail top-5) on
  per-query top-5 correctness via paired bootstrap.

Determinism: every set is sorted to a list before use and every RNG is
explicitly seeded, so identical ``seeds`` yield identical result dicts.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Iterable

from sma.eval.agentic.memories import IndexItem, Memory, Query
from sma.eval.agentic.metrics import (
    ABSENT_RANK,
    novelty_f1,
    risk_coverage_aurc,
    tail_topk,
)
from sma.eval.stats import cliffs_delta, paired_bootstrap
from sma.ontology import MountedOntology

# Enterprise-RAG/KG gauntlet — SMA's primary comparison is the best of these.
ENTERPRISE_NAMES = ("bm25", "dense", "hybrid_rrf", "hybrid_rerank", "hippo")

NOVELTY_THRESHOLD = 0.5  # fixed (not per-method tuned); noted caveat in the spec.


# --- ontology IC machinery (closure-propagated term frequency) -------------
# Mirrors sma/eval/ontology_bench.py so the "rare" slice is defined identically.
def _ancestors(term: str, parents: dict[str, tuple[str, ...]], cache: dict[str, set]) -> set:
    if term in cache:
        return cache[term]
    acc: set[str] = set()
    for p in parents.get(term, ()):
        acc.add(p)
        acc |= _ancestors(p, parents, cache)
    cache[term] = acc
    return acc


def _build_ic(
    entity_terms: list[set[str]],
    parents: dict[str, tuple[str, ...]],
    anc_cache: dict[str, set],
) -> dict[str, float]:
    """Information content per term via closure-propagated frequency."""
    n = len(entity_terms)
    freq: dict[str, int] = {}
    for terms in entity_terms:
        clo = set(terms)
        for t in terms:
            clo |= _ancestors(t, parents, anc_cache)
        for t in clo:
            freq[t] = freq.get(t, 0) + 1
    return {t: -math.log(c / n) for t, c in freq.items()} if n else {}


def run_oneshot(
    name: str,
    mounted: MountedOntology,
    records: dict[str, set[str]],
    memories: list[Memory],
    *,
    seeds: Iterable[int] = (7, 17, 23),
    n_index: int = 2000,
    n_query: int = 120,
    holdout_frac: float = 0.1,
) -> dict:
    """Run the one-shot agentic benchmark and return a result dict.

    ``records`` maps ``entity_id -> set(term_id)``. Returns
    ``{"arm", "memories", "n_all", "n_rare", "n_novel", "per_memory", "primary"}``
    where ``per_memory[name]`` carries tail top-k (all + rare slices), AURC, and
    novelty F1, and ``primary`` is the SMA-vs-best-enterprise paired bootstrap.
    """
    graph = mounted.graph
    parents = {tid: tuple(t.parents) for tid, t in graph.terms.items()}

    def term_text(t: str) -> str:
        nm = graph.terms[t].name if t in graph.terms else ""
        return nm or t

    mem_names = [m.name for m in memories]

    # Eligible entities: those with at least one known term (SORTED for determinism).
    eligible = sorted(
        eid
        for eid, terms in records.items()
        if any(t in graph.terms for t in terms)
    )

    # Pooled per-query rows across seeds. Each row holds every memory's rank plus
    # confidence/novelty/flags for that query.
    answerable_rows: list[dict] = []  # rank rows for tail_topk on answerable queries
    per_mem: dict[str, dict[str, list]] = {
        m: {"ans_conf": [], "ans_correct": [], "all_pred_novel": [], "all_is_novel": []}
        for m in mem_names
    }
    # Per-query top-5 correctness on ALL queries (answerable + novel) for the
    # paired bootstrap. A novel query is "correct" only if the memory abstains
    # via novelty; for top-5 retrieval correctness a novel query is always a miss.
    top5_ans: dict[str, list[float]] = {m: [] for m in mem_names}

    n_novel_total = 0

    for seed in seeds:
        rng = random.Random(seed)
        ids = list(eligible)
        rng.shuffle(ids)
        pool = ids[:n_index]

        # Deterministic NOVEL holdout: SORTED -> shuffle -> slice.
        pool_sorted = sorted(pool)
        rng.shuffle(pool_sorted)
        n_holdout = int(round(len(pool_sorted) * holdout_frac))
        novel_ids = sorted(pool_sorted[:n_holdout])
        index_ids = sorted(pool_sorted[n_holdout:])

        # Indexed term-sets (known terms only, SORTED to lists).
        dz = {e: sorted(t for t in records[e] if t in graph.terms) for e in index_ids}
        dz_novel = {e: sorted(t for t in records[e] if t in graph.terms) for e in novel_ids}

        # IC + rare threshold over the INDEXED records only.
        anc_cache: dict[str, set] = {}
        ic = _build_ic([set(v) for v in dz.values()], parents, anc_cache)
        median_ic = statistics.median(ic.values()) if ic else 0.0
        noise_pool = sorted(ic) or sorted({t for v in dz.values() for t in v})

        # Build IndexItems and index every memory (identical input).
        items = [
            IndexItem(
                key=e,
                term_ids=frozenset(dz[e]),
                text=" ".join(term_text(t) for t in dz[e]),
                meta={"id": e},
            )
            for e in index_ids
        ]
        for mem in memories:
            mem.index(items)

        # Query specs: hard partial/imprecise observations for answerable AND novel.
        def make_qspec(terms: list[str]) -> list[str]:
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
            return q

        # Allocate the n_query budget between answerable and novel entities,
        # preserving the holdout proportion. Both kinds get hard queries.
        ans_candidates = [e for e in index_ids if dz[e]]
        nov_candidates = [e for e in novel_ids if dz_novel[e]]
        n_nov = min(len(nov_candidates), int(round(n_query * holdout_frac)))
        n_ans = min(len(ans_candidates), n_query - n_nov)
        ans_q = ans_candidates[:n_ans]
        nov_q = nov_candidates[:n_nov]
        n_novel_total += len(nov_q)

        qspecs: list[tuple[str, list[str], bool]] = []
        for e in ans_q:
            qspecs.append((e, make_qspec(dz[e]), False))
        for e in nov_q:
            qspecs.append((e, make_qspec(dz_novel[e]), True))

        for e, qterms, is_novel in qspecs:
            query = Query(
                term_ids=frozenset(qterms),
                text=" ".join(term_text(t) for t in qterms),
            )
            rare = (
                max((ic.get(t, 0.0) for t in (dz[e] if not is_novel else dz_novel[e])), default=0.0)
                > median_ic
            )
            rank_row = {"rare": rare}
            for mem in memories:
                res = mem.retrieve(query, k=10)
                rank = next((r.rank for r in res if r.key == e), ABSENT_RANK)
                if is_novel:
                    rank = ABSENT_RANK  # unanswerable: true key is not indexed
                conf = res[0].confidence if res else 0.0
                nov = mem.novelty(query)

                rank_row[mem.name] = rank
                top5_ans[mem.name].append(1.0 if (not is_novel and rank <= 5) else 0.0)
                per_mem[mem.name]["all_pred_novel"].append(nov > NOVELTY_THRESHOLD)
                per_mem[mem.name]["all_is_novel"].append(is_novel)
                if not is_novel:
                    correct = rank <= 5
                    per_mem[mem.name]["ans_conf"].append(conf)
                    per_mem[mem.name]["ans_correct"].append(correct)

            if not is_novel:
                answerable_rows.append(rank_row)

    # --- aggregate ---------------------------------------------------------
    tail5 = tail_topk(answerable_rows, k=5)
    tail1 = tail_topk(answerable_rows, k=1)
    tail10 = tail_topk(answerable_rows, k=10)

    per_memory: dict[str, dict] = {}
    for m in mem_names:
        aurc, _curve = risk_coverage_aurc(
            per_mem[m]["ans_conf"], per_mem[m]["ans_correct"]
        )
        f1 = novelty_f1(per_mem[m]["all_pred_novel"], per_mem[m]["all_is_novel"])
        per_memory[m] = {
            "tail": {
                "top1": tail1.get(m, {"all": 0.0, "rare": 0.0}),
                "top5": tail5.get(m, {"all": 0.0, "rare": 0.0}),
                "top10": tail10.get(m, {"all": 0.0, "rare": 0.0}),
            },
            "aurc": aurc,
            "novelty_f1": f1,
        }

    # --- primary: SMA vs best enterprise-RAG on per-query top-5 correctness --
    primary: dict | None = None
    present_enterprise = [m for m in ENTERPRISE_NAMES if m in tail5]
    if "sma" in tail5 and present_enterprise:
        best = max(present_enterprise, key=lambda m: tail5[m]["all"])
        a = top5_ans["sma"]
        b = top5_ans[best]
        bs = paired_bootstrap(a, b)
        primary = {
            "a": "sma",
            "b": best,
            "best_enterprise": best,
            "delta_top5": bs["delta"],
            "ci_low": bs["ci_low"],
            "ci_high": bs["ci_high"],
            "p_value": bs["p_value"],
            "cliffs": cliffs_delta(a, b),
        }

    return {
        "arm": name,
        "memories": mem_names,
        "n_all": len(answerable_rows),
        "n_rare": sum(1 for r in answerable_rows if r["rare"]),
        "n_novel": n_novel_total,
        "per_memory": per_memory,
        "primary": primary,
    }
