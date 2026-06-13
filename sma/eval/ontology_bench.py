"""Shared harness for the multi-domain ontology benchmark suite (gigatest).

One protocol, every golden-ontology domain (configs/preregistration_ontology.md):
mount the ontology, index entities by their annotation term-sets, query with hard
partial/imprecise observations, and rank the true entity. SMA (the universal
adapter) is scored against the ontology-aware SOTA-equivalent (Phenomizer/Resnik
IC best-match) and a lexical floor (Jaccard).

Reproducibility: every set->list is sorted and every RNG is explicitly seeded, so
results do not depend on PYTHONHASHSEED (the variance source in the exploratory
HPO gate). An arm is defined entirely by (mounted ontology, entity->term-set);
no per-domain code lives here.
"""
from __future__ import annotations

import math
import random
import time
from typing import Iterable

from sma.eval.stats import cliffs_delta, holm_bonferroni, paired_bootstrap
from sma.ontology import MountedOntology


# --- ontology IC machinery (closure-propagated term frequency) -------------
def _ancestors(term: str, parents: dict[str, tuple[str, ...]], cache: dict[str, set]) -> set[str]:
    if term in cache:
        return cache[term]
    acc: set[str] = set()
    for p in parents.get(term, ()):  # parents already a tuple per term
        acc.add(p)
        acc |= _ancestors(p, parents, cache)
    cache[term] = acc
    return acc


def _build_ic(entity_terms: list[set[str]], parents, anc_cache):
    n = len(entity_terms)
    freq: dict[str, int] = {}
    for terms in entity_terms:
        clo = set(terms)
        for t in terms:
            clo |= _ancestors(t, parents, anc_cache)
        for t in clo:
            freq[t] = freq.get(t, 0) + 1
    return {t: -math.log(c / n) for t, c in freq.items()}


def _resnik(a, b, parents, anc_cache, ic):
    ca = {a} | _ancestors(a, parents, anc_cache)
    cb = {b} | _ancestors(b, parents, anc_cache)
    return max((ic.get(x, 0.0) for x in ca & cb), default=0.0)


def _phenomizer(query, terms, parents, anc_cache, ic):
    def bma(src, tgt):
        return sum(max((_resnik(q, t, parents, anc_cache, ic) for t in tgt), default=0.0)
                   for q in src) / max(len(src), 1)
    return 0.5 * (bma(query, terms) + bma(terms, query))


def _jaccard(query, terms):
    q = set(query)
    return len(q & terms) / max(len(q | terms), 1)


# --- one arm ---------------------------------------------------------------
def run_arm(
    name: str,
    mounted: MountedOntology,
    records: dict[str, set[str]],
    *,
    seeds: Iterable[int] = (7, 17, 23),
    n_index: int = 2500,
    n_query: int = 150,
    min_terms: int = 7,
    max_terms: int = 30,
    verbose: bool = True,
) -> dict:
    """records: entity_id -> set of ontology term ids. Returns a result dict with
    per-seed metrics, pooled per-query correctness, and SMA-vs-best stats."""
    graph = mounted.graph
    parents = {tid: tuple(t.parents) for tid, t in graph.terms.items()}
    # eligible entities: term count in band AND all terms known to the ontology
    eligible = sorted(
        eid for eid, terms in records.items()
        if min_terms <= len({t for t in terms if t in graph.terms}) <= max_terms
    )

    pooled = {m: {"sma": [], "phen": [], "jac": []} for m in ("t1", "t5", "t10")}
    pooled_rank = {"sma": [], "phen": [], "jac": []}
    per_seed = []

    for seed in seeds:
        rng = random.Random(seed)
        ids = list(eligible)
        rng.shuffle(ids)
        idx_ids = sorted(ids[:n_index])               # sorted -> hash-independent
        dz = {e: sorted(t for t in records[e] if t in graph.terms) for e in idx_ids}
        anc_cache: dict[str, set] = {}
        ic = _build_ic([set(v) for v in dz.values()], parents, anc_cache)
        noise_pool = sorted(ic)

        index = mounted.build_index((e, dz[e], {"id": e}) for e in idx_ids)
        key_of = index.key_of

        query_ids = [e for e in idx_ids if len(dz[e]) >= 8][:n_query]
        ranks = {"sma": [], "phen": [], "jac": []}
        t0 = time.perf_counter()
        for n, e in enumerate(query_ids, 1):
            terms = dz[e]
            keep = rng.sample(terms, min(5, len(terms)))
            q = []
            for t in keep:
                cur = t
                for _ in range(rng.choice([0, 0, 1, 1, 2])):   # imprecision climb
                    ps = parents.get(cur)
                    if ps:
                        cur = rng.choice(sorted(ps))
                q.append(cur)
            q += rng.sample(noise_pool, min(3, len(noise_pool)))

            qcase = mounted.build_case(q)
            res = index.retrieve(qcase, k=10, shortlist=80, fac_budget=40)
            sma_rank = next((i for i, r in enumerate(res, 1) if key_of.get(r.case_id) == e), 999)
            ranks["sma"].append(sma_rank)
            for tag, fn in (("phen", _phenomizer), ("jac", _jaccard)):
                if tag == "phen":
                    scored = sorted(((fn(q, set(dz[o]), parents, anc_cache, ic), o) for o in idx_ids),
                                    key=lambda x: (-x[0], x[1]))
                else:
                    scored = sorted(((fn(q, set(dz[o])), o) for o in idx_ids),
                                    key=lambda x: (-x[0], x[1]))
                ranks[tag].append(next((i for i, (_, o) in enumerate(scored, 1) if o == e), 999))
            if verbose and n % 50 == 0:
                print(f"  [{name} seed {seed}] {n}/{len(query_ids)} ({time.perf_counter()-t0:.0f}s)", flush=True)

        def acc(rs, k):
            return sum(1 for r in rs if r <= k) / max(len(rs), 1)

        def mrr(rs):
            return sum(1 / r for r in rs if r < 999) / max(len(rs), 1)

        per_seed.append({
            "seed": seed, "n": len(query_ids),
            **{f"{m}_{k}": acc(ranks[m], k) for m in ranks for k in (1, 5, 10)},
            **{f"{m}_mrr": mrr(ranks[m]) for m in ranks},
        })
        for m in ranks:
            pooled_rank[m].extend(ranks[m])
            pooled["t1"][_short(m)].extend(1.0 if r <= 1 else 0.0 for r in ranks[m])
            pooled["t5"][_short(m)].extend(1.0 if r <= 5 else 0.0 for r in ranks[m])
            pooled["t10"][_short(m)].extend(1.0 if r <= 10 else 0.0 for r in ranks[m])

    # primary metric: top-5; SMA vs the BEST baseline on top-5
    base_t5 = {"phen": sum(pooled["t5"]["phen"]) / len(pooled["t5"]["phen"]),
               "jac": sum(pooled["t5"]["jac"]) / len(pooled["t5"]["jac"])}
    best = "phen" if base_t5["phen"] >= base_t5["jac"] else "jac"
    sma_c, best_c = pooled["t5"]["sma"], pooled["t5"][best]
    bs = paired_bootstrap(sma_c, best_c)
    result = {
        "arm": name, "best_baseline": best,
        "n_queries": len(sma_c),
        "sma": {k: sum(pooled[k]["sma"]) / len(pooled[k]["sma"]) for k in pooled},
        "phen": {k: sum(pooled[k]["phen"]) / len(pooled[k]["phen"]) for k in pooled},
        "jac": {k: sum(pooled[k]["jac"]) / len(pooled[k]["jac"]) for k in pooled},
        "sma_mrr": sum(1 / r for r in pooled_rank["sma"] if r < 999) / len(pooled_rank["sma"]),
        "primary_delta_t5": bs["delta"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
        "p_value": bs["p_value"], "cliffs": cliffs_delta(sma_c, best_c),
        "per_seed": per_seed,
    }
    if verbose:
        _print_arm(result)
    return result


def _short(m):
    return {"sma": "sma", "phen": "phen", "jac": "jac"}[m]


def _print_arm(r):
    print(f"\n=== arm {r['arm']}: {r['n_queries']} pooled queries ===")
    print(f"{'method':<12}{'top-1':<8}{'top-5':<8}{'top-10':<8}")
    for m, lab in (("sma", "SMA"), ("phen", "Phenomizer"), ("jac", "Jaccard")):
        print(f"{lab:<12}{r[m]['t1']:<8.3f}{r[m]['t5']:<8.3f}{r[m]['t10']:<8.3f}")
    print(f"primary (top-5) SMA-{r['best_baseline']}: delta={r['primary_delta_t5']:+.4f} "
          f"CI[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] p={r['p_value']:.4f} "
          f"cliffs={r['cliffs']:+.3f}")
