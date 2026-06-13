"""Shared harness for the multi-domain ontology benchmark suite (gigatest).

One protocol, every golden-ontology domain (configs/preregistration_ontology.md):
mount the ontology, index entities by their annotation term-sets, query with hard
partial/imprecise observations, and rank the true entity. SMA (the universal
adapter) is scored against FOUR baselines:
  - Phenomizer / Resnik IC best-match  (ontology-AWARE SOTA-equivalent)
  - Jaccard term overlap               (lexical floor)
  - TF-IDF dense cosine                (real dense-RAG over the same annotations)
  - HippoRAG phrase-graph + PPR        (real KG retriever over the same annotations)
Reported on ALL queries and on the registered RARE slice (entities whose rarest
term's IC exceeds the corpus median). Reproducibility: every set->list is sorted
and every RNG is explicitly seeded (hash-independent). No per-domain code here.
"""
from __future__ import annotations

import math
import random
import statistics
import time
from typing import Iterable

from sma.eval.baselines.dense import rank_tfidf_dense_batch
from sma.eval.baselines.hipporag import HippoRAGRetriever
from sma.eval.stats import cliffs_delta, paired_bootstrap
from sma.ontology import MountedOntology

METHODS = ("sma", "phen", "jac", "dense", "hippo")
LABELS = {"sma": "SMA", "phen": "Phenomizer", "jac": "Jaccard",
          "dense": "Dense-RAG", "hippo": "HippoRAG"}


# --- ontology IC machinery (closure-propagated term frequency) -------------
def _ancestors(term, parents, cache):
    if term in cache:
        return cache[term]
    acc: set[str] = set()
    for p in parents.get(term, ()):
        acc.add(p)
        acc |= _ancestors(p, parents, cache)
    cache[term] = acc
    return acc


def _build_ic(entity_terms, parents, anc_cache):
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


def _rank_of(ranked_ids, target):
    return next((i for i, cid in enumerate(ranked_ids, 1) if cid == target), 999)


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
    use_hippo: bool = True,
    verbose: bool = True,
) -> dict:
    """records: entity_id -> set of ontology term ids. Returns a result dict with
    pooled per-query ranks for every method, on ALL queries and the RARE slice."""
    graph = mounted.graph
    parents = {tid: tuple(t.parents) for tid, t in graph.terms.items()}

    def term_text(t):
        nm = graph.terms[t].name if t in graph.terms else ""
        return nm or t

    eligible = sorted(
        eid for eid, terms in records.items()
        if min_terms <= len({t for t in terms if t in graph.terms}) <= max_terms
    )

    # per-query rows pooled across seeds: {method: rank, "rare": bool}
    rows: list[dict] = []
    per_seed = []

    for seed in seeds:
        rng = random.Random(seed)
        ids = list(eligible)
        rng.shuffle(ids)
        idx_ids = sorted(ids[:n_index])
        dz = {e: sorted(t for t in records[e] if t in graph.terms) for e in idx_ids}
        anc_cache: dict[str, set] = {}
        ic = _build_ic([set(v) for v in dz.values()], parents, anc_cache)
        noise_pool = sorted(ic)
        median_ic = statistics.median(ic.values()) if ic else 0.0

        index = mounted.build_index((e, dz[e], {"id": e}) for e in idx_ids)
        key_of = index.key_of
        index_docs = [(e, " ".join(term_text(t) for t in dz[e])) for e in idx_ids]

        # generate the hard queries first (so dense can batch)
        query_ids = [e for e in idx_ids if len(dz[e]) >= 8][:n_query]
        qspecs = []
        for e in query_ids:
            terms = dz[e]
            keep = rng.sample(terms, min(5, len(terms)))
            q = []
            for t in keep:
                cur = t
                for _ in range(rng.choice([0, 0, 1, 1, 2])):
                    ps = parents.get(cur)
                    if ps:
                        cur = rng.choice(sorted(ps))
                q.append(cur)
            q += rng.sample(noise_pool, min(3, len(noise_pool)))
            qspecs.append((e, q))

        qtexts = [" ".join(term_text(t) for t in q) for _, q in qspecs]
        dense_rk = rank_tfidf_dense_batch(qtexts, index_docs, k=20)
        hippo = None
        if use_hippo:
            hippo = HippoRAGRetriever(); hippo.build(index_docs)

        t0 = time.perf_counter()
        seed_ranks = {m: [] for m in METHODS}
        for n, (e, q) in enumerate(qspecs, 1):
            row = {"rare": max((ic.get(t, 0.0) for t in dz[e]), default=0.0) > median_ic}
            # SMA
            res = mounted.build_case(q)
            sres = index.retrieve(res, k=10, shortlist=80, fac_budget=40)
            row["sma"] = _rank_of([key_of.get(r.case_id) for r in sres], e)
            # Phenomizer + Jaccard (rank true entity among all index entities)
            phen = sorted(((_phenomizer(q, set(dz[o]), parents, anc_cache, ic), o) for o in idx_ids),
                          key=lambda x: (-x[0], x[1]))
            row["phen"] = _rank_of([o for _, o in phen], e)
            jac = sorted(((_jaccard(q, set(dz[o])), o) for o in idx_ids), key=lambda x: (-x[0], x[1]))
            row["jac"] = _rank_of([o for _, o in jac], e)
            # Dense-RAG (precomputed batch)
            row["dense"] = _rank_of([cid for cid, _ in dense_rk[n - 1]], e)
            # HippoRAG (KG/PPR)
            row["hippo"] = _rank_of([cid for cid, _ in hippo.retrieve(qtexts[n - 1], k=20)], e) if hippo else 999
            rows.append(row)
            for m in METHODS:
                seed_ranks[m].append(row[m])
            if verbose and n % 50 == 0:
                print(f"  [{name} seed {seed}] {n}/{len(qspecs)} ({time.perf_counter()-t0:.0f}s)", flush=True)

        per_seed.append({"seed": seed, "n": len(qspecs),
                         **{f"{m}_t5": _acc(seed_ranks[m], 5) for m in METHODS}})

    result = {"arm": name, "n_all": len(rows), "n_rare": sum(1 for r in rows if r["rare"]),
              "per_seed": per_seed, "slices": {}}
    for slice_name, sub in (("all", rows), ("rare", [r for r in rows if r["rare"]])):
        result["slices"][slice_name] = _summarize(sub)
    if verbose:
        _print_arm(result)
    return result


def _acc(ranks, k):
    return sum(1 for r in ranks if r <= k) / max(len(ranks), 1)


def _summarize(rows):
    if not rows:
        return None
    metr = {m: {f"t{k}": _acc([r[m] for r in rows], k) for k in (1, 5, 10)} for m in METHODS}
    for m in METHODS:
        metr[m]["mrr"] = sum(1 / r[m] for r in rows if r[m] < 999) / len(rows)
    # primary: SMA vs BEST non-SMA baseline on top-5
    sma_c = [1.0 if r["sma"] <= 5 else 0.0 for r in rows]
    others = [m for m in METHODS if m != "sma"]
    best = max(others, key=lambda m: metr[m]["t5"])
    best_c = [1.0 if r[best] <= 5 else 0.0 for r in rows]
    bs = paired_bootstrap(sma_c, best_c)
    return {"n": len(rows), "metrics": metr, "best_baseline": best,
            "delta_t5": bs["delta"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
            "p_value": bs["p_value"], "cliffs": cliffs_delta(sma_c, best_c)}


def _print_arm(r):
    print(f"\n=== arm {r['arm']}: {r['n_all']} queries ({r['n_rare']} rare) ===")
    for slice_name in ("all", "rare"):
        s = r["slices"][slice_name]
        if not s:
            continue
        print(f"\n  [{slice_name}] n={s['n']}")
        print(f"  {'method':<12}{'top-1':<8}{'top-5':<8}{'top-10':<8}{'MRR':<8}")
        for m in METHODS:
            mm = s["metrics"][m]
            print(f"  {LABELS[m]:<12}{mm['t1']:<8.3f}{mm['t5']:<8.3f}{mm['t10']:<8.3f}{mm['mrr']:<8.3f}")
        print(f"  primary top-5 SMA vs {LABELS[s['best_baseline']]}: "
              f"delta={s['delta_t5']:+.4f} CI[{s['ci_low']:+.4f},{s['ci_high']:+.4f}] "
              f"p={s['p_value']:.4f} cliffs={s['cliffs']:+.3f}")
