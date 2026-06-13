"""Phase 3a regression: reproduce the rare-disease result THROUGH the generic
ontology API, proving the universal loader faithfully generalizes the bespoke
``scripts/rare_disease_test.py``.

This replicates the reference protocol EXACTLY (same diseases, same simulated
hard patients, same Phenomizer-IC and Jaccard baselines, same metric), but swaps
the hand-rolled OBO parsing + lattice + case-building for the new API:

    g  = load_obo(".../hp.obo", name="hpo")
    mo = mount(g)                       # canon + is_a ascension lattice for free
    idx = mo.build_index(records)       # records: (disease_id, terms, {"disease": id})
    qcase = mo.build_case(query_terms)
    res = idx.retrieve(qcase, k=10, shortlist=80, fac_budget=40)
    disease = idx.key_of[res.case_id]

Acceptance gate: SMA top-5 and top-10 must be >= Phenomizer (the known result).
Prints ``REGRESSION PASS`` / ``REGRESSION FAIL`` and the metric table.
"""

from __future__ import annotations

import math
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sma.ontology.loader import load_obo
from sma.ontology.mount import mount

ROOT = pathlib.Path(__file__).resolve().parents[1]
HPO = ROOT / "data" / "raw" / "hpo"


def parse_diseases(min_ph: int = 7, max_ph: int = 30) -> dict[str, set]:
    """Parse phenotype.hpoa exactly as the reference: aspect (col 10) == 'P',
    disease id col 0, HPO term col 3; keep diseases with 7..30 phenotypes."""
    dis: dict[str, set] = {}
    for line in (HPO / "phenotype.hpoa").open():
        if line.startswith(("#", "database_id")):
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 11 or p[10] != "P":
            continue
        dis.setdefault(p[0], set()).add(p[3])
    return {d: ts for d, ts in dis.items() if min_ph <= len(ts) <= max_ph}


def ancestors(term: str, parents: dict[str, list[str]], cache: dict[str, set]) -> set:
    if term in cache:
        return cache[term]
    acc: set = set()
    for p in parents.get(term, []):
        acc.add(p)
        acc |= ancestors(p, parents, cache)
    cache[term] = acc
    return acc


def main(n_index: int = 2500, n_query: int = 200, seed: int = 7) -> None:
    rng = random.Random(seed)

    # --- NEW GENERIC API: load + mount the ontology (parser + lattice for free) ---
    g = load_obo(str(HPO / "hp.obo"), name="hpo")
    mo = mount(g)

    # is_a parents straight from the normalized graph (replaces hand-rolled parse_obo).
    parents: dict[str, list[str]] = {tid: list(t.parents) for tid, t in g.terms.items()}

    diseases = parse_diseases()
    ids = list(diseases)
    rng.shuffle(ids)
    ids = ids[:n_index]
    dz = {d: diseases[d] for d in ids}
    anc_cache: dict[str, set] = {}

    # --- SMA index via the generic builder ---
    records = ((d, dz[d], {"disease": d}) for d in ids)
    index = mo.build_index(records)

    # --- IC for Phenomizer baseline (closure-propagated term frequency) ---
    N = len(dz)
    freq: dict[str, int] = {}
    for d, terms in dz.items():
        clo = set(terms)
        for t in terms:
            clo |= ancestors(t, parents, anc_cache)
        for t in clo:
            freq[t] = freq.get(t, 0) + 1
    IC = {t: -math.log(c / N) for t, c in freq.items()}

    def resnik(a, b):
        ca = {a} | ancestors(a, parents, anc_cache)
        cb = {b} | ancestors(b, parents, anc_cache)
        common = ca & cb
        return max((IC.get(x, 0.0) for x in common), default=0.0)

    def phenomizer(query, terms):
        def bma(src, tgt):
            return sum(max((resnik(q, t) for t in tgt), default=0.0) for q in src) / max(len(src), 1)
        return 0.5 * (bma(query, terms) + bma(terms, query))

    def jaccard(query, terms):
        q = set(query)
        return len(q & terms) / max(len(q | terms), 1)

    # --- simulate the SAME hard patients, evaluate all methods ---
    test_ids = [d for d in ids if len(dz[d]) >= 8][:n_query]
    ranks = {"SMA": [], "Phenomizer": [], "Jaccard": []}
    t0 = time.perf_counter()
    all_diseases = list(dz)
    noise_pool = list(IC)
    for n, d in enumerate(test_ids, 1):
        terms = list(dz[d])
        # HARD, realistic: a partial presentation of a few symptoms, described
        # imprecisely (climbed up the ontology), plus noise.
        keep = rng.sample(terms, min(5, len(terms)))
        q = []
        for t in keep:
            cur = t
            for _ in range(rng.choice([0, 0, 1, 1, 2])):   # imprecision: climb 0-2 levels
                ps = parents.get(cur)
                if ps:
                    cur = rng.choice(ps)
            q.append(cur)
        q += rng.sample(noise_pool, 3)                      # 3 irrelevant findings

        # SMA via generic build_case + retrieve + key_of recovery
        qcase = mo.build_case(q)
        res = index.retrieve(qcase, k=10, shortlist=80, fac_budget=40)
        sma_rank = next((i for i, r in enumerate(res, 1) if index.key_of[r.case_id] == d), 999)
        ranks["SMA"].append(sma_rank)
        # baselines (rank true disease among all)
        for name, fn in (("Phenomizer", phenomizer), ("Jaccard", jaccard)):
            scored = sorted(((fn(q, dz[o]), o) for o in all_diseases), key=lambda x: -x[0])
            rk = next((i for i, (_, o) in enumerate(scored, 1) if o == d), 999)
            ranks[name].append(rk)
        if n % 50 == 0:
            print(f"  {n}/{len(test_ids)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    print(f"\n=== rare-disease diagnosis via GENERIC ontology API "
          f"(n={len(test_ids)} simulated patients, {n_index} candidate diseases) ===")
    print(f"{'method':<12}{'top-1':<8}{'top-5':<8}{'top-10':<8}{'MRR':<8}")
    metrics: dict[str, dict[str, float]] = {}
    for m, rs in ranks.items():
        t1 = sum(1 for r in rs if r == 1) / len(rs)
        t5 = sum(1 for r in rs if r <= 5) / len(rs)
        t10 = sum(1 for r in rs if r <= 10) / len(rs)
        mrr = sum(1 / r for r in rs if r < 999) / len(rs)
        metrics[m] = {"top1": t1, "top5": t5, "top10": t10, "mrr": mrr}
        print(f"{m:<12}{t1:<8.3f}{t5:<8.3f}{t10:<8.3f}{mrr:<8.3f}")

    sma, phen = metrics["SMA"], metrics["Phenomizer"]
    ok = sma["top5"] >= phen["top5"] and sma["top10"] >= phen["top10"]
    print("\nREGRESSION PASS" if ok else "\nREGRESSION FAIL")


if __name__ == "__main__":
    main()
