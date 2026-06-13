"""Exploratory (NOT confirmatory) rare-disease diagnosis benchmark.

Standard setup: diseases -> HPO phenotype sets; simulate patients by sampling a
noisy subset of a disease's phenotypes; rank candidate diseases. SMA encodes each
phenotype as a FUNCTOR over the patient and uses the HPO is-a tree as its
ascension lattice (a specific term climbs to a disease's general term). The
surprisal scorer weights rare phenotypes by -log p == information content.

Baselines: Phenomizer-style IC best-match semantic similarity (the SOTA-equiv),
and Jaccard set overlap. Metric: top-1/5/10 accuracy + MRR. Curiosity only.
"""
from __future__ import annotations

import math
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sma.ir.canon import Canonicalizer
from sma.ir.schema import entity, make_case, stmt
from sma.index.macfac import MacFacIndex
from sma.match.types import MatchConfig

HPO = pathlib.Path("data/raw/hpo")


def parse_obo():
    parents: dict[str, list[str]] = {}
    cur = None
    for line in (HPO / "hp.obo").open():
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = None
        elif line.startswith("id: HP:"):
            cur = line[4:]
            parents.setdefault(cur, [])
        elif line.startswith("is_a: HP:") and cur:
            parents[cur].append(line.split()[1])
    return parents


def ancestors(term, parents, cache):
    if term in cache:
        return cache[term]
    acc = set()
    for p in parents.get(term, []):
        acc.add(p); acc |= ancestors(p, parents, cache)
    cache[term] = acc
    return acc


def parse_diseases(min_ph=7, max_ph=30):
    dis: dict[str, set] = {}
    for line in (HPO / "phenotype.hpoa").open():
        if line.startswith(("#", "database_id")):
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 11 or p[10] != "P":
            continue
        dis.setdefault(p[0], set()).add(p[3])
    return {d: ts for d, ts in dis.items() if min_ph <= len(ts) <= max_ph}


def fid(hp):  # HP:0001250 -> HP_0001250 (functor-safe)
    return hp.replace(":", "_")


def main(n_index=2500, n_query=300, seed=7):
    rng = random.Random(seed)
    parents = parse_obo()
    diseases = parse_diseases()
    ids = list(diseases)
    rng.shuffle(ids)
    ids = ids[:n_index]
    dz = {d: diseases[d] for d in ids}
    anc_cache: dict[str, set] = {}

    # --- SMA: phenotype-functor cases + HPO ascension lattice ---
    canon = Canonicalizer()
    seen_edges = set()
    for d, terms in dz.items():
        for t in terms:
            for p in parents.get(t, []):
                e = (fid(t), fid(p))
                if e not in seen_edges:
                    canon.lattice.add(*e); seen_edges.add(e)
    p_ent = entity("patient", "patient")
    cases = {}
    for d, terms in dz.items():
        c = make_case([stmt(fid(t), p_ent) for t in terms], {"disease": d})
        cases[c.case_id] = (d, c)
    index = MacFacIndex(config=MatchConfig(delta=2, rho=0.95), canon=canon)
    index.build([c for _, c in cases.values()])
    cid_disease = {cid: d for cid, (d, c) in cases.items()}

    # --- IC for Phenomizer baseline (closure-propagated term frequency) ---
    N = len(dz)
    freq: dict[str, int] = {}
    closures = {}
    for d, terms in dz.items():
        clo = set(terms)
        for t in terms:
            clo |= ancestors(t, parents, anc_cache)
        closures[d] = clo
        for t in clo:
            freq[t] = freq.get(t, 0) + 1
    IC = {t: -math.log(c / N) for t, c in freq.items()}

    def resnik(a, b):
        ca = {a} | ancestors(a, parents, anc_cache)
        cb = {b} | ancestors(b, parents, anc_cache)
        common = ca & cb
        return max((IC.get(x, 0.0) for x in common), default=0.0)

    def phenomizer(query, terms):
        # symmetric best-match average IC
        def bma(src, tgt):
            return sum(max((resnik(q, t) for t in tgt), default=0.0) for q in src) / max(len(src), 1)
        return 0.5 * (bma(query, terms) + bma(terms, query))

    def jaccard(query, terms):
        q = set(query)
        return len(q & terms) / max(len(q | terms), 1)

    # --- simulate patients from index diseases, evaluate all methods ---
    test_ids = [d for d in ids if len(dz[d]) >= 8][:n_query]
    ranks = {"SMA": [], "Phenomizer": [], "Jaccard": []}
    t0 = time.perf_counter()
    all_diseases = list(dz)
    noise_pool = list(IC)
    for n, d in enumerate(test_ids, 1):
        terms = list(dz[d])
        # HARD, realistic: a partial presentation of only a few symptoms,
        # often described imprecisely (climbed up the ontology), plus noise.
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

        # SMA
        qcase = make_case([stmt(fid(t), p_ent) for t in q])
        res = index.retrieve(qcase, k=10, shortlist=80, fac_budget=40)
        sma_rank = next((i for i, r in enumerate(res, 1) if cid_disease[r.case_id] == d), 999)
        ranks["SMA"].append(sma_rank)
        # baselines (rank true disease among all)
        for name, fn in (("Phenomizer", phenomizer), ("Jaccard", jaccard)):
            scored = sorted(((fn(q, dz[o]), o) for o in all_diseases), key=lambda x: -x[0])
            rk = next((i for i, (_, o) in enumerate(scored, 1) if o == d), 999)
            ranks[name].append(rk)
        if n % 50 == 0:
            print(f"  {n}/{len(test_ids)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    print(f"\n=== rare-disease diagnosis (n={len(test_ids)} simulated patients, "
          f"{n_index} candidate diseases) ===")
    print(f"{'method':<12}{'top-1':<8}{'top-5':<8}{'top-10':<8}{'MRR':<8}")
    for m, rs in ranks.items():
        t1 = sum(1 for r in rs if r == 1) / len(rs)
        t5 = sum(1 for r in rs if r <= 5) / len(rs)
        t10 = sum(1 for r in rs if r <= 10) / len(rs)
        mrr = sum(1 / r for r in rs if r < 999) / len(rs)
        print(f"{m:<12}{t1:<8.3f}{t5:<8.3f}{t10:<8.3f}{mrr:<8.3f}")


if __name__ == "__main__":
    main()
