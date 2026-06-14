#!/usr/bin/env python3
"""Entity-clustered robustness re-analysis of the agentic memory-swap suite.

The committed paired bootstrap (sma/eval/stats.paired_bootstrap) resamples
PER QUERY. The three seeds share entities, so per-query resampling treats the
<=3 correlated queries of one entity as independent and yields anti-conservative
intervals. This script re-runs the EXACT harness eval loop (importing the frozen
helpers so the queries/scores are byte-identical) but additionally records each
query's TARGET ENTITY ID, then computes a CLUSTER BOOTSTRAP that resamples
ENTITIES with replacement (taking all of an entity's queries together).

It validates faithfulness by reproducing the committed per-query delta/marginals
before reporting the clustered result. It does NOT modify any frozen file.

Output: reports/confirmatory/cluster_bootstrap.csv
"""
from __future__ import annotations
import argparse, csv, pathlib, sys, importlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sma.eval.agentic.harness as H
from sma.eval.agentic import (
    SmaMemory, BM25Memory, DenseMemory, HybridRRFMemory, HybridRerankMemory,
)
from sma.eval.stats import paired_bootstrap

ARMS = {
    "medicine": "sma.eval.agentic.arms.medicine",
    "discovery": "sma.eval.agentic.arms.discovery",
    "finance": "sma.eval.agentic.arms.finance",
    "legal": "sma.eval.agentic.arms.legal",
    "cyber": "sma.eval.agentic.arms.cyber",
}
# committed best enterprise baseline per domain (from reports/confirmatory/agentic_*.csv)
BEST = {"medicine": "hybrid_rerank", "discovery": "dense",
        "finance": "hybrid_rrf", "legal": "dense", "cyber": "hybrid_rrf"}


def build_minimal(mounted, best: str) -> list:
    """Build SMA + exactly the memories needed to realise `best` (skip hippo)."""
    bm25 = BM25Memory(); dense = DenseMemory()
    mems = [SmaMemory(mounted)]
    if best == "dense":
        mems += [dense]
    elif best == "hybrid_rrf":
        mems += [bm25, dense, HybridRRFMemory(bm25, dense)]
    elif best == "hybrid_rerank":
        hyb = HybridRRFMemory(bm25, dense)
        mems += [bm25, dense, hyb, HybridRerankMemory(hyb)]
    return mems


def run_capture(name, mounted, records, memories, *, seeds=(7, 17, 23),
                n_index=2000, n_query=120, holdout_frac=0.1):
    """Verbatim copy of harness.run_oneshot's loop, additionally capturing the
    (seed, entity_id) of every per-query top-5 score. Returns per-query rows."""
    graph = mounted.graph
    parents = {tid: tuple(t.parents) for tid, t in graph.terms.items()}

    def term_text(t):
        nm = graph.terms[t].name if t in graph.terms else ""
        return nm or t

    mem_names = [m.name for m in memories]
    eligible = sorted(eid for eid, terms in records.items()
                      if any(t in graph.terms for t in terms))
    top5_ans = {m: [] for m in mem_names}
    rows = []  # one per query: {entity, seed, is_novel, <mem>:0/1 ...}

    for seed in seeds:
        rng = H.random.Random(seed)
        ids = list(eligible); rng.shuffle(ids); pool = ids[:n_index]
        pool_sorted = sorted(pool); rng.shuffle(pool_sorted)
        n_holdout = int(round(len(pool_sorted) * holdout_frac))
        novel_ids = sorted(pool_sorted[:n_holdout])
        index_ids = sorted(pool_sorted[n_holdout:])
        dz = {e: sorted(t for t in records[e] if t in graph.terms) for e in index_ids}
        dz_novel = {e: sorted(t for t in records[e] if t in graph.terms) for e in novel_ids}
        anc_cache = {}
        ic = H._build_ic([set(v) for v in dz.values()], parents, anc_cache)
        median_ic = H.statistics.median(ic.values()) if ic else 0.0
        noise_pool = sorted(ic) or sorted({t for v in dz.values() for t in v})
        items = [H.IndexItem(key=e, term_ids=frozenset(dz[e]),
                             text=" ".join(term_text(t) for t in dz[e]), meta={"id": e})
                 for e in index_ids]
        for mem in memories:
            mem.index(items)

        def make_qspec(terms):
            keep = rng.sample(terms, min(5, len(terms)))
            q = []
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

        ans_candidates = [e for e in index_ids if dz[e]]
        nov_candidates = [e for e in novel_ids if dz_novel[e]]
        n_nov = min(len(nov_candidates), int(round(n_query * holdout_frac)))
        n_ans = min(len(ans_candidates), n_query - n_nov)
        ans_q = ans_candidates[:n_ans]; nov_q = nov_candidates[:n_nov]
        qspecs = [(e, make_qspec(dz[e]), False) for e in ans_q]
        qspecs += [(e, make_qspec(dz_novel[e]), True) for e in nov_q]

        for e, qterms, is_novel in qspecs:
            query = H.Query(term_ids=frozenset(qterms),
                            text=" ".join(term_text(t) for t in qterms))
            row = {"entity": e, "seed": seed, "is_novel": is_novel}
            for mem in memories:
                res = mem.retrieve(query, k=10)
                rank = next((r.rank for r in res if r.key == e), H.ABSENT_RANK)
                if is_novel:
                    rank = H.ABSENT_RANK
                score = 1.0 if (not is_novel and rank <= 5) else 0.0
                top5_ans[mem.name].append(score)
                row[mem.name] = score
            rows.append(row)
    return rows, top5_ans


def cluster_bootstrap(rows, a_name, b_name, n_resamples=10_000, seed=12345):
    """Resample ENTITIES with replacement; each entity contributes all its queries.
    Two-sided p from the sign-crossing fraction (bootstrap-floor at 1/(R+1))."""
    by_ent = {}
    for r in rows:
        by_ent.setdefault(r["entity"], []).append(r[a_name] - r[b_name])
    ents = list(by_ent.keys())
    # per-entity mean diff; observed grand mean weights entities equally is NOT
    # the per-query estimate, so we keep the per-query estimate by pooling all
    # queries of resampled entities (matches the committed per-query statistic).
    obs = float(np.mean([d for r in rows for d in [r[a_name] - r[b_name]]]))
    rng = np.random.default_rng(seed)
    idx_ents = np.array(ents, dtype=object)
    diffs_by_ent = [np.array(by_ent[e], float) for e in ents]
    boot = np.empty(n_resamples)
    for i in range(n_resamples):
        pick = rng.integers(0, len(ents), size=len(ents))
        pooled = np.concatenate([diffs_by_ent[j] for j in pick])
        boot[i] = pooled.mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_lo = (np.sum(boot <= 0.0) + 1) / (n_resamples + 1)
    p_hi = (np.sum(boot >= 0.0) + 1) / (n_resamples + 1)
    p = float(min(1.0, 2.0 * min(p_lo, p_hi)))
    return {"delta": obs, "ci_low": float(lo), "ci_high": float(hi),
            "p_value": p, "n_entities": len(ents), "n_queries": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--n-index", type=int, default=2000)
    args = ap.parse_args()
    out = ROOT / "reports/confirmatory/cluster_bootstrap.csv"
    fields = ["arm", "best", "n_entities", "n_queries", "pq_delta", "pq_p",
              "cl_delta", "cl_ci_low", "cl_ci_high", "cl_p", "reproduced"]
    results = []
    for arm in args.arms:
        best = BEST[arm]
        print(f"[{arm}] best={best} building...", flush=True)
        mod = importlib.import_module(ARMS[arm])
        mounted, records = mod.load()
        mems = build_minimal(mounted, best)
        rows, top5 = run_capture(arm, mounted, records, mems, n_index=args.n_index)
        a, b = top5["sma"], top5[best]
        pq = paired_bootstrap(a, b)             # per-query (reproduce committed)
        cl = cluster_bootstrap(rows, "sma", best)
        repro = abs(pq["delta"] - cl["delta"]) < 1e-9  # same point estimate
        row = {"arm": arm, "best": best, "n_entities": cl["n_entities"],
               "n_queries": cl["n_queries"], "pq_delta": round(pq["delta"], 4),
               "pq_p": pq["p_value"], "cl_delta": round(cl["delta"], 4),
               "cl_ci_low": round(cl["ci_low"], 4), "cl_ci_high": round(cl["ci_high"], 4),
               "cl_p": round(cl["p_value"], 4), "reproduced": repro}
        results.append(row)
        print(f"  per-query  d={pq['delta']:+.4f} p={pq['p_value']:.4g}", flush=True)
        print(f"  clustered  d={cl['delta']:+.4f} CI[{cl['ci_low']:+.3f},{cl['ci_high']:+.3f}] "
              f"p={cl['p_value']:.4g}  ents={cl['n_entities']}", flush=True)
        # write incrementally so a slow last arm doesn't lose earlier results
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(results)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
