"""Experiment D2 (baseline-fairness): ancestor-expanded RAG + IC-similarity.

The reviewer's sharpest fairness point: "did you give the RAG baselines the same
ontology context (ancestors/definitions), and did you compare to an ontology
IC-similarity retriever, on the evaluated domains?"

This NEW driver re-runs the SAME tasks/splits as the committed de-risk suite
(sma/eval/ontology_bench.py -> reports/confirmatory/ontology_suite.csv) on the
two cleanest, richest domains -- MEDICINE (HPO) and GENOMICS (GO) -- and adds:

  * dense+anc / bm25+anc : every document AND every query term-set is expanded
    with its FULL is-a ancestor closure (term NAMES of all ancestors are folded
    into the text) before indexing/ranking. This isolates the question: is SMA's
    edge just ancestor expansion? If dense+anc / bm25+anc still trail SMA, no.
  * dense / bm25         : the same retrievers WITHOUT closure, as a reference.
  * phen (Phenomizer IC best-match-average) and jac (Jaccard) : the existing
    ontology-AWARE IC-similarity reference points, framed against the same tasks.

To keep SMA byte-identical to the committed run we REPLICATE ontology_bench.run_arm's
exact RNG draw order (shuffle -> idx_ids -> qspec sampling -> noise) and reuse its
FROZEN helpers (_ancestors, _build_ic, _phenomizer, _jaccard, _rank_of, _acc). We
do NOT modify any frozen module -- this is an additive module.

Output:
  reports/confirmatory/ontology_fair_baselines.csv
      (arm, method, t5_rare, t5_all, delta_vs_sma, ci_low, ci_high, p, p_holm)
  reports/confirmatory/ontology_fair_baselines.log   (honest verdict)

Primary comparison: tail (rare) top-5, 3 seeds, paired bootstrap of each method
vs SMA, Holm across the baseline family per arm.

  python3 scripts/ontology_fair_baselines.py             # both arms (hpo, go)
  python3 scripts/ontology_fair_baselines.py --arm go
  python3 scripts/ontology_fair_baselines.py --n-index 400 --n-query 60   # smoke
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import random
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rank_bm25 import BM25Okapi  # real BM25 (IDF + length norm) -- fairest lexical RAG

from sma.eval.baselines.dense import rank_tfidf_dense_batch
# FROZEN helpers, reused verbatim (read-only import):
from sma.eval.ontology_bench import (
    _acc,
    _ancestors,
    _build_ic,
    _jaccard,
    _phenomizer,
    _rank_of,
)
from sma.eval.stats import cliffs_delta, holm_bonferroni, paired_bootstrap
from sma.ontology import load_obo, mount

# the loaders are identical to the committed driver (scripts/bench_ontology_suite.py)
from bench_ontology_suite import ARMS  # noqa: E402  (records loaders + obo paths)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/confirmatory"

# method order for the report.  "sma" is the reference all deltas are measured against.
METHODS = ("sma", "phen", "jac", "dense", "dense+anc", "bm25", "bm25+anc")
LABELS = {
    "sma": "SMA",
    "phen": "Phenomizer(IC-BMA)",
    "jac": "Jaccard",
    "dense": "Dense-RAG(TF-IDF)",
    "dense+anc": "Dense-RAG+ancestors",
    "bm25": "BM25",
    "bm25+anc": "BM25+ancestors",
}
# the baselines whose deltas-vs-SMA enter the per-arm Holm family
BASELINES = tuple(m for m in METHODS if m != "sma")


def _bm25_rank(bm, ids, query_tokens, k):
    """Rank with a PREBUILT BM25Okapi index (corpus is query-independent, so the
    index is built once per seed -- get_scores is vectorised and fast)."""
    scores = bm.get_scores(query_tokens)
    pairs = sorted(zip(ids, (float(s) for s in scores)), key=lambda x: (-x[1], x[0]))
    return [cid for cid, _ in pairs[:k]]


def run_arm(name, mounted, records, *, seeds=(7, 17, 23),
            n_index=2000, n_query=120, min_terms=7, max_terms=30, verbose=True):
    """Replicate ontology_bench.run_arm's task/split generation EXACTLY (same RNG
    draw order so SMA ranks reproduce the committed run), then additionally rank
    every method -- including the ancestor-expanded retrievers -- on the same
    queries.  Returns per-query rows pooled across seeds with a rank per method
    and a `rare` flag, plus per-seed top-5 accuracies for sanity."""
    graph = mounted.graph
    parents = {tid: tuple(t.parents) for tid, t in graph.terms.items()}

    def term_text(t):
        nm = graph.terms[t].name if t in graph.terms else ""
        return nm or t

    eligible = sorted(
        eid for eid, terms in records.items()
        if min_terms <= len({t for t in terms if t in graph.terms}) <= max_terms
    )

    rows: list[dict] = []
    per_seed = []

    for seed in seeds:
        # --- IDENTICAL RNG draw order to ontology_bench.run_arm ----------------
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

        # plain (no-closure) doc text: the term-name record, identical to committed
        index_docs = [(e, " ".join(term_text(t) for t in dz[e])) for e in idx_ids]

        # ancestor-expanded doc text: term NAMES of every term PLUS the names of all
        # its is-a ancestors (full closure).  Baselines now SEE the ontology context.
        def expand_terms(term_list):
            clo = set(term_list)
            for t in term_list:
                clo |= _ancestors(t, parents, anc_cache)
            return sorted(clo)

        index_docs_anc = [
            (e, " ".join(term_text(t) for t in expand_terms(dz[e]))) for e in idx_ids
        ]
        bm25_ids = [e for e, _ in index_docs]
        # build each BM25 index ONCE per seed (corpus is query-independent)
        bm25 = BM25Okapi([text.split() for _, text in index_docs])
        bm25_anc = BM25Okapi([text.split() for _, text in index_docs_anc])

        # --- query generation: byte-identical RNG order to ontology_bench --------
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

        # query text: plain and ancestor-expanded
        qtexts = [" ".join(term_text(t) for t in q) for _, q in qspecs]
        qtexts_anc = [" ".join(term_text(t) for t in expand_terms(q)) for _, q in qspecs]

        # dense batches (precompute)
        dense_rk = rank_tfidf_dense_batch(qtexts, index_docs, k=20)
        dense_rk_anc = rank_tfidf_dense_batch(qtexts_anc, index_docs_anc, k=20)

        t0 = time.perf_counter()
        seed_ranks = {m: [] for m in METHODS}
        for n, (e, q) in enumerate(qspecs, 1):
            row = {"rare": max((ic.get(t, 0.0) for t in dz[e]), default=0.0) > median_ic}
            # SMA -- identical call to committed run
            res = mounted.build_case(q)
            sres = index.retrieve(res, k=10, shortlist=80, fac_budget=40)
            row["sma"] = _rank_of([key_of.get(r.case_id) for r in sres], e)
            # Phenomizer (IC best-match-average) + Jaccard -- frozen helpers
            phen = sorted(
                ((_phenomizer(q, set(dz[o]), parents, anc_cache, ic), o) for o in idx_ids),
                key=lambda x: (-x[0], x[1]))
            row["phen"] = _rank_of([o for _, o in phen], e)
            jac = sorted(((_jaccard(q, set(dz[o])), o) for o in idx_ids),
                         key=lambda x: (-x[0], x[1]))
            row["jac"] = _rank_of([o for _, o in jac], e)
            # Dense-RAG (no closure) and Dense-RAG+ancestors
            row["dense"] = _rank_of([cid for cid, _ in dense_rk[n - 1]], e)
            row["dense+anc"] = _rank_of([cid for cid, _ in dense_rk_anc[n - 1]], e)
            # BM25 (no closure) and BM25+ancestors -- real Okapi BM25
            row["bm25"] = _rank_of(
                _bm25_rank(bm25, bm25_ids, qtexts[n - 1].split(), 20), e)
            row["bm25+anc"] = _rank_of(
                _bm25_rank(bm25_anc, bm25_ids, qtexts_anc[n - 1].split(), 20), e)
            rows.append(row)
            for m in METHODS:
                seed_ranks[m].append(row[m])
            if verbose and n % 40 == 0:
                print(f"  [{name} seed {seed}] {n}/{len(qspecs)} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)

        per_seed.append({"seed": seed, "n": len(seed_ranks["sma"]),
                         **{f"{m}_t5": _acc(seed_ranks[m], 5) for m in METHODS}})

    return {"arm": name, "rows": rows, "per_seed": per_seed,
            "n_all": len(rows), "n_rare": sum(1 for r in rows if r["rare"])}


def summarize(arm_result):
    """Per-arm: for each baseline method, paired bootstrap of (SMA t5) - (method t5)
    on the rare slice, with Holm across the baseline family.  Also reports t5_all."""
    rows = arm_result["rows"]
    rare = [r for r in rows if r["rare"]]
    out = {}
    for m in METHODS:
        out[m] = {
            "t5_rare": _acc([r[m] for r in rare], 5),
            "t5_all": _acc([r[m] for r in rows], 5),
        }
    # paired bootstrap of SMA vs each baseline on the RARE slice (primary)
    sma_rare = [1.0 if r["sma"] <= 5 else 0.0 for r in rare]
    raw_p = {}
    for m in BASELINES:
        m_rare = [1.0 if r[m] <= 5 else 0.0 for r in rare]
        bs = paired_bootstrap(sma_rare, m_rare)
        out[m].update({
            "delta_vs_sma": bs["delta"], "ci_low": bs["ci_low"],
            "ci_high": bs["ci_high"], "p": bs["p_value"],
            "cliffs": cliffs_delta(sma_rare, m_rare),
        })
        raw_p[m] = bs["p_value"]
    holm = holm_bonferroni(raw_p)
    for m in BASELINES:
        out[m]["p_holm"] = holm[m]
    out["sma"].update({"delta_vs_sma": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                       "p": 1.0, "p_holm": 1.0, "cliffs": 0.0})
    return out


def fmt_log(arm_label, summ, per_seed, n_rare, n_all):
    L = []
    L.append(f"########## {arm_label} ##########")
    L.append(f"  pooled queries: n_all={n_all}  n_rare(tail)={n_rare}  seeds={len(per_seed)}")
    L.append("  per-seed SMA top-5 (sanity vs committed ontology_suite.csv): "
             + ", ".join(f"{ps['seed']}={ps['sma_t5']:.3f}" for ps in per_seed))
    L.append("")
    L.append(f"  {'method':<22}{'t5_rare':<10}{'t5_all':<10}{'Δ_vs_SMA':<11}"
             f"{'95% CI':<22}{'p':<9}{'p_holm':<9}{'cliffs'}")
    for m in METHODS:
        s = summ[m]
        ci = f"[{s['ci_low']:+.3f},{s['ci_high']:+.3f}]"
        L.append(f"  {LABELS[m]:<22}{s['t5_rare']:<10.4f}{s['t5_all']:<10.4f}"
                 f"{s['delta_vs_sma']:<+11.4f}{ci:<22}{s['p']:<9.4f}"
                 f"{s['p_holm']:<9.4f}{s['cliffs']:+.3f}")
    return "\n".join(L)


def verdict_block(arm_label, summ):
    """Honest verdict: does ancestor expansion close the gap to SMA?"""
    L = [f"  --- VERDICT [{arm_label}] (rare/tail top-5) ---"]
    sma = summ["sma"]["t5_rare"]

    def line(m):
        s = summ[m]
        beats = s["delta_vs_sma"] > 0 and s["p_holm"] < 0.05
        ties = s["p_holm"] >= 0.05
        tag = ("SMA WINS (Holm-sig)" if beats
               else "PARITY (no Holm-sig gap)" if ties and abs(s["delta_vs_sma"]) >= 0
               else "baseline ahead")
        if s["delta_vs_sma"] < 0 and s["p_holm"] < 0.05:
            tag = "BASELINE BEATS SMA (Holm-sig)"
        return (f"    SMA vs {LABELS[m]:<22} Δ={s['delta_vs_sma']:+.4f} "
                f"p_holm={s['p_holm']:.4f}  ->  {tag}")

    for m in ("bm25", "bm25+anc", "dense", "dense+anc", "phen", "jac"):
        L.append(line(m))

    # the central de-risk question, answered from the numbers
    d_bm25 = summ["bm25"]["t5_rare"]
    d_bm25a = summ["bm25+anc"]["t5_rare"]
    d_dense = summ["dense"]["t5_rare"]
    d_densea = summ["dense+anc"]["t5_rare"]
    L.append("")
    L.append(f"    ancestor-expansion lift  : BM25 {d_bm25:.3f}->{d_bm25a:.3f} "
             f"(Δ{d_bm25a-d_bm25:+.3f}); Dense {d_dense:.3f}->{d_densea:.3f} "
             f"(Δ{d_densea-d_dense:+.3f})")
    best_anc = max(("bm25+anc", "dense+anc"), key=lambda m: summ[m]["t5_rare"])
    gap = sma - summ[best_anc]["t5_rare"]
    closed = summ[best_anc]["p_holm"] >= 0.05 if best_anc in summ and "p_holm" in summ[best_anc] else None
    L.append(f"    best ancestor-RAG = {LABELS[best_anc]} t5_rare={summ[best_anc]['t5_rare']:.3f}; "
             f"SMA={sma:.3f}; residual gap={gap:+.3f}; "
             f"Holm-sig gap remains? {'NO (closed to parity)' if closed else 'YES (SMA still ahead)'}")
    return "\n".join(L)


def main(which, n_index, n_query):
    keys = list(ARMS) if which == "all" else [which]
    results, summaries = [], {}
    for k in keys:
        label, obo, name, loader = ARMS[k]
        print(f"\n########## {label} ##########", flush=True)
        mo = mount(load_obo(str(ROOT / obo), name=name))
        records = loader()
        print(f"  {len(records)} entities loaded; running fair-baseline arm...", flush=True)
        r = run_arm(label, mo, records, n_index=n_index, n_query=n_query)
        results.append(r)
        summaries[label] = summarize(r)

    OUT.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_rows = []
    for r in results:
        s = summaries[r["arm"]]
        for m in METHODS:
            mm = s[m]
            csv_rows.append({
                "arm": r["arm"], "method": LABELS[m],
                "t5_rare": f"{mm['t5_rare']:.4f}", "t5_all": f"{mm['t5_all']:.4f}",
                "delta_vs_sma": f"{mm['delta_vs_sma']:.4f}",
                "ci_low": f"{mm['ci_low']:.4f}", "ci_high": f"{mm['ci_high']:.4f}",
                "p": f"{mm['p']:.4f}", "p_holm": f"{mm['p_holm']:.4f}",
                "cliffs": f"{mm['cliffs']:.4f}",
            })
    csv_path = OUT / "ontology_fair_baselines.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader(); w.writerows(csv_rows)

    # LOG
    log_lines = [
        "Experiment D2: ancestor-expanded RAG + IC-similarity baseline fairness",
        "Same tasks/splits as reports/confirmatory/ontology_suite.csv "
        f"(seeds 7/17/23, n_index={n_index}, n_query={n_query}).",
        "Primary metric: tail (rare) top-5; paired bootstrap of SMA-minus-baseline; "
        "Holm across the baseline family per arm.",
        "Ancestor-expanded arms fold the FULL is-a ancestor closure (term names) into "
        "BOTH document and query text before ranking.",
        "",
    ]
    for r in results:
        s = summaries[r["arm"]]
        log_lines.append(fmt_log(r["arm"], s, r["per_seed"], r["n_rare"], r["n_all"]))
        log_lines.append("")
        log_lines.append(verdict_block(r["arm"], s))
        log_lines.append("")
    log_text = "\n".join(log_lines)
    (OUT / "ontology_fair_baselines.log").write_text(log_text + "\n")

    print("\n" + log_text)
    print(f"\nwrote {csv_path}")
    print(f"wrote {OUT/'ontology_fair_baselines.log'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all", choices=["all", "hpo", "go"])
    ap.add_argument("--n-index", type=int, default=2000)
    ap.add_argument("--n-query", type=int, default=120)
    a = ap.parse_args()
    main(a.arm, a.n_index, a.n_query)
