"""Shared cross-domain arm evaluator (4b).

Guarantees an IDENTICAL index/query partition across arms (generic / drafted /
expert) by splitting on the original record order — NOT on the encoding-dependent
case_id — so the baselines are stable and the only thing that varies between arms
is the SMA encoding. Returns the result dict and writes cd_<domain>_<phase>.csv.
"""
from __future__ import annotations

import csv
import pathlib
import time

from sma.index.macfac import MacFacIndex
from sma.ir.schema import Statement
from sma.eval.baselines.bm25 import rank_bm25_like
from sma.eval.baselines.dense import rank_tfidf_dense_batch
from sma.eval.metrics import macro_f1
from sma.eval.stats import paired_bootstrap, holm_bonferroni, cliffs_delta

OUT = pathlib.Path("reports/confirmatory")


def _ho_density(case) -> float:
    ex = case.expressions()
    ho = sum(1 for s in ex if any(isinstance(a, Statement) for a in s.args))
    return ho / max(len(ex), 1)


def _vote(ranked_ids, label_of, labels):
    pos, neg = labels
    tally = {pos: 0, neg: 0}
    for cid in ranked_ids:
        tally[label_of[cid]] += 1
    return pos if tally[pos] >= tally[neg] else neg


def evaluate_arm(items, encode_fn, row_text_fn, labels, phase, domain,
                 k=10, frac=0.7):
    """items: list of records (each has .label). encode_fn(item)->Case.
    row_text_fn(item)->str for the baselines. labels=(pos,neg)."""
    index_n = int(len(items) * frac)
    index_items, query_items = items[:index_n], items[index_n:]   # FIXED split

    def enc(its):
        return [(it, encode_fn(it)) for it in its]
    idx, qry = enc(index_items), enc(query_items)
    label_of = {c.case_id: it.label for it, c in idx + qry}
    text_of = {c.case_id: row_text_fn(it) for it, c in idx + qry}
    dens = [_ho_density(c) for _, c in idx + qry]
    mean_ho = sum(dens) / len(dens)

    index_cases = [c for _, c in idx]
    index_docs = [(c.case_id, text_of[c.case_id]) for _, c in idx]
    t0 = time.perf_counter()
    sma_index = MacFacIndex(); sma_index.build(index_cases)
    gold, sma_p, bm_p, dn_p = [], [], [], []
    dense_rk = rank_tfidf_dense_batch([text_of[c.case_id] for _, c in qry], index_docs, k=k)
    for qi, (_, qc) in enumerate(qry):
        gold.append(label_of[qc.case_id])
        res = sma_index.retrieve(qc, k=k, shortlist=60, fac_budget=25)
        sma_p.append(_vote([r.case_id for r in res], label_of, labels))
        bm = rank_bm25_like(text_of[qc.case_id], index_docs, k=k)
        bm_p.append(_vote([cid for cid, _ in bm], label_of, labels))
        dn_p.append(_vote([cid for cid, _ in dense_rk[qi]], label_of, labels))
    dt = time.perf_counter() - t0

    f1 = {"SMA": macro_f1(gold, sma_p), "BM25": macro_f1(gold, bm_p), "Dense": macro_f1(gold, dn_p)}
    print(f"[{domain}/{phase}] HO-density={mean_ho:.4f}  macro-F1: SMA {f1['SMA']:.4f}  "
          f"BM25 {f1['BM25']:.4f}  Dense {f1['Dense']:.4f}  ({dt:.0f}s)", flush=True)

    def correct(p):
        return [1.0 if a == g else 0.0 for a, g in zip(p, gold)]
    sma_c = correct(sma_p); pv, summ = {}, []
    for name, pred in (("BM25", bm_p), ("Dense", dn_p)):
        bs = paired_bootstrap(sma_c, correct(pred)); pv[name] = bs["p_value"]
        summ.append({"phase": phase, "domain": domain, "baseline": name,
                     "ho_density": f"{mean_ho:.4f}", "sma_f1": f"{f1['SMA']:.4f}",
                     "baseline_f1": f"{f1[name]:.4f}", "delta": f"{bs['delta']:.4f}",
                     "ci_low": f"{bs['ci_low']:.4f}", "ci_high": f"{bs['ci_high']:.4f}",
                     "cliffs": f"{cliffs_delta(sma_c, correct(pred)):.4f}"})
    holm = holm_bonferroni(pv)
    for s in summ:
        s["p_holm"] = f"{holm[s['baseline']]:.4f}"
    out = OUT / f"cd_{domain}_{phase}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    return {"f1": f1, "ho": mean_ho, "summ": summ}
