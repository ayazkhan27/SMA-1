"""Phase 4b 'before': generic structured adapter on real Diabetes-130.

Measures (a) higher-order-relation density of the generic encoding (expect ~0)
and (b) SMA vs BM25 vs dense on readmission-by-analogy retrieval (expect parity,
because flat tabular gives systematicity nothing to exploit). Deterministic, no
LLM. Writes reports/confirmatory/cd_diabetes_before.csv.
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sma.eval.diabetes import load_encounters, row_csv, row_text
from sma.encoders import get_encoder
from sma.index.macfac import MacFacIndex
from sma.ir.schema import Statement
from sma.eval.baselines.bm25 import rank_bm25_like
from sma.eval.baselines.dense import rank_tfidf_dense_batch
from sma.eval.metrics import macro_f1
from sma.eval.stats import paired_bootstrap, holm_bonferroni, cliffs_delta

OUT = pathlib.Path("reports/confirmatory/cd_diabetes_before.csv")


def ho_density(case) -> float:
    exprs = case.expressions()
    ho = sum(1 for s in exprs if any(isinstance(a, Statement) for a in s.args))
    return ho / max(len(exprs), 1)


def predict_by_vote(ranked_ids, label_of):
    votes = {"early": 0, "not": 0}
    for cid in ranked_ids:
        votes[label_of[cid]] += 1
    return "early" if votes["early"] >= votes["not"] else "not"


def main(sample=1000, index_n=700, k=10):
    encs = load_encounters(sample=sample, seed=7, balanced=True)
    enc_struct = get_encoder("structured")
    cases, label_of, dens = {}, {}, []
    docs = []  # (case_id, text) for baselines
    for e in encs:
        case = enc_struct.encode(row_csv(e), format="csv").case
        cases[case.case_id] = case
        label_of[case.case_id] = e.label
        dens.append(ho_density(case))
        docs.append((case.case_id, row_text(e)))
    mean_ho = sum(dens) / len(dens)
    print(f"encounters={len(encs)}  mean HO-density={mean_ho:.4f}  "
          f"(early={sum(1 for e in encs if e.label=='early')})", flush=True)

    ids = list(cases)
    index_ids, query_ids = ids[:index_n], ids[index_n:]
    index_cases = [cases[i] for i in index_ids]
    index_docs = [(i, dict(docs)[i]) for i in index_ids]
    doc_text = dict(docs)

    # SMA retrieval
    t0 = time.perf_counter()
    sma_index = MacFacIndex()
    sma_index.build(index_cases)
    sma_pred, gold = [], []
    bm_pred, dn_pred = [], []
    q_texts = [doc_text[q] for q in query_ids]
    dense_rk = rank_tfidf_dense_batch(q_texts, index_docs, k=k)
    for qi, q in enumerate(query_ids):
        gold.append(label_of[q])
        res = sma_index.retrieve(cases[q], k=k, shortlist=60, fac_budget=25)
        sma_pred.append(predict_by_vote([r.case_id for r in res], label_of))
        bm = rank_bm25_like(doc_text[q], index_docs, k=k)
        bm_pred.append(predict_by_vote([cid for cid, _ in bm], label_of))
        dn_pred.append(predict_by_vote([cid for cid, _ in dense_rk[qi]], label_of))
    dt = time.perf_counter() - t0

    def f1(pred):
        return macro_f1(gold, pred)
    res = {"SMA": f1(sma_pred), "BM25": f1(bm_pred), "Dense": f1(dn_pred)}
    print(f"macro-F1: {{'SMA': {res['SMA']:.4f}, 'BM25': {res['BM25']:.4f}, "
          f"'Dense': {res['Dense']:.4f}}}  ({dt:.0f}s)", flush=True)

    # per-query correctness for paired stats (SMA vs each baseline)
    def correct(pred):
        return [1.0 if p == g else 0.0 for p, g in zip(pred, gold)]
    sma_c = correct(sma_pred)
    pv, summ = {}, []
    for name, pred in (("BM25", bm_pred), ("Dense", dn_pred)):
        bs = paired_bootstrap(sma_c, correct(pred))
        pv[name] = bs["p_value"]
        summ.append({"phase": "before", "metric": "accuracy", "baseline": name,
                     "ho_density": f"{mean_ho:.4f}", "sma_f1": f"{res['SMA']:.4f}",
                     "baseline_f1": f"{res[name]:.4f}", "delta": f"{bs['delta']:.4f}",
                     "ci_low": f"{bs['ci_low']:.4f}", "ci_high": f"{bs['ci_high']:.4f}",
                     "cliffs": f"{cliffs_delta(sma_c, correct(pred)):.4f}"})
    holm = holm_bonferroni(pv)
    for s in summ:
        s["p_holm"] = f"{holm[s['baseline']]:.4f}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader(); w.writerows(summ)
    print(f"wrote {OUT}", flush=True)
    # Verdict by SIGNIFICANCE, not raw point gap: parity = no Holm-significant
    # difference from any baseline (the predicted outcome for flat encoding).
    sig = any(float(s["p_holm"]) < 0.05 for s in summ)
    print("VERDICT:", "SIGNIFICANT DIFFERENCE vs a baseline — investigate" if sig
          else "PARITY (no significant difference — flat encoding gives SMA no "
               "structural signal, as predicted by H7)", flush=True)


if __name__ == "__main__":
    main()
