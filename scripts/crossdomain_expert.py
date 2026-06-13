"""Phase 4b third arm: the EXPERT hand-designed healthcare adapter vs the generic
(parity) and LLM-drafted (lift-toward-parity) encodings, on real Diabetes-130.

SMA retrieves over expert clinical structure (comorbidity, complication, therapy
class, control->escalation chains); BM25/dense use the same flat record text as
the other arms (fair: does expert STRUCTURE beat value-based retrieval?).
Deterministic, no LLM. Writes reports/confirmatory/cd_diabetes_expert.csv.
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sma.eval.diabetes import load_encounters, row_text
from sma.encoders.healthcare import HealthcareEncoder
from sma.index.macfac import MacFacIndex
from sma.ir.schema import Statement
from sma.eval.baselines.bm25 import rank_bm25_like
from sma.eval.baselines.dense import rank_tfidf_dense_batch
from sma.eval.metrics import macro_f1
from sma.eval.stats import paired_bootstrap, holm_bonferroni, cliffs_delta
from scripts.crossdomain_before import ho_density, vote

LABELS = ("early", "not")
OUT = pathlib.Path("reports/confirmatory/cd_diabetes_expert.csv")


def main(sample=1000, k=10):
    index_n = int(sample * 0.7)
    items = load_encounters(sample=sample, seed=7, balanced=True)
    enc = HealthcareEncoder()
    cases, label_of, dens, docs = {}, {}, [], []
    for it in items:
        case = enc.encode_record(it.fields)
        cases[case.case_id] = case
        label_of[case.case_id] = it.label
        dens.append(ho_density(case))
        docs.append((case.case_id, row_text(it)))
    mean_ho = sum(dens) / len(dens)
    mean_stmts = sum(len(c.expressions()) for c in cases.values()) / len(cases)
    print(f"[expert] items={len(items)}  mean statements={mean_stmts:.1f}  "
          f"mean HO-density={mean_ho:.4f}", flush=True)

    ids = list(cases)
    index_ids, query_ids = ids[:index_n], ids[index_n:]
    doc_text = dict(docs)
    index_docs = [(i, doc_text[i]) for i in index_ids]
    t0 = time.perf_counter()
    sma_index = MacFacIndex(); sma_index.build([cases[i] for i in index_ids])
    gold, sma_pred, bm_pred, dn_pred = [], [], [], []
    dense_rk = rank_tfidf_dense_batch([doc_text[q] for q in query_ids], index_docs, k=k)
    for qi, q in enumerate(query_ids):
        gold.append(label_of[q])
        res = sma_index.retrieve(cases[q], k=k, shortlist=60, fac_budget=25)
        sma_pred.append(vote([r.case_id for r in res], label_of, LABELS))
        bm = rank_bm25_like(doc_text[q], index_docs, k=k)
        bm_pred.append(vote([cid for cid, _ in bm], label_of, LABELS))
        dn_pred.append(vote([cid for cid, _ in dense_rk[qi]], label_of, LABELS))
    dt = time.perf_counter() - t0
    res = {"SMA": macro_f1(gold, sma_pred), "BM25": macro_f1(gold, bm_pred),
           "Dense": macro_f1(gold, dn_pred)}
    print(f"[expert] macro-F1: SMA {res['SMA']:.4f}  BM25 {res['BM25']:.4f}  "
          f"Dense {res['Dense']:.4f}  ({dt:.0f}s)", flush=True)

    def correct(p):
        return [1.0 if a == g else 0.0 for a, g in zip(p, gold)]
    sma_c = correct(sma_pred); pv, summ = {}, []
    for name, pred in (("BM25", bm_pred), ("Dense", dn_pred)):
        bs = paired_bootstrap(sma_c, correct(pred)); pv[name] = bs["p_value"]
        summ.append({"phase": "expert", "domain": "diabetes", "baseline": name,
                     "ho_density": f"{mean_ho:.4f}", "sma_f1": f"{res['SMA']:.4f}",
                     "baseline_f1": f"{res[name]:.4f}", "delta": f"{bs['delta']:.4f}",
                     "ci_low": f"{bs['ci_low']:.4f}", "ci_high": f"{bs['ci_high']:.4f}",
                     "cliffs": f"{cliffs_delta(sma_c, correct(pred)):.4f}"})
    holm = holm_bonferroni(pv)
    for s in summ:
        s["p_holm"] = f"{holm[s['baseline']]:.4f}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    sig_win = all(float(s["delta"]) > 0 and float(s["p_holm"]) < 0.05 for s in summ)
    sig_lose = any(float(s["delta"]) < 0 and float(s["p_holm"]) < 0.05 for s in summ)
    print(f"[expert] wrote {OUT}", flush=True)
    print("[expert] VERDICT:", "SMA SIGNIFICANTLY BEATS both baselines" if sig_win
          else "SMA significantly BELOW a baseline" if sig_lose
          else "PARITY / mixed (see CIs)", flush=True)


if __name__ == "__main__":
    main()
