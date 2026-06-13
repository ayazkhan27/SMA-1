"""Phase 4b 'before': generic structured adapter on a real cross-domain dataset.

Measures (a) higher-order-relation density of the generic encoding (expect ~0)
and (b) SMA vs BM25 vs dense on label-by-analogy retrieval (expect parity,
because flat tabular gives systematicity nothing to exploit). Deterministic, no
LLM. Writes reports/confirmatory/cd_<domain>_before.csv.

  python3 scripts/crossdomain_before.py --domain diabetes
  python3 scripts/crossdomain_before.py --domain ieee
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sma.encoders import get_encoder
from sma.index.macfac import MacFacIndex
from sma.ir.schema import Statement
from sma.eval.baselines.bm25 import rank_bm25_like
from sma.eval.baselines.dense import rank_tfidf_dense_batch
from sma.eval.metrics import macro_f1
from sma.eval.stats import paired_bootstrap, holm_bonferroni, cliffs_delta


def _diabetes():
    from sma.eval.diabetes import load_encounters, row_csv, row_text
    return (lambda s, sd: load_encounters(sample=s, seed=sd, balanced=True),
            row_csv, row_text, ("early", "not"))


def _ieee():
    from sma.eval.ieee_cis import load_transactions, row_csv, row_text
    return (lambda s, sd: load_transactions(sample=s, seed=sd, balanced=True),
            row_csv, row_text, ("fraud", "legit"))


DOMAINS = {"diabetes": _diabetes, "ieee": _ieee}


def ho_density(case) -> float:
    exprs = case.expressions()
    ho = sum(1 for s in exprs if any(isinstance(a, Statement) for a in s.args))
    return ho / max(len(exprs), 1)


def vote(ranked_ids, label_of, labels):
    pos, neg = labels
    tally = {pos: 0, neg: 0}
    for cid in ranked_ids:
        tally[label_of[cid]] += 1
    return pos if tally[pos] >= tally[neg] else neg


def main(domain: str, sample=1000, k=10):
    index_n = int(sample * 0.7)   # 70/30 index/query split, scales with sample
    loader, row_csv, row_text, labels = DOMAINS[domain]()
    items = loader(sample, 7)
    enc_struct = get_encoder("structured")
    cases, label_of, dens, docs = {}, {}, [], []
    for it in items:
        case = enc_struct.encode(row_csv(it), format="csv").case
        cases[case.case_id] = case
        label_of[case.case_id] = it.label
        dens.append(ho_density(case))
        docs.append((case.case_id, row_text(it)))
    mean_ho = sum(dens) / len(dens)
    pos = sum(1 for it in items if it.label == labels[0])
    print(f"[{domain}] items={len(items)}  mean HO-density={mean_ho:.4f}  "
          f"({labels[0]}={pos})", flush=True)

    ids = list(cases)
    index_ids, query_ids = ids[:index_n], ids[index_n:]
    index_cases = [cases[i] for i in index_ids]
    doc_text = dict(docs)
    index_docs = [(i, doc_text[i]) for i in index_ids]

    t0 = time.perf_counter()
    sma_index = MacFacIndex()
    sma_index.build(index_cases)
    gold, sma_pred, bm_pred, dn_pred = [], [], [], []
    dense_rk = rank_tfidf_dense_batch([doc_text[q] for q in query_ids], index_docs, k=k)
    for qi, q in enumerate(query_ids):
        gold.append(label_of[q])
        res = sma_index.retrieve(cases[q], k=k, shortlist=60, fac_budget=25)
        sma_pred.append(vote([r.case_id for r in res], label_of, labels))
        bm = rank_bm25_like(doc_text[q], index_docs, k=k)
        bm_pred.append(vote([cid for cid, _ in bm], label_of, labels))
        dn_pred.append(vote([cid for cid, _ in dense_rk[qi]], label_of, labels))
    dt = time.perf_counter() - t0

    res = {"SMA": macro_f1(gold, sma_pred), "BM25": macro_f1(gold, bm_pred),
           "Dense": macro_f1(gold, dn_pred)}
    print(f"[{domain}] macro-F1: SMA {res['SMA']:.4f}  BM25 {res['BM25']:.4f}  "
          f"Dense {res['Dense']:.4f}  ({dt:.0f}s)", flush=True)

    def correct(pred):
        return [1.0 if p == g else 0.0 for p, g in zip(pred, gold)]
    sma_c = correct(sma_pred)
    pv, summ = {}, []
    for name, pred in (("BM25", bm_pred), ("Dense", dn_pred)):
        bs = paired_bootstrap(sma_c, correct(pred))
        pv[name] = bs["p_value"]
        summ.append({"phase": "before", "domain": domain, "baseline": name,
                     "ho_density": f"{mean_ho:.4f}", "sma_f1": f"{res['SMA']:.4f}",
                     "baseline_f1": f"{res[name]:.4f}", "delta": f"{bs['delta']:.4f}",
                     "ci_low": f"{bs['ci_low']:.4f}", "ci_high": f"{bs['ci_high']:.4f}",
                     "cliffs": f"{cliffs_delta(sma_c, correct(pred)):.4f}"})
    holm = holm_bonferroni(pv)
    for s in summ:
        s["p_holm"] = f"{holm[s['baseline']]:.4f}"
    out = pathlib.Path(f"reports/confirmatory/cd_{domain}_before.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    sig = any(float(s["p_holm"]) < 0.05 for s in summ)
    print(f"[{domain}] wrote {out}", flush=True)
    print(f"[{domain}] VERDICT:", "SIGNIFICANT DIFFERENCE vs a baseline — investigate" if sig
          else "PARITY (no significant difference — flat encoding gives SMA no "
               "structural signal, as predicted by H7)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), default="diabetes")
    ap.add_argument("--sample", type=int, default=1000)
    a = ap.parse_args()
    main(a.domain, sample=a.sample)
