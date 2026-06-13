"""Phase 4b 'after': the dynamic adapter. The LLM DRAFTS higher-order relation
rules for a domain (it never encodes); a deterministic applier emits HO-relation
statements from those rules; we re-encode and re-measure whether SMA's structural
advantage RECOVERS once structure exists (H8).

Drafting is admin-gated in production (ADR-007); here it is a research run.
Drafted rules are content-addressed + 'LLM-proposed, unreviewed' tainted and
saved to reports/confirmatory/cd_<domain>_rules.json.

  python3 scripts/crossdomain_after.py --domain diabetes
  python3 scripts/crossdomain_after.py --domain ieee
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import blake3

from sma.index.macfac import MacFacIndex
from sma.ir.schema import Statement, entity, make_case, stmt
from sma.eval.baselines.bm25 import rank_bm25_like
from sma.eval.baselines.dense import rank_tfidf_dense_batch
from sma.eval.metrics import macro_f1
from sma.eval.stats import paired_bootstrap, holm_bonferroni, cliffs_delta
from scripts.crossdomain_before import DOMAINS, ho_density, vote

ROOT = pathlib.Path(__file__).resolve().parents[1]

DRAFT_SYS = (
    "You design DETERMINISTIC higher-order relation rules for a tabular domain so "
    "a structure-mapping memory can compare records by relational structure, not "
    "just values. You are given the column names and a few example rows. Propose "
    "3-6 rules that connect CAUSALLY, TEMPORALLY, or SEMANTICALLY related columns "
    "(e.g. diagnosis->treatment->outcome; or entity links like same-card/same-"
    "address; or event sequences). Output ONLY a JSON array; each rule is "
    '{"relation": "<shortName>", "subjects": ["col", ...], "objects": ["col", ...], '
    '"kind": "pairwise"|"subject_object"}. "pairwise" relates every pair within '
    'subjects; "subject_object" relates each subject column to each object column. '
    "Use ONLY columns from the provided list. Use the per-column value summary to "
    "judge what each column MEANS (codes, categories, counts, time-deltas) and "
    "relate only columns that are genuinely related; do NOT blanket-connect every "
    "count column to every other - that adds noise, not structure. Prefer a few "
    "high-value relations over many weak ones. No prose.")


def schema_summary(train_items, max_vals: int = 12) -> str:
    """Per-column distinct-value summary from TRAINING rows only, so the drafter
    understands cryptic columns (e.g. C1 is a count, D1 a day-delta) and drafts
    selective rules. No labels, no test rows."""
    from collections import defaultdict
    vals = defaultdict(set)
    for it in train_items:
        for k, v in it.fields.items():
            vals[k].add(v)
    lines = []
    for col in sorted(vals):
        vs = sorted(vals[col])
        shown = ", ".join(vs[:max_vals])
        extra = f" … (+{len(vs)-max_vals} more, {len(vs)} distinct)" if len(vs) > max_vals else ""
        lines.append(f"  {col}: {shown}{extra}")
    return "\n".join(lines)


def _load_env():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())


def propose_rules(llm, columns, summary, samples):
    cols = ", ".join(columns)
    ex = "\n".join("row: " + ", ".join(f"{k}={v}" for k, v in s.items()) for s in samples)
    out = llm.complete([{"role": "system", "content": DRAFT_SYS},
                        {"role": "user", "content":
                         f"Columns: {cols}\n\nPer-column values (training data):\n{summary}"
                         f"\n\nExample rows:\n{ex}"}],
                       max_tokens=900)
    txt = out.strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1].lstrip("json").strip()
    try:
        rules = json.loads(txt)
    except json.JSONDecodeError:
        return []
    colset = set(columns)
    clean = []
    for r in rules if isinstance(rules, list) else []:
        subs = [c for c in r.get("subjects", []) if c in colset]
        objs = [c for c in r.get("objects", []) if c in colset]
        kind = r.get("kind", "pairwise")
        name = str(r.get("relation", "rel"))[:24]
        if subs and (kind == "pairwise" or objs):
            clean.append({"relation": name, "subjects": subs, "objects": objs, "kind": kind})
    return clean


def apply_rules(fields: dict, rules: list) -> list[Statement]:
    """Base flat (col row value) statements PLUS drafted higher-order relations
    over them. Shared first-order sub-statements are hash-consed by make_case."""
    row = entity("row", "row")
    def fo(col):  # first-order statement for a column present in this record
        return stmt(col, row, entity(fields[col], "value")) if col in fields else None
    out = [s for s in (fo(c) for c in fields) if s is not None]
    for r in rules:
        subs = [fo(c) for c in r["subjects"] if c in fields]
        subs = [s for s in subs if s is not None]
        if r["kind"] == "pairwise":
            for i in range(len(subs)):
                for j in range(i + 1, len(subs)):
                    out.append(stmt(r["relation"], subs[i], subs[j]))  # higher-order
        else:
            objs = [fo(c) for c in r["objects"] if c in fields]
            objs = [s for s in objs if s is not None]
            for s in subs:
                for o in objs:
                    out.append(stmt(r["relation"], s, o))
    return out


def main(domain: str, sample=1000, k=10, n_draft_rows=25):
    index_n = int(sample * 0.7)
    loader, row_csv, row_text, labels = DOMAINS[domain]()
    items = loader(sample, 7)
    # Split FIRST. The rule-drafter sees TRAINING (index) rows only — never the
    # query/test rows — so the drafted encoder is not fit to the eval set.
    train_items = items[:index_n]
    _load_env()
    from sma.agent.llm import DeepSeekOrchestrator
    llm = DeepSeekOrchestrator()

    columns = sorted({c for it in train_items for c in it.fields})
    summary = schema_summary(train_items)            # per-column distinct values (train)
    samples = [train_items[i].fields for i in range(min(n_draft_rows, len(train_items)))]
    print(f"[{domain}] drafting HO-rules from {len(columns)} cols + value summary "
          f"+ {len(samples)} train rows (test rows withheld)...", flush=True)
    rules = propose_rules(llm, columns, summary, samples)
    print(f"[{domain}] drafted {len(rules)} rules:",
          [r["relation"] for r in rules], flush=True)
    if not rules:
        sys.exit(f"[{domain}] no rules drafted; aborting after-run.")
    # ADR-007: content-address + taint the drafted rules
    blob = json.dumps(rules, sort_keys=True)
    rec = {"domain": domain, "rules": rules,
           "content_id": blake3.blake3(blob.encode()).hexdigest()[:16],
           "taint": "LLM-proposed, unreviewed"}
    (ROOT / f"reports/confirmatory/cd_{domain}_rules.json").write_text(json.dumps(rec, indent=2))

    cases, label_of, dens, docs = {}, {}, [], []
    for it in items:
        case = make_case(apply_rules(it.fields, rules), {"adapter": "draft", "domain": domain})
        cases[case.case_id] = case
        label_of[case.case_id] = it.label
        dens.append(ho_density(case))
        docs.append((case.case_id, row_text(it)))
    mean_ho = sum(dens) / len(dens)
    print(f"[{domain}] re-encoded; mean HO-density now {mean_ho:.4f} (was 0.0000)", flush=True)

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
        sma_pred.append(vote([r.case_id for r in res], label_of, labels))
        bm = rank_bm25_like(doc_text[q], index_docs, k=k)
        bm_pred.append(vote([cid for cid, _ in bm], label_of, labels))
        dn_pred.append(vote([cid for cid, _ in dense_rk[qi]], label_of, labels))
    dt = time.perf_counter() - t0
    res = {"SMA": macro_f1(gold, sma_pred), "BM25": macro_f1(gold, bm_pred),
           "Dense": macro_f1(gold, dn_pred)}
    print(f"[{domain}] AFTER macro-F1: SMA {res['SMA']:.4f}  BM25 {res['BM25']:.4f}  "
          f"Dense {res['Dense']:.4f}  ({dt:.0f}s)", flush=True)

    def correct(p):
        return [1.0 if a == g else 0.0 for a, g in zip(p, gold)]
    sma_c = correct(sma_pred); pv, summ = {}, []
    for name, pred in (("BM25", bm_pred), ("Dense", dn_pred)):
        bs = paired_bootstrap(sma_c, correct(pred)); pv[name] = bs["p_value"]
        summ.append({"phase": "after", "domain": domain, "baseline": name,
                     "ho_density": f"{mean_ho:.4f}", "sma_f1": f"{res['SMA']:.4f}",
                     "baseline_f1": f"{res[name]:.4f}", "delta": f"{bs['delta']:.4f}",
                     "ci_low": f"{bs['ci_low']:.4f}", "ci_high": f"{bs['ci_high']:.4f}",
                     "cliffs": f"{cliffs_delta(sma_c, correct(pred)):.4f}"})
    holm = holm_bonferroni(pv)
    for s in summ:
        s["p_holm"] = f"{holm[s['baseline']]:.4f}"
    out = ROOT / f"reports/confirmatory/cd_{domain}_after.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    print(f"[{domain}] wrote {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), default="diabetes")
    ap.add_argument("--sample", type=int, default=1000)
    a = ap.parse_args()
    main(a.domain, sample=a.sample)
