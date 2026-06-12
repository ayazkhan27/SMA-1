"""Scorer gauntlet: SES vs MDL vs RRF(SES,MDL) vs surprisal-SES (ADR-004 sequel).

Decides score-v2 by measurement before the calibration freeze:
  A. family-hit@k on HDFS and BGL, stratified into common vs rare families
     (rare = <= 20 same-family sessions in the index);
  B. the EOF micro-case (rare-family retrieval on the 5k HDFS UI corpus);
  C. short-session bias indicator (mean lines of top-5 retrieved vs corpus).
The BGL->Spirit transfer leg for the surprisal scorer runs separately via
sma.eval.transfer_eval --scorer surprisal.

Usage: python3 -u scripts/scorer_gauntlet.py [--out reports/scorer_gauntlet.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random

import numpy as np

from sma.eval.baselines.hybrid_rrf import rrf_fuse
from sma.eval.family_labels import bgl_family, hdfs_family
from sma.eval.loghub_eval import sample_bgl_stratified, sample_hdfs_stratified
from sma.index.macfac import MacFacIndex
from sma.match.types import MatchConfig

SCORERS = ("ses", "mdl", "surprisal")
RARE_THRESHOLD = 20
K = 10


def build_indexes(cases):
    indexes = {}
    for scorer in SCORERS:
        idx = MacFacIndex(config=MatchConfig(scorer=scorer))
        idx.build(cases)
        indexes[scorer] = idx
    return indexes


def family_eval(dataset, sampled, families):
    rng = random.Random(101)
    rng.shuffle(sampled)
    split = int(len(sampled) * 0.8)
    index_data, query_data = sampled[:split], sampled[split:]

    from sma.encoders import get_encoder

    encoder = get_encoder("logs")
    index_cases = [encoder.encode(t, session_id=s).case for s, t, _ in index_data]
    case_meta = {
        c.case_id: (families[s], len(t.splitlines()))
        for c, (s, t, _) in zip(index_cases, index_data)
    }
    fam_counts = {}
    for fam, _ in case_meta.values():
        fam_counts[fam] = fam_counts.get(fam, 0) + 1

    indexes = build_indexes(index_cases)
    corpus_mean_lines = np.mean([lines for _, lines in case_meta.values()])

    stats = {v: {"common": [], "rare": [], "lines": []} for v in (*SCORERS, "rrf")}
    n_q = 0
    for qi, (sid, text, _label) in enumerate(query_data, 1):
        q_family = families[sid]
        if q_family == "normal":
            continue
        n_q += 1
        q_case = encoder.encode(text, session_id=sid).case
        rankings = {}
        for scorer in SCORERS:
            res = indexes[scorer].retrieve(q_case, k=K, shortlist=40, fac_budget=20)
            rankings[scorer] = [(r.case_id, r.ses_n) for r in res]
        rankings["rrf"] = rrf_fuse([rankings["ses"], rankings["mdl"]], top_k=K)

        same_family_available = fam_counts.get(q_family, 0)
        stratum = "rare" if same_family_available <= RARE_THRESHOLD else "common"
        for variant, ranked in rankings.items():
            top5 = [cid for cid, _ in ranked[:5]]
            hits = sum(1 for cid in top5 if case_meta[cid][0] == q_family)
            denom = min(5, same_family_available)
            stats[variant][stratum].append(hits / denom if denom else 0.0)
            stats[variant]["lines"].extend(case_meta[cid][1] for cid in top5)
        if qi % 40 == 0:
            print(f"  {dataset}: {qi}/{len(query_data)} queries", flush=True)

    rows = []
    for variant, s in stats.items():
        rows.append({
            "dataset": dataset, "variant": variant,
            "family_hit5_common": f"{np.mean(s['common']):.4f}" if s["common"] else "",
            "n_common": len(s["common"]),
            "family_hit5_rare": f"{np.mean(s['rare']):.4f}" if s["rare"] else "",
            "n_rare": len(s["rare"]),
            "mean_lines_top5": f"{np.mean(s['lines']):.1f}" if s["lines"] else "",
            "corpus_mean_lines": f"{corpus_mean_lines:.1f}",
        })
        print(f"{dataset} {variant:10s} common={rows[-1]['family_hit5_common']} "
              f"(n={len(s['common'])}) rare={rows[-1]['family_hit5_rare']} (n={len(s['rare'])}) "
              f"top5-lines={rows[-1]['mean_lines_top5']} vs corpus {corpus_mean_lines:.1f}", flush=True)
    return rows


def eof_case():
    rows_src = [json.loads(l) for l in pathlib.Path("data/processed/ui_corpus_hdfs.jsonl").open()]
    random.Random(7).shuffle(rows_src)
    rows_src = rows_src[:5000]
    from sma.encoders import get_encoder

    encoder = get_encoder("logs")
    cases = [encoder.encode(r["text"], session_id=r["id"]).case for r in rows_src]
    text_by_id = {c.case_id: r["text"] for c, r in zip(cases, rows_src)}
    indexes = build_indexes(cases)

    q = ("Last night a tranche of block writes died mid-pipeline: replicas start receiving, then the "
         "receiving datanode hits an EOFException in receiveBlock and writeBlock fails before the "
         "pipeline closes. Historically, is the stream breaking on the sending or receiving replica?")
    q_case = encoder.encode(q).case
    out = []
    rankings = {}
    for scorer in SCORERS:
        res = indexes[scorer].retrieve(q_case, k=5, shortlist=200, fac_budget=30)
        rankings[scorer] = [(r.case_id, r.ses_n) for r in res]
    rankings["rrf"] = rrf_fuse([rankings["ses"], rankings["mdl"]], top_k=5)
    for variant, ranked in rankings.items():
        eof = sum(1 for cid, _ in ranked if "EOFException" in text_by_id[cid])
        out.append({"dataset": "EOF_case", "variant": variant,
                    "family_hit5_common": "", "n_common": "",
                    "family_hit5_rare": f"{eof}/5", "n_rare": 1,
                    "mean_lines_top5": "", "corpus_mean_lines": ""})
        print(f"EOF case {variant:10s}: {eof}/5 EOF-family in top-5", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/scorer_gauntlet.csv")
    args = parser.parse_args()
    all_rows = []

    print("=== Part A1: HDFS family strata ===", flush=True)
    hdfs = sample_hdfs_stratified(pathlib.Path("data/raw/loghub_raw/HDFS_v1.zip"), 1000, seed=42)
    hdfs_families = {sid: hdfs_family(text, label) for sid, text, label in hdfs}
    all_rows += family_eval("HDFS", hdfs, hdfs_families)

    print("=== Part A2: BGL family strata ===", flush=True)
    bgl_zip = pathlib.Path("data/raw/loghub_raw/BGL.zip")
    bgl = sample_bgl_stratified(bgl_zip, 1000, seed=42)
    bgl_families = bgl_family(bgl_zip, [sid for sid, _, _ in bgl])
    bgl_families = {sid: bgl_families.get(sid, "normal") for sid, _, _ in bgl}
    all_rows += family_eval("BGL", bgl, bgl_families)

    print("=== Part B: EOF micro-case ===", flush=True)
    all_rows += eof_case()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
