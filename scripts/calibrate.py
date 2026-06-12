"""Calibration grid (blueprint 8.6): tune dials on VALIDATION data only.

Sweeps the registered open dials over validation samples that are seed-
disjoint from the final test protocol (test uses seed 42; validation here
uses seed 7 samples and SSB seeds 29/31). Outputs one row per config to
reports/calibration_grid.csv. The freeze (prereg tag) selects from this grid;
after the freeze these dials never move again.

Dials and their registered priors/evidence:
- scorer:        ses vs surprisal   (ADR-005/006: EOF-case vs rare-strata split)
- normalization: max vs target      (ADR-006: transfer vs haystack split)
- gamma:         0.125 / 0.25 / 0.5 (systematicity trickle-down)
- rho:           0.90 / 0.95        (lattice ascension penalty; SSB-sensitive)
"""

from __future__ import annotations

import csv
import itertools
import json
import pathlib
import random
import time

from sma.encoders import get_encoder
from sma.eval.family_labels import hdfs_family
from sma.eval.loghub_eval import sample_hdfs_stratified
from sma.eval.ssb_generator import build_canonicalizer, generate_triples
from sma.index.macfac import MacFacIndex
from sma.match.types import MatchConfig

OUT = pathlib.Path("reports/calibration_grid.csv")

GRID = {
    "scorer": ["ses", "surprisal"],
    "normalization": ["max", "target"],
    "gamma": [0.125, 0.25, 0.5],
    "rho": [0.90, 0.95],
}

VAL_SEED = 7          # disjoint from the test seed (42)
SSB_VAL_SEEDS = (29, 31)  # disjoint from fixture seeds (11, 19, 23)


def ssb_score(cfg: MatchConfig, n: int = 12) -> float:
    """Fraction of SSB validation triples where the analog ranks first."""
    hits = total = 0
    for seed in SSB_VAL_SEEDS:
        triples = generate_triples(n, seed=seed)
        canon = build_canonicalizer(triples)
        ssb_cfg = MatchConfig(scorer=cfg.scorer, normalization=cfg.normalization,
                              gamma=cfg.gamma, rho=cfg.rho, delta=2)
        index = MacFacIndex(config=ssb_cfg, canon=canon)
        lib = []
        for t in triples:
            lib.extend([t.analog, t.distractor])
        index.build(lib)
        for t in triples:
            res = index.retrieve(t.query, k=5, shortlist=2 * n, fac_budget=50)
            hits += bool(res) and res[0].case_id == t.analog.case_id
            total += 1
    return hits / total


def family_scores(cfg: MatchConfig, sampled, families) -> tuple[float, float]:
    """family-hit@5 (common, rare) on the HDFS validation sample."""
    rng = random.Random(101)
    data = list(sampled)
    rng.shuffle(data)
    split = int(len(data) * 0.8)
    index_data, query_data = data[:split], data[split:]
    encoder = get_encoder("logs")
    index_cases = [encoder.encode(t, session_id=s).case for s, t, _ in index_data]
    fam_of = {c.case_id: families[s] for c, (s, _, _) in zip(index_cases, index_data)}
    fam_counts: dict[str, int] = {}
    for f in fam_of.values():
        fam_counts[f] = fam_counts.get(f, 0) + 1
    index = MacFacIndex(config=cfg)
    index.build(index_cases)
    common, rare = [], []
    for sid, text, _label in query_data:
        fam = families[sid]
        if fam == "normal":
            continue
        q = encoder.encode(text, session_id=sid).case
        res = index.retrieve(q, k=5, shortlist=40, fac_budget=20)
        avail = fam_counts.get(fam, 0)
        denom = min(5, avail)
        if not denom:
            continue
        hit = sum(1 for r in res if fam_of.get(r.case_id) == fam) / denom
        (rare if avail <= 20 else common).append(hit)
    c = sum(common) / len(common) if common else 0.0
    r = sum(rare) / len(rare) if rare else 0.0
    return c, r


def haystack_score(cfg: MatchConfig, rows, n_probes: int = 10) -> float:
    """Leave-one-out needle retrieval on the Liberty validation slice."""
    encoder = get_encoder("logs")
    index = MacFacIndex(config=cfg)
    cases = {}
    for r in rows:
        c = encoder.encode(r["text"], session_id=r["id"]).case
        cases[r["id"]] = c
        index.add(c)
    label_of = {cases[r["id"]].case_id: r["label"] for r in rows}
    needles = [r for r in rows if r["label"] == "Anomaly"][:n_probes]
    scores = []
    for r in needles:
        res = index.retrieve(cases[r["id"]], k=6, shortlist=200, fac_budget=30)
        top = [x for x in res if x.case_id != cases[r["id"]].case_id][:5]
        scores.append(sum(1 for x in top if label_of.get(x.case_id) == "Anomaly") / 5)
    return sum(scores) / len(scores) if scores else 0.0


def main() -> int:
    print("loading validation data (seed 7, disjoint from test seed 42)...", flush=True)
    hdfs = sample_hdfs_stratified(
        pathlib.Path("data/raw/loghub_raw/HDFS_v1.zip"), sample_size=500, seed=VAL_SEED)
    families = {sid: hdfs_family(text, label) for sid, text, label in hdfs}
    liberty = [json.loads(l) for l in
               pathlib.Path("data/processed/ui_corpus_liberty.jsonl").open()]
    random.Random(VAL_SEED).shuffle(liberty)
    liberty = liberty[:1500]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scorer", "normalization", "gamma", "rho",
              "ssb_r1", "hdfs_family_common", "hdfs_family_rare", "haystack_needles", "seconds"]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        combos = list(itertools.product(*GRID.values()))
        for i, (scorer, norm, gamma, rho) in enumerate(combos, 1):
            cfg = MatchConfig(scorer=scorer, normalization=norm, gamma=gamma, rho=rho)
            t0 = time.perf_counter()
            ssb = ssb_score(cfg)
            fam_c, fam_r = family_scores(cfg, hdfs, families)
            hay = haystack_score(cfg, liberty)
            secs = time.perf_counter() - t0
            row = {"scorer": scorer, "normalization": norm, "gamma": gamma, "rho": rho,
                   "ssb_r1": f"{ssb:.4f}", "hdfs_family_common": f"{fam_c:.4f}",
                   "hdfs_family_rare": f"{fam_r:.4f}", "haystack_needles": f"{hay:.4f}",
                   "seconds": f"{secs:.0f}"}
            writer.writerow(row)
            fh.flush()
            print(f"[{i}/{len(combos)}] {row}", flush=True)
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
