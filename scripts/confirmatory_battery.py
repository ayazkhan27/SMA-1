#!/usr/bin/env python3
"""Single-shot confirmatory battery (configs/preregistration.md section 4).

Implements the frozen test protocol EXACTLY:

  T1   cross-system transfer: BGL->Spirit, BGL->Thunderbird, HDFS->OpenStack;
       index 800 / query 200 per seed; macro-F1, label-hit@1; methods = SMA
       plus every baseline wired into sma.eval.transfer_eval (picked up
       dynamically from its retrievers dict).
  T2   within-system: HDFS family-hit@5 (common/rare strata) and BGL triage;
       1000-session stratified samples.
  T3   code: BugsInPy leave-one-project-out category@1 (deterministic, no
       seed dimension).
  T4   haystack: Liberty 5000-session corpus @ 5% needles with OUT-OF-CORPUS
       needle probes (sampled beyond the corpus line slice of liberty2.gz);
       hybrid fused (RRF of bm25+dense+sma) is the registered primary, solo
       modes recorded alongside.
  SSB  forced-choice + library r1/MRR vs BM25/TF-IDF dense.

Test seeds {201..205} (T1/T2/T4) and SSB seeds {41, 43} n=100 are REGISTERED
and were never used during development or calibration. The battery is
single-shot: a task whose output already exists refuses to run; crash reruns
must be logged in docs/STATUS.md first and require --force-rerun-logged.

Statistics (prereg section 5, sma.eval.stats): per-query paired bootstrap
(10,000 resamples) for SMA-vs-baseline deltas with 95% CIs, Holm-Bonferroni
within each dataset's family of baseline comparisons, Cliff's delta effect
sizes, multi-seed means +/- s.d.

Usage:
  python3 -u scripts/confirmatory_battery.py --task t1 --smoke   # plumbing
  python3 -u scripts/confirmatory_battery.py --task all          # THE run

--smoke runs tiny sizes on VALIDATION seed 7 (SSB seed 29) only and writes
*_smoke.csv files; it never touches the registered outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import statistics
import sys
import time
from collections import Counter, defaultdict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sma.eval.stats import cliffs_delta, holm_bonferroni, paired_bootstrap  # noqa: E402

# ---------------------------------------------------------------------------
# Registered constants (prereg section 4). DO NOT EDIT after prereg-v1.
# ---------------------------------------------------------------------------
TEST_SEEDS = (201, 202, 203, 204, 205)
SSB_TEST_SEEDS = (41, 43)
SSB_N = 100

# Validation-only seeds for --smoke plumbing runs (same seeds calibration
# used; disjoint from every registered test seed above).
SMOKE_SEED = 7
SSB_SMOKE_SEED = 29

OUT_DIR = REPO_ROOT / "reports" / "confirmatory"
RAW_DIR = REPO_ROOT / "data" / "raw" / "loghub_raw"
HDFS_ZIP = RAW_DIR / "HDFS_v1.zip"
BGL_ZIP = RAW_DIR / "BGL.zip"
LIBERTY_GZ = RAW_DIR / "liberty2.gz"
BUGSINPY_ROOT = REPO_ROOT / "data" / "raw" / "bugsinpy"
T4_CORPUS = REPO_ROOT / "data" / "processed" / "ui_corpus_liberty.jsonl"

T1_PAIRS = "BGL->Spirit,BGL->Thunderbird,HDFS->OpenStack"
T1_INDEX_SIZE, T1_QUERY_SIZE = 800, 200
T2_SAMPLE_SIZE = 1000
RARE_FAMILY_MAX = 20  # calibrate.family_scores stratum boundary (<=20 = rare)

# ui_corpus_liberty.jsonl was sampled from liberty2.gz lines 30M-60M (the
# alert-storm transition region; docs/STATUS.md 2026-06-12). Out-of-corpus
# needle probes therefore come from STRICTLY BEYOND that slice.
T4_PROBE_LINE_START = 60_000_000
T4_PROBE_LINE_CAP = 80_000_000
T4_PROBES_PER_SEED = 40
T4_RRF_DEPTH = 20  # hybrid posture: RRF over bm25+dense+sma top-20s
T4_SESSION_LINE_CAP = 60  # sample_cfdr_haystack stores at most 60 lines

SMA_METHOD = "SMA"

STAT_FIELDS = [
    "task", "leg", "metric", "reference", "baseline", "n_queries",
    "reference_mean", "baseline_mean", "delta", "ci_low", "ci_high",
    "p_raw", "p_holm", "cliffs_delta",
]
SUMMARY_FIELDS = ["task", "leg", "method", "metric", "mean", "sd", "n_seeds", "per_seed"]


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


def out_path(stem: str, smoke: bool) -> pathlib.Path:
    return OUT_DIR / f"{stem}{'_smoke' if smoke else ''}.csv"


def single_shot_guard(task: str, primary: pathlib.Path, force: bool, smoke: bool) -> bool:
    """Enforce the one-execution-per-cell discipline of prereg section 4."""
    if smoke or not primary.exists():
        return True
    if force:
        print(f"[{task}] {primary} exists but --force-rerun-logged was supplied; "
              f"proceeding on the operator's assertion that the rerun is logged "
              f"in docs/STATUS.md.", flush=True)
        return True
    bar = "!" * 78
    print(bar)
    print(f"REFUSING TO RUN {task.upper()}: {primary} already exists.")
    print("The confirmatory battery is SINGLE-SHOT (configs/preregistration.md")
    print("section 4): one execution per cell, no reruns except after crashes.")
    print("A crash rerun MUST be logged in docs/STATUS.md BEFORE it is launched;")
    print("once logged, re-invoke with --force-rerun-logged to proceed.")
    print(bar, flush=True)
    return False


def write_csv(path: pathlib.Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}", flush=True)


def per_query_scores(rows: list[dict], metric_field: str) -> dict:
    """rows -> {leg: {method: {seed|query_id: score}}} (blank scores skipped)."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        value = r.get(metric_field, "")
        if value == "" or value is None:
            continue
        qid = f"{r.get('seed', '')}|{r['query_id']}"
        out[r["leg"]][r["method"]][qid] = float(value)
    return out


def compute_stats(task: str, per_query: dict, metric: str, reference: str = SMA_METHOD) -> list[dict]:
    """Paired bootstrap + Holm (within each leg's comparison family) + Cliff's
    delta for reference-vs-every-other-method, pooled across seeds."""
    rows = []
    for leg in sorted(per_query):
        methods = per_query[leg]
        if reference not in methods:
            continue
        ref_scores = methods[reference]
        prelim = []
        for m in sorted(methods):
            if m == reference:
                continue
            shared = sorted(set(ref_scores) & set(methods[m]))
            if not shared:
                continue
            a = [ref_scores[q] for q in shared]
            b = [methods[m][q] for q in shared]
            prelim.append((m, a, b, paired_bootstrap(a, b)))
        if not prelim:
            continue
        holm = holm_bonferroni({m: boot["p_value"] for m, _, _, boot in prelim})
        for m, a, b, boot in prelim:
            rows.append({
                "task": task, "leg": leg, "metric": metric,
                "reference": reference, "baseline": m,
                "n_queries": len(a),
                "reference_mean": f"{statistics.mean(a):.4f}",
                "baseline_mean": f"{statistics.mean(b):.4f}",
                "delta": f"{boot['delta']:.4f}",
                "ci_low": f"{boot['ci_low']:.4f}",
                "ci_high": f"{boot['ci_high']:.4f}",
                "p_raw": f"{boot['p_value']:.6f}",
                "p_holm": f"{holm[m]:.6f}",
                "cliffs_delta": f"{cliffs_delta(a, b):.4f}",
            })
            print(f"[{task} stats] {leg} {metric}: {reference} vs {m}: "
                  f"delta={rows[-1]['delta']} CI=[{rows[-1]['ci_low']},{rows[-1]['ci_high']}] "
                  f"p_holm={rows[-1]['p_holm']} cliffs={rows[-1]['cliffs_delta']}", flush=True)
    return rows


def summarize_rows(task: str, rows: list[dict], metric_fields: list[str],
                   with_macro_f1: bool = False) -> list[dict]:
    """Per (leg, method, metric): per-seed means, then mean +/- sd across seeds."""
    per_seed_vals: dict = defaultdict(list)  # (leg, method, metric, seed) -> [scores]
    labels: dict = defaultdict(lambda: ([], []))  # (leg, method, seed) -> (y_true, y_pred)
    for r in rows:
        seed = r.get("seed", "")
        for mf in metric_fields:
            value = r.get(mf, "")
            if value == "" or value is None:
                continue
            per_seed_vals[(r["leg"], r["method"], mf, seed)].append(float(value))
        if with_macro_f1 and r.get("true_label") and r.get("pred_label"):
            y_true, y_pred = labels[(r["leg"], r["method"], seed)]
            y_true.append(r["true_label"])
            y_pred.append(r["pred_label"])

    seed_means: dict = defaultdict(dict)  # (leg, method, metric) -> {seed: mean}
    for (leg, method, mf, seed), vals in per_seed_vals.items():
        seed_means[(leg, method, mf)][seed] = statistics.mean(vals)
    if with_macro_f1 and labels:
        from sklearn.metrics import f1_score
        for (leg, method, seed), (y_true, y_pred) in labels.items():
            seed_means[(leg, method, "macro_f1")][seed] = f1_score(
                y_true, y_pred, average="macro")

    out = []
    for (leg, method, mf) in sorted(seed_means):
        by_seed = seed_means[(leg, method, mf)]
        means = [by_seed[s] for s in sorted(by_seed, key=str)]
        sd = statistics.stdev(means) if len(means) > 1 else 0.0
        out.append({
            "task": task, "leg": leg, "method": method, "metric": mf,
            "mean": f"{statistics.mean(means):.4f}", "sd": f"{sd:.4f}",
            "n_seeds": len(means),
            "per_seed": ";".join(f"{s}:{by_seed[s]:.4f}" for s in sorted(by_seed, key=str)),
        })
    return out


def get_dense_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# T1: cross-system transfer
# ---------------------------------------------------------------------------


def run_t1(smoke: bool, force: bool) -> bool:
    rows_path = out_path("t1_rows", smoke)
    if not single_shot_guard("t1", rows_path, force, smoke):
        return False
    from sma.eval import transfer_eval

    if smoke:
        seeds, pairs = (SMOKE_SEED,), "HDFS->OpenStack"
        index_size, query_size = 60, 20
        print("[t1 smoke] HDFS->OpenStack only (Spirit/Thunderbird 20M-line "
              "scans excluded from plumbing verification)", flush=True)
    else:
        seeds, pairs = TEST_SEEDS, T1_PAIRS
        index_size, query_size = T1_INDEX_SIZE, T1_QUERY_SIZE

    raw_metrics_path = out_path("t1_transfer_metrics", smoke)
    if raw_metrics_path.exists():
        raw_metrics_path.unlink()  # append-mode file; never mix executions

    per_query: list[dict] = []
    for seed in seeds:
        print(f"\n[t1] seed {seed}: pairs={pairs} index={index_size} "
              f"query={query_size} scorer=surprisal (frozen)", flush=True)
        random.seed(seed)
        before = len(per_query)
        transfer_eval.run_named_pairs(
            pairs, "surprisal", seed, index_size, query_size,
            out_path=raw_metrics_path, per_query_rows=per_query,
        )
        for r in per_query[before:]:
            r["seed"] = seed

    for r in per_query:
        r["leg"] = r["split"].split("[seed")[0]

    fields = ["seed", "leg", "split", "method", "query_id",
              "true_label", "pred_label", "hit@1", "hit@5", "hit@10"]
    write_csv(rows_path, per_query, fields)
    write_csv(out_path("t1_summary", smoke),
              summarize_rows("t1", per_query, ["hit@1", "hit@5", "hit@10"],
                             with_macro_f1=True),
              SUMMARY_FIELDS)
    stats = compute_stats("t1", per_query_scores(per_query, "hit@1"), "label_hit@1")
    write_csv(out_path("t1_stats", smoke), stats, STAT_FIELDS)
    return True


# ---------------------------------------------------------------------------
# T2: within-system (HDFS family-hit@5 common/rare + BGL triage)
# ---------------------------------------------------------------------------


def hdfs_family_leg(sampled, families, seed, dense_model) -> list[dict]:
    """family-hit@5 with common/rare strata; pattern of calibrate.family_scores
    extended to BM25 and Dense RAG baselines, with per-query rows."""
    from rank_bm25 import BM25Okapi
    from sentence_transformers import util

    from sma.encoders import get_encoder
    from sma.index.macfac import MacFacIndex
    from sma.match.types import MatchConfig

    data = list(sampled)
    random.Random(101).shuffle(data)  # repo-wide 80/20 split protocol
    cut = int(len(data) * 0.8)
    index_data, query_data = data[:cut], data[cut:]

    encoder = get_encoder("logs")
    print(f"[t2 seed {seed}] encoding {len(index_data)} HDFS index cases...", flush=True)
    index_cases, index_docs = [], []
    for sid, text, _label in index_data:
        case = encoder.encode(text, session_id=sid).case
        index_cases.append(case)
        index_docs.append((case.case_id, text))
    fam_of = {c.case_id: families[sid] for c, (sid, _, _) in zip(index_cases, index_data)}
    fam_counts = Counter(fam_of.values())
    doc_ids = [cid for cid, _ in index_docs]

    print(f"[t2 seed {seed}] building SMA (frozen MatchConfig), BM25, Dense indexes...",
          flush=True)
    sma_index = MacFacIndex(config=MatchConfig())  # frozen score-v2-final dials
    sma_index.build(index_cases)
    bm25 = BM25Okapi([t.lower().split() for _, t in index_docs])
    embeddings = dense_model.encode([t for _, t in index_docs],
                                    convert_to_tensor=True, show_progress_bar=False)

    def top5_sma(q_case, q_text):
        res = sma_index.retrieve(q_case, k=5, shortlist=40, fac_budget=20)
        return [r.case_id for r in res]

    def top5_bm25(q_case, q_text):
        scores = bm25.get_scores(q_text.lower().split())
        ranked = sorted(zip(doc_ids, scores), key=lambda row: (-row[1], row[0]))
        return [cid for cid, _ in ranked[:5]]

    def top5_dense(q_case, q_text):
        q_emb = dense_model.encode(q_text, convert_to_tensor=True, show_progress_bar=False)
        scores = util.cos_sim(q_emb, embeddings)[0].cpu().tolist()
        ranked = sorted(zip(doc_ids, scores), key=lambda row: (-row[1], row[0]))
        return [cid for cid, _ in ranked[:5]]

    methods = {"SMA": top5_sma, "BM25": top5_bm25, "Dense RAG": top5_dense}
    rows = []
    n_scored = 0
    for sid, text, _label in query_data:
        fam = families[sid]
        if fam == "normal":
            continue
        available = fam_counts.get(fam, 0)
        denom = min(5, available)
        if not denom:
            continue
        q_case = encoder.encode(text, session_id=sid).case
        stratum = "rare" if available <= RARE_FAMILY_MAX else "common"
        for method, top5 in methods.items():
            retrieved = top5(q_case, text)
            hit = sum(1 for cid in retrieved if fam_of.get(cid) == fam) / denom
            rows.append({
                "seed": seed, "leg": f"HDFS_family_{stratum}", "stratum": stratum,
                "method": method, "query_id": sid, "family": fam,
                "true_label": "", "pred_label": "",
                "hit@1": "", "hit@5": hit, "hit@10": "",
            })
        n_scored += 1
        if n_scored % 20 == 0:
            print(f"[t2 seed {seed}] {n_scored} anomalous HDFS queries scored...", flush=True)
    print(f"[t2 seed {seed}] HDFS family leg done: {n_scored} queries scored", flush=True)
    return rows


def run_t2(smoke: bool, force: bool) -> bool:
    rows_path = out_path("t2_rows", smoke)
    if not single_shot_guard("t2", rows_path, force, smoke):
        return False
    from sma.eval import transfer_eval
    from sma.eval.family_labels import hdfs_family
    from sma.eval.loghub_eval import sample_bgl_stratified, sample_hdfs_stratified

    seeds = (SMOKE_SEED,) if smoke else TEST_SEEDS
    sample_size = 120 if smoke else T2_SAMPLE_SIZE
    dense_model = get_dense_model()

    all_rows: list[dict] = []
    for seed in seeds:
        print(f"\n[t2] seed {seed}: sampling HDFS ({sample_size} stratified)...", flush=True)
        sampled = sample_hdfs_stratified(HDFS_ZIP, sample_size=sample_size, seed=seed)
        families = {sid: hdfs_family(text, label) for sid, text, label in sampled}
        all_rows.extend(hdfs_family_leg(sampled, families, seed, dense_model))

        print(f"[t2] seed {seed}: sampling BGL ({sample_size} stratified) for triage...",
              flush=True)
        bgl = sample_bgl_stratified(BGL_ZIP, sample_size=sample_size, seed=seed)
        data = list(bgl)
        random.Random(101).shuffle(data)  # repo-wide 80/20 split protocol
        cut = int(len(data) * 0.8)
        triage_rows: list[dict] = []
        transfer_eval.run_transfer(
            data[:cut], data[cut:], f"BGL_triage[seed{seed}]",
            scorer="surprisal", per_query_rows=triage_rows,
        )
        for r in triage_rows:
            r.update({"seed": seed, "leg": "BGL_triage", "stratum": "", "family": ""})
        all_rows.extend(triage_rows)

    fields = ["seed", "leg", "stratum", "method", "query_id", "family",
              "true_label", "pred_label", "hit@1", "hit@5", "hit@10"]
    write_csv(rows_path, [{k: r.get(k, "") for k in fields} for r in all_rows], fields)
    write_csv(out_path("t2_summary", smoke),
              summarize_rows("t2", all_rows, ["hit@1", "hit@5", "hit@10"],
                             with_macro_f1=True),
              SUMMARY_FIELDS)

    stats = compute_stats("t2", per_query_scores(
        [r for r in all_rows if r["leg"].startswith("HDFS_family")], "hit@5"),
        "family_hit@5")
    stats += compute_stats("t2", per_query_scores(
        [r for r in all_rows if r["leg"] == "BGL_triage"], "hit@1"),
        "label_hit@1")
    write_csv(out_path("t2_stats", smoke), stats, STAT_FIELDS)
    return True


# ---------------------------------------------------------------------------
# T3: BugsInPy leave-one-project-out category@1
# ---------------------------------------------------------------------------


def run_t3(smoke: bool, force: bool) -> bool:
    rows_path = out_path("t3_rows", smoke)
    if not single_shot_guard("t3", rows_path, force, smoke):
        return False
    if not BUGSINPY_ROOT.exists():
        print(f"[t3] missing dataset at {BUGSINPY_ROOT}; clone soarsmu/BugsInPy first.",
              flush=True)
        return False
    import bugsinpy_eval as be  # scripts/ is on sys.path

    from sma.index.macfac import MacFacIndex
    from sma.match.types import MatchConfig

    print(f"[t3] loading bugs from {BUGSINPY_ROOT} ...", flush=True)
    entries = be.prepare(BUGSINPY_ROOT, 40 if smoke else None)
    inventory = Counter(e["category"] for e in entries)
    other_share = inventory.get("other", 0) / len(entries)
    excluded = {"other"} if other_share > be.OTHER_EXCLUSION_THRESHOLD else set()
    print(f"[t3] {len(entries)} bugs; 'other' share {other_share:.1%} "
          f"({'excluded from' if excluded else 'kept in'} metric denominators)",
          flush=True)

    dense_model = get_dense_model()
    print("[t3] pre-encoding all patch texts...", flush=True)
    cache = be.EmbeddingCache(dense_model, entries)

    def build_sma_frozen(index_entries):
        # bugsinpy_eval.build_sma predates the freeze (scorer="ses"); the
        # confirmatory matcher uses the frozen MatchConfig defaults.
        idx = MacFacIndex(config=MatchConfig())
        idx.build([e["case"] for e in index_entries])

        def retrieve(query):
            results = idx.retrieve(query["case"], k=5, shortlist=40, fac_budget=20)
            return [r.case_id for r in results]

        return retrieve

    def norm(rid):
        return rid.split(":", 1)[1] if rid.startswith("bugsinpy:") else rid

    projects = sorted({e["project"] for e in entries})
    rows: list[dict] = []
    for fold_i, project in enumerate(projects, start=1):
        query_entries = [e for e in entries if e["project"] == project]
        index_entries = [e for e in entries if e["project"] != project]
        if not query_entries or not index_entries:
            continue
        print(f"[t3] fold {fold_i}/{len(projects)} {project}: "
              f"{len(query_entries)} queries vs {len(index_entries)} index bugs",
              flush=True)
        retrievers = {
            "SMA": build_sma_frozen(index_entries),
            "BM25": be.build_bm25(index_entries),
            "Dense RAG": be.build_dense(index_entries, dense_model, cache),
        }
        index_cats = {e["key"]: e["category"] for e in index_entries}
        cat_counts = Counter(index_cats.values())
        for query in query_entries:
            q_cat = query["category"]
            available = cat_counts.get(q_cat, 0)
            scored = available > 0 and q_cat not in excluded
            for method, retrieve in retrievers.items():
                ranked = [norm(r) for r in retrieve(query)]
                row = {
                    "seed": "", "leg": "bugsinpy_lopo", "fold": project,
                    "method": method, "query_id": query["key"], "category": q_cat,
                    "available": available, "excluded": q_cat in excluded,
                    "hit@1": "", "hit@5": "",
                }
                if scored:
                    for k in (1, 5):
                        hits = sum(1 for key in ranked[:k] if index_cats.get(key) == q_cat)
                        row[f"hit@{k}"] = hits / min(k, available)
                rows.append(row)

    fields = ["leg", "fold", "method", "query_id", "category", "available",
              "excluded", "hit@1", "hit@5"]
    write_csv(rows_path, [{k: r.get(k, "") for k in fields} for r in rows], fields)
    write_csv(out_path("t3_summary", smoke),
              summarize_rows("t3", rows, ["hit@1", "hit@5"]),
              SUMMARY_FIELDS)
    stats = compute_stats("t3", per_query_scores(rows, "hit@1"), "category_hit@1")
    stats += compute_stats("t3", per_query_scores(rows, "hit@5"), "category_hit@5")
    write_csv(out_path("t3_stats", smoke), stats, STAT_FIELDS)
    return True


# ---------------------------------------------------------------------------
# T4: Liberty haystack with out-of-corpus needle probes
# ---------------------------------------------------------------------------


def sample_liberty_needles(seeds, per_seed, line_start, line_cap, exclude_ids):
    """Out-of-corpus needle (alert-window) probes from liberty2.gz.

    Slicing logic mirrors scripts/prepare_ui_corpus.py sample_cfdr_haystack
    (per-node 60-second windows, >=3 lines, alert-tag column stripped), but
    scans the [line_start, line_cap) slice ONCE and draws each seed's probe
    set from the same eligible pool, so the 5-seed battery pays one two-pass
    scan instead of ten.
    """
    import gzip

    counts: dict[str, int] = defaultdict(int)
    is_alert: dict[str, bool] = defaultdict(bool)
    t0 = time.perf_counter()
    print(f"[t4] pass 1: scanning liberty2.gz lines [{line_start:,}, {line_cap:,})...",
          flush=True)
    with gzip.open(LIBERTY_GZ, "rt", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if i < line_start:
                continue
            if i >= line_cap:
                break
            parts = line.split(maxsplit=4)
            if len(parts) < 5:
                continue
            try:
                window = int(parts[1]) // 60
            except ValueError:
                continue
            key = f"liberty_{parts[3]}_{window}"
            counts[key] += 1
            if parts[0] != "-":
                is_alert[key] = True
    needles = sorted(
        k for k, n in counts.items()
        if n >= 3 and is_alert[k] and k not in exclude_ids
    )
    print(f"[t4] pass 1 done in {time.perf_counter() - t0:.0f}s: "
          f"{len(needles)} eligible out-of-corpus needle sessions", flush=True)
    if not needles:
        return {seed: [] for seed in seeds}

    chosen = {
        seed: sorted(random.Random(seed).sample(needles, min(per_seed, len(needles))))
        for seed in seeds
    }
    union = set().union(*chosen.values())

    texts: dict[str, list[str]] = defaultdict(list)
    t0 = time.perf_counter()
    print(f"[t4] pass 2: extracting {len(union)} needle sessions...", flush=True)
    with gzip.open(LIBERTY_GZ, "rt", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if i < line_start:
                continue
            if i >= line_cap:
                break
            parts = line.split(maxsplit=4)
            if len(parts) < 5:
                continue
            try:
                window = int(parts[1]) // 60
            except ValueError:
                continue
            key = f"liberty_{parts[3]}_{window}"
            if key in union and len(texts[key]) < T4_SESSION_LINE_CAP:
                # Strip the alert-tag column (ground truth, not log content).
                texts[key].append(line.partition(" ")[2].rstrip())
    print(f"[t4] pass 2 done in {time.perf_counter() - t0:.0f}s", flush=True)
    return {
        seed: [(k, "\n".join(texts[k])) for k in chosen[seed] if texts.get(k)]
        for seed in seeds
    }


def run_t4(smoke: bool, force: bool) -> bool:
    rows_path = out_path("t4_rows", smoke)
    if not single_shot_guard("t4", rows_path, force, smoke):
        return False
    from rank_bm25 import BM25Okapi
    from sentence_transformers import util

    from sma.encoders import get_encoder
    from sma.eval.baselines.hybrid_rrf import rrf_fuse
    from sma.index.macfac import MacFacIndex
    from sma.match.types import MatchConfig

    corpus = [json.loads(line) for line in T4_CORPUS.open(encoding="utf-8")]
    if smoke:
        seeds, per_seed = (SMOKE_SEED,), 5
        corpus = corpus[:400]
        # Plumbing-only slice: alerts only get dense enough late in the file;
        # the smoke window overlaps the corpus region, which is fine because
        # probe ids are still excluded against the corpus id set.
        line_start, line_cap = 40_000_000, 43_000_000
    else:
        seeds, per_seed = TEST_SEEDS, T4_PROBES_PER_SEED
        line_start, line_cap = T4_PROBE_LINE_START, T4_PROBE_LINE_CAP
    print(f"[t4] corpus: {len(corpus)} sessions "
          f"({sum(r['label'] == 'Anomaly' for r in corpus)} needles)", flush=True)

    probes_by_seed = sample_liberty_needles(
        seeds, per_seed, line_start, line_cap,
        exclude_ids={r["id"] for r in corpus},
    )
    if not any(probes_by_seed.values()):
        print("[t4] ERROR: no out-of-corpus needle probes found in the slice; "
              "nothing to evaluate.", flush=True)
        return False

    encoder = get_encoder("logs")
    print(f"[t4] encoding {len(corpus)} corpus sessions (Tier-0)...", flush=True)
    corpus_cases, label_of, doc_ids, doc_texts = [], {}, [], []
    for i, r in enumerate(corpus, start=1):
        case = encoder.encode(r["text"], session_id=r["id"]).case
        corpus_cases.append(case)
        label_of[case.case_id] = r["label"]
        doc_ids.append(case.case_id)
        doc_texts.append(r["text"])
        if i % 500 == 0:
            print(f"[t4] encoded {i}/{len(corpus)}...", flush=True)

    print("[t4] building SMA index (frozen MatchConfig)...", flush=True)
    sma_index = MacFacIndex(config=MatchConfig())
    sma_index.build(corpus_cases)
    print("[t4] building BM25 index...", flush=True)
    bm25 = BM25Okapi([t.lower().split() for t in doc_texts])
    print("[t4] building Dense RAG index (all-MiniLM-L6-v2)...", flush=True)
    dense_model = get_dense_model()
    embeddings = dense_model.encode(doc_texts, convert_to_tensor=True,
                                    show_progress_bar=False)

    def rank_sma(q_case):
        res = sma_index.retrieve(q_case, k=T4_RRF_DEPTH, shortlist=200, fac_budget=30)
        return [(r.case_id, r.ses_n) for r in res]

    def rank_bm25(q_text):
        scores = bm25.get_scores(q_text.lower().split())
        ranked = sorted(zip(doc_ids, scores), key=lambda row: (-row[1], row[0]))
        return ranked[:T4_RRF_DEPTH]

    def rank_dense(q_text):
        q_emb = dense_model.encode(q_text, convert_to_tensor=True, show_progress_bar=False)
        scores = util.cos_sim(q_emb, embeddings)[0].cpu().tolist()
        ranked = sorted(zip(doc_ids, scores), key=lambda row: (-row[1], row[0]))
        return ranked[:T4_RRF_DEPTH]

    rows: list[dict] = []
    for seed in seeds:
        probes = probes_by_seed[seed]
        print(f"\n[t4] seed {seed}: {len(probes)} out-of-corpus needle probes", flush=True)
        for p_idx, (probe_id, text) in enumerate(probes, start=1):
            q_case = encoder.encode(text, session_id=probe_id).case
            ranked_by_method = {
                "BM25": rank_bm25(text),
                "Dense RAG": rank_dense(text),
                "SMA": rank_sma(q_case),
            }
            # Registered primary haystack posture: hybrid fused (RRF of
            # bm25+dense+sma); solo modes recorded alongside.
            ranked_by_method["Hybrid-RRF"] = rrf_fuse(
                [ranked_by_method["BM25"], ranked_by_method["Dense RAG"],
                 ranked_by_method["SMA"]],
                top_k=10,
            )
            for method, ranked in ranked_by_method.items():
                top5 = [cid for cid, _ in ranked[:5]]
                denom = min(5, len(ranked)) or 1
                hit5 = sum(1 for cid in top5 if label_of.get(cid) == "Anomaly") / denom
                hit1 = 1.0 if top5 and label_of.get(top5[0]) == "Anomaly" else 0.0
                rows.append({
                    "seed": seed, "leg": "liberty_haystack", "method": method,
                    "query_id": probe_id, "needle_hit@5": hit5, "hit@1": hit1,
                })
            if p_idx % 10 == 0 or p_idx == len(probes):
                print(f"[t4] seed {seed}: {p_idx}/{len(probes)} probes done", flush=True)

    fields = ["seed", "leg", "method", "query_id", "needle_hit@5", "hit@1"]
    write_csv(rows_path, rows, fields)
    write_csv(out_path("t4_summary", smoke),
              summarize_rows("t4", rows, ["needle_hit@5", "hit@1"]),
              SUMMARY_FIELDS)
    # Hybrid fused is the registered primary; the solo-SMA comparison family
    # is also reported (prereg section 1 registered caveat).
    per_q = per_query_scores(rows, "needle_hit@5")
    stats = compute_stats("t4", per_q, "needle_hit@5", reference="Hybrid-RRF")
    stats += compute_stats("t4", per_q, "needle_hit@5", reference=SMA_METHOD)
    write_csv(out_path("t4_stats", smoke), stats, STAT_FIELDS)
    return True


# ---------------------------------------------------------------------------
# SSB: forced-choice + library r1/MRR
# ---------------------------------------------------------------------------


def run_ssb(smoke: bool, force: bool) -> bool:
    rows_path = out_path("ssb_rows", smoke)
    if not single_shot_guard("ssb", rows_path, force, smoke):
        return False
    from sma.eval import ssb_eval

    seeds = (SSB_SMOKE_SEED,) if smoke else SSB_TEST_SEEDS
    n = 12 if smoke else SSB_N

    rows: list[dict] = []
    for seed in seeds:
        print(f"\n[ssb] seed {seed}: forced-choice (n={n})...", flush=True)
        fc = ssb_eval.evaluate_forced_choice(n=n, seed=seed)
        for i, rank in enumerate(fc.ranks):
            rows.append({
                "seed": seed, "leg": "forced_choice", "method": "SMA",
                "query_id": f"fc_{seed}_{i}", "rank": rank,
                "r1": float(rank == 1), "rr": (1.0 / rank) if rank else 0.0,
            })
        print(f"[ssb] seed {seed}: forced-choice r1={fc.metrics['r1']} "
              f"mrr={fc.metrics['mrr']}", flush=True)

        print(f"[ssb] seed {seed}: library (n={n}, 2n cases)...", flush=True)
        lib = ssb_eval.evaluate_library(n=n, seed=seed, k=10)
        for method, ranks in lib["ranks"].items():
            for qid, rank in zip(lib["query_ids"], ranks):
                rows.append({
                    "seed": seed, "leg": "library", "method": method,
                    "query_id": qid, "rank": rank,
                    "r1": float(rank == 1), "rr": (1.0 / rank) if rank else 0.0,
                })
        for metric in lib["metrics"]:
            print(f"[ssb] seed {seed}: {metric['split']}: r1={metric['r1']} "
                  f"mrr={metric['mrr']}", flush=True)

    fields = ["seed", "leg", "method", "query_id", "rank", "r1", "rr"]
    write_csv(rows_path, rows, fields)
    write_csv(out_path("ssb_summary", smoke),
              summarize_rows("ssb", rows, ["r1", "rr"]),
              SUMMARY_FIELDS)
    library_rows = [r for r in rows if r["leg"] == "library"]
    stats = compute_stats("ssb", per_query_scores(library_rows, "r1"), "r1")
    stats += compute_stats("ssb", per_query_scores(library_rows, "rr"), "mrr")
    write_csv(out_path("ssb_stats", smoke), stats, STAT_FIELDS)
    return True


# ---------------------------------------------------------------------------


RUNNERS = {"t1": run_t1, "t2": run_t2, "t3": run_t3, "t4": run_t4, "ssb": run_ssb}
TASK_ORDER = ["t1", "t2", "t3", "t4", "ssb"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=[*TASK_ORDER, "all"])
    parser.add_argument("--smoke", action="store_true",
                        help="tiny plumbing run on validation seed 7 (SSB seed 29); "
                             "writes *_smoke.csv, never the registered outputs")
    parser.add_argument("--force-rerun-logged", action="store_true",
                        help="override the single-shot guard AFTER logging the "
                             "crash rerun in docs/STATUS.md")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "SMOKE (validation seeds)" if args.smoke else "CONFIRMATORY (registered test seeds)"
    print(f"=== Confirmatory battery: task={args.task} mode={mode} ===", flush=True)
    if not args.smoke:
        print(f"Test seeds {TEST_SEEDS}, SSB seeds {SSB_TEST_SEEDS} n={SSB_N} "
              f"(prereg section 4, frozen at prereg-v1)", flush=True)

    tasks = TASK_ORDER if args.task == "all" else [args.task]
    blocked = []
    for task in tasks:
        t0 = time.perf_counter()
        ran = RUNNERS[task](args.smoke, args.force_rerun_logged)
        if ran:
            print(f"=== {task} complete in {time.perf_counter() - t0:.0f}s ===", flush=True)
        else:
            blocked.append(task)

    if blocked:
        print(f"\nTasks not run: {', '.join(blocked)}", flush=True)
        # --task all skips completed tasks by design (resume after a crash);
        # an explicitly named blocked task is an error.
        return 1 if args.task != "all" else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
