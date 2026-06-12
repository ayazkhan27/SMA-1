"""T3 bug-fix memory evaluation on BugsInPy (fix-category-hit@k).

Case = deterministic Tier-0 structure of a bug's fix context (unified-diff
structure + failing-test names, ``sma.eval.bugsinpy``). Ground truth =
ordered-regex fix-pattern categories over the patch
(``sma.eval.bugsinpy_families``). Question: given a NEW bug, does retrieval
surface a PAST bug fixed by the same kind of change (the fix), not just a
textually similar file?

Methods:
  SMA-ses  MacFacIndex over bug cases (scorer="ses")
  BM25     rank_bm25.BM25Okapi over raw patch text
  Dense    all-MiniLM-L6-v2 over raw patch text

Split modes:
  stratified  80/20 over bugs, stratified by fix category, seed 42
  lopo        leave-one-project-out (17 folds): queries = held-out project,
              index = the other 16 projects -- cross-project transfer

Metric (mirrors family-hit@k from scripts/family_eval.py):
  category_hit@k = |top-k retrieved with same category| /
                   min(k, number of same-category index bugs)
Queries with zero same-category index bugs are skipped (counted). If the
"other" category exceeds 40% of bugs it is excluded from metric
denominators (reported either way; the with-other numbers are printed).

Outputs:
  reports/bugsinpy_metrics.csv    split_mode, method, category_hit@1,
                                  category_hit@5, n_queries, p50_ms
  reports/bugsinpy_inventory.csv  category, n_bugs

Usage: python3 scripts/bugsinpy_eval.py [--smoke N] [--root PATH]
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import random
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sma.eval.bugsinpy import BugRecord, bug_case, load_bugs, parse_patch
from sma.eval.bugsinpy_families import categorize
from sma.index.macfac import MacFacIndex
from sma.match.types import MatchConfig

ROOT = pathlib.Path("data/raw/bugsinpy")
OTHER_EXCLUSION_THRESHOLD = 0.40
KS = (1, 5)


def prepare(root: pathlib.Path, smoke: int | None):
    bugs = load_bugs(root)
    if smoke:
        rng = random.Random(42)
        bugs = sorted(rng.sample(bugs, min(smoke, len(bugs))), key=lambda b: b.key)
    entries = []
    for bug in bugs:
        facts = parse_patch(bug.patch_text)
        entries.append(
            {
                "key": bug.key,
                "project": bug.project,
                "category": categorize(facts),
                "case": bug_case(bug, facts),
                "text": bug.patch_text,
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Retrieval methods. Each builder takes the index entries and returns a
# retrieve(query_entry) -> ranked list of index keys (len <= max k).
# ---------------------------------------------------------------------------


def build_sma(index_entries):
    idx = MacFacIndex(config=MatchConfig(scorer="ses"))
    idx.build([e["case"] for e in index_entries])

    def retrieve(query):
        results = idx.retrieve(query["case"], k=5, shortlist=40, fac_budget=20)
        return [r.case_id for r in results]

    return retrieve


def build_bm25(index_entries):
    from rank_bm25 import BM25Okapi

    keys = [e["key"] for e in index_entries]
    bm25 = BM25Okapi([e["text"].lower().split() for e in index_entries])

    def retrieve(query):
        scores = bm25.get_scores(query["text"].lower().split())
        ranked = sorted(zip(keys, scores), key=lambda row: (-row[1], row[0]))
        return [k for k, _ in ranked[:5]]

    return retrieve


def build_dense(index_entries, dense_model, embedding_cache):
    from sentence_transformers import util

    keys = [e["key"] for e in index_entries]
    index_emb = embedding_cache.stack([e["key"] for e in index_entries])

    def retrieve(query):
        q_emb = dense_model.encode(
            query["text"], convert_to_tensor=True, show_progress_bar=False
        )
        scores = util.cos_sim(q_emb, index_emb)[0].cpu().tolist()
        ranked = sorted(zip(keys, scores), key=lambda row: (-row[1], row[0]))
        return [k for k, _ in ranked[:5]]

    return retrieve


class EmbeddingCache:
    """Encode every patch once; folds reuse the vectors (identical values)."""

    def __init__(self, model, entries):
        import torch

        texts = [e["text"] for e in entries]
        emb = model.encode(
            texts, convert_to_tensor=True, show_progress_bar=False, batch_size=16
        )
        self._by_key = {e["key"]: emb[i] for i, e in enumerate(entries)}
        self._torch = torch

    def stack(self, keys):
        return self._torch.stack([self._by_key[k] for k in keys])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_fold(retrievers, index_entries, query_entries, excluded, accum):
    """Run every method over one (index, queries) fold; accumulate hits and
    latencies into ``accum[method]``."""
    # SMA case_id == "bugsinpy:<key>"; normalize retrieved ids to bug keys.
    def norm(rid):
        return rid.split(":", 1)[1] if rid.startswith("bugsinpy:") else rid

    index_cats = {e["key"]: e["category"] for e in index_entries}
    cat_counts = Counter(index_cats.values())

    for q_idx, query in enumerate(query_entries, start=1):
        q_cat = query["category"]
        available = cat_counts.get(q_cat, 0)
        for method, retrieve in retrievers.items():
            t0 = time.perf_counter()
            ranked = [norm(r) for r in retrieve(query)]
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            acc = accum[method]
            acc["latency_ms"].append(elapsed_ms)
            if available == 0:
                acc["n_no_support"] += 1
                continue
            fracs = {}
            for k in KS:
                hits = sum(1 for key in ranked[:k] if index_cats.get(key) == q_cat)
                fracs[k] = hits / min(k, available)
                acc[f"all_hit@{k}"].append(fracs[k])
            if q_cat in excluded:
                acc["n_excluded"] += 1
                continue
            for k in KS:
                acc[f"hit@{k}"].append(fracs[k])
        if q_idx % 50 == 0 or q_idx == len(query_entries):
            print(f"    {q_idx}/{len(query_entries)} queries done")


def new_accumulator(methods):
    return {
        m: {
            "hit@1": [],
            "hit@5": [],
            "all_hit@1": [],
            "all_hit@5": [],
            "latency_ms": [],
            "n_excluded": 0,
            "n_no_support": 0,
        }
        for m in methods
    }


def summarize(split_mode, accum):
    rows = []
    for method, acc in accum.items():
        n = len(acc["hit@1"])
        h1 = sum(acc["hit@1"]) / n if n else 0.0
        h5 = sum(acc["hit@5"]) / n if n else 0.0
        n_all = len(acc["all_hit@1"])
        a1 = sum(acc["all_hit@1"]) / n_all if n_all else 0.0
        a5 = sum(acc["all_hit@5"]) / n_all if n_all else 0.0
        p50 = statistics.median(acc["latency_ms"]) if acc["latency_ms"] else 0.0
        rows.append(
            {
                "split_mode": split_mode,
                "method": method,
                "category_hit@1": f"{h1:.4f}",
                "category_hit@5": f"{h5:.4f}",
                "n_queries": n,
                "p50_ms": f"{p50:.2f}",
            }
        )
        print(
            f"  {method:<8} cat_hit@1={h1:.4f} cat_hit@5={h5:.4f} n_scored={n} | "
            f"with-other: @1={a1:.4f} @5={a5:.4f} n={n_all} | "
            f"excluded_other={acc['n_excluded']} "
            f"no_same_cat_support={acc['n_no_support']} p50={p50:.2f}ms"
        )
    return rows


# ---------------------------------------------------------------------------
# Split modes
# ---------------------------------------------------------------------------


def stratified_split(entries, seed=42, train_frac=0.8):
    by_cat: dict[str, list] = {}
    for e in entries:
        by_cat.setdefault(e["category"], []).append(e)
    rng = random.Random(seed)
    index_entries, query_entries = [], []
    for cat in sorted(by_cat):
        group = sorted(by_cat[cat], key=lambda e: e["key"])
        rng.shuffle(group)
        cut = int(len(group) * train_frac)
        if cut == len(group) and len(group) > 1:
            cut -= 1
        index_entries.extend(group[:cut])
        query_entries.extend(group[cut:])
    query_entries.sort(key=lambda e: e["key"])
    return index_entries, query_entries


def run_stratified(entries, methods_factory, excluded):
    index_entries, query_entries = stratified_split(entries)
    print(
        f"\n--- stratified split: {len(index_entries)} index bugs, "
        f"{len(query_entries)} query bugs ---"
    )
    retrievers = methods_factory(index_entries)
    accum = new_accumulator(retrievers)
    score_fold(retrievers, index_entries, query_entries, excluded, accum)
    return summarize("stratified", accum)


def run_lopo(entries, methods_factory, excluded):
    projects = sorted({e["project"] for e in entries})
    print(f"\n--- leave-one-project-out: {len(projects)} folds ---")
    accum = None
    for project in projects:
        query_entries = [e for e in entries if e["project"] == project]
        index_entries = [e for e in entries if e["project"] != project]
        print(
            f"  fold {project}: {len(query_entries)} queries vs "
            f"{len(index_entries)} index bugs"
        )
        retrievers = methods_factory(index_entries)
        if accum is None:
            accum = new_accumulator(retrievers)
        score_fold(retrievers, index_entries, query_entries, excluded, accum)
    return summarize("lopo", accum)


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument(
        "--smoke", type=int, default=None, help="run on a deterministic N-bug sample"
    )
    args = parser.parse_args()

    if not args.root.exists():
        print(f"Missing dataset at {args.root}; clone soarsmu/BugsInPy there first.")
        return

    print(f"Loading bugs from {args.root} ...")
    entries = prepare(args.root, args.smoke)
    projects = Counter(e["project"] for e in entries)
    print(
        f"{len(entries)} bugs with non-empty patches across {len(projects)} projects."
    )

    inventory = Counter(e["category"] for e in entries)
    print("\nFix-category inventory:")
    for cat, n in inventory.most_common():
        print(f"  {n:4d}  {n / len(entries):5.1%}  {cat}")

    other_share = inventory.get("other", 0) / len(entries)
    excluded = {"other"} if other_share > OTHER_EXCLUSION_THRESHOLD else set()
    print(
        f"\n'other' share = {other_share:.1%} "
        + (
            f"(> {OTHER_EXCLUSION_THRESHOLD:.0%}: excluded from metric denominators)"
            if excluded
            else f"(<= {OTHER_EXCLUSION_THRESHOLD:.0%}: kept in metrics)"
        )
    )

    print("\nLoading dense model once (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer

    dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Pre-encoding all patch texts...")
    embedding_cache = EmbeddingCache(dense_model, entries)

    def methods_factory(index_entries):
        return {
            "SMA-ses": build_sma(index_entries),
            "BM25": build_bm25(index_entries),
            "Dense": build_dense(index_entries, dense_model, embedding_cache),
        }

    rows = []
    rows.extend(run_stratified(entries, methods_factory, excluded))
    rows.extend(run_lopo(entries, methods_factory, excluded))

    reports = pathlib.Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""

    inv_path = reports / f"bugsinpy_inventory{suffix}.csv"
    with inv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["category", "n_bugs"])
        writer.writeheader()
        for cat, n in inventory.most_common():
            writer.writerow({"category": cat, "n_bugs": n})
    print(f"\nSaved {inv_path}")

    met_path = reports / f"bugsinpy_metrics{suffix}.csv"
    with met_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "split_mode",
                "method",
                "category_hit@1",
                "category_hit@5",
                "n_queries",
                "p50_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {met_path}")

    print("\n=== BugsInPy fix-category retrieval ===")
    for row in rows:
        print(
            f"{row['split_mode']:<11} {row['method']:<8} "
            f"cat@1={row['category_hit@1']} cat@5={row['category_hit@5']} "
            f"n={row['n_queries']} p50={row['p50_ms']}ms"
        )


if __name__ == "__main__":
    main()
