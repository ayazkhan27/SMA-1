#!/usr/bin/env python3
"""Reviewer-grade baseline ladder for the LogHub within-system triage protocol.

Runs six stronger baselines under EXACTLY the protocol of
sma.eval.loghub_eval.run_evaluation (1000 stratified sessions seed 42, 80/20
split via random.Random(101).shuffle, weighted top-5 label vote, macro-F1 +
label_hit_rate@{1,5,10} + p50/p95 per-query latency, diagnostic alerts), so
rows are directly comparable to reports/triage_metrics.csv:

  1. Hybrid-RRF              reciprocal-rank fusion (k=60) of BM25Okapi + BGE dense
  2. BGE-dense               BAAI/bge-base-en-v1.5 (the blueprint's B2 embedder)
  3. SPLADE                  naver/splade-cocondenser-ensembledistil learned sparse
  4. Hybrid+Rerank           cross-encoder/ms-marco-MiniLM-L-6-v2 over Hybrid-RRF top-20
  5. WL-kernel               2-iteration WL graph kernel on SMA's own Tier-0 cases
  6. B6-LongContext-DeepSeek top-20 BM25 precedents stuffed into one deepseek-chat
                             prompt (HDFS only; temperature 0; one retry per row)

Per-query latency includes the full per-query work of each method (query
encoding, fusion, reranking, API call), matching the loghub_eval convention.
Index/build costs are excluded there too.

Usage:
  python3 -u scripts/baseline_ladder.py                  # full run, writes reports/baseline_ladder_metrics.csv
  python3 -u scripts/baseline_ladder.py --smoke          # 60 index docs / 20 queries, B6 capped at 3 calls
  python3 -u scripts/baseline_ladder.py --datasets HDFS  # subset
  python3 -u scripts/baseline_ladder.py --skip-b6        # no API leg
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import random
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sklearn.metrics import f1_score  # noqa: E402

from sma.encoders import get_encoder  # noqa: E402
from sma.eval.loghub_eval import sample_bgl_stratified, sample_hdfs_stratified  # noqa: E402
from sma.eval.baselines.bge_dense import BGEDenseRetriever  # noqa: E402
from sma.eval.baselines.hybrid_rrf import rrf_fuse  # noqa: E402
from sma.eval.baselines.longcontext_llm import LongContextDeepSeek  # noqa: E402
from sma.eval.baselines.rerank import CrossEncoderReranker  # noqa: E402
from sma.eval.baselines.splade import SpladeRetriever  # noqa: E402
from sma.eval.baselines.wl_kernel import WLKernelRetriever  # noqa: E402

OUTPUT_CSV = REPO_ROOT / "reports" / "baseline_ladder_metrics.csv"
FIELDNAMES = [
    "dataset",
    "split",
    "method",
    "macro_f1",
    "label_hit_rate@1",
    "label_hit_rate@5",
    "label_hit_rate@10",
    "p50_ms",
    "p95_ms",
]

RRF_DEPTH = 100  # fusion depth per leg; final cut is top-10 / top-20
RERANK_POOL = 20
B6_PRECEDENTS = 20


def check_manifest(dataset_name, sampled_data):
    """Verify our sample matches the ids the original triage run recorded.

    reports/loghub_sample_manifest.csv was written by sma.eval.loghub_eval at
    the time triage_metrics.csv was produced; identical id sets prove the
    ladder rows are computed on the very same sessions.
    """
    manifest = REPO_ROOT / "reports" / "loghub_sample_manifest.csv"
    if not manifest.exists():
        print(f"[manifest] {manifest} missing; skipping sample-identity check")
        return
    recorded = set()
    with manifest.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["dataset"] == dataset_name:
                recorded.add((row["session_id"], row["label"]))
    ours = {(sid, label) for sid, _, label in sampled_data}
    if ours == recorded:
        print(f"[manifest] {dataset_name}: sample matches the original triage run "
              f"({len(ours)} sessions)")
    else:
        print(f"[manifest] WARNING {dataset_name}: sample differs from original run "
              f"(ours={len(ours)}, recorded={len(recorded)}, "
              f"overlap={len(ours & recorded)}) -- rows may not be strictly comparable")


def split_protocol(sampled_data):
    """EXACT split convention of loghub_eval.run_evaluation."""
    sampled_data = list(sampled_data)
    random.Random(101).shuffle(sampled_data)
    split_idx = int(len(sampled_data) * 0.8)
    return sampled_data[:split_idx], sampled_data[split_idx:]


def weighted_vote(ranked, index_labels, top=5):
    """EXACT weighted top-5 vote of loghub_eval.run_evaluation."""
    voting = {"Anomaly": 0.0, "Normal": 0.0}
    for case_id, score in ranked[:top]:
        voting[index_labels[case_id]] += score
    return max(voting, key=voting.get) if sum(voting.values()) > 0 else "Normal"


def summarize_method(
    dataset_name, method, preds, recalls, latencies, true_labels, index_labels
):
    """Metric computation + diagnostic-alert convention from run_evaluation."""
    f1 = f1_score(true_labels, preds, average="macro")

    relevant_by_label = {
        label: {cid for cid, lab in index_labels.items() if lab == label}
        for label in ("Anomaly", "Normal")
    }
    r_lists = {1: [], 5: [], 10: []}
    for q_label, ret_ids in zip(true_labels, recalls):
        relevant_ids = relevant_by_label[q_label]
        for k in (1, 5, 10):
            hits = len(set(ret_ids[:k]).intersection(relevant_ids))
            denom = min(len(relevant_ids), k)
            r_lists[k].append(hits / denom if denom > 0 else 0.0)
    r1, r5, r10 = (sum(r_lists[k]) / len(r_lists[k]) for k in (1, 5, 10))

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)

    rows = [{
        "dataset": "LogHub",
        "split": f"{dataset_name}_MVP_diagnostic",
        "method": method,
        "macro_f1": f"{f1:.4f}",
        "label_hit_rate@1": f"{r1:.4f}",
        "label_hit_rate@5": f"{r5:.4f}",
        "label_hit_rate@10": f"{r10:.4f}",
        "p50_ms": f"{p50:.3f}",
        "p95_ms": f"{p95:.3f}",
    }]
    print(f"Method: {method}")
    print(f"  Macro-F1: {f1:.4f}")
    print(f"  label_hit_rate@1: {r1:.4f}, label_hit_rate@5: {r5:.4f}, label_hit_rate@10: {r10:.4f}")
    print(f"  p50 Latency: {p50:.3f} ms, p95 Latency: {p95:.3f} ms")

    unique_preds = set(preds)
    if f1 == 0.0 or f1 == 1.0 or len(unique_preds) <= 1:
        if f1 == 0.0:
            reason = "F1 is 0.0: Retrieval collapse or dataset imbalance"
        elif f1 == 1.0:
            reason = ("F1 is 1.0: Suspiciously perfect classification - potential "
                      "data leakage or indexing overlap")
        else:
            reason = f"Retrieval collapse: predicted only '{list(unique_preds)[0]}' sessions"
        rows.append({
            "dataset": "DIAGNOSTIC",
            "split": f"{dataset_name}_MVP_diagnostic",
            "method": f"{method}_alert",
            "macro_f1": reason,
            "label_hit_rate@1": "ALERT",
            "label_hit_rate@5": "ALERT",
            "label_hit_rate@10": "ALERT",
            "p50_ms": "0.000",
            "p95_ms": "0.000",
        })
        print(f"  [DIAGNOSTIC ALERT] {reason}")
    return rows


def evaluate_dataset(dataset_name, sampled_data, smoke=False):
    """Run the five retrieval baselines; return (rows, b6_context)."""
    print(f"\n--- Baseline ladder on {dataset_name} ({len(sampled_data)} cases) ---")
    index_data, query_data = split_protocol(sampled_data)
    if smoke:
        index_data, query_data = index_data[:60], query_data[:20]
        print(f"[smoke] index={len(index_data)} queries={len(query_data)}")

    log_encoder = get_encoder("logs")
    print("Encoding index cases (Tier-0)...")
    index_cases, index_docs, index_labels = [], [], {}
    for sid, text, label in index_data:
        case = log_encoder.encode(text, session_id=sid).case
        index_cases.append(case)
        index_docs.append((case.case_id, text))
        index_labels[case.case_id] = label
    print("Encoding query cases (Tier-0)...")
    query_cases, query_docs, query_labels = [], [], {}
    for sid, text, label in query_data:
        case = log_encoder.encode(text, session_id=sid).case
        query_cases.append(case)
        query_docs.append((case.case_id, text))
        query_labels[case.case_id] = label

    doc_ids = [doc_id for doc_id, _ in index_docs]
    doc_text = dict(index_docs)

    print("Building BM25 index...")
    from rank_bm25 import BM25Okapi

    bm25_index = BM25Okapi([text.lower().split() for _, text in index_docs])

    print("Building BGE dense index (BAAI/bge-base-en-v1.5)...")
    t0 = time.perf_counter()
    bge = BGEDenseRetriever()
    bge.build(index_docs)
    print(f"  BGE index built in {time.perf_counter() - t0:.1f}s")

    print("Building SPLADE index (naver/splade-cocondenser-ensembledistil)...")
    t0 = time.perf_counter()
    splade = SpladeRetriever()
    splade.build(index_docs)
    print(f"  SPLADE index built in {time.perf_counter() - t0:.1f}s")

    print("Loading cross-encoder reranker (ms-marco-MiniLM-L-6-v2)...")
    reranker = CrossEncoderReranker()

    print("Building WL-kernel index over Tier-0 cases...")
    t0 = time.perf_counter()
    wl = WLKernelRetriever()
    wl.build(index_cases)
    print(f"  WL index built in {time.perf_counter() - t0:.1f}s")

    def bm25_ranking(q_text, k):
        scores = bm25_index.get_scores(q_text.lower().split())
        ranked = sorted(zip(doc_ids, scores), key=lambda row: (-row[1], row[0]))
        return ranked[:k]

    def hybrid_ranking(q_text, k):
        return rrf_fuse(
            [bm25_ranking(q_text, RRF_DEPTH), bge.retrieve(q_text, k=RRF_DEPTH)],
            top_k=k,
        )

    def retrieve_hybrid(q_case, q_text):
        return hybrid_ranking(q_text, 10)

    def retrieve_bge(q_case, q_text):
        return bge.retrieve(q_text, k=10)

    def retrieve_splade(q_case, q_text):
        return splade.retrieve(q_text, k=10)

    def retrieve_rerank(q_case, q_text):
        pool = hybrid_ranking(q_text, RERANK_POOL)
        candidates = [(cid, doc_text[cid]) for cid, _ in pool]
        return reranker.rerank(q_text, candidates, top_k=10)

    def retrieve_wl(q_case, q_text):
        return wl.retrieve(q_case, k=10)

    retrievers = {
        "Hybrid-RRF": retrieve_hybrid,
        "BGE-dense": retrieve_bge,
        "SPLADE": retrieve_splade,
        "Hybrid+Rerank": retrieve_rerank,
        "WL-kernel": retrieve_wl,
    }
    metrics = {m: {"recalls": [], "preds": [], "latencies": []} for m in retrievers}

    print("Starting retrieval runs...")
    total = len(query_cases)
    for idx, (q_case, (q_case_id, q_text)) in enumerate(zip(query_cases, query_docs), start=1):
        for method, retriever in retrievers.items():
            t0 = time.perf_counter()
            ranked = retriever(q_case, q_text)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            data = metrics[method]
            data["recalls"].append([cid for cid, _ in ranked])
            data["latencies"].append(elapsed_ms)
            data["preds"].append(weighted_vote(ranked, index_labels))
        if idx % 20 == 0 or idx == total:
            print(f"Processed {idx}/{total} retrieval runs...")

    true_labels = [query_labels[c.case_id] for c in query_cases]
    rows = []
    for method in retrievers:
        data = metrics[method]
        rows.extend(summarize_method(
            dataset_name, method, data["preds"], data["recalls"],
            data["latencies"], true_labels, index_labels,
        ))

    b6_context = {
        "dataset_name": dataset_name,
        "bm25_ranking": bm25_ranking,
        "doc_text": doc_text,
        "index_labels": index_labels,
        "query_docs": query_docs,
        "true_labels": true_labels,
    }
    return rows, b6_context


def run_b6(b6_context, limit=None):
    """B6 long-context leg: HDFS only, run LAST (it is the only API consumer).

    The 'retrieved' list reported for hit rates is the BM25 top-10 -- the
    candidate set the LLM conditions on; B6 itself contributes the label
    decision, not a new ranking. Failed rows (after one retry) fall back to
    'Normal' for metric computation and are counted in the failure report.
    """
    dataset_name = b6_context["dataset_name"]
    query_docs = b6_context["query_docs"]
    if limit is not None:
        query_docs = query_docs[:limit]
        true_labels = b6_context["true_labels"][:limit]
    else:
        true_labels = b6_context["true_labels"]

    print(f"\n--- B6-LongContext-DeepSeek on {dataset_name} ({len(query_docs)} queries) ---")
    client = LongContextDeepSeek()
    if not client.api_key:
        print("[B6] SMA_DEEPSEEK_API_KEY not available; skipping B6 leg entirely.")
        return [], client

    preds, recalls, latencies = [], [], []
    for idx, (q_case_id, q_text) in enumerate(query_docs, start=1):
        pool = b6_context["bm25_ranking"](q_text, B6_PRECEDENTS)
        precedents = [
            (b6_context["index_labels"][cid], b6_context["doc_text"][cid])
            for cid, _ in pool
        ]
        t0 = time.perf_counter()
        label = client.classify(q_case_id, q_text, precedents)
        latencies.append((time.perf_counter() - t0) * 1000)
        preds.append(label if label is not None else "Normal")
        recalls.append([cid for cid, _ in pool[:10]])
        if idx % 20 == 0 or idx == len(query_docs):
            print(f"Processed {idx}/{len(query_docs)} B6 calls "
                  f"({len(client.failures)} failed so far)...")

    rows = summarize_method(
        dataset_name, "B6-LongContext-DeepSeek", preds, recalls, latencies,
        true_labels, b6_context["index_labels"],
    )
    print(f"[B6] API calls: {client.calls}, failed rows: {len(client.failures)}, "
          f"prompt tokens: {client.total_prompt_tokens}, "
          f"completion tokens: {client.total_completion_tokens}")
    for failure in client.failures:
        print(f"[B6 failure] {failure}")
    return rows, client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="60 index docs / 20 queries per dataset; B6 capped at 3 calls")
    parser.add_argument("--datasets", nargs="+", default=["HDFS", "BGL"],
                        choices=["HDFS", "BGL"])
    parser.add_argument("--skip-b6", action="store_true")
    parser.add_argument("--out", default=str(OUTPUT_CSV))
    args = parser.parse_args()

    # Be a polite tenant: other evals share this machine.
    try:
        import torch

        torch.set_num_threads(8)
    except ImportError:
        pass

    random.seed(42)  # match run_loghub_eval's global seeding
    hdfs_zip = REPO_ROOT / "data/raw/loghub_raw/HDFS_v1.zip"
    bgl_zip = REPO_ROOT / "data/raw/loghub_raw/BGL.zip"

    all_rows = []
    b6_context = None
    if "HDFS" in args.datasets:
        if not hdfs_zip.exists():
            print("Missing HDFS_v1.zip; run scripts/fetch_datasets.py first.")
            return 1
        print("Sampling HDFS sessions (1000 stratified, seed 42)...")
        hdfs_sampled = sample_hdfs_stratified(hdfs_zip, sample_size=1000, seed=42)
        check_manifest("HDFS", hdfs_sampled)
        rows, b6_context = evaluate_dataset("HDFS", hdfs_sampled, smoke=args.smoke)
        all_rows.extend(rows)
    if "BGL" in args.datasets:
        if not bgl_zip.exists():
            print("Missing BGL.zip; run scripts/fetch_datasets.py first.")
            return 1
        print("Sampling BGL sessions (1000 stratified, seed 42)...")
        bgl_sampled = sample_bgl_stratified(bgl_zip, sample_size=1000, seed=42)
        check_manifest("BGL", bgl_sampled)
        rows, _ = evaluate_dataset("BGL", bgl_sampled, smoke=args.smoke)
        all_rows.extend(rows)

    # B6 runs LAST: it is the only leg that spends API budget.
    if not args.skip_b6 and b6_context is not None:
        b6_rows, _client = run_b6(b6_context, limit=3 if args.smoke else None)
        all_rows.extend(b6_rows)
    elif not args.skip_b6:
        print("B6 leg skipped: HDFS was not evaluated in this invocation.")

    out_path = pathlib.Path(args.out)
    if args.smoke:
        out_path = out_path.with_name(out_path.stem + "_smoke.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
