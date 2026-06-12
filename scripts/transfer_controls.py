"""Cross-system transfer controls: do WL-kernel and the production RAG stack transfer?

The baseline ladder showed WL-kernel (generic graph similarity on SMA's own
Tier-0 extraction) BEATS full SME matching within-system on HDFS (0.9799 vs
0.9549). The decisive question for the matcher's existence: does it also
transfer cross-system, where SMA-SES scored 0.92-0.965 (BGL->Spirit, frozen
ontology) while BM25/dense/KG sat near chance? Same question for Hybrid-RRF
and Hybrid+Rerank — the stack production RAG actually ships.

Protocol mirrors sma/eval/transfer_eval.py: index = 800 stratified BGL
sessions (seed 42), query = 200 stratified Spirit sessions (seed 42),
weighted top-5 vote, macro-F1 + label_hit_rate@k + latency.

Usage: python3 -u scripts/transfer_controls.py [--out reports/transfer_controls_metrics.csv]
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import time

import numpy as np
from sklearn.metrics import f1_score

from sma.encoders import get_encoder
from sma.eval.baselines.hybrid_rrf import rrf_fuse
from sma.eval.baselines.rerank import CrossEncoderReranker
from sma.eval.baselines.wl_kernel import WLKernelRetriever
from sma.eval.loghub_eval import sample_bgl_stratified
from sma.eval.transfer_eval import sample_spirit

FIELDS = ["dataset", "split", "method", "macro_f1", "label_hit_rate@1",
          "label_hit_rate@5", "label_hit_rate@10", "p50_ms", "p95_ms"]
SPLIT = "BGL->spirit_first20M[seed42][controls]"


def weighted_vote(ranked, index_labels, top=5):
    voting = {"Anomaly": 0.0, "Normal": 0.0}
    for doc_id, score in ranked[:top]:
        voting[index_labels[doc_id]] += score
    return max(voting, key=voting.get) if sum(voting.values()) > 0 else "Normal"


def hit_rates(ranked_ids, q_label, index_labels, relevant_count):
    out = []
    for k in (1, 5, 10):
        hits = sum(1 for doc_id in ranked_ids[:k] if index_labels[doc_id] == q_label)
        denom = min(relevant_count, k)
        out.append(hits / denom if denom else 0.0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/transfer_controls_metrics.csv")
    parser.add_argument("--index-size", type=int, default=800)
    parser.add_argument("--query-size", type=int, default=200)
    args = parser.parse_args()

    bgl = pathlib.Path("data/raw/loghub_raw/BGL.zip")
    spirit = pathlib.Path("data/raw/loghub_raw/spirit2.gz")
    print("sampling BGL index set...", flush=True)
    index_data = sample_bgl_stratified(bgl, sample_size=args.index_size, seed=42)
    print("sampling Spirit query set...", flush=True)
    query_data = sample_spirit(spirit, sample_size=args.query_size, seed=42)
    print(f"index={len(index_data)} query={len(query_data)}", flush=True)

    encoder = get_encoder("logs")
    print("encoding index cases (Tier-0)...", flush=True)
    index_cases = [encoder.encode(text, session_id=sid).case for sid, text, _ in index_data]
    index_ids = [case.case_id for case in index_cases]
    index_labels = {case.case_id: label for case, (_, _, label) in zip(index_cases, index_data)}
    index_texts = {case.case_id: text for case, (_, text, _) in zip(index_cases, index_data)}

    print("building WL index...", flush=True)
    wl = WLKernelRetriever()
    wl.build(index_cases)

    print("building BM25 index...", flush=True)
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([text.lower().split() for _, text, _ in index_data])

    print("building BGE dense index...", flush=True)
    from sentence_transformers import SentenceTransformer

    bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
    bge_index = bge.encode([text for _, text, _ in index_data],
                           normalize_embeddings=True, show_progress_bar=False, batch_size=16)

    print("loading reranker...", flush=True)
    reranker = CrossEncoderReranker()

    def bm25_ranking(q_text, k=20):
        scores = bm25.get_scores(q_text.lower().split())
        return sorted(zip(index_ids, scores), key=lambda r: (-r[1], r[0]))[:k]

    def bge_ranking(q_text, k=20):
        qv = bge.encode([q_text], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = bge_index @ qv
        return sorted(zip(index_ids, scores.tolist()), key=lambda r: (-r[1], r[0]))[:k]

    def hybrid_ranking(q_text, k=20):
        return rrf_fuse([bm25_ranking(q_text), bge_ranking(q_text)], top_k=k)

    methods = {
        "WL-kernel": lambda q_case, q_text: wl.retrieve(q_case, k=10),
        "Hybrid-RRF": lambda q_case, q_text: hybrid_ranking(q_text, k=10),
        "Hybrid+Rerank": lambda q_case, q_text: reranker.rerank(
            q_text, [(doc_id, index_texts[doc_id]) for doc_id, _ in hybrid_ranking(q_text, k=20)], top_k=10
        ),
    }

    per_method = {m: {"preds": [], "rates": [], "lat": []} for m in methods}
    true_labels = []
    n_anom = sum(1 for _, _, label in index_data if label == "Anomaly")
    n_norm = len(index_data) - n_anom

    print("running queries...", flush=True)
    for qi, (sid, q_text, q_label) in enumerate(query_data, start=1):
        q_case = encoder.encode(q_text, session_id=sid).case
        true_labels.append(q_label)
        relevant = n_anom if q_label == "Anomaly" else n_norm
        for name, fn in methods.items():
            t0 = time.perf_counter()
            ranked = fn(q_case, q_text)
            dt = (time.perf_counter() - t0) * 1000
            data = per_method[name]
            data["lat"].append(dt)
            data["preds"].append(weighted_vote(ranked, index_labels))
            data["rates"].append(hit_rates([d for d, _ in ranked], q_label, index_labels, relevant))
        if qi % 20 == 0:
            print(f"  {qi}/{len(query_data)}", flush=True)

    rows = []
    for name, data in per_method.items():
        f1 = f1_score(true_labels, data["preds"], average="macro")
        rates = np.array(data["rates"])
        row = {
            "dataset": "LogHub",
            "split": SPLIT,
            "method": name,
            "macro_f1": f"{f1:.4f}",
            "label_hit_rate@1": f"{rates[:, 0].mean():.4f}",
            "label_hit_rate@5": f"{rates[:, 1].mean():.4f}",
            "label_hit_rate@10": f"{rates[:, 2].mean():.4f}",
            "p50_ms": f"{np.percentile(data['lat'], 50):.3f}",
            "p95_ms": f"{np.percentile(data['lat'], 95):.3f}",
        }
        rows.append(row)
        print(f"{name:14s} F1={f1:.4f} hit@1={rates[:,0].mean():.4f} "
              f"p50={np.percentile(data['lat'],50):.0f}ms", flush=True)
        unique_preds = set(data["preds"])
        if len(unique_preds) <= 1:
            print(f"  [DIAGNOSTIC ALERT] {name}: predicted only {unique_preds}", flush=True)
            rows.append({"dataset": "DIAGNOSTIC", "split": SPLIT, "method": f"{name}_alert",
                         "macro_f1": f"Retrieval collapse: predicted only {list(unique_preds)[0]!r}",
                         "label_hit_rate@1": "ALERT", "label_hit_rate@5": "ALERT",
                         "label_hit_rate@10": "ALERT", "p50_ms": "0.000", "p95_ms": "0.000"})

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    with out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"appended {len(rows)} rows to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
