"""Failure-family retrieval evaluation (family-hit@k) for HDFS and BGL.

Re-runs the within-system protocol of ``sma.eval.loghub_eval`` (1000
stratified sessions seed 42, 80/20 index/query split seed 101, identical
retrieval parameters) but scores retrieval on deterministic FAILURE
FAMILIES (see ``sma.eval.family_labels``) instead of the shallow binary
Anomaly/Normal label: did retrieval surface the correct root-cause
family, not just "an anomaly"?

Methods: SMA-ses, SMA-mdl, BM25, Dense (all-MiniLM-L6-v2).

Per query whose family != "normal":
  family_hit@k = |top-k retrieved with the same family| /
                 min(k, number of same-family sessions in the index)
Queries with zero same-family index sessions cannot be hit and are
excluded from the family averages (counts reported). Binary macro-F1
(weighted top-5 vote, as in loghub_eval) is reported for continuity.

Outputs:
  reports/family_metrics.csv    dataset, method, n_query_families,
                                family_hit@1/@5/@10, macro_f1_binary
  reports/family_inventory.csv  dataset, family, n_sessions

Usage: python3 scripts/family_eval.py [--sample-size 1000]
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import random
import sys
from collections import Counter

from sklearn.metrics import f1_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sma.encoders import get_encoder
from sma.eval.family_labels import bgl_family, hdfs_family
from sma.eval.loghub_eval import sample_bgl_stratified, sample_hdfs_stratified
from sma.index.macfac import MacFacIndex
from sma.match.types import MatchConfig

HDFS_ZIP = pathlib.Path("data/raw/loghub_raw/HDFS_v1.zip")
BGL_ZIP = pathlib.Path("data/raw/loghub_raw/BGL.zip")


def build_family_map(
    dataset: str, sampled: list[tuple[str, str, str]]
) -> dict[str, str]:
    """Map session id -> failure family for every sampled session."""
    if dataset == "HDFS":
        return {sid: hdfs_family(text, label) for sid, text, label in sampled}

    # BGL: families come from the raw log's alert-category column (the
    # column is stripped from session text by the sampler).
    keys = {sid for sid, _, _ in sampled}
    fam = bgl_family(BGL_ZIP, keys)

    # Verify the key scheme matches the sampler's by reconstruction:
    # every sampled key must resolve, and family=="normal" must agree
    # with the sampler's binary label exactly.
    missing = [sid for sid, _, _ in sampled if sid not in fam]
    mismatched = [
        sid
        for sid, _, label in sampled
        if sid in fam and (fam[sid] == "normal") != (label == "Normal")
    ]
    if missing or mismatched:
        raise RuntimeError(
            f"BGL key verification FAILED: {len(missing)} missing keys, "
            f"{len(mismatched)} binary-label mismatches "
            f"(examples: {missing[:3] + mismatched[:3]})"
        )
    print(
        f"BGL key verification OK: {len(keys)} sampled keys all resolved, "
        "binary labels agree with alert-category derivation."
    )
    return {sid: fam[sid] for sid, _, _ in sampled}


def run_family_evaluation(
    dataset: str,
    sampled_data: list[tuple[str, str, str]],
    families: dict[str, str],
    dense_model,
) -> list[dict]:
    """Four-method family-hit evaluation on one dataset."""
    print(f"\n--- Family evaluation on {dataset} ({len(sampled_data)} sessions) ---")

    # Identical split protocol to loghub_eval.run_evaluation.
    random.Random(101).shuffle(sampled_data)
    split_idx = int(len(sampled_data) * 0.8)
    index_data = sampled_data[:split_idx]
    query_data = sampled_data[split_idx:]

    log_encoder = get_encoder("logs")

    print("Encoding index cases...")
    index_cases, index_docs = [], []
    index_labels, index_families = {}, {}
    for sid, text, label in index_data:
        case = log_encoder.encode(text, session_id=sid).case
        index_cases.append(case)
        index_docs.append((case.case_id, text))
        index_labels[case.case_id] = label
        index_families[case.case_id] = families[sid]

    print("Encoding query cases...")
    query_cases, query_docs = [], []
    query_labels, query_families = {}, {}
    for sid, text, label in query_data:
        case = log_encoder.encode(text, session_id=sid).case
        query_cases.append(case)
        query_docs.append((case.case_id, text))
        query_labels[case.case_id] = label
        query_families[case.case_id] = families[sid]

    index_family_counts = Counter(index_families.values())
    doc_ids = [doc_id for doc_id, _ in index_docs]

    print("Building SMA-ses index...")
    sma_ses = MacFacIndex(config=MatchConfig(scorer="ses"))
    sma_ses.build(index_cases)
    print("Building SMA-mdl index...")
    sma_mdl = MacFacIndex(config=MatchConfig(scorer="mdl"))
    sma_mdl.build(index_cases)

    print("Building BM25 index...")
    from rank_bm25 import BM25Okapi

    bm25_index = BM25Okapi([text.lower().split() for _, text in index_docs])

    print("Building Dense index (all-MiniLM-L6-v2)...")
    from sentence_transformers import util

    index_embeddings = dense_model.encode(
        [text for _, text in index_docs], convert_to_tensor=True, show_progress_bar=False
    )

    def retrieve_sma(index):
        def _r(q_case, q_text):
            results = index.retrieve(q_case, k=10, shortlist=40, fac_budget=20)
            return [(r.case_id, r.ses_n) for r in results]

        return _r

    def retrieve_bm25(q_case, q_text):
        scores = bm25_index.get_scores(q_text.lower().split())
        ranked = sorted(zip(doc_ids, scores), key=lambda row: (-row[1], row[0]))
        return ranked[:10]

    def retrieve_dense(q_case, q_text):
        q_emb = dense_model.encode(q_text, convert_to_tensor=True, show_progress_bar=False)
        scores = util.cos_sim(q_emb, index_embeddings)[0].cpu().tolist()
        ranked = sorted(zip(doc_ids, scores), key=lambda row: (-row[1], row[0]))
        return ranked[:10]

    def weighted_vote(ranked, top=5):
        voting = {"Anomaly": 0.0, "Normal": 0.0}
        for case_id, score in ranked[:top]:
            voting[index_labels[case_id]] += score
        return max(voting, key=voting.get) if sum(voting.values()) > 0 else "Normal"

    retrievers = {
        "SMA-ses": retrieve_sma(sma_ses),
        "SMA-mdl": retrieve_sma(sma_mdl),
        "BM25": retrieve_bm25,
        "Dense": retrieve_dense,
    }
    per_method = {m: {"rankings": [], "preds": []} for m in retrievers}

    total = len(query_cases)
    print("Running retrieval...")
    for idx, (q_case, (q_case_id, q_text)) in enumerate(
        zip(query_cases, query_docs), start=1
    ):
        for method, retriever in retrievers.items():
            ranked = retriever(q_case, q_text)
            per_method[method]["rankings"].append([cid for cid, _ in ranked])
            per_method[method]["preds"].append(weighted_vote(ranked))
        if idx % 20 == 0 or idx == total:
            print(f"  {idx}/{total} queries done")

    true_labels = [query_labels[c.case_id] for c in query_cases]
    rows = []
    for method, data in per_method.items():
        f1 = f1_score(true_labels, data["preds"], average="macro")

        hit_lists = {1: [], 5: [], 10: []}
        scored_families = set()
        n_scored = n_skipped = 0
        for q_idx, q_case in enumerate(query_cases):
            q_fam = query_families[q_case.case_id]
            if q_fam == "normal":
                continue
            available = index_family_counts.get(q_fam, 0)
            if available == 0:
                n_skipped += 1
                continue
            n_scored += 1
            scored_families.add(q_fam)
            ret = data["rankings"][q_idx]
            for k in (1, 5, 10):
                hits = sum(1 for cid in ret[:k] if index_families[cid] == q_fam)
                hit_lists[k].append(hits / min(k, available))

        fh = {
            k: (sum(v) / len(v) if v else 0.0) for k, v in hit_lists.items()
        }
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n_query_families": len(scored_families),
                "family_hit@1": f"{fh[1]:.4f}",
                "family_hit@5": f"{fh[5]:.4f}",
                "family_hit@10": f"{fh[10]:.4f}",
                "macro_f1_binary": f"{f1:.4f}",
            }
        )
        print(
            f"{method}: family_hit@1={fh[1]:.4f} @5={fh[5]:.4f} @10={fh[10]:.4f} "
            f"binary_macro_f1={f1:.4f} "
            f"(scored {n_scored} anomalous queries over {len(scored_families)} "
            f"families; {n_skipped} skipped, no same-family index session)"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()

    if not HDFS_ZIP.exists() or not BGL_ZIP.exists():
        print("Missing log datasets. Run fetch_datasets.py first.")
        return

    random.seed(42)

    print("Loading dense model once...")
    from sentence_transformers import SentenceTransformer

    dense_model = SentenceTransformer("all-MiniLM-L6-v2")

    inventory_rows = []
    metric_rows = []

    for dataset, sampler, zip_path in (
        ("HDFS", sample_hdfs_stratified, HDFS_ZIP),
        ("BGL", sample_bgl_stratified, BGL_ZIP),
    ):
        print(f"\nSampling {dataset} (n={args.sample_size}, seed=42)...")
        sampled = sampler(zip_path, sample_size=args.sample_size, seed=42)
        families = build_family_map(dataset, sampled)

        inv = Counter(families.values())
        print(f"{dataset} family inventory:")
        for fam, n in inv.most_common():
            print(f"  {n:5d}  {fam}")
            inventory_rows.append(
                {"dataset": dataset, "family": fam, "n_sessions": n}
            )

        metric_rows.extend(
            run_family_evaluation(dataset, sampled, families, dense_model)
        )

    reports = pathlib.Path("reports")
    reports.mkdir(parents=True, exist_ok=True)

    inv_path = reports / "family_inventory.csv"
    with inv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "family", "n_sessions"])
        writer.writeheader()
        writer.writerows(inventory_rows)
    print(f"\nSaved {inv_path}")

    met_path = reports / "family_metrics.csv"
    with met_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset",
                "method",
                "n_query_families",
                "family_hit@1",
                "family_hit@5",
                "family_hit@10",
                "macro_f1_binary",
            ],
        )
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"Saved {met_path}")

    print("\n=== Family metrics ===")
    for row in metric_rows:
        print(
            f"{row['dataset']:<5} {row['method']:<8} "
            f"fam@1={row['family_hit@1']} fam@5={row['family_hit@5']} "
            f"fam@10={row['family_hit@10']} binF1={row['macro_f1_binary']}"
        )


if __name__ == "__main__":
    main()
