"""LogHub MVP Diagnostic Evaluation script.

Performs stratified sampling on HDFS and BGL datasets, sessionizes logs,
indexes cases, runs retrieval via SMA, BM25, Dense RAG, and KG-PPR Proxy,
and outputs performance and latency metrics.
"""

from __future__ import annotations

import csv
import pathlib
import random
import time
import zipfile
from collections import defaultdict, Counter

import numpy as np
from sklearn.metrics import f1_score

from sma.encoders import get_encoder
from sma.index.macfac import MacFacIndex
from sma.match.types import MatchConfig


def load_hdfs_blocks(hdfs_zip_path: pathlib.Path) -> dict[str, str]:
    """Load block labels from the HDFS anomaly label CSV."""
    labels = {}
    with zipfile.ZipFile(hdfs_zip_path, "r") as z:
        with z.open("preprocessed/anomaly_label.csv") as fh:
            reader = csv.reader(fh.read().decode("utf-8").splitlines())
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    labels[row[0]] = row[1]
    return labels


def extract_hdfs_sessions(
    hdfs_zip_path: pathlib.Path, sampled_blocks: set[str]
) -> dict[str, list[str]]:
    """Stream HDFS.log and extract log lines for sampled blocks."""
    block_lines = defaultdict(list)
    import re
    block_re = re.compile(r"blk_-?\d+")
    
    with zipfile.ZipFile(hdfs_zip_path, "r") as z:
        with z.open("HDFS.log") as fh:
            for line_bytes in fh:
                line = line_bytes.decode("utf-8", errors="ignore")
                match = block_re.search(line)
                if match:
                    bid = match.group(0)
                    if bid in sampled_blocks:
                        block_lines[bid].append(line)
    return block_lines


def get_hdfs_first_timestamps(hdfs_zip_path: pathlib.Path) -> dict[str, float]:
    """Scan HDFS.log to find the first occurrence time (in seconds) of each block."""
    import re
    from datetime import datetime
    block_re = re.compile(r"blk_-?\d+")
    block_times = {}
    
    # Scan the entire HDFS log stream for unbiased temporal stratification
    with zipfile.ZipFile(hdfs_zip_path, "r") as z:
        with z.open("HDFS.log") as fh:
            for line_bytes in fh:
                line = line_bytes.decode("utf-8", errors="ignore")
                match = block_re.search(line)
                if match:
                    bid = match.group(0)
                    if bid not in block_times:
                        ts_str = line[:13]
                        if len(ts_str) == 13 and ts_str[6] == ' ' and ts_str[:6].isdigit() and ts_str[7:].isdigit():
                            try:
                                dt = datetime.strptime(ts_str, "%y%m%d %H%M%S")
                                block_times[bid] = dt.timestamp()
                            except ValueError:
                                block_times[bid] = 0.0
                        else:
                            block_times[bid] = 0.0
    return block_times


def sample_hdfs_stratified(
    hdfs_zip_path: pathlib.Path, sample_size: int = 1000, seed: int = 42
) -> list[tuple[str, str, str]]:
    """Sample stratified HDFS block sessions."""
    labels = load_hdfs_blocks(hdfs_zip_path)
    block_times = get_hdfs_first_timestamps(hdfs_zip_path)
    
    # Filter to blocks that we found timestamps for
    valid_blocks = [b for b in labels if b in block_times]
    
    anom_blocks = [b for b in valid_blocks if labels[b] == "Anomaly"]
    norm_blocks = [b for b in valid_blocks if labels[b] == "Normal"]
    
    rng = random.Random(seed)
    
    def get_stratified_subset(blocks, target_n):
        # Sort by timestamp
        sorted_blocks = sorted(blocks, key=lambda b: block_times[b])
        if len(sorted_blocks) <= target_n:
            return sorted_blocks
        
        # Divide into 5 bins
        bins = np.array_split(sorted_blocks, 5)
        subset = []
        per_bin = target_n // 5
        for b in bins:
            subset.extend(rng.sample(list(b), min(len(b), per_bin)))
        # Fill remainder
        while len(subset) < target_n and sorted_blocks:
            rem = list(set(sorted_blocks) - set(subset))
            if not rem:
                break
            subset.append(rng.choice(rem))
        return subset

    sampled_anom = get_stratified_subset(anom_blocks, sample_size // 2)
    sampled_norm = get_stratified_subset(norm_blocks, sample_size // 2)
    sampled_set = set(sampled_anom + sampled_norm)
    
    # Extract log texts
    block_lines = extract_hdfs_sessions(hdfs_zip_path, sampled_set)
    
    results = []
    for bid in sampled_anom + sampled_norm:
        lines = block_lines.get(bid, [])
        if lines:
            results.append((bid, "".join(lines), labels[bid]))
    return results


def sample_bgl_stratified(
    bgl_zip_path: pathlib.Path, sample_size: int = 1000, seed: int = 42
) -> list[tuple[str, str, str]]:
    """Sessionize and sample stratified BGL logs using two passes to save memory."""
    # Pass 1: Gather metadata for sessionization and labels
    session_counts = Counter()
    labels = defaultdict(bool)
    timestamps = {}
    
    with zipfile.ZipFile(bgl_zip_path, "r") as z:
        with z.open("BGL.log") as fh:
            for line_bytes in fh:
                line = line_bytes.decode("utf-8", errors="ignore")
                parts = line.split(maxsplit=5)
                if len(parts) < 5:
                    continue
                label = parts[0]
                try:
                    timestamp = int(parts[1])
                except ValueError:
                    continue
                node_id = parts[3]
                
                # Group BGL into 60-second windows per Node ID as per blueprint
                window = timestamp // 60
                session_key = f"bgl_{node_id}_{window}"
                session_counts[session_key] += 1
                if label != "-":
                    labels[session_key] = True
                if session_key not in timestamps:
                    timestamps[session_key] = timestamp

    # Filter sessions with length >= 3 to avoid tiny cases
    filtered_keys = [k for k, count in session_counts.items() if count >= 3]
    anom_keys = [k for k in filtered_keys if labels[k]]
    norm_keys = [k for k in filtered_keys if not labels[k]]
    
    rng = random.Random(seed)
    
    def get_stratified_subset(keys, target_n):
        sorted_keys = sorted(keys, key=lambda k: timestamps[k])
        if len(sorted_keys) <= target_n:
            return sorted_keys
        bins = np.array_split(sorted_keys, 5)
        subset = []
        per_bin = target_n // 5
        for b in bins:
            subset.extend(rng.sample(list(b), min(len(b), per_bin)))
        while len(subset) < target_n and sorted_keys:
            rem = list(set(sorted_keys) - set(subset))
            if not rem:
                break
            subset.append(rng.choice(rem))
        return subset

    sampled_anom = get_stratified_subset(anom_keys, sample_size // 2)
    sampled_norm = get_stratified_subset(norm_keys, sample_size // 2)
    sampled_set = set(sampled_anom + sampled_norm)
    
    # Pass 2: Extract actual lines for the sampled set
    sessions_lines = defaultdict(list)
    with zipfile.ZipFile(bgl_zip_path, "r") as z:
        with z.open("BGL.log") as fh:
            for line_bytes in fh:
                line = line_bytes.decode("utf-8", errors="ignore")
                parts = line.split(maxsplit=5)
                if len(parts) < 5:
                    continue
                try:
                    timestamp = int(parts[1])
                except ValueError:
                    continue
                node_id = parts[3]
                window = timestamp // 60
                session_key = f"bgl_{node_id}_{window}"
                if session_key in sampled_set:
                    # Drop the leading alert-category column: it is the ground-truth
                    # label, not log content. Keeping it leaks labels to every
                    # retriever (BGL '-' = normal, anything else = anomaly).
                    sessions_lines[session_key].append(line.partition(" ")[2] or line)
                    
    results = []
    for k in sampled_anom + sampled_norm:
        lines = sessions_lines.get(k, [])
        if lines:
            results.append((k, "".join(lines), "Anomaly" if labels[k] else "Normal"))
    return results


def run_evaluation(
    dataset_name: str,
    sampled_data: list[tuple[str, str, str]],
    output_manifest_rows: list[dict],
    scorer: str = "ses",
) -> list[dict]:
    """Execute four-way evaluation comparison on the sampled dataset."""
    print(f"\n--- Running evaluation on {dataset_name} ({len(sampled_data)} cases) ---")
    
    # Save to manifest list
    for sid, _, label in sampled_data:
        output_manifest_rows.append({
            "dataset": dataset_name,
            "session_id": sid,
            "label": label
        })
        
    # Split into 80% Index / 20% Query
    random.Random(101).shuffle(sampled_data)
    split_idx = int(len(sampled_data) * 0.8)
    index_data = sampled_data[:split_idx]
    query_data = sampled_data[split_idx:]
    
    # Parse and encode cases
    log_encoder = get_encoder("logs")
    
    print("Encoding index cases...")
    index_cases = []
    index_docs = []  # List of (case_id, text)
    index_labels = {}
    for sid, text, label in index_data:
        case = log_encoder.encode(text, session_id=sid).case
        index_cases.append(case)
        index_docs.append((case.case_id, text))
        index_labels[case.case_id] = label
        
    print("Encoding query cases...")
    query_cases = []
    query_docs = []
    query_labels = {}
    for sid, text, label in query_data:
        case = log_encoder.encode(text, session_id=sid).case
        query_cases.append(case)
        query_docs.append((case.case_id, text))
        query_labels[case.case_id] = label

    # Build indexes ONCE before the query loop
    # 1. Build SMA MAC/FAC index
    print(f"Building SMA Index (scorer={scorer})...")
    sma_index = MacFacIndex(config=MatchConfig(scorer=scorer))
    sma_index.build(index_cases)
    
    # 2. Build BM25 Index
    print("Building BM25 Index...")
    from rank_bm25 import BM25Okapi
    tokenized_index = [text.lower().split() for _, text in index_docs]
    bm25_index = BM25Okapi(tokenized_index)
    
    # 3. Build Dense RAG Index (SentenceTransformers)
    print("Building Dense RAG Index (SentenceTransformers)...")
    from sentence_transformers import SentenceTransformer, util
    dense_model = SentenceTransformer('all-MiniLM-L6-v2')
    index_texts = [text for _, text in index_docs]
    index_embeddings = dense_model.encode(index_texts, convert_to_tensor=True, show_progress_bar=False)
    
    # 4. Build KG-PPR Proxy index
    print("Building KG-PPR Proxy Index...")
    index_entity_counters = {
        ic.case_id: Counter(e.name for e in ic.entities())
        for ic in index_cases
    }
    
    # Per-query ranked retrieval for each method, as (case_id, score) pairs.
    def retrieve_sma(q_case, q_text):
        # shortlist=40, fac_budget=20 keeps CPU latency bounded
        results = sma_index.retrieve(q_case, k=10, shortlist=40, fac_budget=20)
        return [(r.case_id, r.ses_n) for r in results]

    def retrieve_bm25(q_case, q_text):
        scores = bm25_index.get_scores(q_text.lower().split())
        ranked = sorted(zip(doc_ids, scores), key=lambda row: (-row[1], row[0]))
        return ranked[:10]

    def retrieve_dense(q_case, q_text):
        query_embedding = dense_model.encode(q_text, convert_to_tensor=True, show_progress_bar=False)
        scores = util.cos_sim(query_embedding, index_embeddings)[0].cpu().tolist()
        ranked = sorted(zip(doc_ids, scores), key=lambda row: (-row[1], row[0]))
        return ranked[:10]

    def retrieve_kg(q_case, q_text):
        q_counter = Counter(e.name for e in q_case.entities())
        ranked = sorted(
            (
                (ic_id, float(sum(min(v, counts.get(k, 0)) for k, v in q_counter.items())))
                for ic_id, counts in index_entity_counters.items()
            ),
            key=lambda row: (-row[1], row[0]),
        )
        return ranked[:10]

    def weighted_vote(ranked, top=5):
        voting = {"Anomaly": 0.0, "Normal": 0.0}
        for case_id, score in ranked[:top]:
            voting[index_labels[case_id]] += score
        return max(voting, key=voting.get) if sum(voting.values()) > 0 else "Normal"

    retrievers = {
        "SMA": retrieve_sma,
        "BM25": retrieve_bm25,
        "Dense RAG": retrieve_dense,
        "KG-PPR Proxy": retrieve_kg,
    }
    methods = list(retrievers)
    metrics_by_method = {m: {"recalls": [], "preds": [], "latencies": []} for m in methods}
    doc_ids = [doc_id for doc_id, _ in index_docs]

    print("Starting retrieval runs...")
    total_queries = len(query_cases)
    for idx, (q_case, (q_case_id, q_text)) in enumerate(zip(query_cases, query_docs), start=1):
        for method, retriever in retrievers.items():
            t0 = time.perf_counter()
            ranked = retriever(q_case, q_text)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            data = metrics_by_method[method]
            data["recalls"].append([case_id for case_id, _ in ranked])
            data["latencies"].append(elapsed_ms)
            data["preds"].append(weighted_vote(ranked))

        if idx % 20 == 0 or idx == total_queries:
            print(f"Processed {idx}/{total_queries} retrieval runs...")

    # Calculate final metrics
    triage_rows = []
    true_labels = [query_labels[c.case_id] for c in query_cases]
    
    for m in methods:
        data = metrics_by_method[m]
        preds = data["preds"]
        recalls = data["recalls"]
        latencies = data["latencies"]
        
        # F1 Score
        f1 = f1_score(true_labels, preds, average="macro")
        
        # label_hit_rate @ 1, 5, 10
        r1_list = []
        r5_list = []
        r10_list = []
        for q_idx, q_case in enumerate(query_cases):
            q_label = query_labels[q_case.case_id]
            ret_ids = recalls[q_idx]
            
            # Find all relevant index cases for this query
            relevant_ids = {ic.case_id for ic in index_cases if index_labels[ic.case_id] == q_label}
            
            # Hit rate at k = count of retrieved relevant / min(relevant_ids, k)
            def compute_hit_rate_k(k):
                hits = len(set(ret_ids[:k]).intersection(relevant_ids))
                denom = min(len(relevant_ids), k)
                return hits / denom if denom > 0 else 0.0
                
            r1_list.append(compute_hit_rate_k(1))
            r5_list.append(compute_hit_rate_k(5))
            r10_list.append(compute_hit_rate_k(10))
            
        r1 = sum(r1_list) / len(r1_list)
        r5 = sum(r5_list) / len(r5_list)
        r10 = sum(r10_list) / len(r10_list)
        
        # Latency p50, p95
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        
        triage_rows.append({
            "dataset": "LogHub",
            "split": f"{dataset_name}_MVP_diagnostic",
            "method": m,
            "macro_f1": f"{f1:.4f}",
            "label_hit_rate@1": f"{r1:.4f}",
            "label_hit_rate@5": f"{r5:.4f}",
            "label_hit_rate@10": f"{r10:.4f}",
            "p50_ms": f"{p50:.3f}",
            "p95_ms": f"{p95:.3f}"
        })
        
        # Print results
        print(f"Method: {m}")
        print(f"  Macro-F1: {f1:.4f}")
        print(f"  label_hit_rate@1: {r1:.4f}, label_hit_rate@5: {r5:.4f}, label_hit_rate@10: {r10:.4f}")
        print(f"  p50 Latency: {p50:.3f} ms, p95 Latency: {p95:.3f} ms")
        
        # Diagnostic alerts for collapsed or suspiciously perfect runs
        unique_preds = set(preds)
        is_suspicious = (f1 == 0.0 or f1 == 1.0 or len(unique_preds) <= 1)
        if is_suspicious:
            reason = ""
            if f1 == 0.0:
                reason = "F1 is 0.0: Retrieval collapse or dataset imbalance"
            elif f1 == 1.0:
                reason = "F1 is 1.0: Suspiciously perfect classification - potential data leakage or indexing overlap"
            elif len(unique_preds) <= 1:
                reason = f"Retrieval collapse: predicted only '{list(unique_preds)[0]}' sessions"
                
            triage_rows.append({
                "dataset": "DIAGNOSTIC",
                "split": f"{dataset_name}_MVP_diagnostic",
                "method": f"{m}_alert",
                "macro_f1": reason,
                "label_hit_rate@1": "ALERT",
                "label_hit_rate@5": "ALERT",
                "label_hit_rate@10": "ALERT",
                "p50_ms": "0.000",
                "p95_ms": "0.000"
            })
            print(f"  [DIAGNOSTIC ALERT] {reason}")
        
    return triage_rows


def run_loghub_eval(scorer: str = "ses") -> list[dict]:
    """Execute both HDFS and BGL evaluations and write manifests."""
    random.seed(42)
    
    hdfs_zip = pathlib.Path("data/raw/loghub_raw/HDFS_v1.zip")
    bgl_zip = pathlib.Path("data/raw/loghub_raw/BGL.zip")
    
    if not hdfs_zip.exists() or not bgl_zip.exists():
        print("Missing log datasets. Run fetch_datasets.py first.")
        return []
        
    manifest_rows = []
    
    # 1. HDFS stratified sampling & evaluation
    print("Sampling HDFS sessions...")
    hdfs_sampled = sample_hdfs_stratified(hdfs_zip, sample_size=1000, seed=42)
    hdfs_rows = run_evaluation("HDFS", hdfs_sampled, manifest_rows, scorer=scorer)

    # 2. BGL stratified sampling & evaluation
    print("Sampling BGL sessions...")
    bgl_sampled = sample_bgl_stratified(bgl_zip, sample_size=1000, seed=42)
    bgl_rows = run_evaluation("BGL", bgl_sampled, manifest_rows, scorer=scorer)
    
    # Save the sampled manifest for reproducibility
    manifest_path = pathlib.Path("reports/loghub_sample_manifest.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "session_id", "label"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Saved manifest to {manifest_path}")
    
    return hdfs_rows + bgl_rows


if __name__ == "__main__":
    run_loghub_eval()
