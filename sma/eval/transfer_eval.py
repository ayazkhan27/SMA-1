"""Cross-system transfer evaluation (blueprint section 8.3, task T2-b).

Indexes incidents from one log system and queries with incidents from a
DIFFERENT system: HDFS->OpenStack and BGL->Thunderbird. Vocabularies differ
across systems but failure motifs recur, so this is the unseen-concept test
on real data. Compares the same four retrieval methods as loghub_eval
(SMA, BM25, Dense RAG, KG-PPR Proxy) with weighted vote, label_hit_rate@k
and latency metrics, but WITHOUT an 80/20 split: the index set comes
entirely from system A and the query set entirely from system B.

Run as: python3 -u -m sma.eval.transfer_eval [--scorer ses|mdl]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import pathlib
import random
import re
import tarfile
import time
import zipfile
from collections import defaultdict, Counter

import numpy as np
from sklearn.metrics import f1_score

from sma.encoders import get_encoder
from sma.index.macfac import MacFacIndex
from sma.match.types import MatchConfig
from sma.eval.loghub_eval import sample_hdfs_stratified, sample_bgl_stratified

# Expected checksum of a complete Thunderbird.tar.gz (a previous copy was
# corrupt; we refuse to evaluate against anything that does not match).
THUNDERBIRD_MD5 = "0891b048df2919dc78c99c4428686b44"

# Thunderbird is huge (~30GB uncompressed, ~211M lines). For tractability we
# cap both streaming passes at the first 20 million lines; the split name
# records this cap as "thunderbird_first20M".
THUNDERBIRD_LINE_CAP = 20_000_000

# Spirit (Sandia supercomputer, Oliner & Stearley 2007) held-out transfer
# target. Source: USENIX CFDR hpc4/spirit2.gz (NOT in the LogHub Zenodo
# records; see data/manifests/datasets.json source_note). Same alert-flag
# format family as BGL/Thunderbird. md5 verified before every evaluation,
# like Thunderbird.
SPIRIT_MD5 = "ba6271c4f454bc21634b19c406d9769c"

# Spirit is ~37GB uncompressed (~272M lines). Same tractability cap as
# Thunderbird: both streaming passes stop at the first 20 million lines;
# the split name records this cap as "spirit_first20M".
SPIRIT_LINE_CAP = 20_000_000

OPENSTACK_INSTANCE_RE = re.compile(r"instance: ([0-9a-f-]{36})")


def get_stratified_subset(keys, target_n, sort_key, rng):
    """Sample target_n keys stratified over 5 temporal bins (same scheme as
    the nested helpers in loghub_eval)."""
    sorted_keys = sorted(keys, key=sort_key)
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


def sample_openstack(
    path: pathlib.Path, sample_size: int = 200, seed: int = 42
) -> list[tuple[str, str, str]]:
    """Sessionize and sample stratified OpenStack logs by VM instance id.

    The LogHub OpenStack archive contains openstack_normal1.log,
    openstack_normal2.log and openstack_abnormal.log. Sessions are grouped by
    the VM instance id appearing as "[instance: <uuid>]"; a session is labeled
    Anomaly iff it comes from the abnormal log (instance-id sets of the normal
    and abnormal runs are disjoint).
    """
    members = [
        ("openstack_normal1.log", "Normal"),
        ("openstack_normal2.log", "Normal"),
        ("openstack_abnormal.log", "Anomaly"),
    ]

    # Pass 1: gather session sizes, labels and first-seen order
    session_counts = Counter()
    labels = {}
    first_seen = {}
    line_no = 0
    with tarfile.open(path, "r:gz") as tar:
        for member_name, label in members:
            with tar.extractfile(member_name) as fh:
                for line_bytes in fh:
                    line_no += 1
                    line = line_bytes.decode("utf-8", errors="ignore")
                    match = OPENSTACK_INSTANCE_RE.search(line)
                    if not match:
                        continue
                    key = f"openstack_{match.group(1)}"
                    session_counts[key] += 1
                    if label == "Anomaly":
                        labels[key] = "Anomaly"
                    else:
                        labels.setdefault(key, "Normal")
                    # Files are time-ordered, so first-seen line index is a
                    # monotone proxy for the first timestamp (used only for
                    # the 5-bin temporal stratification below).
                    if key not in first_seen:
                        first_seen[key] = line_no

    # Filter sessions with length >= 3 to avoid tiny cases (BGL convention)
    filtered_keys = [k for k, count in session_counts.items() if count >= 3]
    anom_keys = [k for k in filtered_keys if labels[k] == "Anomaly"]
    norm_keys = [k for k in filtered_keys if labels[k] == "Normal"]

    rng = random.Random(seed)
    sampled_anom = get_stratified_subset(
        anom_keys, sample_size // 2, lambda k: first_seen[k], rng
    )
    sampled_norm = get_stratified_subset(
        norm_keys, sample_size // 2, lambda k: first_seen[k], rng
    )
    sampled_set = set(sampled_anom + sampled_norm)

    # Pass 2: extract actual lines for the sampled set
    sessions_lines = defaultdict(list)
    with tarfile.open(path, "r:gz") as tar:
        for member_name, _label in members:
            with tar.extractfile(member_name) as fh:
                for line_bytes in fh:
                    line = line_bytes.decode("utf-8", errors="ignore")
                    match = OPENSTACK_INSTANCE_RE.search(line)
                    if not match:
                        continue
                    key = f"openstack_{match.group(1)}"
                    if key in sampled_set:
                        # Drop the leading source-filename column: which file
                        # a line came from perfectly encodes the session label
                        # (normal vs abnormal run), so keeping it would leak
                        # labels into the text just like the BGL alert column.
                        sessions_lines[key].append(line.partition(" ")[2] or line)

    results = []
    for k in sampled_anom + sampled_norm:
        lines = sessions_lines.get(k, [])
        if lines:
            results.append((k, "".join(lines), labels[k]))
    return results


def check_thunderbird(path: pathlib.Path) -> str | None:
    """Return None if Thunderbird.tar.gz is present and checksum-verified,
    otherwise a human-readable reason to skip the BGL->Thunderbird pair."""
    if not path.exists():
        return f"{path} is missing (download may still be in progress)"
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != THUNDERBIRD_MD5:
        return (
            f"{path} md5 mismatch: expected {THUNDERBIRD_MD5}, got {actual} "
            "(file incomplete or corrupt; a previous copy was corrupt too)"
        )
    return None


def sample_thunderbird(
    path: pathlib.Path, sample_size: int = 200, seed: int = 42
) -> list[tuple[str, str, str]]:
    """Sessionize and sample stratified Thunderbird logs (BGL-like format).

    Streams the tar.gz in two passes without extracting to disk or holding
    lines in memory, like sample_bgl_stratified. Sessionizes per node into
    60-second windows; the first whitespace-separated field is the
    ground-truth label column ("-" = normal, anything else = alert category)
    and is STRIPPED from extracted text to avoid label leakage. Both passes
    are capped at the first THUNDERBIRD_LINE_CAP (20M) lines for tractability
    (the split name "thunderbird_first20M" records the cap).
    """
    skip_reason = check_thunderbird(path)
    if skip_reason:
        print(f"Skipping Thunderbird sampling: {skip_reason}")
        return []

    def stream_lines(tb_path):
        """Yield decoded lines of the first log member, capped at 20M."""
        with tarfile.open(tb_path, "r|gz") as tar:
            for member in tar:
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                for line_no, line_bytes in enumerate(fh, start=1):
                    if line_no > THUNDERBIRD_LINE_CAP:
                        return
                    yield line_bytes.decode("utf-8", errors="ignore")
                return  # only the first (log) member matters

    # Pass 1: gather metadata for sessionization and labels
    session_counts = Counter()
    labels = defaultdict(bool)
    timestamps = {}
    for line in stream_lines(path):
        parts = line.split(maxsplit=5)
        if len(parts) < 5:
            continue
        label = parts[0]
        try:
            timestamp = int(parts[1])
        except ValueError:
            continue
        node_id = parts[3]

        # Group Thunderbird into 60-second windows per node, like BGL
        window = timestamp // 60
        session_key = f"tbird_{node_id}_{window}"
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
    sampled_anom = get_stratified_subset(
        anom_keys, sample_size // 2, lambda k: timestamps[k], rng
    )
    sampled_norm = get_stratified_subset(
        norm_keys, sample_size // 2, lambda k: timestamps[k], rng
    )
    sampled_set = set(sampled_anom + sampled_norm)

    # Pass 2: extract actual lines for the sampled set
    sessions_lines = defaultdict(list)
    for line in stream_lines(path):
        parts = line.split(maxsplit=5)
        if len(parts) < 5:
            continue
        try:
            timestamp = int(parts[1])
        except ValueError:
            continue
        node_id = parts[3]
        window = timestamp // 60
        session_key = f"tbird_{node_id}_{window}"
        if session_key in sampled_set:
            # Drop the leading alert-category column: it is the ground-truth
            # label, not log content. Keeping it leaks labels to every
            # retriever (Thunderbird '-' = normal, anything else = anomaly) -
            # the same bug previously shipped and fixed in BGL.
            sessions_lines[session_key].append(line.partition(" ")[2] or line)

    results = []
    for k in sampled_anom + sampled_norm:
        lines = sessions_lines.get(k, [])
        if lines:
            results.append((k, "".join(lines), "Anomaly" if labels[k] else "Normal"))
    return results


def check_spirit(path: pathlib.Path) -> str | None:
    """Return None if spirit2.gz is present and checksum-verified, otherwise
    a human-readable reason to skip Spirit pairs (mirrors check_thunderbird)."""
    if not path.exists():
        return f"{path} is missing (download may still be in progress)"
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != SPIRIT_MD5:
        return (
            f"{path} md5 mismatch: expected {SPIRIT_MD5}, got {actual} "
            "(file incomplete or corrupt)"
        )
    return None


def sample_spirit(
    path: pathlib.Path, sample_size: int = 200, seed: int = 42
) -> list[tuple[str, str, str]]:
    """Sessionize and sample stratified Spirit logs (BGL/Thunderbird family).

    Modeled exactly on sample_thunderbird: streams the plain gzip in two
    passes without extracting to disk or holding lines in memory. Sessionizes
    per node into 60-second windows with a >=3 line minimum; the first
    whitespace-separated field is the ground-truth alert label column
    ("-" = normal, anything else = alert category, e.g. R_HDA_NR) and is
    STRIPPED from extracted text to avoid label leakage. Both passes are
    capped at the first SPIRIT_LINE_CAP (20M) lines for tractability (the
    split name "spirit_first20M" records the cap).

    Spirit line format (verified against CFDR hpc4/spirit2.gz):
        LABEL EPOCH DATE NODE Month Day HH:MM:SS src daemon[pid]: message
    so parts[0]=label, parts[1]=epoch seconds, parts[3]=node id - identical
    field positions to Thunderbird.
    """
    skip_reason = check_spirit(path)
    if skip_reason:
        print(f"Skipping Spirit sampling: {skip_reason}")
        return []

    def stream_lines(sp_path):
        """Yield decoded lines of the gzipped log, capped at 20M."""
        with gzip.open(sp_path, "rb") as fh:
            for line_no, line_bytes in enumerate(fh, start=1):
                if line_no > SPIRIT_LINE_CAP:
                    return
                yield line_bytes.decode("utf-8", errors="ignore")

    # Pass 1: gather metadata for sessionization and labels
    session_counts = Counter()
    labels = defaultdict(bool)
    timestamps = {}
    for line in stream_lines(path):
        parts = line.split(maxsplit=5)
        if len(parts) < 5:
            continue
        label = parts[0]
        try:
            timestamp = int(parts[1])
        except ValueError:
            continue
        node_id = parts[3]

        # Group Spirit into 60-second windows per node, like BGL/Thunderbird
        window = timestamp // 60
        session_key = f"spirit_{node_id}_{window}"
        session_counts[session_key] += 1
        if label != "-":
            labels[session_key] = True
        if session_key not in timestamps:
            timestamps[session_key] = timestamp

    # Filter sessions with length >= 3 to avoid tiny cases
    filtered_keys = [k for k, count in session_counts.items() if count >= 3]
    anom_keys = [k for k in filtered_keys if labels[k]]
    norm_keys = [k for k in filtered_keys if not labels[k]]
    print(
        f"Spirit (first {SPIRIT_LINE_CAP // 1_000_000}M lines): "
        f"{len(session_counts)} sessions, {len(filtered_keys)} with >=3 lines "
        f"({len(anom_keys)} anomalous / {len(norm_keys)} normal)"
    )

    rng = random.Random(seed)
    sampled_anom = get_stratified_subset(
        anom_keys, sample_size // 2, lambda k: timestamps[k], rng
    )
    sampled_norm = get_stratified_subset(
        norm_keys, sample_size // 2, lambda k: timestamps[k], rng
    )
    sampled_set = set(sampled_anom + sampled_norm)

    # Pass 2: extract actual lines for the sampled set
    sessions_lines = defaultdict(list)
    for line in stream_lines(path):
        parts = line.split(maxsplit=5)
        if len(parts) < 5:
            continue
        try:
            timestamp = int(parts[1])
        except ValueError:
            continue
        node_id = parts[3]
        window = timestamp // 60
        session_key = f"spirit_{node_id}_{window}"
        if session_key in sampled_set:
            # Drop the leading alert-category column: it is the ground-truth
            # label, not log content. Keeping it would leak labels to every
            # retriever (Spirit "-" = normal, anything else = anomaly), the
            # same leak previously found and fixed in BGL and Thunderbird.
            sessions_lines[session_key].append(line.partition(" ")[2] or line)

    results = []
    for k in sampled_anom + sampled_norm:
        lines = sessions_lines.get(k, [])
        if lines:
            results.append((k, "".join(lines), "Anomaly" if labels[k] else "Normal"))
    sampled_counts = Counter(label for _, _, label in results)
    print(
        f"Spirit sample: {len(results)} sessions "
        f"({sampled_counts.get('Anomaly', 0)} Anomaly / "
        f"{sampled_counts.get('Normal', 0)} Normal)"
    )
    return results


def run_transfer(
    index_data: list[tuple[str, str, str]],
    query_data: list[tuple[str, str, str]],
    pair_name: str,
    scorer: str = "ses",
) -> list[dict]:
    """Execute four-way cross-system transfer comparison.

    Adapted from loghub_eval.run_evaluation but WITHOUT the 80/20 split:
    index_data is the full index set (system A), query_data the full query
    set (system B).
    """
    split_name = f"{pair_name}[{scorer}]"
    print(
        f"\n--- Running transfer evaluation {split_name} "
        f"({len(index_data)} index / {len(query_data)} query cases) ---"
    )

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
    transfer_rows = []
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

        transfer_rows.append({
            "dataset": "LogHub",
            "split": split_name,
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

            transfer_rows.append({
                "dataset": "DIAGNOSTIC",
                "split": split_name,
                "method": f"{m}_alert",
                "macro_f1": reason,
                "label_hit_rate@1": "ALERT",
                "label_hit_rate@5": "ALERT",
                "label_hit_rate@10": "ALERT",
                "p50_ms": "0.000",
                "p95_ms": "0.000"
            })
            print(f"  [DIAGNOSTIC ALERT] {reason}")

    return transfer_rows


def append_transfer_rows(
    rows: list[dict], out_path: str | pathlib.Path = "reports/transfer_metrics.csv"
) -> None:
    """Append metric rows to a transfer metrics CSV (triage schema).

    Defaults to reports/transfer_metrics.csv (the original behavior)."""
    if not rows:
        return
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "split", "method", "macro_f1",
        "label_hit_rate@1", "label_hit_rate@5", "label_hit_rate@10",
        "p50_ms", "p95_ms",
    ]
    write_header = not out_path.exists()
    with out_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Appended {len(rows)} rows to {out_path}")


# Registry of samplable systems for --pairs. Each entry maps the system name
# (as written in a "A->B" pair spec) to (archive filename, sampler, display
# name used in the split string, optional integrity-check function).
SYSTEMS = {
    "HDFS": ("HDFS_v1.zip", sample_hdfs_stratified, "HDFS", None),
    "BGL": ("BGL.zip", sample_bgl_stratified, "BGL", None),
    "OpenStack": ("OpenStack.tar.gz", sample_openstack, "OpenStack", None),
    "Thunderbird": (
        "Thunderbird.tar.gz", sample_thunderbird, "thunderbird_first20M",
        check_thunderbird,
    ),
    "Spirit": ("spirit2.gz", sample_spirit, "spirit_first20M", check_spirit),
}


def run_named_pairs(pairs_spec, scorer, seed, index_size, query_size, out_path):
    """Run a comma-separated list of "A->B" transfer pairs (e.g.
    "BGL->Spirit,HDFS->Spirit") with an explicit seed, appending rows to
    out_path. Additive entry point used by --pairs; the default (no --pairs)
    code path in main() is unchanged."""
    raw_dir = pathlib.Path("data/raw/loghub_raw")
    all_rows = []
    sample_cache = {}  # (system, size, seed) -> sampled sessions

    def sample_system(name, size):
        key = (name, size, seed)
        if key in sample_cache:
            return sample_cache[key]
        filename, sampler, _display, check = SYSTEMS[name]
        path = raw_dir / filename
        if not path.exists():
            print(f"Skipping {name}: {path} is missing. Run fetch_datasets.py first.")
            data = []
        else:
            skip = check(path) if check else None
            if skip:
                print(f"Skipping {name}: {skip}")
                data = []
            else:
                print(f"Sampling {name} sessions (size={size}, seed={seed})...")
                data = sampler(path, sample_size=size, seed=seed)
                counts = Counter(label for _, _, label in data)
                print(
                    f"{name} class counts: {counts.get('Anomaly', 0)} Anomaly / "
                    f"{counts.get('Normal', 0)} Normal"
                )
        sample_cache[key] = data
        return data

    for pair in [p.strip() for p in pairs_spec.split(",") if p.strip()]:
        if "->" not in pair:
            print(f"Skipping malformed pair spec '{pair}' (expected 'A->B').")
            continue
        src, dst = (s.strip() for s in pair.split("->", 1))
        if src not in SYSTEMS or dst not in SYSTEMS:
            known = ", ".join(SYSTEMS)
            print(f"Skipping pair '{pair}': unknown system (known: {known}).")
            continue
        index_data = sample_system(src, index_size)
        query_data = sample_system(dst, query_size)
        if not index_data or not query_data:
            print(f"Skipping pair '{pair}': empty index or query sample.")
            continue
        pair_name = f"{SYSTEMS[src][2]}->{SYSTEMS[dst][2]}[seed{seed}]"
        all_rows.extend(run_transfer(index_data, query_data, pair_name, scorer=scorer))

    append_transfer_rows(all_rows, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-system transfer evaluation (T2-b)")
    parser.add_argument("--scorer", choices=["ses", "mdl", "surprisal"], default="ses")
    parser.add_argument("--index-size", type=int, default=800,
                        help="stratified sessions to index from system A")
    parser.add_argument("--query-size", type=int, default=200,
                        help="stratified sessions to query from system B")
    parser.add_argument("--pairs", default=None,
                        help="comma-separated 'A->B' pairs to run instead of the "
                             "default HDFS->OpenStack and BGL->Thunderbird pairs, "
                             "e.g. 'BGL->Spirit,HDFS->Spirit'")
    parser.add_argument("--seed", type=int, default=42,
                        help="sampling seed threaded into both samplers")
    parser.add_argument("--out", default="reports/transfer_metrics.csv",
                        help="CSV path to append metric rows to")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.pairs:
        run_named_pairs(
            args.pairs, args.scorer, args.seed,
            args.index_size, args.query_size, args.out,
        )
        return

    raw_dir = pathlib.Path("data/raw/loghub_raw")
    hdfs_zip = raw_dir / "HDFS_v1.zip"
    bgl_zip = raw_dir / "BGL.zip"
    openstack_tar = raw_dir / "OpenStack.tar.gz"
    thunderbird_tar = raw_dir / "Thunderbird.tar.gz"

    all_rows = []

    # Pair 1: HDFS -> OpenStack
    if not hdfs_zip.exists():
        print(f"Skipping HDFS->OpenStack: {hdfs_zip} is missing. Run fetch_datasets.py first.")
    elif not openstack_tar.exists():
        print(f"Skipping HDFS->OpenStack: {openstack_tar} is missing. Run fetch_datasets.py first.")
    else:
        print("Sampling HDFS sessions (index set)...")
        hdfs_index = sample_hdfs_stratified(hdfs_zip, sample_size=args.index_size, seed=args.seed)
        print("Sampling OpenStack sessions (query set)...")
        openstack_query = sample_openstack(openstack_tar, sample_size=args.query_size, seed=args.seed)
        all_rows.extend(
            run_transfer(hdfs_index, openstack_query, "HDFS->OpenStack", scorer=args.scorer)
        )

    # Pair 2: BGL -> Thunderbird (first 20M lines, see THUNDERBIRD_LINE_CAP)
    tbird_skip = check_thunderbird(thunderbird_tar)
    if not bgl_zip.exists():
        print(f"Skipping BGL->Thunderbird: {bgl_zip} is missing. Run fetch_datasets.py first.")
    elif tbird_skip:
        print(f"Skipping BGL->Thunderbird: {tbird_skip}")
    else:
        print("Sampling BGL sessions (index set)...")
        bgl_index = sample_bgl_stratified(bgl_zip, sample_size=args.index_size, seed=args.seed)
        print("Sampling Thunderbird sessions (query set, first 20M lines)...")
        tbird_query = sample_thunderbird(thunderbird_tar, sample_size=args.query_size, seed=args.seed)
        if tbird_query:
            all_rows.extend(
                run_transfer(bgl_index, tbird_query, "BGL->thunderbird_first20M", scorer=args.scorer)
            )
        else:
            print("Skipping BGL->Thunderbird: no Thunderbird sessions sampled.")

    append_transfer_rows(all_rows, args.out)


if __name__ == "__main__":
    main()
