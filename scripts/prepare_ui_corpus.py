"""One-time extraction of large labeled LogHub samples for the UI workbench.

Writes JSONL files (one {"id", "text", "label"} per line) to data/processed/
so the Gradio workbench can index thousands of real incidents without
re-scanning the raw zips on every load.

Usage: python3 -u scripts/prepare_ui_corpus.py [--hdfs-n 5000] [--bgl-n 2500]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from sma.eval.loghub_eval import sample_bgl_stratified, sample_hdfs_stratified

OUT_DIR = pathlib.Path("data/processed")


def write_jsonl(path: pathlib.Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sid, text, label in rows:
            fh.write(json.dumps({"id": sid, "text": text, "label": label}) + "\n")
    print(f"wrote {len(rows)} sessions -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs-n", type=int, default=5000)
    parser.add_argument("--bgl-n", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hdfs_zip = pathlib.Path("data/raw/loghub_raw/HDFS_v1.zip")
    bgl_zip = pathlib.Path("data/raw/loghub_raw/BGL.zip")

    if hdfs_zip.exists():
        print(f"sampling {args.hdfs_n} HDFS sessions (two full scans, takes minutes)...")
        rows = sample_hdfs_stratified(hdfs_zip, sample_size=args.hdfs_n, seed=args.seed)
        write_jsonl(OUT_DIR / "ui_corpus_hdfs.jsonl", rows)
    else:
        print(f"missing {hdfs_zip}; skipping HDFS")

    if bgl_zip.exists():
        print(f"sampling {args.bgl_n} BGL sessions...")
        rows = sample_bgl_stratified(bgl_zip, sample_size=args.bgl_n, seed=args.seed)
        write_jsonl(OUT_DIR / "ui_corpus_bgl.jsonl", rows)
    else:
        print(f"missing {bgl_zip}; skipping BGL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def sample_cfdr_haystack(
    gz_path: pathlib.Path,
    prefix: str,
    sample_size: int = 5000,
    anomaly_rate: float = 0.05,
    seed: int = 42,
    line_cap: int = 20_000_000,
    line_start: int = 0,
) -> list[tuple[str, str, str]]:
    """Needle-in-haystack sample from a CFDR-format syslog (Liberty/Spirit).

    Unlike the 50/50 eval samplers, this keeps a realistic class balance:
    sample_size sessions of which ~anomaly_rate are alert windows. Format is
    BGL-family: first whitespace field is the admin alert tag ('-' = normal),
    STRIPPED from stored text (label-leak discipline). Sessions are per-node
    60-second windows with >= 3 lines, scanned over the first line_cap lines
    (two-pass stream).
    """
    import gzip
    import random
    from collections import defaultdict

    counts: dict[str, int] = defaultdict(int)
    is_alert: dict[str, bool] = defaultdict(bool)
    with gzip.open(gz_path, "rt", errors="ignore") as fh:
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
            key = f"{prefix}_{parts[3]}_{window}"
            counts[key] += 1
            if parts[0] != "-":
                is_alert[key] = True

    eligible = [k for k, n in counts.items() if n >= 3]
    anom = sorted(k for k in eligible if is_alert[k])
    norm = sorted(k for k in eligible if not is_alert[k])
    rng = random.Random(seed)
    n_anom = min(len(anom), int(sample_size * anomaly_rate))
    n_norm = min(len(norm), sample_size - n_anom)
    chosen_anom = rng.sample(anom, n_anom)
    chosen_norm = rng.sample(norm, n_norm)
    chosen = set(chosen_anom) | set(chosen_norm)
    print(f"{prefix}: {len(eligible)} eligible sessions "
          f"({len(anom)} alert / {len(norm)} normal) -> sampling {n_anom} + {n_norm}")

    texts: dict[str, list[str]] = defaultdict(list)
    with gzip.open(gz_path, "rt", errors="ignore") as fh:
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
            key = f"{prefix}_{parts[3]}_{window}"
            if key in chosen and len(texts[key]) < 60:
                # Strip the alert-tag column (ground truth, not log content).
                texts[key].append(line.partition(" ")[2].rstrip())

    rows = []
    for key in chosen_anom + chosen_norm:
        lines = texts.get(key, [])
        if lines:
            rows.append((key, "\n".join(lines), "Anomaly" if is_alert[key] else "Normal"))
    rng.shuffle(rows)
    return rows
