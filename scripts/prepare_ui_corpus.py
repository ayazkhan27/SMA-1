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
