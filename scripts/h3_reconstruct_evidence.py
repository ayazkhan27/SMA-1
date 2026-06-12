"""Reconstruct the H3 mini-study evidence sets for judging.

Re-runs the deterministic retrieval from scripts/h3_mini_study.py (same corpus,
same seed, same k) and dumps the 100 (question, mode) evidence sets to JSON so
the judging pass can compare answers against the evidence the LLMs saw.

Caveat: the logs encoder changed (v0.2.0) after the study ran, so SMA-mode
evidence may differ slightly from what answers actually saw.

Usage:
  python3 -u scripts/h3_reconstruct_evidence.py [--out /tmp/h3_evidence.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from h3_mini_study import QUESTIONS, load_corpus  # noqa: E402

from sma.agent.comparison import MODES, ComparisonFramework  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-docs", type=int, default=5000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="/tmp/h3_evidence.json")
    args = parser.parse_args()

    framework = ComparisonFramework()
    print(f"indexing {args.n_docs} HDFS sessions...", flush=True)
    load_corpus(framework, args.n_docs)

    out: dict[str, dict] = {}
    for qid, answerable, question in QUESTIONS:
        for mode in MODES:
            canonical, evidence = framework.evidence_for(
                question, mode, adapter_id="logs", k=args.k
            )
            out[f"{qid}|{canonical}"] = {
                "question_id": qid,
                "answerable": answerable,
                "question": question,
                "mode": canonical,
                "evidence": [
                    {
                        "source_id": row.get("source_id"),
                        "label": row.get("label"),
                        "score": row.get("score"),
                        "provenance": row.get("provenance"),
                        "mode_detail": row.get("mode_detail"),
                        "text": row.get("text"),
                    }
                    for row in evidence
                ],
            }
            print(f"{qid} {canonical}: {len(evidence)} items", flush=True)

    path = pathlib.Path(args.out)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
