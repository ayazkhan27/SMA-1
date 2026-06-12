"""H3 mini-study: verifiable answers vs confabulation across memory modes and LLMs.

Runs a fixed question set over the prepared HDFS UI corpus. Half the questions
are answerable from session evidence; half are unanswerable or carry a false
premise. Every mode x LLM cell is scored mechanically for abstention; columns
for human correctness/confabulation ratings are left blank for review.

Usage:
  python3 -u scripts/h3_mini_study.py [--n-docs 5000] [--limit N]
      [--modes sma bm25 "dense rag" "knowledge graph" "context only"]
      [--llms local deepseek] [--k 5] [--out reports/h3_mini_study.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import re
import time

from sma.agent.comparison import MODES, ComparisonFramework

# (id, answerable, question). Answerable = the HDFS corpus plausibly contains
# supporting sessions. Unanswerable = false premise, wrong domain, beyond the
# session window, or outside what retrieval evidence can support.
QUESTIONS: list[tuple[str, bool, str]] = [
    ("A1", True, "Show me precedents where a block write died with an EOFException in receiveBlock mid-pipeline. Which component threw the exception?"),
    ("A2", True, "Find sessions where a PacketResponder terminated during block receipt. What happened immediately before the termination in those sessions?"),
    ("A3", True, "Pull sessions where the same block was received from multiple source addresses. Is that replication fan-in pattern normal in our history?"),
    ("A4", True, "Find sessions where the namenode allocated a block but the write pipeline subsequently hit an exception. What exception types appear in the precedents?"),
    ("A5", True, "Show me sessions where blocks were deleted as routine cleanup after serving. What does that lifecycle look like?"),
    ("A6", True, "Retrieve precedents of NameSystem.addStoredBlock blockMap updates following a successful receive - the normal full write lifecycle."),
    ("A7", True, "Find sessions with a java.io.IOException during writeBlock and tell me which remote addresses the transfers involved."),
    ("A8", True, "Find precedents where a block was added to the invalidSet. What tends to precede the invalidation within the same session?"),
    ("A9", True, "Retrieve sessions where 'Received block of size' completed normally end to end - the canonical healthy write."),
    ("A10", True, "Find sessions where a block transfer failed with a connection reset by peer. Which side logged the reset?"),
    ("U1", False, "Find the incidents where our ZooKeeper session expiry caused the namenode to crash-loop last quarter."),
    ("U2", False, "Show precedents where Kubernetes evicted our datanode pods under memory pressure and HDFS writes stalled."),
    ("U3", False, "What percentage of all block writes in our history failed on the first replica?"),
    ("U4", False, "For the blocks that failed with EOFException, how long did re-replication take to restore the target replication factor afterwards?"),
    ("U5", False, "Pull the GPU out-of-memory incidents from the training cluster that corrupted HDFS writes."),
    ("U6", False, "Find the incident where an unauthorized client wrote blocks while bypassing namenode ACL checks."),
    ("U7", False, "Compare our HDFS EOFException rate with the BGL machine-check rate over the same period."),
    ("U8", False, "Which on-call engineer acknowledged the under-replication pages, and how fast did they respond?"),
    ("U9", False, "When did we change dfs.replication from 2 to 3, and which incidents drove that configuration change?"),
    ("U10", False, "If we had enabled rack-aware placement earlier, which of these past pipeline failures would have been avoided?"),
]

ABSTAIN_PATTERNS = [
    r"(no|not|doesn'?t|does not|insufficient|cannot|can'?t|unable)[^.]{0,50}(evidence|determine|answer|support|contain|find|tell|say)",
    r"evidence (does not|doesn'?t|is insufficient|lacks)",
    r"not (present|available|recorded|contained) in the evidence",
    r"no (such|matching|relevant) (incident|precedent|evidence|session)",
    r"(beyond|outside) (the|this|these) (evidence|session|window)",
]
ABSTAIN_RE = re.compile("|".join(ABSTAIN_PATTERNS), re.IGNORECASE)


def detected_abstention(answer: str) -> bool:
    return bool(ABSTAIN_RE.search(answer))


def load_corpus(framework: ComparisonFramework, n_docs: int, seed: int = 7) -> None:
    path = pathlib.Path("data/processed/ui_corpus_hdfs.jsonl")
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    for row in rows[:n_docs]:
        framework.add_document(row["text"], adapter_id="logs", label=row.get("label", ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-docs", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=0, help="limit number of questions (0 = all)")
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--llms", nargs="+", default=["local", "deepseek"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="reports/h3_mini_study.csv")
    args = parser.parse_args()

    framework = ComparisonFramework()
    print(f"indexing {args.n_docs} HDFS sessions...", flush=True)
    load_corpus(framework, args.n_docs)

    questions = QUESTIONS[: args.limit] if args.limit else QUESTIONS
    fieldnames = [
        "question_id", "answerable", "mode", "llm", "auto_abstained",
        "n_evidence", "evidence_anomaly", "evidence_normal", "latency_s",
        "answer", "human_correct", "human_confabulated", "notes",
    ]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_cells = len(questions) * len(args.modes) * len(args.llms)
    done = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for qid, answerable, question in questions:
            for mode in args.modes:
                for llm in args.llms:
                    t0 = time.perf_counter()
                    result = framework.ask(question, mode, adapter_id="logs", k=args.k, llm=llm)
                    latency = time.perf_counter() - t0
                    labels = [row.get("label") for row in result.evidence if row.get("label")]
                    writer.writerow(
                        {
                            "question_id": qid,
                            "answerable": answerable,
                            "mode": result.mode,
                            "llm": llm,
                            "auto_abstained": detected_abstention(result.answer),
                            "n_evidence": len(result.evidence),
                            "evidence_anomaly": labels.count("Anomaly"),
                            "evidence_normal": labels.count("Normal"),
                            "latency_s": f"{latency:.2f}",
                            "answer": result.answer.replace("\n", " ")[:1500],
                            "human_correct": "",
                            "human_confabulated": "",
                            "notes": "",
                        }
                    )
                    fh.flush()
                    done += 1
                    print(f"[{done}/{n_cells}] {qid} {result.mode} {llm} "
                          f"abstain={detected_abstention(result.answer)} {latency:.1f}s", flush=True)

    # Summary: abstention discipline per mode x llm
    print("\nSummary (correct abstention behavior):")
    import collections
    cells = collections.defaultdict(lambda: {"good": 0, "n": 0})
    for row in csv.DictReader(out.open(encoding="utf-8")):
        key = (row["mode"], row["llm"])
        answerable = row["answerable"] == "True"
        abstained = row["auto_abstained"] == "True"
        # Good behavior: abstain on unanswerable, answer on answerable.
        cells[key]["good"] += int(abstained == (not answerable))
        cells[key]["n"] += 1
    for (mode, llm), c in sorted(cells.items()):
        print(f"  {mode:16s} x {llm:8s}: {c['good']}/{c['n']} cells behaved correctly (auto-detected)")
    print(f"\nwrote {out} - human_correct/human_confabulated columns await review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
