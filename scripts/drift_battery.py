"""Single-shot drift battery (Phase 4a). Mirrors confirmatory_battery discipline.

  python3 scripts/drift_battery.py --smoke           # 5 instances, deterministic stub, no network
  python3 scripts/drift_battery.py --limit 20        # cost pilot (real DeepSeek), oracle data
  python3 scripts/drift_battery.py                   # full single-shot, oracle data
  python3 scripts/drift_battery.py --full            # full, 277MB realistic-haystack data
"""
from __future__ import annotations
import argparse, csv, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sma.eval.longmemeval import load_instances, grade_answer
from sma.eval.memory_backends.context_only import ContextOnly
from sma.eval.memory_backends.rag_notes import RagNotes
from sma.eval.memory_backends.sma_memory import SmaMemory
from sma.eval.stats import paired_bootstrap, holm_bonferroni, cliffs_delta

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "longmemeval"
ORACLE = DATA_DIR / "longmemeval_oracle.json"
FULL = DATA_DIR / "longmemeval_s_cleaned.json"
OUT = ROOT / "reports" / "confirmatory"

def make_backends(llm):
    backends = [ContextOnly(llm), RagNotes(llm), SmaMemory(llm)]
    try:
        from sma.eval.memory_backends.zep_graphiti import ZepGraphiti, ZEP_AVAILABLE
        if ZEP_AVAILABLE:
            backends.append(ZepGraphiti(llm))
    except Exception as exc:
        print(f"[zep] skipped: {exc}", flush=True)
    return backends

class _Stub:
    """Deterministic offline LLM for --smoke: extracts a crude fact, echoes
    the most recent retrieved item as the answer. No network."""
    def complete(self, messages, max_tokens=600, temperature=0.0):
        sysmsg = messages[0]["content"]
        user = messages[-1]["content"]
        if "Extract" in sysmsg:
            # one short fact from the message
            frag = user.strip().replace('"', "'")[:50]
            return '["' + frag + '"]'
        # answer: echo the last memory line if present, else 'unknown'
        if "Memory:" in user:
            lines = [l[2:] for l in user.splitlines() if l.startswith("- ")]
            return lines[-1][:60] if lines else "unknown"
        return "unknown"

def get_llm(smoke):
    if smoke:
        return _Stub()
    from sma.agent.llm import DeepSeekOrchestrator
    return DeepSeekOrchestrator()

def run(limit, smoke, full):
    rows_path = OUT / ("t5_rows_smoke.csv" if smoke else "t5_rows.csv")
    if rows_path.exists() and not smoke:
        sys.exit(f"REFUSE: {rows_path} exists (single-shot). Log a rerun in STATUS.md and delete to force.")
    data = FULL if full else ORACLE
    insts = load_instances(data)
    if smoke: insts = insts[:5]
    elif limit: insts = insts[:limit]
    llm = get_llm(smoke)
    rows = []
    for inst in insts:
        for b in make_backends(llm):
            b.reset()
            for s in inst.sessions:
                b.ingest(s)
            r = b.query(inst.question)
            rows.append({"qid": inst.question_id, "category": inst.category,
                         "method": b.name, "correct": grade_answer(r.answer, inst.answer),
                         "drift": int(inst.is_drift), "flagged": int(r.drift_flagged)})
        print(f"[{len(rows)}] {inst.question_id} done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with rows_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["qid", "category", "method", "correct", "drift", "flagged"])
        w.writeheader(); w.writerows(rows)
    methods = sorted({r["method"] for r in rows})
    drift_rows = [r for r in rows if r["drift"]]
    by = lambda m: [r["correct"] for r in drift_rows if r["method"] == m]
    if "sma" in methods:
        pvals, summary = {}, []
        for m in methods:
            if m == "sma": continue
            if not by(m) or not by("sma"): continue
            bs = paired_bootstrap(by("sma"), by(m))
            pvals[m] = bs["p_value"]
            summary.append({"baseline": m, "delta": bs["delta"],
                            "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
                            "cliffs": cliffs_delta(by("sma"), by(m))})
        if pvals:
            holm = holm_bonferroni(pvals)
            for s in summary: s["p_holm"] = holm[s["baseline"]]
            spath = OUT / ("t5_stats_smoke.csv" if smoke else "t5_stats.csv")
            with spath.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["baseline", "delta", "ci_low", "ci_high", "cliffs", "p_holm"])
                w.writeheader(); w.writerows(summary)
        print("drift-category accuracy:", {m: round(sum(by(m))/max(len(by(m)),1), 3) for m in methods})
    print(f"wrote {rows_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--full", action="store_true", help="use longmemeval_s_cleaned (277MB); default oracle")
    a = ap.parse_args()
    run(a.limit, a.smoke, a.full)
