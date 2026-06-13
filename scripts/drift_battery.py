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

class _ExtractCache:
    """Wraps an LLM and memoizes EXTRACTION calls (system prompt contains
    'Extract') keyed by the turn text. This (a) guarantees extraction is truly
    held constant across backends — rag-notes and sma get byte-identical facts
    for the same turn — and (b) halves token cost by not extracting the same
    turn twice. Answer calls (per-backend, different context) are never cached."""
    def __init__(self, llm):
        self.llm = llm
        self._cache: dict[str, str] = {}
        self.hits = 0
        self.misses = 0
    def complete(self, messages, max_tokens=600, temperature=0.0):
        if messages and "Extract" in messages[0].get("content", ""):
            key = messages[-1]["content"]
            if key in self._cache:
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            out = self.llm.complete(messages, max_tokens=max_tokens, temperature=temperature)
            self._cache[key] = out
            return out
        return self.llm.complete(messages, max_tokens=max_tokens, temperature=temperature)

def get_llm(smoke):
    if smoke:
        return _Stub()
    from sma.agent.llm import DeepSeekOrchestrator
    return _ExtractCache(DeepSeekOrchestrator())

FIELDS = ["qid", "category", "method", "correct", "drift", "flagged"]

def run(limit, smoke, full):
    rows_path = OUT / ("t5_rows_smoke.csv" if smoke else "t5_rows.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    data = FULL if full else ORACLE
    insts = load_instances(data)
    if smoke: insts = insts[:5]
    elif limit: insts = insts[:limit]
    # Resume: rows are written per-instance so a crash never loses prior work.
    # Re-launching skips instances already in the CSV (single-shot per qid).
    rows: list[dict] = []
    done: set[str] = set()
    if rows_path.exists():
        rows = list(csv.DictReader(rows_path.open()))
        for r in rows:
            r["correct"] = float(r["correct"]); r["drift"] = int(r["drift"]); r["flagged"] = int(r["flagged"])
        done = {r["qid"] for r in rows}
        remaining = [i for i in insts if i.question_id not in done]
        if not remaining:
            print(f"all {len(done)} instances already done in {rows_path}; computing stats only", flush=True)
        else:
            print(f"resuming: {len(done)} done, {len(remaining)} remaining", flush=True)
        insts = remaining
    llm = get_llm(smoke)
    new_file = not rows_path.exists()
    fh = rows_path.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
    for n, inst in enumerate(insts, 1):
        for b in make_backends(llm):
            b.reset()
            for s in inst.sessions:
                b.ingest(s)
            r = b.query(inst.question)
            row = {"qid": inst.question_id, "category": inst.category,
                   "method": b.name, "correct": grade_answer(r.answer, inst.answer),
                   "drift": int(inst.is_drift), "flagged": int(r.drift_flagged)}
            rows.append(row); writer.writerow(row)
        fh.flush()  # checkpoint after every instance
        print(f"[{len(done)+n}] {inst.question_id} done", flush=True)
    fh.close()
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
