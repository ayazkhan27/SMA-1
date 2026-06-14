#!/usr/bin/env python3
"""Cross-domain trustworthy-QA (exploratory) — replicate the Phase-5 medicine
LLM-QA paradigm on a NON-medicine domain to test whether the SMA-grounded agent's
accuracy/selectivity/attribution advantage over dense RAG generalises beyond the
single pre-registered domain.

DESIGN (honest scope):
  * This is EXPLORATORY, not part of pre-registration v2 (which covered medicine).
  * It does NOT modify the frozen ``sma/eval/agentic_qa`` package. It reuses the
    frozen QAAgent, metrics, calibration logic, and memories unchanged, and only
    (a) supplies a DOMAIN-NEUTRAL prompt via a thin subclass that overrides the two
    prompt-building methods (the validated retrieval / calibration-gate / parsing /
    result-assembly logic is inherited verbatim), and (b) builds the question pools
    from an existing domain arm's records (e.g. genomics = ``discovery.load()``),
    mirroring ``agentic_qa.pools.build_pools``.
  * The medicine prompt says "diagnose the disease"; for any domain we instead say
    "choose the candidate entity whose features best match the case", which is the
    same selection task without the medical framing.

Holds the LLM (DeepSeek, temperature 0) and prompt FIXED; swaps only the memory
(closed-book / dense-RAG / SMA). Calibrates the cite-or-abstain gate on a disjoint
split (retrieval-only, no LLM spend), exactly as the medicine driver does.

  python3 scripts/crossdomain_qa.py --arm genomics --memory sma --mock        # free smoke
  python3 scripts/crossdomain_qa.py --arm genomics --memory sma --n-answerable 20 --n-held 20
"""
from __future__ import annotations
import argparse, csv, importlib, pathlib, random, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sma.eval.agentic import IndexItem, Query, SmaMemory, DenseMemory
from sma.eval.agentic_qa.agent import QAAgent, MockLLM, ABSTAIN
from sma.eval.agentic_qa.pools import QAItem
from sma.eval.agentic_qa.metrics import (
    accuracy, citation_faithfulness, abstention, novelty_f1, novelty_recall,
    grounding_auroc, auroc,
)

ARMS = {"genomics": "sma.eval.agentic.arms.discovery",
        "finance": "sma.eval.agentic.arms.finance",
        "cyber": "sma.eval.agentic.arms.cyber",
        "legal": "sma.eval.agentic.arms.legal"}

# ---- domain-neutral prompts (the ONLY substantive change vs the medicine agent) --
NEUTRAL_SYSTEM = (
    "You are a careful classification assistant. You are given a case described by a "
    "set of features and a numbered list of candidate entities retrieved from a "
    "grounded knowledge base, each with a few of its characteristic features. Choose "
    "the single candidate whose characteristic features best match the case. Answer "
    "ONLY when a candidate genuinely grounds the case; if none of the candidates fit, "
    "abstain. Reply with STRICT one-line JSON and nothing else: "
    '{"choice": <candidate number, or 0 for none / abstain>}.'
)
NEUTRAL_CLOSED_SYSTEM = (
    "You are a careful classification assistant. You are given a case described by a "
    "set of features and no external knowledge. Name the single most likely entity, "
    "or abstain if you are not confident. Reply with STRICT one-line JSON and nothing "
    'else: {"diagnosis": "<entity name, or ABSTAIN>"}.'
)


class NeutralQAAgent(QAAgent):
    """QAAgent with domain-neutral prompts. Overrides ONLY the two prompt builders;
    every other behaviour (retrieve, calibrated gate, candidate rendering, JSON
    parsing, result assembly, metrics) is inherited verbatim from the frozen agent."""

    def _answer_grounded(self, item: QAItem) -> dict:
        query = Query(item.case_terms, item.case_text)
        retrieved = self.memory.retrieve(query, self.k)
        confidence = retrieved[0].confidence if retrieved else 0.0
        grounding_score = retrieved[0].score if retrieved else 0.0
        if self.score_threshold is not None and grounding_score < self.score_threshold:
            return self._result(item, abstained=True, pred_id=None, answer=ABSTAIN,
                                 novelty_flag=True, confidence=confidence,
                                 grounding_score=grounding_score)
        candidates_text, keys = self._render_candidates(retrieved)
        if self.score_threshold is not None:
            novelty_flag = False
        else:
            novelty_flag = bool(self.memory.novelty(query) > self.novelty_threshold)
        user = (
            f"Case features:\n{item.case_text}\n\n"
            f"Candidate entities:\n{candidates_text or '(none retrieved)'}\n\n"
            "Rule: choose the candidate whose characteristic features best match the "
            "case; answer only if a candidate genuinely grounds the case, otherwise "
            "choose 0 to abstain.\n"
            'Reply with STRICT one-line JSON: {"choice": <candidate number or 0>}.'
        )
        reply = self.llm.complete(
            [{"role": "system", "content": NEUTRAL_SYSTEM}, {"role": "user", "content": user}],
            max_tokens=600, temperature=0.0)
        choice = self._parse_choice(reply, n_candidates=len(keys))
        if choice == 0:
            pred_id, answer, abstained = None, ABSTAIN, True
        else:
            pred_id = keys[choice - 1]
            answer = self.key_to_name.get(pred_id, pred_id)
            abstained = False
        return self._result(item, abstained=abstained, pred_id=pred_id, answer=answer,
                            novelty_flag=novelty_flag, confidence=confidence,
                            grounding_score=grounding_score)

    def _answer_closed_book(self, item: QAItem) -> dict:
        user = (
            f"Case features:\n{item.case_text}\n\n"
            "Name the single most likely entity, or abstain if not confident.\n"
            'Reply with STRICT one-line JSON: {"diagnosis": "<entity name or ABSTAIN>"}.'
        )
        reply = self.llm.complete(
            [{"role": "system", "content": NEUTRAL_CLOSED_SYSTEM}, {"role": "user", "content": user}],
            max_tokens=600, temperature=0.0)
        from sma.eval.agentic_qa.agent import _parse_json_object
        obj = _parse_json_object(reply) or {}
        name = str(obj.get("diagnosis", "")).strip()
        abstained = (not name) or name.upper() == ABSTAIN
        return self._result(item, abstained=abstained, pred_id=None,
                            answer=(ABSTAIN if abstained else name), novelty_flag=False,
                            confidence=0.0, grounding_score=None)


def build_pools(mounted, records, *, seed=7, n_index=1500, n_answerable=120,
                n_held=120, n_calib=60):
    """Mirror of agentic_qa.pools.build_pools but over an arbitrary arm's records
    (entity_id -> set(term_id)) with a DOMAIN-NEUTRAL case rendering."""
    graph = mounted.graph
    def tname(t): return (graph.terms[t].name if t in graph.terms else "") or t
    parents = {tid: tuple(term.parents) for tid, term in graph.terms.items()}
    known = {e: sorted(t for t in terms if t in graph.terms)
             for e, terms in records.items()}
    known = {e: ts for e, ts in known.items() if ts}
    eligible = sorted(known)
    rng = random.Random(seed); rng.shuffle(eligible)
    indexed_ids, held_ids = eligible[:n_index], eligible[n_index:]
    index_items = [IndexItem(key=e, term_ids=frozenset(known[e]),
                             text=" ".join(tname(t) for t in known[e]), meta={"name": e})
                   for e in indexed_ids]
    noise_pool = sorted({t for e in indexed_ids for t in known[e]})

    def make_case(terms):
        keep = rng.sample(terms, min(5, len(terms)))
        q = []
        for t in keep:
            cur = t
            for _ in range(rng.choice([0, 0, 1, 1, 2])):
                ps = parents.get(cur)
                if ps:
                    cur = rng.choice(sorted(ps))
            q.append(cur)
        if noise_pool:
            q += rng.sample(noise_pool, min(3, len(noise_pool)))
        return frozenset(q), ", ".join(tname(t) for t in q)

    def qitems(ids, n, *, answerable):
        out = []
        for e in ids[:n]:
            ct, txt = make_case(known[e])
            out.append(QAItem(case_text=txt, case_terms=ct, gold_id=e, gold_name=e,
                              answerable=answerable, novel=not answerable))
        return out

    answerable = qitems(indexed_ids, n_answerable, answerable=True)
    novel = qitems(held_ids, n_held, answerable=False)
    return {"index_items": index_items, "answerable": answerable, "novel": novel,
            "calib_answerable": qitems(indexed_ids[n_answerable:], n_calib, answerable=True),
            "calib_ook": qitems(held_ids[n_held:], n_calib, answerable=False)}


def calibrate_threshold(memory, calib_items, k):
    pos, neg = [], []
    for it in calib_items:
        r = memory.retrieve(Query(it.case_terms, it.case_text), k)
        (pos if it.answerable else neg).append(r[0].score if r else 0.0)
    if not pos or not neg:
        return None, None
    observed = sorted(set(pos + neg))
    cands = [observed[0] - 1.0] + [(observed[i] + observed[i + 1]) / 2
                                   for i in range(len(observed) - 1)] + [observed[-1] + 1.0]
    best_t, best_j = cands[0], -2.0
    for t in cands:
        j = sum(s >= t for s in pos) / len(pos) - sum(s >= t for s in neg) / len(neg)
        if j > best_j:
            best_j, best_t = j, t
    return best_t, auroc(pos, neg)


def make_llm(mock):
    if mock:
        return MockLLM()
    from sma.agent.llm import DeepSeekOrchestrator
    return DeepSeekOrchestrator()


def compute_metrics(results):
    abst = abstention(results); nf1 = novelty_f1(results)
    return {"accuracy": accuracy(results),
            "citation_faithfulness": citation_faithfulness(results),
            "abstain_recall": abst["abstain_recall"], "false_abstain": abst["false_abstain"],
            "selective_accuracy": abst["selective_accuracy"], "aurc": abst["aurc"],
            "grounding_auroc": grounding_auroc(results),
            "novelty_recall": novelty_recall(results), "novelty_f1": nf1["f1"]}


def run(arm, memory_name, *, mock, n_answerable, n_held, k, n_index, n_calib):
    mod = importlib.import_module(ARMS[arm])
    mounted, records = mod.load()
    pools = build_pools(mounted, records, n_index=n_index, n_answerable=n_answerable,
                        n_held=n_held, n_calib=n_calib)
    idx = pools["index_items"]
    calib = list(pools["calib_answerable"]) + list(pools["calib_ook"])
    print(f"[{arm}/{memory_name}] index={len(idx)} answerable={len(pools['answerable'])} "
          f"held={len(pools['novel'])} calib={len(calib)}", flush=True)
    key_to_name = {it.key: it.meta.get("name", it.key) for it in idx}
    key_to_terms = {it.key: it.term_ids for it in idx}
    memory = None
    if memory_name == "dense":
        memory = DenseMemory(); memory.index(idx)
    elif memory_name == "sma":
        memory = SmaMemory(mounted); memory.index(idx)
    thr, cauroc = (None, None)
    if memory is not None and calib:
        thr, cauroc = calibrate_threshold(memory, calib, k)
        print(f"  calibrated threshold={thr} (calib AUROC {cauroc})", flush=True)
    agent = NeutralQAAgent(make_llm(mock), memory, key_to_name=key_to_name,
                          key_to_terms=key_to_terms, k=k, score_threshold=thr)
    items = pools["answerable"] + pools["novel"]
    results = [agent.answer(it) for it in items]
    return {"memory": memory_name, "metrics": compute_metrics(results),
            "score_threshold": thr, "calib_auroc": cauroc, "n": len(items)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), default="genomics")
    ap.add_argument("--memory", choices=("none", "dense", "sma"), default="sma")
    ap.add_argument("--all-memories", action="store_true", help="run none+dense+sma")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--n-answerable", type=int, default=120)
    ap.add_argument("--n-held", type=int, default=120)
    ap.add_argument("--n-index", type=int, default=1500)
    ap.add_argument("--n-calib", type=int, default=60)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    mems = ["none", "dense", "sma"] if args.all_memories else [args.memory]
    rows = []
    for m in mems:
        r = run(args.arm, m, mock=args.mock, n_answerable=args.n_answerable,
                n_held=args.n_held, k=args.k, n_index=args.n_index, n_calib=args.n_calib)
        met = r["metrics"]
        print(f"  acc={met['accuracy']:.3f} cite={met['citation_faithfulness']} "
              f"abstain_recall={met['abstain_recall']:.3f} groundingAUROC={met['grounding_auroc']} "
              f"noveltyF1={met['novelty_f1']:.3f}", flush=True)
        rows.append({"arm": args.arm, "memory": m, "n": r["n"],
                     "score_threshold": r["score_threshold"], **met})
    out = ROOT / "reports/confirmatory" / f"crossdomain_qa_{args.arm}.csv"
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
