#!/usr/bin/env python3
"""Agentic ontology routing (exploratory) — the missing piece of the universal
adapter's "route across domains" claim.

The shipped DomainRouter (sma/ontology/router.py) routes DETERMINISTICALLY by term-id
prefix (HP:->hpo) or an explicit domain label; it never reads the query. The
realistic cross-domain setting has neither: a user poses a case in natural language
(term NAMES, no ids, no domain tag) and the system must FIRST decide which curated
ontology to consult. This benchmark measures that routing step.

Mixed query set: from each of the five evaluated domains we sample cases (an
entity's noisy term-name set) with the domain label hidden. Three routers map each
query to one of the five ontologies:
  * LLM         — the agent (DeepSeek) is shown the query and the five domains and
                  picks one (the agentic capability the paper claims).
  * dense       — nearest domain by cosine of the query embedding to a domain
                  prototype (mean embedding of that domain's indexed case texts).
  * majority    — always predict the most frequent domain (chance floor).
Metric: routing accuracy (did it pick the gold domain?). End-to-end cross-domain QA
then composes this with the per-domain trustworthy-QA results (crossdomain_qa.py).

  python3 scripts/crossdomain_routing.py --per-domain 40 --mock     # free smoke
  python3 scripts/crossdomain_routing.py --per-domain 40
"""
from __future__ import annotations
import argparse, csv, importlib, json, pathlib, random, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DOMAINS = {"medicine": "sma.eval.agentic.arms.medicine",
           "genomics": "sma.eval.agentic.arms.discovery",
           "finance": "sma.eval.agentic.arms.finance",
           "cyber": "sma.eval.agentic.arms.cyber",
           "legal": "sma.eval.agentic.arms.legal"}
LABELS = list(DOMAINS)

ROUTE_SYSTEM = (
    "You are a routing assistant. You are given a case described by a list of "
    "feature terms. Decide which single knowledge domain it belongs to, from: "
    "medicine, genomics, finance, cyber, legal. Reply with STRICT one-line JSON "
    'and nothing else: {"domain": "<one of medicine|genomics|finance|cyber|legal>"}.'
)


def sample_queries(per_domain, seed=11):
    """Return (queries, prototypes): queries = [(domain, text)], prototypes =
    {domain: [indexed case texts]} for the dense router."""
    queries, protos = [], {}
    for dom, path in DOMAINS.items():
        mounted, records = importlib.import_module(path).load()
        graph = mounted.graph
        def nm(t): return (graph.terms[t].name if t in graph.terms else "") or t
        known = {e: sorted(x for x in ts if x in graph.terms) for e, ts in records.items()}
        known = {e: ts for e, ts in known.items() if ts}
        ids = sorted(known); rng = random.Random(seed); rng.shuffle(ids)
        proto_ids, q_ids = ids[:300], ids[300:300 + per_domain] or ids[:per_domain]
        protos[dom] = [" ".join(nm(t) for t in known[e]) for e in proto_ids]
        for e in q_ids:
            terms = known[e]
            keep = rng.sample(terms, min(6, len(terms)))
            queries.append((dom, ", ".join(nm(t) for t in keep)))
    return queries, protos


def route_llm(queries, mock):
    if mock:
        # deterministic stand-in: keyword vote (NO api spend) for the smoke test
        kw = {"medicine": ["seizure", "intellectual", "ataxia", "abnormal", "phenotype"],
              "genomics": ["binding", "activity", "process", "regulation", "cell"],
              "finance": ["asset", "income", "liability", "equity", "tax", "revenue"],
              "cyber": ["technique", "credential", "execution", "command", "lateral"],
              "legal": ["device", "method", "apparatus", "system", "circuit"]}
        out = []
        for _dom, text in queries:
            t = text.lower()
            best = max(LABELS, key=lambda d: sum(w in t for w in kw[d]))
            out.append(best)
        return out
    from sma.agent.llm import DeepSeekOrchestrator
    o = DeepSeekOrchestrator()
    out = []
    for i, (_dom, text) in enumerate(queries):
        try:
            r = o.complete([{"role": "system", "content": ROUTE_SYSTEM},
                            {"role": "user", "content": f"Case features: {text}"}],
                           max_tokens=20, temperature=0.0)
            try:
                d = str(json.loads(r.strip().strip("`")).get("domain", "")).lower()
            except Exception:
                d = next((x for x in LABELS if x in r.lower()), LABELS[0])
        except Exception as exc:  # transient network/API error on one call
            print(f"  [llm route call {i} failed: {type(exc).__name__}; falling back]", flush=True)
            d = LABELS[0]
        out.append(d if d in LABELS else LABELS[0])
    return out


def route_dense(queries, protos):
    from sentence_transformers import SentenceTransformer
    import numpy as np
    m = SentenceTransformer("BAAI/bge-small-en-v1.5")
    cents = {d: m.encode(protos[d], normalize_embeddings=True).mean(0) for d in LABELS}
    C = np.stack([cents[d] for d in LABELS])
    out = []
    for _dom, text in queries:
        q = m.encode([text], normalize_embeddings=True)[0]
        out.append(LABELS[int((C @ q).argmax())])
    return out


def acc(queries, preds):
    n = len(queries)
    return sum(1 for (g, _), p in zip(queries, preds) if g == p) / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=40)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    queries, protos = sample_queries(args.per_domain)
    print(f"mixed query set: {len(queries)} ({args.per_domain}/domain)", flush=True)
    maj = Counter(g for g, _ in queries).most_common(1)[0][0]
    results = {"llm": route_llm(queries, args.mock),
               "dense": route_dense(queries, protos),
               "majority": [maj] * len(queries)}
    rows = []
    for method, preds in results.items():
        a = acc(queries, preds)
        # per-domain recall
        per = {d: acc([(g, t) for (g, t) in queries if g == d],
                      [p for (g, _), p in zip(queries, preds) if g == d]) for d in LABELS}
        print(f"  {method:9s} routing acc={a:.3f}  per-domain={ {k: round(v,2) for k,v in per.items()} }", flush=True)
        rows.append({"method": method, "n": len(queries), "routing_accuracy": round(a, 4),
                     **{f"acc_{d}": round(per[d], 4) for d in LABELS}})
    out = ROOT / "reports/confirmatory/crossdomain_routing.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
