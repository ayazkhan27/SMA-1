"""Experiment D1 - mechanism-stack ablation (Nature MI revision).

Isolates what each SMA mechanism contributes on the agentic memory-swap
benchmark, answering "what does structure-mapping add beyond IC subsumption?"
and testing the paper's untested prediction that relation-rich ontologies
benefit more from the typed-relation rung.

A LADDER of configurations, holding the harness (queries, index, seeds,
rare-slice definition) fixed and swapping ONLY the retrieval mechanism:

  rung 0  lexical-only     BM25 over term-name documents (baseline floor).
  rung 1  +is-a ascension  SMA, MatchConfig(delta=2, scorer="ses"): IC OFF
                           (unit weights), is-a-only graph (typed relations
                           suppressed).
  rung 2  +IC weighting    SMA, MatchConfig(delta=2, scorer="surprisal"): IC ON
                           (corpus surprisal costs), is-a-only graph.
  rung 3  +typed relations SMA, full frozen config (surprisal/max/gamma0.25/
                           rho0.95/delta2) WITH typed/higher-order relations
                           mounted.

Toggles (verified against frozen code, never modified):
  * IC off vs on  -> MatchConfig.scorer "ses" vs "surprisal". For "surprisal"
    the MacFacIndex lazily derives corpus_costs (-log2 p per functor); for
    "ses" bound_costs/cost_fn are None => unit weights (sma/index/macfac.py:74-79,
    sma/match/engine.py:24).
  * typed relations on vs off -> MountedOntology.build_case iterates
    graph.typed_relations() (sma/ontology/mount.py:58). A graph whose terms all
    carry relations=() yields no higher-order statements; is-a parents are kept,
    so the ascension lattice is identical (sma/ontology/graph.py:52-58).
  * is-a ascension -> MatchConfig.delta (delta=0 off, delta=2 two hops).

The frozen run_oneshot is imported and reused read-only. It is run ONCE per
domain with all four rung-memories present, so every rung sees byte-identical
queries / index / rare-slice. Point estimates (rare-slice tail top-5) come
straight from its result. For the adjacent-rung paired bootstrap on the RARE
slice (run_oneshot's built-in `primary` bootstrap is on the all-slice only), we
reconstruct per-query rare-slice top-5 correctness by replaying the harness's
deterministic query loop, and HARD-ASSERT the reconstructed point estimate
matches run_oneshot's reported value to 1e-9. If it diverges, the run aborts
rather than reporting a number we cannot trust.

Output:
  reports/confirmatory/ablation_mechanism.csv
  reports/confirmatory/ablation_mechanism.log
"""

from __future__ import annotations

import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sma.eval.agentic import BM25Memory, SmaMemory, run_oneshot
from sma.eval.agentic.harness import _build_ic  # frozen IC machinery (reuse, no copy)
from sma.eval.agentic.metrics import ABSENT_RANK
from sma.eval.stats import cliffs_delta, paired_bootstrap
from sma.ontology import OntologyGraph, Term, mount
from sma.match.types import MatchConfig

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/confirmatory"

# Same seeds and index size as the committed agentic runs (scripts/agentic_suite.py
# defaults: seeds=(7,17,23) baked into run_oneshot; n_index=2000, n_query=120).
SEEDS = (7, 17, 23)
# Tractable ablation scale: the full n_index=2000 x 3 seeds x 4 rungs x 2 domains
# is compute-bound (SME alignment cost is super-linear in index size); n_index=400
# runs in minutes per the pilot and the relative mechanism contributions (the point
# of the ablation) are robust to index size. Reported honestly at this scale.
N_INDEX = 400
N_QUERY = 90
HOLDOUT_FRAC = 0.1

DOMAINS = {
    "genomics": "sma.eval.agentic.arms.discovery",  # GO: relation-rich
    "medicine": "sma.eval.agentic.arms.medicine",   # HPO: pure is-a control
}

# Frozen production MatchConfig (rung 3) fields.
RHO = 0.95
GAMMA = 0.25
DELTA = 2
NORM = "max"


def strip_relations(graph: OntologyGraph) -> OntologyGraph:
    """Return a copy of ``graph`` with every term's typed relations removed.

    is-a parents are preserved, so the mounted ascension lattice is unchanged;
    only the higher-order typed-relation statements are suppressed.
    """
    terms = {
        tid: Term(id=t.id, name=t.name, parents=t.parents, relations=(), obsolete=t.obsolete)
        for tid, t in graph.terms.items()
    }
    return OntologyGraph(name=f"{graph.name}_isaonly", version=graph.version, terms=terms)


class RecordingMemory:
    """Transparent proxy that records each retrieve() call's ranked key list.

    Wraps a real :class:`Memory` so the frozen harness drives it exactly as
    usual; the only side effect is appending the returned (key, rank) list to
    ``self.calls`` in call order. This lets us recover per-query ranks from the
    single run_oneshot pass without a second (expensive) SMA retrieval pass and
    without copying any harness logic.
    """

    def __init__(self, inner, name: str):
        self.inner = inner
        self.name = name
        self.calls: list[list[tuple[str, int]]] = []

    def index(self, items):
        return self.inner.index(items)

    def retrieve(self, query, k):
        res = self.inner.retrieve(query, k)
        self.calls.append([(r.key, r.rank) for r in res])
        return res

    def novelty(self, query):
        return self.inner.novelty(query)


def rung_memories(graph: OntologyGraph, record: bool = False):
    """Build the four ladder memories. Returns list[(rung, config_label, memory)].

    With ``record=True`` each memory is wrapped in :class:`RecordingMemory` so
    its per-query ranked outputs are captured during run_oneshot.
    """
    stripped = strip_relations(graph)

    m_isa = mount(stripped, MatchConfig(delta=DELTA, rho=RHO, scorer="ses", gamma=GAMMA, normalization=NORM))
    m_ic = mount(stripped, MatchConfig(delta=DELTA, rho=RHO, scorer="surprisal", gamma=GAMMA, normalization=NORM))
    m_full = mount(graph, MatchConfig(delta=DELTA, rho=RHO, scorer="surprisal", gamma=GAMMA, normalization=NORM))

    specs = [
        (0, "lexical-only(bm25)", BM25Memory(), "bm25"),
        (1, "isa-ascension delta2 scorer=ses IC-off (isa-only)", SmaMemory(m_isa), "sma_r1_isa"),
        (2, "+IC scorer=surprisal delta2 (isa-only)", SmaMemory(m_ic), "sma_r2_ic"),
        (3, "+typed-relations surprisal/max/g0.25/r0.95/d2 (full)", SmaMemory(m_full), "sma_r3_full"),
    ]
    out = []
    for rung, label, mem, name in specs:
        if record:
            mem = RecordingMemory(mem, name)
        else:
            mem.name = name
        out.append((rung, label, mem))
    return out


def query_plan(graph, records):
    """Reproduce the harness's deterministic per-query (gold, is_novel, rare) plan.

    This is the bookkeeping HALF of the harness loop with NO retrieval: it yields
    the exact ordered sequence of queries the harness issues to every memory, so
    recorded per-call ranks can be paired with their gold key and rare flag. The
    IC build is imported from the harness (sma.eval.agentic.harness._build_ic);
    only the seed/shuffle/make_qspec stepping is mirrored. Returns a flat list of
    ``(gold_key, is_novel, rare)`` in harness call order, and is hard-checked
    against run_oneshot's reported rare top-5 in main().
    """
    parents = {tid: tuple(t.parents) for tid, t in graph.terms.items()}
    eligible = sorted(
        eid for eid, terms in records.items() if any(t in graph.terms for t in terms)
    )
    plan: list[tuple[str, bool, bool]] = []
    for seed in SEEDS:
        rng = random.Random(seed)
        ids = list(eligible)
        rng.shuffle(ids)
        pool = ids[:N_INDEX]

        pool_sorted = sorted(pool)
        rng.shuffle(pool_sorted)
        n_holdout = int(round(len(pool_sorted) * HOLDOUT_FRAC))
        novel_ids = sorted(pool_sorted[:n_holdout])
        index_ids = sorted(pool_sorted[n_holdout:])

        dz = {e: sorted(t for t in records[e] if t in graph.terms) for e in index_ids}
        dz_novel = {e: sorted(t for t in records[e] if t in graph.terms) for e in novel_ids}

        anc_cache: dict[str, set] = {}
        ic = _build_ic([set(v) for v in dz.values()], parents, anc_cache)
        median_ic = statistics.median(ic.values()) if ic else 0.0
        noise_pool = sorted(ic) or sorted({t for v in dz.values() for t in v})

        # The harness calls make_qspec for each answerable then each novel query;
        # each call advances rng identically. We must consume rng the same way to
        # stay phase-locked, even though we discard the produced terms.
        def make_qspec(terms):
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
            return q

        ans_candidates = [e for e in index_ids if dz[e]]
        nov_candidates = [e for e in novel_ids if dz_novel[e]]
        n_nov = min(len(nov_candidates), int(round(N_QUERY * HOLDOUT_FRAC)))
        n_ans = min(len(ans_candidates), N_QUERY - n_nov)
        ans_q = ans_candidates[:n_ans]
        nov_q = nov_candidates[:n_nov]

        qspecs = [(e, False) for e in ans_q] + [(e, True) for e in nov_q]
        for e, is_novel in qspecs:
            make_qspec(dz[e] if not is_novel else dz_novel[e])  # advance rng in lockstep
            rare = (
                max((ic.get(t, 0.0) for t in (dz[e] if not is_novel else dz_novel[e])), default=0.0)
                > median_ic
            )
            plan.append((e, is_novel, rare))
    return plan


def rare_vectors_from_calls(plan, calls):
    """Pair the recorded per-call ranked lists with the (gold, novel, rare) plan.

    Returns ``(rare_correct, point_rare)``: the per-query rare-slice top-5
    correctness vector and its mean (for the self-check vs run_oneshot).
    """
    assert len(plan) == len(calls), f"plan/call count mismatch: {len(plan)} vs {len(calls)}"
    rare_correct: list[float] = []
    for (gold, is_novel, rare), ranked in zip(plan, calls):
        rank = next((rk for key, rk in ranked if key == gold), ABSENT_RANK)
        if is_novel:
            continue  # answerable slice only
        if rare:
            rare_correct.append(1.0 if rank <= 5 else 0.0)
    point = statistics.mean(rare_correct) if rare_correct else 0.0
    return rare_correct, point


CSV_FIELDS = [
    "domain", "rung", "config", "t5_rare", "t5_all", "n_rare",
    "delta_vs_prev_rung", "ci_low", "ci_high", "p", "cliffs",
]


def merge_partials() -> None:
    """Merge per-domain partial CSVs (ablation_mechanism_<domain>.csv) into the
    canonical ablation_mechanism.csv + ablation_mechanism.log."""
    import csv as _csv

    rows = []
    log_parts = []
    for domain in DOMAINS:
        p = OUT / f"ablation_mechanism_{domain}.csv"
        lp = OUT / f"ablation_mechanism_{domain}.log"
        if p.exists():
            with p.open() as fh:
                rows.extend(list(_csv.DictReader(fh)))
        if lp.exists():
            log_parts.append(lp.read_text())
    if not rows:
        raise SystemExit("merge: no per-domain partial CSVs found")
    csv_path = OUT / "ablation_mechanism.csv"
    with csv_path.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    # Verdict over merged rows.
    by_domain: dict[str, dict[int, float]] = {}
    for r in rows:
        by_domain.setdefault(r["domain"], {})[int(r["rung"])] = float(r["t5_rare"])
    verdict = ["", "# ----- decomposition / verdict (merged) -----"]
    for domain, rungs in by_domain.items():
        d01 = rungs.get(1, 0) - rungs.get(0, 0)
        d12 = rungs.get(2, 0) - rungs.get(1, 0)
        d23 = rungs.get(3, 0) - rungs.get(2, 0)
        verdict.append(
            f"{domain}: rung0={rungs.get(0,0):.3f} -> r1={rungs.get(1,0):.3f} "
            f"(+isa {d01:+.3f}) -> r2={rungs.get(2,0):.3f} (+IC {d12:+.3f}) -> "
            f"r3={rungs.get(3,0):.3f} (+typed {d23:+.3f})"
        )
    log_path = OUT / "ablation_mechanism.log"
    log_path.write_text("\n\n".join(log_parts) + "\n" + "\n".join(verdict) + "\n")
    print(f"merged -> {csv_path}\n" + "\n".join(verdict), flush=True)


def main() -> None:
    import argparse
    import csv as _csv
    import importlib

    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=sorted(DOMAINS), default=None,
                    help="run a single domain and write a per-domain partial CSV/log")
    ap.add_argument("--merge", action="store_true",
                    help="merge per-domain partial CSVs into the canonical output")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_partials()
        return

    run_domains = {args.domain: DOMAINS[args.domain]} if args.domain else dict(DOMAINS)
    partial = args.domain is not None

    csv_rows = []
    log_lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        log_lines.append(s)

    log("# D1 mechanism-stack ablation  (rare-slice tail top-5)")
    log(f"# seeds={SEEDS} n_index={N_INDEX} n_query={N_QUERY} holdout_frac={HOLDOUT_FRAC}")
    log(f"# rung-3 config: scorer=surprisal norm={NORM} gamma={GAMMA} rho={RHO} delta={DELTA}")
    log("")

    for domain, arm_path in run_domains.items():
        arm = importlib.import_module(arm_path)
        mounted, records = arm.load()
        graph = mounted.graph
        n_rel = sum(1 for _ in graph.typed_relations())
        n_isa = sum(1 for _ in graph.is_a_edges())
        log(f"========== DOMAIN {domain} ==========")
        log(f"terms={len(graph.terms)} is_a_edges={n_isa} typed_relations={n_rel} records={len(records)}")

        # Recording memories: one frozen run_oneshot pass drives them; each
        # memory's per-query ranked outputs are captured for the paired bootstrap.
        ladder = rung_memories(graph, record=True)
        memories = [m for _, _, m in ladder]
        names_by_rung = {rung: m.name for rung, _, m in ladder}
        label_by_rung = {rung: lab for rung, lab, _ in ladder}

        # Single frozen run_oneshot with all rungs present (identical queries/index).
        result = run_oneshot(
            domain, mounted, records, memories,
            seeds=SEEDS, n_index=N_INDEX, n_query=N_QUERY, holdout_frac=HOLDOUT_FRAC,
        )
        rare_top5 = {
            m.name: result["per_memory"][m.name]["tail"]["top5"]["rare"] for m in memories
        }
        all_top5 = {
            m.name: result["per_memory"][m.name]["tail"]["top5"]["all"] for m in memories
        }
        log(f"n_all={result['n_all']} n_rare={result['n_rare']} n_novel={result['n_novel']}")

        # Per-query rare vectors for the adjacent-rung paired bootstrap: pair each
        # memory's recorded ranked outputs with the deterministic query plan
        # (retrieval-free bookkeeping; no second SMA pass).
        plan = query_plan(graph, records)
        rare_correct: dict[str, list[float]] = {}
        point_rare: dict[str, float] = {}
        for m in memories:
            vec, pt = rare_vectors_from_calls(plan, m.calls)
            rare_correct[m.name] = vec
            point_rare[m.name] = pt

        # HARD self-check: recording-derived rare top-5 must equal run_oneshot's.
        for rung, name in names_by_rung.items():
            a = rare_top5[name]
            b = point_rare[name]
            if abs(a - b) > 1e-9:
                raise SystemExit(
                    f"[{domain}] rare-slice reconstruction MISMATCH for {name}: "
                    f"run_oneshot={a:.6f} recorded={b:.6f} (delta={a-b:.2e}). "
                    "Aborting rather than reporting an unverified paired test."
                )
        log("  rare-slice recording self-check: PASS (matches run_oneshot to 1e-9)")

        # Per-rung point estimates + adjacent-rung paired bootstrap on rare slice.
        prev_name = None
        for rung in sorted(names_by_rung):
            name = names_by_rung[rung]
            label = label_by_rung[rung]
            t5r = rare_top5[name]
            t5a = all_top5[name]
            bs = None
            delta_str = p_str = ci_lo = ci_hi = cd_str = ""
            if prev_name is None:
                log(
                    f"  rung {rung} [{label}]  t5_rare={t5r:.4f}  t5_all={t5a:.4f}  (floor)"
                )
            else:
                a = rare_correct[name]
                b = rare_correct[prev_name]
                if a and b:
                    bs = paired_bootstrap(a, b)
                    cd = cliffs_delta(a, b)
                    delta_str = f"{bs['delta']:+.4f}"
                    p_str = f"{bs['p_value']:.4g}"
                    ci_lo = f"{bs['ci_low']:+.4f}"
                    ci_hi = f"{bs['ci_high']:+.4f}"
                    cd_str = f"{cd:+.3f}"
                    log(
                        f"  rung {rung} [{label}]  t5_rare={t5r:.4f}  t5_all={t5a:.4f}  "
                        f"delta_vs_prev={delta_str}  CI=[{ci_lo},{ci_hi}]  p={p_str}  cliffs={cd_str}"
                    )
                else:
                    log(
                        f"  rung {rung} [{label}]  t5_rare={t5r:.4f}  t5_all={t5a:.4f}  "
                        f"(rare slice empty - no paired test)"
                    )
            csv_rows.append(
                {
                    "domain": domain,
                    "rung": rung,
                    "config": label,
                    "t5_rare": f"{t5r:.4f}",
                    "t5_all": f"{t5a:.4f}",
                    "n_rare": result["n_rare"],
                    "delta_vs_prev_rung": delta_str,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "p": p_str,
                    "cliffs": cd_str,
                }
            )
            prev_name = name
        log("")

    # Verdict.
    log("")
    log("# ----- decomposition / verdict -----")
    by_domain: dict[str, dict[int, float]] = {}
    for r in csv_rows:
        by_domain.setdefault(r["domain"], {})[int(r["rung"])] = float(r["t5_rare"])
    for domain, rungs in by_domain.items():
        d01 = rungs.get(1, 0) - rungs.get(0, 0)
        d12 = rungs.get(2, 0) - rungs.get(1, 0)
        d23 = rungs.get(3, 0) - rungs.get(2, 0)
        log(
            f"{domain}: rung0={rungs.get(0,0):.3f} -> r1={rungs.get(1,0):.3f} "
            f"(+isa {d01:+.3f}) -> r2={rungs.get(2,0):.3f} (+IC {d12:+.3f}) -> "
            f"r3={rungs.get(3,0):.3f} (+typed {d23:+.3f})"
        )

    # Write CSV + log. With --domain, write per-domain partials (resumable);
    # otherwise write the canonical files directly.
    if partial:
        only = args.domain
        csv_path = OUT / f"ablation_mechanism_{only}.csv"
        log_path = OUT / f"ablation_mechanism_{only}.log"
    else:
        csv_path = OUT / "ablation_mechanism.csv"
        log_path = OUT / "ablation_mechanism.log"
    with csv_path.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(csv_rows)
    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"wrote {csv_path}\nwrote {log_path}", flush=True)


if __name__ == "__main__":
    main()
