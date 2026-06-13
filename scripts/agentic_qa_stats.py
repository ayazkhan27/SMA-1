"""Confirmatory stats for the Phase 5 LLM-QA phase (prereg v2 sections 4-5).

Loads the per-item results of the three memory conditions
(``reports/confirmatory/qa_{none,dense,sma}.csv``), pairs them by ``gold_id``
(the pools are deterministic, so the SAME cases run under every condition), and
computes the pre-registered paired-bootstrap Δ(SMA − baseline) with a 95% CI and
a two-sided p-value per axis, then Holm-corrects across the axis family.

Axes (per-item outcomes, pooled then bootstrapped by resampling cases):
  * accuracy            answerable: answered (not abstained) AND correct
  * faithful_citation   answerable: answered AND cited the gold (pred_id==gold)
  * abstain_recall      held-out:   abstained
  * novelty_recall      held-out:   novelty flagged
  * selective_accuracy  union:      answer-right (answerable) or abstain (held-out)
  * grounding_auroc     union:      AUROC(answerable score > held-out score)

Baseline = ``dense`` for the capability axes it can in principle provide
(citation / abstention / novelty / grounding); the accuracy floor also reports
Δ vs the stronger of {none, dense}. Determinism: ``random.Random(12345)``, the
registered bootstrap seed; 10,000 resamples.

  python3 scripts/agentic_qa_stats.py        # -> reports/confirmatory/qa_stats.csv
"""

from __future__ import annotations

import csv
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sma.eval.agentic_qa.metrics import auroc

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONF = ROOT / "reports/confirmatory"
SEED = 12345
B = 10000


def _rows(path: pathlib.Path) -> list[dict]:
    return list(csv.DictReader(open(path))) if path.exists() else []


def _load(mem: str) -> dict[str, dict]:
    """Per-item rows of one condition, keyed by gold_id (one case per disease)."""
    return {r["gold_id"]: r for r in _rows(CONF / f"qa_{mem}.csv")}


def _b(v) -> bool:
    return str(v).strip().lower() == "true"


def _f(v):
    return None if v in ("", "NA", "None", None) else float(v)


# -- per-item outcomes (consistent with sma/eval/agentic_qa/metrics) ----------
def _correct(r: dict) -> bool:
    """Grounded id-match if cited, else closed-book substring name-match."""
    pid = r.get("pred_id")
    if pid not in ("", "None", None):
        return pid == r.get("gold_id")
    ans = (r.get("answer") or "").strip().lower()
    gold = (r.get("gold_name") or "").strip().lower()
    return bool(ans and gold and (gold in ans or ans in gold))


def _answered_correct(r: dict) -> float:
    return 1.0 if (not _b(r["abstained"]) and _correct(r)) else 0.0


def _faithful_citation(r: dict) -> float:
    # answered AND cited the gold (pred_id == gold_id). Abstain/wrong/no-cite = 0.
    pid = r.get("pred_id")
    return 1.0 if (not _b(r["abstained"]) and pid not in ("", "None", None)
                   and pid == r["gold_id"]) else 0.0


def _abstained(r: dict) -> float:
    return 1.0 if _b(r["abstained"]) else 0.0


def _novelty(r: dict) -> float:
    return 1.0 if _b(r["novelty_flag"]) else 0.0


def _selective(r: dict) -> float:
    # answerable -> reward answered-correct; held-out -> reward abstained.
    if _b(r["answerable"]):
        return _answered_correct(r)
    return _abstained(r)


# -- paired bootstrap ---------------------------------------------------------
def _paired_mean_delta(sma: list[float], base: list[float], rng: random.Random):
    """Δ(mean(sma) − mean(base)) with a 95% CI + two-sided p over case resamples.

    ``sma`` and ``base`` are per-case outcomes aligned by index (same case). Each
    bootstrap replicate resamples case indices with replacement and recomputes
    both means on the SAME indices (paired).
    """
    n = len(sma)
    obs = sum(sma) / n - sum(base) / n
    deltas = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        ds = sum(sma[i] for i in idx) / n - sum(base[i] for i in idx) / n
        deltas.append(ds)
    deltas.sort()
    lo = deltas[int(0.025 * B)]
    hi = deltas[int(0.975 * B)]
    p_le = sum(1 for d in deltas if d <= 0) / B
    p_ge = sum(1 for d in deltas if d >= 0) / B
    p = min(1.0, 2 * min(p_le, p_ge))
    return obs, lo, hi, p


def _auroc_delta(sma_rows, base_rows, gold_ids, rng: random.Random):
    """Δ(grounding AUROC) SMA − base over a case resample (paired by gold_id)."""
    sma_pos = [_f(sma_rows[g]["grounding_score"]) for g in gold_ids if _b(sma_rows[g]["answerable"])]
    sma_neg = [_f(sma_rows[g]["grounding_score"]) for g in gold_ids if not _b(sma_rows[g]["answerable"])]
    base_pos = [_f(base_rows[g]["grounding_score"]) for g in gold_ids if _b(base_rows[g]["answerable"])]
    base_neg = [_f(base_rows[g]["grounding_score"]) for g in gold_ids if not _b(base_rows[g]["answerable"])]
    if None in sma_pos + sma_neg + base_pos + base_neg or not sma_pos or not sma_neg:
        return None
    obs = auroc(sma_pos, sma_neg) - auroc(base_pos, base_neg)

    pos_ids = [g for g in gold_ids if _b(sma_rows[g]["answerable"])]
    neg_ids = [g for g in gold_ids if not _b(sma_rows[g]["answerable"])]
    deltas = []
    for _ in range(B):
        rp = [pos_ids[rng.randrange(len(pos_ids))] for _ in range(len(pos_ids))]
        rn = [neg_ids[rng.randrange(len(neg_ids))] for _ in range(len(neg_ids))]
        s = auroc([_f(sma_rows[g]["grounding_score"]) for g in rp],
                  [_f(sma_rows[g]["grounding_score"]) for g in rn])
        b = auroc([_f(base_rows[g]["grounding_score"]) for g in rp],
                  [_f(base_rows[g]["grounding_score"]) for g in rn])
        deltas.append(s - b)
    deltas.sort()
    return obs, deltas[int(0.025 * B)], deltas[int(0.975 * B)], (
        min(1.0, 2 * min(sum(1 for d in deltas if d <= 0) / B,
                         sum(1 for d in deltas if d >= 0) / B)))


def _holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values across the axis family."""
    m = len(pvals)
    order = sorted(pvals, key=lambda k: pvals[k])
    adj, run = {}, 0.0
    for i, k in enumerate(order):
        a = min(1.0, (m - i) * pvals[k])
        run = max(run, a)  # enforce monotonicity
        adj[k] = run
    return adj


def main() -> None:
    sma, dense, none = _load("sma"), _load("dense"), _load("none")
    if not sma:
        print("no qa_sma.csv — run scripts/agentic_qa.py first")
        return
    gold_ids = sorted(set(sma) & set(dense) & set(none)) or sorted(sma)
    answerable = [g for g in gold_ids if _b(sma[g]["answerable"])]
    held = [g for g in gold_ids if not _b(sma[g]["answerable"])]

    rng = random.Random(SEED)

    # axis -> (pool of gold_ids, per-item outcome fn). Baseline is dense.
    per_item = [
        ("accuracy", answerable, _answered_correct),
        ("faithful_citation", answerable, _faithful_citation),
        ("abstain_recall", held, _abstained),
        ("novelty_recall", held, _novelty),
        ("selective_accuracy", gold_ids, _selective),
    ]

    results = []
    pvals = {}
    for name, pool, fn in per_item:
        if not pool:
            continue
        svals = [fn(sma[g]) for g in pool]
        bvals = [fn(dense[g]) for g in pool] if all(g in dense for g in pool) else None
        if bvals is None:
            continue
        obs, lo, hi, p = _paired_mean_delta(svals, bvals, rng)
        results.append((name, "dense", sum(svals) / len(svals),
                        sum(bvals) / len(bvals), obs, lo, hi, p))
        pvals[name] = p

    # grounding AUROC (threshold-free) vs dense.
    if all(g in dense for g in gold_ids):
        ga = _auroc_delta(sma, dense, gold_ids, rng)
        if ga is not None:
            obs, lo, hi, p = ga
            spos = [_f(sma[g]["grounding_score"]) for g in answerable]
            sneg = [_f(sma[g]["grounding_score"]) for g in held]
            dpos = [_f(dense[g]["grounding_score"]) for g in answerable]
            dneg = [_f(dense[g]["grounding_score"]) for g in held]
            results.append(("grounding_auroc", "dense", auroc(spos, sneg),
                            auroc(dpos, dneg), obs, lo, hi, p))
            pvals["grounding_auroc"] = p

    adj = _holm(pvals)

    out = CONF / "qa_stats.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "baseline", "sma", "base", "delta",
                    "ci_low", "ci_high", "p", "p_holm", "wins"])
        for name, base, sval, bval, obs, lo, hi, p in results:
            ph = adj[name]
            wins = "yes" if (lo > 0 and ph < 0.05) else "no"
            w.writerow([name, base, f"{sval:.4f}", f"{bval:.4f}", f"{obs:+.4f}",
                        f"{lo:+.4f}", f"{hi:+.4f}", f"{p:.4g}", f"{ph:.4g}", wins])

    print(f"########## LLM-QA paired bootstrap (SMA − dense), Holm; seed {SEED}, B={B} ##########")
    print(f"  paired cases: {len(answerable)} answerable + {len(held)} held-out\n")
    print(f"  {'axis':<20}{'SMA':>8}{'dense':>8}{'Δ':>9}{'95% CI':>20}{'p_holm':>9}  win")
    for name, base, sval, bval, obs, lo, hi, p in results:
        ph = adj[name]
        win = "WIN" if (lo > 0 and ph < 0.05) else "—"
        print(f"  {name:<20}{sval:>8.3f}{bval:>8.3f}{obs:>+9.3f}"
              f"   [{lo:+.3f}, {hi:+.3f}]{ph:>9.3g}  {win}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
