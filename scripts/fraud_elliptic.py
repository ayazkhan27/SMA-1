"""Driver for the structural-fraud arm (Elliptic Bitcoin transaction graph).

Encodes each transaction's local graph neighbourhood (predecessor/successor
bitcoin-flow topology) as a case of higher-order typology relations — the
cross-record structure flat-tabular fraud lacks — and runs a retrieval-by-analogy
illicit-detection evaluation: SMA (structure-mapping) vs dense-RAG vs BM25, with
a flat logistic-regression-on-features baseline for context. Macro-F1, ROC-AUC,
and a SMA-vs-best paired bootstrap are written to
``reports/confirmatory/agentic_fraud_elliptic.csv``; the honest verdict +
mechanism go to ``agentic_fraud_elliptic.log``.

  # full run (default seeds, capped index for tractability):
  python3 scripts/fraud_elliptic.py

  # quick smoke:
  python3 scripts/fraud_elliptic.py --n-max 1500 --seeds 7
"""

from __future__ import annotations

import argparse
import csv
import datetime
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sma.eval.fraud_elliptic.encoder import load_elliptic  # noqa: E402
from sma.eval.fraud_elliptic.eval import run_elliptic  # noqa: E402

DATA_DIR = ROOT / "data/raw/elliptic/elliptic_bitcoin_dataset"
OUT_CSV = ROOT / "reports/confirmatory/agentic_fraud_elliptic.csv"
OUT_LOG = ROOT / "reports/confirmatory/agentic_fraud_elliptic.log"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 23])
    ap.add_argument("--frac-test", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=15, help="kNN analogs per query")
    ap.add_argument("--n-max", type=int, default=4000,
                   help="cap on labelled nodes per seed (class ratio preserved)")
    ap.add_argument("--calib-frac", type=float, default=0.3,
                   help="fraction of train held out to calibrate each method's threshold")
    ap.add_argument("--no-logreg", action="store_true",
                   help="skip the flat logistic-regression context baseline")
    args = ap.parse_args()

    if not DATA_DIR.exists():
        sys.exit(
            f"Elliptic data not found at {DATA_DIR}.\n"
            "Acquire it with:\n"
            "  kaggle datasets download -d ellipticco/elliptic-data-set "
            "-p data/raw/elliptic/\n"
            "  cd data/raw/elliptic && unzip elliptic-data-set.zip\n"
        )

    print(f"Loading Elliptic from {DATA_DIR} ...", flush=True)
    g = load_elliptic(str(DATA_DIR))
    print(f"  {len(g.feats)} nodes, {len(g.labelled_ids())} labelled "
          f"(illicit/licit), running ...", flush=True)

    res = run_elliptic(
        g,
        seeds=tuple(args.seeds),
        frac_test=args.frac_test,
        k=args.k,
        n_max=args.n_max,
        calib_frac=args.calib_frac,
        include_logreg=not args.no_logreg,
    )

    # --- write CSV ---------------------------------------------------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "method", "macro_f1", "illicit_f1", "roc_auc",
                    "calib_threshold", "n_pred", "n_test_pooled", "n_illicit",
                    "k", "seeds"])
        for m, d in res["per_method"].items():
            w.writerow([
                res["arm"], m,
                f"{d['macro_f1']:.4f}", f"{d['illicit_f1']:.4f}",
                ("" if d["roc_auc"] != d["roc_auc"] else f"{d['roc_auc']:.4f}"),
                f"{d['threshold']:.4f}",
                d["n"], res["n_test_pooled"], res["n_illicit"],
                res["k"], "|".join(map(str, res["seeds"])),
            ])
        p = res["primary"]
        w.writerow([])
        w.writerow(["primary", f"{p['a']}_vs_{p['b']}", "delta_acc", f"{p['delta_acc']:.4f}",
                    "ci_low", f"{p['ci_low']:.4f}", "ci_high", f"{p['ci_high']:.4f}",
                    "p_value", f"{p['p_value']:.4g}"])

    # --- verdict + mechanism -> log ---------------------------------------
    pm = res["per_method"]
    p = res["primary"]
    sma_f1 = pm["sma"]["macro_f1"]
    best_base = p["b"]
    base_f1 = pm[best_base]["macro_f1"]
    delta_f1 = sma_f1 - base_f1
    win = (delta_f1 > 0) and (p["ci_low"] > 0) and (p["p_value"] < 0.05)
    verdict = "WIN" if win else "NULL"

    lines = []
    lines.append("=" * 72)
    lines.append("STRUCTURAL-FRAUD ARM — Elliptic Bitcoin transaction graph")
    lines.append(f"generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("MECHANISM")
    lines.append("-" * 72)
    lines.append(
        "Each transaction is encoded NOT as its 166 flat features but as a case\n"
        "of graph-neighbourhood typology terms over a licit/illicit lattice:\n"
        "  - fanIn_* / fanOut_*  : predecessor / successor degree class\n"
        "  - inVal_* / outVal_*  : incoming / outgoing value tier\n"
        "  - temp_*              : temporal-step bucket (49 steps -> early/mid/late)\n"
        "  - nbrIllicit_* / nbrLicit_* : neighbour LABEL context (leak-guarded:\n"
        "      train-visible labels only; a node's OWN class is never emitted)\n"
        "Higher-order relations flowsFrom(fanIn, nbrCtx) and flowsTo(fanOut, nbrCtx)\n"
        "wire the node's own topology to its neighbour context — the cross-record\n"
        "structure flat-tabular encodings discard. SMA mounts the lattice and\n"
        "retrieves illicit analogs by structural match; dense/BM25 see the same\n"
        "term text; logreg sees the raw 166 features (the flat-tabular control).\n"
        "Evaluation: retrieval-by-analogy kNN (k={k}) — vote retrieved analogs'\n"
        "known labels into an illicit score; metric = macro-F1 + ROC-AUC.".format(k=res["k"])
    )
    lines.append("")
    lines.append("RESULTS (pooled over seeds {})".format(res["seeds"]))
    lines.append("-" * 72)
    lines.append(f"{'method':<10}{'macro_f1':>10}{'illicit_f1':>12}{'roc_auc':>10}{'cal_thr':>9}")
    for m, d in pm.items():
        auc = "NA" if d["roc_auc"] != d["roc_auc"] else f"{d['roc_auc']:.4f}"
        lines.append(f"{m:<10}{d['macro_f1']:>10.4f}{d['illicit_f1']:>12.4f}{auc:>10}{d['threshold']:>9.3f}")
    lines.append("")
    lines.append(f"n_test_pooled={res['n_test_pooled']}  n_illicit={res['n_illicit']}  "
                 f"k={res['k']}  (per-method thresholds calibrated on a train slice)")
    lines.append("")
    lines.append("PRIMARY — SMA vs best retrieval baseline ({}), paired bootstrap on".format(best_base))
    lines.append("per-node thresholded accuracy:")
    lines.append(f"  delta_acc = {p['delta_acc']:+.4f}  "
                 f"95% CI [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]  p = {p['p_value']:.4g}")
    lines.append(f"  macro-F1: sma={sma_f1:.4f}  {best_base}={base_f1:.4f}  "
                 f"delta_F1={delta_f1:+.4f}")
    lines.append("")
    lines.append("VERDICT")
    lines.append("-" * 72)
    lines.append(f"  {verdict}")
    if win:
        lines.append(
            "  SMA's structure-mapping over the bitcoin-flow neighbourhood beats the\n"
            "  best vector/lexical retrieval baseline on illicit detection, with a\n"
            "  CI excluding zero and p < 0.05. The cross-record graph structure that\n"
            "  flat-tabular fraud (4b null) lacked is the source of the edge.")
    else:
        lines.append(
            "  No significant SMA edge over the best retrieval baseline on this\n"
            "  encoding (CI includes zero or p >= 0.05). Reported honestly: the\n"
            "  graph neighbourhood as encoded did not yield a structure-mapping\n"
            "  advantage here. See macro-F1 / ROC-AUC above for where each method\n"
            "  lands; the paper's flat-vs-graph boundary claim stands either way.")
    lines.append("")

    OUT_LOG.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwrote {OUT_CSV}")
    print(f"wrote {OUT_LOG}")


if __name__ == "__main__":
    main()
