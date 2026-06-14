"""Retrieval-by-analogy illicit-detection evaluation on the Elliptic graph.

Each transaction is encoded as a *case of graph-neighbourhood typology terms*
(see :mod:`sma.eval.fraud_elliptic.encoder`). The labelled nodes are split into
a train (index) set and a test set. A memory indexes the train cases; for each
test node we retrieve its top-k analogs and vote their (known) labels — weighted
by retrieval confidence — into an illicit score. This is the *same* analogical
retrieval SMA is built for, now used as a kNN classifier so the metric is
detection quality (macro-F1, ROC-AUC), not key-recall.

Compared memories (frozen, reused read-only from :mod:`sma.eval.agentic`):

* ``sma``   — mount the typology lattice; index neighbourhood cases via MacFac;
* ``dense`` — BGE-small embeddings over the term-name text of each case;
* ``bm25``  — lexical BM25 over the same term-name text.

A ``logreg`` baseline (logistic regression on the raw 166 flat features) is run
for context — the flat-tabular method that the 4b finance null showed SMA cannot
beat when there is no cross-record structure.

Leak guard: the encoder reads neighbour labels ONLY from the train split, and a
node's own class is never emitted. Test nodes are encoded against train-visible
labels, so no test label ever enters any case.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from sma.eval.agentic.memories import (
    BM25Memory,
    DenseMemory,
    IndexItem,
    Query,
    SmaMemory,
)
from sma.eval.fraud_elliptic.encoder import (
    ILLICIT,
    LICIT,
    EllipticGraph,
    NeighbourhoodEncoder,
    build_typology,
)
from sma.ontology import mount

# Positive class = illicit.
POS = ILLICIT


@dataclass
class Split:
    train: list[str]
    test: list[str]


def stratified_split(g: EllipticGraph, frac_test: float, seed: int,
                     n_max: int | None = None) -> Split:
    """Stratified train/test split over labelled nodes (class-balanced holdout)."""
    rng = random.Random(seed)
    illicit = [t for t in g.labelled_ids() if g.label[t] == ILLICIT]
    licit = [t for t in g.labelled_ids() if g.label[t] == LICIT]
    rng.shuffle(illicit)
    rng.shuffle(licit)
    if n_max is not None:
        # Cap total while preserving the natural class ratio (illicit is ~10%).
        ratio = len(illicit) / (len(illicit) + len(licit))
        n_ill = min(len(illicit), int(round(n_max * ratio)))
        n_lic = min(len(licit), n_max - n_ill)
        illicit, licit = illicit[:n_ill], licit[:n_lic]

    def cut(xs: list[str]) -> tuple[list[str], list[str]]:
        k = int(round(len(xs) * frac_test))
        return xs[k:], xs[:k]  # train, test

    tr_i, te_i = cut(illicit)
    tr_l, te_l = cut(licit)
    return Split(train=sorted(tr_i + tr_l), test=sorted(te_i + te_l))


def _case_text(terms: list[str]) -> str:
    """Term-name text for the lexical / dense baselines (names == ids here)."""
    return " ".join(t.replace("_", " ") for t in terms)


def _build_memories(mounted):
    """Fresh instances of the three frozen retrieval memories."""
    return [SmaMemory(mounted), DenseMemory(), BM25Memory()]


def _knn_vote(mem, query: Query, train_label: dict[str, str], k: int) -> float:
    """Confidence-weighted illicit vote over a memory's top-k analogs in [0,1]."""
    res = mem.retrieve(query, k=k)
    if not res:
        return 0.0
    num = 0.0
    den = 0.0
    for r in res:
        lab = train_label.get(r.key)
        if lab is None:
            continue
        w = max(r.score, 1e-9)
        den += w
        if lab == POS:
            num += w
    return num / den if den > 0 else 0.0


def _best_f1_threshold(scores: list[float], labels: list[int]) -> float:
    """Threshold (on a calibration split) that maximizes macro-F1.

    The kNN illicit vote is strongly compressed toward 0 by the ~10% illicit
    base rate, so a fixed 0.5 cut makes every method predict all-licit. Each
    method therefore gets its own threshold, chosen on a DISJOINT calibration
    slice of the train split (never on test) by sweeping candidate cuts.
    """
    s = np.asarray(scores)
    y = np.asarray(labels)
    if len(set(labels)) < 2 or s.size == 0:
        return 0.5
    cands = sorted(set(s.tolist()))
    best_t, best_f1 = 0.5, -1.0
    for c in cands:
        f1 = f1_score(y, (s > c).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(c)
    return best_t


def run_elliptic(
    g: EllipticGraph,
    *,
    seeds=(7, 17, 23),
    frac_test: float = 0.3,
    k: int = 15,
    n_max: int | None = 4000,
    calib_frac: float = 0.3,
    include_logreg: bool = True,
) -> dict:
    """Run the retrieval-by-analogy illicit-detection evaluation.

    Returns a result dict with per-method pooled macro-F1 / ROC-AUC across seeds
    and a per-test-node paired record for the bootstrap (SMA vs best baseline).
    Each method's decision threshold is calibrated per seed on a disjoint slice
    of the train split (``calib_frac``); ROC-AUC is threshold-free.
    """
    typ = build_typology()
    mounted = mount(typ)

    method_names = ["sma", "dense", "bm25"] + (["logreg"] if include_logreg else [])
    # Pooled per-node arrays across seeds.
    scores: dict[str, list[float]] = {m: [] for m in method_names}
    preds: dict[str, list[int]] = {m: [] for m in method_names}
    truth: list[int] = []
    thresholds: dict[str, list[float]] = {m: [] for m in method_names}

    for seed in seeds:
        split = stratified_split(g, frac_test=frac_test, seed=seed, n_max=n_max)
        # Carve a calibration slice out of train (disjoint from index and test).
        rng = random.Random(seed * 7919 + 1)
        train_all = list(split.train)
        rng.shuffle(train_all)
        n_cal = int(round(len(train_all) * calib_frac))
        calib_ids = sorted(train_all[:n_cal])
        index_ids = sorted(train_all[n_cal:])

        # Encoder reads neighbour labels ONLY from the indexed train split.
        index_label = {t: g.label[t] for t in index_ids}
        enc = NeighbourhoodEncoder(graph=g, visible_labels=index_label)

        # Index train cases in every retrieval memory (identical input).
        items = []
        for t in index_ids:
            terms = enc.encode(t)
            items.append(IndexItem(
                key=t, term_ids=frozenset(terms), text=_case_text(terms), meta={"id": t}
            ))
        memories = _build_memories(mounted)
        for mem in memories:
            mem.index(items)

        def knn_scores(node_ids):
            out = {m.name: [] for m in memories}
            for t in node_ids:
                qterms = enc.encode(t)
                query = Query(term_ids=frozenset(qterms), text=_case_text(qterms))
                for mem in memories:
                    out[mem.name].append(_knn_vote(mem, query, index_label, k))
            return out

        # Calibrate each retrieval method's threshold on the calibration slice.
        cal_truth = [1 if g.label[t] == POS else 0 for t in calib_ids]
        cal_scores = knn_scores(calib_ids)
        seed_thresh = {
            m: _best_f1_threshold(cal_scores[m], cal_truth) for m in cal_scores
        }

        # Score + predict the test split.
        test_scores = knn_scores(split.test)
        for t in split.test:
            truth.append(1 if g.label[t] == POS else 0)
        for m in cal_scores:
            thr = seed_thresh[m]
            thresholds[m].append(thr)
            scores[m].extend(test_scores[m])
            preds[m].extend(int(s > thr) for s in test_scores[m])

        # Flat logistic-regression baseline on the raw 166 features (context).
        if include_logreg:
            Xtr = np.array([g.feats[t][1:] for t in index_ids])  # drop time at idx0
            ytr = np.array([1 if g.label[t] == POS else 0 for t in index_ids])
            Xcal = np.array([g.feats[t][1:] for t in calib_ids])
            Xte = np.array([g.feats[t][1:] for t in split.test])
            scaler = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(scaler.transform(Xtr), ytr)
            cal_p = clf.predict_proba(scaler.transform(Xcal))[:, 1]
            lr_thr = _best_f1_threshold(list(cal_p), cal_truth)
            te_p = clf.predict_proba(scaler.transform(Xte))[:, 1]
            thresholds["logreg"].append(lr_thr)
            scores["logreg"].extend(float(p) for p in te_p)
            preds["logreg"].extend(int(p > lr_thr) for p in te_p)

    truth_arr = np.array(truth)
    per_method: dict[str, dict] = {}
    for m in method_names:
        s = np.array(scores[m])
        pred = np.array(preds[m])
        try:
            auc = float(roc_auc_score(truth_arr, s)) if len(set(truth)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")
        per_method[m] = {
            "macro_f1": float(f1_score(truth_arr, pred, average="macro", zero_division=0)),
            "illicit_f1": float(f1_score(truth_arr, pred, pos_label=1, zero_division=0)),
            "roc_auc": auc,
            "threshold": float(np.mean(thresholds[m])) if thresholds[m] else 0.5,
            "n": int(len(s)),
        }

    # Primary: SMA vs best retrieval baseline (by macro-F1) on per-node squared
    # error of the illicit score (lower better) -> paired bootstrap on accuracy.
    from sma.eval.stats import paired_bootstrap

    retrieval_baselines = [m for m in ("dense", "bm25") if m in per_method]
    best = max(retrieval_baselines, key=lambda m: per_method[m]["macro_f1"])
    # Per-node correctness (calibrated prediction) for SMA vs best baseline.
    sma_correct = [(1.0 if preds["sma"][i] == truth[i] else 0.0) for i in range(len(truth))]
    base_correct = [(1.0 if preds[best][i] == truth[i] else 0.0) for i in range(len(truth))]
    bs = paired_bootstrap(sma_correct, base_correct)
    primary = {
        "a": "sma", "b": best,
        "delta_acc": bs["delta"], "ci_low": bs["ci_low"],
        "ci_high": bs["ci_high"], "p_value": bs["p_value"],
    }

    return {
        "arm": "fraud_elliptic",
        "n_test_pooled": len(truth),
        "n_illicit": int(truth_arr.sum()),
        "k": k, "seeds": list(seeds),
        "per_method": per_method,
        "primary": primary,
    }
