"""Certified MAC/FAC retrieval over a case library."""

from __future__ import annotations

from dataclasses import dataclass

from sma.ir.canon import Canonicalizer, default_canonicalizer
from sma.ir.schema import Case
from sma.match.engine import match_cases
from sma.match.ses import self_score
from sma.match.types import MatchConfig

from .ann import AnnIndex
from .content_vectors import functor_vector
from .inverted import InvertedIndex


@dataclass(frozen=True)
class RetrievalResult:
    case_id: str
    ses_n: float
    score: float
    u_bound: float
    certified: bool


class MacFacIndex:
    def __init__(self, config: MatchConfig | None = None, canon: Canonicalizer | None = None):
        self.config = config or MatchConfig()
        self.canon = canon or default_canonicalizer()
        self.cases: dict[str, Case] = {}
        self.ann = AnnIndex()
        self.inverted = InvertedIndex()
        self._score_cache: dict[tuple, tuple[float, float]] = {}

    def add(self, case: Case) -> None:
        vector = functor_vector(case, canon=self.canon)
        self.cases[case.case_id] = case
        self.ann.add(case.case_id, vector)
        self.inverted.add(case.case_id, vector)
        if self.config.functor_costs is not None:
            # Corpus changed; surprisal costs and cached scores are stale.
            self.config.functor_costs = None
            self._score_cache.clear()

    def build(self, cases: list[Case]) -> None:
        for case in cases:
            self.add(case)

    def corpus_costs(self) -> dict[str, float]:
        """Corpus surprisal (-log2 p, KT-smoothed) per canonical functor."""
        import math
        from collections import Counter

        counts: Counter[str] = Counter()
        for vector in self.inverted.vectors.values():
            for feature, n in vector.items():
                if feature.startswith("f:"):
                    counts[feature[2:]] += n
        total = sum(counts.values())
        vocab = max(len(counts), 1)
        return {
            functor: -math.log2((count + 0.5) / (total + 0.5 * vocab))
            for functor, count in counts.items()
        }

    def retrieve(
        self, query: Case, k: int = 10, shortlist: int = 200, fac_budget: int | None = None
    ) -> list[RetrievalResult]:
        if self.config.scorer == "surprisal" and self.config.functor_costs is None:
            # Lazily derive costs from the indexed corpus; deterministic given
            # contents. Stale after add() — cleared there.
            self.config.functor_costs = self.corpus_costs()
        qvec = functor_vector(query, canon=self.canon)
        ann_ids = [case_id for case_id, _ in self.ann.search(qvec, k=min(shortlist, len(self.cases)))]
        if not ann_ids:
            ann_ids = sorted(self.inverted.candidates(qvec))
        bound_costs = self.config.functor_costs if self.config.scorer == "surprisal" else None
        bounded = [
            (case_id, self.inverted.bound(qvec, case_id, max_score_per_mh=4.0, costs=bound_costs))
            for case_id in ann_ids
        ]
        bounded.sort(key=lambda row: (-row[1], row[0]))
        # ses_n = score / max(self(base), self(target)) <= U_bound / self(target),
        # so dividing the raw-score bound by the query's self-score gives an
        # admissible bound in ses_n units (weighted consistently for the
        # surprisal scorer). The MDL scorer has no such bound, so it never
        # early-stops on bounds (budget only).
        ses_n_denom = None
        if self.config.scorer in ("ses", "surprisal") and self.config.normalization != "min":
            cost_fn = None
            if bound_costs:
                costs = bound_costs

                def cost_fn(mh):
                    from sma.ir.schema import Statement

                    if isinstance(mh.base, Statement):
                        return costs.get(self.canon.canonical(mh.base.functor), 1.0)
                    return 1.0

            ses_n_denom = max(self_score(query, gamma=self.config.gamma, cost_fn=cost_fn), 1e-9)
        scored: list[tuple[str, float, float, float]] = []
        kth_ses_n = float("-inf")
        n_examined = 0
        for case_id, bound in bounded:
            if fac_budget is not None and n_examined >= fac_budget:
                break
            if (
                ses_n_denom is not None
                and len(scored) >= k
                and bound / ses_n_denom < kth_ses_n
            ):
                break
            score, ses_n = self._score_case(case_id, query)
            scored.append((case_id, ses_n, score, bound))
            n_examined += 1
            if len(scored) >= k:
                kth_ses_n = sorted(s[1] for s in scored)[-k]
        # The top-k is certified exact over the shortlist iff no unexamined
        # candidate's bound could still beat the k-th best ses_n.
        remaining = bounded[n_examined:]
        certified = not remaining or (
            ses_n_denom is not None
            and len(scored) >= k
            and remaining[0][1] / ses_n_denom < kth_ses_n
        )
        scored.sort(key=lambda row: (-row[1], row[0]))
        return [
            RetrievalResult(case_id=cid, ses_n=ses_n, score=score, u_bound=bound, certified=certified)
            for cid, ses_n, score, bound in scored[:k]
        ]

    def brute_force(self, query: Case, k: int = 10) -> list[RetrievalResult]:
        qvec = functor_vector(query, canon=self.canon)
        results: list[RetrievalResult] = []
        for case_id, case in self.cases.items():
            score, ses_n = self._score_case(case_id, query)
            results.append(
                RetrievalResult(
                    case_id=case_id,
                    ses_n=ses_n,
                    score=score,
                    u_bound=self.inverted.bound(qvec, case_id, max_score_per_mh=4.0),
                    certified=True,
                )
            )
        return sorted(results, key=lambda row: (-row.ses_n, row.case_id))[:k]

    def _score_case(self, case_id: str, query: Case) -> tuple[float, float]:
        key = (case_id, query.case_id, self.config.scorer, self.config.normalization)
        if key not in self._score_cache:
            gmap = match_cases(self.cases[case_id], query, config=self.config)
            self._score_cache[key] = (gmap.score, gmap.normalized_score)
        return self._score_cache[key]
