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
        self._score_cache: dict[tuple[str, str, str], tuple[float, float]] = {}

    def add(self, case: Case) -> None:
        vector = functor_vector(case, canon=self.canon)
        self.cases[case.case_id] = case
        self.ann.add(case.case_id, vector)
        self.inverted.add(case.case_id, vector)

    def build(self, cases: list[Case]) -> None:
        for case in cases:
            self.add(case)

    def retrieve(
        self, query: Case, k: int = 10, shortlist: int = 200, fac_budget: int | None = None
    ) -> list[RetrievalResult]:
        qvec = functor_vector(query, canon=self.canon)
        ann_ids = [case_id for case_id, _ in self.ann.search(qvec, k=min(shortlist, len(self.cases)))]
        if not ann_ids:
            ann_ids = sorted(self.inverted.candidates(qvec))
        bounded = [
            (case_id, self.inverted.bound(qvec, case_id, max_score_per_mh=4.0)) for case_id in ann_ids
        ]
        bounded.sort(key=lambda row: (-row[1], row[0]))
        # ses_n = score / max(self(base), self(target)) <= U_bound / self(target),
        # so dividing the raw-score bound by the query's self-score gives an
        # admissible bound in ses_n units. The MDL scorer has no such bound, so
        # it never early-stops on bounds (budget only).
        ses_n_denom = (
            max(self_score(query, gamma=self.config.gamma), 1e-9)
            if self.config.scorer == "ses"
            else None
        )
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
        key = (case_id, query.case_id, self.config.scorer)
        if key not in self._score_cache:
            gmap = match_cases(self.cases[case_id], query, config=self.config)
            self._score_cache[key] = (gmap.score, gmap.normalized_score)
        return self._score_cache[key]
