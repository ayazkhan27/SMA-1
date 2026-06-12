"""SAGE-style generalization pools."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from sma.ir.schema import Case, make_case
from sma.ir.sexpr import dumps_statement, loads_statement
from sma.match.engine import match_cases
from sma.match.types import MatchConfig

from .probabilities import support_probability


@dataclass
class Generalization:
    gen_id: str
    constituents: list[str] = field(default_factory=list)
    fact_counts: Counter[str] = field(default_factory=Counter)

    def probabilities(self) -> dict[str, float]:
        total = max(len(self.constituents), 1)
        return {fact: support_probability(count, total) for fact, count in self.fact_counts.items()}

    def schema_case(self, probability_cutoff: float = 0.6, min_constituents: int = 3) -> Case:
        total = len(self.constituents)
        facts = []
        for sexpr, count in sorted(self.fact_counts.items()):
            if total < min_constituents or support_probability(count, total) >= probability_cutoff:
                facts.append(loads_statement(sexpr))
        return make_case(facts, {"adapter": "sage", "generalization": self.gen_id})


@dataclass
class SagePool:
    pool_id: str
    config: MatchConfig = field(default_factory=MatchConfig)
    assimilation_threshold: float = 0.25
    probability_cutoff: float = 0.6
    min_constituents: int = 3
    generalizations: list[Generalization] = field(default_factory=list)
    outliers: list[Case] = field(default_factory=list)

    def assimilate(self, case: Case) -> str:
        best_idx = -1
        best_score = float("-inf")
        for idx, gen in enumerate(self.generalizations):
            gmap = match_cases(gen.schema_case(self.probability_cutoff, self.min_constituents), case, self.config)
            if gmap.normalized_score > best_score:
                best_score = gmap.normalized_score
                best_idx = idx
        if best_idx >= 0 and best_score >= self.assimilation_threshold:
            self._add_to_generalization(self.generalizations[best_idx], case)
            return self.generalizations[best_idx].gen_id
        for outlier in list(self.outliers):
            gmap = match_cases(outlier, case, self.config)
            if gmap.normalized_score >= self.assimilation_threshold:
                gen = Generalization(gen_id=f"{self.pool_id}_gen_{len(self.generalizations)}")
                self._add_to_generalization(gen, outlier)
                self._add_to_generalization(gen, case)
                self.generalizations.append(gen)
                self.outliers.remove(outlier)
                return gen.gen_id
        self.outliers.append(case)
        return "outlier"

    def _add_to_generalization(self, gen: Generalization, case: Case) -> None:
        if case.case_id not in gen.constituents:
            gen.constituents.append(case.case_id)
        for statement in case.statements:
            gen.fact_counts[dumps_statement(statement)] += 1

    def stats(self) -> dict:
        return {
            "pool_id": self.pool_id,
            "n_generalizations": len(self.generalizations),
            "n_outliers": len(self.outliers),
            "generalizations": [
                {
                    "gen_id": gen.gen_id,
                    "n_constituents": len(gen.constituents),
                    "n_facts": len(gen.fact_counts),
                    "probabilities": gen.probabilities(),
                }
                for gen in self.generalizations
            ],
        }

