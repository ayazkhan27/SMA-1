"""Candidate inference projection."""

from __future__ import annotations

from sma.ir.schema import Entity, Node, Statement
from sma.ir.sexpr import dumps_statement

from .types import CandidateInference, GMap, node_key


def candidate_inferences(gmap: GMap) -> tuple[CandidateInference, ...]:
    base_to_target: dict[str, Node] = {mh.base_key: mh.target for mh in gmap.hypotheses}
    mapped_statement_keys = {
        mh.base_key for mh in gmap.hypotheses if isinstance(mh.base, Statement)
    }
    out: list[CandidateInference] = []
    skolem_counter = 0

    def project(node: Node) -> Node:
        nonlocal skolem_counter
        mapped = base_to_target.get(node_key(node))
        if mapped is not None:
            return mapped
        if isinstance(node, Entity):
            skolem_counter += 1
            return Entity(f"AnalogySkolemFn_{skolem_counter}", node.type)
        return Statement(node.functor, tuple(project(arg) for arg in node.args), ascension=node.ascension)

    for statement in gmap.base.statements:
        if node_key(statement) in mapped_statement_keys:
            continue
        if not any(node_key(entity) in base_to_target for entity in statement.entities()):
            continue
        skolem_counter = 0
        projected = project(statement)
        skolems = tuple(
            entity.name for entity in projected.entities() if entity.name.startswith("AnalogySkolemFn_")
        )
        ascensions = tuple(
            f"{mh.base_key}->{mh.ancestor}" for mh in gmap.hypotheses if mh.distance > 0 and mh.ancestor
        )
        out.append(
            CandidateInference(
                inference_sexpr=dumps_statement(projected),
                base_case_id=gmap.base.case_id,
                target_case_id=gmap.target.case_id,
                ses_n=gmap.normalized_score,
                support=tuple(sorted(mapped_statement_keys)),
                skolems=skolems,
                ascensions=ascensions,
            )
        )
    return tuple(out)

