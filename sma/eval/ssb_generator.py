"""Synthetic Structural Benchmark generator."""

from __future__ import annotations

import random
from dataclasses import dataclass

from sma.ir.schema import Case, Statement, entity, make_case, stmt


@dataclass(frozen=True)
class SSBTriple:
    query: Case
    analog: Case
    distractor: Case
    gold: dict[str, str]


def generate_triples(n: int = 100, seed: int = 13) -> list[SSBTriple]:
    rng = random.Random(seed)
    return [generate_triple(rng, i) for i in range(n)]


def generate_triple(rng: random.Random, idx: int) -> SSBTriple:
    depth = rng.randint(2, 4)
    width = rng.randint(2, 4)
    # Each triple gets its own functor namespace; otherwise triples sharing
    # (depth, width) produce structurally identical cases across the library
    # and the labeled analog is no longer the unique correct answer.
    namespace = f"t{idx}"
    query = schema_case(f"q{idx}", depth, width, namespace)
    analog = rename_case(query, f"a{idx}", "far")
    distractor = rewire_case(query, f"d{idx}", namespace)
    gold = {f"E:{e.name}": f"E:{e.name.replace(f'q{idx}', f'a{idx}')}" for e in query.entities()}
    return SSBTriple(query=query, analog=analog, distractor=distractor, gold=gold)


def schema_case(prefix: str, depth: int, width: int, namespace: str) -> Case:
    entities = [entity(f"{prefix}_e{i}") for i in range(width)]
    base_rels: list[Statement] = []
    for i in range(width - 1):
        base_rels.append(stmt(f"{namespace}_rel{i}", entities[i], entities[i + 1]))
    current = base_rels
    statements = list(base_rels)
    for layer in range(1, depth):
        next_layer: list[Statement] = []
        for i in range(max(1, len(current) - 1)):
            relation = stmt(
                f"{namespace}_ho{layer}", current[i], current[min(i + 1, len(current) - 1)]
            )
            next_layer.append(relation)
            statements.append(relation)
        current = next_layer
    return make_case(statements, {"adapter": "ssb", "tier": 0})


def rename_case(case: Case, entity_prefix: str, functor_prefix: str) -> Case:
    def rename_stmt(s: Statement) -> Statement:
        args = []
        for arg in s.args:
            if isinstance(arg, Statement):
                args.append(rename_stmt(arg))
            else:
                parts = arg.name.split("_", 1)
                suffix = parts[1] if len(parts) == 2 else arg.name
                args.append(entity(f"{entity_prefix}_{suffix}", arg.type))
        return Statement(f"{functor_prefix}_{s.functor}", tuple(args))

    return make_case([rename_stmt(s) for s in case.statements], {"adapter": "ssb", "tier": 0})


def rewire_case(case: Case, prefix: str, namespace: str) -> Case:
    ents = [entity(f"{prefix}_e{i}") for i, _ in enumerate(case.entities())]
    statements: list[Statement] = []
    functors = [s.functor for s in case.statements if s.arity == 2 and not any(isinstance(a, Statement) for a in s.args)]
    for i, functor in enumerate(functors):
        statements.append(stmt(functor, ents[i % len(ents)], ents[(i + 2) % len(ents)]))
    if len(statements) >= 2:
        statements.append(stmt(f"{namespace}_ho1", statements[-1], statements[0]))
    return make_case(statements or [stmt("empty", entity(prefix))], {"adapter": "ssb", "tier": 0})

