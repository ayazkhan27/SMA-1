"""Synthetic Structural Benchmark generator (de-circularized).

Each triple is (query, analog, distractor) with full gold correspondences:

- query: a seeded relational schema with its own functor vocabulary;
- analog: the SAME structure under a DISJOINT functor vocabulary and renamed
  entities - zero lexical overlap with the query. The two vocabularies are
  bridged ONLY by a declared predicate lattice (each query functor and its
  analog counterpart share an abstract parent concept), so matching requires
  minimal ascension (delta >= 2) at the rho^dist penalty. No string trick
  (the old far_-prefix bijection was known to the canonicalizer - circular);
- distractor: the query's own vocabulary (matched content vector) with the
  relational structure rewired - same words, broken structure.

build_canonicalizer(triples) returns the Canonicalizer carrying the lattice;
evaluations MUST use it together with a delta>=2 MatchConfig, otherwise
analogs are unreachable by construction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from sma.ir.canon import Canonicalizer
from sma.ir.schema import Case, Statement, entity, make_case, stmt


@dataclass(frozen=True)
class SSBTriple:
    query: Case
    analog: Case
    distractor: Case
    gold: dict[str, str]
    lattice_pairs: tuple[tuple[str, str], ...]  # (child_functor, parent_concept)


def generate_triples(n: int = 100, seed: int = 13) -> list[SSBTriple]:
    rng = random.Random(seed)
    return [generate_triple(rng, i) for i in range(n)]


def _fresh_name(rng: random.Random, prefix: str) -> str:
    return f"{prefix}{rng.randrange(1 << 24):06x}"


def generate_triple(rng: random.Random, idx: int) -> SSBTriple:
    depth = rng.randint(2, 4)
    width = rng.randint(2, 4)
    # Functor slots for this schema; each slot gets a query name and an analog
    # name drawn from disjoint random pools, joined only through a concept.
    slots = [f"rel{i}" for i in range(width - 1)] + [f"ho{layer}" for layer in range(1, depth)]
    q_name: dict[str, str] = {}
    a_name: dict[str, str] = {}
    lattice_pairs: list[tuple[str, str]] = []
    for slot in slots:
        q = _fresh_name(rng, f"q{idx}")
        a = _fresh_name(rng, f"a{idx}")
        concept = f"c{idx}_{slot}"
        q_name[slot], a_name[slot] = q, a
        lattice_pairs.append((q, concept))
        lattice_pairs.append((a, concept))

    query = schema_case(f"q{idx}", depth, width, q_name)
    analog = schema_case(f"a{idx}", depth, width, a_name)
    distractor = rewire_case(query, f"d{idx}", q_name)
    gold = {f"E:{e.name}": f"E:{e.name.replace(f'q{idx}', f'a{idx}')}" for e in query.entities()}
    return SSBTriple(query=query, analog=analog, distractor=distractor, gold=gold,
                     lattice_pairs=tuple(lattice_pairs))


def schema_case(prefix: str, depth: int, width: int, names: dict[str, str]) -> Case:
    entities = [entity(f"{prefix}_e{i}") for i in range(width)]
    base_rels: list[Statement] = []
    for i in range(width - 1):
        base_rels.append(stmt(names[f"rel{i}"], entities[i], entities[i + 1]))
    current = base_rels
    statements = list(base_rels)
    for layer in range(1, depth):
        next_layer: list[Statement] = []
        for i in range(max(1, len(current) - 1)):
            relation = stmt(names[f"ho{layer}"], current[i], current[min(i + 1, len(current) - 1)])
            next_layer.append(relation)
            statements.append(relation)
        current = next_layer
    return make_case(statements, {"adapter": "ssb", "tier": 0})


def rewire_case(case: Case, prefix: str, names: dict[str, str]) -> Case:
    """Same vocabulary as the query, broken structure.

    The base relations form a STAR (every relation points into one hub
    entity) instead of the query's chain. A chain has no entity with
    in-degree >= 2, a star does, so for width >= 3 the two are provably
    non-isomorphic under ordered relations - the old (i, i+2 mod n)
    rewiring could reproduce the chain up to relabeling at small widths
    (the matcher then correctly scored the 'distractor' 1.0)."""
    ents = [entity(f"{prefix}_e{i}") for i, _ in enumerate(case.entities())]
    statements: list[Statement] = []
    functors = [s.functor for s in case.statements
                if s.arity == 2 and not any(isinstance(a, Statement) for a in s.args)]
    hub = ents[-1]
    for i, functor in enumerate(functors):
        statements.append(stmt(functor, ents[i % max(len(ents) - 1, 1)], hub))
    if len(statements) >= 2 and "ho1" in names:
        statements.append(stmt(names["ho1"], statements[-1], statements[0]))
    return make_case(statements or [stmt("empty", entity(prefix))], {"adapter": "ssb", "tier": 0})


def build_canonicalizer(triples: list[SSBTriple]) -> Canonicalizer:
    """Canonicalizer whose lattice is the ONLY bridge between vocabularies."""
    canon = Canonicalizer()
    for triple in triples:
        for child, parent in triple.lattice_pairs:
            canon.lattice.add(child, parent)
    return canon
