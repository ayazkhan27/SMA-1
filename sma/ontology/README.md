# `sma.ontology` — universal OWL/OBO loader, mount, registry, router

The universal ontology layer described in `docs/PAPER_SPINE.md`. It generalizes
the hand-rolled HPO mount (`scripts/rare_disease_test.py`) into a reusable
pipeline that turns *any* OBO/OWL ontology into structure-mapping cases: parse
to a normalized graph, mount the is-a hierarchy onto a predicate lattice, build
a `MacFacIndex`, and retrieve by analogy. Parsing is stdlib-only
(`xml.etree` for OWL — `rdflib` is **not** required).

## The `OntologyGraph` contract

```python
@dataclass
class Term:
    id: str                                    # "HP:0001250"
    name: str = ""
    parents: tuple[str, ...] = ()              # is_a parent ids
    relations: tuple[tuple[str, str], ...] = ()  # (rel_type, target_id)
    obsolete: bool = False

@dataclass
class OntologyGraph:
    name: str                                  # short id, e.g. "hpo"
    version: str = ""                          # data-version / owl versionIRI
    terms: dict[str, Term]
    def active_terms(self) -> dict[str, Term]            # non-obsolete only
    def is_a_edges(self) -> Iterator[tuple[str, str]]    # (child, parent)
    def typed_relations(self) -> Iterator[tuple[str, str, str]]  # (subj, rel, obj)
```

Edge iterators skip any edge that touches an obsolete term. `fid(term_id)`
maps an id to a functor-safe symbol (`"HP:0001250"` -> `"HP_0001250"`).

## Usage (load -> mount -> build_index -> retrieve)

```python
from sma.ontology import load_obo, mount
g = load_obo("data/hp.obo", name="hpo")
mounted = mount(g)                                   # MatchConfig(delta=2, rho=0.95)
index = mounted.build_index([("disease:1", ["HP:0001250", "HP:0001263"], None)])
hits = index.retrieve(mounted.build_case(["HP:0001250"]), k=10, shortlist=80)
print([h.case_id for h in hits])
```

`build_case` emits `stmt(fid(term_id), subject)` for each present term, and a
higher-order `stmt(rel, stmt(fid(s),subj), stmt(fid(o),subj))` for each typed
relation whose subject **and** object are both present.

## Registry and router

```python
from sma.ontology import OntologyRegistry, DomainRouter
reg = OntologyRegistry()
reg.register("hpo", "data/hp.obo")          # format inferred from extension
mounted = reg.get("hpo")                     # load + mount, cached

router = DomainRouter(reg)
router.register_prefix("HP:", "hpo")
router.register_domain("medicine", "hpo")
router.route(term_ids=["HP:0001250"])        # -> ["hpo"]
```

## Rule: merge WITHIN an ecosystem, route ACROSS

Ontologies in the same ecosystem (e.g. the OBO Foundry — HPO, GO, MONDO that
share cross-references and a common upper level) are **merged** into one mounted
lattice so analogies can span them. Ontologies from different ecosystems (e.g.
medical HPO vs. cybersecurity ATT&CK) are kept separate and the `DomainRouter`
selects the right one per query by id prefix or domain. Never merge across
ecosystems; route across, merge within.

## Validation

- `scripts/ontology_hpo_regression.py` — the HPO regression **gate** (reproduces
  the `rare_disease_test.py` numbers through the universal loader).
- `scripts/ontology_attack_demo.py` — the **second domain** (MITRE ATT&CK STIX),
  demonstrating the loader/registry/router generalize beyond medicine.
