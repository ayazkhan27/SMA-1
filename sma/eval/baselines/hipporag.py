"""HippoRAG-2-style KG retrieval comparator (blueprint B5, deterministic adaptation).

Adapted from HippoRAG / HippoRAG 2 (Gutierrez et al., NeurIPS'24 / ICML'25):
  - a phrase graph over OpenIE triples with passage (document) nodes,
  - synonym edges linking near-identical phrases,
  - Personalized PageRank (damping 0.5, as published) with personalization
    mass on query phrase nodes, weighted by node specificity 1/df
    (HippoRAG's inverse passage-frequency seed weighting),
  - documents scored by the PPR mass landing on their document nodes
    (chosen over summing contained-entity mass: it is the published
    HippoRAG 2 passage-node scoring and needs no extra idf heuristic).

Substitutions for a fair, deterministic, LLM-free comparison:
  - LLM OpenIE is replaced by rule-based triple extraction: per log/code
    line, regex-extracted entity-like tokens (block ids, IPs, hostnames,
    paths, hex ids, dotted/CamelCase identifiers, content words) and a
    fixed verb lexicon for relations; lines with no relation token fall
    back to entity co-occurrence edges.
  - The embedding-based synonym model is replaced by case/punctuation-
    normalized string equality plus token-Jaccard >= 0.8 on split
    identifiers.
No randomness anywhere: iteration is over sorted structures and PageRank
is the deterministic scipy power iteration.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import networkx as nx

DAMPING = 0.5  # HippoRAG's published PPR damping factor
JACCARD_SYNONYM = 0.8
MAX_ENTITIES_PER_LINE = 16

# Pattern priority matters: earlier patterns mask their spans so later,
# more generic ones do not re-extract fragments.
ENTITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"blk_-?\d+"),                                  # HDFS block ids
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"),       # IPv4(:port)
    re.compile(r"\b0x[0-9a-fA-F]+\b|\b[0-9a-f]{8,}\b"),        # hex ids
    re.compile(r"(?<![\w/])/(?:[\w.\-]+/)+[\w.\-]+"),          # file paths
    re.compile(r"\b[A-Za-z_][\w$\-]*(?:\.[A-Za-z_][\w$\-]*)+\b"),  # dotted: classes, hostnames
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b"),      # CamelCase (exceptions, services)
    re.compile(r"\b[A-Za-z][\w]*(?:[-_][\w]+)+\b"),            # snake/hyphen identifiers
]
WORD_PATTERN = re.compile(r"\b[A-Za-z]{2,}\b")

STOPWORDS = frozenset(
    "the a an to of in on at for is was were be been being and or not with by "
    "from this that it its as are has have had while when then than but if "
    "else into over under after before during no none null true false via per "
    "info debug trace warn".split()
)
RELATION_LEXICON = frozenset(
    "received receiving sent send sending terminating terminated starting "
    "started start stop stopping stopped failed failing fails fail connect "
    "connected connecting disconnect disconnected deleting deleted delete "
    "created creating create opened opening open closed closing close read "
    "reading write writing wrote allocated allocating exceeded aborted "
    "aborting retrying retried refused raised threw throw throws thrown "
    "caught calling called returned returning killed killing launched "
    "launching completed completing finished exited timed waiting blocked "
    "serving served added adding removed removing updated updating "
    "registered succeeded".split()
)


def _normalize(phrase: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", phrase.lower())


def _phrase_tokens(phrase: str) -> frozenset[str]:
    """Split an identifier on case boundaries and punctuation for Jaccard."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", phrase)
    return frozenset(t.lower() for t in re.split(r"[^0-9A-Za-z]+", spaced) if t)


def _line_triples(line: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]], list[str]]:
    """Extract (triples, co-occurrence pairs, all entities) from one line."""
    found: list[tuple[int, str, str]] = []
    masked = line
    for pattern in ENTITY_PATTERNS:
        for m in pattern.finditer(masked):
            found.append((m.start(), "entity", m.group(0)))
        masked = pattern.sub(lambda m: " " * len(m.group(0)), masked)
    for m in WORD_PATTERN.finditer(masked):
        word = m.group(0)
        lower = word.lower()
        if lower in STOPWORDS:
            continue
        if lower in RELATION_LEXICON or (len(lower) >= 5 and lower.endswith(("ing", "ed"))):
            found.append((m.start(), "relation", lower))
        else:
            found.append((m.start(), "entity", word))
    found.sort()
    ents = [(pos, s) for pos, kind, s in found if kind == "entity"][:MAX_ENTITIES_PER_LINE]
    rels = [(pos, s) for pos, kind, s in found if kind == "relation"]
    entities = [s for _, s in ents]
    triples: list[tuple[str, str, str]] = []
    pairs: list[tuple[str, str]] = []
    if rels:
        for (p1, e1), (p2, e2) in zip(ents, ents[1:]):
            between = [r for pr, r in rels if p1 < pr < p2]
            rel = between[0] if between else rels[0][1]
            triples.append((e1, rel, e2))
    else:
        for i, (_, e1) in enumerate(ents):
            for _, e2 in ents[i + 1:]:
                if e1 != e2:
                    pairs.append((e1, e2))
    return triples, pairs, entities


def extract_phrases(text: str) -> list[str]:
    """Entity-like phrases for a whole text, in order of first appearance."""
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines() or [text]:
        _, _, entities = _line_triples(line)
        for e in entities:
            if e not in seen:
                seen.add(e)
                out.append(e)
    return out


class HippoRAGRetriever:
    """Phrase graph + Personalized PageRank retriever (see module docstring)."""

    def __init__(self, damping: float = DAMPING):
        self.damping = damping
        self.graph = nx.Graph()
        self.doc_ids: list[str] = []
        self._norm_index: dict[str, list[str]] = {}

    @staticmethod
    def _doc_node(doc_id: str) -> str:
        return f"d::{doc_id}"

    @staticmethod
    def _ent_node(phrase: str) -> str:
        return f"e::{phrase}"

    def _bump(self, u: str, v: str, w: float) -> None:
        if self.graph.has_edge(u, v):
            self.graph[u][v]["weight"] += w
        else:
            self.graph.add_edge(u, v, weight=w)

    def build(self, documents: list[tuple[str, str]]) -> None:
        self.graph = nx.Graph()
        self.doc_ids = [doc_id for doc_id, _ in documents]
        for doc_id, text in documents:
            dnode = self._doc_node(doc_id)
            self.graph.add_node(dnode, kind="doc")
            doc_counts: Counter[str] = Counter()
            for line in text.splitlines() or [text]:
                triples, pairs, entities = _line_triples(line)
                doc_counts.update(entities)
                for s, _rel, o in triples:
                    if s != o:
                        self._bump(self._ent_node(s), self._ent_node(o), 1.0)
                for e1, e2 in pairs:
                    self._bump(self._ent_node(e1), self._ent_node(e2), 1.0)
            for phrase in sorted(doc_counts):
                enode = self._ent_node(phrase)
                self.graph.add_node(enode, kind="entity")
                self._bump(enode, dnode, float(doc_counts[phrase]))
        self._add_synonym_edges()
        self._norm_index = defaultdict(list)
        for node in sorted(self.graph.nodes):
            if node.startswith("e::"):
                self._norm_index[_normalize(node[3:])].append(node)
        self._norm_index = dict(self._norm_index)

    def _add_synonym_edges(self) -> None:
        phrases = sorted(n[3:] for n in self.graph.nodes if n.startswith("e::"))
        by_norm: dict[str, list[str]] = defaultdict(list)
        for p in phrases:
            by_norm[_normalize(p)].append(p)
        for _norm, group in sorted(by_norm.items()):
            for i, p1 in enumerate(group):
                for p2 in group[i + 1:]:
                    self._bump(self._ent_node(p1), self._ent_node(p2), 1.0)
        # Token-Jaccard synonyms among multi-token identifiers sharing a token.
        token_sets = {p: _phrase_tokens(p) for p in phrases}
        by_token: dict[str, list[str]] = defaultdict(list)
        for p in phrases:
            if len(token_sets[p]) >= 2:
                for t in sorted(token_sets[p]):
                    by_token[t].append(p)
        compared: set[tuple[str, str]] = set()
        for _token, group in sorted(by_token.items()):
            for i, p1 in enumerate(group):
                for p2 in group[i + 1:]:
                    key = (p1, p2)
                    if key in compared or _normalize(p1) == _normalize(p2):
                        continue
                    compared.add(key)
                    t1, t2 = token_sets[p1], token_sets[p2]
                    jac = len(t1 & t2) / len(t1 | t2)
                    if jac >= JACCARD_SYNONYM:
                        self._bump(self._ent_node(p1), self._ent_node(p2), 1.0)

    def _specificity(self, node: str) -> float:
        df = sum(1 for nb in self.graph[node] if nb.startswith("d::"))
        return 1.0 / max(1, df)

    def retrieve(self, query_text: str, k: int = 10) -> list[tuple[str, float]]:
        if not self.doc_ids:
            return []
        seeds: dict[str, float] = {}
        for phrase in extract_phrases(query_text):
            for node in self._norm_index.get(_normalize(phrase), []):
                seeds[node] = max(seeds.get(node, 0.0), self._specificity(node))
        personalization = dict(sorted(seeds.items())) if seeds else None
        scores = nx.pagerank(
            self.graph,
            alpha=self.damping,
            personalization=personalization,
            weight="weight",
        )
        ranked = sorted(
            ((doc_id, float(scores.get(self._doc_node(doc_id), 0.0))) for doc_id in self.doc_ids),
            key=lambda row: (-row[1], row[0]),
        )
        return ranked[:k]


def rank_hipporag(query_text: str, documents: list[tuple[str, str]], k: int = 10) -> list[tuple[str, float]]:
    retriever = HippoRAGRetriever()
    retriever.build(documents)
    return retriever.retrieve(query_text, k=k)
