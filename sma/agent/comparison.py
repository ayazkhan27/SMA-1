"""Toggleable SMA / BM25 / dense-RAG / knowledge-graph / context-only comparison.

The modes mirror the LogHub evaluation baselines so what you see in the UI is
the same retrieval mathematics that produced reports/triage_metrics.csv:

- sma:             MAC/FAC retrieval + SME mapping with candidate inferences
- bm25:            lexical retrieval (rank_bm25 BM25Okapi over raw text)
- dense rag:       sentence-transformer embeddings (all-MiniLM-L6-v2), with a
                   deterministic TF-IDF fallback when the model is unavailable
- knowledge graph: deterministic Tier-0 entity graph, entity-overlap +
                   neighbor-bonus scoring (KG-PPR proxy)
- context only:    no retrieval; first k corpus items stuffed into context
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sma.encoders import get_encoder
from sma.encoders.coverage import coverage_warning, rule_coverage
from sma.encoders.draft_adapter import DraftAdapter
from sma.index.macfac import MacFacIndex
from sma.ir.schema import Case
from sma.match.engine import match_cases
from sma.match.explain import alignment_summary
from sma.match.infer import candidate_inferences
from sma.match.types import MatchConfig

from .llm import LocalOrchestrator, default_deepseek, default_orchestrator

MODES = ("sma", "bm25", "dense rag", "knowledge graph", "hybrid (fused)", "context only")
MODE_ALIASES = {
    "rag": "dense rag",
    "kg": "knowledge graph",
    "context": "context only",
    "hybrid": "hybrid (fused)",
    "rrf": "hybrid (fused)",
    "fused": "hybrid (fused)",
}
LLM_BACKENDS = ("local", "deepseek")

DENSE_MODEL_NAME = "all-MiniLM-L6-v2"

_dense_model = None
_dense_model_error: str | None = None


def dense_model():
    """Lazily load the sentence-transformer once per process; None if unavailable."""
    global _dense_model, _dense_model_error
    if _dense_model is None and _dense_model_error is None:
        try:
            from sentence_transformers import SentenceTransformer

            _dense_model = SentenceTransformer(DENSE_MODEL_NAME)
        except Exception as exc:
            _dense_model_error = f"{type(exc).__name__}: {exc}"
    return _dense_model


@dataclass
class CorpusItem:
    item_id: str
    text: str
    adapter_id: str
    case: Case
    label: str = ""


@dataclass
class ModeResult:
    mode: str
    answer: str
    evidence: list[dict]
    llm_status: dict


class ComparisonFramework:
    def __init__(self, orchestrator: LocalOrchestrator | None = None):
        self.items: list[CorpusItem] = []
        self.index = MacFacIndex()
        self.orchestrator = orchestrator or default_orchestrator
        self.orchestrators = {"local": self.orchestrator, "deepseek": default_deepseek}
        # Session flag: when set, logs-adapter encoding (corpus AND queries)
        # goes through the LLM-proposed draft adapter and every SMA evidence
        # row is provenance-stamped as unreviewed.
        self.draft_adapter: DraftAdapter | None = None
        # Per-corpus caches, rebuilt when _version changes.
        self._version = 0
        self._bm25 = None
        self._bm25_version = -1
        self._dense_embeddings = None
        self._dense_version = -1
        self._entity_graphs: list[dict[str, set[str]]] = []
        self._graph_version = -1

    def clear(self) -> None:
        self.items.clear()
        self.index = MacFacIndex(config=self.index.config)
        self._version += 1

    def set_scorer(self, scorer: str) -> None:
        """Switch the SMA scoring regime (ses or mdl) without reindexing.

        The MAC index structures are scorer-independent; only the FAC scoring
        step reads config, and its cache key already includes the scorer.
        """
        if scorer not in ("ses", "mdl", "surprisal"):
            raise ValueError(f"unknown scorer: {scorer!r}; expected 'ses', 'mdl' or 'surprisal'")
        self.index.config = MatchConfig(scorer=scorer)

    @property
    def draft_note(self) -> str | None:
        """Provenance stamp shown on every SMA evidence row and the chips strip."""
        if self.draft_adapter is None:
            return None
        return f"draft-adapter (LLM-proposed, unreviewed) hash={self.draft_adapter.draft_hash[:8]}"

    def apply_draft_adapter(self, adapter: DraftAdapter) -> int:
        """Re-encode the whole corpus through the draft adapter (texts/labels kept)."""
        self.draft_adapter = adapter
        return self._reencode_all()

    def revert_draft_adapter(self) -> int:
        """Restore base adapters: re-encode every item with its original encoder."""
        self.draft_adapter = None
        return self._reencode_all()

    def _reencode_all(self) -> int:
        self.index = MacFacIndex(config=self.index.config)
        for item in self.items:
            item.case = self._encode(item.text, item.adapter_id)
            self.index.add(item.case)
        self._version += 1
        return len(self.items)

    def _encode(self, text: str, adapter_id: str) -> Case:
        if self.draft_adapter is not None and adapter_id == "logs":
            return self.draft_adapter.encode(text).case
        return get_encoder(adapter_id).encode(text).case

    def add_document(self, text: str, adapter_id: str = "logs", label: str = "") -> CorpusItem:
        case = self._encode(text, adapter_id)
        item = CorpusItem(
            item_id=f"doc_{len(self.items)}",
            text=text,
            adapter_id=adapter_id,
            case=case,
            label=label,
        )
        self.items.append(item)
        self.index.add(case)
        self._version += 1
        return item

    def load_lines(
        self,
        corpus_text: str,
        adapter_id: str = "logs",
        max_items: int = 50,
        single_case: bool = False,
    ) -> list[CorpusItem]:
        """Load raw text as corpus items.

        single_case=True keeps the whole text as ONE incident. Otherwise text
        splits on blank lines into blocks; a single block with no blank lines
        falls back to one item per line (the legacy behavior that quietly
        explodes a pasted session - hence the explicit flag).
        """
        if single_case and corpus_text.strip():
            return [self.add_document(corpus_text.strip(), adapter_id=adapter_id)]
        added = []
        blocks = split_corpus(corpus_text)
        for block in blocks[:max_items]:
            added.append(self.add_document(block, adapter_id=adapter_id))
        return added

    def evidence_for(self, question: str, mode: str, adapter_id: str = "logs", k: int = 4) -> tuple[str, list[dict]]:
        """Resolve a mode name and run its retriever; returns (canonical_mode, evidence)."""
        mode = MODE_ALIASES.get(mode.lower(), mode.lower())
        if mode == "sma":
            return mode, self.sma_evidence(question, adapter_id, k)
        if mode == "bm25":
            return mode, self.bm25_evidence(question, k)
        if mode == "dense rag":
            return mode, self.dense_evidence(question, k)
        if mode == "knowledge graph":
            return mode, self.kg_evidence(question, k)
        if mode == "hybrid (fused)":
            return mode, self.hybrid_evidence(question, adapter_id, k)
        if mode == "context only":
            return mode, self.context_evidence(k)
        raise ValueError(f"unknown mode: {mode!r}; expected one of {MODES}")

    def ask(
        self,
        question: str,
        mode: str,
        adapter_id: str = "logs",
        k: int = 4,
        llm: str = "local",
        history: list[dict] | None = None,
    ) -> ModeResult:
        mode, evidence = self.evidence_for(question, mode, adapter_id=adapter_id, k=k)
        orchestrator = self.orchestrators.get(llm)
        if orchestrator is None:
            raise ValueError(f"unknown llm backend: {llm!r}; expected one of {LLM_BACKENDS}")
        answer = orchestrator.answer(question, mode, evidence, history=history)
        return ModeResult(mode=mode, answer=answer, evidence=evidence, llm_status=orchestrator.status)

    def ask_all(self, question: str, adapter_id: str = "logs", k: int = 4,
                modes: tuple[str, ...] | list[str] = MODES, llm: str = "local") -> dict[str, ModeResult]:
        return {mode: self.ask(question, mode, adapter_id=adapter_id, k=k, llm=llm) for mode in modes}

    # --- mode implementations -------------------------------------------------

    def sma_evidence(self, question: str, adapter_id: str, k: int) -> list[dict]:
        query_case = self._encode(question, adapter_id)
        # Lattice-miss tripwire (blueprint 12-R3): how much of the query's
        # vocabulary the frozen class rules actually cover.
        coverage = rule_coverage(
            question,
            extra_classes=self.draft_adapter.rules.classes if self.draft_adapter else None,
        )
        mode_detail = "SME mapping + MAC/FAC retrieval"
        if self.draft_note:
            mode_detail += f"; {self.draft_note}"
        # MAC/FAC budgets keep large corpora interactive: the MAC stage screens
        # everything, full SME mapping runs only on the budgeted shortlist.
        shortlist = min(max(k, len(self.items)), 200)
        fac_budget = 30 if len(self.items) > 100 else None
        results = self.index.retrieve(query_case, k=k, shortlist=shortlist, fac_budget=fac_budget)
        evidence = []
        case_to_item = {item.case.case_id: item for item in self.items}
        for result in results:
            item = case_to_item[result.case_id]
            gmap = match_cases(item.case, query_case, config=self.index.config)
            inferences = candidate_inferences(gmap)
            evidence.append(
                {
                    "source_id": item.item_id,
                    "label": item.label,
                    "score": f"{result.ses_n:.4f}",
                    "text": item.text,
                    "provenance": f"case={item.case.case_id}; ses_n={result.ses_n:.4f}; certified={result.certified}",
                    "mode_detail": mode_detail,
                    "alignment": alignment_summary(gmap),
                    "inferences": [inf.inference_sexpr for inf in inferences[:3]],
                    "coverage": coverage,
                }
            )
        warning = coverage_warning(coverage)
        if warning:
            evidence.insert(0, self._coverage_warning_row(coverage, warning))
        return evidence

    def _coverage_warning_row(self, coverage: dict, warning: str) -> dict:
        """Pseudo-row prepended to SMA evidence when coverage trips 12-R3."""
        return {
            "source_id": "coverage-tripwire",
            "label": "",
            "score": f"{coverage['fraction']:.2f}",
            "text": "",
            "provenance": (
                f"rule_coverage={coverage['covered_lines']}/{coverage['total_lines']} "
                "non-empty lines fired EVENT_CLASS_RULES (blueprint 12-R3)"
            ),
            "mode_detail": "structural coverage tripwire",
            "warning": warning,
            "coverage": coverage,
        }

    # --- candidate rankings shared by hybrid fusion ---------------------------

    def _bm25_ranking(self, question: str, n: int) -> list[tuple[str, float]]:
        if self._bm25_version != self._version:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([item.text.lower().split() for item in self.items])
            self._bm25_version = self._version
        scores = self._bm25.get_scores(question.lower().split())
        ranked = sorted(zip(self.items, scores), key=lambda row: (-row[1], row[0].item_id))
        return [(item.item_id, float(score)) for item, score in ranked[:n]]

    def _dense_ranking(self, question: str, n: int) -> list[tuple[str, float]]:
        model = dense_model()
        if model is None:
            from sma.eval.baselines.dense import rank_tfidf_dense

            return rank_tfidf_dense(
                question, [(item.item_id, item.text) for item in self.items], k=n
            )
        if self._dense_version != self._version:
            self._dense_embeddings = model.encode(
                [item.text for item in self.items], convert_to_tensor=True, show_progress_bar=False
            )
            self._dense_version = self._version
        from sentence_transformers import util

        query_embedding = model.encode(question, convert_to_tensor=True, show_progress_bar=False)
        sims = util.cos_sim(query_embedding, self._dense_embeddings)[0].cpu().tolist()
        ranked = sorted(zip(self.items, sims), key=lambda row: (-row[1], row[0].item_id))
        return [(item.item_id, float(score)) for item, score in ranked[:n]]

    def _sma_ranking(self, query_case: Case, n: int) -> list[tuple[str, float]]:
        shortlist = min(max(n, len(self.items)), 200)
        fac_budget = 30 if len(self.items) > 100 else None
        results = self.index.retrieve(query_case, k=n, shortlist=shortlist, fac_budget=fac_budget)
        case_to_item = {item.case.case_id: item for item in self.items}
        return [
            (case_to_item[result.case_id].item_id, result.ses_n)
            for result in results
            if result.case_id in case_to_item
        ]

    def hybrid_evidence(self, question: str, adapter_id: str, k: int) -> list[dict]:
        """RRF-fused bm25+dense+sma candidates, SME alignment receipts on each.

        Candidate generation is the union of the top-20 from each retriever,
        fused with reciprocal-rank fusion; the fused top-k then get full SME
        receipts (match_cases + alignment_summary) so accountability rides on
        every candidate regardless of which retriever found it.
        """
        if not self.items:
            return []
        from sma.eval.baselines.hybrid_rrf import rrf_fuse

        query_case = self._encode(question, adapter_id)
        coverage = rule_coverage(
            question,
            extra_classes=self.draft_adapter.rules.classes if self.draft_adapter else None,
        )
        n = 20
        rankings = {
            "bm25": self._bm25_ranking(question, n),
            "dense": self._dense_ranking(question, n),
            "sma": self._sma_ranking(query_case, n),
        }
        fused = rrf_fuse(list(rankings.values()), top_k=k)
        rank_of = {
            name: {doc_id: rank for rank, (doc_id, _score) in enumerate(ranking, start=1)}
            for name, ranking in rankings.items()
        }
        mode_detail = "RRF(bm25+dense+sma) candidates, SME alignment receipts"
        if self.draft_note:
            mode_detail += f"; {self.draft_note}"
        by_id = {item.item_id: item for item in self.items}
        evidence = []
        for doc_id, fused_score in fused:
            item = by_id[doc_id]
            gmap = match_cases(item.case, query_case, config=self.index.config)
            inferences = candidate_inferences(gmap)
            ranks = ", ".join(
                f"{name}={rank_of[name].get(doc_id, '-')}" for name in ("bm25", "dense", "sma")
            )
            evidence.append(
                {
                    "source_id": item.item_id,
                    "label": item.label,
                    "score": f"{fused_score:.4f}",
                    "text": item.text,
                    "provenance": f"rrf={fused_score:.4f}; ranks({ranks}); case={item.case.case_id}",
                    "mode_detail": mode_detail,
                    "alignment": alignment_summary(gmap),
                    "inferences": [inf.inference_sexpr for inf in inferences[:3]],
                    "coverage": coverage,
                }
            )
        warning = coverage_warning(coverage)
        if warning:
            evidence.insert(0, self._coverage_warning_row(coverage, warning))
        return evidence

    def bm25_evidence(self, question: str, k: int) -> list[dict]:
        if not self.items:
            return []
        if self._bm25_version != self._version:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([item.text.lower().split() for item in self.items])
            self._bm25_version = self._version
        scores = self._bm25.get_scores(question.lower().split())
        ranked = sorted(zip(self.items, scores), key=lambda row: (-row[1], row[0].item_id))
        return [
            {
                "source_id": item.item_id,
                "label": item.label,
                "score": f"{score:.4f}",
                "text": item.text,
                "provenance": f"bm25_okapi={score:.4f}",
                "mode_detail": "BM25 lexical retrieval (rank_bm25)",
            }
            for item, score in ranked[:k]
        ]

    def dense_evidence(self, question: str, k: int) -> list[dict]:
        if not self.items:
            return []
        model = dense_model()
        if model is None:
            from sma.eval.baselines.dense import rank_tfidf_dense

            ranked = rank_tfidf_dense(
                question, [(item.item_id, item.text) for item in self.items], k=k
            )
            by_id = {item.item_id: item for item in self.items}
            return [
                {
                    "source_id": item_id,
                    "label": by_id[item_id].label,
                    "score": f"{score:.4f}",
                    "text": by_id[item_id].text,
                    "provenance": f"tfidf_cosine={score:.4f}",
                    "mode_detail": (
                        "TF-IDF fallback (sentence-transformers unavailable: "
                        f"{_dense_model_error})"
                    ),
                }
                for item_id, score in ranked
            ]
        if self._dense_version != self._version:
            self._dense_embeddings = model.encode(
                [item.text for item in self.items], convert_to_tensor=True, show_progress_bar=False
            )
            self._dense_version = self._version
        from sentence_transformers import util

        query_embedding = model.encode(question, convert_to_tensor=True, show_progress_bar=False)
        sims = util.cos_sim(query_embedding, self._dense_embeddings)[0].cpu().tolist()
        ranked = sorted(zip(self.items, sims), key=lambda row: (-row[1], row[0].item_id))
        return [
            {
                "source_id": item.item_id,
                "label": item.label,
                "score": f"{score:.4f}",
                "text": item.text,
                "provenance": f"dense_cosine={score:.4f}; model={DENSE_MODEL_NAME}",
                "mode_detail": "Dense RAG (sentence-transformers)",
            }
            for item, score in ranked[:k]
        ]

    def kg_evidence(self, question: str, k: int) -> list[dict]:
        if not self.items:
            return []
        if self._graph_version != self._version:
            self._entity_graphs = [case_entity_graph(item.case) for item in self.items]
            self._graph_version = self._version
        q_entities = {token.lower() for token in entityish_tokens(question)}
        rows = []
        for item, graph in zip(self.items, self._entity_graphs):
            matched = q_entities & set(graph)
            neighbor_bonus = sum(len(graph[token]) for token in matched)
            score = len(matched) + 0.1 * neighbor_bonus
            rows.append((score, item, sorted(matched)))
        rows.sort(key=lambda row: (-row[0], row[1].item_id))
        return [
            {
                "source_id": item.item_id,
                "label": item.label,
                "score": f"{score:.4f}",
                "text": item.text,
                "provenance": f"entity_overlap_ppr_proxy={score:.4f}; matched={','.join(matched) or 'none'}",
                "mode_detail": "Tier-0 entity graph, overlap + neighbor bonus (KG-PPR proxy)",
            }
            for score, item, matched in rows[:k]
        ]

    def context_evidence(self, k: int) -> list[dict]:
        return [
            {
                "source_id": item.item_id,
                "label": item.label,
                "score": "context",
                "text": item.text,
                "provenance": "raw_context_window",
                "mode_detail": "No retrieval; first corpus items stuffed into context",
            }
            for item in self.items[:k]
        ]

    def corpus_table(self) -> list[list[str]]:
        return [
            [
                item.item_id,
                item.adapter_id,
                str(len(item.case.statements)),
                item.case.case_id[:12],
                item.text[:160].replace("\n", " "),
            ]
            for item in self.items
        ]


def split_corpus(corpus_text: str) -> list[str]:
    blocks = [block.strip() for block in corpus_text.split("\n\n") if block.strip()]
    if len(blocks) <= 1:
        blocks = [line.strip() for line in corpus_text.splitlines() if line.strip()]
    return blocks


def entityish_tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[A-Za-z0-9_.:/-]{2,}", text)


def case_entity_graph(case: Case) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for expr in case.expressions():
        entities = [entity.name.lower() for entity in expr.entities()]
        for left in entities:
            for right in entities:
                if left != right:
                    graph[left].add(right)
    return graph


def demo_corpus() -> str:
    return "\n".join(
        [
            "INFO DataNode blk_123 timeout connecting to 10.0.0.1",
            "WARN DataNode blk_123 retry after timeout",
            "ERROR DataNode blk_123 failed after retry",
            "INFO API service accepted request /checkout",
            "ERROR DB connection timeout caused retry storm in checkout service",
            "WARN worker restarted after queue saturation cleared",
        ]
    )


def challenge_corpus() -> str:
    """Adversarial incident library for mode comparison (logs adapter).

    Blank-line separated blocks become one case each. Designed traps:
    context-only grabs the flashy first block; BM25/dense are pulled by shared
    rare tokens; the KG proxy is pulled by shared component entities; only the
    causal anatomy (timeout -> retries -> failure) identifies true analogs.
    """
    return "\n\n".join(
        [
            # 0: context-only trap — flashy, totally unrelated, sits first.
            "ERROR AuthService invalid signature on admin token\n"
            "ERROR AuthService possible credential stuffing detected\n"
            "WARN AuthService source addresses blocked by waf",
            # 1: full cascade, payment vocabulary (structural target).
            "ERROR PaymentGateway connection timeout to db-shard-7\n"
            "WARN PaymentGateway retrying transaction batch\n"
            "WARN PaymentGateway retrying transaction batch\n"
            "ERROR PaymentGateway transaction failed after repeated retry\n"
            "ERROR PaymentGateway worker pool exhausted failure",
            # 2: benign timeout mention, no cascade (surface distractor).
            "INFO SearchService deployment completed successfully\n"
            "INFO SearchService timeout setting increased to 30s by operator\n"
            "INFO SearchService cache warmed and serving",
            # 3: healthy pipeline, BackupAgent (entity/component trap).
            "INFO BackupAgent nightly snapshot started on host 10.0.0.1\n"
            "INFO BackupAgent snapshot uploaded to object storage\n"
            "INFO BackupAgent snapshot rotation complete",
            # 4: transient blip sharing rare tokens (fetch/asset/bundle) with Q4.
            "WARN CacheNode fetch timeout for asset bundle\n"
            "INFO CacheNode fetch recovered on second attempt\n"
            "INFO CacheNode asset bundle served normally",
            # 5: retry storm, ApiEdge vocabulary (structural target for Q4).
            "WARN ApiEdge upstream timeout on route /search\n"
            "WARN ApiEdge retrying upstream call\n"
            "WARN ApiEdge retrying upstream call\n"
            "WARN ApiEdge retrying upstream call\n"
            "ERROR ApiEdge circuit breaker opened after retry failure",
            # 6: second cascade, broker vocabulary (alternate structural target).
            "ERROR MsgBroker session timeout waiting for heartbeat ack\n"
            "WARN MsgBroker consumer retrying fetch offset\n"
            "ERROR MsgBroker consumer fetch failed\n"
            "ERROR MsgBroker partition leader election failed",
            # 7: maintenance window, restart without any failure chain.
            "INFO Scheduler maintenance window opened\n"
            "INFO Scheduler workers restarted in rolling order\n"
            "INFO Scheduler maintenance window closed",
        ]
    )


__all__ = [
    "ComparisonFramework",
    "CorpusItem",
    "ModeResult",
    "MODES",
    "demo_corpus",
    "challenge_corpus",
]
