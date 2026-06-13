"""The one-shot QA agent for the Phase 5 LLM-QA "trustworthy specialist" phase.

A single :class:`QAAgent` holds the LLM and prompt FIXED and swaps only the
retrieval ``Memory`` (none / dense-RAG / SMA), exactly as registered in
``configs/preregistration_v2_llmqa.md`` section 2. For each
:class:`~sma.eval.agentic_qa.pools.QAItem` it runs one agent turn and returns a
result dict carrying every field the trustworthy-QA metrics read
(``sma.eval.agentic_qa.metrics``): ``gold_id``, ``gold_name``, ``answerable``,
``novel``, ``abstained``, ``pred_id``, ``answer``, ``novelty_flag``,
``confidence``, ``grounding_score``.

Two grounding regimes:

* **grounded** (a memory is given) — retrieve top-k candidates, render them as a
  numbered list, and ask the LLM for a strict one-line JSON ``{"choice": <n>}``
  where ``n`` is a candidate number or ``0`` to abstain. ``pred_id`` is the
  chosen candidate's key (the disease id), so correctness/citation can be checked
  structurally against the gold. When a calibrated ``score_threshold`` is given,
  a case whose top RAW grounding score falls below it is abstained AND flagged
  novel *before* the LLM call (the structural score, not the saturated confidence
  or the expectation-violation flag, is what separates known from unknown); with
  no threshold the novelty flag falls back to ``memory.novelty(query)``.
* **closed-book** (``memory is None``) — the LLM answers from the case alone with
  a strict one-line JSON ``{"diagnosis": "<name or ABSTAIN>"}``; ``pred_id`` is
  ``None`` (no citation), ``confidence`` is a flat ``0.5``, and novelty is N/A.

JSON parsing is defensive (strips ```` ``` ```` code fences, scans for the first
``{...}`` object) and falls back to ABSTAIN on any parse/validation failure, so a
malformed model reply degrades to the safe action rather than crashing the run.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from sma.eval.agentic import Query
from sma.eval.agentic_qa.pools import QAItem

# How many characteristic feature names to show per candidate (keeps the prompt
# bounded; the candidate is identified by its number, not by an exhaustive list).
_FEATURES_PER_CANDIDATE = 6

SYSTEM_PROMPT = (
    "You are a careful diagnostic assistant. You are given a clinical case and a "
    "numbered list of candidate diseases retrieved from a grounded knowledge base, "
    "each with a few of its characteristic features. Choose the single candidate "
    "whose characteristic features best match the case. Answer ONLY when a "
    "candidate genuinely grounds the case; if none of the candidates fit, abstain. "
    "Reply with STRICT one-line JSON and nothing else: "
    '{"choice": <candidate number, or 0 for none / abstain>}.'
)

CLOSED_BOOK_SYSTEM_PROMPT = (
    "You are a careful diagnostic assistant. You are given a clinical case and no "
    "external knowledge. Name the single most likely disease, or abstain if you are "
    "not confident. Reply with STRICT one-line JSON and nothing else: "
    '{"diagnosis": "<disease name, or ABSTAIN>"}.'
)

ABSTAIN = "ABSTAIN"


class LLM(Protocol):
    """The fixed LLM backend (``DeepSeekOrchestrator`` or a mock in tests)."""

    def complete(
        self, messages: list[dict], max_tokens: int = 600, temperature: float = 0.0
    ) -> str: ...


class MockLLM:
    """A deterministic stand-in for the real LLM (NEVER calls DeepSeek).

    Used by the tests and the ``--mock`` driver so the whole harness can run with
    zero API spend. By default it picks candidate ``1`` in the grounded regime and
    echoes a fixed diagnosis closed-book; pass ``choice`` / ``diagnosis`` to script
    other behaviours (e.g. ``choice=0`` to exercise the abstain path). When
    ``raw`` is set it is returned verbatim, to test defensive JSON parsing.
    """

    def __init__(
        self,
        choice: int = 1,
        diagnosis: str = "Mock disease",
        raw: str | None = None,
    ):
        self.choice = choice
        self.diagnosis = diagnosis
        self.raw = raw
        self.calls: list[list[dict]] = []

    def complete(
        self, messages: list[dict], max_tokens: int = 600, temperature: float = 0.0
    ) -> str:
        self.calls.append(messages)
        if self.raw is not None:
            return self.raw
        # Closed-book prompts ask for a "diagnosis" key; grounded ask for "choice".
        system = messages[0]["content"] if messages else ""
        if "diagnosis" in system:
            return json.dumps({"diagnosis": self.diagnosis})
        return json.dumps({"choice": self.choice})


def _strip_fences(text: str) -> str:
    """Drop Markdown code fences so JSON wrapped in ```` ```json ... ``` ```` parses."""
    t = text.strip()
    if t.startswith("```"):
        # Remove the opening fence (with optional language tag) and closing fence.
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t.strip())
    return t.strip()


def _parse_json_object(text: str) -> dict | None:
    """Best-effort parse of a single JSON object from a (possibly noisy) reply.

    Tries the whole stripped string first, then falls back to the first balanced
    ``{...}`` substring. Returns ``None`` when nothing parses to a dict.
    """
    stripped = _strip_fences(text)
    for candidate in (stripped, _first_brace_object(stripped)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _first_brace_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, or ``None``."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


class QAAgent:
    """One-shot retrieve-then-answer agent with a swappable retrieval memory.

    ``memory`` is one of the frozen ``Memory`` retrievers (``SmaMemory`` /
    ``DenseMemory`` / ...) or ``None`` for the closed-book condition. ``key_to_name``
    / ``key_to_terms`` map an :class:`IndexItem` key (disease id) to its display
    name and its ontology term ids, used to render the numbered candidate list;
    pass the same maps that back the indexed knowledge. ``k`` is the retrieval
    depth and ``novelty_threshold`` is the cut for the ``expectation_violation``
    novelty flag (only meaningful for SMA).
    """

    def __init__(
        self,
        llm: LLM,
        memory: Any | None,
        *,
        key_to_name: dict[str, str] | None = None,
        key_to_terms: dict[str, frozenset[str]] | None = None,
        k: int = 5,
        novelty_threshold: float = 0.5,
        score_threshold: float | None = None,
    ):
        self.llm = llm
        self.memory = memory
        self.key_to_name = key_to_name or {}
        self.key_to_terms = key_to_terms or {}
        self.k = k
        self.novelty_threshold = novelty_threshold
        # Calibrated cite-or-abstain: the RAW structural grounding score (not the
        # saturated normalized confidence, nor the expectation-violation flag — both
        # of which fail to separate known/unknown, AUROC~0.48) is the abstention
        # signal. Below this threshold the memory has no grounding -> abstain + flag
        # novel, WITHOUT spending an LLM call. None = no gate (LLM-only abstention).
        self.score_threshold = score_threshold

    # -- rendering ----------------------------------------------------------
    def _feature_text(self, key: str) -> str:
        """A few characteristic feature NAMES for a candidate disease."""
        terms = sorted(self.key_to_terms.get(key, frozenset()))
        names = [self._term_name(t) for t in terms[:_FEATURES_PER_CANDIDATE]]
        return ", ".join(n for n in names if n)

    def _term_name(self, term_id: str) -> str:
        """Resolve a term id to a human name via the SMA ontology when available."""
        mounted = getattr(self.memory, "mounted", None)
        if mounted is not None:
            term = mounted.graph.terms.get(term_id)
            if term is not None and term.name:
                return term.name
        return term_id

    def _render_candidates(self, retrieved: list) -> tuple[str, list[str]]:
        """Build the numbered candidate block and the parallel key list.

        Returns ``(text, keys)`` where ``keys[i]`` is the disease id of candidate
        ``i + 1`` (so a parsed ``{"choice": n}`` maps to ``keys[n - 1]``).
        """
        lines: list[str] = []
        keys: list[str] = []
        for i, r in enumerate(retrieved, 1):
            keys.append(r.key)
            name = self.key_to_name.get(r.key, r.key)
            features = self._feature_text(r.key)
            feat = f" -- characteristic features: {features}" if features else ""
            lines.append(f"[{i}] {name}{feat}")
        return "\n".join(lines), keys

    # -- answer -------------------------------------------------------------
    def answer(self, item: QAItem) -> dict:
        """Run one agent turn over ``item`` and return the metrics result dict."""
        if self.memory is None:
            return self._answer_closed_book(item)
        return self._answer_grounded(item)

    def _result(
        self,
        item: QAItem,
        *,
        abstained: bool,
        pred_id: str | None,
        answer: str,
        novelty_flag: bool,
        confidence: float,
        grounding_score: float | None,
    ) -> dict:
        """Assemble the per-item result dict the trustworthy-QA metrics read."""
        return {
            "gold_id": item.gold_id,
            "gold_name": item.gold_name,
            "answerable": item.answerable,
            "novel": item.novel,
            "abstained": abstained,
            "pred_id": pred_id,
            "answer": answer,
            "novelty_flag": novelty_flag,
            "confidence": confidence,
            # The RAW top structural grounding score (None closed-book). This is
            # the signal that actually separates known from unknown; the metrics
            # use it for threshold-free discrimination AUROC.
            "grounding_score": grounding_score,
        }

    def _answer_grounded(self, item: QAItem) -> dict:
        query = Query(item.case_terms, item.case_text)
        retrieved = self.memory.retrieve(query, self.k)
        confidence = retrieved[0].confidence if retrieved else 0.0
        grounding_score = retrieved[0].score if retrieved else 0.0

        # Calibrated cite-or-abstain. If the top RAW grounding score is below the
        # validation-calibrated threshold, the memory does not structurally ground
        # this case -> ABSTAIN and FLAG NOVEL, WITHOUT spending an LLM call. The
        # raw structural match score is the discriminating signal (answerable vs
        # out-of-knowledge AUROC ~0.84); the squashed confidence (top hit always
        # ~1.0) and the expectation-violation flag are not (AUROC ~0.48). A None
        # threshold disables the gate -> pure LLM-mediated abstention (legacy).
        if self.score_threshold is not None and grounding_score < self.score_threshold:
            return self._result(
                item,
                abstained=True,
                pred_id=None,
                answer=ABSTAIN,
                novelty_flag=True,
                confidence=confidence,
                grounding_score=grounding_score,
            )

        candidates_text, keys = self._render_candidates(retrieved)

        # With a calibrated gate, the structural signal IS the novelty signal:
        # above threshold here -> not flagged. Without a gate, fall back to the
        # memory's own expectation-violation novelty vs novelty_threshold.
        if self.score_threshold is not None:
            novelty_flag = False
        else:
            novelty_flag = bool(self.memory.novelty(query) > self.novelty_threshold)

        user = (
            f"Clinical case:\n{item.case_text}\n\n"
            f"Candidate diseases:\n{candidates_text or '(none retrieved)'}\n\n"
            "Rule: choose the candidate whose characteristic features best match "
            "the case; answer only if a candidate genuinely grounds the case, "
            "otherwise choose 0 to abstain.\n"
            'Reply with STRICT one-line JSON: {"choice": <candidate number or 0>}.'
        )
        reply = self.llm.complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=600,
            temperature=0.0,
        )
        choice = self._parse_choice(reply, n_candidates=len(keys))

        if choice == 0:
            pred_id: str | None = None
            answer = ABSTAIN
            abstained = True
        else:
            pred_id = keys[choice - 1]
            answer = self.key_to_name.get(pred_id, pred_id)
            abstained = False

        return self._result(
            item,
            abstained=abstained,
            pred_id=pred_id,
            answer=answer,
            novelty_flag=novelty_flag,
            confidence=confidence,
            grounding_score=grounding_score,
        )

    def _answer_closed_book(self, item: QAItem) -> dict:
        user = (
            f"Clinical case:\n{item.case_text}\n\n"
            "Name the single most likely disease, or abstain if not confident.\n"
            'Reply with STRICT one-line JSON: {"diagnosis": "<disease name or ABSTAIN>"}.'
        )
        reply = self.llm.complete(
            [
                {"role": "system", "content": CLOSED_BOOK_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=600,
            temperature=0.0,
        )
        diagnosis = self._parse_diagnosis(reply)
        abstained = diagnosis.strip().upper() == ABSTAIN
        answer = ABSTAIN if abstained else diagnosis

        return self._result(
            item,
            abstained=abstained,
            pred_id=None,
            answer=answer,
            novelty_flag=False,
            confidence=0.5,
            grounding_score=None,
        )

    # -- parsing ------------------------------------------------------------
    @staticmethod
    def _parse_choice(reply: str, *, n_candidates: int) -> int:
        """Parse ``{"choice": n}`` -> int in ``0..n_candidates``; abstain on failure.

        Any parse error, missing/ill-typed ``choice``, or out-of-range index
        collapses to ``0`` (abstain), the safe action.
        """
        obj = _parse_json_object(reply)
        if obj is None or "choice" not in obj:
            return 0
        try:
            choice = int(obj["choice"])
        except (TypeError, ValueError):
            return 0
        if choice < 0 or choice > n_candidates:
            return 0
        return choice

    @staticmethod
    def _parse_diagnosis(reply: str) -> str:
        """Parse ``{"diagnosis": "..."}`` -> str; abstain on failure."""
        obj = _parse_json_object(reply)
        if obj is None:
            return ABSTAIN
        value = obj.get("diagnosis")
        if not isinstance(value, str) or not value.strip():
            return ABSTAIN
        return value.strip()
