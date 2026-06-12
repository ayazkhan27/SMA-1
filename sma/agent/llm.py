"""LLM orchestration layer.

The LLM is intentionally downstream of extraction and retrieval. It receives a
mode, a user question, and already-retrieved evidence, then verbalizes an
answer. It cannot write facts to memory or affect candidate generation.

Two interchangeable backends:
- LocalOrchestrator:    quantized Qwen GGUF via llama-cpp (CPU, offline)
- DeepSeekOrchestrator: DeepSeek's OpenAI-compatible API (needs
                        SMA_DEEPSEEK_API_KEY in the environment or repo .env)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
DEFAULT_MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
DEFAULT_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_KEY_ENV = "SMA_DEEPSEEK_API_KEY"

SYSTEM_PROMPT = (
    "You are the answer writer for SMA-1, an agentic memory system. Retrieval is already "
    "complete; you only verbalize. The evidence items are PAST incidents retrieved from memory "
    "as candidate precedents for the user's input. If the input is itself a new incident (e.g. "
    "raw log lines), do NOT look for those literal entries - hostnames, dates and IDs will "
    "always differ. Instead say which precedent has the most similar failure pattern (the "
    "sequence/causal shape of events), what happened in it, and what that suggests here, citing "
    "items like [1]. Only when no precedent shares even the failure pattern, say so plainly. "
    "For ordinary questions, answer strictly from the evidence. Reply in 2-5 short sentences. "
    "Never repeat yourself."
)


MAX_HISTORY_TURNS = 8


def build_messages(
    question: str, mode: str, evidence: list[dict], history: list[dict] | None = None
) -> list[dict]:
    # Plain numbered texts only: provenance hashes and scores are for the UI
    # and audit trail, not the verbalizer — small models parrot them back.
    # 900 chars/item: the H3 judge pass showed a 400-char cap truncates exactly
    # the anomaly lines questions ask about (abstention artifacts); 900 x 5
    # items stays within the local model's 4k context alongside chat history.
    def _item(i: int, row: dict) -> str:
        why = row.get("alignment")
        head = f"[{i + 1}]" + (f" (why retrieved: {why})" if why else "")
        return f"{head} {row.get('text', '')[:900]}"

    evidence_text = "\n".join(_item(i, row) for i, row in enumerate(evidence[:8]))
    window_caveat = (
        "\nCaveat: each evidence item is one bounded session window. Events outside a "
        "window are not recorded in it - the absence of an event in the evidence is NOT "
        "evidence that it did not happen. Never infer outcomes from absence."
        if evidence
        else ""
    )
    user = (
        f"Evidence retrieved by the '{mode}' memory for the latest question:\n"
        f"{evidence_text or '(none retrieved)'}{window_caveat}\n\n"
        f"Question: {question}"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-MAX_HISTORY_TURNS * 2:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user})
    return messages


def _env_key(name: str) -> str | None:
    """Read a key from the environment, falling back to the repo .env file."""
    value = os.environ.get(name)
    if value:
        return value
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return None


@dataclass(frozen=True)
class OrchestratorConfig:
    model_path: str = DEFAULT_MODEL_PATH
    n_ctx: int = 4096
    n_threads: int = 8
    temperature: float = 0.2
    max_tokens: int = 220
    repeat_penalty: float = 1.25
    top_p: float = 0.9


class LocalOrchestrator:
    name = "local"

    def __init__(self, config: OrchestratorConfig | None = None):
        self.config = config or OrchestratorConfig(
            model_path=os.environ.get("SMA_LLM_MODEL", DEFAULT_MODEL_PATH)
        )
        self._llm = None
        self._load_error: str | None = None

    @property
    def status(self) -> dict:
        if self._llm is not None:
            return {
                "backend": "llama_cpp",
                "model": Path(self.config.model_path).name,
                "loaded": True,
            }
        path = Path(self.config.model_path)
        return {
            "backend": "deterministic_fallback" if not path.exists() else "llama_cpp",
            "model": Path(self.config.model_path).name,
            "loaded": False,
            "load_error": self._load_error or ("model file missing" if not path.exists() else ""),
            "recommended_model": f"{DEFAULT_MODEL_REPO}/{DEFAULT_MODEL_FILE}",
        }

    def _ensure_loaded(self) -> bool:
        if self._llm is not None:
            return True
        path = Path(self.config.model_path)
        if not path.exists():
            self._load_error = "model file missing; run scripts/fetch_model.py"
            return False
        try:
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=str(path),
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads,
                verbose=False,
            )
            return True
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            self._load_error = str(exc)
            return False

    def answer(
        self, question: str, mode: str, evidence: list[dict], history: list[dict] | None = None
    ) -> str:
        if not self._ensure_loaded():
            return fallback_answer(question, mode, evidence, self.status)
        # Chat completion uses the model's own instruct template (raw completion
        # makes small Qwen models ramble); repeat_penalty + tight max_tokens
        # stop the looping/repetition failure mode of 0.5B quantized models.
        response = self._llm.create_chat_completion(  # type: ignore[union-attr]
            messages=build_messages(question, mode, evidence, history),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repeat_penalty=self.config.repeat_penalty,
        )
        text = (response["choices"][0]["message"]["content"] or "").strip()
        return text or fallback_answer(question, mode, evidence, self.status)


class DeepSeekOrchestrator:
    name = "deepseek"

    def __init__(self, model: str = DEEPSEEK_MODEL, api_key: str | None = None):
        self.model = model
        self._api_key = api_key or _env_key(DEEPSEEK_KEY_ENV)
        self._last_error: str | None = None

    @property
    def status(self) -> dict:
        return {
            "backend": "deepseek_api",
            "model": self.model,
            "key_present": bool(self._api_key),
            "last_error": self._last_error or "",
        }

    def answer(
        self, question: str, mode: str, evidence: list[dict], history: list[dict] | None = None
    ) -> str:
        if not self._api_key:
            self._last_error = f"{DEEPSEEK_KEY_ENV} not set"
            return fallback_answer(question, mode, evidence, self.status)
        try:
            import httpx

            response = httpx.post(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self.model,
                    "messages": build_messages(question, mode, evidence, history),
                    "temperature": 0.3,
                    "max_tokens": 400,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            self._last_error = None
            text = (response.json()["choices"][0]["message"]["content"] or "").strip()
            return text or fallback_answer(question, mode, evidence, self.status)
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            return fallback_answer(question, mode, evidence, self.status)


def fallback_answer(question: str, mode: str, evidence: list[dict], status: dict) -> str:
    if not evidence:
        return (
            f"No evidence was retrieved for `{mode}`. Local LLM status: "
            f"{status.get('load_error') or status.get('last_error') or status.get('backend')}."
        )
    top = evidence[0]
    lines = [
        f"Mode `{mode}` retrieved {len(evidence)} evidence item(s).",
        f"Top evidence: {top.get('text', '')}",
        f"Provenance: {top.get('provenance', top.get('source_id', ''))}",
        "Local LLM is not available, so this is a deterministic evidence summary "
        f"rather than generated prose ({status.get('load_error') or status.get('last_error') or 'no backend'}).",
    ]
    return "\n".join(lines)


default_orchestrator = LocalOrchestrator()
default_deepseek = DeepSeekOrchestrator()
