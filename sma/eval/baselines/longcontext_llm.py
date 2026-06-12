"""B6 long-context LLM baseline (blueprint section 8.1 B6).

Controls for "maybe you don't need retrieval at all": stuff the query session
plus its top-20 BM25 candidate precedents (labels included) into one prompt
and ask deepseek-chat (temperature 0, max_tokens 10) to label the query.

API key comes from SMA_DEEPSEEK_API_KEY or the repo .env, via the same lookup
the agent layer uses (sma.agent.llm._env_key). Errors get exactly one retry;
a second failure (or an unparseable reply) marks the row failed.
"""

from __future__ import annotations

import time

from sma.agent.llm import DEEPSEEK_BASE_URL, DEEPSEEK_KEY_ENV, DEEPSEEK_MODEL, _env_key

CANDIDATE_CHARS = 800
QUERY_CHARS = 1600

SYSTEM_PROMPT = (
    "You are an incident triage assistant. You label log sessions as Anomaly or "
    "Normal by analogy to labeled precedent sessions. Reply with exactly one word: "
    "Anomaly or Normal."
)


def build_prompt(query_text: str, precedents: list[tuple[str, str]]) -> str:
    """precedents: list of (label, session_text), already ranked."""
    lines = [
        "Query session (label unknown):",
        query_text[:QUERY_CHARS],
        "",
        "Labeled precedent sessions (retrieved by lexical similarity, most similar first):",
    ]
    for i, (label, text) in enumerate(precedents, start=1):
        lines.append(f"[{i}] ({label}) {text[:CANDIDATE_CHARS]}")
    lines.append("")
    lines.append(
        "Based on these precedents, is the query session Anomaly or Normal? "
        "Answer with exactly one word."
    )
    return "\n".join(lines)


def parse_label(content: str) -> str | None:
    lowered = content.lower()
    has_anomaly = "anomal" in lowered
    has_normal = "normal" in lowered
    if has_anomaly and not has_normal:
        return "Anomaly"
    if has_normal and not has_anomaly:
        return "Normal"
    if has_anomaly and has_normal:  # ambiguous reply
        first_a = lowered.find("anomal")
        first_n = lowered.find("normal")
        return "Anomaly" if first_a < first_n else "Normal"
    return None


class LongContextDeepSeek:
    def __init__(
        self,
        model: str = DEEPSEEK_MODEL,
        api_key: str | None = None,
        timeout: float = 60.0,
        retry_sleep: float = 2.0,
    ):
        self.model = model
        self.api_key = api_key or _env_key(DEEPSEEK_KEY_ENV)
        self.timeout = timeout
        self.retry_sleep = retry_sleep
        self.calls = 0
        self.failures: list[str] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _call_once(self, prompt: str) -> str:
        import httpx

        response = httpx.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 10,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        self.calls += 1
        usage = payload.get("usage", {})
        self.total_prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.total_completion_tokens += int(usage.get("completion_tokens", 0))
        return (payload["choices"][0]["message"]["content"] or "").strip()

    def classify(
        self, query_id: str, query_text: str, precedents: list[tuple[str, str]]
    ) -> str | None:
        """Return 'Anomaly'/'Normal', or None when the row failed (after one retry)."""
        if not self.api_key:
            self.failures.append(f"{query_id}: {DEEPSEEK_KEY_ENV} not set")
            return None
        prompt = build_prompt(query_text, precedents)
        last_error = ""
        for attempt in range(2):  # one initial call + one retry
            try:
                content = self._call_once(prompt)
                label = parse_label(content)
                if label is not None:
                    return label
                last_error = f"unparseable reply: {content!r}"
            except Exception as exc:  # noqa: BLE001 - any transport/API error retries once
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt == 0:
                time.sleep(self.retry_sleep)
        self.failures.append(f"{query_id}: {last_error}")
        return None
