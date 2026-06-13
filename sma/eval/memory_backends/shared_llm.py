"""Extraction + answering shared by ALL variants (extraction held constant)."""
from __future__ import annotations
import json

_EXTRACT_SYS = ("Extract the durable user facts from the message as a JSON "
                "array of short strings. Only facts that could be asked about "
                "later. No commentary.")
_ANSWER_SYS = ("Answer the question using ONLY the provided memory items. "
               "If the memory contradicts itself, prefer the most recent. "
               "Answer concisely; if unknown, say 'unknown'.")

def extract_facts(llm, message: str) -> list[str]:
    """Call llm to extract durable facts from a chat message; returns list of strings."""
    out = llm.complete(
        [{"role": "system", "content": _EXTRACT_SYS},
         {"role": "user", "content": message}], max_tokens=300)
    try:
        facts = json.loads(out)
        return [str(f) for f in facts] if isinstance(facts, list) else []
    except (json.JSONDecodeError, TypeError):
        return []

def answer_from(llm, question: str, retrieved: list[str]) -> str:
    """Answer question using only the provided retrieved memory items."""
    mem = "\n".join(f"- {r}" for r in retrieved) or "(no memory)"
    out = llm.complete(
        [{"role": "system", "content": _ANSWER_SYS},
         {"role": "user", "content": f"Memory:\n{mem}\n\nQuestion: {question}"}],
        max_tokens=120)
    return out.strip()
