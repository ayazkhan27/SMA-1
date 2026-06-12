# ADR-003: Canonical MAC/FAC Retrieval and LLM Orchestration Boundary

## Context

The fixture report showed `mapping_f1=1.0` but `r1=0.0`, which means SME could map the
correct analog once supplied, while MAC/FAC retrieval screened it out. The root cause was that
SME used canonicalized functors, but MAC vectors and upper bounds used raw functor names.

The UI also needs an agentic LLM comparison experience without violating the blueprint rule that
LLMs never perform extraction or retrieval.

## Decision

- Canonicalize MAC/FAC content and WL-1 features with the same canonicalizer used by SME.
- Evaluate SSB retrieval in forced-choice analog-vs-distractor form for report rows.
- Add a local quantized Qwen GGUF orchestrator via optional `llama-cpp-python`; default model is
  `Qwen/Qwen2.5-0.5B-Instruct-GGUF` / `qwen2.5-0.5b-instruct-q4_k_m.gguf`, selected because the
  laptop has CPU-only inference and about 8 GiB available RAM.
- Keep the LLM strictly downstream of deterministic extraction and retrieval. It receives evidence
  from one selected mode: SMA, RAG, knowledge graph, or context-only.

## Consequences

The retrieval red flag is now covered by G4 tests. The Gradio app can compare generated or
fallback evidence-grounded responses across memory modes without letting the LLM write facts or
affect candidate generation.
