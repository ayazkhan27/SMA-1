# SMA-1 as an MCP server

`sma.mcp` exposes the structure-mapping memory over the **Model Context Protocol**, so
an agentic LLM (Codex CLI, Claude, the OpenAI Agents SDK, …) can mount curated
ontologies and retrieve **structurally-analogous** prior cases — by logical structure
(is-a subsumption + typed relations + rarity weighting), with a checkable structural
citation, a cite-or-abstain decision, and an expectation-violation **novelty** flag.
This is the analogical-memory layer for a discovery loop: the LLM generates and
verifies; SMA grounds each step in real precedent and flags the genuinely never-seen —
which vector RAG structurally cannot do.

## Install & run

```bash
pip install -e ".[mcp]"          # adds the `mcp` SDK + the `sma-mcp` console script
python -m sma.mcp                 # starts a stdio MCP server (or: sma-mcp)
```

## Wire into Codex CLI

Add an MCP server block to `~/.codex/config.toml` (see `examples/codex_config.toml`):

```toml
[mcp_servers.sma]
command = "python"
args = ["-m", "sma.mcp"]
env = { SMA_MANIFEST = "/abs/path/to/sma_manifest.json" }
```

or `codex mcp add sma -- python -m sma.mcp`. Confirm with `/mcp` in the Codex TUI.
(Refs: [Codex MCP docs](https://developers.openai.com/codex/mcp); the same slot Harvard's
Zitnik Lab uses to wire Codex into an [AI scientist via ToolUniverse](https://zitniklab.hms.harvard.edu/ToolUniverse/guide/building_ai_scientists/codex_cli.html).)

## Tools

| Tool | Purpose |
|---|---|
| `list_ontologies` | Registered ontologies, concept/case counts, supported formats |
| `mount_ontology(name, path, format="auto")` | Mount a curated ontology (OBO/OWL auto; or stix, cpc, xbrl, cwe, capec, mitre_xml, rdf) |
| `index_cases(ontology, cases)` | Add cases `{key, term_ids, text?}`; `key` is what gets cited |
| `encode_text(ontology, text)` | Deterministic text → term-ids by term-name match (no LLM) |
| `retrieve(ontology, text\|term_ids, k, ground_threshold?)` | Structural analogs + citations + abstain + novelty |
| `novelty(ontology, text\|term_ids)` | Expectation-violation score + flag (high = never-seen) |

`retrieve` returns ranked `citations` `{id, score, confidence, rank}`, an `abstain`
boolean (true ⇒ no structural precedent — the LLM must not fabricate), and a `novelty`
score. Cite-or-abstain gates on the **raw** grounding score (the normalized confidence
saturates and doesn't separate known/unknown), so set a per-ontology `ground_threshold`.

## Configure many ontologies

Point `SMA_MANIFEST` at a JSON manifest (`examples/sma_manifest.example.json`). Ontologies
mount **lazily** on first use, so registering ~100 is cheap. Env knobs:
`SMA_GROUND_THRESHOLD` (default cite-or-abstain cut), `SMA_NOVELTY_THRESHOLD` (default 0.5).

## The discovery loop

```
Codex (the reasoner)  ── "find a method that beats SOTA on X"
   │
   ├─ corpus MCP (ToolUniverse / journals / benchmarks) ── pulls papers, datasets
   │
   ├─ your encoder ── paper → SMA case (term-ids over the field's ontology)
   │
   └─ SMA MCP ── retrieve structural analogs + citation + novelty flag
        │
        └─ Codex reasons over the analogs → proposes → a benchmark harness RUNS it → iterate
```

## Honest scope (read this)

- **You supply the ontology + the encoder.** SMA's edge exists only where a curated,
  discriminative ontology exists and cases are encoded as relational structure over its
  term-ids. The bundled `encode_text` is a lexical starter; production recall needs a
  domain encoder (deterministic by design — no LLM in the extraction path). Without this
  you get parity with vector RAG, not an edge.
- **SMA retrieves and flags; it does not discover or verify.** The creative leap is the
  LLM's; "beats SOTA" is settled by a benchmark harness, never by SMA.
- **Analogy ≠ causality.** Structure-mapping finds relational similarity ("A is to B as C
  is to D"); it does not establish causation. On real domains the measured advantage is
  carried by subsumption + rarity weighting; the higher-order-relation machinery is
  decisive only on a synthetic structure-only control. Lean on subsumption + rarity +
  novelty-flagging.

## Calibrating the abstain threshold

`ground_threshold` is in raw-score units, per index. The repo's method
(`sma/eval/agentic_qa/`) sets it by Youden's J on a held-out known/unknown split using
retrieval scores only — no LLM spend, no leakage. Borrow it, freeze the number, set it
per ontology (manifest `ground_threshold` or `SMA_GROUND_THRESHOLD`).
