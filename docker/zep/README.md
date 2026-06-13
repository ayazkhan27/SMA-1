# Zep / Graphiti SOTA Baseline

The Zep baseline uses [Graphiti](https://github.com/getzep/graphiti) — a temporal
knowledge-graph memory — running against FalkorDB as the graph store.

## Enabling the Zep variant

1. Install the Python package:

   ```bash
   pip install graphiti-core
   ```

2. Start FalkorDB (the graph store Graphiti writes to):

   ```bash
   docker compose -f docker/zep/docker-compose.yml up -d
   ```

3. Set the DeepSeek environment variables so Graphiti's extraction uses the same
   backbone as the other SMA-1 backends (equal-footing comparison):

   ```bash
   export SMA_DEEPSEEK_API_KEY="<your key>"
   export GRAPHITI_LLM_BASE_URL="https://api.deepseek.com/v1"
   export GRAPHITI_LLM_MODEL="deepseek-chat"
   ```

4. Run the drift battery:

   ```bash
   python3 -m pytest tests/test_drift.py -v
   ```

## Without these steps

Without `graphiti-core` installed, `ZEP_AVAILABLE` is `False` and the
`test_zep_imports_or_skips` test is **automatically skipped** with the message
"graphiti not installed; Zep baseline runs in its container". All other three
backends (ContextOnly, RagNotes, SmaMemory) continue to run normally.
