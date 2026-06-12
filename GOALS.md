# /goal SMA-MVP-1

MISSION: Build and evaluate Structure-Mapping Agentic Memory exactly per blueprint sections 2-8.

DEFINITION OF DONE:

- D1. `pytest -m "gate_G0 or gate_G1 or gate_G2 or gate_G3 or gate_G4 or gate_G5 or gate_G6"` passes.
- D2. Canonical battery includes water-flow/heat-flow and solar-system/atom mappings, plus SME v4 oracle fixtures.
- D3. Certified retrieval verified: best-first FAC top-k equals brute-force top-k on a fixed sample.
- D4. Preregistration exists and is frozen before test-set runs.
- D5. Full evaluation emits `reports/report.html` from `make report`.
- D6. Every agent-surfaced claim in the demo carries provenance or is explicitly unsupported.
- D7. H1-H6 are reported whether positive or negative.

STOP CONDITIONS: D1-D7 satisfied, or a blueprint kill criterion fires and the prescribed write-up is delivered.

