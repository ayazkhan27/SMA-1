# SMA-1 Positioning: the production pain we replace

**One sentence.** SMA-1 replaces the vector-search retrieval layer inside
incident-management and RCA-assist tooling with structure-mapped incident
memory: "have we seen this before, what caused it, what fixed it" — with a
checkable provenance record on every suggestion and zero LLM tokens spent on
extraction or indexing.

**The wedge (committed).** Incident memory for SRE/ops and adjacent
pattern-recurrence domains (fraud ops, SOC). NOT general agent memory, NOT
tool-state memory, NOT document QA — those are research extensions (drift
protocol T5), claimed only when their experiments exist.

**Success metrics for this wedge (what we measure, in order):**
1. failure-family-hit@k — does retrieval surface the correct root-cause
   family, not just "an anomaly"
2. fix/mitigation retrieval precision (BugsInPy fix-category match@k; human-
   rated RCA suggestion precision)
3. unsupported-claim rate + evidence-faithfulness of surfaced answers
   (automatable because evidence is structured)
4. wrong-precedent rate at rank 1 (proxy for "wrong fix avoided")
5. abstention quality: does the system say "no precedent" when that is true
6. latency: p95 < 300 ms at 100k cases (gate, not aspiration)
7. cost: $0 extraction tokens; CPU-only indexing (measured per 10k docs)

**Why structure (the differentiated claim).** Within one system, lexical and
embedding retrieval are strong and we report honest parity-or-loss (BGL).
The wedge case is vocabulary shift - new services, renamed components,
cross-system reuse - where surface methods decay and structure transfers
(BGL->Thunderbird: SMA 0.909 vs dense 0.741; held-out confirmation in
progress). Fraud and SOC are adversarial vocabulary-shift settings by nature.

**Enterprise constraints honored by design.** Deterministic extraction
(auditable, reproducible bit-for-bit), provenance on every claim, CPU-only,
on-prem/air-gapped friendly, no per-document LLM cost.

**Current maturity (honest).** Research prototype. Funded-prototype evidence
bar: frozen ontology + held-out multi-seed transfer + RCA/fix retrieval
metrics + drift wrong-action prevention. This document changes only by ADR.
