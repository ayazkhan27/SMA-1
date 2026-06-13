# Preregistration v2 — LLM-QA "Trustworthy Specialist" phase

Status: **REGISTERED 2026-06-13, before any run below.** Builds on the frozen
`adapter-v1` (ADR-008) and the agentic retrieval results (`docs/STATUS.md`,
5/5 domains). Companion to `configs/preregistration_ontology.md` and
`docs/PAPER_SPINE.md`. This registers the END-TO-END LLM phase: the retrieval
benchmark proved SMA beats RAG/KG on the tail; this phase asks whether an LLM
*agent* equipped with SMA memory becomes a **verifiable specialist** — measured
not by accuracy alone but by capabilities RAG cannot provide.

## 0. Why this phase (and what it is NOT)

The retrieval suite isolated the retriever. This phase puts a real LLM in the
loop and measures the thing that matters in deployment: a wrong-but-confident
answer is catastrophic in medicine/law/security/finance. We test whether the
SMA-grounded agent is **trustworthy** — it cites, abstains when it should, and
flags novelty — where a RAG-grounded agent is confident, opaque, and cannot.
This is NOT a generic accuracy leaderboard chase.

## 1. Claim under test

**C-QA:** Holding the LLM and prompt fixed and swapping only the retrieval memory
(none / dense-RAG / SMA), the **SMA-grounded agent dominates on a trustworthy-QA
composite** — accuracy + citation-faithfulness + abstention-calibration +
novelty-recall — and in particular wins the three capability axes RAG
structurally lacks, while at least matching accuracy.

## 2. Design — memory-swap agent (clean attribution)

- **Fixed:** the LLM (DeepSeek, `DeepSeekOrchestrator.complete`, temperature 0),
  the system+answer prompt, the retrieval top-k, the question set.
- **Swapped (the only variable): the memory** —
  (a) **none** (closed-book LLM), (b) **dense-RAG** (BGE neural dense over the
  same knowledge), (c) **SMA** (the frozen universal adapter over the domain
  golden ontology). Optional (d) hybrid-RAG.
- **Agent turn (one-shot):** receive a case → query the memory for top-k grounded
  candidates → the LLM produces `{answer, citation, abstain}` under the rule
  "answer only if the memory grounds it; cite the supporting evidence; else
  abstain." The SMA condition additionally surfaces an
  `expectation_violation` novelty flag.

## 3. Task & data

Primary task = **ontology-grounded identification/diagnosis QA** on the flagship
**medicine** domain (it leverages the proven HPO/MONDO arm and is the highest-
stakes): given a clinical case (phenotype set in NL), the agent must name the
disease and cite the grounding. Three registered question pools:
1. **Answerable** — cases drawn from indexed diseases (the agent should answer + cite).
2. **Out-of-knowledge** — cases whose true disease is NOT indexed (the agent should ABSTAIN).
3. **Novel** — held-out diseases unseen at index time (the agent should FLAG NOVEL).

External benchmarks used as methodology/comparators (registered; run if feasible
within budget): **ALCE** (citation-quality metric), **MIRAGE** (medical-QA
accuracy reference), and a hallucination/abstention reference
(SimpleQA-style "I don't know" scoring). If a benchmark's corpus is text-only
(not ontology-grounded), it is used to validate the *metric*, not as an SMA arm.

## 4. Metrics (pre-registered) — the trustworthy-QA composite

- **Accuracy** — answer correct on answerable pool.
- **Citation-faithfulness** — does the cited evidence actually entail the answer?
  (ALCE-style support score; for SMA = the structural receipt is checked against
  the gold grounding.)
- **Abstention-calibration** — risk–coverage / selective-prediction AUROC over
  {answerable vs out-of-knowledge}: does the agent answer when it can and abstain
  when it can't?
- **Novelty-recall / F1** — on the novel pool, is the case flagged?
- **Trustworthy-QA composite** — pre-registered weighting (equal weights across
  the four axes, reported alongside each axis; the composite is descriptive, the
  per-axis tests are confirmatory).
- **Stats:** paired bootstrap (10k, seed 12345) on per-question outcomes,
  SMA-agent vs each baseline-agent, per axis; Holm across the axis family.

## 5. Falsifiers

- **Per-axis:** SMA-agent's 95% CI for Δ(SMA − best baseline) on citation-
  faithfulness / abstention-AUROC / novelty-F1 includes 0 after Holm → that axis
  is reported as no-advantage.
- **Headline:** if SMA-agent does not win ≥2 of the three capability axes, the
  "verifiable specialist" claim is narrowed to retrieval-only (the phase-1 result).
- **Accuracy floor:** if SMA-agent accuracy is significantly *below* the RAG-agent,
  report it (the capability gains must not come at an accuracy cost).

## 6. Budget & guards

- One-shot (1 LLM call/question) keeps cost low. Estimate: ~3 pools × ~200
  questions × 3 memory conditions ≈ 1,800 calls (+ a citation-faithfulness judge
  call where automated). DeepSeek context-caching on the shared prompt amortizes.
  Log spend; hard cap; abort-safe. Top up budget before the full run; pilot on a
  small slice first and report projected cost (as in the drift battery).
- Extraction/answer prompt held CONSTANT across all memory conditions.

## 7. Registered caveats

- The agent is one-shot (retrieve→answer); the interactive AgentClinic-style loop
  is a separate, single-domain case study, not part of this confirmatory test.
- Citation-faithfulness for the `none` (closed-book) condition is undefined
  (no retrieval) → that cell is N/A, not 0.
- Cross-domain LLM-QA (cyber/legal/finance) is a follow-up; medicine is the
  registered primary because it is the highest-stakes and best-grounded.

## 8. Addendum — abstention mechanism + pool realization (2026-06-13, PRE full run)

Registered **after the mock + a small real pilot, before the confirmatory run**.
The prereg explicitly permits a pilot to surface implementation decisions (§6);
these are recorded here transparently. None of it peeks at the test split.

**(a) Pool realization — out-of-knowledge ≡ novel on the held-out set.** The §3
pools 2 (out-of-knowledge → should ABSTAIN) and 3 (novel → should FLAG) are
realized on the **same held-out diseases**: an unindexed disease is *both*
out-of-knowledge and novel, and the correct trustworthy behaviour on it is to do
*both* (abstain AND flag). So abstention is scored over {answerable vs held-out}
and novelty over {held-out}, on the same items. This is the faithful reading and
it makes abstention *harder* (every held-out case is a should-abstain case), not
easier.

**(b) Cite-or-abstain signal = the RAW structural grounding score, calibrated.**
The pilot's retrieval-only diagnostic showed the LLM-only abstention and SAGE's
`expectation_violation` flag do **not** separate known from unknown
(answerable-vs-held-out AUROC ≈ 0.48), and the squashed top-hit confidence
saturates (the rank-1 hit is always ≈1.0 by construction). The **raw top
structural match score** *does* separate them (AUROC ≈ 0.84). Therefore the
registered cite-or-abstain signal is the raw grounding score, gated by a
threshold **calibrated on a DISJOINT calibration split** (`calib_answerable` /
`calib_ook`, drawn from spare indexed / held-out diseases, never the test pools)
by maximising **Youden's J** (TPR−FPR) for "answer iff score ≥ t". Calibration is
**retrieval-only (no LLM spend, no test leakage)** and **per-memory** (each
retriever — SMA and dense — fits its own threshold; the fairest comparison).
Below threshold the agent abstains AND flags novel **without an LLM call**. This
replaces nothing in the claim (§1) or falsifiers (§5); it specifies *how* the
abstain/novelty decision is made, identically for every memory.

**(c) Added reported axis — threshold-free grounding-AUROC.** Alongside the
realized operating-point metrics (abstain-recall, false-abstain,
selective-accuracy, risk-coverage AURC) we report the **threshold-free**
grounding-AUROC: the AUROC that the raw score is higher on answerable than on
held-out cases. It is the cleanest "can the memory tell known from unknown"
statistic because it does not depend on where the threshold sits. Novelty is
reported as recall **and** precision/F1 (held-out positive, answerable
false-alarm negative) so "flag everything" cannot win.

**(d) Falsifiers unchanged.** §5 stands: per-axis Δ(SMA − best baseline) 95% CI
(paired bootstrap, seed 12345) must exclude 0 after Holm; SMA must win ≥2 of the
three capability axes; the accuracy floor still guards against a coverage-for-
accuracy trade. Determinism: seeded pools + a deterministic Youden sweep +
`PYTHONHASHSEED=0`.
