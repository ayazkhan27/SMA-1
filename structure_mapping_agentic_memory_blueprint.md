# SMA‑1: Structure‑Mapping Agentic Memory
## A complete, plug‑and‑play implementation blueprint for an agentic memory system built on the mathematical core of Gentner's Structure‑Mapping Engine (SME), with MAC/FAC retrieval and SAGE consolidation — no LLM in the extraction path

**Version:** 1.0 (2026‑06‑11) · **Status:** Ready for execution by a human team or a CLI coding agent
**License target:** Apache‑2.0 · **Hardware target:** one 16‑core / 64 GB RAM CPU machine (no GPU required for the system itself; a small GPU or CPU embeddings only for *baselines*)

---

## 0. Honest feasibility verdict (read first)

**Verdict: feasible, with one explicitly scoped carve‑out.**

1. **Feasible with high confidence:** A domain‑agnostic agentic memory whose *retrieval, similarity, inference, and consolidation* are governed entirely by the mathematics of structure mapping (match hypotheses → structural consistency → kernels → merge → structural evaluation → candidate inference), with MAC/FAC two‑stage retrieval for scalability and SAGE‑style analogical generalization for consolidation. All of this machinery is published, has run for decades at Northwestern's Qualitative Reasoning Group, has known complexity bounds (greedy merge is O(n² log n); the exact problem is NP‑hard via Veale & Keane 1997), and the SME v4 reference source (Common Lisp) is publicly downloadable from QRG and usable as a *test oracle*. Nothing in this layer needs an LLM.

2. **Feasible with engineering confidence:** Fully deterministic, LLM‑free structure extraction for **semi‑structured and structured domains** — logs, stack traces, code, configs, JSON/tabular data, form‑like legal/HR documents. Grammar‑based parsers (Drain3 template mining, tree‑sitter ASTs, regex grammars, schema mappers) produce predicate‑calculus cases that are bit‑for‑bit reproducible and auditable. This covers the user's stated 10k‑case log/bug corpus and scales to enterprise volumes.

3. **The honest carve‑out:** Fully general, high‑fidelity extraction of *arbitrary free prose* (e.g., subtle legal testimony) **without any statistical model whatsoever** is not achievable at research‑grade fidelity today; this is the field's open "representation problem," acknowledged by Forbus's group itself. The blueprint therefore defines two encoder tiers: **Tier‑0 (strict)** — pure grammar/rule/lookup extraction, zero statistical components, used for the MVP and all headline claims; **Tier‑1 (standard)** — rule‑based open‑IE over a *non‑generative, deterministic‑at‑inference* dependency parse (Stanza/spaCy) plus a Penn‑Discourse‑Treebank explicit‑connective lexicon for higher‑order relations (CAUSE, CONDITION, CONTRAST). Tier‑1 contains a statistical parser (a classifier, not a generative LM); it is clearly flagged, optional, and never used for the paper's primary claims. This is stated as a limitation, not hidden. Everything else in the user's mandate — no LLM/VLM anywhere except orchestration, non‑ad‑hoc weights, scalability, domain‑agnosticism, testability against RAG/KG — is met in full.

4. **The scientific hunch itself (structure beats semantics on unseen concepts and drift) is a hypothesis, not a guarantee.** It has supporting evidence (LLMs collapse on counterfactual/far analogies where surface cues vanish — Lewis & Mitchell 2024; GPT‑4 scores *below random* on far analogies in ARN; structure mapping by construction ignores surface vocabulary), and this blueprint converts the hunch into six pre‑registered, falsifiable hypotheses (§8) with kill criteria (§12). If H1 fails after the diagnostics in §12, the correct output is a negative‑result paper — also scientifically valuable.

---

## 1. Research grounding (what was verified online, 2026‑06‑11)

Every load‑bearing design decision below cites one of these. Full bibliography in §13.

| # | Finding | Source | Used for |
|---|---|---|---|
| R1 | SME algorithm: match hypotheses, structural consistency (one‑to‑one, parallel connectivity), gmaps, structural evaluation, candidate inferences; "most steps polynomial, typically O(N²)" | Falkenhainer, Forbus & Gentner 1986 (AAAI), 1989 (*Artificial Intelligence* 41:1–63) | §2 matcher math |
| R2 | Systematicity principle: prefer interconnected relational systems governed by higher‑order relations; relations over attributes | Gentner 1983, *Cognitive Science* 7:155–170 | §2 scoring |
| R3 | Modern SME: greedy merge constructs best interpretations in **O(n² log n)**; incremental mapping; SME now uses overall similarity (matches at every level, structure still wins via systematicity) | Forbus, Ferguson, Lovett & Gentner 2017, *Cognitive Science* 41(5):1152–1201 | §2.5 merge, §2.8 complexity |
| R4 | Exact structure mapping is **NP‑hard** | Veale & Keane 1997 (cited in R3 and in Crouse et al.) | §2.8; justifies anytime exact/greedy design |
| R5 | MAC/FAC: stage 1 = **content vectors** (functor occurrence counts), whose **dot product estimates the structural match score**; stage 2 = SME on the shortlist | Forbus, Gentner & Law 1995, *Cognitive Science* 19:141–205 | §2.7 retrieval; we strengthen "estimate" to a provable upper bound |
| R6 | SAGE: generalization pools of generalizations + outliers; assimilate when SME score (normalized) exceeds an **assimilation threshold**; facts carry **probabilities = support frequencies**; low‑probability facts wear away; incremental (no retraining); "k‑means with outliers but the number of clusters is discovered" | McLure, Friedman & Forbus 2015; Kuehne et al. 2000 (SEQL); Halstead & Forbus 2005; AILEEN papers (arXiv 2006.01962, 2210.11731) | §2.9 consolidation |
| R7 | SME v4 reference implementation (Common Lisp) + algorithm/complexity documentation is downloadable from Northwestern QRG | qrg.northwestern.edu/software/sme4 | §7 P2 oracle tests |
| R8 | 2025 retrospective by Gentner & Forbus confirms SME remains the canonical SMT implementation | Gentner & Forbus 2025, *Current Directions in Psychological Science* | motivation |
| R9 | LLM analogical reasoning is **brittle off‑distribution**: on counterfactual variants of analogy problems, human performance stays high while GPT models decline sharply (contested by Webb et al. 2024 reply for letter‑string tasks with code execution — we report the debate honestly) | Lewis & Mitchell 2024 (arXiv 2402.08955); Webb et al. 2024 (arXiv 2404.13070) | motivation; H1 design |
| R10 | ARN (TACL 2024): narrative analogy benchmark built on the surface/system‑mapping distinction; LLMs handle near analogies but degrade ~35 points on **far analogies**; GPT‑4 zero‑shot **below random** on far analogies | Sourati, Ilievski, Sommerauer & Jiang 2024 | T4 benchmark |
| R11 | Agent failure modes over long horizons: context pollution, error **compounding through autoregressive feedback** (outputs become inputs), state drift; calls for metrics tracking divergence of internal state from ground truth | arXiv 2601.04170 "Agent Drift"; arXiv 2602.16666 "Towards a Science of AI Agent Reliability"; context‑rot literature | T5 drift protocol; motivation |
| R12 | HippoRAG (NeurIPS'24): KG + Personalized PageRank memory; HippoRAG 2 (ICML'25): the strongest published KG‑memory baseline; both use **LLM OpenIE for extraction** (cost/fragility we avoid by design) | Gutiérrez et al. 2024, 2025; github.com/OSU-NLP-Group/HippoRAG | B4/B5 baselines |
| R13 | LogHub: HDFS v1 (11.17M messages; 575,061 block sessions; 16,838 labeled anomalous), BGL (4.75M messages, per‑message labels, 348,460 anomalous), Thunderbird, OpenStack; Loghub‑2k provides **oracle templates** for parser validation | He et al., ISSRE 2023; github.com/logpai/loghub | T2 datasets; P3 exit gates |
| R14 | Minimal‑complexity (Kolmogorov/MDL) analogy solving: humans prefer the lowest‑complexity solution; computable MDL surrogates work in practice | Murena, Al‑Ghossein, Dessalles & Cornuéjols, IJCAI 2020; Murena et al. 2017 | §2.6 parameter‑free scorer |
| R15 | Neural Analogical Matching (AMN): a neural approximation of SMT exists (AAAI 2021) — noted and **excluded** from the core (it is a learned model; we keep the matcher symbolic), but reused as a related‑work anchor | Crouse, Nakos, Abdelaziz & Forbus 2021 | related work |

---

## 2. Formal core: definitions, theorems, scoring, and complexity

This section is the contract. The matcher implemented in §7‑P2 must satisfy it exactly; the property‑based tests in §7 encode it.

### 2.1 Representation: typed predicate cases as labeled DAGs

A **case** is the unit of memory (one incident, one bug, one testimony, one episode of agent experience).

- **Vocabulary.** Σ = E ⊎ F ⊎ A ⊎ R: entities/constants, functions, attributes (arity‑1 predicates), relations (arity ≥ 2). A designated subset HO ⊆ R of *higher‑order* relations (e.g. `cause`, `implies`, `enables`, `before`, `during`, `prevents`, `partOfPlan`) takes *statements* as arguments. Every functor has a fixed **signature**: arity, argument types, commutativity flag (for `and`, `sameAs` — handled by sorted canonical argument order at encode time, per Ferguson 2003's commutative‑expression problem).
- **Case.** C = a finite set of ground statements; each statement is an expression tree over Σ. The union of statement trees forms a rooted **DAG** D_C = (V_C, A_C): vertices are expression occurrences and entities; arcs go from each expression to its arguments, labeled by argument position i. Sub‑expressions are shared (hash‑consed), so identical sub‑statements are a single vertex — this is what makes "interconnected systems" a graph‑theoretic object.
- **Order.** order(v) = 0 for entities/constants; order(expr) = 1 + max over arguments. Attributes are order 1; first‑order relations order 1; HO relations ≥ 2. **Depth** d(C) = max order; in practice d ≤ 6.
- **Canonical serialization.** S‑expressions with interned 32‑bit symbol ids, e.g. `(cause (timeout svc-A db-1) (retryStorm svc-A))`. Case id = BLAKE3 hash of canonical form (content‑addressed; dedup for free; audit trail for free).

*Why this is domain‑agnostic:* the IR contains **no domain commitments** — only typed terms and statements. Domains enter solely through (a) which adapter produced the statements and (b) the canonicalization lattice (§4.3). This is exactly the SME/Cyc‑style discipline used across physics, sketches, moral dilemmas, stories, and strategy games in the QRG literature (R3, R6), which is the strongest available evidence that the IR itself generalizes.

### 2.2 Match hypotheses and structural consistency (the SMT constraints, stated precisely)

Given base B and target T:

- **MH seeding (tiered identicality).** H₀ = { (b,t) ∈ expr(B)×expr(T) : κ(functor(b)) = κ(functor(t)) ∧ arity(b)=arity(t) }, where κ is the canonicalization map (§4.3). **Minimal ascension:** if no identical match exists for a statement, allow (b,t) when the functors share a least common subsumer within ≤ δ steps in the predicate lattice L, at multiplicative score penalty ρ^dist (ρ, δ fitted in §8.6, defaults δ=2). Entity and function MHs are *induced*, never seeded: (arg_i(b), arg_i(t)) is added for every seeded/induced expression MH — this enforces **parallel connectivity** downward by construction.
- **Structural consistency.** A set M ⊆ H of MHs is *structurally consistent* iff
  (i) **one‑to‑one:** the induced correspondence on V_B×V_T is a partial **injective function** (no base item maps to two targets; no two base items map to one target);
  (ii) **support:** for every expression MH (b,t) ∈ M and every position i, (arg_i(b), arg_i(t)) ∈ M.
- **Kernels.** A *root* MH is one not required as an argument descendant of any other MH. kernel(h) = the downward support closure of root h. A kernel that internally violates (i) is discarded (its root MH is structurally impossible). Kernels are the atoms of interpretation — exactly SME's design (R1, R3).
- **Gmaps (global mappings).** Maximal unions of kernels that are structurally consistent.

**Lemma 1 (pairwise sufficiency — the exactness license for the merge step).**
*A union of internally consistent kernels is structurally consistent iff every pair of kernels in the union is mutually consistent.*
**Proof.** Condition (ii) is closed under union because each kernel already contains its own support closure. Condition (i) — being a partial injective function — fails only via a witness: either some b with (b,t),(b,t′)∈M, t≠t′ (functionality), or some t with (b,t),(b′,t)∈M, b≠b′ (injectivity). Each witness involves exactly two MHs, each contained in some kernel; the violation is therefore visible in the union of (at most) those two kernels. Conversely pairwise consistency rules out every two‑MH witness. ∎
**Consequence.** Gmap construction is exactly **Maximum‑Weight Independent Set (MWIS)** on the *kernel conflict graph* G_K (vertices = kernels weighted by their SES contribution; edges = mutually inconsistent kernel pairs, detectable by hash‑join on entity bindings in O(total binding size)). This licenses the solver design of §2.5 and explains *why* SME's greedy merge is an approximation to a well‑posed combinatorial optimum rather than an ad‑hoc procedure.

### 2.3 Structural evaluation score (SES) by trickle‑down — with the systematicity bias as a theorem‑shaped property

Compute over the mapped justification DAG, roots first:

```
s(h) = σ₀ · asc(h) + γ · Σ_{p ∈ parents_M(h)} s(p)        (trickle‑down)
SES(M) = Σ_{h ∈ M} s(h)
```

where σ₀ = 1 (a unit; only ratios matter), asc(h) = ρ^dist(h) ∈ (0,1] is the minimal‑ascension penalty (1 for identical functors), and γ > 0 is the trickle‑down factor. Parents pass evidence to the arguments they justify; therefore an MH embedded under k nested higher‑order matched relations receives score Θ(Σ_{i≤k} γ^i · #ancestor‑chains): **deep, interconnected matched systems strictly dominate equal‑sized flat collections of isolated matches.** That is Gentner's systematicity (R2) realized as arithmetic, identical in shape to SME's published trickle‑down (R1, R3).

**Boundedness/stability condition (replaces a magic constant with a constraint).** If Δ = max parent fan‑in in any justification DAG and d = max depth, then s(h) ≤ σ₀·Σ_{i=0}^{d}(γΔ)^i. Choose γ < 1/Δ for library‑scale stability **or** use per‑comparison normalization SESₙ(M) = SES(M)/max(SES_self(B), SES_self(T)) (self‑match score), which is what SAGE uses for thresholding (R6) and is scale‑free across domains. We use SESₙ everywhere a threshold is compared across cases — this is one of the two pillars of the "non‑heuristic weights" requirement (§2.6 is the other).

### 2.4 Candidate inference (the memory's *output*, with provenance)

For a selected gmap M with substitution σ_M (base→target over matched items):

```
CI(M) = { f ∈ statements(B) \ dom(M) :  f is arc‑connected in D_B to dom(M) }
project(f) = σ_M(f) with every unmapped entity replaced by a fresh skolem  (skolem fn = AnalogySkolemFn, flagged :hypothetical)
```

Every projected inference is emitted as a record `{inference, base_case_id, gmap_id, SESₙ, support := mapped justifications above f, skolems, ascensions_used}`. **This is the anti‑hallucination mechanism:** unlike generative recall, an SMA inference is a checkable object — the verifier (§3.6) can type‑check it, the orchestrating LLM can only *verbalize or act on* it, and a human can audit exactly which past case and which correspondences produced it. (This is standard SME candidate‑inference semantics, R1/R3, repurposed as a provenance discipline.)

### 2.5 Merge solver: exact‑anytime (CP‑SAT) with the published greedy as guaranteed fallback

- **Exact (default for ≤ ~400 kernels):** MWIS via Google OR‑Tools **CP‑SAT**: maximize Σ_k w_k x_k, x∈{0,1}, with x_{k₁}+x_{k₂} ≤ 1 for each conflict edge; w_k = SES contribution of kernel k (computable per‑kernel because trickle‑down is confined to a kernel's own justification DAG; cross‑kernel parents do not exist by the root definition). Time budget 250 ms; CP‑SAT is anytime and reports the optimality gap — log it.
- **Fallback (always available, scales unconditionally):** SME's greedy merge — sort kernels by w_k descending, add if pairwise‑consistent with the current set, optionally with b restarts seeded by the top‑b kernels. Complexity **O(n² log n)** as published (R3). Lemma 1 guarantees greedy never emits an inconsistent gmap.
- **Justification:** the exact problem is NP‑hard (R4); an *anytime exact solver with a certified gap, falling back to the literature's own polynomial approximation*, is the most principled stance possible — no silent heuristic, every approximation is measured and reported.

### 2.6 Eliminating ad‑hoc weights: two complementary, defensible regimes

The user's hardest requirement. Two regimes, both implemented; the eval (§8.7) compares them.

**Regime A — identified statistical parameters (γ, ρ, δ, θ_a, θ_p), never hand‑set:**
- The score model has exactly two continuous shape parameters (γ, ρ) and three structural ones (δ, assimilation θ_a, probability cutoff θ_p). None is chosen by feel. Estimation:
  1. **Human‑fit MLE:** Bradley–Terry on pairwise soundness judgments — P(M₁ ≻ M₂) = σ(β·(SESₙ(M₁) − SESₙ(M₂))) — fitted on (a) the Gentner, Rattermann & Forbus (1993) "Karla the Hawk"‑style story sets (literal‑similarity vs analogy vs mere‑appearance vs first‑order‑only variants; the materials are published in the paper's appendix and transcribable), and (b) our synthetic SSB gold (§8.3) where ground‑truth soundness order is known by construction. Report point estimates, profile‑likelihood CIs, and a γ‑sensitivity sweep (γ ∈ {0, .125, .25, .5, 1/Δ}).
  2. **Task calibration:** θ_a, θ_p selected by maximizing held‑out macro‑F1 on T2 triage via grid search on the *validation* split only, reported with the grid. (This is how SAGE's modeler‑set thresholds become identified quantities; published SAGE configs such as AILEEN's {assimilation 0.01... 0.2, probability 0.6} are recorded as priors/reference points, R6.)
- **Domain‑agnosticity argument:** SESₙ is a function of *structure only* (canonical‑functor identity, arity, connectivity) — it contains no lexical, frequency, or domain features; γ and ρ modulate the universal geometry of justification DAGs, not content. Fitting them on stories + synthetic structures and *freezing* them before touching logs/code (pre‑registered, §8.6) is the test of domain transfer, not an assumption of it.

**Regime B — parameter‑free MDL scorer (the mathematically purest answer):**
Define a prefix code over canonical statements: cost ℓ(f) = Σ_{symbols σ in f} −log₂ p̂(σ) + ℓ_arity, with p̂ the corpus unigram distribution over canonical functors (universal/Krichevsky–Trofimov smoothing) and Rissanen universal integer codes for indices. Then for mapping M:

```
G(M) = L(T) − L(T | B, M)
L(T | B, M) = Σ_{matched systems S} [ log₂|roots(B)| ]                (one pointer per connected matched system)
            + |τ(M)| · [ log₂|E_B| + log₂|E_T| ]                       (the entity substitution table)
            + Σ_{unmatched f ∈ T} ℓ(f)                                  (fresh encoding of the residual)
```

- **Proposition (systematicity = compression).** For a fixed set of matched statements, G(M) increases as the number of connected matched systems k decreases (each system costs one pointer; bindings are shared across a system). Hence the maximal‑gain mapping prefers few, large, deeply interconnected relational systems — systematicity *derived* rather than weighted. Moreover **one‑to‑one is necessary for decodability**: τ must be a function for the decoder to invert the reference encoding — i.e., SMT's hardest constraint falls out of the requirement that an analogy be a lossless code. This grounds the constraints in Rissanen's MDL and matches the empirical finding that humans select minimal‑complexity analogies (R14).
- Regime B has **zero tunable weights**; its cost is that p̂ injects mild frequency sensitivity (rare shared structure compresses more — arguably a feature: surprisal‑weighted systematicity). The ablation §8.7 reports A vs B head‑to‑head; the paper leads with whichever wins on SSB human‑order recovery, reporting both.

### 2.7 MAC stage: content vectors upgraded from "estimate" to **admissible bound** → certified top‑k retrieval

- **Content vector.** c(C) ∈ ℕ^{|Σ̂|}: counts of canonical functors (this is exactly MAC's vector, R5) **plus** Weisfeiler–Leman‑1 refinement features `(functor(parent), i, functor(child_i))` — strictly more selective, still a bag, still inner‑product‑able; WL features are the standard isomorphism‑aware bag refinement (Shervashidze et al. 2011) and cost nothing extra at encode time.
- **Lemma 2 (screening bound).** Every MH consumes one occurrence of a shared canonical functor in each case (injectivity, §2.2‑i), and each MH's trickle‑down score is ≤ s̄ = σ₀·Σ_{i≤d}(γΔ)^i. Therefore
  `SES(B,T) ≤ s̄ · Σ_σ min(c_B(σ), c_T(σ)) =: U(B,T)` — the **histogram‑intersection kernel times a constant** is an admissible upper bound on SES. (With ascension enabled, replace c by counts over the ≤δ ancestor closure in L — still a lookup.) An analogous bound holds for Regime B: G(M) ≤ Σ_σ ℓ(σ)·min(c_B(σ),c_T(σ)).
- **Retrieval algorithm (certified MAC/FAC):**
  1. ANN pre‑screen: cosine over feature‑hashed (2¹³‑dim) L2‑normalized vectors in HNSW → top‑K₀ (K₀=200) in O(log N) amortized;
  2. exact bound U on the K₀ shortlist via the functor inverted index (sparse min‑intersection);
  3. **best‑first FAC:** run the full matcher on candidates in descending U, maintaining the k‑th best SESₙ so far; stop when U(next) < kth‑best. ⇒ the returned top‑k is *provably exact w.r.t. SES over the shortlist*; shortlist recall (the only uncertified step) is itself measured against brute force on a 1k sample (§7‑P4 exit gate).
- **Why this honors MAC/FAC and improves it:** Forbus/Gentner/Law's dot product "estimates how well the structural representations will match" (R5); Lemma 2 turns the estimate into a bound, which buys early stopping with no ranking regret — the property that makes 10⁴→10⁷ scaling principled (shard the ANN; FAC is embarrassingly parallel per candidate).

### 2.8 Complexity budget (per operation, n = statements per case, N = library size, m = |MHs|, K = shortlist)

| Operation | Bound | Notes |
|---|---|---|
| Encode (Tier‑0) | O(doc length) | parsers are linear/near‑linear |
| MH seeding | O(Σ_σ c_B(σ)·c_T(σ)) ≤ O(n_B·n_T) | sparse in practice; grouped by functor |
| Support closure + kernels | O(m·ā) | ā = mean arity |
| Conflict graph | O(k² ) pair checks via binding hash‑join | k = #kernels, typically ≪ m |
| Merge: greedy | **O(m² log m)** (published, R3) | unconditional |
| Merge: CP‑SAT | NP‑hard worst case; 250 ms budget, anytime, gap logged | exact on typical instances |
| SES trickle‑down | O(m) topological pass | DAG |
| MAC query | O(log N) ANN + O(K·n̄) bounds + best‑first FAC | FAC parallel across candidates |
| Memory | O(n) per case (~2–8 KB zstd) | 10⁷ cases ≈ tens of GB |

Targets (§7‑P4 gate): median pair‑match ≤ 50 ms at n=300 (greedy), library query p95 ≤ 1.5 s at N=10⁴ on the reference box.

### 2.9 Consolidation: SAGE generalization with frequency‑probabilities (no learned weights)

Per *generalization context* (one per task family, created by the orchestrator):
1. New case → MAC/FAC against the context's generalizations + outliers (R6).
2. If best SESₙ ≥ θ_a: **assimilate** — align via the gmap; replace corresponded entities by role‑preserving skolems `(GenEntFn i ctx)`; every fact's probability is the **support frequency** n_f/N_g over constituents (pure counting with Laplace +1 smoothing — there is nothing to tune beyond θ_p, which is calibrated in §2.6‑A); facts with p < θ_p after N_g ≥ N_min constituents are pruned ("wear away", R6).
3. Else store as outlier; two mutually‑assimilable outliers seed a new generalization. Disjunctive concepts emerge as multiple generalizations per pool; the number of clusters is discovered, not set (R6).
**Why it matters for the user's goals:** generalizations are *schemas* — when a genuinely novel incident arrives, it can match the schema even when it matches no single exemplar, which is the concrete mechanism behind "robust to unseen concepts." Probabilities are frequencies, satisfying the no‑heuristic‑weights mandate.

### 2.10 The drift argument, stated precisely (and how we test it rather than assert it)

Mechanism claim: in a context‑accumulating agent, per‑step generation error ε feeds back autoregressively (outputs become inputs), giving state‑fidelity decay on the order of (1−ε)^t plus context‑pollution effects; this failure mode is documented in the agent‑drift literature (R11). SMA changes the dependency structure: at every step the working state is **re‑derived by the deterministic encoder from the environment** (not from prior generations), beliefs live in the case store (not the context window), and the only things the LLM injects into memory are raw artifacts routed through the encoder. Therefore per‑step error is bounded by encoder error and does **not** compound through generation. This is an architectural argument, not a proof about task success — so it is tested as **H4** with the T5 protocol (§8.4): measure state‑fidelity F1 vs ground truth as a function of horizon t for SMA‑memory vs RAG‑notes memory vs context‑only, and compare decay slopes.

---

## 3. System architecture

```
                         ┌─────────────────────────────────────────────┐
 raw artifacts ───────▶  │ ENCODER LAYER (deterministic, versioned)    │
 (logs, code, traces,    │  adapters: logs/ code/ traces/ json/ prose* │──┐
  docs, agent obs.)      └─────────────────────────────────────────────┘  │ cases (s‑expr, content‑addressed)
                                                                          ▼
   ┌───────────────┐   write   ┌───────────────────┐   index   ┌────────────────────┐
   │ ORCHESTRATOR  │──────────▶│  CASE STORE        │──────────▶│  INDEX SERVICE      │
   │ (the only LLM)│  encode() │  LMDB + zstd, WAL, │           │  functor inverted   │
   │ tool‑calls    │           │  append‑only audit │           │  index + HNSW (WL‑1 │
   │ only — never  │◀──────────│  schema registry   │◀──────────│  hashed vectors)    │
   │ writes facts  │  records  └───────────────────┘  vectors   └────────────────────┘
   └──────┬────────┘                                                   │ shortlist + U bounds
          │ retrieve()/map()/project()/verify()/generalize()           ▼
          │                ┌──────────────────────────────────────────────────────┐
          └───────────────▶│ MATCHER (sme‑core): MH seeding → kernels → conflict   │
                           │ graph → CP‑SAT/greedy merge → SESₙ or MDL‑gain →      │
                           │ candidate inferences (+provenance)                    │
                           └──────────────┬───────────────────────────────────────┘
                                          ▼
                           ┌──────────────────────────┐    ┌──────────────────────┐
                           │ VERIFIER: signatures,     │    │ CONSOLIDATOR (SAGE): │
                           │ skolem policy, domain     │    │ gen‑pools, frequency │
                           │ validators (pluggable)    │    │ probabilities, decay │
                           └──────────────────────────┘    └──────────────────────┘
```

### 3.1 Component contracts (the orchestrating LLM sees only these tools)

```json
encode(artifact, adapter_id) -> {case_id, n_statements, warnings[]}        // ONLY path by which content enters memory
retrieve(case_id | inline_case, k, pool) -> [{case_id, SES_n, U_bound, certified: bool}]
map(base_id, target_id, scorer: "ses"|"mdl") -> {gmap_id, correspondences[], SES_n, gap, kernels_used}
project(gmap_id) -> [{inference_sexpr, provenance{base_id,gmap_id,SES_n,support[],skolems[],ascensions[]}}]
verify(inference) -> {status: pass|type_fail|domain_fail|hypothetical, reasons[]}
store(case_id, outcome_annotations) -> ok                                   // annotations are themselves statements via encode()
generalize(pool_id) | pool_stats(pool_id) -> schema summaries with fact probabilities
explain(gmap_id) -> human‑readable correspondence table                     // what the LLM verbalizes
```

Hard rules enforced in the API layer (not by prompt): (1) no tool accepts free‑text "facts" — `store` annotations are routed through `encode`; (2) every answer surfaced to a user must reference ≥1 provenance record or be labeled `unsupported‑by‑memory`; (3) the LLM may *choose* adapters, *sequence* tools, *verbalize* results, and *act* — nothing else. This is the operational meaning of "LLM only for orchestration."

### 3.2 Storage
- **Case store:** LMDB (memory‑mapped, ACID, zero‑server) with zstd‑compressed canonical s‑expressions; keys = BLAKE3 content ids; append‑only WAL table for audit/replay. Postgres optional at enterprise scale (swap behind the same repository interface).
- **Indexes:** (a) functor → posting list (for exact min‑intersection bounds, Lemma 2); (b) hnswlib index over 8192‑dim feature‑hashed WL‑1 vectors; (c) metadata SQLite (timestamps, adapter version, labels, pool membership). All rebuildable from the WAL — the store is the single source of truth.

---

## 4. Extraction without LLMs (the make‑or‑break layer)

**Stance (restated as policy):** the encode path contains **no generative model under any circumstances**. Tier‑0 contains no statistical model at all. Tier‑1 may contain a *non‑generative, deterministic‑at‑inference* dependency parser, clearly flagged in every case's metadata; Tier‑1 is excluded from headline claims. Determinism requirement: identical input + adapter version ⇒ identical case bytes (CI enforces with golden files).

### 4.1 Tier‑0 adapters (MVP scope; pure rules/grammars/lookups)

**A. logs adapter** (the user's primary corpus):
1. **Templates as event predicates:** Drain3 (fixed config, masking rules versioned) mines templates online; each template id becomes a functor `evt_<hash>` with a curated alias (`blockReceiveError`, …) added lazily; template parameters become typed entities (ip, blockId, path, duration — typed by masking rules).
2. **Case assembly:** one case per session (HDFS: block id; BGL/Thunderbird: 60 s window — the standard groupings, R13). Statements: event instances; `before(e_i, e_j)` from timestamps; `count(evt, n)` and `burst(evt, window)` from deterministic aggregation; component/host attributes from structured fields.
3. **Higher‑order relations from a rule lexicon (the systematicity carrier):** per‑adapter declarative rules over event sequences, e.g. `timeout(X,Y) ∧ retry(X,Y) within w ⇒ cause(timeout(X,Y), retryStorm(X))`, `Caused by:`‑chains in exceptions ⇒ `cause`, supervisor‑restart patterns ⇒ `enables/prevents`. Rules are data (YAML), versioned, domain‑extensible — this is where an SRE's knowledge lives, *at the boundary*, leaving the core domain‑free.
**Rationale:** log text is machine‑generated from format strings; template mining + rules recovers structure essentially losslessly — the strongest possible footing for the deterministic‑extraction requirement, validated against Loghub‑2k oracle templates (R13) in gate G3.

**B. code/bugs adapter:** tree‑sitter parses source to ASTs (grammars are deterministic, 40+ languages); extract `defines/calls/imports/throws/catches/assigns/reads` relations; stack traces via a regex grammar → `frame/at/calledFrom` chains; exception cause‑chains → `cause`; diffs (for fixes) → `adds/removes/modifies(node)` statements. Test failures (pytest/JUnit XML) → assertion statements.

**C. structured‑data adapter:** JSON/CSV/XML with a declared schema map → relations directly (column = attribute/relation; foreign keys = relations). Covers tickets, CRM rows, form‑like legal/HR records — the deterministic slice of "legal testimony, UX, marketing" data.

**D. agent‑observation adapter:** the agent's own tool outputs (exit codes, file diffs, HTTP statuses) → statements; this is how SMA becomes *agentic* memory rather than a document index.

### 4.2 Tier‑1 prose adapter (flagged, optional, not in headline claims)
UD parse (Stanza 1.8, fixed seeds/greedy decoding) → ClausIE/MinIE‑style **rule** extraction of (arg, pred, arg) clauses with negation/modality flags → neo‑Davidsonian reification (`event e; agent(e,X); theme(e,Y)`); **higher‑order relations only from the PDTB explicit‑connective lexicon** ("because/therefore/although/if" → `cause/implies/contrast/condition` over clause statements) — connective lookup is deterministic. Fidelity ceiling stated openly: implicit discourse relations are *not* extracted (that's where statistical/generative parsing would creep in). This tier exists to run T4 (ARN, Karla) and to demonstrate the domain‑agnostic IR; its cases carry `tier:1` metadata forever.

### 4.3 Canonicalization κ and the predicate lattice L (symbolic only)
κ = rule lemmatizer → adapter alias table → lattice node. L = union of (a) WordNet hypernym closure + VerbNet class memberships + FrameNet frame membership (all NLTK corpus readers — static lookup tables, no models), (b) per‑adapter mini‑ontologies (e.g. `connTimeout ⊑ timeout ⊑ failureEvent`). Tiered identicality (§2.2) consults L only when exact matching fails, within δ, at penalty ρ^dist — mirroring SME's minimal ascension and MAC/FAC's identicality‑constrained semantics, which outperformed similarity‑table models (Law, Forbus & Gentner 1994). Coverage gaps in technical vocabularies are expected; the mitigation is adapter ontologies + a `lattice‑miss` counter surfaced in eval (§12‑R3).

---

## 5. Exact software stack (state‑of‑the‑repo, pin minor versions at P0 and record in ADR‑001)

| Package (PyPI) | Version floor | Role | Why this one |
|---|---|---|---|
| python | 3.11 | runtime | perf + typing |
| pydantic | 2.x | IR schemas, tool contracts | validation at every boundary |
| rustworkx | ≥0.15 | DAGs, toposort, components | Rust speed, NetworkX‑like API |
| networkx | 3.x | prototyping, PPR for KG baseline | ubiquity |
| ortools | ≥9.10 | CP‑SAT merge (§2.5) | best free anytime exact solver, gap reporting |
| hnswlib | ≥0.8 | ANN over hashed WL‑1 vectors | tiny, fast, CPU; (faiss‑cpu 1.8 acceptable swap) |
| scikit‑learn | 1.5 | HashingVectorizer (feature hashing), metrics | standard |
| numpy / scipy | 2.x / 1.13 | math, sparse min‑intersection | standard |
| drain3 | ≥0.9.11 | log template mining | de‑facto standard online Drain |
| tree‑sitter + tree‑sitter‑language‑pack | ≥0.23 | code ASTs | deterministic grammars, many languages |
| regex | latest | trace/connective grammars | possessive quantifiers, stability |
| nltk | 3.9 | WordNet/VerbNet/FrameNet readers | symbolic lattices, no models |
| stanza | 1.8 | Tier‑1 UD parse ONLY | non‑generative; flagged tier |
| lmdb / zstandard | 1.4 / 0.22 | case store | mmap ACID + compression |
| blake3 | latest | content addressing | fast, collision‑resistant |
| fastapi + uvicorn | latest | tool API for the orchestrator | typed OpenAPI tools |
| typer | latest | CLI (`sma encode/retrieve/map/eval …`) | agent‑drivable |
| pytest + hypothesis | latest | unit + property tests (invariants §2.2) | property tests are the spec |
| ranx | latest | R@k/MRR/nDCG | standard IR eval |
| statsmodels | latest | Bradley–Terry fit, bootstrap, Holm | §2.6/§8.8 |
| rank‑bm25 | latest | baseline B1 | simple, faithful BM25 |
| sentence‑transformers | latest | baseline B2/B3 ONLY (bge‑base‑en‑v1.5) | embeddings never touch SMA core |
| kuzu (or networkx PPR) | ≥0.6 | baseline B4 KG store + Personalized PageRank | embedded graph DB, no server |
| hipporag | latest | baseline B5 (as‑published, LLM‑IE cost comparator) | R12 |
| hydra‑core or tomli | latest | config; every run reproducible from one file | repro |
| docker | — | release image, CPU‑only | repro |

Reference oracle (not a dependency): **SME v4 Common Lisp source from QRG** (R7) run via SBCL in a side container to cross‑check mappings on a 25‑pair regression battery (G2). Check QRG's license terms before redistribution; use as test fixture generator only.

---

## 6. Repository layout

```
sma/
├── sma/ir/            # schema.py (pydantic), sexpr.py, canon.py (κ, lattice), signatures.py
├── sma/store/         # lmdb_store.py, wal.py, registry.py
├── sma/encoders/      # base.py, logs_drain.py, code_treesitter.py, traces.py, structured.py,
│                      # agentobs.py, prose_tier1.py, rules/ (*.yaml HO‑relation lexicons)
├── sma/match/         # mh.py, kernels.py, conflicts.py, merge_greedy.py, merge_cpsat.py,
│                      # ses.py (trickle‑down), mdl.py (Regime B), infer.py (CWSG), explain.py
├── sma/index/         # content_vectors.py (WL‑1 + hashing), inverted.py (U bound), ann.py, macfac.py
├── sma/sage/          # pools.py, assimilate.py, probabilities.py
├── sma/agent/         # api.py (FastAPI tools), policies.py (hard rules §3.1), loop_demo.py
├── sma/eval/          # ssb_generator.py, loghub.py, bugsinpy.py, arn.py, karla/, drift_env.py,
│                      # baselines/{bm25,dense,hybrid,kg_ppr,hipporag}.py, metrics.py, stats.py, report.py
├── tests/             # unit/, property/ (hypothesis invariants), oracle/ (SME v4 battery), golden/
├── configs/           # default.toml per phase; calibration grids; preregistration.md
├── docs/              # this blueprint, ADR/ (decision records), STATUS.md (ledger §10.2)
└── docker/, .github/workflows/ci.yml, GOALS.md (/goal of §10)
```

---

## 7. Build plan — phases with who / when / where / how / why / exit gates

Team: **E1** KR/algorithms engineer (matcher, math), **E2** systems engineer (encoders, store, index, API), **E3** eval scientist (benchmarks, calibration, stats, paper). A single capable CLI agent serializes E1→E2→E3 (multiply durations ×2.5). Where: the reference CPU box + GitHub CI. All gates are pytest markers (`-m gate_G2` etc.) so a CLI agent can self‑verify.

| Phase | When | Who | What & how | Exit gate (machine‑checkable) |
|---|---|---|---|---|
| **P0 bootstrap** | wk 0 | all | repo, CI, ADR‑001 (pinned versions), fetch LogHub/BugsInPy/ARN, transcribe Karla materials, adopt GOALS.md | **G0:** CI green on skeleton; datasets checksum‑verified; /goal accepted in ADR‑002 |
| **P1 IR + store** | wk 1–2 | E2 | schemas, s‑expr codec, hash‑consing, LMDB store, WAL, signatures | **G1:** round‑trip property tests (parse∘print=id); store survives 10⁵ writes + replay |
| **P2 matcher** | wk 2–4 | E1 | §2.2–§2.6 exactly; greedy + CP‑SAT; SES + MDL scorers; CWSG inference | **G2:** (a) hypothesis‑tests of invariants (injectivity, support closure, Lemma‑1 randomized check) pass; (b) canonical battery: water‑flow/heat‑flow yields pressure↔temperature with flow inference, solar‑system/atom yields sun↔nucleus, matching published SME output (R1) and the SME‑v4 oracle on ≥25 pairs; (c) perf: ≤50 ms median @ n=300 greedy |
| **P3 Tier‑0 encoders** | wk 3–5 | E2 | logs (Drain3+rules), code (tree‑sitter), traces, structured, agent‑obs; golden files | **G3:** byte‑determinism on goldens; template grouping vs Loghub‑2k oracle templates ≥ Drain's published accuracy on HDFS/BGL; 50‑case human audit ≥90% statement precision |
| **P4 MAC/FAC index** | wk 5–6 | E2+E1 | WL‑1 vectors, hashing, HNSW, inverted U‑bounds, best‑first FAC | **G4:** certified‑top‑k equals brute force on 1k‑case sample (exactness check of Lemma 2 path); query p95 ≤1.5 s @ N=10⁴; shortlist recall@200 ≥0.98 vs brute force |
| **P5 agent tools** | wk 6–7 | E2 | FastAPI tools §3.1, hard policies, verifier, demo loop on incident triage | **G5:** end‑to‑end demo: new incident → retrieve → map → project → verify → answer **with provenance on 100% of emitted claims**; policy tests prove free‑text facts are rejected |
| **P6 consolidation** | wk 7–8 | E1 | SAGE pools, frequency probabilities, decay | **G6:** on synthetic streams, recovered schemas match generating schemas (fact‑set F1 ≥0.9); probabilities equal analytic frequencies exactly |
| **P7 evaluation** | wk 8–11 | E3 | calibration (γ,ρ,δ,θ) per §2.6 then **freeze**; run §8 in full; ablations; stats | **G7:** preregistered analysis executed verbatim; report.html generated from one command; all numbers reproduce from seeds |
| **P8 paper & release** | wk 11–12 | E3+all | paper per §9; Docker; artifact DOI | **G8:** `docker run sma:paper make all` reproduces every table/figure |

**Why this order:** the matcher (P2) is the scientific core and the riskiest math — front‑load it against the canonical battery before any data touches it; encoders (P3) are independent and parallelizable; retrieval (P4) depends on both; calibration is quarantined to P7 *before freezing*, so no metric leakage into design.

---

## 8. Evaluation: SMA vs RAG vs knowledge graphs (the experiment that makes or breaks the paper)

### 8.1 Baselines (all share SMA's Tier‑0 *facts* where applicable, so the comparison isolates the *memory mathematics*)
- **B1 BM25** (rank‑bm25) over raw case text.
- **B2 Dense RAG:** bge‑base‑en‑v1.5 embeddings + HNSW cosine over raw case text (embeddings are permitted here precisely because *this is what RAG is*; they never enter SMA).
- **B3 Hybrid:** reciprocal‑rank fusion of B1+B2 (the strong practical RAG).
- **B4 KG‑PPR (fair KG):** the *same deterministic Tier‑0 triples* loaded into Kuzu as an entity graph; query = entity seeding + Personalized PageRank over the graph (HippoRAG‑style mechanics without LLM OpenIE, R12) → ranked cases. This isolates "graph semantics" vs "structure mapping" with extraction held constant — the apples‑to‑apples KG comparison the user asked for.
- **B5 HippoRAG 2 as published** (LLM OpenIE + PPR): the cost/fragility comparator; we report its accuracy *and* its extraction token bill vs SMA's $0.
- **B6 Long‑context stuffing** (concatenate top‑U cases into the orchestrator context): controls for "maybe you don't need retrieval at all."

### 8.2 The decisive instrument — SSB, the Synthetic Structural Benchmark (T1)
Generator (`sma/eval/ssb_generator.py`, seeded):
1. Sample a relational **schema** S: depth d∈{2..5}, branching b∈{2..4}, HO‑relation density h∈{0.2..0.6}, entity count e∈{4..12}.
2. **True analog** A(Q): apply a *vocabulary bijection* (every functor renamed via a disjoint lexicon), entity renaming, statement order shuffle, and optional paraphrase templates → same structure, **zero lexical overlap**.
3. **Surface distractor** D(Q): same functor unigram histogram as Q (matched content vector!) but relations re‑wired to break the justification DAG (degree‑preserving rewiring) → same vocabulary, broken structure.
4. Emit triples (Q, A, D) with full gold correspondences. 5,000 triples across a 2×2 of {near/far vocabulary} × {intact/scrambled structure}.
**Why this is the right instrument:** it operationalizes exactly Gentner's surface/structure orthogonalization (R2, R10) under perfect ground truth. Prediction: B1–B3 rank D over A in the far‑vocabulary cell (their similarity *is* the matched histogram); SMA ranks A over D (its score *is* the structure). If that prediction fails, the theory fails honestly — see §12 kill criteria.

### 8.3 Real‑data tasks
- **T2 incident memory (primary applied claim):** LogHub HDFS (575,061 sessions, 16,838 anomalous), BGL, OpenStack (R13). Cases = sessions. Tasks: (a) **triage‑by‑analogy** — retrieve top‑k labeled past incidents, predict label by SESₙ‑weighted vote; (b) **cross‑system transfer** — index BGL, query Thunderbird (and HDFS→OpenStack): vocabularies differ, failure *motifs* (timeout→retry→saturation cascades) recur — the unseen‑concept test on real data; (c) **resolution suggestion** on incident pairs with known fixes (candidate‑inference precision, human‑rated n=200).
- **T3 bug‑fix memory:** BugsInPy (and optionally Defects4J): case = AST+trace structure of bug; retrieve analogous fixed bug; metric: fix‑pattern category match @k.
- **T4 narrative analogy (Tier‑1, flagged):** ARN far/near partitions (R10) as a retrieval/binary task for SMA vs baselines; Karla‑the‑Hawk‑style sets for **H5** (does SESₙ reproduce the human soundness ordering: analogy ≈ literal > surface/first‑order?).
- **T5 agent‑drift protocol:** a seeded synthetic ops world (services, dependencies, injected faults) where ground‑truth state is known. Agent variants share the same orchestrator LLM and differ ONLY in memory: (i) context‑only, (ii) RAG‑notes (B3 store of the agent's own text notes), (iii) SMA. Run horizons t=10..200 steps; measure per‑step **state‑fidelity F1** (agent beliefs vs ground truth via probe queries), contradiction rate, task success vs t. H4 = SMA's decay slope is shallower (R11 motivates the metric).

### 8.4 Pre‑registered hypotheses (write configs/preregistration.md at P0; freeze before P7 data)
- **H1 (structure > semantics off‑surface):** on SSB far‑vocabulary and T2 cross‑system, SMA beats best of B1–B4 on R@10/MRR with large effect (Cliff's δ ≥ 0.474).
- **H2 (no within‑domain tax):** on T2 within‑system triage, SMA is non‑inferior to B3 (margin: −2 pts F1).
- **H3 (verifiable inference > generative recall):** human‑rated precision of SMA candidate inferences ≥ B3+LLM‑generated suggestions on T2(c), with 100% provenance coverage vs ~0%.
- **H4 (drift):** T5 fidelity‑decay slope (i) > (ii) > (iii)=SMA; SMA slope ≈ encoder error, near‑zero compounding.
- **H5 (human alignment):** SESₙ reproduces the canonical soundness ordering on Karla sets; Spearman ρ ≥ 0.6 vs ratings.
- **H6 (cost):** SMA extraction/index token cost = $0 LLM tokens; report B5's per‑10k‑doc OpenIE bill and wall‑clock; SMA CPU‑only.

### 8.5 Metrics & statistics
R@{1,5,10}, MRR, nDCG (ranx); mapping‑correspondence F1 vs SSB gold; inference precision; macro‑F1 (triage); decay slopes (OLS on F1 vs t); latency/throughput; $ and joules. Paired bootstrap (10⁴ resamples) for every comparison; Holm–Bonferroni across H1–H6; report effect sizes, not just p; all seeds fixed and published.

### 8.6 Calibration protocol (no leakage)
Fit (γ, ρ, δ) by Bradley–Terry on Karla + SSB‑validation only (§2.6‑A); fit (θ_a, θ_p) on T2 validation split only; **freeze**, version as `score‑v1`, then run all test sets once. Sensitivity appendix: every headline number at γ ∈ {0, .125, .25, .5} and under Regime B (MDL).

### 8.7 Ablations (each is a one‑flag config)
γ=0 (kills systematicity → expect far‑analogy collapse, the internal validity check); no canonicalization lattice; no HO relations in encoders (flat events only); greedy vs CP‑SAT (quality‑gap vs latency); WL‑1 off (pure MAC content vectors, i.e., faithful 1995 MAC); SES vs MDL; SAGE off (exemplars only).

---

## 9. MVP for an academic paper

- **Claim set (scoped to what the evidence can carry):** (C1) a certified MAC/FAC retrieval bound (Lemma 2 + best‑first exactness) — a small but real algorithmic contribution; (C2) an exact‑anytime gmap merge via the MWIS reduction (Lemma 1 + CP‑SAT) — modernizes greedy SME with measured optimality gaps; (C3) an MDL derivation of systematicity and one‑to‑one (§2.6‑B) — theory contribution linking SMT to R14; (C4) the empirical H1–H6 results on SSB/LogHub/ARN/drift; (C5) the open CPU‑only artifact.
- **Paper skeleton:** Intro (drift + off‑distribution brittleness, R9–R11) → SMT background (R1–R6) → SMA formalism (§2) → system (§3–§5) → experiments (§8) → debate‑honest related work (incl. Webb et al. reply, AMN R15, HippoRAG R12) → limitations (§0.3, §12).
- **Venues:** primary CogSci or AAAI (theory+system fit); ACL Findings if T4 leads; a NeurIPS/ICLR memory‑or‑reasoning workshop as the fast first airing. Pre‑registration cited in the paper.
- **Reproducibility bar:** one Docker image, `make all` regenerates every number; datasets fetched by checksum scripts; no GPU needed for the system.

---

## 10. /goal — paste this into GOALS.md; it is the implementing agent's termination contract

```
/goal SMA-MVP-1
MISSION: Build and evaluate Structure-Mapping Agentic Memory exactly per blueprint §2–§8.
DEFINITION OF DONE (all must hold):
  D1. pytest -m "gate_G0 or gate_G1 or gate_G2 or gate_G3 or gate_G4 or gate_G5 or gate_G6" → 100% pass on CI.
  D2. Canonical battery: water-flow/heat-flow and solar-system/atom mappings match published SME results;
      ≥25-pair agreement with the SME v4 oracle (correspondence F1 ≥ 0.95 per pair).
  D3. Certified retrieval verified: top-k by best-first FAC == brute-force top-k on the 1k sample (exact match).
  D4. Preregistration frozen before any test-set run (git tag prereg-v1 precedes eval commits).
  D5. Full §8 evaluation executed; report.html reproduces from `make report` with fixed seeds.
  D6. Every agent-surfaced claim in the P5 demo carries a provenance record (automated check).
  D7. Honest-outcome rule: H1–H6 results are reported whether positive or negative; negative H1 triggers §12.K1
      diagnostics and, if confirmed, the negative-result write-up — that ALSO satisfies this goal.
STOP CONDITIONS: D1–D7 satisfied, OR a §12 kill criterion fires and its prescribed pivot/write-up is delivered.
NON-GOALS (do not drift into): training any model; LLM-based extraction; UI polish; >2 domains beyond logs/code in MVP.
BUDGET: 12 person-weeks (×2.5 if solo agent); one 16-core/64GB CPU box; LLM tokens only for orchestration/demo/judging.
```

### 10.1 Incremental‑learning protocol for the implementer (what to do as reality bites)
1. **STATUS.md ledger** after every work session: date, phase, gates attempted/passed, metrics observed, surprises, next action. Append‑only.
2. **ADRs** (docs/ADR/NNN.md) for every decision that touches §2 math, §5 versions, or §8 design: context → options → decision → consequences. *No silent changes*: any change to scoring/constraints requires a new `score‑vN`, re‑running G2 and the sensitivity sweep.
3. **New external findings rule:** when a relevant paper/tool surfaces mid‑build, log it in ADR with one of {adopt‑now (only if it removes a blocker), defer‑to‑v2, cite‑only}. Default is defer — protect the preregistration.
4. **Escalation rule for an agent implementer:** if any gate fails twice after the tripwire response below, halt that phase, write the failure analysis to STATUS.md, and request human review rather than improvising around the spec.

### 10.2 Tripwire table (observation → predefined response; no improvisation)
| Tripwire | Predefined response |
|---|---|
| G2 battery mismatch vs SME v4 | diff at MH level first (seeding), then kernels, then merge; the oracle defines correctness — fix code, never "reinterpret" the battery |
| G3 template accuracy below Drain's published numbers | masking config bug; restore reference config from Drain3 benchmarks; re‑golden |
| Encoder statement precision <90% on audit | tighten rules; shrink claim scope per adapter; never add a statistical fixer to Tier‑0 |
| FAC too slow (>50 ms median @ n=300) | (1) cap MHs per functor group via U‑ordering, (2) greedy‑only mode, (3) port `mh.py`/`kernels.py` hot loops to Rust (rustworkx ext) — in that order |
| CP‑SAT gap >5% at budget on >10% of pairs | report greedy as primary, CP‑SAT as oracle subset; this is a finding, not a failure |
| MAC shortlist recall <0.98 | raise K₀; add WL‑2 features; only then consider learned re‑ranker **as a flagged ablation, never core** |
| Lattice‑miss rate >30% on a domain | extend adapter ontology (data change), log coverage; do not loosen δ globally |
| SSB shows B2 ≈ SMA on far‑vocabulary | check paraphrase leakage in generator (bijection must be total); verify D(Q) histogram matching; then accept the result (→ §12.K1) |
| θ_a grid flat / SAGE over‑merges | switch normalization to SESₙ‑both‑ways min; report instability as a limitation |

---

## 11. Where each user requirement is satisfied (traceability)
- *No LLM extraction, LLM orchestration only* → §3.1 hard API rules, §4 policy, enforced by tests (G5).
- *Mathematical core of SME, not its source* → §2.2–§2.5 reimplements the published algorithm; SME v4 used only as a test oracle (G2).
- *Non‑heuristic weights* → §2.6: identified estimation with CIs + sensitivity (Regime A) and a zero‑parameter MDL scorer with a derivation of systematicity/one‑to‑one (Regime B); SAGE probabilities are frequencies (§2.9).
- *Domain‑agnostic* → IR has no domain terms (§2.1); domains live in boundary adapters + lattices (§4); transfer is *tested*, not assumed (T2‑b, H1).
- *Scalable* → certified MAC/FAC (§2.7), O(n² log n) merge fallback (R3), shardable ANN, parallel FAC, complexity table §2.8, perf gates G2/G4.
- *Tested against RAG and KGs* → §8.1 B1–B6 incl. extraction‑held‑constant KG (B4) and as‑published HippoRAG (B5).
- *MVP for a paper* → §9; preregistration §8.4; reproducibility G8.
- */goal + adaptation to findings/obstacles* → §10.
- *who/when/why/how/where* → §7 table; rationale lines throughout.

## 12. Risk register and kill criteria
- **R1 Extraction ceiling (free prose).** Highest scientific risk; mitigated by scoping headline claims to Tier‑0 domains (§0.3). Residual: accept.
- **R2 Representation sensitivity.** SME quality tracks encoding consistency (the classic critique). Mitigation: canonicalization, adapter conventions, plus an **encoder‑noise robustness test** (perturb 10–20% of statements; report score degradation curves).
- **R3 Lattice coverage** in technical vocab → tripwire row; surfaced as a measured coverage number, not hidden.
- **R4 NP‑hardness** → anytime exact + published greedy; gaps logged (§2.5).
- **R5 Hypothesis risk.** *Within‑domain*, surface retrieval is genuinely strong (MAC/FAC's own psychology says superficial remindings dominate retrieval, R5) — hence H2 is parity, and the win condition is off‑surface transfer + verifiability + cost, not universal dominance.
- **K1 (kill/pivot):** if, with a verified generator and encoders (tripwires exhausted), SMA fails H1 on SSB with δ_Cliff < 0.147 — the core conjecture is disconfirmed in the cleanest possible setting → publish the negative result with the diagnostic decomposition (retrieval vs mapping vs scoring), which is itself a contribution.
- **K2:** if Tier‑0 encoders cannot reach G3 precision on logs/code after 2 extra weeks → the bottleneck finding is reported; pivot MVP to SSB‑only theory paper (C1–C3 stand on their own).
- **K3:** if latency gates miss by >5× after the Rust port → reduce N claims to 10⁴ and state the engineering frontier explicitly.

## 13. References (verified 2026‑06‑11)
Falkenhainer, Forbus & Gentner (1986 AAAI; 1989 *Artif. Intell.* 41:1–63) — SME. · Gentner (1983, *Cog. Sci.* 7) — structure‑mapping theory/systematicity. · Forbus, Gentner & Law (1995, *Cog. Sci.* 19) — MAC/FAC. · Law, Forbus & Gentner (1994 CogSci) — identicality vs similarity tables. · Forbus & Oblinger (1990 CogSci) — greedy SME. · Forbus, Ferguson, Lovett & Gentner (2017, *Cog. Sci.* 41(5)) — large‑scale SME, O(n² log n). · Veale & Keane (1997 IJCAI) — NP‑hardness. · Kuehne et al. (2000) SEQL; Halstead & Forbus (2005); McLure, Friedman & Forbus (2015) — SAGE; AILEEN (arXiv 2006.01962; 2210.11731) — SAGE thresholds/probabilities. · Gentner, Rattermann & Forbus (1993, *Cog. Psych.*) — soundness/retrieval materials. · Gentner & Forbus (2025, *Curr. Dir. Psych. Sci.*) — SME retrospective. · QRG SME v4: qrg.northwestern.edu/software/sme4. · Lewis & Mitchell (2024, arXiv 2402.08955) and Webb, Holyoak & Lu (2024 reply, arXiv 2404.13070) — the LLM‑analogy robustness debate. · Sourati et al. (2024, TACL) — ARN. · Gutiérrez et al. (NeurIPS 2024; ICML 2025) — HippoRAG/2. · He et al. (ISSRE 2023) — LogHub; github.com/logpai/loghub. · Murena et al. (IJCAI 2020; 2017) — minimal‑complexity analogy. · Rissanen (1978) — MDL. · Shervashidze et al. (2011, JMLR) — WL kernels. · Crouse et al. (AAAI 2021) — Neural Analogical Matching. · Agent drift / reliability: arXiv 2601.04170; arXiv 2602.16666.

*End of blueprint. A competent engineer or CLI agent should be able to start at §7‑P0 and proceed gate by gate without further design decisions; every remaining judgment call has a predefined tripwire response in §10.2.*
