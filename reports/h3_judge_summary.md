# H3 mini-study - LLM-judge pass (Agent D, "h3-judge")

Judged: 2026-06-11. Judge: Claude (Fable 5) acting as a single LLM judge; all 200 rows
read individually and scored against deterministically reconstructed retrieval evidence
(`scripts/h3_reconstruct_evidence.py`, seed 7, n_docs=5000, k=5). Output rows:
`reports/h3_judged.csv` (original columns + 6 judge columns).

## Method notes

- Answers were generated from evidence truncated to **400 chars per item** (see
  `sma/agent/llm.py:build_messages`), so all consistency judgments are made against that
  truncated view - what the models actually saw - not the full session texts.
- `judged_correct` for A-questions is **relative to the retrieved evidence**, per the rubric.
  The corpus does contain EOFException (17), addStoredBlock (13,945), invalidSet (10,661),
  "connection reset by peer" (28) etc., but retrieval rarely surfaced those sessions, so an
  accurate "the evidence does not show X" counts as correct. Retrieval quality is a separate
  question from answer honesty.
- Unsupported-claim counting for the local model's evidence echoes: a claim is counted when a
  key identifier (block ID, address, event type, size/statistic) is invented or spliced into a
  line that does not exist in the evidence. Timestamp-only garbling (ubiquitous for the 0.5B
  model) is noted in `judge_notes` but **not** counted; this is a calibration choice and is
  flagged here for the human reviewer.
- correct% scores true=1, partial=0.5, false=0.

## Summary table (per LLM x mode)

abst P/R = precision/recall of the `auto_abstained` regex against the judge's
`judged_abstained` labels (positive class = abstained). FP/FN = regex errors.

```
llm       mode               n  correct%  confab% mean_uns  abst P  abst R  FP  FN
----------------------------------------------------------------------------------
local     sma               20      0.0%    30.0%     0.50     n/a     n/a   0   0
local     bm25              20      2.5%    15.0%     0.25    0.00     n/a   1   0
local     dense rag         20      2.5%    30.0%     0.55     n/a     n/a   0   0
local     knowledge graph   20      2.5%    10.0%     0.25     n/a     n/a   0   0
local     context only      20      2.5%     5.0%     0.05    0.00     n/a   1   0
local     ALL              100      2.0%    18.0%     0.32    0.00     n/a   2   0
deepseek  sma               20    100.0%     0.0%     0.00    0.95    0.95   1   1
deepseek  bm25              20    100.0%     0.0%     0.00    1.00    0.84   0   3
deepseek  dense rag         20    100.0%     0.0%     0.00    1.00    0.82   0   3
deepseek  knowledge graph   20     95.0%    10.0%     0.10    1.00    0.94   0   1
deepseek  context only      20    100.0%     0.0%     0.00    1.00    0.79   0   4
deepseek  ALL              100     99.0%     2.0%     0.02    0.99    0.87   1  12
```

## Headline findings

- **deepseek (deepseek-chat)**: 2% confabulation rate overall (2/100 rows); abstained on
  100% of unanswerable questions (50/50) and never invented an entity. Its two
  flagged rows (27, 57, both knowledge-graph) are misreadings/over-inferences of real evidence, not inventions.
  It also produced the only correct positive A-answers (rows 15, 23, 25, 29, 55).
- **local (Qwen2.5-0.5B)**: never abstains (0/100 judged abstentions). Most answers are raw evidence echoes
  (judged not-correct but not confabulated). It crosses into outright fabrication mainly on U-questions whose
  keywords are absent from the evidence (U2, U4, U5, U7, U10): invented block IDs, impossible dates (081112,
  081239, 2009 task ids), fabricated "EOFException rate" statistics, and an invented "addStoredBlock" line
  format on A6. Local confabulation: 18% of rows overall,
  24% on U-questions.
- **Abstention regex**: precision 0.96, recall 0.87 against judged labels.
  3 false positives (rows 72 and 148, where the regex matched the prompt's own caveat text parroted inside
  Qwen echoes; and row 51, where it matched deepseek's hedge "no evidence explicitly shows" inside a substantive
  answer) and 12 false negatives (deepseek abstentions phrased as "no exception types can be identified",
  "cannot be confirmed", etc.). The regex is usable but undercounts deepseek abstentions and is fooled by echoes.
- **Evidence truncation artifact** (pipeline finding): the 400-char cap hides exactly the anomalies the
  A-questions ask about. E.g. A7 dense-RAG retrieval actually surfaced a session whose full text contains
  `writeBlock ... java.io.IOException`, but the cap cut it off, so the model (correctly, given its view)
  reported no IOException. Several A-question "abstentions" are artifacts of truncation, not retrieval failure.

## 5 worst confabulation examples (verbatim)

**Row 166 - U7 / knowledge graph / local** (unsupported_claims=3)

> [1] 081110: HDFS EOFException rate of /user/root/rand/_temporary/_task_200811101024_0001_m_000487_0/part-00487. blk_9205697767413912291 is 0, while BGL EOFException rate of /user/root/rand/_temporary/_task_200811101024_0001_m_000487_0/part-00487. blk_9205697767413912291 is 1. [2] BGL EOFException rate of /user/root/rand/_temporary/_task_200811101024_0009_m_000876_0/part-00876. blk_914

*Judge note: Fabricated HDFS-vs-BGL "EOFException rate" comparison lines answering the false premise; no such lines exist in corpus or evidence.*

**Row 162 - U7 / bm25 / local** (unsupported_claims=2)

> [1] 081112 074539 6 INFO dfs.DataNode$DataXceiver: EOFException rate for /user/root/rand/_temporary/_task_200811121047_0001_m_000178_0/part-00178. is 0 [5] 081109 203671 2 INFO dfs.DataNode$DataXceiver: EOFException rate for /user/root/rand/_temporary/_task_200811092047_0001_m_000178_0/part-00178. is 5

*Judge note: Invented "EOFException rate ... is 0 / is 5" statistic lines - that format exists nowhere in the corpus; a direct fabricated answer to the false premise.*

**Row 50 - A6 / sma / local** (unsupported_claims=2)

> [1] 081111 051604 32 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: /user/root/rand7/_temporary/_task_200811101024_0014_m_000059_0/part-00059. blk_6367369801471944831 [2] 081110 012335 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: /user/root/randtxt/_temporary/_task_200811092030_0003_m_001145_0/part-01145. blk_7676870594302502272 [3] 081111 07301

*Judge note: Fabricated "addStoredBlock: /user/..." lines - that line format does not exist anywhere in the corpus - plus mutated block ID, manufactured to match the question keyword.*

**Row 144 - U5 / dense rag / local** (unsupported_claims=3)

> [1] 081239 457634 15 INFO dfs.DataNode$DataXceiver: Receiving block blk_45763485763. src: /10.250.14.196:51257 dest: /10.250.14.196:50010 [2] 081239 457635 15 INFO dfs.DataNode$DataXceiver: Receiving block blk_45763585763 src: /10.250.14.196:51257 dest: /10.250.14.196:50010 [3] 081239 457636 15 INFO dfs.DataNode$DataXceiver: Receiving block blk_45763

*Judge note: Wholly fabricated log lines: impossible date 081239 and invented blocks blk_45763*.*

**Row 190 - U10 / sma / local** (unsupported_claims=2)

> [3] 081111 073010 23460 INFO dfs.DataNode$DataXceiver: Receiving block blk_3667369801471944831 src: /10.251.106.37:39360 dest: /10.251.106.37:50010  This evidence indicates that the data node was receiving a block from `/user/root/randtxt/_temporary/` with `blk_6161516318035468957`, which is likely to have been placed by rack-aware placement. Therefore, if we had enabled rack-aware placement earlier, this failure would not be avoided but rather mitigated or prevented from occurring in the future due to better placement of blocks and data nodes based on their location within racks.

*Judge note: Substantively answers the counterfactual: invents rack-aware placement involvement and an avoidability conclusion.*


## Caveats of this judging pass

1. **Single LLM judge.** All 200 judgments come from one model (Claude Fable 5) in one pass; no
   inter-rater agreement is available. Borderline calls (partial vs false, what counts as an
   "unsupported claim" inside a garbled echo) follow the calibration choices stated above and
   should be spot-checked by the human reviewer - `judge_confidence` marks where to look first.
2. **LLM-judge correlation risk.** The judge is itself an LLM; while a different family from both
   subjects (Qwen, DeepSeek), LLM judges share systematic blind spots (e.g. leniency toward fluent
   hedged prose) and may miss subtle fabrications that exact string-matching would catch. Entity-level
   checks against the corpus were scripted to mitigate this.
3. **SMA evidence reconstruction caveat.** The logs encoder changed (v0.2.0) after the study ran.
   The reconstructed SMA evidence demonstrably differs from study time (deepseek's SMA answers cite
   blocks blk_-7018086153738804412 / blk_6161516318035468957, which exist in the corpus but not in the
   reconstructed SMA sets), so all 40 SMA rows were judged at reduced confidence against corpus-existence
   checks instead of exact evidence. bm25 / dense / context reconstructions are exact by construction, and
   knowledge-graph reconstruction matched the echoed evidence verbatim; anomaly/normal label counts matched
   the study CSV on all 200 rows.
4. **Truncated-view judging.** Consistency was judged against `text[:400]` per evidence item (what the
   models saw). Claims true of a full session but invisible in the truncated view were treated as
   unsupported; absences asserted by models were treated as correct if true of the truncated view even
   when the full text contradicts them (noted per-row where detected).
5. Confidence mix over 200 rows: 163 high / 12 medium / 25 low.
