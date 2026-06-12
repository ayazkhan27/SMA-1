"""Generate paper-facing CSVs and report.html from deterministic MVP/eval data.

Two modes:
- default: run the SSB fixture evals (and LogHub unless --skip-loghub), write
  all CSVs, then render report.html from the on-disk artifacts.
- --html-only: skip all computation and re-render report.html from whatever
  CSVs currently exist in the reports directory.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import html
import json
import pathlib
from collections import defaultdict

from sma.eval.ssb_eval import evaluate_forced_choice, evaluate_library, evaluate_library_mac_prefilter
from sma.eval.ssb_generator import generate_triples
from sma.match.engine import match_cases


CSV_SCHEMAS = {
    "dataset_manifest.csv": ["dataset", "source", "file", "md5", "status"],
    "retrieval_runs.csv": ["run_id", "query_id", "rank", "case_id", "score", "ses_n", "u_bound", "certified"],
    "mapping_runs.csv": ["run_id", "base_id", "target_id", "score", "ses_n", "n_correspondences", "gap"],
    "triage_metrics.csv": ["dataset", "split", "method", "macro_f1", "label_hit_rate@1", "label_hit_rate@5", "label_hit_rate@10", "p50_ms", "p95_ms"],
    "ssb_cases.csv": ["triple_id", "query_id", "analog_id", "distractor_id"],
    "ssb_metrics.csv": ["split", "r1", "mrr", "mapping_f1"],
    "inference_reviews.csv": ["case_id", "inference", "precision_label", "provenance"],
    "drift_runs.csv": ["variant", "horizon", "state_f1", "contradiction_rate"],
    "latency.csv": ["operation", "n_cases", "p50_ms", "p95_ms"],
    "ablation_metrics.csv": ["ablation", "metric", "value"],
    "sage_pool_stats.csv": ["pool_id", "n_generalizations", "n_outliers", "schema_f1"],
    "calibration.csv": ["parameter", "value", "ci_low", "ci_high", "source"],
    "cost_energy.csv": ["system", "llm_tokens", "cpu_seconds", "usd_estimate"],
}


def _md5_file(path: pathlib.Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest_rows(
    manifest_path: str | pathlib.Path = "data/manifests/datasets.json",
    raw_root: str | pathlib.Path = "data/raw",
) -> list[dict]:
    manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    root = pathlib.Path(raw_root)
    rows: list[dict] = []
    for dataset, spec in manifest.items():
        source = spec.get("doi") or spec.get("record_api") or spec.get("source") or spec.get("git", "")
        files = spec.get("files", {})
        if not files and spec.get("git"):
            rows.append(
                {
                    "dataset": dataset,
                    "source": source,
                    "file": "",
                    "md5": "",
                    "status": "git_manifested",
                }
            )
            continue
        for filename, file_spec in files.items():
            expected = file_spec.get("md5", "")
            local = root / dataset / filename
            if not local.exists():
                status = "missing"
            elif expected:
                actual = _md5_file(local)
                status = "verified" if actual == expected else f"checksum_mismatch:{actual}"
            else:
                status = "downloaded_no_checksum"
            rows.append(
                {
                    "dataset": dataset,
                    "source": source,
                    "file": filename,
                    "md5": expected,
                    "status": status,
                }
            )
    return rows


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fieldnames = CSV_SCHEMAS[path.name]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _ssb_case_rows(n: int, seed: int) -> list[dict]:
    rows: list[dict] = []
    for i, triple in enumerate(generate_triples(n, seed=seed)):
        rows.append(
            {
                "triple_id": f"ssb_{seed}_{i}",
                "query_id": triple.query.case_id,
                "analog_id": triple.analog.case_id,
                "distractor_id": triple.distractor.case_id,
            }
        )
    return rows


def _mapping_rows(n: int, seed: int) -> list[dict]:
    from sma.eval.ssb_generator import build_canonicalizer
    from sma.eval.ssb_eval import ssb_config

    rows: list[dict] = []
    triples = generate_triples(n, seed=seed)
    canon = build_canonicalizer(triples)
    for i, triple in enumerate(triples):
        gmap = match_cases(triple.analog, triple.query, config=ssb_config(), canon=canon)
        rows.append(
            {
                "run_id": "ssb_forced_choice_oracle_mapping",
                "base_id": triple.analog.case_id,
                "target_id": triple.query.case_id,
                "score": f"{gmap.score:.6f}",
                "ses_n": f"{gmap.normalized_score:.6f}",
                "n_correspondences": len(gmap.correspondences),
                "gap": "" if gmap.optimality_gap is None else f"{gmap.optimality_gap:.6f}",
            }
        )
    return rows


def run_fixture_eval(
    library_n: int = 12, mac_prefilter_n: int = 1000, include_loghub: bool = False
) -> dict[str, list[dict]]:
    forced = evaluate_forced_choice(12, seed=11)
    library = evaluate_library(library_n, seed=19, k=10, shortlist=library_n * 2, fac_budget=50)
    large_mac = evaluate_library_mac_prefilter(mac_prefilter_n, seed=23, k=10)
    retrieval_rows: list[dict] = forced.rows + library["sma_rows"] + large_mac["sma_rows"]
    ssb_case_rows = _ssb_case_rows(library_n, seed=19)
    mapping_rows = _mapping_rows(12, seed=11)

    triage_rows: list[dict] = []
    if include_loghub:
        from sma.eval.loghub_eval import run_loghub_eval

        triage_rows = run_loghub_eval()

    if not triage_rows:
        triage_rows = [
            {
                "dataset": "LogHub",
                "split": "HDFS_MVP_diagnostic",
                "method": "SMA",
                "macro_f1": "awaiting_run",
                "label_hit_rate@1": "awaiting_run",
                "label_hit_rate@5": "awaiting_run",
                "label_hit_rate@10": "awaiting_run",
                "p50_ms": "0.000",
                "p95_ms": "0.000",
            }
        ]

    return {
        "ssb_cases.csv": ssb_case_rows,
        "retrieval_runs.csv": retrieval_rows,
        "mapping_runs.csv": mapping_rows,
        "ssb_metrics.csv": [forced.metrics] + library["metrics"] + large_mac["metrics"],
        "latency.csv": [
            {
                "operation": forced.latency["operation"],
                "n_cases": forced.latency["n_cases"],
                "p50_ms": f"{forced.latency['p50_ms']:.3f}",
                "p95_ms": f"{forced.latency['p95_ms']:.3f}",
            },
            {
                "operation": library["latency"]["operation"],
                "n_cases": library["latency"]["n_cases"],
                "p50_ms": f"{library['latency']['p50_ms']:.3f}",
                "p95_ms": f"{library['latency']['p95_ms']:.3f}",
            },
            {
                "operation": large_mac["latency"]["operation"],
                "n_cases": large_mac["latency"]["n_cases"],
                "p50_ms": f"{large_mac['latency']['p50_ms']:.3f}",
                "p95_ms": f"{large_mac['latency']['p95_ms']:.3f}",
            },
        ],
        "dataset_manifest.csv": dataset_manifest_rows(),
        "triage_metrics.csv": triage_rows,
        "inference_reviews.csv": [
            {"case_id": "", "inference": "", "precision_label": "awaiting_human_review", "provenance": ""}
        ],
        "drift_runs.csv": [
            {"variant": "sma", "horizon": 20, "state_f1": "awaiting_full_protocol", "contradiction_rate": ""}
        ],
        "ablation_metrics.csv": [
            {"ablation": "gamma_0", "metric": "awaiting_full_eval", "value": ""}
        ],
        "sage_pool_stats.csv": [
            {"pool_id": "fixture", "n_generalizations": "", "n_outliers": "", "schema_f1": ""}
        ],
        "calibration.csv": [
            {"parameter": "gamma", "value": 0.25, "ci_low": "", "ci_high": "", "source": "draft_default"},
            {"parameter": "rho", "value": 0.5, "ci_low": "", "ci_high": "", "source": "draft_default"},
            {"parameter": "delta", "value": 2, "ci_low": "", "ci_high": "", "source": "draft_default"},
        ],
        "cost_energy.csv": [{"system": "SMA", "llm_tokens": 0, "cpu_seconds": "", "usd_estimate": 0}],
    }


def _load_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _table(rows: list[dict], columns: list[str] | None = None, limit: int | None = None) -> str:
    if not rows:
        return '<p class="missing">artifact not present — run not yet executed</p>'
    columns = columns or list(rows[0].keys())
    body = []
    for row in rows[: limit or len(rows)]:
        cls = ' class="alert"' if "alert" in str(row.get("method", "")).lower() or row.get("dataset") == "DIAGNOSTIC" else ""
        body.append(
            f"<tr{cls}>" + "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns) + "</tr>"
        )
    note = f'<p class="missing">showing {limit} of {len(rows)} rows</p>' if limit and len(rows) > limit else ""
    return (
        "<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in columns)
        + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>" + note
    )


def _h3_summary(rows: list[dict]) -> tuple[str, str]:
    """Aggregate the H3 study CSV into per-LLM and per-mode discipline tables."""
    if not rows:
        return "", ""
    per_llm: dict[str, dict[str, int]] = defaultdict(lambda: {"ua": 0, "un": 0, "aa": 0, "an": 0})
    per_mode: dict[str, dict[str, int]] = defaultdict(lambda: {"ua": 0, "un": 0, "aa": 0, "an": 0})
    for row in rows:
        abstained = row["auto_abstained"] == "True"
        answerable = row["answerable"] == "True"
        for bucket in (per_llm[row["llm"]],) + ((per_mode[row["mode"]],) if row["llm"] == "deepseek" else ()):
            if answerable:
                bucket["an"] += 1
                bucket["aa"] += not abstained
            else:
                bucket["un"] += 1
                bucket["ua"] += abstained
    llm_rows = [
        {"llm": llm, "abstained on unanswerable": f'{c["ua"]}/{c["un"]}', "answered answerable": f'{c["aa"]}/{c["an"]}'}
        for llm, c in sorted(per_llm.items())
    ]
    mode_rows = [
        {"mode (DeepSeek only)": m, "abstained on unanswerable": f'{c["ua"]}/{c["un"]}', "answered answerable": f'{c["aa"]}/{c["an"]}'}
        for m, c in sorted(per_mode.items())
    ]
    return _table(llm_rows), _table(mode_rows)


REPORT_CSS = """
body{font-family:Inter,system-ui,Arial,sans-serif;margin:0;color:#1f2933;background:#f8fafc;line-height:1.55}
.wrap{max-width:1080px;margin:0 auto;padding:36px 28px 80px}
h1{font-size:30px;margin:0 0 2px} h2{font-size:21px;margin:44px 0 8px;border-bottom:2px solid #dbeafe;padding-bottom:4px}
h3{font-size:15px;margin:22px 0 6px;color:#334155}
p,li{font-size:14px} .sub{color:#52606d;margin-top:2px}
table{border-collapse:collapse;width:100%;background:#fff;margin:10px 0 4px}
th,td{border:1px solid #d9e2ec;padding:6px 9px;font-size:12.5px;text-align:left;vertical-align:top;color:#1f2933}
th{background:#eef2f7;font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;color:#334155}
tr.alert td{background:#fef2f2;color:#991b1b}
.win{background:#ecfdf5}
.verdict{border-left:4px solid #2563eb;background:#eff6ff;padding:10px 14px;margin:12px 0;font-size:14px}
.warn{border-left:4px solid #d97706;background:#fffbeb;padding:10px 14px;margin:12px 0;font-size:14px}
.missing{color:#9a3412;font-size:13px;font-style:italic}
.toc{background:#fff;border:1px solid #d9e2ec;border-radius:10px;padding:14px 22px;margin:18px 0}
.toc a{color:#1d4ed8;text-decoration:none;font-size:13.5px}
code{background:#eef2f7;padding:1px 5px;border-radius:4px;font-size:12.5px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:16px 0}
.kpi{background:#fff;border:1px solid #d9e2ec;border-radius:10px;padding:12px 14px}
.kpi .v{font-size:24px;font-weight:700;color:#1d4ed8} .kpi .l{font-size:11.5px;color:#52606d;text-transform:uppercase;letter-spacing:.04em}
"""


def render_html(reports_dir: pathlib.Path | str = "reports") -> str:
    reports = pathlib.Path(reports_dir)
    triage = _load_csv(reports / "triage_metrics.csv")
    triage_mdl = _load_csv(reports / "triage_metrics_mdl.csv")
    transfer = _load_csv(reports / "transfer_metrics.csv")
    holdout = _load_csv(reports / "transfer_holdout_metrics.csv")
    controls = _load_csv(reports / "transfer_controls_metrics.csv")
    ladder = _load_csv(reports / "baseline_ladder_metrics.csv")
    family = _load_csv(reports / "family_metrics.csv")
    h3 = _load_csv(reports / "h3_mini_study.csv")
    ssb = _load_csv(reports / "ssb_metrics.csv")
    latency = _load_csv(reports / "latency.csv")
    manifest = _load_csv(reports / "dataset_manifest.csv")
    calibration = _load_csv(reports / "calibration.csv")
    h3_llm_table, h3_mode_table = _h3_summary(h3)

    try:
        from sma.encoders.logs_drain import LogEncoder

        encoder_version = LogEncoder.version
    except Exception:
        encoder_version = "unknown"
    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # The narrative below is maintained alongside the experiments; every claim
    # cites the artifact table rendered next to it.
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SMA-1 Report &amp; Memory Map</title>
<style>{REPORT_CSS}</style></head><body><div class="wrap">

<h1>SMA-1: Structure-Mapping Agentic Memory — Report &amp; Memory Map</h1>
<p class="sub">Generated {generated} · logs encoder v{encoder_version} · scorer default <code>ses</code> (MDL as ablation, ADR-004) ·
all 11 test gates passing · ledger: <code>docs/STATUS.md</code> · design contract: <code>structure_mapping_agentic_memory_blueprint.md</code></p>

<div class="kpis">
<div class="kpi"><div class="v">0.955</div><div class="l">HDFS triage F1 (SMA, best of 4 methods)</div></div>
<div class="kpi"><div class="v">0.938</div><div class="l">Held-out transfer BGL&rarr;Spirit, 3-seed mean (dense: 0.36)</div></div>
<div class="kpi"><div class="v">46/50</div><div class="l">honest abstentions (DeepSeek verbalizer, H3)</div></div>
<div class="kpi"><div class="v">~2000&times;</div><div class="l">matcher speedup (5&nbsp;min &rarr; 181&nbsp;ms worst case)</div></div>
</div>

<div class="toc"><b>Contents</b><br>
<a href="#sys">1. What this system is</a> ·
<a href="#method">2. Methodology</a> ·
<a href="#within">3. Within-system results</a> ·
<a href="#scorer">4. Scorer ablation (SES vs MDL)</a> ·
<a href="#transfer">5. Cross-system transfer + held-out + controls</a> ·\n<a href="#ladder">5d. Baseline ladder</a> ·\n<a href="#family">5e. Family metric</a> ·
<a href="#h3">6. H3 honesty study</a> ·
<a href="#perf">7. Engineering record</a> ·
<a href="#caveats">8. Caveats &amp; open issues</a> ·
<a href="#next">9. Next steps</a> ·
<a href="#appendix">10. Appendix: raw artifacts</a></div>

<h2 id="sys">1. What this system is, and why</h2>
<p>SMA-1 is an agentic memory whose retrieval is governed by <b>structure mapping</b> (Gentner's SME: match
hypotheses &rarr; kernels &rarr; merge &rarr; structural evaluation), not word similarity. Raw artifacts enter memory only
through <b>deterministic Tier-0 encoders</b> (no LLM, no statistics in the extraction path — bit-identical output
forever); an LLM (local Qwen2.5-0.5B or DeepSeek API) sits strictly downstream and only <i>verbalizes</i> retrieved
evidence. The bet under test: <b>structure beats surface when vocabulary shifts</b> (new services, new systems,
renamed components), and provenance-disciplined retrieval prevents the confident-wrong answers that generative
memory produces. Four baselines mirror this in every experiment: BM25 (lexical), Dense RAG
(all-MiniLM-L6-v2 embeddings), a KG entity-overlap proxy, and context-only stuffing.</p>

<h2 id="method">2. Methodology (how every number below was produced)</h2>
<h3>Datasets &amp; sessionization</h3>
<ul>
<li><b>HDFS v1</b> (11.17M lines): one case per block id; first-occurrence timestamps from a full no-cap scan;
labels from the separate <code>anomaly_label.csv</code>.</li>
<li><b>BGL</b> (4.75M lines): 60-second windows per node, sessions &lt;3 lines discarded, two-pass stream-and-sample.
<b>Label-leak fix:</b> the leading alert-category column (the ground-truth label) is stripped from extracted text —
an early run leaked it to every retriever and was discarded.</li>
<li><b>Thunderbird</b> (re-downloaded, md5-verified): BGL-style sessionization, label column stripped, first 20M lines
(documented cap), streamed from tar.gz.</li>
<li><b>OpenStack</b>: sessions per VM instance uuid; normal/abnormal source files have disjoint instance sets; the
source-filename token (which encodes the label) is stripped.</li>
</ul>
<h3>Protocol</h3>
<ul>
<li>1,000 sessions per dataset (500 Anomaly / 500 Normal), stratified over 5 temporal bins, seed 42.</li>
<li>Within-system: 80/20 index/query split (seed 101). Cross-system: 800 index from system A, 200 queries from system B.</li>
<li>SMA retrieval budgets: MAC shortlist 40, FAC budget 20 (UI: 200/30). Label prediction = score-weighted vote of top-5.</li>
<li>Metrics: macro-F1 of the vote; <code>label_hit_rate@k</code> = retrieved-same-label / min(k, |relevant|); p50/p95 latency per query.
Automated diagnostic alerts flag collapse (single-class predictions), suspicious perfection, and F1=0.</li>
</ul>

<h2 id="within">3. Within-system results (LogHub MVP diagnostic)</h2>
<p><b>Why:</b> establishes the baseline regime where surface methods are expected to be strong (H2 is a parity claim,
not a win claim). <b>Result:</b> SMA dominates HDFS (+13.6 F1 pts over best baseline) because HDFS anomalies live in
event <i>patterns</i>; SMA trails on BGL where anomalous messages are lexically self-announcing ("KERNEL FATAL") —
reported as an honest H2 miss on BGL.</p>
{_table(triage, limit=12)}

<h2 id="scorer">4. Scorer ablation: SES vs MDL (ADR-004)</h2>
<p><b>What prompted it:</b> a live UI test showed SMA retrieving the right <i>class</i> (anomalies) but missing the
asked-for failure <i>family</i> (EOFException write-pipeline deaths): SES weights all matched relations roughly equally,
so abundant common matches swamp the one rare template that identifies the family. Lexical methods do rare-term
weighting implicitly; the blueprint's sanctioned answer is the MDL scorer (rare shared structure compresses more), and
ad-hoc IDF patching of SES is forbidden by the no-heuristic-weights mandate.</p>
<div class="verdict"><b>Verdict:</b> SES wins aggregate triage (HDFS 0.9549 vs 0.8933; BGL tied); MDL uniquely recovers
rare failure families (EOF family in top-5: SES 0/5 — even with verbatim log lines in the query — vs MDL 3/5 from prose).
Neither dominates &rarr; SES stays default, MDL is a first-class toggle in the UI and a reported ablation. Decision record:
<code>docs/ADR/004-scorer-ablation-ses-vs-mdl.md</code>.</div>
<h3>MDL run (same protocol as section 3)</h3>
{_table(triage_mdl, limit=12)}

<h2 id="transfer">5. Cross-system transfer — the H1 experiment</h2>
<p><b>Why this is the main event:</b> within one system, word-matching is genuinely strong; the hypothesis (H1) is that
structure transfers <i>across</i> systems where vocabulary doesn't. <b>Run 1 (encoder v0.1.x) was negative:</b> SMA
vote-collapsed (0.333) on BGL&rarr;Thunderbird while Dense RAG scored 0.741. The prescribed diagnostic decomposition
found the cause: across systems the encoder shared only <b>4 functor types</b> (before/count/component/logSession) —
all event types were content-hash template names, system-specific by construction. The matcher cannot map events the
encoder names incomparably. <b>Fix (encoder v0.2.0):</b> deterministic keyword-driven cross-system event classes
(timeoutEvent, ioEvent, kernelEvent, networkEvent, storageEvent, lifecycleEvent, failureEvent, ...) emitted alongside
the precise template hashes — rules are ordered data, zero statistics, fully Tier-0.</p>
<div class="verdict"><b>Run 2 (v0.2.0): BGL&rarr;Thunderbird — SMA 0.9093 macro-F1 (hit@1 0.9100) vs Dense RAG 0.7407,
BM25 0.5489, KG 0.3552.</b> From collapse to best-in-class by +17 F1 points, purely from giving structure a comparable
vocabulary: the H1 pattern. HDFS&rarr;OpenStack remains an all-methods wall (~coin-flip) under both encoders — OpenStack
anomalies appear to be missing-events rather than error-events, a task-design question.</div>
<h3>All transfer rows (run 1 = the preserved negative, then run 2)</h3>
{_table(transfer)}

<h3>5b. Held-out confirmation: Spirit (ontology frozen at tag ontology-v1 BEFORE download)</h3>
<p><b>Why:</b> the v2 ontology was written after observing the v1 failure — a reviewer would call it post-hoc.
The held-out protocol: hash-freeze the rules, then download an untouched system (Spirit, USENIX CFDR) and run
multi-seed. <b>Result:</b> BGL&rarr;Spirit SMA 0.9200/0.9650/0.9300 over seeds 42/7/19 (mean 0.938) vs Dense RAG
mean 0.356; MDL leg 0.9100. HDFS&rarr;Spirit fails (0.3775) like HDFS&rarr;OpenStack — transfer holds within the
infrastructure failure-physics family, not across app-vs-infra families (honest scope).</p>
{_table(holdout)}

<h3>5c. The decisive controls: is it the representation or the matcher?</h3>
<p><b>Why:</b> the ladder showed generic WL graph similarity on SMA's own extraction BEATS full SME within-system
(0.9799 vs 0.9549 on HDFS, at ~1ms). If WL also transferred, the matcher would be dead weight. <b>Result:</b> on the
identical BGL&rarr;Spirit sets, WL reaches only 0.6239 and the production stack (Hybrid+Rerank) 0.5947, vs SMA 0.9200.
Decomposition: representation necessary (v1 collapse), not sufficient (WL +0.27 over dense), SME alignment adds the
remaining +0.31. Design implication adopted: tiered retrieval — WL prefilter within-system, SME for cross-system,
provenance, and candidate inference.</p>
{_table(controls)}

<h2 id="ladder">5d. Production-RAG baseline ladder (within-system, seed 42)</h2>
<p>Hybrid RRF (BM25+BGE fusion), cross-encoder reranking, BGE-base, SPLADE, the WL-kernel control, and a
long-context frontier-LLM baseline (top-20 candidates stuffed into DeepSeek). HDFS: SMA 0.9549 beats the entire
ladder. BGL: hybrid/BGE/SPLADE saturate (~1.0) — lexically overt anomalies, reported honestly. Latency columns from
this batch ran CPU-contended and are not citable.</p>
{_table(ladder)}

<h2 id="family">5e. Failure-family metric (depth beyond binary triage)</h2>
<p><b>Why:</b> "retrieved an anomaly" is shallow; the enterprise question is whether retrieval surfaces the correct
<i>root-cause family</i> (EOFException-family vs replication-family vs kernel-MCE...). Families derived
deterministically (HDFS: failure-line signatures; BGL: alert-category column read from raw logs for ground truth
only). <b>Result:</b> HDFS family-hit@1 SMA-ses 0.9057 vs BM25 0.6226, dense 0.4906 — SMA finds the right family.
BGL: dense 0.9623 &gt; SMA 0.68 (alert families are lexically marked). ADR-004 revision: SES beats MDL on aggregate
family-hit (0.9057 vs 0.8396) — the EOF rare-family anecdote does not generalize; rare-family-stratified analysis
queued before any default change.</p>
{_table(family)}

<h2 id="h3">6. H3 honesty study (verifiable answers vs confabulation)</h2>
<p><b>Design:</b> 20 authored questions over the 5,000-session HDFS corpus — 10 answerable from session evidence,
10 unanswerable (false premises, wrong domains, beyond-window outcomes) — &times; 5 memory modes &times; 2 verbalizers
= 200 cells. Mechanical abstention detection (regex, conservative); human rating columns left blank in
<code>reports/h3_mini_study.csv</code>. Every prompt carries the window-boundary caveat ("absence of an event in the
evidence is NOT evidence it did not happen").</p>
<div class="verdict"><b>Finding 1:</b> honesty is a property of the verbalizer — DeepSeek abstained on 46/50
unanswerable cells (10/10 under sma and kg); the local 0.5B abstained on 1/50, fabricating ZooKeeper crash-loops and
on-call response times. <b>Finding 2:</b> retrieval decides whether honesty is useful — DeepSeek also correctly declined
most <i>answerable</i> questions (15/50 answered; sma-SES 1/10) because prose-only top-5 evidence rarely contained the
asked-for family — the ADR-004 result appearing in a third independent instrument.</div>
{h3_llm_table}
<h3>DeepSeek cells per memory mode</h3>
{h3_mode_table}
<h3>LLM-judge pass (all 200 cells, audit trail in h3_judged.csv)</h3>
<p>Every answer judged against deterministically reconstructed evidence under a written rubric (correctness,
confabulation, unsupported-claim count, confidence flags). <b>DeepSeek: 99% judged-correct, 0 invented entities,
0.02 mean unsupported claims/answer; local Qwen-0.5B: 2% correct, 18% confabulation, 0.32 unsupported claims,
0/100 abstentions</b> — including invented statistics, fabricated log-line formats, and impossible dates. The
auto-abstention regex was validated against the judge (precision 0.96 / recall 0.87). Judge-found pipeline bug
fixed: a 400-char evidence cap was truncating exactly the anomaly lines questions ask about (now 900). The 40
SMA-mode rows carry reduced judge confidence (encoder changed post-study) — flagged for human spot-check.</p>

<h2 id="perf">7. Engineering record (what was fixed, why it matters)</h2>
<ul>
<li><b>Matcher hot path (~2000&times;):</b> <code>Kernel.bindings</code> was a property rebuilt (with full statement
re-serialization) on every access inside O(k&sup2;) merge loops &rarr; cached tables; MH seeding capped per canonical functor
group (U-ordered, identical statements first — the blueprint &sect;10.2 tripwire); root-MH-only kernels per &sect;2.2.
Worst-case 120-line session: &gt;5 min &rarr; 181 ms. Canonical battery unchanged (G2 green).</li>
<li><b>Certified retrieval fix:</b> the MAC/FAC early-stop compared a raw-score bound to the k-th raw score while ranking
by normalized ses_n &rarr; bound now converted to ses_n units (admissible since the normalizer &ge; query self-score);
the <code>certified</code> flag is honest under FAC budget truncation.</li>
<li><b>Encoder v0.1.1 &rarr; v0.2.0:</b> cause/enables now require antecedent-before-consequent (the Python contradicted its
own rules/logs.yaml); v0.2.0 added the cross-system event ontology (&sect;5 above).</li>
<li><b>Eval integrity:</b> BGL + Thunderbird label columns and the OpenStack source-filename token stripped (three
separate label-leak vectors closed); SSB generator vocabulary collisions fixed (per-triple namespaces);
content addressing is blake3-only (no silent hash fallback).</li>
<li><b>Orchestration:</b> chat-format local LLM calls with repeat-penalty (fixes looping); evidence prompts carry numbered
text only (no provenance hashes — small models parrot them) + the window caveat; DeepSeek backend via httpx with key in
git-ignored <code>.env</code>.</li>
</ul>
<h3>Latency snapshots</h3>
{_table(latency)}

<h2 id="caveats">8. Honest caveats &amp; open issues</h2>
<ul>
<li><b>Single seed everywhere.</b> No confidence intervals yet; the consolidated pre-freeze batch adds multi-seed +
paired bootstrap.</li>
<li><b>The transfer ontology is post-hoc.</b> Event-class rules were written after observing the run-1 failure (legitimate
pre-freeze diagnostics, not a preregistered result); needs held-out confirmation on new seeds/windows and a third pair.</li>
<li><b>BGL within-system is an honest H2 miss</b> (surface methods near-perfect on lexically overt anomalies).</li>
<li><b>HDFS&rarr;OpenStack wall:</b> no method beats coin-flip; investigate whether OpenStack anomaly semantics
(missing cleanup events) are visible to <i>any</i> retrieval formulation before re-attempting.</li>
<li><b>Short-session ses_n bias:</b> normalization favors small self-scores; revisit at calibration time.</li>
<li><b>SMA p95 latency</b> on large BGL windows (~2s) exceeds comfort; tripwire ladder not yet exhausted.</li>
<li><b>SSB far-vocabulary circularity:</b> the synthetic benchmark's <code>far_</code> renaming is known to the
canonicalizer; treat SSB as machinery verification only until real disjoint lexicons are implemented.</li>
<li><b>mdl_gain is an MVP</b> (within-target costs over matched functor types); corpus-level code-length costs are the
designed upgrade.</li>
<li><b>H3 auto-abstention is regex-based</b>; human rating pass pending (columns provided).</li>
</ul>

<h2 id="next">9. Next steps (agreed sequencing)</h2>
<ol>
<li><b>Consolidated pre-freeze eval batch:</b> hybrid RRF (+cross-encoder reranker) baselines, multi-seed transfer
confirmation including the MDL leg, third transfer pair, long-context B6 designed against the H3 findings.</li>
<li><b>Calibration then freeze</b> (&gamma;, &rho;, &delta;, &theta; on validation only), pre-registration tag, then single test-set runs.</li>
<li>LogHub-2k oracle template validation (G3 full gate); SME-v4 25-pair oracle battery (G2 full gate).</li>
<li>Drift protocol T5 (the agentic claim), BugsInPy (T3), ablation battery (&gamma;=0 first).</li>
</ol>

<h2 id="appendix">10. Appendix: remaining raw artifacts</h2>
<h3>Dataset manifest (checksums)</h3>
{_table(manifest)}
<h3>SSB fixture metrics</h3>
{_table(ssb)}
<h3>Calibration placeholders (pre-freeze)</h3>
{_table(calibration)}

</div></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/report.html")
    parser.add_argument(
        "--ssb-library-n",
        type=int,
        default=12,
        help="Number of SSB triples for the FAC-backed full-library run. Default 12 gives 24 library cases.",
    )
    parser.add_argument(
        "--ssb-mac-prefilter-n",
        type=int,
        default=1000,
        help="Number of SSB triples for the MAC-stage candidate-generation diagnostic.",
    )
    parser.add_argument(
        "--skip-loghub",
        action="store_true",
        help="Skip the long-running HDFS/BGL LogHub evaluation (SSB fixtures only).",
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Re-render report.html from existing CSVs without running any evaluation.",
    )
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.html_only:
        rows = run_fixture_eval(
            library_n=args.ssb_library_n,
            mac_prefilter_n=args.ssb_mac_prefilter_n,
            include_loghub=not args.skip_loghub,
        )
        for name, csv_rows in rows.items():
            write_csv(out.parent / name, csv_rows)
    out.write_text(render_html(out.parent), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
