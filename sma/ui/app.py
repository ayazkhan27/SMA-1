"""Gradio comparison workbench for SMA-1.

Chat with an LLM (local Qwen GGUF or DeepSeek API) whose memory mode is
toggleable per turn, or compare all memory modes side by side. Extraction and
retrieval are deterministic; the LLM only verbalizes retrieved evidence.
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib

from sma.agent.adapter_draft import draft_rules
from sma.agent.comparison import MODES, ComparisonFramework, challenge_corpus, demo_corpus
from sma.agent.llm import DEFAULT_MODEL_FILE, DEFAULT_MODEL_REPO, DEEPSEEK_MODEL
from sma.encoders.draft_adapter import (
    DraftAdapter,
    check_determinism,
    rules_from_json,
    rules_hash,
    rules_to_json,
)
from sma.eval.arn import DEFAULT_ARN_PATH, arn_choice_corpus

UI_CORPORA = {
    "HDFS sample (5,000 labeled sessions)": pathlib.Path("data/processed/ui_corpus_hdfs.jsonl"),
    "BGL sample (2,500 labeled sessions)": pathlib.Path("data/processed/ui_corpus_bgl.jsonl"),
}

LLM_CHOICES = {
    "Local (Qwen2.5-0.5B, CPU)": "local",
    f"DeepSeek API ({DEEPSEEK_MODEL})": "deepseek",
}

MODE_ACCENTS = {
    "sma": "#2563eb",
    "bm25": "#b45309",
    "dense rag": "#7c3aed",
    "knowledge graph": "#047857",
    "hybrid (fused)": "#0e7490",
    "context only": "#64748b",
}

HARD_QUESTIONS = [
    "ERROR StreamIngest connector timeout polling source kafka-9\n"
    "WARN StreamIngest connector retrying poll\n"
    "WARN StreamIngest connector retrying poll\n"
    "ERROR StreamIngest sink write failed after repeated retry\n"
    "ERROR StreamIngest backpressure queue overflow failure",
    "INFO PaymentGateway deployment completed successfully\n"
    "INFO PaymentGateway timeout setting increased to 45s by operator\n"
    "INFO PaymentGateway connection pool resized",
    "ERROR BackupAgent snapshot timeout on host 10.0.0.99\n"
    "WARN BackupAgent retrying snapshot upload\n"
    "ERROR BackupAgent snapshot failed permanently",
    "WARN CdnOrigin fetch timeout for asset bundle\n"
    "WARN CdnOrigin retrying fetch\n"
    "WARN CdnOrigin retrying fetch\n"
    "WARN CdnOrigin retrying fetch\n"
    "ERROR CdnOrigin served stale asset after retry failure",
    "ERROR DeviceHub firmware download timeout for sensor fleet\n"
    "WARN DeviceHub retrying firmware push\n"
    "ERROR DeviceHub provisioning failed for batch\n"
    "ERROR DeviceHub heartbeat lost after failure",
]

CSS = """
.sma-shell {max-width: 1480px; margin: 0 auto;}
/* All custom surfaces hard-code light backgrounds, so every text color is
   pinned too — dark-mode themes must never bleed white text into them. */
.sma-cards {display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 16px;}
.sma-card {border: 1px solid #cbd5e1; border-radius: 12px; background: #ffffff; color: #111827;
           overflow: hidden; box-shadow: 0 2px 6px rgba(15, 23, 42, .10);
           display: flex; flex-direction: column; transition: box-shadow .15s ease;}
.sma-card:hover {box-shadow: 0 4px 14px rgba(15, 23, 42, .16);}
.sma-card * {color: inherit;}
.sma-card-head {padding: 9px 14px; color: #ffffff; font-weight: 700; font-size: 13px;
                text-transform: uppercase; letter-spacing: .06em; display: flex;
                justify-content: space-between; align-items: center;}
.sma-card-head .sma-llm-badge {color: rgba(255,255,255,.92); font-weight: 600; font-size: 10px;
                background: rgba(255,255,255,.18); border-radius: 99px; padding: 2px 8px;
                text-transform: none; letter-spacing: 0;}
.sma-card-body {padding: 13px 15px; font-size: 14px; line-height: 1.6; white-space: pre-wrap;
                color: #111827; flex: 1;}
.sma-evidence {border-top: 1px solid #e2e8f0; padding: 9px 15px; font-size: 12px; color: #111827;}
.sma-evidence summary {cursor: pointer; color: #334155; font-weight: 700;}
.sma-ev-item {margin: 8px 0; padding: 9px; background: #f1f5f9; border-radius: 8px; color: #1f2933;}
.sma-ev-meta {color: #475569; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
              font-size: 11px; word-break: break-all;}
.sma-ev-text {margin-top: 5px; white-space: pre-wrap; font-family: ui-monospace, Menlo, monospace;
              font-size: 11px; max-height: 130px; overflow-y: auto; color: #1f2933;}
.sma-inference {color: #1d4ed8; font-family: ui-monospace, Menlo, monospace; font-size: 11px;
                margin-top: 4px;}
.sma-ev-warning {margin: 8px 0; padding: 9px; background: #fff7ed; border: 1px solid #fed7aa;
                 border-radius: 8px; color: #9a3412; font-size: 12px; font-weight: 600;}
.sma-detail {color: #64748b; font-size: 11px; padding: 4px 15px 11px;}
.sma-empty {color: inherit; font-size: 14px;}
.sma-chips {display: flex; gap: 8px; flex-wrap: wrap; margin: 2px 0 6px;}
.sma-chip {background: #eef2ff; color: #3730a3; border: 1px solid #c7d2fe; border-radius: 99px;
           padding: 3px 12px; font-size: 12px; font-weight: 600;}
.sma-chip.ok {background: #ecfdf5; color: #065f46; border-color: #a7f3d0;}
.sma-chip.warn {background: #fff7ed; color: #9a3412; border-color: #fed7aa;}
.sma-table {width: 100%; border-collapse: collapse; background: #ffffff; color: #111827;
            border: 1px solid #cbd5e1; border-radius: 10px; overflow: hidden; font-size: 13px;}
.sma-table th {background: #f1f5f9; color: #334155; text-align: left; padding: 8px 10px;
               font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
               border-bottom: 1px solid #cbd5e1;}
.sma-table td {padding: 7px 10px; border-bottom: 1px solid #e2e8f0; color: #1f2933;
               vertical-align: top;}
.sma-table td.mono {font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: #475569;}
.sma-table tr:last-child td {border-bottom: none;}
.sma-panel {border: 1px solid #cbd5e1; border-radius: 12px; background: #ffffff; color: #111827;
            overflow: hidden;}
.sma-panel * {color: inherit;}
.sma-panel-head {padding: 8px 14px; color: #ffffff; font-weight: 700; font-size: 12px;
                 text-transform: uppercase; letter-spacing: .06em;}
.sma-panel-body {padding: 6px 12px 10px;}
.sma-label {border-radius: 99px; padding: 1px 8px; font-size: 10px; font-weight: 700;
            margin-left: 6px;}
.sma-label.bad {background: #fee2e2; color: #991b1b;}
.sma-label.good {background: #dcfce7; color: #166534;}
.sma-vote {font-size: 12px; color: #334155; margin: 2px 0 8px;}
"""


def coverage_chip(evidence: list[dict]) -> str:
    """Structural-coverage chip (blueprint 12-R3): amber below threshold, green otherwise."""
    coverage = next((row.get("coverage") for row in evidence if row.get("coverage")), None)
    if not coverage:
        return ""
    cls = "warn" if coverage.get("low") else "ok"
    return (
        f'<span class="sma-chip {cls}">structural coverage: '
        f'{coverage.get("percent", 0)}%</span>'
    )


def evidence_items_html(evidence: list[dict]) -> str:
    items = []
    for row in evidence:
        if row.get("warning"):
            items.append(
                '<div class="sma-ev-warning">&#9888; '
                f'{html.escape(row["warning"])}<br>'
                f'<span class="sma-ev-meta">{html.escape(row.get("provenance", ""))}</span></div>'
            )
            continue
        inferences = "".join(
            f'<div class="sma-inference">&#8627; {html.escape(s)}</div>'
            for s in row.get("inferences", [])
        )
        label = row.get("label") or ""
        label_chip = (
            f'<span class="sma-label {"bad" if label == "Anomaly" else "good"}">{html.escape(label)}</span>'
            if label else ""
        )
        alignment = (
            f'<br><b>{html.escape(row["alignment"])}</b>' if row.get("alignment") else ""
        )
        items.append(
            '<div class="sma-ev-item">'
            f'<div class="sma-ev-meta">{html.escape(row.get("source_id", ""))}{label_chip} · '
            f'score={html.escape(str(row.get("score", "")))}<br>'
            f'{html.escape(row.get("provenance", ""))}{alignment}</div>'
            f'<div class="sma-ev-text">{html.escape(row.get("text", ""))}</div>'
            f"{inferences}</div>"
        )
    return "".join(items)


def render_cards(results: dict, llm_label: str) -> str:
    cards = []
    for mode, result in results.items():
        accent = MODE_ACCENTS.get(mode, "#334155")
        detail = next(
            (row["mode_detail"] for row in result.evidence
             if row.get("mode_detail") and not row.get("warning")),
            "no evidence retrieved",
        )
        chip = coverage_chip(result.evidence)
        chip_html = f'<div class="sma-chips">{chip}</div>' if chip else ""
        cards.append(
            '<div class="sma-card">'
            f'<div class="sma-card-head" style="background:{accent}">'
            f"<span>{html.escape(mode)}</span>"
            f'<span class="sma-llm-badge">{html.escape(llm_label)}</span></div>'
            f'<div class="sma-card-body">{html.escape(result.answer)}</div>'
            f'<div class="sma-detail">{chip_html}{html.escape(detail)}{label_vote_line(result.evidence)}</div>'
            '<details class="sma-evidence"><summary>'
            f"Evidence ({len(result.evidence)})</summary>{evidence_items_html(result.evidence)}</details>"
            "</div>"
        )
    return f'<div class="sma-cards">{"".join(cards)}</div>'


def label_vote_line(evidence: list[dict]) -> str:
    labels = [row.get("label") for row in evidence if row.get("label")]
    if not labels:
        return ""
    anomalies = sum(1 for label in labels if label == "Anomaly")
    normals = len(labels) - anomalies
    verdict = "Anomaly" if anomalies > normals else "Normal"
    return (
        f'<div class="sma-vote">retrieved labels: {anomalies} Anomaly / {normals} Normal '
        f"&rarr; vote: <b>{verdict}</b></div>"
    )


def render_evidence_panel(mode: str, evidence: list[dict]) -> str:
    accent = MODE_ACCENTS.get(mode, "#334155")
    chip = coverage_chip(evidence)
    chip_html = f'<div class="sma-chips">{chip}</div>' if chip else ""
    n_items = sum(1 for row in evidence if not row.get("warning"))
    return (
        '<div class="sma-panel">'
        f'<div class="sma-panel-head" style="background:{accent}">'
        f"evidence · {html.escape(mode)} · {n_items} item(s)</div>"
        f'<div class="sma-panel-body">{chip_html}{label_vote_line(evidence)}'
        f'{evidence_items_html(evidence) or "<p class=sma-empty>none retrieved</p>"}</div>'
        "</div>"
    )


def render_corpus_table(framework: ComparisonFramework) -> str:
    if not framework.items:
        return '<p class="sma-empty">Corpus is empty — paste raw text above and click Load.</p>'
    rows = []
    shown = framework.items[:200]
    for item in shown:
        rows.append(
            "<tr>"
            f'<td class="mono">{html.escape(item.item_id)}</td>'
            f"<td>{html.escape(item.adapter_id)}</td>"
            f"<td>{len(item.case.statements)}</td>"
            f'<td class="mono">{html.escape(item.case.case_id[:12])}</td>'
            f"<td>{html.escape(item.text[:150])}</td>"
            "</tr>"
        )
    note = (
        f'<p class="sma-empty">Showing first {len(shown)} of {len(framework.items)} items.</p>'
        if len(framework.items) > len(shown) else ""
    )
    return (
        '<table class="sma-table"><thead><tr>'
        "<th>item</th><th>adapter</th><th>statements</th><th>case id</th><th>text</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>" + note
    )


def render_chips(framework: ComparisonFramework) -> str:
    local = framework.orchestrators["local"].status
    deepseek = framework.orchestrators["deepseek"].status
    local_cls, local_text = (
        ("ok", "local LLM ready") if local.get("loaded") or local.get("backend") == "llama_cpp"
        else ("warn", "local LLM missing")
    )
    ds_cls, ds_text = (
        ("ok", "DeepSeek key present") if deepseek.get("key_present")
        else ("warn", "DeepSeek key missing")
    )
    draft_chip = (
        f'<span class="sma-chip warn">{html.escape(framework.draft_note)}</span>'
        if framework.draft_note else ""
    )
    return (
        '<div class="sma-chips">'
        f'<span class="sma-chip">{len(framework.items)} corpus items</span>'
        f'<span class="sma-chip {local_cls}">{local_text}</span>'
        f'<span class="sma-chip {ds_cls}">{ds_text}</span>'
        f"{draft_chip}"
        "</div>"
    )


def build_demo(framework: ComparisonFramework | None = None):
    framework = framework or ComparisonFramework()
    if not framework.items:
        # Pre-load the challenge corpus so chat and compare work immediately.
        framework.load_lines(challenge_corpus(), adapter_id="logs")
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install gradio to run the UI: pip install gradio") from exc

    adapters = ["logs", "code", "traces", "structured", "agentobs", "prose_tier1"]

    def chat_send(message, history, mode, llm_label, adapter_id, k, scorer):
        message = (message or "").strip()
        history = history or []
        if not message:
            return history, "", gr.skip()
        if not framework.items:
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "The corpus is empty — load one on the Corpus tab first."},
            ]
            return history, "", gr.skip()
        llm = LLM_CHOICES.get(llm_label, "local")
        framework.set_scorer(scorer)
        resolved_mode, evidence = framework.evidence_for(message, mode, adapter_id=adapter_id, k=int(k))
        orchestrator = framework.orchestrators[llm]
        answer = orchestrator.answer(message, resolved_mode, evidence, history=history)
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return history, "", render_evidence_panel(resolved_mode, evidence)

    def chat_clear():
        return [], "", '<p class="sma-empty">Evidence for the latest turn appears here.</p>'

    def run_comparison(question, adapter_id, k, selected_modes, llm_label, scorer):
        if not framework.items:
            return '<p class="sma-empty">Load a corpus first (Corpus tab).</p>'
        if not selected_modes:
            return '<p class="sma-empty">Select at least one memory mode.</p>'
        llm = LLM_CHOICES.get(llm_label, "local")
        framework.set_scorer(scorer)
        results = framework.ask_all(
            question, adapter_id=adapter_id, k=int(k), modes=selected_modes, llm=llm
        )
        return render_cards(results, llm_label)

    def load_corpus(corpus_text, adapter_id, max_items, clear_existing):
        if clear_existing:
            framework.clear()
        added = framework.load_lines(corpus_text, adapter_id=adapter_id, max_items=int(max_items))
        status = {"added": len(added), "total": len(framework.items), "adapter": adapter_id}
        return render_corpus_table(framework), json.dumps(status, indent=2), render_chips(framework)

    def fill_challenge():
        return challenge_corpus(), "logs"

    def fill_demo():
        return demo_corpus(), "logs"

    def load_loghub(corpus_name, n_items, clear_existing, progress=gr.Progress()):
        path = UI_CORPORA.get(corpus_name)
        if path is None or not path.exists():
            status = {
                "error": f"{corpus_name} not prepared",
                "fix": "python3 -u scripts/prepare_ui_corpus.py",
            }
            return render_corpus_table(framework), json.dumps(status, indent=2), render_chips(framework)
        if clear_existing:
            framework.clear()
        n_items = int(n_items)
        added = 0
        with path.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        # The JSONL stores all anomalies first; shuffle deterministically so a
        # partial load keeps the stratified label balance.
        import random

        random.Random(7).shuffle(rows)
        rows = rows[:n_items]
        for row in progress.tqdm(rows, desc=f"Encoding {corpus_name}"):
            framework.add_document(row["text"], adapter_id="logs", label=row.get("label", ""))
            added += 1
        labels = [item.label for item in framework.items if item.label]
        status = {
            "added": added,
            "total": len(framework.items),
            "anomaly": sum(1 for label in labels if label == "Anomaly"),
            "normal": sum(1 for label in labels if label == "Normal"),
            "note": "First dense-RAG query embeds the whole corpus once (~1-2 min at 5k); later queries are fast.",
        }
        return render_corpus_table(framework), json.dumps(status, indent=2), render_chips(framework)

    def load_arn_sample(max_items, clear_existing):
        if not DEFAULT_ARN_PATH.exists():
            status = {
                "error": "ARN CSV is not downloaded",
                "expected_path": str(DEFAULT_ARN_PATH),
                "fetch": "python3 scripts/fetch_datasets.py --manifest data/manifests/datasets.json --only arn",
            }
            return (
                gr.skip(),
                gr.skip(),
                render_corpus_table(framework),
                json.dumps(status, indent=2),
                render_chips(framework),
            )
        corpus_text, suggested_query = arn_choice_corpus(limit=int(max_items))
        if clear_existing:
            framework.clear()
        added = framework.load_lines(corpus_text, adapter_id="prose_tier1", max_items=int(max_items) * 2)
        status = {
            "added": len(added),
            "total": len(framework.items),
            "adapter": "prose_tier1",
            "suggested_query": suggested_query,
            "tier_note": "ARN uses flagged Tier-1 prose extraction; not part of headline Tier-0 claims.",
        }
        return (
            corpus_text,
            "prose_tier1",
            render_corpus_table(framework),
            json.dumps(status, indent=2),
            render_chips(framework),
        )

    def draft_adapter_from_corpus(llm_label):
        if not framework.items:
            return "", "", json.dumps({"error": "corpus is empty - load one first"}, indent=2)
        llm = LLM_CHOICES.get(llm_label, "deepseek")
        rules, note = draft_rules([item.text for item in framework.items], llm=llm)
        if not rules.classes:
            return "", "", json.dumps({"error": note, "backend": llm}, indent=2)
        status = {
            "note": note,
            "backend": llm,
            "discipline": "LLM proposed RULES (data); encoding stays deterministic. Review before trusting.",
        }
        return rules_to_json(rules), rules_hash(rules), json.dumps(status, indent=2)

    def apply_draft_adapter(rules_json):
        try:
            rules = rules_from_json(rules_json or "")
            adapter = DraftAdapter(rules)
            probe = framework.items[0].text if framework.items else "probe timeout error line"
            check_determinism(adapter, probe)
            count = framework.apply_draft_adapter(adapter)
        except Exception as exc:
            status = {"error": f"{type(exc).__name__}: {exc}"}
            return (
                render_corpus_table(framework),
                gr.skip(),
                json.dumps(status, indent=2),
                render_chips(framework),
            )
        status = {
            "applied": framework.draft_note,
            "reencoded_items": count,
            "draft_hash": adapter.draft_hash,
            "case_metadata": {"adapter": "draft", "draft_hash": adapter.draft_hash},
        }
        return (
            render_corpus_table(framework),
            adapter.draft_hash,
            json.dumps(status, indent=2),
            render_chips(framework),
        )

    def revert_draft_adapter():
        count = framework.revert_draft_adapter()
        status = {"reverted_to": "base adapters", "reencoded_items": count}
        return render_corpus_table(framework), json.dumps(status, indent=2), render_chips(framework)

    def backend_status():
        return json.dumps(
            {name: orch.status for name, orch in framework.orchestrators.items()}, indent=2
        )

    with gr.Blocks(title="SMA-1 Agentic Memory Workbench") as demo:
        with gr.Column(elem_classes=["sma-shell"]):
            gr.Markdown(
                "# SMA-1 Agentic Memory Workbench\n"
                "Chat with toggleable memory, or compare all memory modes side by side. "
                "Extraction is deterministic (Tier-0 adapters); the LLM — local or DeepSeek — "
                "only verbalizes retrieved evidence."
            )
            chips = gr.HTML(render_chips(framework))
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=430, label="Conversation")
                        msg = gr.Textbox(
                            lines=3,
                            label="Message (paste log lines as a new incident, or ask a follow-up)",
                            placeholder="Describe or paste an incident… then ask follow-ups.",
                        )
                        with gr.Row():
                            send = gr.Button("Send", variant="primary")
                            clear_chat = gr.Button("Clear chat")
                        gr.Examples(
                            examples=[[q] for q in HARD_QUESTIONS],
                            inputs=[msg],
                            label="Challenge incidents (paired with the pre-loaded challenge corpus)",
                        )
                    with gr.Column(scale=2):
                        chat_mode = gr.Radio(list(MODES), value="sma", label="Memory mode")
                        chat_llm = gr.Radio(
                            list(LLM_CHOICES), value=list(LLM_CHOICES)[0], label="Answer model"
                        )
                        chat_scorer = gr.Radio(
                            ["surprisal", "ses", "mdl"], value="surprisal",
                            label="SMA scorer (ses: systematicity; mdl: surprisal-weighted, finds rare failure families)",
                        )
                        with gr.Row():
                            chat_adapter = gr.Dropdown(adapters, value="logs", label="Query adapter")
                            chat_k = gr.Number(value=5, precision=0, label="Evidence k")
                        evidence_panel = gr.HTML(
                            '<p class="sma-empty">Evidence for the latest turn appears here.</p>'
                        )
                chat_inputs = [msg, chatbot, chat_mode, chat_llm, chat_adapter, chat_k, chat_scorer]
                send.click(chat_send, chat_inputs, [chatbot, msg, evidence_panel])
                msg.submit(chat_send, chat_inputs, [chatbot, msg, evidence_panel])
                clear_chat.click(chat_clear, None, [chatbot, msg, evidence_panel])
            with gr.Tab("Compare"):
                with gr.Row():
                    question = gr.Textbox(
                        value=HARD_QUESTIONS[2],
                        lines=4,
                        scale=4,
                        label="Question / new incident",
                    )
                    query_adapter = gr.Dropdown(adapters, value="logs", label="Query adapter", scale=1)
                    top_k = gr.Number(value=5, precision=0, label="Evidence k", scale=1)
                with gr.Row():
                    modes = gr.CheckboxGroup(
                        choices=list(MODES),
                        value=list(MODES),
                        label="Memory modes (toggle to compare)",
                        scale=3,
                    )
                    llm_choice = gr.Radio(
                        choices=list(LLM_CHOICES),
                        value=list(LLM_CHOICES)[0],
                        label="Answer model",
                        scale=2,
                    )
                    compare_scorer = gr.Radio(
                        ["surprisal", "ses", "mdl"], value="surprisal", label="SMA scorer", scale=1
                    )
                run = gr.Button("Run comparison", variant="primary")
                cards = gr.HTML()
                run.click(run_comparison, [question, query_adapter, top_k, modes, llm_choice, compare_scorer], cards)
                gr.Examples(
                    examples=[[q] for q in HARD_QUESTIONS],
                    inputs=[question],
                    label="Challenge incidents",
                )
            with gr.Tab("Corpus"):
                with gr.Row():
                    adapter = gr.Dropdown(adapters, value="logs", label="Deterministic adapter")
                    max_items = gr.Number(value=50, precision=0, label="Max items")
                    clear = gr.Checkbox(value=True, label="Clear existing corpus")
                with gr.Row():
                    loghub_choice = gr.Dropdown(
                        list(UI_CORPORA), value=list(UI_CORPORA)[0], label="Real LogHub sample", scale=2
                    )
                    loghub_n = gr.Number(value=5000, precision=0, label="Sessions to index", scale=1)
                    load_loghub_btn = gr.Button("Load LogHub sample", variant="primary", scale=1)
                corpus = gr.Textbox(value=challenge_corpus(), lines=10, label="Raw corpus (manual)")
                with gr.Row():
                    load = gr.Button("Load pasted corpus")
                    use_challenge = gr.Button("Fill challenge corpus")
                    use_demo = gr.Button("Fill demo corpus")
                    load_arn = gr.Button("Load downloaded ARN sample")
                corpus_table = gr.HTML(render_corpus_table(framework))
                load_status = gr.Code(language="json", label="Load status")
                load.click(load_corpus, [corpus, adapter, max_items, clear], [corpus_table, load_status, chips])
                load_loghub_btn.click(
                    load_loghub, [loghub_choice, loghub_n, clear], [corpus_table, load_status, chips]
                )
                use_challenge.click(fill_challenge, None, [corpus, adapter])
                use_demo.click(fill_demo, None, [corpus, adapter])
                load_arn.click(
                    load_arn_sample,
                    [max_items, clear],
                    [corpus, adapter, corpus_table, load_status, chips],
                )
                gr.Markdown(
                    "### Draft adapter (LLM proposes rules; encoding stays deterministic)\n"
                    "The LLM drafts extra keyword class rules as data for the frozen logs "
                    "encoder - it never writes facts. Drafts are content-addressed (blake3) "
                    "and remain flagged *LLM-proposed, unreviewed* until reverted or promoted."
                )
                with gr.Row():
                    draft_llm = gr.Radio(
                        list(LLM_CHOICES), value=list(LLM_CHOICES)[1], label="Drafting model", scale=2
                    )
                    draft_btn = gr.Button("Draft adapter from corpus (LLM)", scale=1)
                    apply_draft_btn = gr.Button("Apply draft adapter", variant="primary", scale=1)
                    revert_draft_btn = gr.Button("Revert to base adapter", scale=1)
                draft_json = gr.Textbox(
                    lines=12,
                    label="Proposed rules (editable JSON: classes + maskings)",
                    placeholder='{"classes": [{"name": "...Event", "keywords": ["..."]}], "maskings": []}',
                )
                draft_hash_box = gr.Textbox(label="Draft blake3 hash", interactive=False)
                draft_status = gr.Code(language="json", label="Draft status")
                draft_btn.click(
                    draft_adapter_from_corpus, [draft_llm], [draft_json, draft_hash_box, draft_status]
                )
                apply_draft_btn.click(
                    apply_draft_adapter,
                    [draft_json],
                    [corpus_table, draft_hash_box, draft_status, chips],
                )
                revert_draft_btn.click(
                    revert_draft_adapter, None, [corpus_table, draft_status, chips]
                )
            with gr.Tab("System"):
                gr.Markdown(
                    f"**Local model:** `{DEFAULT_MODEL_REPO}/{DEFAULT_MODEL_FILE}` "
                    "(fetch with `python3 scripts/fetch_model.py`).\n\n"
                    f"**API model:** `{DEEPSEEK_MODEL}` via DeepSeek — set `SMA_DEEPSEEK_API_KEY` "
                    "in the environment or the repo `.env` file.\n\n"
                    "Extraction and retrieval never use either model. Low-level "
                    "`encode/retrieve/map/project/verify` tools: `make api`."
                )
                status = gr.Code(value=backend_status(), language="json", label="LLM backend status")
                refresh = gr.Button("Refresh status")
                refresh.click(backend_status, None, status)
    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    demo = build_demo()
    import gradio as gr

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        theme=gr.themes.Soft(),
        css=CSS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
