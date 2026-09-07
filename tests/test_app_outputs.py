"""Adapter contract tests for the Gradio callbacks in app.py.

These guard the runtime data plumbing only: that every callback branch returns
exactly the registered number of outputs, in order, as plain values (never
deprecated ``gr.update()`` prop-updates), and that the dynamic sections carry
the real values the pipeline produced.

No network, no Gemini, no retrieval index is touched — the graph is stubbed.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import app


STUB_GUIDANCE = [
    {
        "id": "pkcert-whatsapp-2026::otp",
        "parent_id": "pkcert-whatsapp-2026",
        "title": "WhatsApp Account Hijacking Campaign advisory",
        "organization": "PKCERT / National CERT Pakistan",
        "country": "Pakistan",
        "primary_topic": "whatsapp",
        "section_title": "Verification-code theft",
        "source": "https://pkcert.example.invalid/advisory/26/1.pdf",
        "text": "Never share a verification code.",
        "context": "Never share a verification code.",
        "retrieval_score": 0.694,
        "retrieval_backend": "chroma+hf+bm25+mmr+rrf",
        "reviewed_at": "2026-01-15",
    },
    {
        "id": "pkcert-online-transactions::after-suspicious-payment",
        "parent_id": "pkcert-online-transactions",
        "title": "Secure online transactions",
        "organization": "PKCERT / National CERT Pakistan",
        "country": "Pakistan",
        "primary_topic": "payments",
        "section_title": "Account recovery",
        "source": "https://pkcert.example.invalid/advisory/9/1.pdf",
        "text": "Contact your bank immediately.",
        "context": "Contact your bank immediately.",
        "retrieval_score": 0.512,
        "retrieval_backend": "chroma+hf+bm25+mmr+rrf",
        "reviewed_at": "2026-02-02",
    },
]

STUB_TRACE = [
    {"stage": "evidence_extraction", "status": "skipped", "note": "text-only submission"},
    {"stage": "input", "input_type": "Text", "text_chars": 363, "urls_detected": 0},
    {"stage": "indicators", "labels": ["credential_request", "urgency"]},
    {"stage": "url_analysis", "url_count": 0, "structural_warnings": ["none"], "max_heuristic_score": 0},
    {
        "stage": "retrieval",
        "phase": "initial",
        "semantic_search": "completed",
        "bm25": "completed",
        "fused_results": 2,
        "confidence": 0.694,
        "confidence_threshold": 0.43,
        "threshold_met": True,
        "sources": [{"id": item["id"], "title": item["title"]} for item in STUB_GUIDANCE],
        "parent_context": "restored for 2/2 source(s)",
    },
    {"stage": "multi_query", "activated": False, "reason": "first-pass retrieval confidence was sufficient"},
    {"stage": "assessment", "gemini_status": "completed", "risk_level": "high", "scam_type": "OTP phishing"},
    {"stage": "render", "status": "completed"},
]

STUB_RESULT = {
    "case_id": "SCAM-TEST01",
    "ai_status": "completed",
    "input_kind": "text",
    "assessment": {
        "risk_score": 88,
        "risk_level": "high",
        "confidence": 91,
        "scam_type": "OTP phishing / bank impersonation",
        "summary": "Impersonates a bank and demands a one-time code under time pressure.",
        "evidence": [{"indicator": "OTP request", "evidence": "asks for a code", "severity": "high"}],
        "benign_indicators": [],
        "sender_authenticity": "suspicious",
        "incident_timeline": [],
        "immediate_actions": ["Do not share the code."],
        "recovery_actions": [],
        "questions_to_verify": ["Call the number printed on your card."],
        "reporting_steps": ["Report the sender to the platform."],
        "safe_reply": "",
        "uncertainty_note": "Sender identity cannot be confirmed from the message alone.",
    },
    "response_markdown": "# ScamShield Assessment\n\n**Case ID:** `SCAM-TEST01`\n\n## Summary\nStub summary.\n",
    "report_markdown": "# ScamShield Case Report\n\n- Case ID: SCAM-TEST01\n",
    "detected_urls": [],
    "url_findings": [],
    "retrieval_hints": {"topics": ["whatsapp"], "country": "PK"},
    "query_variants": [],
    "retrieval_confidence": 0.694,
    "retrieval_strategy": ["BM25 keyword search", "ChromaDB semantic search"],
    "retrieved_guidance": STUB_GUIDANCE,
    "pipeline_trace": STUB_TRACE,
}

TEXT_ARGS = (
    "Main bank security team se hun. Apna OTP foran bhejein warna account block ho jayega.",
    "", None, None, "Roman Urdu", "", False, False, False, False,
)


class _StubGraph:
    """Stands in for the compiled LangGraph so no model or index is touched."""

    def __init__(self, result: dict):
        self._result = result
        self.calls: list[dict] = []

    def invoke(self, payload, *args, **kwargs):
        self.calls.append(payload)
        return json.loads(json.dumps(self._result))


def _block_fn(name: str):
    for fn in app.demo.fns.values():
        if getattr(fn, "name", None) == name:
            return fn
    raise AssertionError(f"no registered Gradio dependency named {name!r}")


def _is_prop_update(value) -> bool:
    return isinstance(value, dict) and "update" in str(value.get("__type__", ""))


@pytest.fixture()
def stubbed(monkeypatch, tmp_path):
    """Stub the graph and point the report directory at an isolated temp path."""
    graph_stub = _StubGraph(STUB_RESULT)
    monkeypatch.setattr(app, "graph", graph_stub)
    report_dir = tmp_path / "nested" / "scamshield-reports"
    monkeypatch.setattr(app, "REPORT_DIR", report_dir)
    return graph_stub, report_dir


def test_analyze_returns_exactly_the_registered_outputs(stubbed):
    values = app.analyze_submission(*TEXT_ARGS)
    outputs = _block_fn("analyze_submission").outputs
    assert len(values) == len(outputs) == app.OUTPUT_COUNT
    # No branch may rely on the deprecated gr.update() prop-update path.
    assert not any(_is_prop_update(value) for value in values)


def test_clear_form_returns_inputs_plus_outputs(stubbed):
    values = app.clear_form()
    clear_outputs = _block_fn("clear_form").outputs
    assert len(values) == len(clear_outputs) == app.INPUT_COUNT + app.OUTPUT_COUNT
    assert not any(_is_prop_update(value) for value in values)


def test_dynamic_sections_carry_real_runtime_values(stubbed):
    verdict, signals, actions_left, actions_right, sources, source_rows, detail, trace_rows, report_path = (
        app.analyze_submission(*TEXT_ARGS)
    )

    # assessment / verdict is non-empty and reflects the returned risk level.
    assert "HIGH" in verdict and "88/100" in verdict
    assert signals and actions_left and actions_right and sources

    # Detailed source table: real retrieved rows, real metadata, correct types.
    assert len(source_rows) == len(STUB_GUIDANCE) >= 1
    for row in source_rows:
        assert len(row) == 6
        organization, title, section, region, score, url = row
        assert isinstance(organization, str) and organization
        assert isinstance(title, str) and title
        assert isinstance(section, str) and section
        assert isinstance(region, str) and region
        assert isinstance(score, float)
        assert isinstance(url, str) and url
    assert source_rows[0][0] == "PKCERT / National CERT Pakistan"
    assert source_rows[0][2] == "Verification-code theft"

    # Full assessment detail is the markdown the graph actually rendered.
    assert "SCAM-TEST01" in detail and "Stub summary." in detail

    # Pipeline trace survived the graph state and the adapter.
    assert len(trace_rows) == len(STUB_TRACE) >= 1
    flat = " | ".join(cell for row in trace_rows for cell in row)
    for expected in (
        "Evidence extraction", "Text", "credential_request", "semantic: completed",
        "BM25: completed", "0.694", "0.43", "pkcert-whatsapp-2026::otp",
        "restored for 2/2", "final risk: high",
    ):
        assert expected in flat
    for row in trace_rows:
        assert len(row) == 3 and all(isinstance(cell, str) for cell in row)

    # Case report export points at a file that really exists.
    assert isinstance(report_path, str)
    assert Path(report_path).is_file()
    assert "SCAM-TEST01-report.md" in Path(report_path).name


def test_report_directory_is_created_on_demand(stubbed, monkeypatch):
    _, report_dir = stubbed
    assert not report_dir.exists()
    app.analyze_submission(*TEXT_ARGS)
    assert report_dir.is_dir()
    written = list(report_dir.glob("*-report.md"))
    assert len(written) == 1
    assert written[0].read_text(encoding="utf-8").startswith("# ScamShield Case Report")


def test_report_path_is_portable_and_filename_safe(stubbed, monkeypatch):
    monkeypatch.setattr(app, "REPORT_DIR", Path(app.tempfile.gettempdir()) / "scamshield-adapter-test")
    path = app._save_report("../../etc/p@sswörd", "# report")
    name = Path(path).name
    assert "/" not in name and "\\" not in name and ".." not in name
    assert Path(path).is_file()
    assert not Path(path).is_absolute() or Path(path).exists()


def test_degraded_branch_keeps_the_same_output_arity(stubbed, monkeypatch):
    degraded = dict(STUB_RESULT)
    degraded["assessment"] = None
    degraded["ai_status"] = "temporarily_unavailable"
    degraded["response_markdown"] = "# ScamShield Assessment\n\nAI status: temporarily_unavailable\n"
    degraded["report_markdown"] = "# ScamShield Case Report\n\n- Risk level: not produced\n"
    monkeypatch.setattr(app, "graph", _StubGraph(degraded))

    values = app.analyze_submission(*TEXT_ARGS)
    assert len(values) == app.OUTPUT_COUNT
    assert not any(_is_prop_update(value) for value in values)
    verdict, _signals, _left, _right, _sources, source_rows, detail, trace_rows, report_path = values
    assert "AI assessment unavailable" in verdict
    assert len(source_rows) == len(STUB_GUIDANCE)
    assert "temporarily_unavailable" in detail
    assert trace_rows and Path(report_path).is_file()


def test_developer_debug_is_not_rendered_but_is_logged(stubbed, caplog):
    labels = json.dumps(app.demo.config, default=str)
    assert "Developer debug" not in labels
    assert "Structured analysis + retrieval trace" not in labels

    with caplog.at_level("INFO", logger="scamshield.app"):
        app.analyze_submission(*TEXT_ARGS)

    records = [r.getMessage() for r in caplog.records if r.name == "scamshield.app"]
    assert records, "expected a server-side runtime diagnostic log line"
    summary = json.loads(records[-1].split("runtime ", 1)[1])
    assert set(summary) == {
        "ai_status", "input_modality", "retrieval_executed", "source_count",
        "retrieval_confidence", "multi_query_activated", "trace_event_count",
        "risk_band", "report_generated",
    }
    assert summary["ai_status"] == "completed"
    assert summary["input_modality"] == "text"
    assert summary["retrieval_executed"] is True
    assert summary["source_count"] == 2
    assert summary["retrieval_confidence"] == 0.694
    assert summary["multi_query_activated"] is False
    assert summary["trace_event_count"] == len(STUB_TRACE)
    assert summary["risk_band"] == "high"
    assert summary["report_generated"] is True

    # The diagnostic must stay non-sensitive.
    blob = json.dumps(summary).lower()
    for forbidden in ("api_key", "google_api_key", "secret", "password", "otp", "analysistemplate"):
        assert forbidden not in blob


def test_technical_trace_accordion_stays_collapsed_by_default():
    accordions = [
        component for component in app.demo.config["components"]
        if component.get("type") == "accordion"
    ]
    by_label = {component["props"].get("label"): component["props"] for component in accordions}
    trace_label = "🔬 Technical analysis trace (for judges & reviewers)"
    assert trace_label in by_label
    assert by_label[trace_label]["open"] is False


def test_dynamic_output_components_are_never_hidden():
    """The root cause of the blank sections: components created with visible=False
    needed a gr.update() prop-update to appear, and that path dropped their value."""
    outputs = _block_fn("analyze_submission").outputs
    for component in outputs:
        assert component.visible is not False, f"{type(component).__name__} starts hidden"


def test_demo_examples_populate_the_link_email_column():
    assert len(app.DEMO_EXAMPLES) == 4
    for example in app.DEMO_EXAMPLES:
        assert len(example) == app.INPUT_COUNT

    email_examples = [example for example in app.DEMO_EXAMPLES if example[1]]
    assert len(email_examples) == 1, "exactly one example should fill the link/email column"
    email = email_examples[0][1]
    assert "https://secure-bank-alert.example/verify-account" in email
    assert "no-reply@secure-bank-alert.example" in email
    # Reserved .example TLD only — no real host may ever appear in a demo.
    assert ".example" in email
    for forbidden in (".com", ".org", ".net", ".pk", ".io", "http://"):
        assert forbidden not in email


def test_postprocessed_payload_is_populated_and_update_free(stubbed):
    """Assert on the exact bytes Gradio sends to the browser, without a browser."""
    try:
        from gradio.state_holder import SessionState
    except ImportError:  # pragma: no cover - gradio internal moved
        pytest.skip("gradio SessionState API unavailable")

    block_fn = _block_fn("analyze_submission")
    values = list(app.analyze_submission(*TEXT_ARGS))
    payload = asyncio.run(app.demo.postprocess_data(block_fn, values, SessionState(app.demo)))

    assert len(payload) == app.OUTPUT_COUNT
    assert not any(_is_prop_update(item) for item in payload)

    source_table = block_fn.outputs[5].postprocess(values[5])
    assert source_table.data == values[5]
    assert source_table.headers == [
        "Organization", "Source", "Matched section", "Region", "Score", "URL",
    ]

    trace_table = block_fn.outputs[7].postprocess(values[7])
    assert len(trace_table.data) == len(STUB_TRACE)
    assert trace_table.headers == ["Pipeline step", "Status", "Observation"]

    # gr.Markdown strips trailing whitespace; compare on content.
    assert payload[6].strip() == values[6].strip()
    assert "SCAM-TEST01" in payload[6]
    assert payload[8] is not None
