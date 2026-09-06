"""Tests for the safe pipeline trace, indicator labels, and graceful degradation."""
from __future__ import annotations

from scamshield.graph import render_response
from scamshield.query_planner import detect_indicator_labels
from scamshield.trace import format_trace_rows, summarize_event
from scamshield.url_tools import analyze_url


def test_indicator_labels_detect_otp_and_urgency():
    labels = detect_indicator_labels("Main bank security se hun. OTP foran bhejo warna account block ho jayega.")
    assert "otp_request" in labels
    assert "urgency" in labels


def test_indicator_labels_benign_service_code():
    labels = detect_indicator_labels("Your verification code is 123456. Do not share it.")
    assert "reassuring:do_not_share_code" in labels
    assert "otp_request" not in labels


def test_indicator_labels_payment_request():
    labels = detect_indicator_labels("Deposit a registration fee of PKR 3,000 right now to activate your job account.")
    assert "payment_request" in labels
    assert "urgency" in labels


def test_url_labels_are_compact():
    finding = analyze_url("https://bit.ly/secure-login")
    assert "link_shortener" in finding["labels"]
    assert "suspicious_terms" in finding["labels"]


def test_trace_rows_are_judge_friendly():
    trace = [
        {"stage": "input", "input_type": "Text + Screenshot", "text_chars": 412, "urls_detected": 1},
        {"stage": "evidence_extraction", "status": "completed", "modality": "screenshot", "extracted_chars": 210},
        {"stage": "indicators", "labels": ["otp_request", "urgency"]},
        {"stage": "url_analysis", "url_count": 1, "structural_warnings": ["link_shortener"], "max_heuristic_score": 34},
        {
            "stage": "retrieval",
            "phase": "initial",
            "semantic_search": "completed",
            "bm25": "completed",
            "fused_results": 4,
            "confidence": 0.81,
            "confidence_threshold": 0.43,
            "threshold_met": True,
            "sources": [{"id": "pkcert-phishing::intro", "title": "Phishing awareness"}],
            "parent_context": "restored for 4/4 source(s)",
        },
        {"stage": "multi_query", "activated": False},
        {"stage": "assessment", "gemini_status": "completed", "risk_level": "high", "scam_type": "OTP phishing"},
        {"stage": "render", "status": "completed"},
    ]
    rows = format_trace_rows(trace)
    assert [row[0] for row in rows] == [
        "Input",
        "Evidence extraction",
        "Indicators (heuristic)",
        "URL analysis",
        "Retrieval",
        "Multi-Query",
        "Gemini assessment",
        "Response rendering",
    ]
    flat = " | ".join(row[1] for row in rows)
    assert "Text + Screenshot" in flat
    assert "0.81" in flat
    assert "Not needed" in flat
    assert "pkcert-phishing::intro" in flat
    assert "final risk: high" in flat


def test_trace_summaries_never_include_raw_user_content():
    # Trace events are constructed from counts/labels only; the formatter must
    # not reintroduce raw user text fields.
    trace = [
        {
            "stage": "retrieval",
            "phase": "initial",
            "semantic_search": "completed",
            "bm25": "completed",
            "fused_results": 2,
            "confidence": 0.5,
            "confidence_threshold": 0.43,
            "threshold_met": True,
            "sources": [{"id": "pkcert-whatsapp-2026::s1", "title": "WhatsApp scams"}],
            "parent_context": "restored for 2/2 source(s)",
        }
    ]
    text = str(format_trace_rows(trace))
    for forbidden in ("otp", "123456", "cnic", "password"):
        assert forbidden not in text.lower()
    assert "pkcert-whatsapp-2026::s1" in text


def test_degraded_render_when_assessment_missing():
    state = {
        "case_id": "SCAM-TEST01",
        "assessment": None,
        "ai_status": "temporarily_unavailable",
        "input_kind": "text",
        "preferred_language": "English",
        "normalized_text": "OTP 123456 send karo warna account block",
        "detected_urls": [],
        "url_findings": [],
        "retrieved_guidance": [],
        "retrieval_confidence": 0.31,
        "retrieval_strategy": ["BM25 keyword search"],
    }
    rendered = render_response(state)
    response_md = rendered["response_markdown"]
    report_md = rendered["report_markdown"]
    assert "temporarily unavailable" in response_md
    assert "temporarily unavailable" in report_md
    # No risk classification may be fabricated when the model never produced one.
    assert "not produced" in report_md
    for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        assert f"**Risk:** **{level}" not in response_md
    trace = rendered["pipeline_trace"][0]
    assert trace["stage"] == "render"
    assert "degraded" in trace["status"]


def test_multi_query_not_needed_event_summary():
    text = summarize_event({"stage": "multi_query", "activated": False})
    assert "Not needed" in text
