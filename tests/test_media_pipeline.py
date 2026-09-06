"""Tests for the multimodal evidence pipeline (mocked; no live Gemini calls)."""
from __future__ import annotations

import types

import pytest

from scamshield import media


def test_detect_modality():
    assert media.detect_modality("screenshot.png") == "image"
    assert media.detect_modality("photo.JPG") == "image"
    assert media.detect_modality("voice.wav") == "audio"
    assert media.detect_modality("note.mp3") == "audio"
    assert media.detect_modality("document.pdf") is None


def test_unsupported_file_type_is_graceful(tmp_path):
    path = tmp_path / "document.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    summary = media.run_media_pipeline(path)
    assert summary["modality"] is None
    assert summary["extraction_succeeded"] is False
    assert summary["error"] == "unsupported file type"


def test_missing_file_is_graceful(tmp_path):
    summary = media.run_media_pipeline(tmp_path / "missing.png")
    assert summary["extraction_succeeded"] is False
    assert summary["error"] == "file could not be read"


def test_extraction_failure_is_graceful(tmp_path, monkeypatch):
    path = tmp_path / "voice.wav"
    path.write_bytes(b"not real audio data")

    def boom(*args, **kwargs):
        raise RuntimeError("corrupt file")

    monkeypatch.setattr(media, "extract_media_evidence", boom)
    summary = media.run_media_pipeline(path)
    assert summary["modality"] == "audio"
    assert summary["extraction_succeeded"] is False
    assert summary["error"] == "RuntimeError"


def test_successful_media_run_feeds_the_same_pipeline(tmp_path, monkeypatch):
    path = tmp_path / "scam-shot.png"
    path.write_bytes(b"\x89PNG fake bytes")
    monkeypatch.setattr(media, "extract_media_evidence", lambda *a, **k: "Send me the OTP code now!")

    fake_graph = types.SimpleNamespace(
        invoke=lambda payload: {
            "case_id": "SCAM-MEDIA01",
            "assessment": {"risk_level": "high", "scam_type": "OTP phishing"},
            "ai_status": "completed",
            "retrieval_confidence": 0.72,
            "retrieved_guidance": [{"id": "pkcert-phishing::intro", "title": "Phishing awareness"}],
            "pipeline_trace": [
                {"stage": "input", "input_type": "Screenshot"},
                {"stage": "multi_query", "activated": False},
            ],
        }
    )
    import scamshield.graph as graph_module

    monkeypatch.setattr(graph_module, "graph", fake_graph)

    summary = media.run_media_pipeline(path)
    assert summary["modality"] == "image"
    assert summary["extraction_succeeded"] is True
    assert summary["extracted_chars"] == len("Send me the OTP code now!")
    assert summary["retrieval_executed"] is True
    assert summary["retrieval_confidence"] == pytest.approx(0.72)
    assert summary["multi_query_activated"] is False
    assert summary["source_ids"] == ["pkcert-phishing::intro"]
    assert summary["assessment_available"] is True
    # Raw media contents must never be part of the technical summary.
    assert "PNG fake bytes" not in str(summary)


def test_media_run_reports_unavailable_assessment(tmp_path, monkeypatch):
    path = tmp_path / "note.m4a"
    path.write_bytes(b"fake audio")
    monkeypatch.setattr(media, "extract_media_evidence", lambda *a, **k: "extracted words")

    fake_graph = types.SimpleNamespace(
        invoke=lambda payload: {
            "case_id": "SCAM-MEDIA02",
            "assessment": None,
            "ai_status": "temporarily_unavailable",
            "retrieval_confidence": 0.44,
            "retrieved_guidance": [],
            "pipeline_trace": [],
        }
    )
    import scamshield.graph as graph_module

    monkeypatch.setattr(graph_module, "graph", fake_graph)

    summary = media.run_media_pipeline(path)
    assert summary["extraction_succeeded"] is True
    assert summary["retrieval_executed"] is True
    assert summary["assessment_available"] is False
    assert summary["ai_status"] == "temporarily_unavailable"
