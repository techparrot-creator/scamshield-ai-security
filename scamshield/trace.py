"""Safe, judge-friendly pipeline trace helpers.

The pipeline trace records only observable system-level events (stage status,
counts, labels, confidence numbers, source IDs). It deliberately NEVER contains
raw user messages, OTPs, CNICs, passwords, card numbers, phone numbers, full
emails, or any hidden model reasoning / chain-of-thought.
"""
from __future__ import annotations

STAGE_LABELS = {
    "input": "Input",
    "evidence_extraction": "Evidence extraction",
    "indicators": "Indicators (heuristic)",
    "url_analysis": "URL analysis",
    "retrieval": "Retrieval",
    "multi_query": "Multi-Query",
    "query_expansion": "Query expansion",
    "assessment": "Gemini assessment",
    "render": "Response rendering",
}

# Canonical presentation order for the step trace view.
STAGE_ORDER = [
    "evidence_extraction",
    "input",
    "indicators",
    "url_analysis",
    "retrieval",
    "multi_query",
    "query_expansion",
    "assessment",
    "render",
]
_ORDER_INDEX = {stage: index for index, stage in enumerate(STAGE_ORDER)}


def event(stage: str, **fields) -> dict:
    """Build one trace event. Values must stay non-sensitive by construction."""
    return {"stage": stage, **fields}


def _join_list(values) -> str:
    if not values:
        return "none"
    return ", ".join(str(v) for v in values)


def summarize_event(ev: dict) -> str:
    """Turn one trace event into a single human-readable observation line."""
    stage = ev.get("stage", "other")
    if stage == "input":
        return f"{ev.get('input_type', 'Text')} · {ev.get('text_chars', 0)} chars · {ev.get('urls_detected', 0)} URL(s) detected"
    if stage == "evidence_extraction":
        if ev.get("status") == "skipped":
            return f"Skipped ({ev.get('note', 'text-only submission')})"
        modality = ev.get("modality", "media")
        return f"Completed · modality: {modality} · extracted {ev.get('extracted_chars', 0)} chars"
    if stage == "indicators":
        labels = ev.get("labels") or ["none detected"]
        return _join_list(labels)
    if stage == "url_analysis":
        count = ev.get("url_count", 0)
        if not count:
            return "No URLs detected"
        warnings = _join_list(ev.get("structural_warnings") or ["none"])
        return f"{count} URL(s) · max heuristic score {ev.get('max_heuristic_score', 0)}/100 · warnings: {warnings}"
    if stage == "retrieval":
        threshold_met = "met" if ev.get("threshold_met") else "not met"
        sources = _join_list(item.get("id") if isinstance(item, dict) else item for item in ev.get("sources", []))
        return (
            f"{ev.get('phase', 'initial')} · semantic: {ev.get('semantic_search', 'n/a')} · BM25: {ev.get('bm25', 'n/a')} · "
            f"{ev.get('fused_results', 0)} fused result(s) · confidence {ev.get('confidence', 0)} "
            f"(threshold {ev.get('confidence_threshold', 0)} {threshold_met}) · "
            f"parent context {ev.get('parent_context', 'n/a')} · sources: {sources}"
        )
    if stage == "multi_query":
        if ev.get("activated"):
            return f"Activated ({ev.get('reason', 'weak first-pass retrieval')})"
        return "Not needed (first-pass retrieval confidence was sufficient)"
    if stage == "query_expansion":
        return ev.get("status", "n/a").capitalize()
    if stage == "assessment":
        status = str(ev.get("gemini_status", "n/a"))
        if ev.get("risk_level"):
            return f"Completed · final risk: {ev['risk_level']} · category: {ev.get('scam_type', 'n/a')}"
        return f"Status: {status}"
    if stage == "render":
        return ev.get("status", "completed").capitalize()
    return _join_list(f"{k}: {v}" for k, v in ev.items() if k != "stage")


def format_trace_rows(pipeline_trace: list[dict] | None) -> list[list[str]]:
    """Format trace events into [stage, observation] table rows for the UI."""
    rows: list[list[str]] = []
    for ev in pipeline_trace or []:
        if not isinstance(ev, dict):
            continue
        stage = STAGE_LABELS.get(ev.get("stage", ""), str(ev.get("stage", "Other")).title())
        rows.append([stage, summarize_event(ev)])
    return rows


def format_trace_steps(pipeline_trace: list[dict] | None) -> list[list[str]]:
    """Format trace events into [step, status, observation] rows in pipeline order."""
    events = [ev for ev in pipeline_trace or [] if isinstance(ev, dict)]
    events.sort(key=lambda ev: _ORDER_INDEX.get(str(ev.get("stage", "")), len(STAGE_ORDER)))
    rows: list[list[str]] = []
    for ev in events:
        stage = str(ev.get("stage", ""))
        label = STAGE_LABELS.get(stage, str(stage or "Other").title())
        status = (
            ev.get("status")
            or ev.get("gemini_status")
            or ("activated" if stage == "multi_query" and ev.get("activated") else "completed")
        )
        rows.append([label, str(status), summarize_event(ev)])
    return rows
