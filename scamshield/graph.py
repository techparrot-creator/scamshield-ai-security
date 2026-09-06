from __future__ import annotations

import json
import os
import time
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from scamshield.config import settings
from scamshield.llm_client import GeminiConfigurationError, invoke_with_retry
from scamshield.privacy import redact_sensitive_text
from scamshield.prompts import ANALYSIS_TEMPLATE, QUERY_EXPANSION_PROMPT, SYSTEM_PROMPT
from scamshield.query_planner import (
    classify_secret_flow,
    detect_indicator_labels,
    infer_retrieval_hints,
    should_expand_query,
)
from scamshield.retrieval import retrieve_guidance
from scamshield.schemas import QueryExpansion, ScamAssessment, ScamShieldState
from scamshield.trace import event
from scamshield.url_tools import analyze_url, extract_urls

load_dotenv()

AI_STATUS_MESSAGES = {
    "temporarily_unavailable": (
        "AI assessment is temporarily unavailable. Extracted evidence and verified "
        "safety sources are shown below. Please retry shortly."
    ),
    "configuration_error": (
        "AI assessment is unavailable because the model is not configured correctly. "
        "Extracted evidence and verified safety sources are shown below."
    ),
}


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _describe_input(raw_input: str, input_kind: str) -> str:
    has_text = "SUBMITTED MESSAGE OR LINK:" in raw_input
    has_media = "EXTRACTED FROM" in raw_input
    media_name = {"image": "Screenshot", "audio": "Voice note"}.get(input_kind, "Media")
    if has_text and has_media:
        return f"Text + {media_name}"
    if has_media:
        return media_name
    return "Text"


def _retrieval_event(phase: str, results: list[dict], confidence: float, start: float) -> dict:
    if results:
        semantic = "completed" if any("chroma" in str(item.get("retrieval_backend", "")) for item in results) else "bm25 fallback only"
    else:
        semantic = "no results"
    restored = sum(1 for item in results if item.get("context"))
    return event(
        "retrieval",
        phase=phase,
        semantic_search=semantic,
        bm25="completed",
        fused_results=len(results),
        confidence=round(float(confidence), 3),
        confidence_threshold=settings.rag_expand_threshold,
        threshold_met=bool(confidence >= settings.rag_expand_threshold),
        sources=[{"id": item["id"], "title": item["title"]} for item in results],
        parent_context=f"restored for {restored}/{len(results)} source(s)",
        elapsed_ms=_ms(start),
    )


def _get_model() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY in .env before running ScamShield AI.")
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0,
        api_key=api_key,
        vertexai=False,
    )


def normalize_input(state: ScamShieldState) -> dict:
    start = time.perf_counter()
    raw_input = (state.get("raw_input") or "").strip()
    if not raw_input:
        raise ValueError("No evidence was provided.")
    urls = extract_urls(raw_input)
    labels = detect_indicator_labels(raw_input)
    input_kind = state.get("input_kind", "text")
    trace = [
        event(
            "input",
            input_type=_describe_input(raw_input, input_kind),
            text_chars=len(raw_input),
            urls_detected=len(urls),
            elapsed_ms=_ms(start),
        ),
        event("indicators", labels=labels or ["none detected"], elapsed_ms=_ms(start)),
    ]
    return {
        "case_id": state.get("case_id") or f"SCAM-{uuid.uuid4().hex[:8].upper()}",
        "normalized_text": raw_input,
        "detected_urls": urls,
        "preferred_language": state.get("preferred_language", "English"),
        "input_kind": input_kind,
        "retrieval_hints": infer_retrieval_hints(raw_input),
        "secret_flow_signal": classify_secret_flow(raw_input),
        "pipeline_trace": trace,
    }


def inspect_urls(state: ScamShieldState) -> dict:
    start = time.perf_counter()
    findings = [analyze_url(url) for url in state.get("detected_urls", [])]
    warnings = sorted({label for finding in findings for label in finding.get("labels", [])})
    trace = [
        event(
            "url_analysis",
            url_count=len(findings),
            structural_warnings=warnings or ["none"],
            max_heuristic_score=max((finding["heuristic_score"] for finding in findings), default=0),
            elapsed_ms=_ms(start),
        )
    ]
    return {"url_findings": findings, "pipeline_trace": trace}


def initial_retrieval(state: ScamShieldState) -> dict:
    start = time.perf_counter()
    query = state["normalized_text"]
    if state.get("url_findings"):
        query += "\nCyber safety context: suspicious URL phishing account verification."
    results, confidence, strategy = retrieve_guidance(query, hints=state.get("retrieval_hints"))
    return {
        "retrieved_guidance": results,
        "retrieval_confidence": confidence,
        "retrieval_strategy": strategy,
        "pipeline_trace": [_retrieval_event("initial", results, confidence, start)],
    }


def route_retrieval(state: ScamShieldState) -> str:
    weak = should_expand_query(
        float(state.get("retrieval_confidence", 0.0)),
        len(state.get("retrieved_guidance", [])),
        settings.rag_expand_threshold,
    )
    return "expand_queries" if weak else "analyze_case"


def expand_queries(state: ScamShieldState) -> dict:
    start = time.perf_counter()
    trace = [event("multi_query", activated=True, reason="weak first-pass retrieval confidence")]
    fallback_topics = state.get("retrieval_hints", {}).get("topics", [])
    fallback = [
        f"{state['normalized_text']} cyber scam warning signs",
        f"{' '.join(fallback_topics) or 'online fraud'} prevention reporting recovery",
        "Pakistan cybercrime reporting phishing fraud recovery",
    ]
    try:
        expander = _get_model().with_structured_output(QueryExpansion)
        expanded = invoke_with_retry(
            lambda: expander.invoke(
                [
                    SystemMessage(content=QUERY_EXPANSION_PROMPT),
                    HumanMessage(content=state["normalized_text"]),
                ]
            ),
            description="Query expansion",
        )
        queries = [q.strip() for q in expanded.queries if q.strip()]
        trace.append(event("query_expansion", status="completed", variants=len(queries[:3]), elapsed_ms=_ms(start)))
    except Exception as exc:
        print(f"[ScamShield RAG] query expansion fallback: {type(exc).__name__}")
        queries = fallback
        trace.append(event("query_expansion", status="fallback used", variants=len(queries[:3]), elapsed_ms=_ms(start)))
    return {"query_variants": queries[:3], "pipeline_trace": trace}


def expanded_retrieval(state: ScamShieldState) -> dict:
    start = time.perf_counter()
    results, confidence, strategy = retrieve_guidance(
        state["normalized_text"],
        hints=state.get("retrieval_hints"),
        query_variants=state.get("query_variants", []),
    )
    return {
        "retrieved_guidance": results,
        "retrieval_confidence": confidence,
        "retrieval_strategy": strategy,
        "pipeline_trace": [_retrieval_event("expanded (multi-query)", results, confidence, start)],
    }


def analyze_case(state: ScamShieldState) -> dict:
    start = time.perf_counter()
    trace: list[dict] = []
    if not state.get("query_variants"):
        trace.append(event("multi_query", activated=False, reason="first-pass retrieval confidence was sufficient"))
    try:
        model = _get_model().with_structured_output(ScamAssessment)
    except Exception as exc:
        print(f"[ScamShield Gemini] model configuration error: {type(exc).__name__}: {exc}")
        trace.append(event("assessment", gemini_status="configuration_error", elapsed_ms=_ms(start)))
        return {"assessment": None, "ai_status": "configuration_error", "pipeline_trace": trace}

    guidance_text = "\n\n".join(
        f"SOURCE: {item['title']} ({item['organization']})\n"
        f"URL: {item['source']}\n"
        f"MATCHED SECTION: {item['section_title']}\n"
        f"CONTEXT: {item['context']}"
        for item in state.get("retrieved_guidance", [])
    ) or "No sufficiently relevant verified guidance was retrieved."
    prompt = ANALYSIS_TEMPLATE.format(
        language=state.get("preferred_language", "English"),
        input_kind=state.get("input_kind", "text"),
        evidence=state["normalized_text"],
        url_findings=json.dumps(state.get("url_findings", []), ensure_ascii=False, indent=2),
        secret_flow=state.get("secret_flow_signal", "none"),
        guidance=guidance_text,
        retrieval_strategy=", ".join(state.get("retrieval_strategy", [])),
    )
    try:
        assessment = invoke_with_retry(
            lambda: model.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]),
            description="Scam assessment",
        )
        payload = assessment.model_dump()
        trace.append(
            event(
                "assessment",
                gemini_status="completed",
                risk_level=payload["risk_level"],
                scam_type=payload["scam_type"],
                elapsed_ms=_ms(start),
            )
        )
        return {"assessment": payload, "ai_status": "completed", "pipeline_trace": trace}
    except GeminiConfigurationError as exc:
        print(f"[ScamShield Gemini] assessment unavailable: {exc}")
        trace.append(event("assessment", gemini_status="configuration_error", elapsed_ms=_ms(start)))
        return {"assessment": None, "ai_status": "configuration_error", "pipeline_trace": trace}
    except Exception as exc:
        # Graceful degradation: keep evidence extraction, URL findings, and RAG sources.
        print(f"[ScamShield Gemini] assessment unavailable: {type(exc).__name__}")
        trace.append(event("assessment", gemini_status="temporarily_unavailable", elapsed_ms=_ms(start)))
        return {"assessment": None, "ai_status": "temporarily_unavailable", "pipeline_trace": trace}


def _bullets(items: list[str], empty: str = "None") -> str:
    return "\n".join(f"- {item}" for item in items) if items else f"- {empty}"


def _sources_markdown(state: ScamShieldState) -> str:
    return _bullets(
        [
            f"[{item['title']}]({item['source']}) — {item['section_title']} · score {item['retrieval_score']:.3f} · {item['retrieval_backend']}"
            for item in state.get("retrieved_guidance", [])
        ],
        "No sufficiently relevant verified source was retrieved.",
    )


def _url_markdown(state: ScamShieldState) -> str:
    if not state.get("url_findings"):
        return ""
    return "\n## URL checks\n" + _bullets(
        [
            f"`{item['url']}` — heuristic {item['heuristic_score']}/100. {'; '.join(item.get('signals', []))}"
            for item in state["url_findings"]
        ]
    ) + "\n"


def _render_degraded(state: ScamShieldState) -> dict:
    """Render evidence, URL findings, and verified sources when Gemini is unavailable.

    No risk classification is fabricated: the model never produced one.
    """
    status = state.get("ai_status", "temporarily_unavailable")
    message = AI_STATUS_MESSAGES.get(status, AI_STATUS_MESSAGES["temporarily_unavailable"])
    url_md = _url_markdown(state)
    sources_md = _sources_markdown(state)
    response = f"""# ScamShield Assessment

**Case ID:** `{state.get('case_id', 'unknown')}`  
**AI status:** {status}

> ⚠️ {message}
{url_md}
## Verified guidance already retrieved
{sources_md}

## Safe immediate steps
- Do not reply, click links, or share OTPs, passwords, PINs, card details, or CNIC images.
- Retry this analysis shortly; if urgent, contact your bank or an official helpline you already trust.

> ScamShield provides risk guidance, not a legal finding or a guarantee of safety.
"""
    report = f"""# ScamShield Case Report

- Case ID: {state.get('case_id', 'unknown')}
- Input type: {state.get('input_kind', 'text')}
- AI status: {status}
- Risk level: not produced (model unavailable)
- RAG confidence: {float(state.get('retrieval_confidence', 0.0)):.3f}

## Notice
{message}

## Verified sources
{sources_md}

ScamShield is decision support and does not make a legal determination.
"""
    return {"response_markdown": response, "report_markdown": report}


def render_response(state: ScamShieldState) -> dict:
    start = time.perf_counter()
    assessment = state.get("assessment")
    if not assessment:
        rendered = _render_degraded(state)
        rendered["pipeline_trace"] = [event("render", status="degraded (model unavailable)", elapsed_ms=_ms(start))]
        return rendered
    evidence_md = _bullets(
        [f"**{item['indicator']} ({item['severity']})** — {item['evidence']}" for item in assessment.get("evidence", [])],
        "No strong indicator was identified.",
    )
    benign_md = _bullets(
        assessment.get("benign_indicators", []),
        "No benign indicator was recorded.",
    )
    timeline_md = _bullets(
        [f"**{item['step']}. {item['event']}** — {item['significance']}" for item in assessment.get("incident_timeline", [])],
        "No reliable timeline could be constructed from the evidence.",
    )
    sources_md = _sources_markdown(state)
    url_md = _url_markdown(state)

    response = f"""# ScamShield Assessment

**Case ID:** `{state['case_id']}`  
**Risk:** **{assessment['risk_level'].upper()} — {assessment['risk_score']}/100**  
**Confidence:** {assessment['confidence']}%  
**Likely type:** {assessment['scam_type']}  
**Sender authenticity:** {assessment.get('sender_authenticity', 'unverifiable')}  
**RAG confidence:** {float(state.get('retrieval_confidence', 0.0)):.2f}

## Summary
{assessment['summary']}

## Evidence and red flags
{evidence_md}
{url_md}
## Benign signals
{benign_md}

## Incident timeline
{timeline_md}

## Do this now
{_bullets(assessment.get('immediate_actions', []))}

## Recovery steps
{_bullets(assessment.get('recovery_actions', []), 'No recovery step was generated.')}

## Safe verification questions
{_bullets(assessment.get('questions_to_verify', []), 'No additional verification question was generated.')}

## Reporting steps
{_bullets(assessment.get('reporting_steps', []), 'Preserve evidence and use an official reporting channel if needed.')}

## Safe reply
{assessment.get('safe_reply') or 'Do not reply. Verify independently through an official channel you already trust.'}

## Uncertainty
{assessment['uncertainty_note']}

## Verified guidance used
{sources_md}

> ScamShield provides risk guidance, not a legal finding or a guarantee of safety. Never share OTPs, passwords, PINs, card details, recovery codes, CNIC images, seed phrases, or private keys.
"""

    redacted_evidence = redact_sensitive_text(state["normalized_text"])
    report = f"""# ScamShield Case Report

- Case ID: {state['case_id']}
- Input type: {state.get('input_kind', 'text')}
- Preferred language: {state.get('preferred_language', 'English')}
- Risk level: {assessment['risk_level']}
- Risk score: {assessment['risk_score']}/100
- Confidence: {assessment['confidence']}%
- Sender authenticity: {assessment.get('sender_authenticity', 'unverifiable')}
- RAG confidence: {float(state.get('retrieval_confidence', 0.0)):.3f}
- Likely scam type: {assessment['scam_type']}

## Submitted evidence (privacy-redacted export)
{redacted_evidence}

## Assessment
{assessment['summary']}

## Indicators
{evidence_md}

## Benign signals
{benign_md}

## Timeline
{timeline_md}

## Immediate actions
{_bullets(assessment.get('immediate_actions', []))}

## Recovery actions
{_bullets(assessment.get('recovery_actions', []))}

## Reporting
{_bullets(assessment.get('reporting_steps', []))}

## Verified sources
{sources_md}

## Retrieval strategy
{_bullets(state.get('retrieval_strategy', []))}

ScamShield is decision support and does not make a legal determination.
"""
    return {
        "response_markdown": response,
        "report_markdown": report,
        "pipeline_trace": [event("render", status="completed", elapsed_ms=_ms(start))],
    }


builder = StateGraph(ScamShieldState)
builder.add_node("normalize_input", normalize_input)
builder.add_node("inspect_urls", inspect_urls)
builder.add_node("initial_retrieval", initial_retrieval)
builder.add_node("expand_queries", expand_queries)
builder.add_node("expanded_retrieval", expanded_retrieval)
builder.add_node("analyze_case", analyze_case)
builder.add_node("render_response", render_response)

builder.add_edge(START, "normalize_input")
builder.add_edge("normalize_input", "inspect_urls")
builder.add_edge("inspect_urls", "initial_retrieval")
builder.add_conditional_edges(
    "initial_retrieval",
    route_retrieval,
    {"expand_queries": "expand_queries", "analyze_case": "analyze_case"},
)
builder.add_edge("expand_queries", "expanded_retrieval")
builder.add_edge("expanded_retrieval", "analyze_case")
builder.add_edge("analyze_case", "render_response")
builder.add_edge("render_response", END)

graph = builder.compile()
