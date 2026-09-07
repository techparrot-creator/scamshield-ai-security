from __future__ import annotations

import json
import logging
import os
import tempfile
from html import escape
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from scamshield.graph import AI_STATUS_MESSAGES, graph
from scamshield.llm_client import GeminiServiceError
from scamshield.media import extract_media_evidence
from scamshield.retrieval import rag_health
from scamshield.trace import format_trace_steps

load_dotenv()

logger = logging.getLogger("scamshield.app")
if not logger.handlers:
    logging.basicConfig(level=os.getenv("SCAMSHIELD_LOG_LEVEL", "INFO").upper(), format="%(levelname)s %(name)s %(message)s")

APP_TITLE = "ScamShield AI"
DISCLAIMER = (
    "Decision support only. Do not submit passwords, OTPs, PINs, full card numbers, seed phrases, recovery codes, or private keys."
)

# Cross-platform runtime report directory. Never a hard-coded absolute path, and
# created on demand so a fresh Linux / Hugging Face Space container works as-is.
REPORT_DIR = Path(
    os.getenv("SCAMSHIELD_REPORT_DIR") or Path(tempfile.gettempdir()) / "scamshield-reports"
)

# Callback arity contract. `analyze_submission` returns OUTPUT_COUNT values and
# `clear_form` returns INPUT_COUNT + OUTPUT_COUNT values, in build_demo's order.
INPUT_COUNT = 10
OUTPUT_COUNT = 9

# Verdict wording never guarantees safety or criminality — it expresses likelihood only.
VERDICT_BY_LEVEL = {
    "low": (
        "Likely Legitimate", "v-ok", "✅",
        "No clear request for secrets or payment was found. This is not a guarantee of authenticity — verify the sender through an official channel.",
    ),
    "medium": (
        "Needs Verification", "v-caution", "🔍",
        "Signals are ambiguous. Do not share codes, credentials, or money until the sender is verified independently.",
    ),
    "high": (
        "Likely Scam", "v-danger", "🚨",
        "Strong scam indicators were found. Do not reply, pay, or share any code, password, or PIN.",
    ),
    "critical": (
        "Likely Scam", "v-danger", "🚨",
        "Severe scam indicators — possibly with active loss. Follow the containment steps immediately.",
    ),
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _file_path(uploaded) -> str | None:
    if uploaded is None:
        return None
    if isinstance(uploaded, str):
        return uploaded
    return getattr(uploaded, "name", None)


def _safe_case_id(case_id: str) -> str:
    """Keep the generated case id filesystem-safe on Linux, Windows and macOS."""
    cleaned = "".join(ch for ch in (case_id or "") if ch.isalnum() or ch in "-_")
    return cleaned or "case"


def _save_report(case_id: str, markdown: str) -> str:
    """Write the privacy-redacted report and return a path that really exists."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{_safe_case_id(case_id)}-report.md"
    path.write_text(markdown or "", encoding="utf-8")
    if not path.is_file():
        raise gr.Error("The case report could not be written on the server. Please retry.")
    return str(path)


def _runtime_summary(
    *,
    ai_status: str,
    input_kind: str,
    pipeline_trace: list[dict],
    source_count: int,
    retrieval_confidence: float,
    risk_band: str,
    report_generated: bool,
) -> dict:
    """Compact developer diagnostic — server-side only, never sent to the browser.

    Holds counts, statuses and labels only: no API keys, no environment
    variables, no raw prompts and no hidden model reasoning.
    """
    retrieval_executed = any(ev.get("stage") == "retrieval" for ev in pipeline_trace if isinstance(ev, dict))
    multi_query_activated = any(
        ev.get("stage") == "multi_query" and bool(ev.get("activated"))
        for ev in pipeline_trace
        if isinstance(ev, dict)
    )
    return {
        "ai_status": ai_status,
        "input_modality": input_kind,
        "retrieval_executed": retrieval_executed,
        "source_count": source_count,
        "retrieval_confidence": round(float(retrieval_confidence), 4),
        "multi_query_activated": multi_query_activated,
        "trace_event_count": len(pipeline_trace),
        "risk_band": risk_band,
        "report_generated": report_generated,
    }


def _bullets(items: list[str], empty: str = "None provided.") -> str:
    return "\n".join(f"- {item}" for item in items) if items else f"- {empty}"


# ---------------------------------------------------------------------------
# Result rendering (HTML cards built only from values the pipeline returned)
# ---------------------------------------------------------------------------

def _empty_state_html() -> str:
    return (
        "<div class='ss-empty'>"
        "<div class='empty-icon'>🛡️</div>"
        "<div class='empty-title'>No case analyzed yet.</div>"
        "<div class='empty-sub'>Submit suspicious evidence above to begin.</div>"
        "</div>"
    )


def _verdict_html(assessment: dict | None, ai_status: str, retrieval_confidence: float, case_id: str) -> str:
    if not assessment:
        message = AI_STATUS_MESSAGES.get(ai_status, AI_STATUS_MESSAGES["temporarily_unavailable"])
        return (
            "<div class='ss-verdict v-caution'>"
            "<div class='v-top'>"
            "<div class='v-label'>⚠️ AI assessment unavailable</div>"
            f"<div class='v-meta'>Case <code>{escape(case_id)}</code> · No risk level was fabricated</div>"
            "</div>"
            f"<div class='v-note'>{escape(message)}</div>"
            "<div class='v-note'>Extracted evidence, URL checks, and verified safety sources are still shown below.</div>"
            "</div>"
        )
    level = assessment.get("risk_level", "medium")
    label, css_class, icon, note = VERDICT_BY_LEVEL.get(level, VERDICT_BY_LEVEL["medium"])
    score = assessment.get("risk_score", 0)
    return (
        f"<div class='ss-verdict {css_class}'>"
        "<div class='v-top'>"
        f"<div class='v-label'>{icon} {escape(label)}</div>"
        f"<div class='v-meta'>Risk level: <strong>{escape(level.upper())}</strong> · Risk score: <strong>{score}/100</strong></div>"
        "</div>"
        "<div class='v-grid'>"
        f"<div><span class='v-k'>Category</span><span class='v-v'>{escape(assessment.get('scam_type', 'Not classified'))}</span></div>"
        f"<div><span class='v-k'>Confidence</span><span class='v-v'>{assessment.get('confidence', 0)}%</span></div>"
        f"<div><span class='v-k'>RAG confidence</span><span class='v-v'>{float(retrieval_confidence):.2f}</span></div>"
        f"<div><span class='v-k'>Sender authenticity</span><span class='v-v'>{escape(str(assessment.get('sender_authenticity', 'unverifiable')))}</span></div>"
        "</div>"
        f"<div class='v-note'>{escape(note)}</div>"
        "</div>"
    )


def _signals_html(assessment: dict | None) -> str:
    if not assessment:
        return ""
    suspicious = assessment.get("evidence") or []
    benign = assessment.get("benign_indicators") or []
    warn_chips = "".join(
        f"<span class='chip chip-warn' title=\"{escape(item.get('evidence', ''), quote=True)}\">"
        f"⚠️ {escape(item.get('indicator', 'indicator'))} · {escape(str(item.get('severity', '')))}</span>"
        for item in suspicious
    ) or "<span class='chip chip-none'>No suspicious indicator was flagged.</span>"
    ok_chips = "".join(
        f"<span class='chip chip-ok'>✓ {escape(str(item))}</span>" for item in benign
    ) or "<span class='chip chip-none'>No reassuring indicator was recorded.</span>"
    return (
        "<div class='ss-signals'>"
        "<div class='sig-group'><div class='sig-title sig-title-warn'>⚠️ Suspicious indicators</div>"
        f"<div class='ss-chips'>{warn_chips}</div></div>"
        "<div class='sig-group'><div class='sig-title sig-title-ok'>✓ Reassuring indicators</div>"
        f"<div class='ss-chips'>{ok_chips}</div></div>"
        "</div>"
    )


def _sources_html(items: list[dict]) -> str:
    if not items:
        return "<div class='ss-empty small'>No sufficiently relevant verified source was retrieved for this case.</div>"
    cards = []
    for item in items:
        link = item.get("source", "")
        link_html = (
            f"<a class='src-link' href='{escape(link, quote=True)}' target='_blank' rel='noopener noreferrer'>Open source ↗</a>"
            if link else ""
        )
        cards.append(
            "<div class='ss-source-card'>"
            f"<div class='src-org'>🏛️ {escape(item.get('organization', 'Verified source'))} · {escape(str(item.get('country', '')))}</div>"
            f"<div class='src-title'>{escape(item.get('title', ''))}</div>"
            f"<div class='src-section'>Matched section: {escape(item.get('section_title', ''))}</div>"
            f"<div class='src-score'>Retrieval score {float(item.get('retrieval_score', 0.0)):.3f} · {escape(str(item.get('retrieval_backend', '')))}</div>"
            f"{link_html}"
            "</div>"
        )
    return f"<div class='ss-sources'>{''.join(cards)}</div>"


def _actions_markdown(
    assessment: dict | None,
    clicked_link: bool,
    sent_money: bool,
    shared_secret: bool,
    installed_remote_access: bool,
) -> tuple[str, str]:
    if not assessment:
        left = (
            "### 🚨 Do this now\n"
            "- Do not reply, click links, or share OTPs, passwords, PINs, card details, or CNIC images.\n"
            "- Retry this analysis shortly.\n"
            "- If urgent, contact your bank or an official helpline you already trust."
        )
        right = (
            "### ℹ️ Assessment status\n"
            "- The AI model did not produce a verdict, so no risk level is shown.\n"
            "- Verified safety sources below still apply."
        )
        return left, right
    exposure = clicked_link or sent_money or shared_secret or installed_remote_access
    recovery = assessment.get("recovery_actions") or []
    if exposure:
        recovery_header = "### 🧯 You marked exposure — containment and recovery first"
    else:
        recovery_header = "### 🧯 If you already paid or shared credentials"
    left = f"### 🚨 Do this now\n{_bullets(assessment.get('immediate_actions', []))}\n\n"
    left += f"{recovery_header}\n{_bullets(recovery, 'No exposure was marked. Keep it that way — do not share secrets or send money.')}"
    right = (
        f"### 📢 Reporting\n{_bullets(assessment.get('reporting_steps', []), 'Preserve evidence and use an official reporting channel if needed.')}\n\n"
        f"### 🔎 Verify safely\n{_bullets(assessment.get('questions_to_verify', []), 'Verify through an official channel you already trust.')}"
    )
    return left, right


# ---------------------------------------------------------------------------
# Analysis pipeline entry point (feeds the SAME LangGraph workflow)
# ---------------------------------------------------------------------------

def analyze_submission(
    suspicious_text: str,
    link_email_text: str,
    screenshot_file,
    voice_file,
    preferred_language: str,
    additional_context: str,
    clicked_link: bool,
    sent_money: bool,
    shared_secret: bool,
    installed_remote_access: bool,
):
    suspicious_text = (suspicious_text or "").strip()
    link_email_text = (link_email_text or "").strip()
    media_paths = [(p, "image") for p in [_file_path(screenshot_file)] if p]
    media_paths += [(p, "audio") for p in [_file_path(voice_file)] if p]
    if not suspicious_text and not link_email_text and not media_paths:
        raise gr.Error("Paste a suspicious message/link, or upload a screenshot or voice note first.")

    parts: list[str] = []
    trace_seed: list[dict] = []
    modalities: set[str] = set()
    if suspicious_text:
        parts.append(f"SUBMITTED MESSAGE OR LINK:\n{suspicious_text}")
    if link_email_text:
        parts.append(f"SUBMITTED MESSAGE OR LINK:\n{link_email_text}")

    for file_path, modality in media_paths:
        path = Path(file_path)
        modalities.add(modality)
        modality_label = "screenshot" if modality == "image" else "voice note"
        try:
            extracted = extract_media_evidence(path.read_bytes(), path.name)
        except GeminiServiceError as exc:
            raise gr.Error(
                f"Evidence extraction from the {modality_label} is unavailable right now ({type(exc).__name__}). "
                "You can still analyze pasted text, or retry shortly."
            ) from exc
        except Exception as exc:
            raise gr.Error(f"Uploaded evidence could not be processed: {exc}") from exc
        parts.append(f"EXTRACTED FROM {path.name}:\n{extracted}")
        trace_seed.append(
            {"stage": "evidence_extraction", "status": "completed", "modality": modality_label, "extracted_chars": len(extracted)}
        )

    if media_paths and (suspicious_text or link_email_text):
        input_kind = "mixed"
    elif modalities == {"image"}:
        input_kind = "image"
    elif modalities == {"audio"}:
        input_kind = "audio"
    elif media_paths:
        input_kind = "mixed"
    else:
        input_kind = "text"
    if not media_paths:
        trace_seed.append({"stage": "evidence_extraction", "status": "skipped", "note": "text-only submission"})

    parts.append(
        "USER INTERACTION STATUS:\n"
        f"Clicked suspicious link: {'yes' if clicked_link else 'no'}\n"
        f"Sent money or made payment: {'yes' if sent_money else 'no'}\n"
        f"Shared OTP/password/PIN/recovery code: {'yes' if shared_secret else 'no'}\n"
        f"Installed or granted remote-access software: {'yes' if installed_remote_access else 'no'}"
    )
    if (additional_context or "").strip():
        parts.append(f"ADDITIONAL CONTEXT:\n{additional_context.strip()}")

    try:
        result = graph.invoke(
            {
                "raw_input": "\n\n".join(parts),
                "input_kind": input_kind,
                "preferred_language": preferred_language,
                "pipeline_trace": trace_seed,
            }
        )
    except Exception as exc:
        raise gr.Error(f"ScamShield analysis failed: {exc}") from exc

    assessment = result.get("assessment")
    ai_status = result.get("ai_status", "completed")
    case_id = result.get("case_id") or "SCAM-UNKNOWN"
    retrieval_confidence = float(result.get("retrieval_confidence") or 0.0)
    pipeline_trace = result.get("pipeline_trace") or []
    guidance = result.get("retrieved_guidance") or []

    verdict = _verdict_html(assessment, ai_status, retrieval_confidence, case_id)
    signals = _signals_html(assessment)
    actions_left, actions_right = _actions_markdown(
        assessment, clicked_link, sent_money, shared_secret, installed_remote_access
    )
    sources = _sources_html(guidance)
    # Real retrieved RAG metadata only — every cell is coerced to the column type
    # declared on the Dataframe so the grid can never blank out on a stray None.
    source_rows = [
        [
            str(item.get("organization") or ""),
            str(item.get("title") or ""),
            str(item.get("section_title") or ""),
            str(item.get("country") or ""),
            float(item.get("retrieval_score") or 0.0),
            str(item.get("source") or ""),
        ]
        for item in guidance
    ]
    trace_rows = format_trace_steps(pipeline_trace)
    detail_markdown = result.get("response_markdown") or ""
    report_path = _save_report(case_id, result.get("report_markdown") or "")

    # Developer diagnostics stay server-side; nothing sensitive reaches the browser.
    logger.info(
        "runtime %s",
        json.dumps(
            _runtime_summary(
                ai_status=ai_status,
                input_kind=result.get("input_kind") or input_kind,
                pipeline_trace=pipeline_trace,
                source_count=len(source_rows),
                retrieval_confidence=retrieval_confidence,
                risk_band=str((assessment or {}).get("risk_level") or "not_produced"),
                report_generated=bool(report_path),
            ),
            sort_keys=True,
        ),
    )

    # Plain values in exactly the order of OUTPUT_COUNT registered outputs.
    # No gr.update()/prop-update dicts: that deprecated path made Gradio rebuild
    # each component with its value stripped, which is why the collapsed sections
    # rendered empty even though the backend payload was correct.
    return (
        verdict,
        signals,
        actions_left,
        actions_right,
        sources,
        source_rows,
        detail_markdown,
        trace_rows,
        report_path,
    )


def clear_form():
    """Reset every input and output. Plain values only, same order as the wiring."""
    return (
        # ---- 10 inputs -------------------------------------------------------
        "", "", None, None, "Roman Urdu", "", False, False, False, False,
        # ---- 9 outputs (must match OUTPUT_COUNT and the order in build_demo) --
        _empty_state_html(),  # verdict_html
        "",                   # signals_html
        "",                   # actions_left
        "",                   # actions_right
        "",                   # sources_html
        [],                   # sources_table
        "",                   # result_markdown
        [],                   # trace_steps
        None,                 # report_file
    )


# ---------------------------------------------------------------------------
# Design system (dark cybersecurity theme, responsive)
# ---------------------------------------------------------------------------

CSS = """
/* ---------- Base surfaces ---------- */
.gradio-container, .dark .gradio-container {
  max-width: 1240px !important;
  margin: 0 auto !important;
  background: linear-gradient(180deg, #070d1a 0%, #0a1424 100%) !important;
  color: #e6edf7 !important;
  color-scheme: dark;
  overflow-x: hidden;
}
body { font-family: "Inter", "Segoe UI", system-ui, "Noto Naskh Arabic", "Noto Nastaliq Urdu", sans-serif; }
.gradio-container .block {
  background: #0f1b2d !important;
  border: 1px solid #1e2d45 !important;
  border-radius: 14px !important;
  box-shadow: 0 2px 10px rgba(2, 8, 20, 0.35);
}
.gradio-container .prose, .gradio-container label span, .gradio-container p,
.gradio-container .markdown, .gradio-container h1, .gradio-container h2, .gradio-container h3 {
  color: #e6edf7;
}
.gradio-container .label-wrap span, .gradio-container label > span { color: #b8c7dd !important; font-weight: 600; }
.gradio-container textarea, .gradio-container input[type="text"], .gradio-container input[type="password"] {
  background: #0b1526 !important; color: #e6edf7 !important; border: 1px solid #24385a !important;
  border-radius: 10px !important; line-height: 1.6;
}
.gradio-container textarea:focus, .gradio-container input:focus, .gradio-container button:focus-visible {
  outline: 2px solid #22d3ee !important; outline-offset: 2px;
}

/* ---------- Tabs ---------- */
.gradio-container .tab-nav { border-bottom: 1px solid #1e2d45 !important; gap: 4px; }
.gradio-container .tab-nav button {
  color: #8fa3bf !important; border-radius: 10px 10px 0 0; padding: 10px 16px; font-weight: 600;
  border: 1px solid transparent !important; background: transparent !important;
}
.gradio-container .tab-nav button.selected {
  color: #22d3ee !important; background: #0f1b2d !important; border: 1px solid #1e2d45 !important; border-bottom-color: #0f1b2d !important;
}

/* ---------- Buttons ---------- */
.gradio-container button.primary {
  background: linear-gradient(135deg, #06b6d4, #0ea5e9) !important;
  color: #04222b !important; font-weight: 700; font-size: 1.02rem;
  border: none !important; border-radius: 12px !important; padding: 12px 20px;
  box-shadow: 0 4px 14px rgba(14, 165, 233, 0.28);
}
.gradio-container button.primary:hover { filter: brightness(1.08); }
.gradio-container button.secondary {
  background: #12233c !important; color: #b8c7dd !important;
  border: 1px solid #24385a !important; border-radius: 12px !important; padding: 12px 20px;
}
.gradio-container button.secondary:hover { background: #16304f !important; }

/* ---------- Hero ---------- */
.ss-hero {
  display: flex; align-items: center; justify-content: space-between; gap: 14px;
  padding: 16px 20px; margin-bottom: 10px;
  background: linear-gradient(135deg, #0c1a30 0%, #0e2a3d 100%);
  border: 1px solid #1e3a55; border-radius: 16px;
}
.ss-brand { font-size: 1.45rem; font-weight: 800; letter-spacing: .01em; color: #f1f6fd; }
.ss-brand .shield { margin-right: 6px; }
.ss-tagline { color: #8fd6e8; font-size: .95rem; margin-top: 2px; }
.ss-status {
  display: inline-flex; align-items: center; gap: 7px; font-size: .82rem; font-weight: 600;
  color: #7ce7b6; background: rgba(52, 211, 153, 0.09); border: 1px solid rgba(52, 211, 153, 0.35);
  padding: 6px 12px; border-radius: 999px; white-space: nowrap;
}
.ss-status .dot { width: 8px; height: 8px; border-radius: 50%; background: #34d399; box-shadow: 0 0 8px rgba(52,211,153,.8); }
.ss-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
.ss-pill {
  font-size: .78rem; font-weight: 600; color: #9fb8d8; background: #101f36;
  border: 1px solid #23375a; border-radius: 999px; padding: 4px 12px; letter-spacing: .02em;
}

/* ---------- Section titles ---------- */
.ss-section-title { font-size: 1.05rem; font-weight: 700; color: #dbe8f8; margin: 4px 0 2px 2px; }
.ss-section-sub { font-size: .85rem; color: #8fa3bf; margin: 0 0 10px 2px; }

/* ---------- Empty state ---------- */
.ss-empty {
  text-align: center; padding: 34px 18px; border: 1px dashed #24385a; border-radius: 14px;
  background: #0b1526; color: #8fa3bf;
}
.ss-empty.small { padding: 18px 14px; font-size: .9rem; }
.ss-empty .empty-icon { font-size: 1.8rem; margin-bottom: 8px; }
.ss-empty .empty-title { font-weight: 700; color: #c7d6ea; }
.ss-empty .empty-sub { font-size: .85rem; margin-top: 4px; }

/* ---------- Verdict card ---------- */
.ss-verdict { border-radius: 16px; padding: 18px 20px; border: 1px solid; }
.ss-verdict .v-top { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px; }
.ss-verdict .v-label { font-size: 1.35rem; font-weight: 800; letter-spacing: .01em; }
.ss-verdict .v-meta { font-size: .9rem; color: #c7d6ea; }
.ss-verdict .v-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;
  margin: 14px 0 10px 0;
}
.ss-verdict .v-k { display: block; font-size: .74rem; text-transform: uppercase; letter-spacing: .08em; opacity: .75; }
.ss-verdict .v-v { display: block; font-weight: 700; font-size: .98rem; margin-top: 2px; }
.ss-verdict .v-note { font-size: .88rem; opacity: .92; border-top: 1px solid rgba(255,255,255,.14); padding-top: 10px; }
.ss-verdict code { background: rgba(0,0,0,.3); border-radius: 6px; padding: 1px 6px; }
.v-ok { background: linear-gradient(135deg, rgba(6,78,59,.55), rgba(16,64,80,.45)); border-color: rgba(52,211,153,.5); color: #d7fbe9; }
.v-caution { background: linear-gradient(135deg, rgba(120,53,15,.5), rgba(112,66,20,.35)); border-color: rgba(245,158,11,.55); color: #fdeed3; }
.v-danger { background: linear-gradient(135deg, rgba(127,29,29,.55), rgba(94,23,46,.4)); border-color: rgba(248,113,113,.55); color: #ffe1e1; }

/* ---------- Signal chips ---------- */
.ss-signals { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.sig-title { font-size: .85rem; font-weight: 700; margin-bottom: 8px; }
.sig-title-warn { color: #fca5a5; }
.sig-title-ok { color: #6ee7b7; }
.ss-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
  font-size: .8rem; padding: 6px 11px; border-radius: 999px; border: 1px solid; max-width: 100%;
  overflow-wrap: anywhere;
}
.chip-warn { color: #fecaca; background: rgba(248,113,113,.1); border-color: rgba(248,113,113,.4); }
.chip-ok { color: #a7f3d0; background: rgba(52,211,153,.09); border-color: rgba(52,211,153,.35); }
.chip-none { color: #8fa3bf; background: #0b1526; border-color: #24385a; }

/* ---------- Source cards ---------- */
.ss-sources { display: grid; grid-template-columns: repeat(auto-fit, minmax(255px, 1fr)); gap: 12px; }
.ss-source-card {
  background: #0b1526; border: 1px solid #24385a; border-radius: 13px; padding: 14px 16px;
  display: flex; flex-direction: column; gap: 6px;
}
.ss-source-card .src-org { font-size: .75rem; text-transform: uppercase; letter-spacing: .07em; color: #7dd3fc; font-weight: 700; }
.ss-source-card .src-title { font-weight: 700; color: #eef4fc; }
.ss-source-card .src-section { font-size: .84rem; color: #b8c7dd; }
.ss-source-card .src-score { font-size: .76rem; color: #8fa3bf; }
.ss-source-card .src-link { color: #22d3ee; font-size: .84rem; font-weight: 600; text-decoration: none; margin-top: auto; }
.ss-source-card .src-link:hover { text-decoration: underline; }

/* ---------- Misc ---------- */
.ss-disclaimer {
  font-size: .82rem; color: #fcd9a0; background: rgba(245,158,11,.08);
  border: 1px solid rgba(245,158,11,.35); border-radius: 12px; padding: 10px 14px;
}
.gradio-container .table-wrap { overflow-x: auto; }
.gradio-container table { font-size: .85rem; }
.gradio-container .gradio-container { max-width: none !important; }
.ss-examples { font-size: .82rem; opacity: .92; }

/* ---------- Responsive ---------- */
@media (max-width: 1024px) {
  .gradio-container { max-width: 100% !important; }
  .ss-signals { grid-template-columns: 1fr; }
}
@media (max-width: 768px) {
  .ss-hero { flex-direction: column; align-items: flex-start; padding: 14px 16px; }
  .ss-verdict .v-label { font-size: 1.15rem; }
  .gradio-container { padding-left: 8px !important; padding-right: 8px !important; }
}
@media (max-width: 640px) {
  .gradio-container button.primary, .gradio-container button.secondary { width: 100%; }
  .ss-sources { grid-template-columns: 1fr; }
  .ss-verdict .v-grid { grid-template-columns: 1fr 1fr; }
}
"""


def _hero_html(system_ready: bool) -> str:
    status = (
        "<span class='ss-status'><span class='dot'></span>System ready</span>"
        if system_ready
        else "<span class='ss-status' style='color:#fcd9a0;border-color:rgba(245,158,11,.4);background:rgba(245,158,11,.08);'><span class='dot' style='background:#f59e0b;'></span>Index missing — run build_index.py</span>"
    )
    return (
        "<div class='ss-hero'>"
        "<div>"
        "<div class='ss-brand'><span class='shield'>🛡️</span>ScamShield AI</div>"
        "<div class='ss-tagline'>Detect. Understand. Respond safely.</div>"
        "</div>"
        f"{status}"
        "</div>"
        "<div class='ss-pills'>"
        "<span class='ss-pill'>English</span><span class='ss-pill'>اردو · Urdu</span>"
        "<span class='ss-pill'>Roman Urdu</span><span class='ss-pill'>✓ Verified RAG</span>"
        "<span class='ss-pill'>🖼️ 🎙️ Multimodal</span>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# UI — demo examples
#
# Column order must match `inputs` in build_demo():
# suspicious_text, link_email_text, screenshot_file, voice_file,
# preferred_language, additional_context,
# clicked_link, sent_money, shared_secret, installed_remote_access
# Every host name below uses the RFC 2606 reserved `.example` TLD, so the
# fictional phishing sample can never resolve to a real site.
# ---------------------------------------------------------------------------

FICTIONAL_PHISHING_EMAIL = (
    "From: \"SecureBank Alert Team\" <no-reply@secure-bank-alert.example>\n"
    "Subject: Urgent - your account will be locked within 2 hours\n"
    "\n"
    "Dear valued customer,\n"
    "\n"
    "We detected an unusual sign-in attempt on your account from an unrecognized device.\n"
    "To keep your account active you must re-verify your identity right now through our\n"
    "secure portal:\n"
    "\n"
    "    https://secure-bank-alert.example/verify-account?ref=8f31c2\n"
    "\n"
    "If verification is not completed within 2 hours your account will be permanently\n"
    "locked and all pending transfers will be cancelled.\n"
    "\n"
    "Do not reply to this message.\n"
    "SecureBank Security Operations"
)

DEMO_EXAMPLES = [
    [
        "Assalam o Alaikum, main bank security team se hun. Apna OTP foran bhejein warna account block ho jayega.",
        "", None, None, "Roman Urdu", "WhatsApp par unknown number se aya.",
        False, False, False, False,
    ],
    [
        "We found your CV. Earn PKR 20,000 daily from home. Reply YES on WhatsApp and deposit a training fee to start.",
        "", None, None, "English", "Unexpected recruiter message.",
        False, False, False, False,
    ],
    [
        "G-483920 is your Google verification code. Do not share this code with anyone.",
        "", None, None, "English", "Arrived while I was signing in to my own account.",
        False, False, False, False,
    ],
    [
        "",
        FICTIONAL_PHISHING_EMAIL, None, None, "English",
        "Fictional sample email — arrived from an unknown sender with a look-alike link.",
        False, False, False, False,
    ],
]


def build_demo() -> gr.Blocks:
    diagnostics = rag_health()
    system_ready = diagnostics.get("indexed_sections", 0) > 0
    with gr.Blocks(title=APP_TITLE, fill_width=True, delete_cache=(86400, 86400)) as demo:
        gr.HTML(_hero_html(system_ready))

        # ---- 1. Evidence input -------------------------------------------------
        gr.HTML(
            "<div class='ss-section-title'>1 · Submit suspicious evidence</div>"
            "<div class='ss-section-sub'>Choose how you received it. Every mode is analyzed by the same pipeline.</div>"
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("💬 Text / Message"):
                        suspicious_text = gr.Textbox(
                            label="Suspicious message (SMS, WhatsApp, chat)",
                            lines=7,
                            placeholder="Example: Main bank security team se hun. Apna OTP foran bhejein warna account block ho jayega…",
                        )
                    with gr.Tab("🖼️ Screenshot"):
                        screenshot_file = gr.Image(
                            label="Upload a screenshot (preview shown below the file)",
                            type="filepath",
                            sources=["upload"],
                        )
                    with gr.Tab("🎙️ Voice Note"):
                        voice_file = gr.Audio(
                            label="Record a voice note or upload an audio file",
                            sources=["upload", "microphone"],
                            type="filepath",
                        )
                    with gr.Tab("🔗 Link / Email"):
                        link_email_text = gr.Textbox(
                            label="Paste the link or the full email body (include sender headers if you have them)",
                            lines=7,
                            placeholder="Example: From: security@bank-login-alert.xyz … Click here to verify your account: http://…",
                        )
                additional_context = gr.Textbox(
                    label="Optional context — where did it arrive and what happened?",
                    lines=2,
                    placeholder="e.g. WhatsApp par unknown number se aya…",
                )
            # ---- 2. Case context ----------------------------------------------
            with gr.Column(scale=2):
                gr.HTML("<div class='ss-section-title'>2 · What already happened?</div>")
                clicked_link = gr.Checkbox(label="I clicked the link")
                sent_money = gr.Checkbox(label="I sent money / made a payment")
                shared_secret = gr.Checkbox(label="I shared an OTP, password, PIN, or recovery code")
                installed_remote_access = gr.Checkbox(label="I installed or allowed remote-access software")
                preferred_language = gr.Dropdown(
                    ["English", "Urdu", "Roman Urdu"], value="Roman Urdu", label="Response language"
                )
                gr.HTML(f"<div class='ss-disclaimer'>⚠️ {DISCLAIMER}</div>")

        # ---- 3. Primary CTA ----------------------------------------------------
        with gr.Row():
            analyze_button = gr.Button("🛡️  Analyze safely", variant="primary", scale=3, elem_classes=["cta-primary"])
            clear_button = gr.Button("↺  Clear / New case", variant="secondary", scale=1)

        # ---- 4. Verdict ---------------------------------------------------------
        gr.HTML("<div class='ss-section-title'>3 · Assessment</div>")
        verdict_html = gr.HTML(_empty_state_html())
        signals_html = gr.HTML("")

        # ---- 5. Actions -----------------------------------------------------------
        with gr.Row(equal_height=True):
            actions_left = gr.Markdown("")
            actions_right = gr.Markdown("")

        # ---- 6. Verified RAG sources ---------------------------------------------
        # These components are always rendered; the Accordion alone controls the
        # collapse. Hiding the component itself (visible=False) required a
        # gr.update() prop-update to reveal it, and that path dropped the value.
        gr.HTML("<div class='ss-section-title'>4 · Verified evidence used for this assessment</div>")
        sources_html = gr.HTML("")
        with gr.Accordion("Detailed source table", open=False):
            sources_table = gr.Dataframe(
                headers=["Organization", "Source", "Matched section", "Region", "Score", "URL"],
                datatype=["str", "str", "str", "str", "number", "str"],
                interactive=False,
                wrap=True,
            )
        with gr.Accordion("Full assessment detail", open=False):
            result_markdown = gr.Markdown("")

        # ---- 7. Technical trace (collapsed for ordinary users) --------------------
        with gr.Accordion("🔬 Technical analysis trace (for judges & reviewers)", open=False):
            gr.Markdown(
                "Observable system-level events only — input modality, extraction status, indicator labels, retrieval "
                "metrics (semantic, BM25, RRF/MMR, multi-query, parent context), and Gemini status. "
                "Never raw secrets or hidden model reasoning. Deeper developer diagnostics are written to the "
                "server log only."
            )
            trace_steps = gr.Dataframe(
                headers=["Pipeline step", "Status", "Observation"],
                datatype=["str", "str", "str"],
                interactive=False,
                wrap=True,
            )
        with gr.Accordion("📄 Case report export", open=False):
            report_file = gr.File(label="Privacy-redacted report", interactive=False)

        # ---- 8. About the analysis --------------------------------------------------
        with gr.Accordion("ℹ️ About the analysis", open=False):
            gr.Markdown(
                f"- **Verified sources:** {diagnostics['parent_documents']} curated cyber-safety documents\n"
                f"- **Indexed sections:** {diagnostics['indexed_sections']}\n"
                f"- **Embedding model:** `{diagnostics['embedding_model']}`\n"
                "- **Pipeline:** LangGraph conditional workflow · Gemini assessment · ChromaDB + BM25 hybrid retrieval · RRF/MMR fusion · conditional Multi-Query · parent-context restoration\n"
                "- **Languages:** English, Urdu, Roman Urdu\n\n"
                "ScamShield provides risk guidance, not a legal finding or a guarantee of safety."
            )

        inputs = [
            suspicious_text, link_email_text, screenshot_file, voice_file,
            preferred_language, additional_context,
            clicked_link, sent_money, shared_secret, installed_remote_access,
        ]
        outputs = [
            verdict_html, signals_html, actions_left, actions_right, sources_html,
            sources_table, result_markdown, trace_steps, report_file,
        ]
        # Fail loudly at build time if the callback contract ever drifts, instead
        # of silently blanking half the UI at request time.
        assert len(inputs) == INPUT_COUNT, f"expected {INPUT_COUNT} inputs, wired {len(inputs)}"
        assert len(outputs) == OUTPUT_COUNT, f"expected {OUTPUT_COUNT} outputs, wired {len(outputs)}"
        assert len(clear_form()) == INPUT_COUNT + OUTPUT_COUNT, "clear_form arity drifted"

        analyze_button.click(fn=analyze_submission, inputs=inputs, outputs=outputs, show_progress="full")
        clear_button.click(fn=clear_form, inputs=[], outputs=inputs + outputs)

        with gr.Column(elem_classes=["ss-examples"]):
            gr.Examples(
                examples=DEMO_EXAMPLES,
                inputs=inputs,
                cache_examples=False,
                label=(
                    "Demo examples (OTP/bank impersonation · fake job · legitimate security notification · "
                    "fictional phishing email — reserved .example domains only)"
                ),
            )
    return demo


demo = build_demo()

if __name__ == "__main__":
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    demo.queue(default_concurrency_limit=2, max_size=20).launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        css=CSS,
        show_error=True,
        # Lets gr.File serve the generated report on read-only-root Linux images
        # such as Hugging Face Spaces, where /tmp is the writable location.
        allowed_paths=[str(REPORT_DIR)],
    )
