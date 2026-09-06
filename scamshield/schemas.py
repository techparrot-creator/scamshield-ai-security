from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

LanguageChoice = Literal["English", "Urdu", "Roman Urdu"]
RiskLevel = Literal["low", "medium", "high", "critical"]


class EvidenceItem(BaseModel):
    indicator: str = Field(description="Specific suspicious or reassuring indicator.")
    evidence: str = Field(description="Short factual observation from the submitted evidence.")
    severity: Literal["low", "medium", "high"]


class IncidentStep(BaseModel):
    step: int = Field(ge=1, le=12)
    event: str
    significance: str


class QueryExpansion(BaseModel):
    queries: list[str] = Field(min_length=2, max_length=4)


class ScamAssessment(BaseModel):
    risk_score: int = Field(ge=0, le=100)
    risk_level: RiskLevel
    confidence: int = Field(ge=0, le=100)
    scam_type: str
    summary: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    benign_indicators: list[str] = Field(
        default_factory=list,
        description="Reassuring signals that argue against a scam, e.g. no request for secrets or payment.",
    )
    sender_authenticity: Literal["verified", "unverifiable", "suspicious"] = Field(
        default="unverifiable",
        description="Whether sender identity could be established from the evidence alone.",
    )
    incident_timeline: list[IncidentStep] = Field(default_factory=list)
    immediate_actions: list[str] = Field(default_factory=list)
    recovery_actions: list[str] = Field(default_factory=list)
    questions_to_verify: list[str] = Field(default_factory=list)
    reporting_steps: list[str] = Field(default_factory=list)
    safe_reply: str = Field(
        description="Short non-confrontational reply, or empty string if the safest action is not to reply."
    )
    uncertainty_note: str


class ScamShieldState(TypedDict, total=False):
    raw_input: str
    input_kind: str
    preferred_language: LanguageChoice
    case_id: str
    normalized_text: str
    detected_urls: list[str]
    url_findings: list[dict]
    retrieval_hints: dict
    query_variants: list[str]
    retrieved_guidance: list[dict]
    retrieval_confidence: float
    retrieval_strategy: list[str]
    secret_flow_signal: str
    # "completed", "temporarily_unavailable", or "configuration_error".
    ai_status: str
    assessment: dict | None
    response_markdown: str
    report_markdown: str
    # Observable system-level events only; never raw user content or model reasoning.
    # The add-reducer lets every node append its own events without overwriting.
    pipeline_trace: Annotated[list[dict], operator.add]
