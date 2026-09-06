from __future__ import annotations

import json
from pathlib import Path

from scamshield.prompts import ANALYSIS_TEMPLATE, SYSTEM_PROMPT
from scamshield.query_planner import classify_secret_flow

EVAL_FILE = Path(__file__).resolve().parent.parent / "data" / "eval_cases.jsonl"


def _load_cases() -> list[dict]:
    return [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_benign_service_code_english():
    assert classify_secret_flow("Your verification code is 123456. Do not share it.") == "service_warning_do_not_share"


def test_code_request_english():
    assert classify_secret_flow("Send me the verification code you just received.") == "code_requested_from_user"


def test_code_request_roman_urdu():
    assert classify_secret_flow("Main bank security se hun. OTP foran bhejo warna account block ho jayega.") == "code_requested_from_user"


def test_benign_do_not_share_roman_urdu():
    assert classify_secret_flow("Aap ka OTP 558899 hai. Is code ko kisi se share na karein.") == "service_warning_do_not_share"


def test_no_secret_flow_in_benign_notification():
    assert classify_secret_flow("Your parcel is out for delivery today.") == "none"


def test_adversarial_pair_flows_in_opposite_directions():
    by_id = {case["id"]: case for case in _load_cases()}
    benign, scam = by_id["pair-otp-benign"], by_id["pair-otp-scam"]
    assert classify_secret_flow(benign["text"]) == "service_warning_do_not_share"
    assert classify_secret_flow(scam["text"]) == "code_requested_from_user"
    assert benign["expected_risk"] == "low"
    assert scam["expected_risk"] in {"high", "critical"}


def test_eval_dataset_has_legitimate_negative_controls():
    cases = _load_cases()
    legitimate = [case for case in cases if case.get("kind") == "legitimate"]
    assert len(legitimate) >= 6
    assert all(case["expected_risk"] == "low" for case in legitimate)


def test_existing_scam_cases_preserved():
    ids = {case["id"] for case in _load_cases()}
    assert {
        "otp-bank-roman-ur",
        "job-fee-en",
        "crypto-invest-en",
        "whatsapp-code-roman",
        "payment-recovery",
        "romance-money",
    } <= ids


def test_prompts_include_false_positive_calibration():
    assert "{secret_flow}" in ANALYSIS_TEMPLATE
    assert "false positives" in ANALYSIS_TEMPLATE.lower()
    assert "NOT to share" in ANALYSIS_TEMPLATE  # OTP-direction distinction
    assert "unverifiable" in ANALYSIS_TEMPLATE  # brand authenticity handling
    assert "background knowledge" in SYSTEM_PROMPT.lower()
