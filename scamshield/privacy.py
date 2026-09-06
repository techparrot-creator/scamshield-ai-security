from __future__ import annotations

import re


def _mask_digits(value: str, keep: int = 4) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) <= keep:
        return "*" * len(digits)
    return "*" * (len(digits) - keep) + digits[-keep:]


def redact_sensitive_text(text: str) -> str:
    """Redact likely secrets and personal identifiers from downloadable reports.

    The live analysis still sees the evidence supplied by the user; this function is for
    safer report export. It intentionally keeps some context so evidence remains useful.
    """
    value = text or ""

    # Labelled secrets: OTP: 123456, PIN 1234, password=abc123.
    value = re.sub(
        r"(?i)\b(otp|pin|password|passcode|recovery\s*code)\b\s*[:=\-]?\s*([A-Za-z0-9@#$%^&*!._-]{4,32})",
        lambda m: f"{m.group(1)}: [REDACTED]",
        value,
    )

    # Pakistan CNIC with or without dashes.
    value = re.sub(r"\b\d{5}-?\d{7}-?\d\b", "*****-*******-*", value)

    # Common Pakistani mobile numbers.
    value = re.sub(
        r"\b(?:\+92|0092|0)?3\d{9}\b",
        lambda m: "[PHONE:" + _mask_digits(m.group(0), keep=3) + "]",
        value,
    )

    # Card-like 13-19 digit sequences, spaces/dashes allowed.
    def card_repl(match: re.Match) -> str:
        raw = match.group(0)
        digits = re.sub(r"\D", "", raw)
        if 13 <= len(digits) <= 19:
            return "[CARD:" + _mask_digits(digits, keep=4) + "]"
        return raw

    value = re.sub(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)", card_repl, value)

    # Partially mask email local-parts.
    value = re.sub(
        r"\b([A-Za-z0-9._%+-]{2,})@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b",
        lambda m: f"{m.group(1)[0]}***@{m.group(2)}",
        value,
    )
    return value
