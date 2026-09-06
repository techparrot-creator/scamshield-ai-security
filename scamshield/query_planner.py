from __future__ import annotations

import re

TOPIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "phishing": ("phish", "link", "login", "verify", "password", "credential", "email", "sms", "smishing"),
    "otp_account_takeover": ("otp", "verification code", "6 digit", "six digit", "account block", "account locked"),
    "payment_fraud": ("payment", "paid", "sent money", "easypaisa", "jazzcash", "bank transfer", "wallet", "refund"),
    "job_scam": ("job", "recruiter", "salary", "hiring", "work from home", "task", "registration fee"),
    "investment_scam": ("investment", "crypto", "forex", "profit", "return", "trading", "guaranteed return"),
    "romance_scam": ("dating", "romance", "love", "relationship", "marriage", "fiance"),
    "blackmail": ("blackmail", "sextortion", "intimate", "leak photos", "threat", "expose"),
    "whatsapp_hijack": ("whatsapp", "qr code", "linked device", "call forwarding"),
    "impersonation": ("police", "government", "bank officer", "support team", "courier", "customs", "authority"),
}

PAKISTAN_TERMS = (
    "pakistan", "nccia", "pta", "pkcert", "cnic", "easypaisa", "jazzcash", "nadra", "fbr", "rs.", "pkr"
)

# Benign pattern: a service sends a code and warns the user NOT to share it.
SECRET_BENIGN_PATTERNS = (
    "do not share", "don't share", "dont share", "never share",
    "treat this code as sensitive",
    "share na karein", "share na kare", "na share karein", "kisi se share na",
    "kisi ko na bata", "share mat karein", "share mat karo", "mat bhejo", "mat bhejein",
)

# Scam pattern: someone asks the user to send/share/forward/reveal a code.
SECRET_REQUEST_PATTERNS = (
    "send me", "give me the code", "give me your code", "tell me the code", "tell me your code",
    "what is the code", "what is your code", "provide the code", "provide your code",
    "share the code", "send the code", "send your code", "code forward",
    "bhejo", "bhej do", "bhej dein", "bhej dain", "forward kar",
    "share karo", "share kar do", "share kar dein", "share karain", "share karein",
    "code bata", "otp bata", "code likh",
)

# Heuristic indicator labels for the pipeline trace (labels only, never raw text).
URGENCY_TERMS = (
    "urgent", "foran", "fauran", "immediately", "right now", "abhi", "warna", "block ho jay",
    "within 24 hours", "within 2 hours", "last chance", "expires", "expire", "final warning", "deadline",
)
PAYMENT_TERMS = (
    "payment", "pay a", "pay the", "paid", "paying", "deposit", "fee", "transfer money", "send money",
    "paise bhej", "paisa bhej", "easypaisa", "jazzcash", "refund", "processing fee", "registration fee",
    "training fee", "clearance fee", "advance fee",
)
CREDENTIAL_TERMS = (
    "password", "pin", "cvv", "card number", "card details", "recovery code", "credentials",
    "login details", "account details", "password bata", "pin bata",
)


def classify_secret_flow(text: str) -> str:
    """Heuristically classify the direction of an OTP/security-code flow.

    Returns one of:
    - "service_warning_do_not_share": a service sent a code and warns not to share it (benign).
    - "code_requested_from_user": someone asks the user to send/share/reveal a code (scam signal).
    - "mixed_signals": both patterns are present.
    - "none": no code-flow pattern was detected.
    """
    normalized = re.sub(r"\s+", " ", (text or "").lower().replace("\u2019", "'").replace("\u2018", "'"))
    benign = any(pattern in normalized for pattern in SECRET_BENIGN_PATTERNS)
    requested = any(pattern in normalized for pattern in SECRET_REQUEST_PATTERNS)
    if requested and benign:
        return "mixed_signals"
    if requested:
        return "code_requested_from_user"
    if benign:
        return "service_warning_do_not_share"
    return "none"


def infer_retrieval_hints(text: str) -> dict:
    normalized = re.sub(r"\s+", " ", (text or "").lower())
    topics = [topic for topic, patterns in TOPIC_PATTERNS.items() if any(p in normalized for p in patterns)]
    country = "Pakistan" if any(term in normalized for term in PAKISTAN_TERMS) else None
    return {"topics": topics[:4], "country": country}


def detect_indicator_labels(text: str) -> list[str]:
    """Detect deterministic indicator labels for the pipeline trace.

    Returns compact labels such as ``otp_request``, ``urgency``, ``payment_request``,
    ``credential_request``, ``impersonation_claim``, or reassuring labels like
    ``reassuring:do_not_share_code``. Only labels are returned, never raw text.
    """
    normalized = re.sub(r"\s+", " ", (text or "").lower())
    labels: set[str] = set()
    flow = classify_secret_flow(text)
    if flow == "code_requested_from_user":
        labels.add("otp_request")
    if flow == "service_warning_do_not_share":
        labels.add("reassuring:do_not_share_code")
    if any(term in normalized for term in URGENCY_TERMS):
        labels.add("urgency")
    if any(term in normalized for term in PAYMENT_TERMS):
        labels.add("payment_request")
    if any(term in normalized for term in CREDENTIAL_TERMS):
        labels.add("credential_request")
    topics = infer_retrieval_hints(text)["topics"]
    if "impersonation" in topics:
        labels.add("impersonation_claim")
    if "phishing" in topics and any(marker in normalized for marker in ("http", "www.", "link", "login", "sign in", "sign-in")):
        labels.add("suspicious_link_or_login")
    if "blackmail" in topics:
        labels.add("threat_pressure")
    informational_only = not labels - {"reassuring:do_not_share_code"}
    if informational_only and flow in {"none", "service_warning_do_not_share"} and not any(
        term in normalized for term in PAYMENT_TERMS + CREDENTIAL_TERMS
    ):
        labels.add("reassuring:no_harmful_request_detected")
    return sorted(labels)


def should_expand_query(confidence: float, result_count: int, threshold: float = 0.43) -> bool:
    return result_count < 2 or confidence < threshold
