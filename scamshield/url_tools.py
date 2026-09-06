from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\]\[\"']+")
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "cutt.ly", "rb.gy", "is.gd", "rebrand.ly", "shorturl.at"
}
SUSPICIOUS_TERMS = {
    "verify", "login", "secure", "update", "wallet", "claim", "reward", "prize", "bonus", "otp", "urgent",
    "bank", "support", "gift", "crypto", "refund", "account", "password"
}


def extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    for match in URL_PATTERN.findall(text or ""):
        cleaned = match.rstrip(".,;:!?)")
        if cleaned not in urls:
            urls.append(cleaned)
    return urls


def _is_ip(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def analyze_url(url: str) -> dict:
    candidate = url if "://" in url else f"https://{url}"
    parsed = urlparse(candidate)
    host = (parsed.hostname or "").lower()
    full = candidate.lower()
    signals: list[str] = []
    labels: list[str] = []
    score = 0

    if not host:
        return {
            "url": url,
            "heuristic_score": 40,
            "signals": ["URL could not be parsed reliably."],
            "labels": ["unparseable"],
            "note": "Structural heuristic only; no live reputation lookup was performed.",
        }

    if _is_ip(host):
        score += 25
        labels.append("raw_ip_host")
        signals.append("Uses a raw IP address instead of a normal domain.")
    if "xn--" in host:
        score += 20
        labels.append("punycode_lookalike")
        signals.append("Contains punycode, which can be used for look-alike domains.")
    if "@" in parsed.netloc:
        score += 20
        labels.append("at_symbol_redirect")
        signals.append("Contains an @ symbol that can hide the real destination.")
    if host in SHORTENERS:
        score += 15
        labels.append("link_shortener")
        signals.append("Uses a link-shortening service that hides the final destination.")
    if host.count(".") >= 3:
        score += 10
        labels.append("many_subdomains")
        signals.append("Contains many subdomains, making the real domain harder to notice.")
    matched_terms = sorted(term for term in SUSPICIOUS_TERMS if term in full)
    if matched_terms:
        score += min(20, 4 * len(matched_terms))
        labels.append("suspicious_terms")
        signals.append(f"Contains account, payment, or urgency terms: {', '.join(matched_terms[:5])}.")
    if parsed.scheme != "https":
        score += 5
        labels.append("no_https")
        signals.append("Does not use HTTPS.")
    if len(candidate) > 120:
        score += 5
        labels.append("long_url")
        signals.append("Unusually long URL.")
    if sum(ch.isdigit() for ch in host) >= 5:
        score += 5
        labels.append("digit_heavy_host")
        signals.append("Domain contains many digits.")

    return {
        "url": url,
        "hostname": host,
        "heuristic_score": min(score, 100),
        "signals": signals or ["No obvious structural red flags were found."],
        "labels": labels,
        "note": "Structural heuristic only; absence of red flags does not prove a URL is safe.",
    }
