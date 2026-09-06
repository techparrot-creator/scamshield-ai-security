"""Centralized resilient Gemini invocation.

Every Gemini call in ScamShield goes through `invoke_with_retry` so that:
- transient failures (network disconnects, connection errors, read timeouts,
  retryable 429/5xx API errors) are retried with exponential backoff;
- authentication/configuration errors (invalid API key, invalid request,
  unsupported model, permission denied) are NOT blindly retried;
- logs keep full failure detail while secrets are redacted;
- callers receive typed errors (`GeminiUnavailableError`,
  `GeminiConfigurationError`) instead of raw SDK tracebacks.
"""
from __future__ import annotations

import os
import time
from typing import Callable, TypeVar

from scamshield.config import settings

try:  # httpx is the transport used by google-genai; keep the import defensive.
    import httpx
except ImportError:  # pragma: no cover - httpx ships with google-genai
    httpx = None

try:
    from google.genai import errors as genai_errors
except ImportError:  # pragma: no cover - google-genai is a hard dependency
    genai_errors = None

try:
    # langchain-google-genai re-raises API failures as langchain_core ModelError
    # subclasses (e.g. GoogleInvalidRequestError), which carry no HTTP code.
    from langchain_core import exceptions as langchain_errors
except ImportError:  # pragma: no cover - langchain-core is a hard dependency
    langchain_errors = None

T = TypeVar("T")

# HTTP statuses considered safe to retry (rate limits and server-side faults).
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
# HTTP statuses that indicate a configuration/authentication problem.
CONFIGURATION_STATUS = {400, 401, 403, 404}

SECRET_ENV_VARS = ("GOOGLE_API_KEY", "GEMINI_API_KEY")


class GeminiServiceError(RuntimeError):
    """Base class for sanitized Gemini service failures."""


class GeminiUnavailableError(GeminiServiceError):
    """A transient failure persisted after all retry attempts."""


class GeminiConfigurationError(GeminiServiceError):
    """A non-retryable authentication or configuration failure."""


def redact_secrets(text: str) -> str:
    """Replace any configured API key value that leaked into an error string."""
    sanitized = text or ""
    for var in SECRET_ENV_VARS:
        secret = os.getenv(var)
        if secret:
            sanitized = sanitized.replace(secret, "[REDACTED]")
    return sanitized


def _http_status(exc: BaseException) -> int | None:
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def classify_gemini_error(exc: BaseException) -> str:
    """Classify a Gemini-call failure as 'retryable', 'configuration', or 'unknown'.

    Only clearly transient failures are retried. Unknown errors surface as-is so
    genuine bugs are never masked by retry loops.
    """
    if isinstance(exc, GeminiConfigurationError):
        return "configuration"
    status = _http_status(exc)
    if status is not None:
        if status in RETRYABLE_STATUS or 500 <= status <= 599:
            return "retryable"
        if status in CONFIGURATION_STATUS:
            return "configuration"
        return "unknown"
    if langchain_errors is not None:
        if isinstance(
            exc,
            (
                langchain_errors.ModelRateLimitError,
                langchain_errors.ModelAPIError,
                langchain_errors.ModelConnectionError,
                langchain_errors.ModelTimeoutError,
            ),
        ):
            return "retryable"
        if isinstance(
            exc,
            (
                langchain_errors.ModelAuthenticationError,
                langchain_errors.ModelPermissionDeniedError,
                langchain_errors.ModelInvalidRequestError,
                langchain_errors.ModelNotFoundError,
                langchain_errors.ContextOverflowError,
            ),
        ):
            return "configuration"
    if httpx is not None and isinstance(exc, httpx.TransportError):
        # Covers ConnectError, ReadError, RemoteProtocolError, read timeouts, etc.
        return "retryable"
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return "retryable"
    if genai_errors is not None:
        if isinstance(exc, genai_errors.ServerError):
            return "retryable"
        if isinstance(exc, genai_errors.ClientError):
            return "configuration"
    return "unknown"


def invoke_with_retry(
    invoker: Callable[[], T],
    *,
    description: str = "Gemini call",
    max_attempts: int | None = None,
    base_delay: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `invoker` with exponential backoff for transient Gemini failures.

    Defaults: up to 3 total attempts with ~1s, 2s waits (exponential).
    Failures are logged with secrets redacted; the raised errors carry only
    sanitized messages.
    """
    attempts = max_attempts or max(1, settings.gemini_max_attempts)
    delay_base = settings.gemini_retry_base_delay if base_delay is None else base_delay
    last_transient: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return invoker()
        except Exception as exc:
            kind = classify_gemini_error(exc)
            detail = redact_secrets(repr(exc))
            if kind == "configuration":
                print(
                    f"[ScamShield Gemini] {description}: non-retryable configuration/auth "
                    f"error on attempt {attempt}/{attempts}: {detail}"
                )
                raise GeminiConfigurationError(
                    f"{description} failed because of a configuration or authentication error "
                    f"({type(exc).__name__}, HTTP status {_http_status(exc) or 'n/a'}). "
                    "Check GOOGLE_API_KEY and model settings."
                ) from exc
            if kind == "unknown":
                print(f"[ScamShield Gemini] {description}: unexpected error on attempt {attempt}/{attempts}: {detail}")
                raise
            print(
                f"[ScamShield Gemini] {description}: transient failure on attempt {attempt}/{attempts}: {detail}"
            )
            last_transient = exc
            if attempt < attempts:
                sleep(delay_base * (2 ** (attempt - 1)))

    raise GeminiUnavailableError(
        f"{description} is temporarily unavailable after {attempts} attempts due to transient errors."
    ) from last_transient
