"""Tests for the centralized resilient Gemini invocation helper."""
from __future__ import annotations

import httpx
import pytest
from google.genai import errors as genai_errors

from scamshield.llm_client import (
    GeminiConfigurationError,
    GeminiUnavailableError,
    classify_gemini_error,
    invoke_with_retry,
)


class _Counter:
    def __init__(self) -> None:
        self.calls = 0

    def bump(self) -> None:
        self.calls += 1


def test_two_transient_failures_then_success():
    counter = _Counter()
    sleeps: list[float] = []

    def invoker():
        counter.bump()
        if counter.calls <= 2:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return "assessment-ok"

    result = invoke_with_retry(invoker, description="test call", sleep=sleeps.append)
    assert result == "assessment-ok"
    assert counter.calls == 3
    # Exponential backoff between the three attempts.
    assert sleeps == [1.0, 2.0]


def test_all_transient_attempts_fail_raises_graceful_unavailable():
    counter = _Counter()

    def invoker():
        counter.bump()
        raise httpx.ConnectError("connection refused")

    with pytest.raises(GeminiUnavailableError):
        invoke_with_retry(invoker, description="test call", sleep=lambda _: None)
    assert counter.calls == 3


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: httpx.ReadTimeout("read timed out"),
        lambda: httpx.NetworkError("network unreachable"),
        lambda: ConnectionResetError("connection reset by peer"),
        lambda: TimeoutError("operation timed out"),
        lambda: genai_errors.ServerError(503, {"error": {"message": "overloaded"}}),
        lambda: genai_errors.ClientError(429, {"error": {"message": "rate limited"}}),
    ],
)
def test_transient_errors_are_retryable(exc_factory):
    assert classify_gemini_error(exc_factory()) == "retryable"


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: genai_errors.ClientError(400, {"error": {"message": "API key not valid"}}),
        lambda: genai_errors.ClientError(401, {"error": {"message": "unauthenticated"}}),
        lambda: genai_errors.ClientError(403, {"error": {"message": "permission denied"}}),
        lambda: genai_errors.ClientError(404, {"error": {"message": "model not found"}}),
    ],
)
def test_configuration_errors_are_not_retryable(exc_factory):
    assert classify_gemini_error(exc_factory()) == "configuration"


def test_non_retryable_configuration_error_is_not_repeatedly_retried():
    counter = _Counter()

    def invoker():
        counter.bump()
        raise genai_errors.ClientError(400, {"error": {"message": "API key not valid"}})

    with pytest.raises(GeminiConfigurationError):
        invoke_with_retry(invoker, description="test call", sleep=lambda _: None)
    assert counter.calls == 1


def test_langchain_wrapped_invalid_request_is_configuration():
    # langchain-google-genai re-raises 400s as GoogleInvalidRequestError without an HTTP code.
    from langchain_google_genai.chat_models import GoogleInvalidRequestError

    counter = _Counter()

    def invoker():
        counter.bump()
        raise GoogleInvalidRequestError("API key not valid. Please pass a valid API key.")

    with pytest.raises(GeminiConfigurationError):
        invoke_with_retry(invoker, description="test call", sleep=lambda _: None)
    assert counter.calls == 1


def test_langchain_wrapped_rate_limit_is_retried():
    from langchain_google_genai.chat_models import GoogleRateLimitError

    assert classify_gemini_error(GoogleRateLimitError("rate limited")) == "retryable"


def test_unknown_error_is_not_retried_or_masked():
    counter = _Counter()

    def invoker():
        counter.bump()
        raise ValueError("unexpected payload shape")

    with pytest.raises(ValueError):
        invoke_with_retry(invoker, description="test call", sleep=lambda _: None)
    assert counter.calls == 1


def test_secrets_never_leak_into_errors_or_logs(capsys, monkeypatch):
    secret = "AIzaSySECRETKEY-DO-NOT-LEAK-123"
    monkeypatch.setenv("GOOGLE_API_KEY", secret)

    def invoker():
        raise httpx.ReadError(f"connection aborted while sending key={secret}")

    with pytest.raises(GeminiUnavailableError) as excinfo:
        invoke_with_retry(invoker, description="test call", sleep=lambda _: None)
    captured = capsys.readouterr()
    assert secret not in str(excinfo.value)
    assert secret not in captured.out
    assert "[REDACTED]" in captured.out
