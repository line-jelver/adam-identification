"""Tests for LLM exception hierarchy."""

from __future__ import annotations

import pytest

from adam_identification.llm.exceptions import (
    AuthenticationError,
    InvalidResponseError,
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
    exception_for_http_status,
)


def test_exception_inheritance() -> None:
    """All specialized exceptions inherit from LLMError."""
    assert issubclass(RateLimitError, LLMError)
    assert issubclass(AuthenticationError, LLMError)
    assert issubclass(ProviderUnavailableError, LLMError)
    assert issubclass(InvalidResponseError, LLMError)
    assert issubclass(LLMError, Exception)


@pytest.mark.parametrize(
    "exc_type",
    [RateLimitError, AuthenticationError, ProviderUnavailableError, InvalidResponseError],
)
def test_exceptions_catchable(exc_type: type[LLMError]) -> None:
    """Each exception type can be caught directly and as LLMError."""
    with pytest.raises(exc_type):
        raise exc_type("boom")
    with pytest.raises(LLMError):
        raise exc_type("boom")


@pytest.mark.parametrize(
    ("status", "exc_type"),
    [
        (429, RateLimitError),
        (401, AuthenticationError),
        (404, InvalidResponseError),
        (400, InvalidResponseError),
        (403, InvalidResponseError),
        (503, ProviderUnavailableError),
    ],
)
def test_exception_for_http_status(status: int, exc_type: type[LLMError]) -> None:
    """HTTP statuses map to retryable vs non-retryable LLM errors."""
    exc = exception_for_http_status("Gemini", status)
    assert isinstance(exc, exc_type)


def test_exception_for_http_status_404_mentions_model() -> None:
    exc = exception_for_http_status("Gemini", 404)
    assert "404" in str(exc)
    assert "model" in str(exc).lower()


def test_exception_for_http_status_includes_slug() -> None:
    exc = exception_for_http_status("OpenAI", 400, model="gpt-6")
    assert "gpt-6" in str(exc)
    assert "400" in str(exc)
