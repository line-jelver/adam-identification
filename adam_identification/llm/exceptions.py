"""Typed exception hierarchy for LLM providers."""

from adam_identification.exceptions import (  # noqa: F401
    AuthenticationError,
    InvalidResponseError,
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
)

__all__ = [
    "AuthenticationError",
    "InvalidResponseError",
    "LLMError",
    "ProviderUnavailableError",
    "RateLimitError",
    "exception_for_http_status",
]


def exception_for_http_status(
    provider: str,
    status: int,
    *,
    model: str | None = None,
) -> LLMError:
    """Map a provider HTTP status code to a typed ``LLMError``.

    429 is retried as a rate limit. 5xx is ``ProviderUnavailableError`` and is
    retried. Other 4xx codes, including 404 for an unknown model slug, are
    ``InvalidResponseError`` and must not be retried.

    Args:
        provider: Display name (e.g. ``"Gemini"``).
        status: HTTP status code from the provider API.
        model: Optional model slug included in 4xx messages.

    Returns:
        An exception instance ready to ``raise ... from`` the HTTP error.
    """
    model_note = f" for model {model!r}" if model else ""
    if status == 429:
        return RateLimitError(f"{provider} rate limit exceeded.")
    if status == 401:
        return AuthenticationError(f"{provider} authentication failed.")
    if 500 <= status < 600:
        return ProviderUnavailableError(f"{provider} service unavailable (HTTP {status}).")
    if status == 404:
        return InvalidResponseError(
            f"{provider} could not find the requested model{model_note} (HTTP 404). "
            "Check the model slug and provider."
        )
    return InvalidResponseError(f"{provider} request failed (HTTP {status}){model_note}.")
