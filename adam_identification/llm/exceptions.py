"""Typed exception hierarchy for LLM providers."""

from adam_identification.exceptions import (  # noqa: F401
    AuthenticationError,
    InvalidResponseError,
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
)
