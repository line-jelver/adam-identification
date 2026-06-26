"""Typed exception hierarchy for adam-identification.

Raised by database clients and the identifier agent so callers can react by
exception type rather than by parsing error strings.
"""


class IdentificationError(Exception):
    """Base class for all adam-identification errors."""


class MaterialNotFoundError(IdentificationError):
    """Raised when a formula or name query returns no usable database match."""


class DatabaseAPIError(IdentificationError):
    """Raised on network, authentication, or malformed-payload failures."""


class LLMError(IdentificationError):
    """Base class for LLM provider errors."""


class RateLimitError(LLMError):
    """Provider rate limit hit. The caller can back off and retry."""


class AuthenticationError(LLMError):
    """API key is missing, expired, or invalid."""


class ProviderUnavailableError(LLMError):
    """Provider service is temporarily unavailable or unreachable."""


class InvalidResponseError(LLMError):
    """Provider returned a response that could not be parsed."""
