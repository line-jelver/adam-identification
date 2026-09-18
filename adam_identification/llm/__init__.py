"""LLM provider factory for adam-identification.

Usage::

    from adam_identification.llm import get_provider

    llm = get_provider(model="gemini-2.5-flash")
    response = await llm.complete_single("Identify silicon")
"""

from __future__ import annotations

from adam_identification._config import ConfigurationError
from adam_identification.llm.base import BaseLLM, LLMResponse, Message, TokenUsage
from adam_identification.llm.retry import (
    DEFAULT_RATE_LIMIT_DELAYS_S,
    RetryResult,
    retry_on_rate_limit,
)

MISSING_MODEL_MESSAGE = (
    "A model slug is required; there is no default. "
    "Pass model=... / --model, for example model='gemini-3.1-pro-preview'. "
    "Smaller or cheaper models often fail to abstain on underspecified queries."
)

_VALID_PROVIDERS = ("openai", "anthropic", "google", "openrouter")
_OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4")
_NATIVE_ORG_PREFIXES = {
    "openai": "openai",
    "google": "google",
    "anthropic": "anthropic",
}

__all__ = [
    "MISSING_MODEL_MESSAGE",
    "infer_provider",
    "get_provider",
    "BaseLLM",
    "Message",
    "TokenUsage",
    "LLMResponse",
    "retry_on_rate_limit",
    "RetryResult",
    "DEFAULT_RATE_LIMIT_DELAYS_S",
]


def infer_provider(model: str) -> str:
    """Return the native provider key for a model slug.

    New slugs are recognized by name family, not an allowlist. ``openai/``,
    ``google/``, and ``anthropic/`` prefixes map to those native APIs.
    Other ``org/model`` slugs map to ``openrouter``.

    Args:
        model: Model slug, e.g. ``"gemini-3.8-flash"`` or ``"openai/gpt-5.4-mini"``.

    Returns:
        ``"google"``, ``"anthropic"``, ``"openai"``, or ``"openrouter"``.

    Raises:
        ConfigurationError: If ``model`` is blank or the name cannot be mapped.
    """
    slug = str(model).strip().lower()
    if not slug:
        raise ConfigurationError(MISSING_MODEL_MESSAGE)
    if "/" in slug:
        org, _, rest = slug.partition("/")
        if rest and org in _NATIVE_ORG_PREFIXES:
            return _NATIVE_ORG_PREFIXES[org]
        return "openrouter"
    if "gemini" in slug:
        return "google"
    if "claude" in slug:
        return "anthropic"
    if "gpt" in slug or slug.startswith(_OPENAI_REASONING_PREFIXES):
        return "openai"
    raise ConfigurationError(
        f"Cannot infer LLM provider from model slug {model!r}. "
        "Pass provider= / --provider (openai, anthropic, google, or openrouter)."
    )


def _native_api_model(provider: str, model: str) -> str:
    """Strip ``openai/`` / ``google/`` / ``anthropic/`` when calling that native API."""
    if provider not in _NATIVE_ORG_PREFIXES:
        return model
    prefix = f"{provider}/"
    stripped = model.strip()
    if stripped.lower().startswith(prefix):
        remainder = stripped[len(prefix) :]
        if remainder:
            return remainder
    return model


def get_provider(
    provider: str | None = None,
    model: str | None = None,
    *,
    disable_thinking: bool = False,
) -> BaseLLM:
    """Construct a provider implementation by name.

    When ``provider`` is omitted, it is inferred from ``model`` via
    :func:`infer_provider`. An explicit provider always wins, including
    ``"openrouter"`` for a slug that also has a native API.

    Args:
        provider: Provider key — ``"openai"``, ``"anthropic"``, ``"google"``,
            or ``"openrouter"``. ``"gemini"`` is not accepted. ``None`` infers
            from ``model``.
        model: Required model slug (no default).
        disable_thinking: When ``True``, suppress extended reasoning /
            chain-of-thought. Currently only honoured by Gemini
            (sets ``thinkingConfig.thinkingBudget=0``).

    Returns:
        A :class:`~adam_identification.llm.base.BaseLLM` instance ready for
        async calls.

    Raises:
        ConfigurationError: If ``model`` is omitted, the provider cannot be
            inferred or is unknown, or the required API key is missing.
    """
    if not model or not str(model).strip():
        raise ConfigurationError(MISSING_MODEL_MESSAGE)
    if provider is None or not str(provider).strip():
        provider = infer_provider(model)
    api_model = _native_api_model(provider, model)
    if provider == "openai":
        from adam_identification.llm.providers.openai import OpenAIProvider

        return OpenAIProvider(model=api_model)
    if provider == "anthropic":
        from adam_identification.llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model=api_model)
    if provider == "google":
        from adam_identification.llm.providers.gemini import GeminiProvider

        return GeminiProvider(model=api_model, disable_thinking=disable_thinking)
    if provider == "openrouter":
        from adam_identification.llm.providers.openrouter import OpenRouterProvider

        return OpenRouterProvider(model=model)
    raise ConfigurationError(
        f"Unknown LLM provider: '{provider}'. Valid options: {', '.join(_VALID_PROVIDERS)}."
    )
