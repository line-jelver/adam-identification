"""LLM provider factory for adam-identification.

Usage::

    from adam_identification.llm import get_provider

    llm = get_provider("google", model="gemini-2.5-flash")
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

__all__ = [
    "MISSING_MODEL_MESSAGE",
    "get_provider",
    "BaseLLM",
    "Message",
    "TokenUsage",
    "LLMResponse",
    "retry_on_rate_limit",
    "RetryResult",
    "DEFAULT_RATE_LIMIT_DELAYS_S",
]


def get_provider(
    provider: str,
    model: str | None = None,
    *,
    disable_thinking: bool = False,
) -> BaseLLM:
    """Construct a provider implementation by name.

    Args:
        provider: Provider key — ``"openai"``, ``"anthropic"``, ``"google"``,
            or ``"openrouter"``. ``"gemini"`` is not accepted.
        model: Required model slug (no default).
        disable_thinking: When ``True``, suppress extended reasoning /
            chain-of-thought. Currently only honoured by Gemini
            (sets ``thinkingConfig.thinkingBudget=0``).

    Returns:
        A :class:`~adam_identification.llm.base.BaseLLM` instance ready for
        async calls.

    Raises:
        ConfigurationError: If ``model`` is omitted, the provider name is
            unknown, or the required API key is missing.
    """
    if not model or not str(model).strip():
        raise ConfigurationError(MISSING_MODEL_MESSAGE)
    if provider == "openai":
        from adam_identification.llm.providers.openai import OpenAIProvider

        return OpenAIProvider(model=model)
    if provider == "anthropic":
        from adam_identification.llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model=model)
    if provider == "google":
        from adam_identification.llm.providers.gemini import GeminiProvider

        return GeminiProvider(model=model, disable_thinking=disable_thinking)
    if provider == "openrouter":
        from adam_identification.llm.providers.openrouter import OpenRouterProvider

        return OpenRouterProvider(model=model)
    raise ConfigurationError(
        f"Unknown LLM provider: '{provider}'. Valid options: openai, anthropic, google, openrouter."
    )
