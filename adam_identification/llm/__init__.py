"""LLM provider factory for adam-identification.

Usage::

    from adam_identification.llm import get_provider

    llm = get_provider("gemini", model="gemini-2.5-flash")
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

__all__ = [
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
        provider: Provider key — ``"openai"``, ``"anthropic"``, ``"gemini"``,
            or ``"openrouter"``.
        model: Model slug. Falls back to each provider's default when ``None``.
        disable_thinking: When ``True``, suppress extended reasoning /
            chain-of-thought. Currently only honoured by Gemini
            (sets ``thinkingConfig.thinkingBudget=0``).

    Returns:
        A :class:`~adam_identification.llm.base.BaseLLM` instance ready for
        async calls.

    Raises:
        ConfigurationError: If the provider name is unknown or the required
            API key is missing.
    """
    if provider == "openai":
        from adam_identification.llm.providers.openai import OpenAIProvider

        return OpenAIProvider(model=model) if model else OpenAIProvider()
    if provider == "anthropic":
        from adam_identification.llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model=model) if model else AnthropicProvider()
    if provider == "gemini":
        from adam_identification.llm.providers.gemini import GeminiProvider

        return (
            GeminiProvider(model=model, disable_thinking=disable_thinking)
            if model
            else GeminiProvider(disable_thinking=disable_thinking)
        )
    if provider == "openrouter":
        from adam_identification.llm.providers.openrouter import OpenRouterProvider

        return OpenRouterProvider(model=model) if model else OpenRouterProvider()
    raise ConfigurationError(
        f"Unknown LLM provider: '{provider}'. "
        "Valid options: openai, anthropic, gemini, openrouter."
    )
