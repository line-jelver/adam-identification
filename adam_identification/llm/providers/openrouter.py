"""OpenRouter provider implementation for adam-identification."""

from __future__ import annotations

from adam_identification._config import ConfigurationError, settings
from adam_identification.llm.providers.openai import OpenAIProvider

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider using the OpenAI-compatible API surface.

    Provides access to 300+ models including DeepSeek, Kimi, Qwen, and others.
    See https://openrouter.ai/models for available model slugs.

    Args:
        model: OpenRouter model slug, e.g. ``"deepseek/deepseek-chat-v3-5"``.
    """

    def __init__(self, model: str = "deepseek/deepseek-chat-v3-5") -> None:
        if not settings.openrouter_api_key:
            raise ConfigurationError("OPENROUTER_API_KEY is not set. Add it to your .env file.")
        self._api_key = settings.openrouter_api_key
        self._model = model
        self._base_url = _OPENROUTER_URL

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        headers["HTTP-Referer"] = "https://github.com/line-jelver/adam-identification"
        headers["X-Title"] = "adam-identification"
        return headers
