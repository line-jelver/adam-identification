"""OpenAI provider implementation for ADaM."""

from __future__ import annotations

from typing import Any, Literal

import httpx

from adam_identification._config import ConfigurationError, settings
from adam_identification.llm.base import BaseLLM, LLMResponse, Message, TokenUsage
from adam_identification.llm.exceptions import (
    InvalidResponseError,
    ProviderUnavailableError,
    exception_for_http_status,
)

_OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# OpenAI models that reject non-default explicit temperature (verified via live API).
_OPENAI_FIXED_TEMPERATURE_MODELS: frozenset[str] = frozenset({"gpt-5.5"})
_OPENAI_NO_EXPLICIT_TEMPERATURE_PREFIXES: tuple[str, ...] = ("gpt-5.5", "o1", "o3")


def openai_model_supports_explicit_temperature(model: str) -> bool:
    """Return whether the OpenAI API accepts arbitrary explicit ``temperature`` values.

    ``gpt-5.5`` only supports the default temperature (1.0); other ``gpt-5.4*``
    slugs such as ``gpt-5.4-mini`` accept explicit values (e.g. 0.0).

    Args:
        model: OpenAI model slug.

    Returns:
        ``False`` when the API is expected to ignore or reject explicit temperature.
    """
    model_lower = model.lower()
    if model_lower in _OPENAI_FIXED_TEMPERATURE_MODELS:
        return False
    blocked = _OPENAI_NO_EXPLICIT_TEMPERATURE_PREFIXES
    return not any(model_lower.startswith(prefix) for prefix in blocked)


class OpenAIProvider(BaseLLM):
    """Provider for OpenAI chat-completions models."""

    _http_provider_name = "OpenAI"

    def __init__(self, model: str = "gpt-5.4-nano") -> None:
        if not settings.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is not set. Add it to your .env file.")
        self._api_key = settings.openai_api_key
        self._model = model
        self._base_url = _OPENAI_URL
        self.provider = "openai"
        self.requested_model = model

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _messages_payload(self, messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    @staticmethod
    def _token_usage(payload: dict[str, Any]) -> TokenUsage:
        usage = payload.get("usage", {})
        return TokenUsage(
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
        )

    def _build_payload(
        self,
        messages: list[Message],
        *,
        response_format: Literal["text", "json"],
        temperature: float | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages_payload(messages),
            "max_completion_tokens": max_tokens,
        }
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        if temperature is not None and openai_model_supports_explicit_temperature(self._model):
            payload["temperature"] = temperature
        return payload

    async def _complete_once(
        self,
        messages: list[Message],
        *,
        response_format: Literal["text", "json"] = "text",
        temperature: float | None = None,
        max_tokens: int = 8000,
    ) -> LLMResponse:
        payload = self._build_payload(
            messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(self._base_url, headers=self._headers(), json=payload)
                response.raise_for_status()
                response_json = response.json()
        except httpx.HTTPStatusError as exc:
            raise exception_for_http_status(
                self._http_provider_name,
                exc.response.status_code,
                model=self._model,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"{self._http_provider_name} network error: {exc}"
            ) from exc

        choices = response_json.get("choices", [])
        if not choices:
            raise InvalidResponseError(f"{self._http_provider_name} response missing choices.")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise InvalidResponseError(f"{self._http_provider_name} returned empty content.")
        return LLMResponse(
            content=content,
            token_usage=self._token_usage(response_json),
            model=str(response_json.get("model", self._model)),
        )
