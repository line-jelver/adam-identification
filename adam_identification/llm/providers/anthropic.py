"""Anthropic provider implementation for ADaM.

JSON mode appends the suffix from ``adam/prompts/templates/llm/anthropic_json_suffix.j2``
to the system field (Anthropic API constraint).
"""

from __future__ import annotations

from typing import Any, Literal

import httpx

from adam_identification._config import ConfigurationError, settings
from adam_identification._prompts import PromptLoader
from adam_identification.llm.base import BaseLLM, LLMResponse, Message, TokenUsage
from adam_identification.llm.exceptions import (
    AuthenticationError,
    InvalidResponseError,
    ProviderUnavailableError,
    RateLimitError,
)

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseLLM):
    """Provider for Anthropic Claude models."""

    def __init__(self, model: str = "claude-3-5-haiku-20241022") -> None:
        if not settings.anthropic_api_key:
            raise ConfigurationError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        self._api_key = settings.anthropic_api_key
        self._model = model
        self._base_url = _ANTHROPIC_URL
        self._json_mode_suffix = PromptLoader().render("llm/anthropic_json_suffix.j2")

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    @staticmethod
    def _split_messages(messages: list[Message]) -> tuple[str, list[dict[str, str]]]:
        system_prompt = ""
        api_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
                continue
            api_messages.append({"role": msg.role, "content": msg.content})
        return system_prompt, api_messages

    @staticmethod
    def _token_usage(payload: dict[str, Any]) -> TokenUsage:
        usage = payload.get("usage", {})
        prompt = int(usage.get("input_tokens", 0) or 0)
        completion = int(usage.get("output_tokens", 0) or 0)
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )

    async def _complete_once(
        self,
        messages: list[Message],
        *,
        response_format: Literal["text", "json"] = "text",
        temperature: float | None = None,
        max_tokens: int = 8000,
    ) -> LLMResponse:
        system_prompt, api_messages = self._split_messages(messages)
        if response_format == "json":
            system_prompt = (system_prompt + self._json_mode_suffix).strip()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(self._base_url, headers=self._headers(), json=payload)
                response.raise_for_status()
                response_json = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                raise RateLimitError("Anthropic rate limit exceeded.") from exc
            if status == 401:
                raise AuthenticationError("Anthropic authentication failed.") from exc
            if 500 <= status < 600:
                raise ProviderUnavailableError(
                    f"Anthropic service unavailable (HTTP {status})."
                ) from exc
            raise ProviderUnavailableError(f"Anthropic request failed (HTTP {status}).") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"Anthropic network error: {exc}") from exc

        content_blocks = response_json.get("content", [])
        if not content_blocks:
            raise InvalidResponseError("Anthropic response missing content.")
        first = content_blocks[0]
        text = first.get("text")
        if not isinstance(text, str) or not text.strip():
            raise InvalidResponseError("Anthropic returned empty content.")
        return LLMResponse(
            content=text,
            token_usage=self._token_usage(response_json),
            model=str(response_json.get("model", self._model)),
        )
