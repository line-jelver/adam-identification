"""Google Gemini provider implementation for adam-identification."""

from __future__ import annotations

from typing import Any, Literal

import httpx

from adam_identification._config import ConfigurationError, settings
from adam_identification.llm.base import BaseLLM, LLMResponse, Message, TokenUsage
from adam_identification.exceptions import (
    AuthenticationError,
    InvalidResponseError,
    ProviderUnavailableError,
    RateLimitError,
)

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(BaseLLM):
    """Provider for Google Gemini models.

    Default sampling temperature is ``1.0`` (applied when ``temperature=None``),
    matching the Gemini API default and the effective default of other providers.
    Pass ``temperature=0.0`` explicitly for deterministic outputs.

    Args:
        model: Gemini model slug, e.g. ``"gemini-2.5-flash"``.
        disable_thinking: When ``True``, add
            ``thinkingConfig: {thinkingBudget: 0}`` to the generation config,
            suppressing extended reasoning traces. Only effective for
            thinking-capable models; silently ignored by others.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        disable_thinking: bool = False,
    ) -> None:
        if not settings.google_api_key:
            raise ConfigurationError("GOOGLE_API_KEY is not set. Add it to your .env file.")
        self._api_key = settings.google_api_key
        self._model = model
        self._disable_thinking = disable_thinking

    def _url(self) -> str:
        return f"{_GEMINI_BASE_URL}/{self._model}:generateContent"

    def _auth_headers(self) -> dict[str, str]:
        # Pass API key as a header so it never appears in URL logs.
        return {"x-goog-api-key": self._api_key}

    @staticmethod
    def _to_payload_messages(
        messages: list[Message],
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        contents: list[dict[str, Any]] = []
        system_parts: list[dict[str, str]] = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append({"text": msg.content})
                continue
            role = "model" if msg.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg.content}]})
        return contents, system_parts

    @staticmethod
    def _token_usage(payload: dict[str, Any]) -> TokenUsage:
        usage = payload.get("usageMetadata", {})
        return TokenUsage(
            prompt_tokens=int(usage.get("promptTokenCount", 0) or 0),
            completion_tokens=int(usage.get("candidatesTokenCount", 0) or 0),
            total_tokens=int(usage.get("totalTokenCount", 0) or 0),
        )

    async def _complete_once(
        self,
        messages: list[Message],
        *,
        response_format: Literal["text", "json"] = "text",
        temperature: float | None = None,
        max_tokens: int = 8000,
    ) -> LLMResponse:
        contents, system_parts = self._to_payload_messages(messages)
        generation_config: dict[str, Any] = {
            "temperature": temperature if temperature is not None else 1.0,
            "maxOutputTokens": max_tokens,
        }
        if response_format == "json":
            generation_config["responseMimeType"] = "application/json"
        if self._disable_thinking:
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    self._url(), json=payload, headers=self._auth_headers()
                )
                response.raise_for_status()
                response_json = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                raise RateLimitError("Gemini rate limit exceeded.") from exc
            if status == 401:
                raise AuthenticationError("Gemini authentication failed.") from exc
            if 500 <= status < 600:
                raise ProviderUnavailableError(
                    f"Gemini service unavailable (HTTP {status})."
                ) from exc
            raise ProviderUnavailableError(f"Gemini request failed (HTTP {status}).") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"Gemini network error: {exc}") from exc

        candidates = response_json.get("candidates", [])
        if not candidates:
            if "promptFeedback" in response_json:
                reason = response_json["promptFeedback"].get("blockReason", "UNKNOWN")
                raise InvalidResponseError(f"Gemini blocked response: {reason}")
            raise InvalidResponseError("Gemini response missing candidates.")

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise InvalidResponseError("Gemini response missing text parts.")
        text = parts[0].get("text")
        if not isinstance(text, str) or not text.strip():
            raise InvalidResponseError("Gemini returned empty content.")
        return LLMResponse(
            content=text,
            token_usage=self._token_usage(response_json),
            model=self._model,
        )
