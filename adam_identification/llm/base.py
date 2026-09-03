"""Base interface and shared types for LLM providers.

All agent code should target this interface. Provider-specific modules live in
``adam_identification/llm/providers`` and are only instantiated through
:func:`~adam_identification.llm.get_provider`.

Usage:
    from adam_identification.llm import get_provider

    provider = get_provider("google", model="gemini-2.5-flash")
    response = await provider.complete_single("Identify silicon")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from adam_identification.llm.retry import retry_on_rate_limit


@dataclass(frozen=True)
class Message:
    """A single chat message.

    Attributes:
        role: Message role, usually ``system``, ``user``, or ``assistant``.
        content: Message text content.
    """

    role: str
    content: str


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response returned from all providers.

    Attributes:
        content: Response text.
        token_usage: Token accounting where available.
        model: Model used to produce the response.
    """

    content: str
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    rate_limit_retries: int = 0
    """Rate-limit backoff retries consumed before this response was obtained."""


class BaseLLM(ABC):
    """Abstract contract for LLM providers.

    Subclasses implement :meth:`_complete_once`; the public :meth:`complete`
    wraps each call with :func:`~adam_identification.llm.retry.retry_on_rate_limit`.
    """

    async def complete(
        self,
        messages: list[Message],
        *,
        response_format: Literal["text", "json"] = "text",
        temperature: float | None = None,
        max_tokens: int = 8000,
    ) -> LLMResponse:
        """Send messages with automatic backoff on provider rate limits."""

        async def _call() -> LLMResponse:
            return await self._complete_once(
                messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        outcome = await retry_on_rate_limit(_call)
        resp = outcome.value
        return LLMResponse(
            content=resp.content,
            token_usage=resp.token_usage,
            model=resp.model,
            rate_limit_retries=outcome.api_retries,
        )

    @abstractmethod
    async def _complete_once(
        self,
        messages: list[Message],
        *,
        response_format: Literal["text", "json"] = "text",
        temperature: float | None = None,
        max_tokens: int = 8000,
    ) -> LLMResponse:
        """Perform one provider API call (no rate-limit retry).

        Args:
            messages: Ordered conversation messages.
            response_format: ``text`` or ``json``.
            temperature: Optional sampling temperature.
            max_tokens: Maximum completion length.
        """

    async def complete_single(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        response_format: Literal["text", "json"] = "text",
        temperature: float | None = None,
        max_tokens: int = 8000,
    ) -> LLMResponse:
        """Convenience wrapper for one user prompt."""
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        return await self.complete(
            messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
