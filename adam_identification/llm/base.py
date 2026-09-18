"""Base interface and shared types for LLM providers.

All agent code should target this interface. Provider-specific modules live in
``adam_identification/llm/providers`` and are only instantiated through
:func:`~adam_identification.llm.get_provider`.

Usage:
    from adam_identification.llm import get_provider

    provider = get_provider(model="gemini-2.5-flash")
    response = await provider.complete_single("Identify silicon")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
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

    provider: str = "unknown"
    requested_model: str = ""

    def _record_call(
        self,
        *,
        messages: list[Message],
        response_format: str,
        temperature: float | None,
        max_tokens: int,
        requested_at: datetime,
        completed_at: datetime,
        response: str | None,
        error: str | None,
        exc_type: str | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        provider_model: str | None,
        rate_limit_retries: int,
    ) -> None:
        """Best-effort append of one LLM call onto the active trace."""
        try:
            from adam_identification.provenance.context import (
                checkpoint,
                current_purpose,
                current_trace,
            )
            from adam_identification.provenance.trace import LLMCallRecord, LLMMessage

            trace = current_trace()
            if trace is None:
                return
            trace.record_llm_call(
                LLMCallRecord(
                    purpose=current_purpose(),
                    provider=self.provider,
                    requested_model=self.requested_model,
                    provider_model=provider_model or None,
                    requested_at=requested_at,
                    completed_at=completed_at,
                    messages=[LLMMessage(role=m.role, content=m.content) for m in messages],
                    response=response,
                    error=error,
                    exc_type=exc_type,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    rate_limit_retries=rate_limit_retries,
                )
            )
            checkpoint()
        except Exception:
            pass

    async def complete(
        self,
        messages: list[Message],
        *,
        response_format: Literal["text", "json"] = "text",
        temperature: float | None = None,
        max_tokens: int = 8000,
    ) -> LLMResponse:
        """Send messages with automatic backoff on provider rate limits.

        Each call is recorded on the active provenance trace when one is bound.
        """

        async def _call() -> LLMResponse:
            return await self._complete_once(
                messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        requested_at = datetime.now(UTC)
        rate_limit_retries = 0
        try:
            outcome = await retry_on_rate_limit(_call)
            rate_limit_retries = outcome.api_retries
            resp = outcome.value
            usage = resp.token_usage
            self._record_call(
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                requested_at=requested_at,
                completed_at=datetime.now(UTC),
                response=resp.content,
                error=None,
                exc_type=None,
                prompt_tokens=usage.prompt_tokens or None,
                completion_tokens=usage.completion_tokens or None,
                total_tokens=usage.total_tokens or None,
                provider_model=resp.model or None,
                rate_limit_retries=rate_limit_retries,
            )
            return LLMResponse(
                content=resp.content,
                token_usage=resp.token_usage,
                model=resp.model,
                rate_limit_retries=outcome.api_retries,
            )
        except Exception as exc:
            retries = getattr(exc, "api_retries", rate_limit_retries)
            if not isinstance(retries, int):
                retries = rate_limit_retries
            self._record_call(
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                requested_at=requested_at,
                completed_at=datetime.now(UTC),
                response=None,
                error=str(exc),
                exc_type=type(exc).__name__,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                provider_model=None,
                rate_limit_retries=retries,
            )
            raise

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
