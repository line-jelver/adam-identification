"""Async retry helpers for LLM API calls.

Providers raise :class:`~adam_identification.llm.exceptions.RateLimitError` on HTTP 429 /
quota exhaustion, or :class:`~adam_identification.llm.exceptions.ProviderUnavailableError` on
HTTP 5xx / transient network errors.  Callers use these helpers to back off
and retry instead of recording a permanent failure.

Pipeline position:
    Used by LLM provider ``complete()`` methods.

Cross-references:
    - ``adam_identification.llm.exceptions.RateLimitError`` — retried exception type.
    - ``adam_identification.llm.exceptions.ProviderUnavailableError`` — also retried.
    - ``adam_identification.llm`` — public re-export of ``retry_on_rate_limit``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from adam_identification.llm.exceptions import ProviderUnavailableError, RateLimitError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Both rate-limit (429) and provider unavailable (5xx / network) are transient.
_RETRYABLE: tuple[type[RateLimitError] | type[ProviderUnavailableError], ...] = (
    RateLimitError,
    ProviderUnavailableError,
)


@dataclass(frozen=True)
class RetryResult(Generic[T]):
    """Outcome of a retried async call with API retry accounting.

    Attributes:
        value: Successful return value from the wrapped coroutine factory.
        api_retries: Number of retried transient errors (:class:`RateLimitError`
            or :class:`ProviderUnavailableError`) before success (0 when the
            first attempt succeeded).
        rate_limit_retries: Backward-compatible alias for ``api_retries``.
    """

    value: T
    api_retries: int = 0

    @property
    def rate_limit_retries(self) -> int:
        """Backward-compatible alias for ``api_retries``."""
        return self.api_retries


# Default backoff schedule for benchmark-scale sweeps (seconds).
DEFAULT_RATE_LIMIT_DELAYS_S: tuple[float, ...] = (30.0, 60.0, 120.0, 120.0)


async def retry_on_rate_limit(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 5,
    delays_s: tuple[float, ...] = DEFAULT_RATE_LIMIT_DELAYS_S,
) -> RetryResult[T]:
    """Call ``coro_factory`` with exponential backoff on transient LLM errors.

    Retries on both :class:`~adam_identification.llm.exceptions.RateLimitError` (HTTP 429)
    and :class:`~adam_identification.llm.exceptions.ProviderUnavailableError` (HTTP 5xx or
    network errors).

    Args:
        coro_factory: Zero-argument callable returning the awaitable to run.
            Invoked fresh on each attempt so callers can rebuild state.
        max_attempts: Total attempts before re-raising the last exception.
        delays_s: Sleep duration before attempts 2..N.  When fewer delays are
            configured than retries needed, the last delay is reused.

    Returns:
        :class:`RetryResult` with the successful value and the number of
        transient retries consumed.

    Raises:
        RateLimitError: When all attempts are exhausted due to rate limits.
        ProviderUnavailableError: When all attempts are exhausted due to
            service unavailability.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_exc: RateLimitError | ProviderUnavailableError | None = None
    api_retries = 0
    for attempt in range(1, max_attempts + 1):
        try:
            value = await coro_factory()
            return RetryResult(value=value, api_retries=api_retries)
        except _RETRYABLE as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            api_retries += 1
            delay = delays_s[min(attempt - 1, len(delays_s) - 1)]
            logger.warning(
                "%s (attempt %d/%d); retrying in %.0fs: %s",
                type(exc).__name__,
                attempt,
                max_attempts,
                delay,
                exc,
            )
            await asyncio.sleep(delay)

    assert last_exc is not None
    last_exc.api_retries = api_retries
    raise last_exc
