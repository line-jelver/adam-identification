"""Async retry helpers for LLM API calls.

Providers raise :class:`~adam_identification.exceptions.RateLimitError` on
HTTP 429 / quota exhaustion. Use :func:`retry_on_rate_limit` to back off and
retry automatically.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from adam_identification.exceptions import RateLimitError

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_RATE_LIMIT_DELAYS_S: tuple[float, ...] = (30.0, 60.0, 120.0, 120.0)


@dataclass(frozen=True)
class RetryResult[T]:
    """Outcome of a retried async call with rate-limit retry accounting.

    Attributes:
        value: Successful return value from the wrapped coroutine factory.
        rate_limit_retries: Number of :class:`RateLimitError` recoveries before
            success (0 when the first attempt succeeded).
    """

    value: T
    rate_limit_retries: int = 0


async def retry_on_rate_limit[T](
    coro_factory: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 5,
    delays_s: tuple[float, ...] = DEFAULT_RATE_LIMIT_DELAYS_S,
) -> RetryResult[T]:
    """Call ``coro_factory`` with exponential backoff on :class:`RateLimitError`.

    Args:
        coro_factory: Zero-argument callable returning the awaitable to run.
            Invoked fresh on each attempt.
        max_attempts: Total attempts before re-raising the last
            :class:`RateLimitError`.
        delays_s: Sleep duration before attempts 2..N.

    Returns:
        :class:`RetryResult` with the successful value and the number of
        rate-limit retries consumed.

    Raises:
        RateLimitError: When all attempts are exhausted.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_exc: RateLimitError | None = None
    rate_limit_retries = 0
    for attempt in range(1, max_attempts + 1):
        try:
            value = await coro_factory()
            return RetryResult(value=value, rate_limit_retries=rate_limit_retries)
        except RateLimitError as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            rate_limit_retries += 1
            delay = delays_s[min(attempt - 1, len(delays_s) - 1)]
            logger.warning(
                "Rate limit hit (attempt %d/%d); retrying in %.0fs",
                attempt,
                max_attempts,
                delay,
            )
            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc
