"""Sync retry helpers for transient database API errors.

PubChem PUG REST can return ``PUGREST.ServerBusy`` under load. Clients use
these helpers with the same backoff schedule as the LLM retry layer.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import httpx

from adam_identification.exceptions import DatabaseAPIError
from adam_identification.llm.retry import DEFAULT_RATE_LIMIT_DELAYS_S

_T = TypeVar("_T")

logger = logging.getLogger(__name__)

DEFAULT_DB_MAX_ATTEMPTS = 5

PUBCHEM_TRANSIENT_MARKERS: tuple[str, ...] = (
    "serverbusy",
    "too many requests",
    "server too busy",
    "service unavailable",
    "timeout",
    "connection error",
    "network error",
    "http 503",
    "http 502",
    "http 504",
    "http 429",
)


@dataclass(frozen=True)
class DbRetryResult(Generic[_T]):
    """Outcome of a retried database API call.

    Attributes:
        value: Successful return value from the wrapped callable.
        transient_retries: Number of transient-error recoveries before success.
    """

    value: _T
    transient_retries: int = 0


def is_transient_database_error(exc: BaseException) -> bool:
    """Return ``True`` when *exc* looks like a retryable transport fault.

    Args:
        exc: The exception to classify.

    Returns:
        ``True`` for httpx network errors and known PubChem busy responses.
    """
    if isinstance(exc, httpx.RequestError):
        return True
    if isinstance(exc, DatabaseAPIError):
        message = str(exc).lower()
        return any(marker in message for marker in PUBCHEM_TRANSIENT_MARKERS)
    return False


def retry_on_transient_error(
    fn: Callable[[], _T],
    *,
    max_attempts: int = DEFAULT_DB_MAX_ATTEMPTS,
    delays_s: tuple[float, ...] = DEFAULT_RATE_LIMIT_DELAYS_S,
    label: str = "Database API",
) -> "DbRetryResult[_T]":
    """Call ``fn`` with exponential backoff on transient database errors.

    Args:
        fn: Zero-argument callable to invoke on each attempt.
        max_attempts: Total attempts before re-raising the last error.
        delays_s: Sleep durations before attempts 2..N.
        label: Short name for log messages, e.g. ``"PubChem"``.

    Returns:
        :class:`DbRetryResult` with the successful value and retry count.

    Raises:
        BaseException: Re-raises the last error when attempts are exhausted or
            the error is not classified as transient.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_exc: BaseException | None = None
    transient_retries = 0
    for attempt in range(1, max_attempts + 1):
        try:
            value = fn()
            return DbRetryResult(value=value, transient_retries=transient_retries)
        except BaseException as exc:
            if not is_transient_database_error(exc) or attempt >= max_attempts:
                raise
            last_exc = exc
            transient_retries += 1
            delay = delays_s[min(attempt - 1, len(delays_s) - 1)]
            logger.warning(
                "%s transient error (attempt %d/%d); retrying in %.0fs: %s",
                label,
                attempt,
                max_attempts,
                delay,
                exc,
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc
