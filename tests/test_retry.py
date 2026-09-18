"""Unit tests for adam_identification.llm.retry."""

from __future__ import annotations

import pytest

from adam_identification.llm.exceptions import InvalidResponseError, RateLimitError
from adam_identification.llm.retry import retry_on_rate_limit


@pytest.mark.asyncio
async def test_retry_on_rate_limit_succeeds_after_backoff() -> None:
    """Second attempt succeeds after one RateLimitError."""
    calls = 0

    async def factory() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RateLimitError("Gemini rate limit exceeded.")
        return "ok"

    result = await retry_on_rate_limit(factory, max_attempts=3, delays_s=(0.0, 0.0))
    assert result.value == "ok"
    assert result.rate_limit_retries == 1
    assert calls == 2


@pytest.mark.asyncio
async def test_retry_on_rate_limit_raises_after_max_attempts() -> None:
    """Exhausted retries re-raise the last RateLimitError."""
    calls = 0

    async def factory() -> str:
        nonlocal calls
        calls += 1
        raise RateLimitError("quota")

    with pytest.raises(RateLimitError, match="quota"):
        await retry_on_rate_limit(factory, max_attempts=2, delays_s=(0.0,))

    assert calls == 2


@pytest.mark.asyncio
async def test_retry_on_rate_limit_does_not_catch_other_errors() -> None:
    """Non-rate-limit exceptions propagate immediately."""

    async def factory() -> str:
        raise ValueError("bad prompt")

    with pytest.raises(ValueError, match="bad prompt"):
        await retry_on_rate_limit(factory)


@pytest.mark.asyncio
async def test_retry_does_not_catch_invalid_response() -> None:
    """Unknown-model 404 must fail immediately, not back off as unavailable."""
    calls = 0

    async def factory() -> str:
        nonlocal calls
        calls += 1
        raise InvalidResponseError("Gemini could not find the requested model (HTTP 404).")

    with pytest.raises(InvalidResponseError, match="404"):
        await retry_on_rate_limit(factory, max_attempts=5, delays_s=(30.0, 60.0))

    assert calls == 1
