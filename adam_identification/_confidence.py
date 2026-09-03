"""Parse model-reported confidence levels from identification LLM JSON.

Identification prompts ask models for a confidence score on a 1 (low) to 5
(high) integer scale. Legacy runs used ``low`` / ``medium`` / ``high``
strings; this module normalises both formats.

Cross-references:
    - ``adam_identification.templates.identification.*`` — prompt schemas.
    - ``adam_identification._phase_lookup`` — selection parsers.
"""

from __future__ import annotations

from typing import Any

_LEGACY_CONFIDENCE_MAP: dict[str, int] = {
    "low": 1,
    "medium": 3,
    "high": 5,
}


def parse_confidence_level(value: Any) -> int | None:
    """Normalise a model confidence value to an integer in ``[1, 5]``.

    Args:
        value: Raw JSON field — integer, numeric string, or legacy label.

    Returns:
        Confidence level ``1`` (low) through ``5`` (high), or ``None`` if missing
        or unrecognised.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        level = int(value)
        return level if 1 <= level <= 5 else None
    text = str(value).strip().lower()
    if not text:
        return None
    if text.isdigit():
        level = int(text)
        return level if 1 <= level <= 5 else None
    return _LEGACY_CONFIDENCE_MAP.get(text)
