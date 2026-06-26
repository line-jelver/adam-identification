"""Utilities for parsing structured JSON from LLM message text.

Strips optional Markdown code fences before :func:`json.loads`, since models
often wrap JSON despite instructions to return raw objects only.
"""

from __future__ import annotations

import json
import re
from typing import Any, cast


def _extract_first_json_object(text: str) -> str | None:
    """Return the substring of the first balanced ``{...}`` object in *text*."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_llm_json_object(content: str) -> dict[str, Any]:
    """Parse a single JSON object from LLM output.

    Args:
        content: Raw model text (possibly wrapped in Markdown fences).

    Returns:
        The parsed JSON object as a ``dict``.

    Raises:
        json.JSONDecodeError: If the payload is not valid JSON after stripping.
        TypeError: If the top-level JSON value is not an object.
    """
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        extracted = _extract_first_json_object(text)
        if extracted is None:
            raise
        parsed = json.loads(extracted)
    if not isinstance(parsed, dict):
        msg = f"Expected JSON object, got {type(parsed).__name__}"
        raise TypeError(msg)
    return cast(dict[str, Any], parsed)
