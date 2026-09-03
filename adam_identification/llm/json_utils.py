"""Utilities for parsing structured JSON from LLM message text.

Strips optional Markdown code fences (for example triple backticks with an
optional ``json`` language tag) before :func:`json.loads`, since models often
wrap JSON despite instructions to return raw objects only.

Pipeline role:
    Used by workflow components and agents that request JSON-shaped output and
    then validate with Pydantic or similar.

Cross-references: ``adam_identification.llm.base`` (``response_format="json"`` still benefits
    from fence stripping).
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


def _try_complete_json(text: str) -> str | None:
    """Attempt to close a truncated JSON object by appending missing braces/quotes.

    Handles the Gemini pattern where the model stops mid-JSON before the
    closing brace(s), producing an ``"Unterminated string"`` or ``"Extra data"``
    parse error.  Only called as a last-resort recovery after ``rfind("}")``
    slicing has failed.

    Args:
        text: Raw LLM output that may contain a truncated ``{...}`` object.

    Returns:
        A patched string that may be parseable as JSON, or ``None`` when no
        ``{`` is found in *text*.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for ch in text[start:]:
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
    if depth > 0:
        close = ('"' if in_string else "") + "}" * depth
        return text[start:] + close
    return None


def parse_llm_json_object(content: str) -> dict[str, Any]:
    """Parse a single JSON object from LLM output.

    Args:
        content: Raw model text (possibly wrapped in Markdown fences).

    Returns:
        The parsed JSON object as a ``dict``.

    Raises:
        json.JSONDecodeError: If the payload is not valid JSON after stripping.
        TypeError: If the top-level JSON value is not an object (``dict``).
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
