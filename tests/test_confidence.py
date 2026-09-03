"""Unit tests for identification confidence parsing."""

from __future__ import annotations

import pytest

from adam_identification._confidence import parse_confidence_level


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (1, 1),
        (5, 5),
        ("3", 3),
        ("low", 1),
        ("medium", 3),
        ("high", 5),
        (None, None),
        ("", None),
        (0, None),
        (6, None),
        (True, None),
    ],
)
def test_parse_confidence_level(raw: object, expected: int | None) -> None:
    assert parse_confidence_level(raw) == expected
