"""Unit tests for confidence level parsing."""

from __future__ import annotations

import pytest

from adam_identification._confidence import parse_confidence_level


def test_integer_in_range() -> None:
    assert parse_confidence_level(3) == 3


def test_integer_out_of_range_returns_none() -> None:
    assert parse_confidence_level(0) is None
    assert parse_confidence_level(6) is None


def test_float_rounds_to_int() -> None:
    assert parse_confidence_level(4.9) == 4


def test_numeric_string() -> None:
    assert parse_confidence_level("5") == 5


def test_legacy_labels() -> None:
    assert parse_confidence_level("low") == 1
    assert parse_confidence_level("medium") == 3
    assert parse_confidence_level("high") == 5


def test_case_insensitive_legacy() -> None:
    assert parse_confidence_level("HIGH") == 5


def test_none_returns_none() -> None:
    assert parse_confidence_level(None) is None


def test_bool_returns_none() -> None:
    assert parse_confidence_level(True) is None


def test_empty_string_returns_none() -> None:
    assert parse_confidence_level("") is None


def test_unknown_string_returns_none() -> None:
    assert parse_confidence_level("excellent") is None
