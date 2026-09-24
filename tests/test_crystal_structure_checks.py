"""Unit tests for provider-neutral crystal-structure validation helpers."""

from __future__ import annotations

import pytest

from adam_identification.database.crystal_structure_checks import (
    assert_ordered_site,
    crystal_system_from_space_group_number,
)
from adam_identification.exceptions import DatabaseAPIError


class TestCrystalSystemFromSpaceGroupNumber:
    @pytest.mark.parametrize(
        ("number", "expected"),
        [
            (1, "Triclinic"),
            (2, "Triclinic"),
            (3, "Monoclinic"),
            (15, "Monoclinic"),
            (16, "Orthorhombic"),
            (74, "Orthorhombic"),
            (75, "Tetragonal"),
            (142, "Tetragonal"),
            (143, "Trigonal"),
            (167, "Trigonal"),
            (168, "Hexagonal"),
            (194, "Hexagonal"),
            (195, "Cubic"),
            (225, "Cubic"),
            (230, "Cubic"),
        ],
    )
    def test_boundaries(self, number: int, expected: str) -> None:
        assert crystal_system_from_space_group_number(number) == expected

    @pytest.mark.parametrize("number", [0, -1, 231, 1000])
    def test_out_of_range_raises(self, number: int) -> None:
        with pytest.raises(DatabaseAPIError):
            crystal_system_from_space_group_number(number)


class TestAssertOrderedSite:
    def test_ordered_site_returns_its_symbol(self) -> None:
        assert assert_ordered_site(["Ti"], [1.0], site_label="Ti") == "Ti"

    def test_mixed_occupancy_raises(self) -> None:
        with pytest.raises(DatabaseAPIError, match="disordered"):
            assert_ordered_site(["Ti", "O"], [0.5, 0.5], site_label="Ti/O")

    def test_partial_occupancy_raises(self) -> None:
        with pytest.raises(DatabaseAPIError, match="disordered"):
            assert_ordered_site(["Ti"], [0.8], site_label="Ti")

    def test_empty_site_raises(self) -> None:
        with pytest.raises(DatabaseAPIError, match="disordered"):
            assert_ordered_site([], [], site_label="empty")

    def test_error_message_includes_site_label(self) -> None:
        with pytest.raises(DatabaseAPIError, match="'my-site'"):
            assert_ordered_site([], [], site_label="my-site")
