"""Unit tests for ``MaterialsProjectCrystalProvider`` (mocked client; no live HTTP)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adam_identification.database.crystal_provider import CrystalCandidate, CrystalProvider
from adam_identification.database.materials_project import MaterialsProjectClient
from adam_identification.database.materials_project_provider import MaterialsProjectCrystalProvider
from adam_identification.exceptions import DatabaseAPIError, MaterialNotFoundError
from adam_identification.models import AtomicPosition, CrystalStructure, Material, MaterialSource


def _make_material(*, mp_id: str, formula: str = "Si", space_group: str = "Fd-3m") -> Material:
    return Material(
        name=formula,
        chemical_formula=formula,
        source=MaterialSource.MATERIALS_PROJECT,
        mp_id=mp_id,
        structure=CrystalStructure(
            unit_cell=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            lattice_parameters={
                "a": 1.0,
                "b": 1.0,
                "c": 1.0,
                "alpha": 90.0,
                "beta": 90.0,
                "gamma": 90.0,
            },
            atomic_positions=[AtomicPosition(element="Si", position=(0.0, 0.0, 0.0))],
            crystal_system="Cubic",
            space_group=space_group,
            nelements=1,
            nsites=1,
        ),
    )


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock(spec=MaterialsProjectClient)


class TestSearchCandidates:
    def test_maps_materials_to_candidates(self, mock_client: MagicMock) -> None:
        mock_client.search_by_formula.return_value = [_make_material(mp_id="mp-149")]

        provider = MaterialsProjectCrystalProvider(mock_client)
        result = provider.search_candidates("Si", max_results=10)

        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        assert candidate.source == MaterialSource.MATERIALS_PROJECT
        assert candidate.source_id == "mp-149"
        assert candidate.structure_ref == "mp-149"
        assert candidate.formula == "Si"
        assert candidate.space_group == "Fd-3m"
        assert candidate.crystal_system == "Cubic"
        assert candidate.nsites == 1
        mock_client.search_by_formula.assert_called_once_with("Si", max_results=10)

    def test_total_matches_and_truncated_are_unknown(self, mock_client: MagicMock) -> None:
        mock_client.search_by_formula.return_value = [_make_material(mp_id="mp-149")]

        provider = MaterialsProjectCrystalProvider(mock_client)
        result = provider.search_candidates("Si", max_results=10)

        assert result.total_matches is None
        assert result.truncated is False

    def test_no_results_returns_empty(self, mock_client: MagicMock) -> None:
        mock_client.search_by_formula.return_value = []

        provider = MaterialsProjectCrystalProvider(mock_client)
        result = provider.search_candidates("XxYz999", max_results=10)

        assert result.candidates == []

    def test_missing_mp_id_raises(self, mock_client: MagicMock) -> None:
        material = _make_material(mp_id="mp-149")
        object.__setattr__(material, "mp_id", None)
        mock_client.search_by_formula.return_value = [material]

        provider = MaterialsProjectCrystalProvider(mock_client)
        with pytest.raises(DatabaseAPIError, match="missing its mp_id"):
            provider.search_candidates("Si", max_results=10)


class TestHydrate:
    def test_returns_cached_material_without_new_request(self, mock_client: MagicMock) -> None:
        material = _make_material(mp_id="mp-149")
        mock_client.search_by_formula.return_value = [material]

        provider = MaterialsProjectCrystalProvider(mock_client)
        result = provider.search_candidates("Si", max_results=10)
        hydrated = provider.hydrate(result.candidates[0])

        assert hydrated is material
        mock_client.get_by_material_id.assert_not_called()

    def test_falls_back_to_get_by_id_when_not_cached(self, mock_client: MagicMock) -> None:
        material = _make_material(mp_id="mp-149")
        mock_client.get_by_material_id.return_value = material
        candidate = CrystalCandidate(
            source=MaterialSource.MATERIALS_PROJECT,
            source_id="mp-149",
            structure_ref="mp-149",
            formula="Si",
        )

        provider = MaterialsProjectCrystalProvider(mock_client)
        hydrated = provider.hydrate(candidate)

        assert hydrated is material
        mock_client.get_by_material_id.assert_called_once_with("mp-149")

    def test_rejects_foreign_provider_candidate(self, mock_client: MagicMock) -> None:
        candidate = CrystalCandidate(
            source=MaterialSource.MC3D,
            source_id="mc3d-18058",
            structure_ref="some-uuid",
            formula="TiO2",
        )

        provider = MaterialsProjectCrystalProvider(mock_client)
        with pytest.raises(DatabaseAPIError, match="Cannot hydrate"):
            provider.hydrate(candidate)
        mock_client.get_by_material_id.assert_not_called()

    def test_not_found_propagates(self, mock_client: MagicMock) -> None:
        mock_client.get_by_material_id.side_effect = MaterialNotFoundError("gone")
        candidate = CrystalCandidate(
            source=MaterialSource.MATERIALS_PROJECT,
            source_id="mp-does-not-exist",
            structure_ref="mp-does-not-exist",
            formula="Si",
        )

        provider = MaterialsProjectCrystalProvider(mock_client)
        with pytest.raises(MaterialNotFoundError):
            provider.hydrate(candidate)


class TestGetById:
    def test_delegates_and_caches(self, mock_client: MagicMock) -> None:
        material = _make_material(mp_id="mp-149")
        mock_client.get_by_material_id.return_value = material

        provider = MaterialsProjectCrystalProvider(mock_client)
        result = provider.get_by_id("mp-149")

        assert result is material
        mock_client.get_by_material_id.assert_called_once_with("mp-149")

        # Cached: a later hydrate() for the same ID makes no further request.
        candidate = CrystalCandidate(
            source=MaterialSource.MATERIALS_PROJECT,
            source_id="mp-149",
            structure_ref="mp-149",
            formula="Si",
        )
        provider.hydrate(candidate)
        mock_client.get_by_material_id.assert_called_once()


class TestProtocolConformance:
    def test_satisfies_crystal_provider_protocol(self, mock_client: MagicMock) -> None:
        provider = MaterialsProjectCrystalProvider(mock_client)
        assert isinstance(provider, CrystalProvider)

    def test_source_attribute(self, mock_client: MagicMock) -> None:
        provider = MaterialsProjectCrystalProvider(mock_client)
        assert provider.source == MaterialSource.MATERIALS_PROJECT
