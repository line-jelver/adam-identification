"""Unit tests for ``CrystalRetriever`` and ``CrystalSourcePolicy``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adam_identification.database.crystal_provider import CrystalCandidate, CrystalSearchResult
from adam_identification.database.crystal_retrieval import (
    CrystalRetriever,
    CrystalSourcePolicy,
    build_crystal_retriever,
    looks_like_explicit_crystal_id,
)
from adam_identification.exceptions import DatabaseAPIError, MaterialNotFoundError
from adam_identification.models import CrystalStructure, Material, MaterialSource


class _FakeProvider:
    """Minimal ``CrystalProvider`` double with scriptable search/hydrate."""

    def __init__(
        self,
        source: MaterialSource,
        *,
        candidates: list[CrystalCandidate] | None = None,
        search_error: Exception | None = None,
        material: Material | None = None,
    ) -> None:
        self.source = source
        self._candidates = candidates or []
        self._search_error = search_error
        self._material = material
        self.search_calls = 0
        self.hydrate_calls = 0
        self.get_by_id_calls = 0

    def search_candidates(self, formula: str, max_results: int) -> CrystalSearchResult:
        self.search_calls += 1
        if self._search_error is not None:
            raise self._search_error
        return CrystalSearchResult(candidates=self._candidates)

    def get_by_id(self, source_id: str) -> Material:
        self.get_by_id_calls += 1
        if self._material is None:
            raise MaterialNotFoundError(source_id)
        return self._material

    def hydrate(self, candidate: CrystalCandidate) -> Material:
        self.hydrate_calls += 1
        if self._material is None:
            raise MaterialNotFoundError(candidate.source_id)
        return self._material


def _candidate(
    source: MaterialSource,
    source_id: str,
    **kwargs: object,
) -> CrystalCandidate:
    return CrystalCandidate(
        source=source,
        source_id=source_id,
        structure_ref=source_id,
        formula="TiO2",
        **kwargs,  # type: ignore[arg-type]
    )


class TestConstruction:
    def test_empty_providers_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CrystalRetriever([])


class TestSearchAuto:
    def test_first_provider_used_when_it_has_candidates(self) -> None:
        mc3d = _FakeProvider(
            MaterialSource.MC3D, candidates=[_candidate(MaterialSource.MC3D, "mc3d-1")]
        )
        mp = _FakeProvider(MaterialSource.MATERIALS_PROJECT)
        retriever = CrystalRetriever([mc3d, mp])

        result = retriever.search("TiO2", 20)

        assert len(result.candidates) == 1
        assert mc3d.search_calls == 1
        assert mp.search_calls == 0

    def test_falls_back_on_empty_result(self) -> None:
        mc3d = _FakeProvider(MaterialSource.MC3D, candidates=[])
        mp_candidate = _candidate(MaterialSource.MATERIALS_PROJECT, "mp-149")
        mp = _FakeProvider(MaterialSource.MATERIALS_PROJECT, candidates=[mp_candidate])
        retriever = CrystalRetriever([mc3d, mp])

        result = retriever.search("TiO2", 20)

        assert result.candidates == [mp_candidate]
        assert mc3d.search_calls == 1
        assert mp.search_calls == 1

    def test_falls_back_on_provider_outage(self) -> None:
        mc3d = _FakeProvider(MaterialSource.MC3D, search_error=DatabaseAPIError("MC3D is down"))
        mp_candidate = _candidate(MaterialSource.MATERIALS_PROJECT, "mp-149")
        mp = _FakeProvider(MaterialSource.MATERIALS_PROJECT, candidates=[mp_candidate])
        retriever = CrystalRetriever([mc3d, mp])

        result = retriever.search("TiO2", 20)

        assert result.candidates == [mp_candidate]

    def test_all_providers_empty_returns_empty_result(self) -> None:
        mc3d = _FakeProvider(MaterialSource.MC3D, candidates=[])
        mp = _FakeProvider(MaterialSource.MATERIALS_PROJECT, candidates=[])
        retriever = CrystalRetriever([mc3d, mp])

        result = retriever.search("XxYz999", 20)

        assert result.candidates == []

    def test_all_providers_raise_reraises_last_error(self) -> None:
        mc3d = _FakeProvider(MaterialSource.MC3D, search_error=DatabaseAPIError("mc3d down"))
        mp = _FakeProvider(
            MaterialSource.MATERIALS_PROJECT, search_error=DatabaseAPIError("mp down")
        )
        retriever = CrystalRetriever([mc3d, mp])

        with pytest.raises(DatabaseAPIError, match="mp down"):
            retriever.search("TiO2", 20)


class TestSearchPinnedPolicy:
    def test_mc3d_policy_never_tries_mp(self) -> None:
        mc3d = _FakeProvider(MaterialSource.MC3D, candidates=[])
        mp = _FakeProvider(
            MaterialSource.MATERIALS_PROJECT,
            candidates=[_candidate(MaterialSource.MATERIALS_PROJECT, "mp-1")],
        )
        retriever = CrystalRetriever([mc3d, mp], policy=CrystalSourcePolicy.MC3D)

        result = retriever.search("TiO2", 20)

        assert result.candidates == []
        assert mp.search_calls == 0

    def test_pinned_policy_outage_propagates_without_fallback(self) -> None:
        mc3d = _FakeProvider(MaterialSource.MC3D, search_error=DatabaseAPIError("mc3d down"))
        mp = _FakeProvider(
            MaterialSource.MATERIALS_PROJECT,
            candidates=[_candidate(MaterialSource.MATERIALS_PROJECT, "mp-1")],
        )
        retriever = CrystalRetriever([mc3d, mp], policy=CrystalSourcePolicy.MC3D)

        with pytest.raises(DatabaseAPIError, match="mc3d down"):
            retriever.search("TiO2", 20)
        assert mp.search_calls == 0

    def test_policy_with_no_matching_provider_raises(self) -> None:
        mc3d = _FakeProvider(MaterialSource.MC3D)
        retriever = CrystalRetriever([mc3d], policy=CrystalSourcePolicy.MATERIALS_PROJECT)

        with pytest.raises(DatabaseAPIError, match="No crystal provider configured"):
            retriever.search("TiO2", 20)


class TestMc3dDeterministicOrdering:
    def test_reorders_experimental_before_theoretical(self) -> None:
        theoretical = _candidate(MaterialSource.MC3D, "mc3d-2", is_theoretical=True)
        experimental = _candidate(MaterialSource.MC3D, "mc3d-1", is_theoretical=False)
        mc3d = _FakeProvider(MaterialSource.MC3D, candidates=[theoretical, experimental])
        retriever = CrystalRetriever([mc3d])

        result = retriever.search("TiO2", 20)

        assert [c.source_id for c in result.candidates] == ["mc3d-1", "mc3d-2"]

    def test_reorders_ambient_before_high_pressure(self) -> None:
        high_pressure = _candidate(MaterialSource.MC3D, "mc3d-2", is_high_pressure=True)
        ambient = _candidate(MaterialSource.MC3D, "mc3d-1", is_high_pressure=False)
        mc3d = _FakeProvider(MaterialSource.MC3D, candidates=[high_pressure, ambient])
        retriever = CrystalRetriever([mc3d])

        result = retriever.search("TiO2", 20)

        assert [c.source_id for c in result.candidates] == ["mc3d-1", "mc3d-2"]

    def test_reorders_by_space_group_number_then_nsites_then_numeric_id(self) -> None:
        c_high_sg = _candidate(MaterialSource.MC3D, "mc3d-30", space_group_number=200)
        c_low_sg_high_id = _candidate(MaterialSource.MC3D, "mc3d-20", space_group_number=100)
        c_low_sg_low_id = _candidate(MaterialSource.MC3D, "mc3d-10", space_group_number=100)
        mc3d = _FakeProvider(
            MaterialSource.MC3D, candidates=[c_high_sg, c_low_sg_high_id, c_low_sg_low_id]
        )
        retriever = CrystalRetriever([mc3d])

        result = retriever.search("TiO2", 20)

        assert [c.source_id for c in result.candidates] == ["mc3d-10", "mc3d-20", "mc3d-30"]

    def test_not_sorted_by_energy(self) -> None:
        low_energy_theoretical = _candidate(
            MaterialSource.MC3D, "mc3d-2", is_theoretical=True, total_energy_ev_per_cell=-100.0
        )
        high_energy_experimental = _candidate(
            MaterialSource.MC3D, "mc3d-1", is_theoretical=False, total_energy_ev_per_cell=-1.0
        )
        mc3d = _FakeProvider(
            MaterialSource.MC3D, candidates=[low_energy_theoretical, high_energy_experimental]
        )
        retriever = CrystalRetriever([mc3d])

        result = retriever.search("TiO2", 20)

        # Experimental (mc3d-1) sorts first despite its much higher raw energy.
        assert [c.source_id for c in result.candidates] == ["mc3d-1", "mc3d-2"]

    def test_mp_candidates_are_not_reordered(self) -> None:
        candidates = [
            _candidate(MaterialSource.MATERIALS_PROJECT, "mp-2"),
            _candidate(MaterialSource.MATERIALS_PROJECT, "mp-1"),
        ]
        mp = _FakeProvider(MaterialSource.MATERIALS_PROJECT, candidates=candidates)
        retriever = CrystalRetriever([mp])

        result = retriever.search("TiO2", 20)

        assert [c.source_id for c in result.candidates] == ["mp-2", "mp-1"]


class TestHydrate:
    def test_delegates_to_matching_provider(self) -> None:
        material = Material(
            name="TiO2",
            chemical_formula="TiO2",
            source=MaterialSource.MC3D,
            mc3d_id="mc3d-1",
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
                atomic_positions=[],
                crystal_system="Cubic",
                space_group="Fm-3m",
                nelements=1,
                nsites=0,
            ),
        )
        mc3d = _FakeProvider(MaterialSource.MC3D, material=material)
        retriever = CrystalRetriever([mc3d])
        candidate = _candidate(MaterialSource.MC3D, "mc3d-1")

        hydrated = retriever.hydrate(candidate)

        assert hydrated is material
        assert mc3d.hydrate_calls == 1

    def test_no_provider_for_source_raises(self) -> None:
        mp = _FakeProvider(MaterialSource.MATERIALS_PROJECT)
        retriever = CrystalRetriever([mp])
        candidate = _candidate(MaterialSource.MC3D, "mc3d-1")

        with pytest.raises(DatabaseAPIError, match="No crystal provider configured"):
            retriever.hydrate(candidate)


class TestRetrievalStatus:
    def test_search_announces_the_provider(self) -> None:
        from adam_identification.database.retrieval_status import (
            bind_retrieval_status,
            reset_retrieval_status,
        )

        messages: list[str] = []
        token = bind_retrieval_status(messages.append)
        try:
            provider = _FakeProvider(MaterialSource.MC3D, candidates=[])
            CrystalRetriever([provider]).search("Si", 5)
        finally:
            reset_retrieval_status(token)
        assert messages == ["Searching MC3D…"]


class TestGetById:
    def test_routes_by_prefix(self) -> None:
        mc3d = _FakeProvider(MaterialSource.MC3D)
        mp = _FakeProvider(MaterialSource.MATERIALS_PROJECT)
        retriever = CrystalRetriever([mc3d, mp])

        with pytest.raises(MaterialNotFoundError):
            retriever.get_by_id("mc3d-18058")
        assert mc3d.get_by_id_calls == 1
        assert mp.get_by_id_calls == 0

    def test_unrecognized_prefix_raises_not_found(self) -> None:
        mc3d = _FakeProvider(MaterialSource.MC3D)
        retriever = CrystalRetriever([mc3d])

        with pytest.raises(MaterialNotFoundError, match="Unrecognized"):
            retriever.get_by_id("cod-12345")

    def test_conflicting_policy_rejects_id_before_calling_provider(self) -> None:
        mc3d = _FakeProvider(MaterialSource.MC3D)
        mp = _FakeProvider(MaterialSource.MATERIALS_PROJECT)
        retriever = CrystalRetriever([mc3d, mp], policy=CrystalSourcePolicy.MATERIALS_PROJECT)

        with pytest.raises(DatabaseAPIError, match="not available under crystal-source policy"):
            retriever.get_by_id("mc3d-18058")
        assert mc3d.get_by_id_calls == 0

    def test_no_configured_provider_for_id_source_raises(self) -> None:
        mc3d = _FakeProvider(MaterialSource.MC3D)
        retriever = CrystalRetriever([mc3d])

        with pytest.raises(DatabaseAPIError):
            retriever.get_by_id("mp-149")


class TestClose:
    def test_closes_every_provider_with_a_close_method(self) -> None:
        mc3d = _FakeProvider(MaterialSource.MC3D)
        mc3d.close = MagicMock()  # type: ignore[method-assign]
        mp = _FakeProvider(MaterialSource.MATERIALS_PROJECT)
        retriever = CrystalRetriever([mc3d, mp])

        retriever.close()

        mc3d.close.assert_called_once()

    def test_ignores_providers_without_close(self) -> None:
        """A provider double with no close() must not raise."""
        mp = _FakeProvider(MaterialSource.MATERIALS_PROJECT)
        retriever = CrystalRetriever([mp])

        retriever.close()  # no AttributeError


def _settings_with_mp_key(api_key: str | None):  # type: ignore[no-untyped-def]
    from dataclasses import replace

    from adam_identification._config import settings as live_settings

    return replace(live_settings, materials_project_api_key=api_key)


class TestBuildCrystalRetriever:
    """Unit tests for the ``--crystal-source``-token factory function."""

    def test_auto_with_key_orders_mc3d_then_mp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from adam_identification.database.materials_project_provider import (
            MaterialsProjectCrystalProvider,
        )
        from adam_identification.database.mc3d import MC3DClient

        monkeypatch.setattr(
            "adam_identification.database.crystal_retrieval.settings",
            _settings_with_mp_key("test-key"),
        )
        with patch(
            "adam_identification.database.crystal_retrieval.MaterialsProjectClient",
            return_value=MagicMock(),
        ):
            retriever = build_crystal_retriever(crystal_source="auto", mc3d_method="pbesol-v2")
        try:
            assert retriever.policy == "auto"
            assert isinstance(retriever._providers[0], MC3DClient)  # noqa: SLF001
            assert isinstance(retriever._providers[1], MaterialsProjectCrystalProvider)  # noqa: SLF001
        finally:
            retriever._providers[0].close()  # noqa: SLF001

    def test_mc3d_ignores_mp_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "adam_identification.database.crystal_retrieval.settings",
            _settings_with_mp_key("test-key"),
        )
        with patch(
            "adam_identification.database.crystal_retrieval.MaterialsProjectClient"
        ) as mock_mp:
            retriever = build_crystal_retriever(crystal_source="mc3d", mc3d_method="pbesol-v1")
        try:
            mock_mp.assert_not_called()
            assert retriever.policy == "mc3d"
            assert len(retriever._providers) == 1  # noqa: SLF001
            assert retriever._providers[0].method == "pbesol-v1"  # noqa: SLF001
        finally:
            retriever._providers[0].close()  # noqa: SLF001

    def test_materials_project_skips_mc3d(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from adam_identification.database.materials_project_provider import (
            MaterialsProjectCrystalProvider,
        )

        monkeypatch.setattr(
            "adam_identification.database.crystal_retrieval.settings",
            _settings_with_mp_key("test-key"),
        )
        with (
            patch(
                "adam_identification.database.crystal_retrieval.MaterialsProjectClient",
                return_value=MagicMock(),
            ) as mock_mp,
            patch("adam_identification.database.crystal_retrieval.MC3DClient") as mock_mc3d,
        ):
            retriever = build_crystal_retriever(
                crystal_source="materials-project",
                mc3d_method="pbesol-v2",
            )
        mock_mc3d.assert_not_called()
        mock_mp.assert_called_once_with(api_key="test-key")
        assert retriever.policy == "materials_project"
        assert isinstance(retriever._providers[0], MaterialsProjectCrystalProvider)  # noqa: SLF001

    def test_unknown_crystal_source_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown crystal source"):
            build_crystal_retriever(crystal_source="cod")

    def test_unknown_mc3d_method_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown MC3D method"):
            build_crystal_retriever(mc3d_method="lda")

    def test_materials_project_without_key_raises_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from adam_identification._config import ConfigurationError

        monkeypatch.setattr(
            "adam_identification.database.crystal_retrieval.settings",
            _settings_with_mp_key(None),
        )
        with pytest.raises(ConfigurationError, match="MATERIALS_PROJECT_API_KEY"):
            build_crystal_retriever(crystal_source="materials-project")


class TestLooksLikeExplicitCrystalId:
    @pytest.mark.parametrize(
        "text",
        ["mc3d-18058", "mp-149", "MC3D-1", " mp-149 ", "Mp-149"],
    )
    def test_matches(self, text: str) -> None:
        assert looks_like_explicit_crystal_id(text) is True

    @pytest.mark.parametrize(
        "text",
        ["TiO2", "silicon", "mp149", "mc3d-", "mp-abc", "rutile mp-149", ""],
    )
    def test_does_not_match(self, text: str) -> None:
        assert looks_like_explicit_crystal_id(text) is False
