"""Unit tests for ``MC3DClient`` and its private wire-mapping helpers.

Uses ``httpx.MockTransport`` against the checked-in fixtures in
``tests/fixtures/mc3d/`` — no live HTTP. See that directory's ``README.md``
for what each fixture represents and the assertion discipline it permits.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from adam_identification.database.crystal_provider import CrystalCandidate
from adam_identification.database.mc3d import MC3DClient, MC3DMethod
from adam_identification.database.mc3d_models import (
    cartesian_to_fractional,
    lattice_parameters,
    to_optimade_reduced_formula,
    validate_species_at_sites,
)
from adam_identification.database.retry import (
    HTTP_RETRYABLE_STATUS_CODES,
    is_transient_database_error,
)
from adam_identification.exceptions import (
    DatabaseAPIError,
    DatabaseHTTPError,
    MaterialNotFoundError,
)
from adam_identification.models import CrystalStructure, MaterialSource, MC3DProperties

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "mc3d"


def _load(name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((FIXTURES_DIR / name).read_text()))


@pytest.fixture(autouse=True)
def _disable_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid real backoff delays when transient-retry paths are exercised."""
    monkeypatch.setattr("adam_identification.database.retry.time.sleep", lambda _seconds: None)


class _RecordingHandler:
    """A ``httpx.MockTransport`` handler that records every request it serves."""

    def __init__(self, responder: Callable[[httpx.Request], httpx.Response]) -> None:
        self._responder = responder
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responder(request)

    @property
    def urls(self) -> list[str]:
        return [str(r.url) for r in self.requests]


def _client(
    responder: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> tuple[MC3DClient, _RecordingHandler]:
    handler = _RecordingHandler(responder)
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, follow_redirects=False)
    return MC3DClient(MC3DMethod.PBESOL_V2, client=http_client, **kwargs), handler


# ── Formula canonicalization ────────────────────────────────────────────────


class TestToOptimadeReducedFormula:
    def test_binary_reorders_and_keeps_counts(self) -> None:
        assert to_optimade_reduced_formula("TiO2") == "O2Ti"

    def test_already_alphabetical_and_reduced(self) -> None:
        assert to_optimade_reduced_formula("Fe2O3") == "Fe2O3"

    def test_unary_formula(self) -> None:
        assert to_optimade_reduced_formula("Si") == "Si"

    def test_reduces_by_gcd(self) -> None:
        # Al2O4 shares a GCD of 2 -> reduced empirical formula AlO2.
        assert to_optimade_reduced_formula("Al2O4") == "AlO2"

    def test_alphabetical_ordering_ignores_input_order(self) -> None:
        assert to_optimade_reduced_formula("NaCl") == "ClNa"

    def test_fractional_input_is_rejected(self) -> None:
        with pytest.raises(DatabaseAPIError, match="parse"):
            to_optimade_reduced_formula("Ti0.5O2")

    def test_empty_formula_is_rejected(self) -> None:
        with pytest.raises(DatabaseAPIError):
            to_optimade_reduced_formula("   ")

    def test_unparseable_formula_is_rejected(self) -> None:
        with pytest.raises(DatabaseAPIError):
            to_optimade_reduced_formula("123")


# ── Cartesian <-> fractional conversion ─────────────────────────────────────
# Space-group-to-crystal-system mapping is provider-neutral and tested in
# test_crystal_structure_checks.py alongside the shared occupancy check.


class TestCartesianToFractional:
    def test_orthogonal_cube(self) -> None:
        lattice = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
        cartesian = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        result = cartesian_to_fractional(cartesian, lattice)
        assert result[0] == pytest.approx((0.0, 0.0, 0.0))
        assert result[1] == pytest.approx((0.5, 0.5, 0.5))

    def test_non_orthogonal_fixture_round_trips(self) -> None:
        """Uses the live-captured FCC primitive cell fixture (60 degree angles)."""
        payload = _load("hydrate_mc3d-19249_nonorthogonal.json")
        attrs = payload["data"]["attributes"]
        result = cartesian_to_fractional(
            attrs["cartesian_site_positions"], attrs["lattice_vectors"]
        )
        assert len(result) == 3
        # Ti sits at an analytically-exact lattice point, but its Cartesian
        # position is stored as tiny (~1e-57) floating-point noise around
        # zero -- must snap to (0, 0, 0), not wrap to ~(1, 1, 0).
        assert result[0] == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
        assert result[1] == pytest.approx((0.75, 0.75, 0.75), abs=1e-6)
        assert result[2] == pytest.approx((0.25, 0.25, 0.25), abs=1e-6)

    def test_singular_lattice_raises(self) -> None:
        lattice = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        with pytest.raises(DatabaseAPIError, match="singular"):
            cartesian_to_fractional([[0.0, 0.0, 0.0]], lattice)

    def test_non_finite_lattice_raises(self) -> None:
        lattice = [[float("nan"), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        with pytest.raises(DatabaseAPIError, match="finite"):
            cartesian_to_fractional([[0.0, 0.0, 0.0]], lattice)

    def test_wraps_into_unit_cell(self) -> None:
        lattice = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        cartesian = [[-0.5, 1.5, 2.0]]
        result = cartesian_to_fractional(cartesian, lattice)
        x, y, z = result[0]
        assert 0.0 <= x < 1.0
        assert 0.0 <= y < 1.0
        assert 0.0 <= z < 1.0


class TestLatticeParameters:
    def test_cubic_lattice_has_right_angles(self) -> None:
        params = lattice_parameters([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        assert params["a"] == pytest.approx(2.0)
        assert params["alpha"] == pytest.approx(90.0)
        assert params["gamma"] == pytest.approx(90.0)

    def test_non_orthogonal_fixture_gives_60_degree_angles(self) -> None:
        payload = _load("hydrate_mc3d-19249_nonorthogonal.json")
        vectors = payload["data"]["attributes"]["lattice_vectors"]
        params = lattice_parameters(vectors)
        assert params["alpha"] == pytest.approx(60.0, abs=0.1)
        assert params["beta"] == pytest.approx(60.0, abs=0.1)
        assert params["gamma"] == pytest.approx(60.0, abs=0.1)


# ── Species/site validation ─────────────────────────────────────────────────


class TestValidateSpeciesAtSites:
    _SPECIES = [
        {"name": "Ti", "chemical_symbols": ["Ti"], "concentration": [1.0]},
        {"name": "O", "chemical_symbols": ["O"], "concentration": [1.0]},
    ]

    def test_maps_each_site_to_its_element(self) -> None:
        mapping = validate_species_at_sites(["Ti", "O", "O"], self._SPECIES)
        assert mapping == {"Ti": "Ti", "O": "O"}

    def test_undefined_species_raises(self) -> None:
        with pytest.raises(DatabaseAPIError, match="undefined species"):
            validate_species_at_sites(["Ti", "N"], self._SPECIES)

    def test_disordered_site_raises(self) -> None:
        disordered = [
            {"name": "Ti/O", "chemical_symbols": ["Ti", "O"], "concentration": [0.5, 0.5]}
        ]
        with pytest.raises(DatabaseAPIError, match="disordered"):
            validate_species_at_sites(["Ti/O"], disordered)

    def test_partial_occupancy_raises(self) -> None:
        vacancy = [{"name": "Ti", "chemical_symbols": ["Ti"], "concentration": [0.8]}]
        with pytest.raises(DatabaseAPIError, match="disordered"):
            validate_species_at_sites(["Ti"], vacancy)

    def test_duplicate_species_name_raises(self) -> None:
        dupes = [*self._SPECIES, {"name": "Ti", "chemical_symbols": ["Ti"], "concentration": [1.0]}]
        with pytest.raises(DatabaseAPIError, match="Duplicate"):
            validate_species_at_sites(["Ti"], dupes)


# ── MC3DMethod / constructor validation ─────────────────────────────────────


class TestMC3DMethodValidation:
    def test_valid_method_strings_accepted(self) -> None:
        client, _ = _client(lambda r: httpx.Response(200, json={}))
        assert client.method == MC3DMethod.PBESOL_V2

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="pbesol-v3|not a valid"):
            MC3DClient("pbesol-v3")

    def test_non_positive_cache_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_core_cache_entries"):
            MC3DClient(MC3DMethod.PBE_V1, max_core_cache_entries=0)

    def test_default_method_is_pbe_v1(self) -> None:
        """Conservative default -- pbesol-v2 must be requested explicitly."""
        client = MC3DClient()
        assert client.method == MC3DMethod.PBE_V1
        client.close()


# ── search_candidates ───────────────────────────────────────────────────────


def _core_base_response(mc3d_id: str) -> httpx.Response:
    if mc3d_id == "mc3d-19249":
        return httpx.Response(200, json=_load("core_base_mc3d-19249.json"))
    return httpx.Response(404, json=_load("core_base_404_body.json"))


class TestSearchCandidates:
    def test_sends_encoded_reduced_formula_and_bounded_page_limit(self) -> None:
        search_payload = _load("formula_search_tio2_page1.json")

        def responder(request: httpx.Request) -> httpx.Response:
            if "/structures?" in str(request.url):
                return httpx.Response(200, json=search_payload)
            mc3d_id = str(request.url).rsplit("/", 1)[-1]
            return _core_base_response(mc3d_id)

        client, handler = _client(responder)
        result = client.search_candidates("TiO2", max_results=5)

        search_request = next(r for r in handler.requests if "/structures?" in str(r.url))
        assert 'chemical_formula_reduced="O2Ti"' in str(search_request.url) or (
            "chemical_formula_reduced%3D%22O2Ti%22" in str(search_request.url)
        )
        assert "page_limit=5" in str(search_request.url)

        # Only the well-formed candidate (mc3d-19249) survives; the other four
        # rows 404 on core_base and must be discarded, not raised.
        assert len(result.candidates) == 1
        assert result.candidates[0].source_id == "mc3d-19249"
        assert result.candidates[0].bravais_lattice == "cF"
        assert result.candidates[0].is_theoretical is True
        # core_base's "ICSD" must be normalized to lowercase for consistency
        # with the OPTIMADE _mcloud_source_db attribute ("icsd").
        assert result.candidates[0].source_database == "icsd"
        assert result.total_matches == 20
        assert result.truncated is True

        # Exactly one search request + one core_base request per returned row.
        assert sum("/structures?" in u for u in handler.urls) == 1
        assert sum("/core_base/" in u for u in handler.urls) == 5
        assert not any("/overview" in u for u in handler.urls)
        assert not any("structure-uuids" in u for u in handler.urls)

    def test_no_results_formula_returns_empty_bounded_result(self) -> None:
        client, _ = _client(
            lambda r: httpx.Response(200, json=_load("formula_search_no_results.json"))
        )
        result = client.search_candidates("ArHe", max_results=5)
        assert result.candidates == []
        assert result.total_matches == 0
        assert result.truncated is False

    def test_non_positive_max_results_makes_no_request(self) -> None:
        client, handler = _client(lambda r: httpx.Response(500, text="should not be called"))
        result = client.search_candidates("TiO2", max_results=0)
        assert result.candidates == []
        assert handler.requests == []

    def test_wide_pass_reuses_cached_core_records(self) -> None:
        """A re-search must not refetch a *successfully* cached core_base row.

        Rows whose ``core_base`` lookup 404s are never cached (there is
        nothing valid to cache), so they are legitimately refetched on a
        later wide pass -- only the one well-formed row (mc3d-19249) must
        not be refetched.
        """
        search_payload = _load("formula_search_tio2_page1.json")

        def responder(request: httpx.Request) -> httpx.Response:
            if "/structures?" in str(request.url):
                return httpx.Response(200, json=search_payload)
            mc3d_id = str(request.url).rsplit("/", 1)[-1]
            return _core_base_response(mc3d_id)

        client, handler = _client(responder)
        client.search_candidates("TiO2", max_results=5)
        first_valid_calls = sum("/core_base/mc3d-19249" in u for u in handler.urls)
        client.search_candidates("TiO2", max_results=5)
        second_valid_calls = sum("/core_base/mc3d-19249" in u for u in handler.urls)
        assert first_valid_calls == 1
        assert second_valid_calls == 1


# ── hydrate() / get_by_id() ──────────────────────────────────────────────────


def _tio2_candidate() -> CrystalCandidate:
    return CrystalCandidate(
        source=MaterialSource.MC3D,
        source_id="mc3d-19249",
        structure_ref="38e42253-a0b6-499e-adea-2a17e81866b9",
        formula="TiO2",
    )


class TestHydrate:
    def _client_for_hydration(self) -> tuple[MC3DClient, _RecordingHandler]:
        core_payload = _load("core_base_mc3d-19249.json")
        hydrate_payload = _load("hydrate_mc3d-19249_nonorthogonal.json")

        def responder(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/core_base/" in url:
                return httpx.Response(200, json=core_payload)
            return httpx.Response(200, json=hydrate_payload)

        return _client(responder)

    def test_hydrate_maps_full_material(self) -> None:
        client, handler = self._client_for_hydration()
        material = client.hydrate(_tio2_candidate())

        assert material.mc3d_id == "mc3d-19249"
        assert material.source == MaterialSource.MC3D
        assert material.chemical_formula == "TiO2"
        structure = material.structure
        assert isinstance(structure, CrystalStructure)
        assert structure.crystal_system == "Cubic"
        assert structure.space_group == "Fm-3m"
        assert structure.nsites == 3
        assert len(structure.atomic_positions) == 3

        props = material.get_properties(MaterialSource.MC3D)
        assert isinstance(props, MC3DProperties)
        assert props.method == "pbesol-v2"
        assert props.structure_uuid == "38e42253-a0b6-499e-adea-2a17e81866b9"
        # core_base returns "ICSD" (uppercase); the client normalizes casing.
        assert props.sources[0].database == "icsd"
        assert props.sources[0].is_theoretical is True
        assert props.provenance_links[0].label == "Final relax calculation"
        assert props.provenance_links[0].url.startswith(
            "https://aiida.materialscloud.org/mc3d-pbesol-v2/api/v4/nodes/"
        )
        assert props.record_url == "https://mc3d.materialscloud.org/?id=mc3d-19249"

        # Exactly one structures request for the hydration.
        assert sum("/structures/" in u for u in handler.urls) == 1

    def test_hydrate_caches_by_structure_uuid(self) -> None:
        client, handler = self._client_for_hydration()
        client.hydrate(_tio2_candidate())
        first_structure_calls = sum("/structures/" in u for u in handler.urls)
        client.hydrate(_tio2_candidate())
        second_structure_calls = (
            sum("/structures/" in u for u in handler.urls) - first_structure_calls
        )
        assert second_structure_calls == 0

    def test_hydrate_rejects_foreign_provider_candidate(self) -> None:
        client, handler = _client(lambda r: httpx.Response(500, text="unused"))
        foreign = CrystalCandidate(
            source=MaterialSource.MATERIALS_PROJECT,
            source_id="mp-149",
            structure_ref="mp-149",
            formula="Si",
        )
        with pytest.raises(DatabaseAPIError, match="Cannot hydrate"):
            client.hydrate(foreign)
        assert handler.requests == []

    def test_get_by_id_fetches_core_then_hydrates(self) -> None:
        client, handler = self._client_for_hydration()
        material = client.get_by_id("mc3d-19249")
        assert material.mc3d_id == "mc3d-19249"
        assert sum("/core_base/" in u for u in handler.urls) == 1
        assert sum("/structures/" in u for u in handler.urls) == 1

    def test_get_by_id_missing_entry_raises_material_not_found(self) -> None:
        client, _ = _client(lambda r: httpx.Response(404, json=_load("core_base_404_body.json")))
        with pytest.raises(MaterialNotFoundError):
            client.get_by_id("mc3d-99999999")

    def test_get_by_id_empty_id_raises_without_request(self) -> None:
        client, handler = _client(lambda r: httpx.Response(500, text="unused"))
        with pytest.raises(MaterialNotFoundError):
            client.get_by_id("   ")
        assert handler.requests == []

    def test_malformed_structure_is_rejected_before_material_construction(self) -> None:
        core_payload = _load("core_base_mc3d-19249.json")
        malformed_payload = _load("hydrate_malformed_structure.json")

        def responder(request: httpx.Request) -> httpx.Response:
            if "/core_base/" in str(request.url):
                return httpx.Response(200, json=core_payload)
            return httpx.Response(200, json=malformed_payload)

        client, _ = _client(responder)
        with pytest.raises(DatabaseAPIError, match="site-count mismatch"):
            client.hydrate(_tio2_candidate())


# ── Transient/HTTP error handling ───────────────────────────────────────────


class TestErrorHandling:
    def test_transient_503_is_retried_then_succeeds(self) -> None:
        core_payload = _load("core_base_mc3d-19249.json")
        attempts = {"n": 0}

        def responder(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] < 3:
                return httpx.Response(503, json=_load("transient_503_body.json"))
            return httpx.Response(200, json=core_payload)

        client, handler = _client(responder)
        core = client._get_core_record("mc3d-19249")  # noqa: SLF001 - exercising retry directly
        assert core.id == "mc3d-19249"
        assert attempts["n"] == 3
        assert client.transient_retries == 2

    def test_core_base_404_is_not_retried(self) -> None:
        client, handler = _client(
            lambda r: httpx.Response(404, json=_load("core_base_404_body.json"))
        )
        with pytest.raises(MaterialNotFoundError):
            client.get_by_id("mc3d-99999999")
        assert len(handler.requests) == 1

    def test_persistent_503_exhausts_retries_and_raises_typed_http_error(self) -> None:
        client, handler = _client(
            lambda r: httpx.Response(503, json=_load("transient_503_body.json"))
        )
        with pytest.raises(DatabaseHTTPError) as exc_info:
            client.get_by_id("mc3d-19249")
        assert exc_info.value.status_code == 503
        assert exc_info.value.provider == "MC3D"
        assert len(handler.requests) == 5  # DEFAULT_DB_MAX_ATTEMPTS

    def test_redirect_is_rejected_not_followed(self) -> None:
        client, handler = _client(
            lambda r: httpx.Response(302, headers={"location": "https://evil.example.com"})
        )
        with pytest.raises(DatabaseAPIError, match="redirect"):
            client.get_by_id("mc3d-19249")
        assert len(handler.requests) == 1

    def test_oversized_response_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("adam_identification.database.mc3d._MAX_RESPONSE_BYTES", 10)
        client, _ = _client(lambda r: httpx.Response(200, json=_load("core_base_mc3d-19249.json")))
        with pytest.raises(DatabaseAPIError, match="byte cap"):
            client.get_by_id("mc3d-19249")

    def test_non_json_body_raises_database_api_error(self) -> None:
        client, _ = _client(lambda r: httpx.Response(200, text="not json"))
        with pytest.raises(DatabaseAPIError, match="non-JSON"):
            client.get_by_id("mc3d-19249")


class TestIsTransientDatabaseError:
    @pytest.mark.parametrize("status_code", sorted(HTTP_RETRYABLE_STATUS_CODES))
    def test_retryable_status_codes(self, status_code: int) -> None:
        error = DatabaseHTTPError(
            "boom", provider="MC3D", operation="test", status_code=status_code
        )
        assert is_transient_database_error(error) is True

    @pytest.mark.parametrize("status_code", [400, 404, 422])
    def test_non_retryable_status_codes(self, status_code: int) -> None:
        error = DatabaseHTTPError(
            "boom", provider="MC3D", operation="test", status_code=status_code
        )
        assert is_transient_database_error(error) is False

    def test_unknown_status_code_is_not_retryable(self) -> None:
        error = DatabaseHTTPError("boom", provider="MC3D", operation="test", status_code=None)
        assert is_transient_database_error(error) is False
