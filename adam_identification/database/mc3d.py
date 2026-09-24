"""MC3D (Materials Cloud Three-Dimensional Structure Database) API client for ADaM.

Implements :class:`~adam_identification.database.crystal_provider.CrystalProvider` against
MC3D's public, unauthenticated OPTIMADE and MCXD-metadata HTTP APIs. No API
key required.

Request budget (never exceeded per call):
    - :meth:`MC3DClient.search_candidates` — one OPTIMADE search request plus
      up to ``max_results`` MCXD ``core_base`` requests (one per returned
      row; cached, so a later search does not refetch a row already seen).
    - :meth:`MC3DClient.hydrate` / :meth:`MC3DClient.get_by_id` — one OPTIMADE
      single-structure request (plus one ``core_base`` request if not
      already cached).
    - Never calls the ``/overview`` or ``/structure-uuids`` endpoints, and
      never requests an unbounded listing.

Cross-references:
    - ``adam_identification.database.crystal_provider`` — ``CrystalCandidate``, ``CrystalProvider``.
    - ``adam_identification.database.mc3d_models`` — private wire models and mapping helpers.
    - ``adam_identification.database.crystal_structure_checks`` — shared, database-neutral
      structure-validation helpers used while mapping a response.
    - ``adam_identification.exceptions`` — ``DatabaseHTTPError``, ``MaterialNotFoundError``.
    - ``adam_identification.database.retry`` — status-code-based transient retry policy.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from enum import StrEnum
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

import httpx
from pydantic import ValidationError

from adam_identification.database.crystal_provider import CrystalCandidate, CrystalSearchResult
from adam_identification.database.crystal_structure_checks import (
    crystal_system_from_space_group_number,
)
from adam_identification.database.mc3d_models import (
    CoreBaseProvenanceLink,
    CoreBaseRecord,
    OptimadeHydrateResponse,
    OptimadeSearchResponse,
    OptimadeStructureAttributes,
    OptimadeStructureResource,
    cartesian_to_fractional,
    lattice_parameters,
    to_optimade_reduced_formula,
    validate_species_at_sites,
)
from adam_identification.database.retry import retry_on_transient_error
from adam_identification.exceptions import (
    DatabaseAPIError,
    DatabaseHTTPError,
    MaterialNotFoundError,
)
from adam_identification.models import (
    AtomicPosition,
    CrystalStructure,
    Material,
    MaterialSource,
    MC3DProperties,
    MC3DProvenanceLink,
    MC3DSourceRecord,
)

logger = logging.getLogger(__name__)

try:
    _ADAM_VERSION = _pkg_version("adam-identification")
except PackageNotFoundError:  # pragma: no cover - editable/dev install edge case
    _ADAM_VERSION = "0.0.0"

_USER_AGENT = f"ADaM/{_ADAM_VERSION} MC3DClient (+https://github.com/line-jelver/ADaM)"

_OPTIMADE_ROOT = "https://optimade.materialscloud.org/main"
_MCXD_ROOT = "https://mcxd-api.materialscloud.org/mc3d"
_AIIDA_ROOT = "https://aiida.materialscloud.org"

#: Decoded-body cap for a single metadata or structure response. MC3D's
#: bounded, single-structure/single-record endpoints never legitimately
#: return anything close to this size; it exists only as a defensive ceiling
#: against an anomalously large response.
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024

_SEARCH_RESPONSE_FIELDS = (
    "chemical_formula_reduced,nsites,_mcloud_mc3d_id,_mcloud_source_db,"
    "_mcloud_source_db_id,_mcloud_total_energy,_mcloud_cell_volume,"
    "_mcloud_total_magnetization,_mcloud_absolute_magnetization"
)

_HYDRATE_RESPONSE_FIELDS = (
    "chemical_formula_reduced,lattice_vectors,cartesian_site_positions,species,"
    "species_at_sites,nsites,structure_features,_mcloud_mc3d_id,_mcloud_source_db,"
    "_mcloud_source_db_id,_mcloud_total_energy,_mcloud_cell_volume,"
    "_mcloud_total_magnetization,_mcloud_absolute_magnetization"
)

#: PBEsol-v2 labels its final relaxation link "Final relax calculation";
#: PBE-v1 labels the equivalent link "Final SCF calculation". Both are
#: checked for here, rather than hard-coding one method's naming.
_FINAL_CALCULATION_LABELS = ("Final relax calculation", "Final SCF calculation")


class MC3DMethod(StrEnum):
    """Allowed MC3D methodology identifiers.

    The only valid values are ``"pbe-v1"``, ``"pbesol-v1"``, and
    ``"pbesol-v2"``. Constructing this enum from any other string raises
    ``ValueError``, which serves as local validation before a URL is built.
    """

    PBE_V1 = "pbe-v1"
    PBESOL_V1 = "pbesol-v1"
    PBESOL_V2 = "pbesol-v2"


class MC3DClient:
    """Client for Materials Cloud's MC3D structure database.

    Implements :class:`~adam_identification.database.crystal_provider.CrystalProvider`
    against MC3D's public OPTIMADE and MCXD-metadata REST APIs.

    Args:
        method: MC3D methodology to query. Defaults to
            :attr:`MC3DMethod.PBE_V1` — the long-established, already-
            qualified method. Materials computed with the newer
            ``pbesol-v2`` methodology have not yet had their pseudopotential
            compatibility and relaxation behavior qualified elsewhere in
            ADaM, so callers that intentionally target PBEsol-v2 must pass
            ``method=MC3DMethod.PBESOL_V2`` explicitly rather than relying
            on a default that could silently change later.
        client: Optional pre-built ``httpx.Client`` (e.g. for tests with a
            mocked transport). If omitted, this instance creates and owns
            one, with redirects disabled unconditionally — MC3D's JSON
            endpoints have no legitimate reason to redirect at all.
        timeout_s: Total read/write/pool timeout in seconds. The connect
            timeout is fixed at 5 seconds regardless of this value.
        max_core_cache_entries: Bounded LRU size for both the ``core_base``
            record cache and the hydrated-``Material`` cache. The cache is
            per-instance and does not persist beyond this object's lifetime.

    Usage::

        with MC3DClient(method=MC3DMethod.PBESOL_V2) as client:
            result = client.search_candidates("TiO2", max_results=20)
            material = client.hydrate(result.candidates[0])
    """

    source: MaterialSource = MaterialSource.MC3D

    def __init__(
        self,
        method: MC3DMethod | str = MC3DMethod.PBE_V1,
        *,
        client: httpx.Client | None = None,
        timeout_s: float = 20.0,
        max_core_cache_entries: int = 100,
    ) -> None:
        self.method = MC3DMethod(method)
        self._optimade_base = f"{_OPTIMADE_ROOT}/mc3d-{self.method.value}/v1"
        self._mcxd_base = f"{_MCXD_ROOT}/{self.method.value}"

        if max_core_cache_entries < 1:
            raise ValueError("max_core_cache_entries must be >= 1")
        self._max_cache_entries = max_core_cache_entries

        if client is not None:
            self._client = client
        else:
            timeout = httpx.Timeout(connect=5.0, read=timeout_s, write=timeout_s, pool=timeout_s)
            self._client = httpx.Client(
                timeout=timeout,
                follow_redirects=False,
                headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
            )
        self._owns_client = client is None

        self._core_cache: OrderedDict[str, CoreBaseRecord] = OrderedDict()
        self._material_cache: OrderedDict[str, Material] = OrderedDict()
        self.transient_retries = 0
        """Transient MC3D retries consumed since client construction."""

    def close(self) -> None:
        """Close the underlying HTTP client if this instance created it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> MC3DClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ── CrystalProvider interface ───────────────────────────────────────────

    def search_candidates(self, formula: str, max_results: int) -> CrystalSearchResult:
        """Return up to ``max_results`` bounded, metadata-only candidates for ``formula``.

        Sends exactly one bounded OPTIMADE search request, then fetches one
        ``core_base`` metadata record per returned row. Never fetches full
        geometry and never requests an unbounded listing.

        Args:
            formula: Chemical formula in any common notation, e.g. ``"TiO2"``.
            max_results: Maximum number of candidates to request and return.

        Returns:
            Bounded :class:`~adam_identification.database.crystal_provider.CrystalSearchResult`.
            A row whose ``core_base`` record is missing or malformed is
            discarded (logged at WARNING), not shown to the caller and not
            raised as a whole-search failure.

        Raises:
            DatabaseAPIError: On a malformed formula, or a transport/schema
                failure for the OPTIMADE search call itself.
        """
        if max_results <= 0:
            return CrystalSearchResult(candidates=[])

        reduced_formula = to_optimade_reduced_formula(formula)
        search_response = self._optimade_search(reduced_formula, max_results)

        candidates: list[CrystalCandidate] = []
        for row in search_response.data:
            mc3d_id = row.attributes.mcloud_mc3d_id
            if not mc3d_id:
                logger.warning("MC3D search row %s has no _mcloud_mc3d_id; discarding.", row.id)
                continue
            try:
                core = self._get_core_record(mc3d_id)
            except (DatabaseAPIError, MaterialNotFoundError) as error:
                logger.warning("Discarding MC3D candidate %s: %s", mc3d_id, error)
                continue
            candidates.append(self._candidate_from_row(mc3d_id, row.attributes, core))

        return CrystalSearchResult(
            candidates=candidates,
            total_matches=search_response.meta.data_returned,
            truncated=search_response.meta.data_returned > len(candidates),
        )

    def get_by_id(self, source_id: str) -> Material:
        """Fetch one MC3D material directly by its ``mc3d-*`` ID.

        Used by the explicit-ID fast path (e.g. a query is exactly
        ``"mc3d-18058"``), bypassing formula search and LLM phase selection
        entirely.

        Args:
            source_id: MC3D ID, e.g. ``"mc3d-18058"``.

        Returns:
            A mapped ``Material`` instance.

        Raises:
            MaterialNotFoundError: If ``source_id`` does not exist in MC3D.
            DatabaseAPIError: On transport or malformed-response failures.
        """
        normalized = source_id.strip()
        if not normalized:
            raise MaterialNotFoundError("Empty MC3D ID was provided.")
        core = self._get_core_record(normalized)
        return self._hydrate_by_core_record(normalized, core)

    def hydrate(self, candidate: CrystalCandidate) -> Material:
        """Return the full ``Material`` for a previously-searched candidate.

        Performs exactly one OPTIMADE single-structure request (skipped
        entirely if this structure was already hydrated by a prior call on
        this instance).

        Args:
            candidate: A candidate previously returned by
                :meth:`search_candidates`.

        Returns:
            The hydrated ``Material``.

        Raises:
            DatabaseAPIError: If ``candidate`` was not sourced from MC3D, or
                the selected structure fails a consistency/validation check.
            MaterialNotFoundError: If the candidate's record no longer exists.
        """
        if candidate.source != self.source:
            raise DatabaseAPIError(
                f"Cannot hydrate a {candidate.source!r} candidate with MC3DClient "
                f"(source={self.source!r})."
            )
        core = self._get_core_record(candidate.source_id)
        return self._hydrate_by_core_record(candidate.source_id, core)

    # ── OPTIMADE search ─────────────────────────────────────────────────────

    def _optimade_search(self, reduced_formula: str, max_results: int) -> OptimadeSearchResponse:
        params = {
            "filter": f'chemical_formula_reduced="{reduced_formula}"',
            "page_limit": str(max_results),
            "response_fields": _SEARCH_RESPONSE_FIELDS,
        }
        payload = self._get_json(
            f"{self._optimade_base}/structures",
            params=params,
            operation="search_candidates",
        )
        try:
            return OptimadeSearchResponse.model_validate(payload)
        except ValidationError as error:
            raise DatabaseAPIError(
                f"Malformed MC3D OPTIMADE search response for formula {reduced_formula!r}: {error}"
            ) from error

    # ── core_base metadata (cached) ─────────────────────────────────────────

    def _get_core_record(self, mc3d_id: str) -> CoreBaseRecord:
        cached = self._core_cache.get(mc3d_id)
        if cached is not None:
            self._core_cache.move_to_end(mc3d_id)
            return cached

        payload = self._get_json(
            f"{self._mcxd_base}/core_base/{mc3d_id}",
            operation="core_base",
            not_found_message=f"MC3D entry {mc3d_id!r} was not found.",
        )
        try:
            record = CoreBaseRecord.model_validate(payload)
        except ValidationError as error:
            raise DatabaseAPIError(
                f"Malformed MC3D core_base record for {mc3d_id!r}: {error}"
            ) from error
        if record.id != mc3d_id:
            raise DatabaseAPIError(
                f"MC3D core_base id mismatch: requested {mc3d_id!r}, got {record.id!r}."
            )

        self._cache_put(self._core_cache, mc3d_id, record)
        return record

    def _candidate_from_row(
        self,
        mc3d_id: str,
        attributes: OptimadeStructureAttributes,
        core: CoreBaseRecord,
    ) -> CrystalCandidate:
        formula = core.general.formula or attributes.chemical_formula_reduced or mc3d_id
        space_group_number = core.general.spacegroup_number
        crystal_system = (
            crystal_system_from_space_group_number(space_group_number)
            if space_group_number is not None
            else None
        )
        primary_source = core.source[0] if core.source else None
        info = primary_source.info if primary_source else None

        # Live responses are inconsistent in casing: core_base.source[].database
        # is "ICSD" while the OPTIMADE _mcloud_source_db attribute is "icsd" for
        # the same record. Normalize to lowercase everywhere.
        return CrystalCandidate(
            source=MaterialSource.MC3D,
            source_id=mc3d_id,
            structure_ref=core.general.structure_uuid,
            formula=formula,
            method=self.method.value,
            space_group=core.general.spacegroup_international,
            space_group_number=space_group_number,
            crystal_system=crystal_system,
            bravais_lattice=core.general.bravais_lattice,
            nsites=attributes.nsites,
            source_database=(
                primary_source.database.lower()
                if primary_source
                else (attributes.mcloud_source_db.lower() if attributes.mcloud_source_db else None)
            ),
            source_database_id=(primary_source.id if primary_source else None)
            or attributes.mcloud_source_db_id,
            is_theoretical=info.is_theoretical if info else None,
            is_high_pressure=info.is_high_pressure if info else None,
            is_high_temperature=info.is_high_temperature if info else None,
            total_energy_ev_per_cell=(
                core.properties.total_energy.value
                if core.properties.total_energy
                else attributes.mcloud_total_energy
            ),
            cell_volume_ang3=(
                core.properties.cell_volume.value
                if core.properties.cell_volume
                else attributes.mcloud_cell_volume
            ),
        )

    # ── Selected-structure hydration (cached) ───────────────────────────────

    def _hydrate_by_core_record(self, mc3d_id: str, core: CoreBaseRecord) -> Material:
        structure_uuid = core.general.structure_uuid
        cached = self._material_cache.get(structure_uuid)
        if cached is not None:
            self._material_cache.move_to_end(structure_uuid)
            return cached

        payload = self._get_json(
            f"{self._optimade_base}/structures/{structure_uuid}",
            params={"response_fields": _HYDRATE_RESPONSE_FIELDS},
            operation="hydrate",
            not_found_message=f"MC3D structure {structure_uuid!r} was not found.",
        )
        try:
            response = OptimadeHydrateResponse.model_validate(payload)
        except ValidationError as error:
            raise DatabaseAPIError(
                f"Malformed MC3D hydration response for {mc3d_id!r}: {error}"
            ) from error

        material = self._map_hydrated_structure(mc3d_id, core, response.data)
        self._cache_put(self._material_cache, structure_uuid, material)
        return material

    def _map_hydrated_structure(
        self,
        mc3d_id: str,
        core: CoreBaseRecord,
        resource: OptimadeStructureResource,
    ) -> Material:
        """Validate and map one hydrated OPTIMADE resource into ``Material``.

        Runs every consistency check below (numbered inline) before
        constructing a ``Material``, so a malformed or inconsistent
        response never reaches downstream code.
        """
        attrs = resource.attributes
        structure_uuid = core.general.structure_uuid

        # Check 1: top-level OPTIMADE id equals the core record's structure UUID.
        if resource.id != structure_uuid:
            raise DatabaseAPIError(
                f"MC3D hydration id mismatch for {mc3d_id!r}: expected structure_uuid "
                f"{structure_uuid!r}, got {resource.id!r}."
            )
        # Check 2: _mcloud_mc3d_id matches the requested candidate/entry ID.
        if attrs.mcloud_mc3d_id != mc3d_id:
            raise DatabaseAPIError(
                f"MC3D hydration mc3d_id mismatch: requested {mc3d_id!r}, "
                f"got {attrs.mcloud_mc3d_id!r}."
            )
        # Check 7: reject disorder/assemblies/implicit atoms/site attachments.
        if attrs.structure_features:
            raise DatabaseAPIError(
                f"MC3D structure {mc3d_id!r} has unsupported structure_features "
                f"{attrs.structure_features}; disordered/assembled structures are "
                "not supported."
            )
        if attrs.lattice_vectors is None or attrs.cartesian_site_positions is None:
            raise DatabaseAPIError(f"MC3D structure {mc3d_id!r} is missing geometry fields.")

        nsites = attrs.nsites
        species_at_sites = attrs.species_at_sites or []
        species = attrs.species or []
        # Check 4: site-count agreement.
        if nsites is None or not (
            len(attrs.cartesian_site_positions) == len(species_at_sites) == nsites
        ):
            raise DatabaseAPIError(
                f"MC3D structure {mc3d_id!r} site-count mismatch: nsites={nsites}, "
                f"positions={len(attrs.cartesian_site_positions)}, "
                f"species_at_sites={len(species_at_sites)}."
            )
        # Checks 5-6: every site name resolves to exactly one ordered species.
        name_to_element = validate_species_at_sites(species_at_sites, species)

        # Check 8: OPTIMADE source id agrees with at least one MCXD source record.
        if attrs.mcloud_source_db_id is not None and core.source:
            known_ids = {s.id for s in core.source}
            if attrs.mcloud_source_db_id not in known_ids:
                raise DatabaseAPIError(
                    f"MC3D structure {mc3d_id!r} OPTIMADE source id "
                    f"{attrs.mcloud_source_db_id!r} does not match any MCXD source record."
                )

        # Check 3 (finite, non-singular lattice) is enforced inside this call.
        fractional_positions = cartesian_to_fractional(
            attrs.cartesian_site_positions, attrs.lattice_vectors
        )
        atomic_positions = [
            AtomicPosition(element=name_to_element[name], position=position)
            for name, position in zip(species_at_sites, fractional_positions, strict=True)
        ]

        space_group_number = core.general.spacegroup_number
        if space_group_number is None:
            raise DatabaseAPIError(f"MC3D structure {mc3d_id!r} is missing a space group number.")

        crystal_structure = CrystalStructure(
            unit_cell=[[float(v) for v in row] for row in attrs.lattice_vectors],
            lattice_parameters=lattice_parameters(attrs.lattice_vectors),
            atomic_positions=atomic_positions,
            crystal_system=crystal_system_from_space_group_number(space_group_number),
            space_group=core.general.spacegroup_international or "",
            nelements=len(species),
            nsites=nsites,
        )

        formula = core.general.formula or attrs.chemical_formula_reduced or mc3d_id
        properties = self._mc3d_properties(mc3d_id, core, structure_uuid)

        return Material(
            name=formula,
            chemical_formula=formula,
            source=MaterialSource.MC3D,
            mc3d_id=mc3d_id,
            structure=crystal_structure,
            properties=[properties],
        )

    def _mc3d_properties(
        self, mc3d_id: str, core: CoreBaseRecord, structure_uuid: str
    ) -> MC3DProperties:
        provenance_links = [self._map_provenance_link(link) for link in core.provenance_links]
        final_calc = next(
            (link for link in core.provenance_links if link.label in _FINAL_CALCULATION_LABELS),
            None,
        )
        return MC3DProperties(
            method=self.method.value,
            structure_uuid=structure_uuid,
            sources=[
                MC3DSourceRecord(
                    database=s.database.lower(),
                    record_id=s.id,
                    version=s.version,
                    is_theoretical=s.info.is_theoretical if s.info else None,
                    is_high_pressure=s.info.is_high_pressure if s.info else None,
                    is_high_temperature=s.info.is_high_temperature if s.info else None,
                )
                for s in core.source
            ],
            total_energy_ev_per_cell=(
                core.properties.total_energy.value if core.properties.total_energy else None
            ),
            cell_volume_ang3=(
                core.properties.cell_volume.value if core.properties.cell_volume else None
            ),
            total_magnetization_mu_b_per_cell=(
                core.properties.total_magnetization.value
                if core.properties.total_magnetization
                else None
            ),
            absolute_magnetization_mu_b_per_cell=(
                core.properties.absolute_magnetization.value
                if core.properties.absolute_magnetization
                else None
            ),
            final_scf_uuid=final_calc.uuid if final_calc else None,
            final_structure_uuid=structure_uuid,
            provenance_links=provenance_links,
            record_url=f"https://mc3d.materialscloud.org/?id={mc3d_id}",
            aiida_node_url=(
                self._aiida_node_url(final_calc.uuid) if final_calc is not None else None
            ),
        )

    def _map_provenance_link(self, link: CoreBaseProvenanceLink) -> MC3DProvenanceLink:
        return MC3DProvenanceLink(
            label=link.label, uuid=link.uuid, url=self._aiida_node_url(link.uuid)
        )

    def _aiida_node_url(self, node_uuid: str) -> str:
        return f"{_AIIDA_ROOT}/mc3d-{self.method.value}/api/v4/nodes/{node_uuid}"

    # ── Bounded LRU cache helper ─────────────────────────────────────────────

    def _cache_put(self, cache: OrderedDict[str, Any], key: str, value: Any) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self._max_cache_entries:
            cache.popitem(last=False)

    # ── HTTP transport (retry + status mapping + bounded streaming) ────────

    def _get_json(
        self,
        url: str,
        *,
        operation: str,
        params: dict[str, str] | None = None,
        not_found_message: str | None = None,
    ) -> Any:
        def _once() -> Any:
            return self._get_json_once(
                url, params=params, operation=operation, not_found_message=not_found_message
            )

        outcome = retry_on_transient_error(_once, label=f"MC3D:{operation}")
        self.transient_retries += outcome.transient_retries
        return outcome.value

    def _get_json_once(
        self,
        url: str,
        *,
        operation: str,
        params: dict[str, str] | None,
        not_found_message: str | None,
    ) -> Any:
        try:
            body, status_code = self._stream_bounded(url, params)
        except httpx.RequestError as error:
            raise DatabaseAPIError(
                f"MC3D network error during {operation} for {url}: {error}"
            ) from error

        if status_code == 404 and not_found_message is not None:
            raise MaterialNotFoundError(not_found_message)

        if status_code >= 400:
            snippet = body.decode("utf-8", errors="replace")[:500]
            raise DatabaseHTTPError(
                f"MC3D {operation} returned HTTP {status_code} for {url}: {snippet}",
                provider="MC3D",
                operation=operation,
                status_code=status_code,
            )

        try:
            return json.loads(body)
        except json.JSONDecodeError as error:
            raise DatabaseAPIError(
                f"MC3D {operation} returned a non-JSON body (HTTP {status_code}) for {url}."
            ) from error

    def _stream_bounded(self, url: str, params: dict[str, str] | None) -> tuple[bytes, int]:
        """GET *url*, rejecting redirects and enforcing ``_MAX_RESPONSE_BYTES``.

        Streams the response body so an oversized payload is aborted mid-
        download rather than fully buffered first.
        """
        with self._client.stream("GET", url, params=params) as response:
            if response.is_redirect:
                location = response.headers.get("location", "<unknown>")
                raise DatabaseAPIError(
                    f"MC3D response redirected to {location!r}; refusing to follow."
                )
            chunks = bytearray()
            for chunk in response.iter_bytes():
                chunks.extend(chunk)
                if len(chunks) > _MAX_RESPONSE_BYTES:
                    raise DatabaseAPIError(
                        f"MC3D response for {url} exceeded the {_MAX_RESPONSE_BYTES}-byte cap."
                    )
            return bytes(chunks), response.status_code


__all__ = ["MC3DClient", "MC3DMethod"]
