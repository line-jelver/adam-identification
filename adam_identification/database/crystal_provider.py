"""Provider-neutral crystal search contract for ADaM.

Defines the small, shared shape needed to search and select a crystal
candidate the same way regardless of which database answered.

A bounded formula-search endpoint may only return compact per-candidate
metadata (formula, space group, site count, source database, energies) —
never lattice vectors or atomic positions. ``CrystalCandidate`` exists to
carry that compact, pre-hydration metadata during the search/selection step,
independent of which database answered. A provider whose search endpoint
already returns full structures can instead build these candidates from a
``Material`` it already holds and make ``hydrate()`` a local lookup rather
than a second network request.

Implemented today by ``adam_identification.database.apis.mc3d.MC3DClient`` directly, and by
Materials Project via a thin composition adapter,
``adam_identification.database.apis.materials_project_provider.MaterialsProjectCrystalProvider``,
that wraps ``MaterialsProjectClient`` without changing it.
:class:`~adam.material_identification.MaterialIdentifier` reaches both
through :class:`~adam_identification.database.crystal_retrieval.CrystalRetriever`.

Cross-references:
    - ``adam_identification.database.models`` — ``Material``, ``MaterialSource``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from adam_identification.models import Material, MaterialSource


class CrystalCandidate(BaseModel):
    """Compact, pre-hydration description of one crystal search result.

    Each field falls into one of three groups: a quality/ranking signal
    (e.g. preferring an experimental structure over a theoretical one, or an
    ambient-condition measurement over a high-pressure/high-temperature
    one), a field rendered by default in the LLM phase-selection prompt (see
    ``adam.material_identification.phase_lookup.candidates_to_selection_json``),
    or a value carried opportunistically because the metadata call that
    populates this candidate already returns it at no extra request cost —
    useful for optional inclusion when the user's query is explicitly about
    that property (e.g. an energy or cell-volume comparison). It never
    carries geometry — call :meth:`CrystalProvider.hydrate` on the selected
    candidate to obtain a full :class:`~adam_identification.database.models.Material`.

    ``total_energy_ev_per_cell`` and ``cell_volume_ang3`` must never be used
    as an energy-above-hull or formation-energy proxy: each MC3D entry's
    total energy is computed independently and is not referenced to a
    common convex hull the way Materials Project's energy-above-hull is, so
    these values are not directly comparable across entries in that sense.
    """

    model_config = ConfigDict(frozen=True)

    source: MaterialSource
    source_id: str
    """Provider-specific display ID, e.g. ``"mc3d-18058"`` or ``"mp-149"``."""

    structure_ref: str
    """Opaque reference passed back to :meth:`CrystalProvider.hydrate`.

    For MC3D this is the OPTIMADE structure UUID. For Materials Project this
    is the same as ``source_id`` (the MP ID) — its provider already returns
    a fully hydrated material during search, so hydration there is a local
    cache lookup rather than a second request.
    """

    formula: str
    method: str | None = None
    """Database methodology, e.g. ``"pbesol-v2"``. ``None`` for providers
    without a versioned methodology."""

    space_group: str | None = None
    space_group_number: int | None = None
    """Ranking signal: a known space group is generally preferable to an
    unknown one when comparing otherwise-similar candidates."""

    crystal_system: str | None = None
    bravais_lattice: str | None = None
    """Pearson-style Bravais lattice symbol, e.g. ``"cF"``. Provenance
    context only."""

    nsites: int | None = None
    """Ranking signal: a known site count is generally preferable to an
    unknown one when comparing otherwise-similar candidates."""

    source_database: str | None = None
    """Original source database of the record, e.g. ``"icsd"``, ``"cod"``.
    Provenance context only."""

    source_database_id: str | None = None
    """Record ID within ``source_database``. Provenance context only."""

    is_theoretical: bool | None = None
    """Ranking signal: an experimentally-derived structure is generally
    preferable to a theoretical one when both match the query."""

    is_high_pressure: bool | None = None
    """Ranking signal: an ambient-condition structure is generally
    preferable to a high-pressure one when both match the query."""

    is_high_temperature: bool | None = None
    """Ranking signal: an ambient-condition structure is generally
    preferable to a high-temperature one when both match the query."""

    total_energy_ev_per_cell: float | None = None
    """Never an energy-above-hull/formation-energy proxy (see class
    docstring). Carried for optional prompt inclusion only when the user's
    query explicitly asks for the lowest-energy database phase."""

    cell_volume_ang3: float | None = None
    """Carried for optional prompt inclusion only when the user's query is
    explicitly about cell volume."""

    energy_above_hull_ev_per_atom: float | None = None
    """Thermodynamic stability referenced to a common convex hull (eV/atom).
    Rendered by default in the LLM phase-selection prompt, which is
    instructed not to use it unless the query explicitly asks for the
    lowest-energy database candidate. Only available from a provider that
    maintains a common hull across its entries (Materials Project today);
    ``None`` otherwise. Never comparable to ``total_energy_ev_per_cell``
    above, which is not hull-referenced."""

    is_metal: bool | None = None
    """Electronic-structure classification from the provider's computed band
    structure. Rendered by default in the LLM phase-selection prompt when
    available. ``None`` when the provider does not report it (e.g. MC3D
    today)."""


class CrystalSearchResult(BaseModel):
    """Bounded result of one candidate search call.

    Attributes:
        candidates: Candidates returned for the current page, already capped
            at the caller's ``max_results``.
        total_matches: Total number of filtered matches reported by the
            provider, when available. May exceed ``len(candidates)`` — this is
            expected (see ``formula_search_tio2_page1.json`` in
            ``tests/fixtures/mc3d/``) and must not be treated as an error.
        truncated: ``True`` when the provider reported more matches than were
            requested (i.e. ``total_matches > len(candidates)``).
    """

    model_config = ConfigDict(frozen=True)

    candidates: list[CrystalCandidate]
    total_matches: int | None = None
    truncated: bool = False


@runtime_checkable
class CrystalProvider(Protocol):
    """Common interface implemented by every crystal-structure source.

    Implemented today by ``MC3DClient`` directly, and by
    ``MaterialsProjectCrystalProvider`` for Materials Project.
    """

    source: MaterialSource

    def search_candidates(self, formula: str, max_results: int) -> CrystalSearchResult:
        """Return up to ``max_results`` bounded, metadata-only candidates for ``formula``.

        Must not fetch full geometry for any candidate and must never request
        an unbounded listing (a provider's full database dump / overview
        endpoint) to answer this call.
        """
        ...

    def get_by_id(self, source_id: str) -> Material:
        """Fetch one material directly by its provider-specific ID.

        Used by the explicit-ID fast path (e.g. a query is exactly
        ``"mc3d-18058"`` or ``"mp-149"``), bypassing formula search and LLM
        phase selection entirely.

        Raises:
            ~adam_identification.database.exceptions.MaterialNotFoundError: If ``source_id``
                does not exist for this provider.
        """
        ...

    def hydrate(self, candidate: CrystalCandidate) -> Material:
        """Return the full ``Material`` for a previously-searched candidate.

        Implementations must verify ``candidate.source`` matches
        :attr:`source` before hydrating.
        """
        ...


__all__ = ["CrystalCandidate", "CrystalSearchResult", "CrystalProvider"]
