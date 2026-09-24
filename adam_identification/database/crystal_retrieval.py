"""Provider-neutral crystal-source policy and retrieval for ADaM.

Combines one or more :class:`~adam_identification.database.crystal_provider.CrystalProvider`
implementations behind a single :class:`CrystalRetriever` so that
:class:`~adam.material_identification.MaterialIdentifier` can search, select,
and hydrate a crystal candidate without knowing which database answered.

Under :attr:`CrystalSourcePolicy.AUTO`, ``search()`` tries each configured
provider in the order given to :class:`CrystalRetriever` and returns the
first one that reports at least one candidate, falling back to the next
provider on either an empty result or a transient/typed database error.
Pinning the policy to one source (:attr:`CrystalSourcePolicy.MC3D` or
:attr:`CrystalSourcePolicy.MATERIALS_PROJECT`) restricts every call to that
provider alone, so a failure there is never silently masked by a fallback.

:func:`build_crystal_retriever` is the single place that turns a
``--crystal-source``-style policy token into a configured
:class:`CrystalRetriever`; the ``adam`` CLI (:mod:`adam.cli._wiring`) and any
other caller (e.g. ``adam_bench``'s identification benchmarks) construct a
retriever the same way rather than each reimplementing provider selection.

Cross-references:
    - ``adam_identification.database.crystal_provider`` — ``CrystalCandidate``,
      ``CrystalProvider``, ``CrystalSearchResult``.
    - ``adam_identification.database.mc3d`` — ``MC3DClient``.
    - ``adam_identification.database.materials_project_provider`` —
      ``MaterialsProjectCrystalProvider``.
    - ``adam_identification.identifier`` — the primary consumer.
    - ``adam_identification.cli`` — builds the CLI's retriever via
      :func:`build_crystal_retriever`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from enum import StrEnum

from adam_identification._config import settings
from adam_identification.database.crystal_provider import (
    CrystalCandidate,
    CrystalProvider,
    CrystalSearchResult,
)
from adam_identification.database.materials_project import MaterialsProjectClient
from adam_identification.database.materials_project_provider import MaterialsProjectCrystalProvider
from adam_identification.database.mc3d import MC3DClient, MC3DMethod
from adam_identification.database.retrieval_status import announce_search
from adam_identification.exceptions import DatabaseAPIError, MaterialNotFoundError
from adam_identification.models import Material, MaterialSource

logger = logging.getLogger(__name__)


class CrystalSourcePolicy(StrEnum):
    """Which crystal database(s) :class:`CrystalRetriever` may use."""

    AUTO = "auto"
    MC3D = "mc3d"
    MATERIALS_PROJECT = "materials_project"


#: Maps an explicit-ID prefix (before the first ``-``) to the database it
#: names, e.g. a query of exactly ``"mc3d-18058"`` or ``"mp-149"``.
_ID_PREFIX_TO_SOURCE: dict[str, MaterialSource] = {
    "mc3d": MaterialSource.MC3D,
    "mp": MaterialSource.MATERIALS_PROJECT,
}

_EXPLICIT_ID_PATTERN = re.compile(rf"^(?:{'|'.join(_ID_PREFIX_TO_SOURCE)})-\d+$", re.IGNORECASE)

_POLICY_TO_SOURCE: dict[CrystalSourcePolicy, MaterialSource] = {
    CrystalSourcePolicy.MC3D: MaterialSource.MC3D,
    CrystalSourcePolicy.MATERIALS_PROJECT: MaterialSource.MATERIALS_PROJECT,
}


def looks_like_explicit_crystal_id(text: str) -> bool:
    """Return whether ``text`` is exactly a crystal-database ID.

    Matches strings like ``"mc3d-18058"`` or ``"mp-149"`` (case-insensitive,
    surrounding whitespace ignored) and nothing else — a formula or natural-
    language description never matches.

    Args:
        text: Candidate query string.

    Returns:
        ``True`` when ``text`` is exactly one recognized ID pattern.
    """
    return bool(_EXPLICIT_ID_PATTERN.match(text.strip()))


def _source_for_id(source_id: str) -> MaterialSource:
    """Infer the owning database from an explicit ID's prefix.

    Raises:
        MaterialNotFoundError: If ``source_id`` has no recognized prefix.
    """
    prefix = source_id.strip().split("-", 1)[0].lower()
    source = _ID_PREFIX_TO_SOURCE.get(prefix)
    if source is None:
        raise MaterialNotFoundError(f"Unrecognized crystal-database ID prefix in {source_id!r}.")
    return source


def _mc3d_candidate_sort_key(candidate: CrystalCandidate) -> tuple[int, int, int, int, int]:
    """Deterministic MC3D candidate ordering: never sorted by energy.

    Prefers, in order: an experimentally-derived structure over a
    theoretical one, an ambient-condition structure over a high-pressure or
    high-temperature one, a known space-group number over an unknown one
    (lower first), a known site count over an unknown one (lower first),
    then the numeric MC3D ID. ``None`` on a ranking field is treated the
    same as the more-preferred value, since an unknown flag must not be
    penalized relative to a known-good one.
    """
    is_theoretical = 1 if candidate.is_theoretical else 0
    is_extreme_condition = 1 if (candidate.is_high_pressure or candidate.is_high_temperature) else 0
    space_group_number = (
        candidate.space_group_number if candidate.space_group_number is not None else 10**9
    )
    nsites = candidate.nsites if candidate.nsites is not None else 10**9
    numeric_id = _numeric_id_suffix(candidate.source_id)
    return (is_theoretical, is_extreme_condition, space_group_number, nsites, numeric_id)


def _numeric_id_suffix(source_id: str) -> int:
    """Extract the trailing integer from an ID like ``"mc3d-18058"`` -> ``18058``."""
    match = re.search(r"(\d+)$", source_id)
    return int(match.group(1)) if match else 10**9


class CrystalRetriever:
    """Search, select, and hydrate crystal candidates across one or more sources.

    Args:
        providers: Configured :class:`CrystalProvider`
            instances, in fallback order. Under
            :attr:`CrystalSourcePolicy.AUTO` this order determines which
            provider is tried first; it is otherwise unused. Each provider's
            :attr:`~adam_identification.database.crystal_provider.CrystalProvider.source`
            must be unique — a second provider for the same source overwrites
            the first internally.
        policy: Which database(s) to use. :attr:`CrystalSourcePolicy.AUTO`
            (default) tries every provider in order; pinning to
            :attr:`CrystalSourcePolicy.MC3D` or
            :attr:`CrystalSourcePolicy.MATERIALS_PROJECT` restricts every
            call to that one provider, so its failures are never masked by a
            fallback.

    Raises:
        ValueError: If ``providers`` is empty.
    """

    def __init__(
        self,
        providers: Sequence[CrystalProvider],
        *,
        policy: CrystalSourcePolicy = CrystalSourcePolicy.AUTO,
    ) -> None:
        if not providers:
            raise ValueError("CrystalRetriever requires at least one CrystalProvider.")
        self._providers: list[CrystalProvider] = list(providers)
        self._provider_by_source: dict[MaterialSource, CrystalProvider] = {
            provider.source: provider for provider in self._providers
        }
        self.policy = CrystalSourcePolicy(policy)

    def _ordered_providers(self) -> list[CrystalProvider]:
        """Providers to try, in order, for the current :attr:`policy`.

        Raises:
            DatabaseAPIError: If :attr:`policy` pins a single source that has
                no configured provider.
        """
        if self.policy == CrystalSourcePolicy.AUTO:
            return self._providers
        target = _POLICY_TO_SOURCE[self.policy]
        provider = self._provider_by_source.get(target)
        if provider is None:
            raise DatabaseAPIError(
                f"No crystal provider configured for policy {self.policy.value!r}."
            )
        return [provider]

    def search(self, formula: str, max_results: int) -> CrystalSearchResult:
        """Search configured providers in order; return the first non-empty result.

        Under :attr:`CrystalSourcePolicy.AUTO`, a provider that returns zero
        candidates or raises :class:`~adam_identification.database.exceptions.DatabaseAPIError`
        is skipped in favor of the next one. When every provider is
        exhausted, the most recent error is re-raised if there was one;
        otherwise an empty :class:`CrystalSearchResult`
        is returned (a genuine no-match, not an outage).

        Args:
            formula: Chemical formula in any common notation.
            max_results: Maximum number of candidates to request and return.

        Returns:
            The first provider's non-empty result, with MC3D results
            re-ordered by :func:`_mc3d_candidate_sort_key` (never by energy).

        Raises:
            DatabaseAPIError: If every provider tried raised one, or if
                :attr:`policy` pins a source with no configured provider.
        """
        last_error: DatabaseAPIError | None = None
        for provider in self._ordered_providers():
            announce_search(provider.source)
            try:
                result = provider.search_candidates(formula, max_results)
            except DatabaseAPIError as error:
                logger.warning(
                    "Crystal provider %s unavailable for formula %r: %s",
                    provider.source,
                    formula,
                    error,
                )
                last_error = error
                continue
            last_error = None
            if not result.candidates:
                continue
            candidates = result.candidates
            if provider.source == MaterialSource.MC3D:
                candidates = sorted(candidates, key=_mc3d_candidate_sort_key)
            return CrystalSearchResult(
                candidates=candidates,
                total_matches=result.total_matches,
                truncated=result.truncated,
            )
        if last_error is not None:
            raise last_error
        return CrystalSearchResult(candidates=[])

    def hydrate(self, candidate: CrystalCandidate) -> Material:
        """Return the full ``Material`` for a previously-searched candidate.

        Args:
            candidate: A candidate previously returned by :meth:`search`.

        Returns:
            The hydrated ``Material``.

        Raises:
            DatabaseAPIError: If no provider is configured for
                ``candidate.source``.
        """
        provider = self._provider_by_source.get(candidate.source)
        if provider is None:
            raise DatabaseAPIError(
                f"No crystal provider configured for candidate source {candidate.source!r}."
            )
        return provider.hydrate(candidate)

    def get_by_id(self, source_id: str) -> Material:
        """Fetch one material directly by its provider-specific ID.

        Used by the explicit-ID fast path (a query that is exactly
        ``"mc3d-18058"`` or ``"mp-149"``), bypassing formula search and LLM
        phase selection entirely. Respects :attr:`policy`: an ID whose
        database is not among the providers :attr:`policy` currently allows
        is rejected, even if a provider for it happens to be configured.

        Args:
            source_id: Provider-specific ID, e.g. ``"mc3d-18058"``.

        Returns:
            A mapped ``Material`` instance.

        Raises:
            MaterialNotFoundError: If ``source_id`` has no recognized
                database prefix, or does not exist in that database.
            DatabaseAPIError: If ``source_id``'s database is not allowed
                under the current :attr:`policy`, or is not configured at
                all.
        """
        source = _source_for_id(source_id)
        allowed_sources = {provider.source for provider in self._ordered_providers()}
        if source not in allowed_sources:
            raise DatabaseAPIError(
                f"ID {source_id!r} belongs to {source.value!r}, which is not available "
                f"under crystal-source policy {self.policy.value!r}."
            )
        announce_search(source)
        return self._provider_by_source[source].get_by_id(source_id)

    def close(self) -> None:
        """Close every configured provider that owns a client.

        Safe to call even when no provider owns a closeable client (e.g. a
        test double). Providers without a ``close()`` method are skipped.
        Callers that hold a retriever only for the duration of one script or
        one identification call should call this when finished — MC3D's
        provider owns an ``httpx.Client`` that is otherwise never closed.
        """
        for provider in self._providers:
            closer = getattr(provider, "close", None)
            if callable(closer):
                closer()


#: Maps a CLI-style, hyphenated crystal-source token to its policy.
#: ``CrystalSourcePolicy.MATERIALS_PROJECT.value`` is underscored
#: (``"materials_project"``); the CLI token is hyphenated
#: (``"materials-project"``) to match ``adam``'s other multi-word flag values.
_CRYSTAL_SOURCE_TOKENS: dict[str, CrystalSourcePolicy] = {
    "auto": CrystalSourcePolicy.AUTO,
    "mc3d": CrystalSourcePolicy.MC3D,
    "materials-project": CrystalSourcePolicy.MATERIALS_PROJECT,
}


def build_crystal_retriever(
    crystal_source: str = "auto",
    mc3d_method: str = "pbesol-v2",
    *,
    mp_api_key: str | None = None,
) -> CrystalRetriever:
    """Build a :class:`CrystalRetriever` for a ``--crystal-source``-style token.

    The single place that turns ``adam``'s ``--crystal-source``/``--mc3d-method``
    CLI tokens into a configured retriever. MC3D needs no credentials, so
    ``"auto"`` and ``"mc3d"`` always include it. ``"auto"`` adds Materials
    Project only when ``MATERIALS_PROJECT_API_KEY`` is already set. An
    explicit ``"materials-project"`` policy includes only Materials Project,
    and raises :class:`~adam_identification._config.ConfigurationError` here when that key is
    missing, before any client is constructed.

    Args:
        crystal_source: ``"auto"`` (default), ``"mc3d"``, or
            ``"materials-project"``.
        mc3d_method: MC3D methodology token: ``"pbe-v1"``, ``"pbesol-v1"``,
            or ``"pbesol-v2"`` (default). Selects the published dataset, not
            the functional used later in a calculation.

    Returns:
        Retriever whose provider list matches ``crystal_source``.

    Raises:
        ValueError: If ``crystal_source`` or ``mc3d_method`` is not supported.
        ConfigurationError: If ``crystal_source`` is ``"materials-project"``
            and ``MATERIALS_PROJECT_API_KEY`` is not set.
    """
    source_token = crystal_source.strip().lower()
    try:
        policy = _CRYSTAL_SOURCE_TOKENS[source_token]
    except KeyError as error:
        allowed = ", ".join(sorted(_CRYSTAL_SOURCE_TOKENS))
        raise ValueError(
            f"Unknown crystal source {crystal_source!r}. Expected one of: {allowed}."
        ) from error

    try:
        method = MC3DMethod(mc3d_method.strip().lower())
    except ValueError as error:
        allowed = ", ".join(item.value for item in MC3DMethod)
        raise ValueError(
            f"Unknown MC3D method {mc3d_method!r}. Expected one of: {allowed}."
        ) from error

    resolved_mp_key = mp_api_key or settings.materials_project_api_key
    providers: list[CrystalProvider] = []
    if policy is CrystalSourcePolicy.MATERIALS_PROJECT:
        if not resolved_mp_key:
            settings.require_materials_project()
        providers.append(
            MaterialsProjectCrystalProvider(MaterialsProjectClient(api_key=resolved_mp_key))
        )
    else:
        providers.append(MC3DClient(method=method))
        if policy is CrystalSourcePolicy.AUTO and resolved_mp_key:
            providers.append(
                MaterialsProjectCrystalProvider(MaterialsProjectClient(api_key=resolved_mp_key))
            )
    return CrystalRetriever(providers, policy=policy)


__all__ = [
    "CrystalRetriever",
    "CrystalSourcePolicy",
    "build_crystal_retriever",
    "looks_like_explicit_crystal_id",
]
