"""Typed exception hierarchy for adam-identification.

Raised by database clients and the identifier agent so callers can react by
exception type rather than by parsing error strings.

API clients raise:
    - ``MaterialNotFoundError`` when a lookup has no matching record.
    - ``DatabaseAPIError`` for transport, authentication, or malformed payload
      failures.

The identification pipeline additionally raises:
    - ``AmbiguousIdentificationError`` when the formula is in the database but
      the query cannot select one candidate uniquely.
    - ``ClarificationNeededError`` when the extraction step cannot start
      retrieval (ambiguous composition or domain at the query level).

LLM providers raise:
    - ``RateLimitError``, ``AuthenticationError``, ``ProviderUnavailableError``,
      ``InvalidResponseError``.
"""

from typing import Any

# Default user-facing table size. Full lists remain on ``.candidates``.
_DISPLAY_CANDIDATE_LIMIT = 8


class IdentificationError(Exception):
    """Base class for all adam-identification errors."""


class DatabaseError(IdentificationError):
    """Base class for database-layer errors."""


class MaterialNotFoundError(DatabaseError):
    """Raised when a material ID or query returns no results."""


class ClarificationNeededError(MaterialNotFoundError):
    """Raised when the extraction step cannot begin retrieval reliably.

    The LLM returned ``decision="clarify"`` because the query does not
    determine a unique chemical composition or domain. The caller (e.g. an
    interactive session) should surface :attr:`user_message` and
    :attr:`suggested_queries` to the user, collect a follow-up, and re-call
    :meth:`~adam_identification.identifier.MaterialIdentifier.identify` with the
    refined query.

    Attributes:
        user_message: A concise explanation from the LLM describing what
            information is missing.
        suggested_queries: Up to five reliable example follow-up queries.
        trace: Optional :class:`~adam_identification.trace.IdentificationTrace`.
    """

    def __init__(
        self,
        message: str,
        *,
        user_message: str = "",
        suggested_queries: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.user_message = user_message
        self.suggested_queries: list[str] = suggested_queries or []
        self.trace: Any = None

    def __str__(self) -> str:
        lines = [self.user_message or super().__str__()]
        if self.suggested_queries:
            lines.append("\nExample queries that would work:")
            for q in self.suggested_queries:
                lines.append(f"  - {q}")
        return "\n".join(lines)


class AmbiguousIdentificationError(MaterialNotFoundError):
    """Raised when a query formula is in the database but multiple candidates match.

    The query provides insufficient information to select one candidate uniquely.
    Callers may inspect :attr:`candidates` for the full candidate list from the
    database, :attr:`suggested_candidates` for the LLM's structured choices, and
    :attr:`user_message` for a user-facing explanation of the ambiguity.

    Attributes:
        query: The original natural-language query string.
        formula: The chemical formula extracted from the query.
        domain: ``"crystal"`` or ``"molecule"``.
        candidates: Candidate dicts from the database. Crystals: ``mp_id``,
            ``formula``, ``space_group``, ``energy_above_hull``. Molecules:
            ``cid``, ``name``, ``formula``, ``smiles``.
        user_message: LLM-generated user-facing explanation of the ambiguity
            and what the user should specify next.
        suggested_candidates: Structured follow-up suggestions from the LLM.
            Each dict has ``index`` (int) and ``suggested_query`` (str).
        trace: Optional :class:`~adam_identification.trace.IdentificationTrace`.
    """

    def __init__(
        self,
        message: str,
        *,
        query: str = "",
        formula: str = "",
        domain: str = "",
        candidates: list[dict[str, Any]] | None = None,
        user_message: str | None = None,
        suggested_candidates: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.query = query
        self.formula = formula
        self.domain = domain
        self.candidates: list[dict[str, Any]] = candidates or []
        self.user_message: str | None = user_message
        self.suggested_candidates: list[dict[str, Any]] = suggested_candidates or []
        self.trace: Any = None

    def __str__(self) -> str:
        base = super().__str__()
        lines = [base]
        if self.user_message:
            lines.extend(["", self.user_message])
        if self.suggested_candidates:
            lines.append("\nSuggested follow-up queries:")
            for suggestion in self.suggested_candidates:
                lines.append(f"  - {suggestion.get('suggested_query', '')}")
        if self.candidates:
            lines.append("")
            lines.append(self._format_candidate_options())
        return "\n".join(lines)

    def candidates_for_display(self, *, verbose: bool = False) -> list[dict[str, Any]]:
        """Return a short, distinct candidate list for user-facing output.

        Suggested indices come first. Remaining rows keep one entry per space
        group (crystals) or name (molecules). Oversized P1 cells are omitted
        unless they were suggested. The full list stays on :attr:`candidates`.

        Args:
            verbose: When ``True``, return every candidate unchanged.

        Returns:
            Candidate dicts to print in the CLI or ``str(exc)``.
        """
        if verbose or len(self.candidates) <= _DISPLAY_CANDIDATE_LIMIT:
            return list(self.candidates)

        selected: list[dict[str, Any]] = []
        seen_idx: set[int] = set()
        for suggestion in self.suggested_candidates:
            raw_idx = suggestion.get("index")
            if not isinstance(raw_idx, int):
                continue
            if raw_idx < 0 or raw_idx >= len(self.candidates) or raw_idx in seen_idx:
                continue
            selected.append(self.candidates[raw_idx])
            seen_idx.add(raw_idx)
            if len(selected) >= _DISPLAY_CANDIDATE_LIMIT:
                return selected

        def _group_key(candidate: dict[str, Any]) -> str:
            if self.domain == "crystal":
                return str(candidate.get("space_group") or candidate.get("mp_id") or "")
            return str(candidate.get("name") or candidate.get("cid") or "")

        def _is_bulky_p1(candidate: dict[str, Any]) -> bool:
            if self.domain != "crystal":
                return False
            space_group = str(candidate.get("space_group") or "").replace(" ", "")
            nsites = candidate.get("nsites")
            return space_group in {"P1", "P-1"} and isinstance(nsites, int) and nsites >= 50

        seen_keys = {_group_key(row) for row in selected}
        for idx, candidate in enumerate(self.candidates):
            if idx in seen_idx or _is_bulky_p1(candidate):
                continue
            key = _group_key(candidate)
            if key and key in seen_keys:
                continue
            selected.append(candidate)
            seen_idx.add(idx)
            if key:
                seen_keys.add(key)
            if len(selected) >= _DISPLAY_CANDIDATE_LIMIT:
                break
        return selected

    def _format_candidate_options(self, *, verbose: bool = False) -> str:
        """Return a compact table of database candidates for the user."""
        shown = self.candidates_for_display(verbose=verbose)
        n_all = len(self.candidates)
        n_show = len(shown)
        out: list[str] = []
        if self.domain == "crystal":
            if n_show < n_all:
                out.append(
                    f"Matching Materials Project candidates ({n_show} of {n_all}; "
                    "full list on .candidates):"
                )
            else:
                out.append(f"Matching Materials Project candidates ({n_all}):")
            for candidate in shown:
                mp_id = candidate.get("mp_id") or "?"
                formula = candidate.get("formula") or "?"
                space_group = candidate.get("space_group") or "?"
                nsites = candidate.get("nsites", "?")
                out.append(f"  - {mp_id}  {formula}  {space_group}  {nsites} sites")
        else:
            if n_show < n_all:
                out.append(
                    f"The formula matches {n_all} PubChem entries "
                    f"(showing {n_show}; full list on .candidates)."
                )
            else:
                out.append(f"The formula matches {n_all} PubChem entries.")
            names: list[str] = []
            for candidate in shown:
                name = candidate.get("name") or "?"
                cid = candidate.get("cid")
                if cid is not None:
                    out.append(f"  - {name} (CID {cid})")
                else:
                    out.append(f"  - {name}")
                names.append(name)
            if names:
                out.append("Candidates include: " + ", ".join(names) + ".")
        return "\n".join(out)


class DatabaseAPIError(DatabaseError):
    """Raised on API/auth/network failures or malformed responses."""


class LLMError(IdentificationError):
    """Base class for LLM provider errors."""


class RateLimitError(LLMError):
    """Provider rate limit hit. The caller can back off and retry."""

    api_retries: int = 0


class AuthenticationError(LLMError):
    """API key is missing, expired, or invalid."""


class ProviderUnavailableError(LLMError):
    """Provider service is temporarily unavailable or unreachable."""

    api_retries: int = 0


class InvalidResponseError(LLMError):
    """Provider returned a response that could not be parsed."""
