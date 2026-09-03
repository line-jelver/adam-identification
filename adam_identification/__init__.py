"""adam-identification: database-grounded LLM agent for atomistic structure identification.

Identifies crystals and molecules from natural-language descriptions by
combining a language model with the Materials Project and PubChem databases.
Implements the database-grounded workflow described in:

    Line Jelver et al., "Database-grounded language-model agents for robust
    atomistic input-structure identification in solids and molecules", 2026.

Quick start::

    from adam_identification import identify

    atoms = identify("silicon", model="gemini-3.1-pro-preview")
    atoms = identify("caffeine", model="gemini-3.1-pro-preview")

    structure = identify("silicon", model="gemini-3.1-pro-preview", output="pymatgen")

    from adam_identification import batch_identify
    results = batch_identify(
        ["silicon", "water", "caffeine"],
        model="gemini-3.1-pro-preview",
        concurrency=5,
    )

API keys are loaded from environment variables or a ``.env`` file:

    GOOGLE_API_KEY            (for Gemini — recommended default)
    OPENAI_API_KEY            (for OpenAI models)
    ANTHROPIC_API_KEY         (for Claude models)
    OPENROUTER_API_KEY        (for DeepSeek, Kimi, Qwen, and others)
    MATERIALS_PROJECT_API_KEY (required for crystal identification)

See the README for full setup instructions.
"""

from __future__ import annotations

import logging
import warnings

# Silence noisy third-party loggers at import time.
for _lib in ("mp_api", "pymatgen", "emmet", "monty"):
    logging.getLogger(_lib).setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="mp_api")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymatgen")
del _lib

from typing import TYPE_CHECKING, Literal

from adam_identification.exceptions import (
    AmbiguousIdentificationError,
    AuthenticationError,
    ClarificationNeededError,
    DatabaseAPIError,
    IdentificationError,
    InvalidResponseError,
    LLMError,
    MaterialNotFoundError,
    ProviderUnavailableError,
    RateLimitError,
)
from adam_identification.identifier import MaterialIdentifier
from adam_identification.models import Material
from adam_identification.result import IdentificationResult
from adam_identification.session import IdentificationSession
from adam_identification.trace import IdentificationTrace

if TYPE_CHECKING:
    import ase
    import pymatgen.core

__version__ = "1.4.0"


def batch_identify(
    queries: list[str],
    *,
    provider: str = "google",
    model: str | None = None,
    mp_api_key: str | None = None,
    concurrency: int = 5,
    output: Literal["ase", "pymatgen", "material", "result"] = "ase",
    minimal_interaction: bool = False,
) -> list:
    """Identify multiple materials concurrently. See :mod:`adam_identification.batch`."""
    from adam_identification.batch import batch_identify as _batch_identify

    return _batch_identify(
        queries,
        provider=provider,
        model=model,
        mp_api_key=mp_api_key,
        concurrency=concurrency,
        output=output,
        minimal_interaction=minimal_interaction,
    )


__all__ = [
    "identify",
    "batch_identify",
    "MaterialIdentifier",
    "IdentificationSession",
    "Material",
    "IdentificationResult",
    "IdentificationTrace",
    "AmbiguousIdentificationError",
    "ClarificationNeededError",
    "IdentificationError",
    "MaterialNotFoundError",
    "DatabaseAPIError",
    "LLMError",
    "RateLimitError",
    "AuthenticationError",
    "ProviderUnavailableError",
    "InvalidResponseError",
    "__version__",
]


def identify(
    query: str,
    *,
    provider: str = "google",
    model: str | None = None,
    output: Literal["ase", "pymatgen", "material", "result"] = "ase",
    mp_api_key: str | None = None,
    minimal_interaction: bool = False,
) -> (
    ase.Atoms | pymatgen.core.IStructure | pymatgen.core.IMolecule | Material | IdentificationResult
):
    """Identify a material from a natural-language description.

    Uses the database-grounded workflow: the language model extracts the
    chemical formula and any disambiguation hints, then the Materials Project
    (for crystals) or PubChem (for molecules) supplies the 3D geometry.

    Args:
        query: Natural-language description, e.g. ``"silicon"`` or
            ``"caffeine"``.
        provider: LLM provider key — ``"google"`` (default), ``"openai"``,
            ``"anthropic"``, or ``"openrouter"``. Model slugs remain
            ``gemini-*`` for Google models.
        model: Required model slug. There is no default.
        output: Return type — ``"ase"`` (default, :class:`ase.Atoms`),
            ``"pymatgen"`` (:class:`pymatgen.core.Structure` or
            :class:`pymatgen.core.Molecule`), ``"material"`` (raw
            :class:`~adam_identification.models.Material`), or ``"result"``
            (:class:`~adam_identification.result.IdentificationResult` with
            atoms, material, and trace).
        mp_api_key: Materials Project API key. Falls back to the
            ``MATERIALS_PROJECT_API_KEY`` environment variable.
        minimal_interaction: When ``True``, the LLM always selects a candidate
            and flags uncertain choices with ``trace.needs_review``. When
            ``False`` (default), underspecified queries raise
            :class:`~adam_identification.exceptions.AmbiguousIdentificationError`.

    Returns:
        ``ase.Atoms``, ``pymatgen.core.Structure``, ``pymatgen.core.Molecule``,
        :class:`~adam_identification.models.Material`, or
        :class:`~adam_identification.result.IdentificationResult` depending on
        ``output``.

    Raises:
        ConfigurationError: If ``model`` is omitted.
        MaterialNotFoundError: If the database search returns no usable match.
        AmbiguousIdentificationError: If the formula is in the database but the
            query lacks enough information to pick one candidate.
        ClarificationNeededError: If composition or domain cannot be determined.
        DatabaseAPIError: On API/network failures.
        LLMError: On LLM provider failures (rate limits, auth errors, etc.).
        ValueError: If the LLM returns an empty formula.

    Example::

        from adam_identification import identify

        atoms = identify("silicon", model="gemini-3.1-pro-preview")
        atoms = identify("water", model="gemini-3.1-pro-preview")
        s = identify("BCC iron", model="gemini-3.1-pro-preview", output="pymatgen")
        result = identify("silicon", model="gemini-3.1-pro-preview", output="result")
        result.trace.outcome
    """
    from adam_identification.database.materials_project import MaterialsProjectClient
    from adam_identification.database.pubchem import PubChemClient
    from adam_identification.llm import get_provider
    from adam_identification.output import to_ase, to_pymatgen

    llm = get_provider(provider, model)
    mp_client = MaterialsProjectClient(api_key=mp_api_key)
    pubchem_client = PubChemClient()
    identifier = MaterialIdentifier(
        llm, mp_client, pubchem_client, minimal_interaction=minimal_interaction
    )

    if output == "result":
        material, trace = identifier.identify(query, return_trace=True)
        atoms = to_ase(material)
        return IdentificationResult(atoms=atoms, material=material, trace=trace)

    material = identifier.identify(query)

    if output == "material":
        return material
    if output == "pymatgen":
        return to_pymatgen(material)
    return to_ase(material)
