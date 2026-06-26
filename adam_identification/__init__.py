"""adam-identification: database-grounded LLM agent for atomistic structure identification.

Identifies crystals and molecules from natural-language descriptions by
combining a language model with the Materials Project and PubChem databases.
Implements the database-grounded workflow described in:

    Line Jelver et al., "Database-grounded language-model agents for robust
    atomistic input-structure identification in solids and molecules", 2026.

Quick start::

    from adam_identification import identify

    # Returns an ase.Atoms object by default
    atoms = identify("silicon")
    atoms = identify("caffeine")
    atoms = identify("hexagonal boron nitride")

    # Or a pymatgen Structure / Molecule
    structure = identify("silicon", output="pymatgen")

    # Batch mode for high-throughput workflows
    from adam_identification import batch_identify
    results = batch_identify(["silicon", "water", "caffeine"], concurrency=5)

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

from adam_identification.batch import batch_identify
from adam_identification.exceptions import (
    AuthenticationError,
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

if TYPE_CHECKING:
    import ase
    import pymatgen.core

__version__ = "1.0.0"
__all__ = [
    "identify",
    "batch_identify",
    "MaterialIdentifier",
    "Material",
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
    provider: str = "gemini",
    model: str | None = None,
    output: Literal["ase", "pymatgen", "material"] = "ase",
    mp_api_key: str | None = None,
) -> "ase.Atoms | pymatgen.core.IStructure | pymatgen.core.IMolecule | Material":
    """Identify a material from a natural-language description.

    Uses the database-grounded workflow: the language model extracts the
    chemical formula and any disambiguation hints, then the Materials Project
    (for crystals) or PubChem (for molecules) supplies the 3D geometry.

    Args:
        query: Natural-language description, e.g. ``"silicon"``,
            ``"caffeine"``, ``"hexagonal boron nitride"``.
        provider: LLM provider key — ``"gemini"`` (default), ``"openai"``,
            ``"anthropic"``, or ``"openrouter"``.
        model: Model slug. Uses each provider's recommended default when
            omitted. See the README for tested models.
        output: Return type — ``"ase"`` (default, :class:`ase.Atoms`),
            ``"pymatgen"`` (:class:`pymatgen.core.Structure` or
            :class:`pymatgen.core.Molecule`), or ``"material"`` (raw
            :class:`~adam_identification.models.Material`).
        mp_api_key: Materials Project API key. Falls back to the
            ``MATERIALS_PROJECT_API_KEY`` environment variable.

    Returns:
        ``ase.Atoms``, ``pymatgen.core.Structure``, ``pymatgen.core.Molecule``,
        or :class:`~adam_identification.models.Material` depending on
        ``output``.

    Raises:
        MaterialNotFoundError: If the database search returns no usable match.
        DatabaseAPIError: On API/network failures.
        LLMError: On LLM provider failures (rate limits, auth errors, etc.).
        ValueError: If the LLM returns an empty formula.

    Example::

        from adam_identification import identify

        atoms = identify("silicon")                  # ase.Atoms, pbc=True
        atoms = identify("water")                    # ase.Atoms, pbc=False
        s = identify("BCC iron", output="pymatgen")  # pymatgen.core.Structure
    """
    from adam_identification.database.materials_project import MaterialsProjectClient
    from adam_identification.database.pubchem import PubChemClient
    from adam_identification.llm import get_provider
    from adam_identification.output import to_ase, to_pymatgen

    llm = get_provider(provider, model)
    mp_client = MaterialsProjectClient(api_key=mp_api_key)
    pubchem_client = PubChemClient()
    identifier = MaterialIdentifier(llm, mp_client, pubchem_client)

    material = identifier.identify(query)

    if output == "material":
        return material
    if output == "pymatgen":
        return to_pymatgen(material)
    return to_ase(material)
