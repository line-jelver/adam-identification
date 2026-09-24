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
    MATERIALS_PROJECT_API_KEY (optional; MC3D needs no key. Required for
                              --crystal-source materials-project, and used as
                              the auto fallback when set)

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

from adam_identification._config import ConfigurationError
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
from adam_identification.provenance.trace import IdentificationSection, Trace
from adam_identification.result import IdentificationResult
from adam_identification.session import IdentificationSession

if TYPE_CHECKING:
    import ase
    import pymatgen.core

__version__ = "2.1.0"


def batch_identify(
    queries: list[str],
    *,
    provider: str | None = None,
    model: str | None = None,
    mp_api_key: str | None = None,
    crystal_source: str = "auto",
    mc3d_method: str = "pbesol-v2",
    concurrency: int = 5,
    output: Literal["ase", "pymatgen", "material", "result"] = "ase",
    minimal_interaction: bool = False,
    work_dir: str | None = None,
) -> list:
    """Identify multiple materials concurrently. See :mod:`adam_identification.batch`."""
    from pathlib import Path

    from adam_identification.batch import batch_identify as _batch_identify

    return _batch_identify(
        queries,
        provider=provider,
        model=model,
        mp_api_key=mp_api_key,
        crystal_source=crystal_source,
        mc3d_method=mc3d_method,
        concurrency=concurrency,
        output=output,
        minimal_interaction=minimal_interaction,
        work_dir=Path(work_dir) if work_dir is not None else None,
    )


__all__ = [
    "identify",
    "batch_identify",
    "MaterialIdentifier",
    "IdentificationSession",
    "Material",
    "IdentificationResult",
    "IdentificationSection",
    "Trace",
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
    "ConfigurationError",
    "__version__",
]


def identify(
    query: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    output: Literal["ase", "pymatgen", "material", "result"] = "ase",
    mp_api_key: str | None = None,
    crystal_source: str = "auto",
    mc3d_method: str = "pbesol-v2",
    minimal_interaction: bool = False,
    work_dir: str | None = None,
) -> (
    ase.Atoms | pymatgen.core.IStructure | pymatgen.core.IMolecule | Material | IdentificationResult
):
    """Identify a material from a natural-language description.

    Always constructs a provenance :class:`~adam_identification.provenance.trace.Trace`.
    Persist it when ``work_dir`` is set. Return it only for ``output="result"``.

    Args:
        query: Natural-language description, e.g. ``"silicon"`` or
            ``"caffeine"``.
        provider: LLM provider key — ``"google"``, ``"openai"``,
            ``"anthropic"``, or ``"openrouter"``. Omitted: inferred from
            ``model``. Pass ``"openrouter"`` to send a native slug through
            that API.
        model: Required model slug. There is no default.
        output: Return type — ``"ase"`` (default), ``"pymatgen"``,
            ``"material"``, or ``"result"`` (atoms, material, and full Trace).
        mp_api_key: Materials Project API key override. Used when
            ``crystal_source`` is ``materials-project``, or as the ``auto``
            fallback when set.
        crystal_source: ``auto`` (default), ``mc3d``, or ``materials-project``.
        mc3d_method: MC3D dataset ``pbe-v1``, ``pbesol-v1``, or ``pbesol-v2``
            (default). This selects the published database, not an execution
            exchange-correlation functional.
        minimal_interaction: When ``True``, always select and flag
            ``trace.needs_review``.
        work_dir: When set, write ``adam.json`` and a structure file here.

    Returns:
        Converted structure, :class:`~adam_identification.models.Material`, or
        :class:`~adam_identification.result.IdentificationResult`.

    Example::

        result = identify("silicon", model="gemini-3.1-pro-preview", output="result")
        result.trace.identification.outcome
    """
    from pathlib import Path

    from adam_identification.output import to_ase, to_pymatgen
    from adam_identification.provenance.lifecycle import run_identification

    wd = Path(work_dir) if work_dir is not None else None
    material, trace = run_identification(
        query,
        provider=provider,
        model=model,
        mp_api_key=mp_api_key,
        crystal_source=crystal_source,
        mc3d_method=mc3d_method,
        minimal_interaction=minimal_interaction,
        work_dir=wd,
        write_artifacts=wd is not None,
    )

    if output == "result":
        return IdentificationResult(atoms=to_ase(material), material=material, trace=trace)
    if output == "material":
        return material
    if output == "pymatgen":
        return to_pymatgen(material)
    return to_ase(material)
