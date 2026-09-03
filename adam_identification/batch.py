"""High-throughput batch identification for adam-identification.

Identifies multiple materials concurrently using the async LLM layer. A
semaphore limits simultaneous in-flight requests to avoid API rate limits.

Example::

    from adam_identification import batch_identify

    results = batch_identify(
        ["silicon", "water", "caffeine", "iron (bcc)"],
        provider="google",
        model="gemini-2.5-flash",
        concurrency=5,
    )
    for query, result in zip(queries, results):
        if isinstance(result, Exception):
            print(f"{query}: FAILED — {result}")
        else:
            print(f"{query}: {result.symbols}")  # ase.Atoms
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from adam_identification.identifier import MaterialIdentifier
from adam_identification.models import Material
from adam_identification.result import IdentificationResult

logger = logging.getLogger(__name__)


def _run_one(
    query: str,
    identifier: MaterialIdentifier,
    output: Literal["ase", "pymatgen", "material", "result"],
) -> Material | IdentificationResult:
    """Identify one query, optionally returning the provenance trace."""
    if output == "result":
        from adam_identification.output import to_ase

        material, trace = identifier.identify(query, return_trace=True)
        return IdentificationResult(atoms=to_ase(material), material=material, trace=trace)
    return identifier.identify(query)


async def _identify_one(
    query: str,
    identifier: MaterialIdentifier,
    semaphore: asyncio.Semaphore,
    output: Literal["ase", "pymatgen", "material", "result"],
) -> Material | IdentificationResult:
    """Identify a single query under a shared concurrency semaphore.

    Args:
        query: Natural-language material description.
        identifier: Shared :class:`~adam_identification.identifier.MaterialIdentifier`.
        semaphore: Concurrency limiter.
        output: Requested output type; ``"result"`` requests a trace.

    Returns:
        Identified material or :class:`~adam_identification.result.IdentificationResult`.
    """
    async with semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_one, query, identifier, output)


async def batch_identify_async(
    queries: list[str],
    *,
    provider: str = "google",
    model: str | None = None,
    mp_api_key: str | None = None,
    concurrency: int = 5,
    output: Literal["ase", "pymatgen", "material", "result"] = "material",
    minimal_interaction: bool = False,
) -> list[Material | IdentificationResult | Exception]:
    """Identify multiple materials concurrently (async version).

    Args:
        queries: List of natural-language material descriptions.
        provider: LLM provider key (``"openai"``, ``"anthropic"``,
            ``"google"``, ``"openrouter"``).
        model: Required model slug. There is no default.
        mp_api_key: Materials Project API key. Falls back to environment variable.
        concurrency: Maximum simultaneous in-flight requests.
        output: ``"result"`` fetches a trace; other values return ``Material``.
        minimal_interaction: When ``True``, always select and flag
            ``needs_review`` instead of raising on ambiguity.

    Returns:
        List of :class:`~adam_identification.models.Material`,
        :class:`~adam_identification.result.IdentificationResult`, or
        :class:`Exception` instances, one per query (order preserved).
    """
    from adam_identification.database.materials_project import MaterialsProjectClient
    from adam_identification.database.pubchem import PubChemClient
    from adam_identification.llm import get_provider

    llm = get_provider(provider, model)
    mp_client = MaterialsProjectClient(api_key=mp_api_key)
    pubchem_client = PubChemClient()
    identifier = MaterialIdentifier(
        llm, mp_client, pubchem_client, minimal_interaction=minimal_interaction
    )

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_identify_one(q, identifier, semaphore, output) for q in queries]
    results: list[Material | IdentificationResult | Exception] = []
    raw = await asyncio.gather(*tasks, return_exceptions=True)
    for item in raw:
        results.append(item)  # type: ignore[arg-type]
    return results


def batch_identify(
    queries: list[str],
    *,
    provider: str = "google",
    model: str | None = None,
    mp_api_key: str | None = None,
    concurrency: int = 5,
    output: Literal["ase", "pymatgen", "material", "result"] = "ase",
    minimal_interaction: bool = False,
) -> list[object]:
    """Identify multiple materials concurrently (synchronous wrapper).

    Args:
        queries: List of natural-language material descriptions.
        provider: LLM provider key (``"openai"``, ``"anthropic"``,
            ``"google"``, ``"openrouter"``).
        model: Required model slug. There is no default.
        mp_api_key: Materials Project API key. Falls back to environment variable.
        concurrency: Maximum simultaneous in-flight requests.
        output: Output type — ``"ase"`` (default), ``"pymatgen"``,
            ``"material"`` (raw :class:`~adam_identification.models.Material`),
            or ``"result"``
            (:class:`~adam_identification.result.IdentificationResult`).
        minimal_interaction: When ``True``, always select and flag
            ``needs_review`` instead of raising on ambiguity.

    Returns:
        List of converted objects or :class:`Exception` instances (order
        preserved). Failed identifications are returned as exceptions, not
        raised, so the rest of the batch still completes.

    Example::

        results = batch_identify(
            ["silicon", "water", "caffeine"],
            model="gemini-3.1-pro-preview",
        )
        # results[0] = ase.Atoms for silicon
        # results[1] = ase.Atoms for water
    """
    raw_results = asyncio.run(
        batch_identify_async(
            queries,
            provider=provider,
            model=model,
            mp_api_key=mp_api_key,
            concurrency=concurrency,
            output=output,
            minimal_interaction=minimal_interaction,
        )
    )

    if output in ("material", "result"):
        boxed: list[object] = []
        boxed.extend(raw_results)
        return boxed

    from adam_identification.output import to_ase, to_pymatgen

    converted: list[object] = []
    for item in raw_results:
        if isinstance(item, Exception):
            converted.append(item)
        else:
            try:
                material = item if isinstance(item, Material) else item.material
                if output == "ase":
                    converted.append(to_ase(material))
                else:
                    converted.append(to_pymatgen(material))
            except Exception as exc:
                logger.warning("Output conversion failed: %s", exc)
                converted.append(exc)
    return converted
