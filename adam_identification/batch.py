"""High-throughput batch identification for adam-identification.

Identifies multiple materials concurrently using the async LLM layer. A
semaphore limits simultaneous in-flight requests to avoid API rate limits.

Example::

    from adam_identification import batch_identify

    results = batch_identify(
        ["silicon", "water", "caffeine", "iron (bcc)"],
        provider="gemini",
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
from typing import TYPE_CHECKING, Literal

from adam_identification.models import Material

if TYPE_CHECKING:
    import ase

logger = logging.getLogger(__name__)


async def _identify_one(
    query: str,
    identifier: MaterialIdentifier,
    semaphore: asyncio.Semaphore,
) -> Material:
    """Identify a single query under a shared concurrency semaphore.

    Args:
        query: Natural-language material description.
        identifier: Shared :class:`~adam_identification.identifier.MaterialIdentifier`.
        semaphore: Concurrency limiter.

    Returns:
        Identified :class:`~adam_identification.models.Material`.
    """
    async with semaphore:
        loop = asyncio.get_running_loop()
        # identifier.identify() is synchronous (wraps asyncio.run internally),
        # so run it in a thread to allow true async concurrency.
        return await loop.run_in_executor(None, identifier.identify, query)


async def batch_identify_async(
    queries: list[str],
    *,
    provider: str = "gemini",
    model: str | None = None,
    mp_api_key: str | None = None,
    concurrency: int = 5,
) -> list[Material | Exception]:
    """Identify multiple materials concurrently (async version).

    Args:
        queries: List of natural-language material descriptions.
        provider: LLM provider key (``"openai"``, ``"anthropic"``,
            ``"gemini"``, ``"openrouter"``).
        model: Model slug. Falls back to the provider's default when ``None``.
        mp_api_key: Materials Project API key. Falls back to environment variable.
        concurrency: Maximum simultaneous in-flight requests.

    Returns:
        List of :class:`~adam_identification.models.Material` or
        :class:`Exception` instances, one per query (order preserved).
    """
    from adam_identification.database.materials_project import MaterialsProjectClient
    from adam_identification.database.pubchem import PubChemClient
    from adam_identification.identifier import MaterialIdentifier
    from adam_identification.llm import get_provider

    llm = get_provider(provider, model)
    mp_client = MaterialsProjectClient(api_key=mp_api_key)
    pubchem_client = PubChemClient()
    identifier = MaterialIdentifier(llm, mp_client, pubchem_client)

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_identify_one(q, identifier, semaphore) for q in queries]
    results: list[Material | Exception] = []
    raw = await asyncio.gather(*tasks, return_exceptions=True)
    for item in raw:
        results.append(item)  # type: ignore[arg-type]
    return results


def batch_identify(
    queries: list[str],
    *,
    provider: str = "gemini",
    model: str | None = None,
    mp_api_key: str | None = None,
    concurrency: int = 5,
    output: Literal["ase", "pymatgen", "material"] = "ase",
) -> list["ase.Atoms | object | Material | Exception"]:
    """Identify multiple materials concurrently (synchronous wrapper).

    Args:
        queries: List of natural-language material descriptions.
        provider: LLM provider key (``"openai"``, ``"anthropic"``,
            ``"gemini"``, ``"openrouter"``).
        model: Model slug. Falls back to the provider's default when ``None``.
        mp_api_key: Materials Project API key. Falls back to environment variable.
        concurrency: Maximum simultaneous in-flight requests.
        output: Output type — ``"ase"`` (default), ``"pymatgen"``, or
            ``"material"`` (raw :class:`~adam_identification.models.Material`).

    Returns:
        List of converted objects or :class:`Exception` instances (order
        preserved). Failed identifications are returned as exceptions, not
        raised, so the rest of the batch still completes.

    Example::

        results = batch_identify(["silicon", "water", "caffeine"])
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
        )
    )

    if output == "material":
        return raw_results

    from adam_identification.output import to_ase, to_pymatgen

    converted: list[object] = []
    for item in raw_results:
        if isinstance(item, Exception):
            converted.append(item)
        else:
            try:
                if output == "ase":
                    converted.append(to_ase(item))
                else:
                    converted.append(to_pymatgen(item))
            except Exception as exc:
                logger.warning("Output conversion failed: %s", exc)
                converted.append(exc)
    return converted
