"""High-throughput batch identification for adam-identification.

Identifies multiple materials concurrently using the async LLM layer. A
semaphore limits simultaneous in-flight requests to avoid API rate limits.

Each query creates and binds its own :class:`~adam_identification.provenance.trace.Trace`
inside the worker thread (``run_in_executor`` does not copy ContextVars).

Example::

    from adam_identification import batch_identify

    results = batch_identify(
        ["silicon", "water", "caffeine", "iron (bcc)"],
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
from pathlib import Path
from typing import Literal

from adam_identification.models import Material
from adam_identification.result import IdentificationResult

logger = logging.getLogger(__name__)


def _safe_stem(query: str, *, limit: int = 40) -> str:
    """Filesystem-safe stem derived from a query string."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in query).strip("_")
    return (safe or "query")[:limit]


def _query_work_dir(work_dir: Path, index: int, query: str) -> Path:
    """Per-query subdirectory under a batch work dir."""
    return Path(work_dir) / f"{index:03d}_{_safe_stem(query)}"


def _run_one(
    query: str,
    index: int,
    *,
    provider: str | None,
    model: str | None,
    mp_api_key: str | None,
    output: Literal["ase", "pymatgen", "material", "result"],
    minimal_interaction: bool,
    work_dir: Path | None,
    save_dir: Path | None,
) -> Material | IdentificationResult:
    """Identify one query under a fresh Trace bound in this worker thread."""
    from adam_identification.output import to_ase
    from adam_identification.provenance.lifecycle import run_identification

    query_dir = _query_work_dir(work_dir, index, query) if work_dir is not None else None
    save_stem = f"{index:03d}_{_safe_stem(query)}" if save_dir is not None else None
    material, trace = run_identification(
        query,
        provider=provider,
        model=model,
        mp_api_key=mp_api_key,
        minimal_interaction=minimal_interaction,
        work_dir=query_dir,
        write_artifacts=query_dir is not None or save_dir is not None,
        save_dir=save_dir,
        save_stem=save_stem,
    )
    if output == "result":
        return IdentificationResult(atoms=to_ase(material), material=material, trace=trace)
    return material


async def _identify_one(
    query: str,
    index: int,
    semaphore: asyncio.Semaphore,
    *,
    provider: str | None,
    model: str | None,
    mp_api_key: str | None,
    output: Literal["ase", "pymatgen", "material", "result"],
    minimal_interaction: bool,
    work_dir: Path | None,
    save_dir: Path | None,
) -> Material | IdentificationResult:
    """Identify a single query under a shared concurrency semaphore."""
    async with semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: _run_one(
                query,
                index,
                provider=provider,
                model=model,
                mp_api_key=mp_api_key,
                output=output,
                minimal_interaction=minimal_interaction,
                work_dir=work_dir,
                save_dir=save_dir,
            ),
        )


async def batch_identify_async(
    queries: list[str],
    *,
    provider: str | None = None,
    model: str | None = None,
    mp_api_key: str | None = None,
    concurrency: int = 5,
    output: Literal["ase", "pymatgen", "material", "result"] = "material",
    minimal_interaction: bool = False,
    work_dir: Path | None = None,
    save_dir: Path | None = None,
) -> list[Material | IdentificationResult | Exception]:
    """Identify multiple materials concurrently (async version).

    Args:
        queries: List of natural-language material descriptions.
        provider: LLM provider key (``"openai"``, ``"anthropic"``,
            ``"google"``, ``"openrouter"``). Omitted: inferred from ``model``.
        model: Required model slug. There is no default.
        mp_api_key: Materials Project API key. Falls back to environment variable.
        concurrency: Maximum simultaneous in-flight requests.
        output: ``"result"`` returns a trace; other values return ``Material``.
        minimal_interaction: When ``True``, always select and flag
            ``needs_review`` instead of raising on ambiguity.
        work_dir: When set, persist each query under a numbered subdirectory.
        save_dir: Optional extra directory for structure files (CLI ``--save-dir``).

    Returns:
        List of :class:`~adam_identification.models.Material`,
        :class:`~adam_identification.result.IdentificationResult`, or
        :class:`Exception` instances, one per query (order preserved).
    """
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        _identify_one(
            q,
            i,
            semaphore,
            provider=provider,
            model=model,
            mp_api_key=mp_api_key,
            output=output,
            minimal_interaction=minimal_interaction,
            work_dir=work_dir,
            save_dir=save_dir,
        )
        for i, q in enumerate(queries, start=1)
    ]
    results: list[Material | IdentificationResult | Exception] = []
    raw = await asyncio.gather(*tasks, return_exceptions=True)
    for item in raw:
        results.append(item)  # type: ignore[arg-type]
    return results


def batch_identify(
    queries: list[str],
    *,
    provider: str | None = None,
    model: str | None = None,
    mp_api_key: str | None = None,
    concurrency: int = 5,
    output: Literal["ase", "pymatgen", "material", "result"] = "ase",
    minimal_interaction: bool = False,
    work_dir: Path | None = None,
    save_dir: Path | None = None,
) -> list[object]:
    """Identify multiple materials concurrently (synchronous wrapper).

    Args:
        queries: List of natural-language material descriptions.
        provider: LLM provider key (``"openai"``, ``"anthropic"``,
            ``"google"``, ``"openrouter"``). Omitted: inferred from ``model``.
        model: Required model slug. There is no default.
        mp_api_key: Materials Project API key. Falls back to environment variable.
        concurrency: Maximum simultaneous in-flight requests.
        output: Output type — ``"ase"`` (default), ``"pymatgen"``,
            ``"material"`` (raw :class:`~adam_identification.models.Material`),
            or ``"result"``
            (:class:`~adam_identification.result.IdentificationResult`).
        minimal_interaction: When ``True``, always select and flag
            ``needs_review`` instead of raising on ambiguity.
        work_dir: When set, persist each query under a numbered subdirectory.
        save_dir: Optional extra directory for structure files.

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
            work_dir=work_dir,
            save_dir=save_dir,
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
