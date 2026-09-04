"""Command-line interface for adam-identification.

Usage::

    adam-identify "silicon" --model gemini-3.1-pro-preview
    adam-identify "glucose" --model gemini-3.1-pro-preview --save glucose.xyz
    adam-identify "BCC iron" --provider anthropic --model claude-haiku-4-5-20251001
    adam-identify --batch queries.txt --model gemini-3.1-pro-preview --save-dir results/
    adam-identify --list-providers
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from adam_identification import __version__
from adam_identification._logging import configure_logging

app = typer.Typer(
    name="adam-identify",
    help="Identify atomistic structures from natural-language descriptions.",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()
err_console = Console(stderr=True)

_PROVIDERS = {
    "google": "Google Gemini — set GOOGLE_API_KEY",
    "openai": "OpenAI — set OPENAI_API_KEY",
    "anthropic": "Anthropic Claude — set ANTHROPIC_API_KEY",
    "openrouter": "OpenRouter (300+ models) — set OPENROUTER_API_KEY",
}


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"adam-identification {__version__}")
        raise typer.Exit()


def _list_providers_callback(value: bool) -> None:
    if value:
        table = Table(title="Available LLM providers", show_header=True)
        table.add_column("Provider key", style="cyan")
        table.add_column("Description")
        for key, desc in _PROVIDERS.items():
            table.add_row(key, desc)
        console.print(table)
        raise typer.Exit()


@app.command()
def identify_cmd(
    query: Annotated[
        str | None,
        typer.Argument(help="Natural-language material description, e.g. 'silicon' or 'caffeine'."),
    ] = None,
    batch: Annotated[
        Path | None,
        typer.Option("--batch", "-b", help="Text file with one query per line (batch mode)."),
    ] = None,
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help="LLM provider: google, openai, anthropic, openrouter.",
        ),
    ] = "google",
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model slug (required for identification). There is no default.",
        ),
    ] = None,
    save: Annotated[
        Path | None,
        typer.Option(
            "--save",
            "-s",
            help="Save structure to file (CIF/XYZ/POSCAR detected by extension).",
        ),
    ] = None,
    save_dir: Annotated[
        Path | None,
        typer.Option("--save-dir", help="Directory for batch output files (one per query)."),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", "-c", help="Max concurrent requests in batch mode."),
    ] = 5,
    mp_api_key: Annotated[
        str | None,
        typer.Option("--mp-api-key", help="Materials Project API key (overrides env var)."),
    ] = None,
    minimal_interaction: Annotated[
        bool,
        typer.Option(
            "--minimal-interaction",
            help="Always select a candidate; flag uncertain choices with needs_review.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show DEBUG logs and the full candidate list."),
    ] = False,
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = None,
    list_providers: Annotated[
        bool | None,
        typer.Option(
            "--list-providers",
            callback=_list_providers_callback,
            is_eager=True,
            help="List available LLM providers and exit.",
        ),
    ] = None,
) -> None:
    """Identify atomistic structures from natural-language descriptions.

    Single query example:

        adam-identify "silicon" --model gemini-3.1-pro-preview

    Batch example (one query per line in queries.txt):

        adam-identify --batch queries.txt --model gemini-3.1-pro-preview

    """
    configure_logging(verbose=verbose)

    if query is None and batch is None:
        err_console.print("[red]Error:[/red] Provide a QUERY argument or --batch FILE.")
        raise typer.Exit(1)

    if not model or not model.strip():
        from adam_identification.llm import MISSING_MODEL_MESSAGE

        err_console.print(f"[red]Error:[/red] {MISSING_MODEL_MESSAGE}")
        raise typer.Exit(1)

    if batch is not None:
        _run_batch(
            batch_file=batch,
            provider=provider,
            model=model,
            save_dir=save_dir,
            concurrency=concurrency,
            mp_api_key=mp_api_key,
            minimal_interaction=minimal_interaction,
        )
    else:
        assert query is not None
        _run_single(
            query=query,
            provider=provider,
            model=model,
            save=save,
            mp_api_key=mp_api_key,
            minimal_interaction=minimal_interaction,
            verbose=verbose,
        )


def _run_single(
    query: str,
    provider: str,
    model: str,
    save: Path | None,
    mp_api_key: str | None,
    minimal_interaction: bool,
    verbose: bool,
) -> None:
    from adam_identification.database.materials_project import MaterialsProjectClient
    from adam_identification.database.pubchem import PubChemClient
    from adam_identification.exceptions import (
        AmbiguousIdentificationError,
        ClarificationNeededError,
    )
    from adam_identification.identifier import MaterialIdentifier
    from adam_identification.llm import get_provider as _get_provider
    from adam_identification.trace import IdentificationTrace

    llm = _get_provider(provider, model)
    mp_client = MaterialsProjectClient(api_key=mp_api_key)
    pubchem_client = PubChemClient()
    identifier = MaterialIdentifier(
        llm, mp_client, pubchem_client, minimal_interaction=minimal_interaction
    )

    # In verbose mode skip the spinner so DEBUG log lines aren't overwritten.
    # Otherwise run inside a status spinner, but capture any exception and only
    # print it *after* the spinner stops to prevent output collisions.
    _caught: Exception | None = None
    if verbose:
        console.print(f"[bold]Identifying:[/bold] {query}")
        try:
            material, trace = identifier.identify(query, return_trace=True)
        except Exception as _exc:
            _caught = _exc
    else:
        with console.status(f"[bold]Identifying:[/bold] {query}"):
            try:
                material, trace = identifier.identify(query, return_trace=True)
            except Exception as _exc:
                _caught = _exc

    # Spinner has now stopped — safe to print without line collisions.
    if _caught is not None:
        err_console.print()  # blank line separates from the spinner/header line
        if isinstance(_caught, AmbiguousIdentificationError):
            _print_ambiguous(_caught, verbose=verbose)
            raise typer.Exit(1) from _caught
        if isinstance(_caught, ClarificationNeededError):
            err_console.print(f"[yellow]Clarification needed:[/yellow] {_caught}")
            raise typer.Exit(1) from _caught
        err_console.print(f"[red]Error:[/red] {_caught}")
        raise typer.Exit(1) from _caught

    if not isinstance(trace, IdentificationTrace):
        trace = IdentificationTrace(query=query, domain="crystal")  # type: ignore[arg-type]
    _print_material(material, query, trace=trace)

    if save is not None:
        _save_material(material, save)


def _run_batch(
    batch_file: Path,
    provider: str,
    model: str,
    save_dir: Path | None,
    concurrency: int,
    mp_api_key: str | None,
    minimal_interaction: bool,
) -> None:
    from adam_identification.batch import batch_identify
    from adam_identification.models import Material

    if not batch_file.exists():
        err_console.print(f"[red]Error:[/red] Batch file not found: {batch_file}")
        raise typer.Exit(1)

    queries = [line.strip() for line in batch_file.read_text().splitlines() if line.strip()]
    if not queries:
        err_console.print("[red]Error:[/red] Batch file contains no queries.")
        raise typer.Exit(1)

    console.print(
        f"Batch: [cyan]{len(queries)}[/cyan] queries, concurrency=[cyan]{concurrency}[/cyan]"
    )

    results = batch_identify(
        queries,
        provider=provider,
        model=model,
        mp_api_key=mp_api_key,
        concurrency=concurrency,
        output="material",
        minimal_interaction=minimal_interaction,
    )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failure = 0
    table = Table(title=f"Batch results ({len(queries)} queries)", show_header=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Query")
    table.add_column("Result")
    table.add_column("ID", style="dim")

    for i, (q, res) in enumerate(zip(queries, results)):
        if isinstance(res, Exception):
            failure += 1
            table.add_row(str(i + 1), q, f"[red]FAILED: {res}[/red]", "")
        else:
            success += 1
            mat: Material = res  # type: ignore[assignment]
            if mat.material_type == "crystal":
                struct = mat.structure
                sg = getattr(struct, "space_group", "?")
                result_str = f"[green]{mat.chemical_formula}[/green] ({sg})"
            else:
                result_str = f"[green]{mat.chemical_formula}[/green] (molecule)"
            db_id = mat.mp_id or (f"CID {mat.pc_cid}" if mat.pc_cid else "")
            table.add_row(str(i + 1), q, result_str, db_id)

            if save_dir is not None:
                safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in q)
                ext = ".cif" if mat.material_type == "crystal" else ".xyz"
                _save_material(mat, save_dir / f"{i + 1:03d}_{safe_name}{ext}")

    console.print(table)
    console.print(
        f"\n[green]{success} succeeded[/green], [red]{failure} failed[/red] "
        f"out of {len(queries)} queries."
    )

    if failure > 0:
        raise typer.Exit(1)


def _print_ambiguous(exc: object, *, verbose: bool = False) -> None:
    """Print a Rich table of ambiguous identification options."""
    from adam_identification.exceptions import AmbiguousIdentificationError

    if not isinstance(exc, AmbiguousIdentificationError):
        err_console.print(f"[red]Error:[/red] {exc}")
        return

    err_console.print("[yellow]Ambiguous identification[/yellow]")
    if exc.user_message:
        err_console.print(exc.user_message)

    shown = exc.candidates_for_display(verbose=verbose)
    table = Table(title="Candidate options", show_header=True)
    if exc.domain == "crystal":
        table.add_column("MP ID", style="cyan")
        table.add_column("Formula")
        table.add_column("Space group")
        table.add_column("Sites", justify="right")
        for candidate in shown:
            table.add_row(
                str(candidate.get("mp_id") or ""),
                str(candidate.get("formula") or ""),
                str(candidate.get("space_group") or ""),
                str(candidate.get("nsites") if candidate.get("nsites") is not None else ""),
            )
    else:
        table.add_column("Name")
        table.add_column("CID", style="cyan")
        table.add_column("Formula")
        for candidate in shown:
            table.add_row(
                str(candidate.get("name") or ""),
                str(candidate.get("cid") or ""),
                str(candidate.get("formula") or ""),
            )
    err_console.print(table)
    if len(shown) < len(exc.candidates):
        err_console.print(
            f"[dim]Showing {len(shown)} of {len(exc.candidates)} database entries. "
            "Use --verbose for the full list.[/dim]"
        )

    if exc.suggested_candidates:
        err_console.print("[bold]Suggested follow-up queries:[/bold]")
        for suggestion in exc.suggested_candidates:
            err_console.print(f"  - {suggestion.get('suggested_query', '')}")
    err_console.print("Resume with IdentificationSession, or re-run with a more specific query.")


def _print_material(material: object, query: str, *, trace: object | None = None) -> None:
    from adam_identification.models import Material

    if not isinstance(material, Material):
        console.print(material)
        return

    struct = material.structure
    if material.material_type == "crystal":
        lp = getattr(struct, "lattice_parameters", {})
        a = lp.get("a", "?")
        b = lp.get("b", "?")
        c = lp.get("c", "?")
        alpha = lp.get("alpha", "?")
        beta = lp.get("beta", "?")
        gamma = lp.get("gamma", "?")
        sg = getattr(struct, "space_group", "?")
        cs = getattr(struct, "crystal_system", "?")
        nsites = getattr(struct, "nsites", "?")
        db_id = material.mp_id or "?"

        props = material.get_properties(material.source) if material.properties else None
        eah_str = ""
        bg_str = ""
        if props is not None:
            eah = getattr(props, "energy_above_hull", None)
            if isinstance(eah, int | float):
                eah_str = f"\n  Energy above hull: {eah:.3f} eV/atom"
            bg = getattr(props, "band_gap", None)
            if isinstance(bg, int | float):
                bg_str = f"\n  Band gap: {bg:.2f} eV"

        text = (
            f"[bold green]Crystal: {material.chemical_formula}[/bold green]\n"
            f"  Materials Project ID: [cyan]{db_id}[/cyan]\n"
            f"  Space group: {sg} ({cs})\n"
            f"  Lattice: a={a:.4g}Å  b={b:.4g}Å  c={c:.4g}Å"
            f"   α={alpha:.4g}°  β={beta:.4g}°  γ={gamma:.4g}°\n"
            f"  Sites: {nsites}"
            f"{eah_str}{bg_str}"
        )
    else:
        nsites = getattr(struct, "nsites", len(struct.atomic_positions))
        smiles = getattr(struct, "smiles", None) or "?"
        cid_str = f"CID {material.pc_cid}" if material.pc_cid else "?"
        text = (
            f"[bold green]Molecule: {material.name}[/bold green]\n"
            f"  Formula: {material.chemical_formula}\n"
            f"  PubChem CID: [cyan]{cid_str}[/cyan]\n"
            f"  SMILES: {smiles}\n"
            f"  Atoms: {nsites}"
        )

    console.print(Panel(text, title=f"[dim]{query}[/dim]", expand=False))

    needs_review = bool(getattr(trace, "needs_review", False))
    if needs_review:
        reason = str(getattr(trace, "selection_reason", "") or "").strip()
        suggestions = getattr(trace, "suggested_candidates", None) or []
        alt_parts = [
            str(item.get("suggested_query", "")).strip()
            for item in suggestions
            if isinstance(item, dict) and item.get("suggested_query")
        ]
        review_text = (
            "[yellow]Needs review[/yellow]: this pick used a conventional assumption "
            "because the query did not uniquely identify one candidate."
        )
        if reason:
            review_text += f"\n  {reason}"
        if alt_parts:
            review_text += "\n  Alternatives: " + "; ".join(alt_parts)
        console.print(Panel(review_text, title="minimal-interaction", expand=False))


def _save_material(material: object, path: Path) -> None:
    from adam_identification.models import Material
    from adam_identification.output import to_ase

    if not isinstance(material, Material):
        return

    try:
        import ase.io as ase_io
    except ImportError:
        err_console.print("[yellow]Warning:[/yellow] ase not available; skipping save.")
        return

    try:
        atoms = to_ase(material)
        ext = path.suffix.lower()
        if ext == ".cif":
            fmt = "cif"
        elif ext in (".xyz", ".extxyz"):
            fmt = "extxyz"
        elif path.name in ("POSCAR", "CONTCAR") or ext == ".vasp":
            fmt = "vasp"
        else:
            fmt = None  # let ASE auto-detect
        ase_io.write(str(path), atoms, format=fmt)
        console.print(f"  Saved: [cyan]{path}[/cyan]")
        if ext == ".cif":
            console.print(
                "  [dim]Note: ASE exports the primitive cell (often as P1), "
                "not the conventional crystallographic cell.[/dim]"
            )
    except Exception as exc:
        err_console.print(f"[yellow]Warning:[/yellow] Could not save to {path}: {exc}")


def main() -> None:
    """Entry point for the ``adam-identify`` CLI command."""
    app()


if __name__ == "__main__":
    main()
