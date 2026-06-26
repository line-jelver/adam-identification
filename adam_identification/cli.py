"""Command-line interface for adam-identification.

Usage::

    adam-identify "silicon"
    adam-identify "glucose" --output pymatgen --save glucose.xyz
    adam-identify "BCC iron" --provider anthropic --model claude-haiku-4-5-20251001
    adam-identify --batch queries.txt --concurrency 10 --save-dir results/
    adam-identify --list-providers
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Optional

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
    "gemini": "Google Gemini — set GOOGLE_API_KEY",
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
        Optional[str],
        typer.Argument(help="Natural-language material description, e.g. 'silicon' or 'caffeine'."),
    ] = None,
    batch: Annotated[
        Optional[Path],
        typer.Option("--batch", "-b", help="Text file with one query per line (batch mode)."),
    ] = None,
    provider: Annotated[
        str,
        typer.Option("--provider", "-p", help="LLM provider: gemini, openai, anthropic, openrouter."),
    ] = "gemini",
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model slug (uses provider default if omitted)."),
    ] = None,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Output format: info (default), ase, pymatgen, json.",
        ),
    ] = "info",
    save: Annotated[
        Optional[Path],
        typer.Option("--save", "-s", help="Save structure to file (CIF/XYZ/POSCAR detected by extension)."),
    ] = None,
    save_dir: Annotated[
        Optional[Path],
        typer.Option("--save-dir", help="Directory for batch output files (one per query)."),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", "-c", help="Max concurrent requests in batch mode."),
    ] = 5,
    mp_api_key: Annotated[
        Optional[str],
        typer.Option("--mp-api-key", help="Materials Project API key (overrides env var)."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show DEBUG-level logs."),
    ] = False,
    version: Annotated[
        Optional[bool],
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = None,
    list_providers: Annotated[
        Optional[bool],
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

        adam-identify "silicon"

    Batch example (one query per line in queries.txt):

        adam-identify --batch queries.txt --concurrency 5

    """
    configure_logging(verbose=verbose)

    if query is None and batch is None:
        err_console.print("[red]Error:[/red] Provide a QUERY argument or --batch FILE.")
        raise typer.Exit(1)

    if batch is not None:
        _run_batch(
            batch_file=batch,
            provider=provider,
            model=model,
            output=output,
            save_dir=save_dir,
            concurrency=concurrency,
            mp_api_key=mp_api_key,
        )
    else:
        assert query is not None
        _run_single(
            query=query,
            provider=provider,
            model=model,
            output=output,
            save=save,
            mp_api_key=mp_api_key,
        )


def _run_single(
    query: str,
    provider: str,
    model: str | None,
    output: str,
    save: Path | None,
    mp_api_key: str | None,
) -> None:
    from adam_identification.database.materials_project import MaterialsProjectClient
    from adam_identification.database.pubchem import PubChemClient
    from adam_identification.identifier import MaterialIdentifier
    from adam_identification.llm import get_provider as _get_provider

    with console.status(f"[bold]Identifying:[/bold] {query}"):
        try:
            llm = _get_provider(provider, model)
            mp_client = MaterialsProjectClient(api_key=mp_api_key)
            pubchem_client = PubChemClient()
            identifier = MaterialIdentifier(llm, mp_client, pubchem_client)
            material = identifier.identify(query)
        except Exception as exc:
            err_console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc

    _print_material(material, query)

    if save is not None:
        _save_material(material, save)

    if output in ("ase", "pymatgen"):
        # Used programmatically; not useful in CLI-only context but allow as a flag.
        pass


def _run_batch(
    batch_file: Path,
    provider: str,
    model: str | None,
    output: str,
    save_dir: Path | None,
    concurrency: int,
    mp_api_key: str | None,
) -> None:
    from adam_identification.batch import batch_identify
    from adam_identification.models import Material

    if not batch_file.exists():
        err_console.print(f"[red]Error:[/red] Batch file not found: {batch_file}")
        raise typer.Exit(1)

    queries = [
        line.strip() for line in batch_file.read_text().splitlines() if line.strip()
    ]
    if not queries:
        err_console.print("[red]Error:[/red] Batch file contains no queries.")
        raise typer.Exit(1)

    console.print(f"Batch: [cyan]{len(queries)}[/cyan] queries, concurrency=[cyan]{concurrency}[/cyan]")

    raw_output = "material"
    results = batch_identify(
        queries,
        provider=provider,
        model=model,
        mp_api_key=mp_api_key,
        concurrency=concurrency,
        output=raw_output,
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
                _save_material(mat, save_dir / f"{i+1:03d}_{safe_name}{ext}")

    console.print(table)
    console.print(
        f"\n[green]{success} succeeded[/green], [red]{failure} failed[/red] "
        f"out of {len(queries)} queries."
    )

    if failure > 0:
        raise typer.Exit(1)


def _print_material(material: "object", query: str) -> None:
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
            if getattr(props, "energy_above_hull", None) is not None:
                eah_str = f"\n  Energy above hull: {props.energy_above_hull:.3f} eV/atom"
            if getattr(props, "band_gap", None) is not None:
                bg_str = f"\n  Band gap: {props.band_gap:.2f} eV"

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


def _save_material(material: "object", path: Path) -> None:
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
    except Exception as exc:
        err_console.print(f"[yellow]Warning:[/yellow] Could not save to {path}: {exc}")


def main() -> None:
    """Entry point for the ``adam-identify`` CLI command."""
    app()


if __name__ == "__main__":
    main()
