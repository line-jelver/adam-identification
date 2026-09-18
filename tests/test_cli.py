"""CLI argument validation that must happen before creating a work directory."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from adam_identification.cli import app

runner = CliRunner()


def test_missing_batch_file_does_not_create_work_dir(tmp_path: Path) -> None:
    work = tmp_path / "work"
    missing = tmp_path / "does-not-exist.txt"
    result = runner.invoke(
        app,
        ["--batch", str(missing), "--model", "gemini-3.1-pro-preview", "--work-dir", str(work)],
    )
    assert result.exit_code == 1
    assert "Batch file not found" in result.output
    assert not work.exists()


def test_empty_batch_file_does_not_create_work_dir(tmp_path: Path) -> None:
    work = tmp_path / "work"
    batch = tmp_path / "empty.txt"
    batch.write_text("\n\n  \n")
    result = runner.invoke(
        app,
        ["--batch", str(batch), "--model", "gemini-3.1-pro-preview", "--work-dir", str(work)],
    )
    assert result.exit_code == 1
    assert "contains no queries" in result.output
    assert not work.exists()


def test_unknown_model_without_provider_fails_to_infer(tmp_path: Path) -> None:
    work = tmp_path / "work"
    result = runner.invoke(
        app,
        ["silicon", "--model", "foo-bar", "--work-dir", str(work)],
    )
    assert result.exit_code == 1
    assert "Cannot infer" in result.output
