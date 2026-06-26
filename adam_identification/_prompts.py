"""Render Jinja2 prompt templates for adam-identification.

Thin wrapper around Jinja2 that loads templates from the bundled
``adam_identification/templates/`` directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

_BUNDLED_ROOT = Path(__file__).resolve().parent / "templates"


class PromptLoader:
    """Load and render prompt templates from the bundled templates directory.

    Args:
        _template_root: Override path for tests. If omitted, uses the bundled
            ``adam_identification/templates/`` directory.

    Raises:
        FileNotFoundError: If the resolved template root does not exist.
    """

    def __init__(self, _template_root: Path | None = None) -> None:
        root = _template_root or _BUNDLED_ROOT
        if not root.exists():
            raise FileNotFoundError(f"Prompt template directory not found: {root}")

        self._template_root = root
        self._env = Environment(
            loader=FileSystemLoader(str(root)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )

    @property
    def template_root(self) -> Path:
        """Return the template root directory."""
        return self._template_root

    def render(self, template_name: str, **context: Any) -> str:
        """Render a template using the provided context.

        Args:
            template_name: Path relative to the template root, e.g.
                ``"identification/formula_extraction.j2"``.
            **context: Template variables.

        Returns:
            Rendered prompt text.

        Raises:
            TemplateNotFound: If ``template_name`` does not exist.
        """
        template = self._env.get_template(template_name)
        return template.render(**context).strip()


__all__ = ["PromptLoader", "TemplateNotFound"]
