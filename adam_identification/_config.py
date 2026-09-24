"""Configuration loader for adam-identification.

Reads API keys from a .env file or from environment variables already set in
the shell. The .env file is searched in the current working directory and any
parent directories, so it does not need to live next to the package.

Usage::

    from adam_identification._config import settings

    mp_key = settings.require_materials_project()  # raises if not set
    llm_key = settings.openai_api_key              # None if not set
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

# Search upward from cwd for a .env file; silently skip if none found.
_dotenv_path = find_dotenv(usecwd=True)
if _dotenv_path:
    load_dotenv(_dotenv_path)


class ConfigurationError(Exception):
    """Raised when a required configuration value is missing or invalid.

    Common causes:

    * ``model`` omitted from :func:`~adam_identification.llm.get_provider` or
      :func:`~adam_identification.identify`.
    * A required API key (``MATERIALS_PROJECT_API_KEY``, ``GOOGLE_API_KEY``,
      etc.) is not set in the environment or ``.env`` file.
    """


def _optional(key: str) -> str | None:
    return os.getenv(key) or None


@dataclass(frozen=True)
class Settings:
    """API key configuration for adam-identification.

    All keys are optional at construction time; provider constructors raise
    :class:`ConfigurationError` when their specific key is missing.

    Attributes:
        openai_api_key: OpenAI API key.
        anthropic_api_key: Anthropic API key.
        google_api_key: Google Gemini API key.
        openrouter_api_key: OpenRouter API key (access to 300+ models).
        materials_project_api_key: Materials Project API key. Required only
            for ``--crystal-source materials-project``, or included under
            ``auto`` when already set. MC3D needs no key. Get yours at
            https://next-gen.materialsproject.org/api
    """

    openai_api_key: str | None
    anthropic_api_key: str | None
    google_api_key: str | None
    openrouter_api_key: str | None
    materials_project_api_key: str | None

    def require_materials_project(self) -> str:
        """Return the MP API key, raising ConfigurationError if not set.

        Returns:
            The Materials Project API key string.

        Raises:
            ConfigurationError: If MATERIALS_PROJECT_API_KEY is not set.
        """
        if not self.materials_project_api_key:
            raise ConfigurationError(
                "MATERIALS_PROJECT_API_KEY is not set.\n"
                "Get your key at https://next-gen.materialsproject.org/api\n"
                "Then set it: export MATERIALS_PROJECT_API_KEY=your_key_here"
            )
        return self.materials_project_api_key


def _load_settings() -> Settings:
    return Settings(
        openai_api_key=_optional("OPENAI_API_KEY"),
        anthropic_api_key=_optional("ANTHROPIC_API_KEY"),
        google_api_key=_optional("GOOGLE_API_KEY"),
        openrouter_api_key=_optional("OPENROUTER_API_KEY"),
        materials_project_api_key=_optional("MATERIALS_PROJECT_API_KEY"),
    )


settings = _load_settings()
