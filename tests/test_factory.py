"""Unit tests for the LLM provider factory."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from adam_identification._config import ConfigurationError, Settings
from adam_identification.llm import get_provider
from adam_identification.llm.providers import anthropic as anthropic_module
from adam_identification.llm.providers import gemini as gemini_module
from adam_identification.llm.providers import openai as openai_module
from adam_identification.llm.providers import openrouter as openrouter_module
from adam_identification.llm.providers.anthropic import AnthropicProvider
from adam_identification.llm.providers.gemini import GeminiProvider
from adam_identification.llm.providers.openai import OpenAIProvider
from adam_identification.llm.providers.openrouter import OpenRouterProvider


def _settings() -> Settings:
    return Settings(
        openai_api_key="sk-test",
        anthropic_api_key="sk-ant-test",
        google_api_key="google-test",
        openrouter_api_key="or-test",
        materials_project_api_key="mp-test",
    )


@contextmanager
def _patch_all_settings(s: Settings) -> Iterator[None]:
    with (
        patch.object(openai_module, "settings", s),
        patch.object(anthropic_module, "settings", s),
        patch.object(gemini_module, "settings", s),
        patch.object(openrouter_module, "settings", s),
    ):
        yield


def test_factory_returns_providers() -> None:
    with _patch_all_settings(_settings()):
        assert isinstance(get_provider("openai", model="gpt-4o-mini"), OpenAIProvider)
        assert isinstance(
            get_provider("anthropic", model="claude-haiku-4-5-20251001"), AnthropicProvider
        )
        assert isinstance(get_provider("google", model="gemini-3.1-pro-preview"), GeminiProvider)
        assert isinstance(
            get_provider("openrouter", model="deepseek/deepseek-chat-v3-5"), OpenRouterProvider
        )


def test_factory_requires_model() -> None:
    with pytest.raises(ConfigurationError, match="no default"):
        get_provider("google")
    with pytest.raises(ConfigurationError, match="no default"):
        get_provider("openai", model="  ")


def test_factory_accepts_model_override() -> None:
    with _patch_all_settings(_settings()):
        provider = get_provider("google", model="gemini-2.5-flash")
    assert isinstance(provider, GeminiProvider)
    assert provider._model == "gemini-2.5-flash"  # noqa: SLF001


def test_factory_rejects_gemini_alias() -> None:
    with pytest.raises(ConfigurationError, match="google"):
        get_provider("gemini", model="gemini-3.1-pro-preview")
    with pytest.raises(ConfigurationError, match="google"):
        get_provider("bogus", model="anything")
