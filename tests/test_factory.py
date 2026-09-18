"""Unit tests for the LLM provider factory."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from adam_identification._config import ConfigurationError, Settings
from adam_identification.llm import get_provider, infer_provider
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
    with pytest.raises(ConfigurationError, match="no default"):
        get_provider()


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


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-3.8-flash", "google"),
        ("gemini-3.1-pro-preview", "google"),
        ("claude-opus-5", "anthropic"),
        ("claude-haiku-4-5-20251001", "anthropic"),
        ("gpt-5.9-mini", "openai"),
        ("gpt-5.4-mini", "openai"),
        ("chatgpt-4o", "openai"),
        ("o1-preview", "openai"),
        ("o3-mini", "openai"),
        ("o4-mini", "openai"),
        ("moonshotai/kimi-k2.6", "openrouter"),
        ("deepseek/deepseek-chat-v3-5", "openrouter"),
        ("openai/gpt-6", "openai"),
        ("openai/gpt-4o", "openai"),
        ("google/gemini-2.5-flash", "google"),
        ("anthropic/claude-sonnet-4-6", "anthropic"),
    ],
)
def test_infer_provider_from_new_and_known_slugs(model: str, expected: str) -> None:
    assert infer_provider(model) == expected


def test_infer_provider_rejects_unknown_slug() -> None:
    with pytest.raises(ConfigurationError, match="Cannot infer"):
        infer_provider("deepseek-v4-pro")
    with pytest.raises(ConfigurationError, match="Cannot infer"):
        infer_provider("foo-bar")


def test_factory_infers_provider_from_model() -> None:
    with _patch_all_settings(_settings()):
        assert isinstance(get_provider(model="gemini-3.8-flash"), GeminiProvider)
        assert isinstance(get_provider(model="gpt-5.9-mini"), OpenAIProvider)
        assert isinstance(get_provider(model="claude-opus-5"), AnthropicProvider)
        assert isinstance(get_provider(model="moonshotai/kimi-k2.6"), OpenRouterProvider)
        openai = get_provider(model="openai/gpt-6")
        assert isinstance(openai, OpenAIProvider)
        assert openai._model == "gpt-6"  # noqa: SLF001
        gemini = get_provider(model="google/gemini-3.8-flash")
        assert isinstance(gemini, GeminiProvider)
        assert gemini._model == "gemini-3.8-flash"  # noqa: SLF001


def test_factory_openrouter_keeps_native_org_slug() -> None:
    with _patch_all_settings(_settings()):
        routed = get_provider("openrouter", model="openai/gpt-6")
    assert isinstance(routed, OpenRouterProvider)
    assert routed._model == "openai/gpt-6"  # noqa: SLF001


def test_factory_openrouter_override_keeps_native_slug() -> None:
    with _patch_all_settings(_settings()):
        gpt = get_provider("openrouter", model="gpt-5.4-mini")
        gemini = get_provider("openrouter", model="gemini-2.5-flash")
    assert isinstance(gpt, OpenRouterProvider)
    assert gpt._model == "gpt-5.4-mini"  # noqa: SLF001
    assert isinstance(gemini, OpenRouterProvider)
    assert gemini._model == "gemini-2.5-flash"  # noqa: SLF001


def test_factory_explicit_provider_wins_over_inference() -> None:
    with _patch_all_settings(_settings()):
        provider = get_provider("google", model="gpt-4o")
    assert isinstance(provider, GeminiProvider)
    assert provider._model == "gpt-4o"  # noqa: SLF001
