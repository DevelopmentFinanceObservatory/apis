"""Tests model settings and stage-specific thinking levels.

A fake Gemini client keeps the suite offline; one test checks `offline_resolve` wiring.
"""

from __future__ import annotations

import pytest
from google.genai.types import ThinkingConfig, ThinkingLevel

from search.engine import config
from search.engine.clients.llm import (
    _STAGE_THINKING_LEVELS,
    LLM,
    _temperature_for_model,
    _thinking_config_for_model,
    for_stage,
)
from tests.engine._harness import make_request, offline_resolve
from tests.fixtures import FakeLLM


class TestGemini3AndEarly31:
    """Bare gemini-3 and 3.1: pinned temperature=0.0, MINIMAL thinking."""

    def test_bare_gemini_3_temperature(self):
        assert _temperature_for_model("gemini-3") == 0.0

    def test_bare_gemini_3_thinking(self):
        config = _thinking_config_for_model("gemini-3")
        assert config == ThinkingConfig(thinking_level=ThinkingLevel.MINIMAL)

    def test_gemini_3_1_flash_lite_temperature(self):
        assert _temperature_for_model("gemini-3.1-flash-lite") == 0.0

    def test_gemini_3_1_flash_lite_thinking(self):
        config = _thinking_config_for_model("gemini-3.1-flash-lite")
        assert config == ThinkingConfig(thinking_level=ThinkingLevel.MINIMAL)


class TestGemini35Plus:
    """Gemini 3.5+ (3.5, 3.6, ...): temperature omitted, LOW thinking."""

    def test_gemini_3_5_flash_lite_temperature(self):
        assert _temperature_for_model("gemini-3.5-flash-lite") is None

    def test_gemini_3_5_flash_lite_thinking(self):
        config = _thinking_config_for_model("gemini-3.5-flash-lite")
        assert config == ThinkingConfig(thinking_level=ThinkingLevel.LOW)

    def test_gemini_3_6_flash_temperature(self):
        assert _temperature_for_model("gemini-3.6-flash") is None

    def test_gemini_3_6_flash_thinking(self):
        config = _thinking_config_for_model("gemini-3.6-flash")
        assert config == ThinkingConfig(thinking_level=ThinkingLevel.LOW)

    def test_multi_digit_minor_gemini_3_10_treated_as_3_5_plus(self):
        """gemini-3.10 parses as minor=10 (>= 5) via the regex's multi-digit \\d+ group."""
        assert _temperature_for_model("gemini-3.10-flash") is None
        config = _thinking_config_for_model("gemini-3.10-flash")
        assert config == ThinkingConfig(thinking_level=ThinkingLevel.LOW)


class TestLatestAliases:
    """The -latest aliases resolve into the 3.5+ generation despite lacking '3'."""

    def test_gemini_flash_latest_temperature(self):
        assert _temperature_for_model("gemini-flash-latest") is None

    def test_gemini_flash_latest_thinking(self):
        config = _thinking_config_for_model("gemini-flash-latest")
        assert config == ThinkingConfig(thinking_level=ThinkingLevel.LOW)

    def test_gemini_flash_lite_latest_temperature(self):
        assert _temperature_for_model("gemini-flash-lite-latest") is None

    def test_gemini_flash_lite_latest_thinking(self):
        config = _thinking_config_for_model("gemini-flash-lite-latest")
        assert config == ThinkingConfig(thinking_level=ThinkingLevel.LOW)


class TestGemini25Era:
    """Pre-3.x models keep the thinking_budget=0 branch and pinned temperature."""

    def test_gemini_2_5_temperature(self):
        assert _temperature_for_model("gemini-2.5-flash") == 0.0

    def test_gemini_2_5_thinking(self):
        config = _thinking_config_for_model("gemini-2.5-flash")
        assert config == ThinkingConfig(thinking_budget=0, include_thoughts=False)


class TestUnknownModels:
    """Unrecognized names: gemma falls to None, other gemini-* falls to MINIMAL."""

    def test_gemma_style_name_temperature(self):
        # No gemini-3.x match, so the pinned-temperature default applies.
        assert _temperature_for_model("gemma-7b") == 0.0

    def test_gemma_style_name_thinking_is_none(self):
        assert _thinking_config_for_model("gemma-7b") is None

    def test_unknown_gemini_name_temperature(self):
        assert _temperature_for_model("gemini-future-model") == 0.0

    def test_unknown_gemini_name_thinking_defaults_to_minimal(self):
        config = _thinking_config_for_model("gemini-future-model")
        assert config == ThinkingConfig(thinking_level=ThinkingLevel.MINIMAL)


class TestThinkingLevelOverride:
    """level_override replaces the chosen level only where a ThinkingLevel-style
    config is the branch's return shape; gemma (None) and gemini-2.5
    (thinking_budget) are untouched by it.
    """

    def test_gemini_3_5_plus_override_applied(self):
        config = _thinking_config_for_model("gemini-3.5-flash-lite", "medium")
        assert config == ThinkingConfig(thinking_level=ThinkingLevel.MEDIUM)

    def test_latest_alias_override_applied(self):
        config = _thinking_config_for_model("gemini-flash-lite-latest", "high")
        assert config == ThinkingConfig(thinking_level=ThinkingLevel.HIGH)

    def test_bare_gemini_3_override_applied(self):
        config = _thinking_config_for_model("gemini-3", "low")
        assert config == ThinkingConfig(thinking_level=ThinkingLevel.LOW)

    def test_unknown_gemini_override_applied(self):
        config = _thinking_config_for_model("gemini-future-model", "minimal")
        assert config == ThinkingConfig(thinking_level=ThinkingLevel.MINIMAL)

    def test_gemma_override_not_applied(self):
        # Helper returns None for gemma regardless of level_override.
        assert _thinking_config_for_model("gemma-7b", "high") is None

    def test_gemini_2_5_override_not_applied(self):
        # thinking_budget branch keeps its shape regardless of level_override.
        config = _thinking_config_for_model("gemini-2.5-flash", "high")
        assert config == ThinkingConfig(thinking_budget=0, include_thoughts=False)

    def test_none_override_keeps_model_default(self):
        config = _thinking_config_for_model("gemini-3.5-flash-lite", None)
        assert config == ThinkingConfig(thinking_level=ThinkingLevel.LOW)


class TestLLMSeedConfigVsKwarg:
    """LLM.__init__'s seed/thinking_level: config supplies the process default,
    a non-None kwarg always overrides it.
    """

    def test_kwarg_seed_overrides_config_default(self, search_settings):
        search_settings(llm_seed=7)
        assert LLM(seed=42)._seed == 42

    def test_none_kwarg_falls_back_to_config_seed(self, search_settings):
        search_settings(llm_seed=7)
        assert LLM()._seed == 7

    def test_kwarg_thinking_level_overrides_config_default(self, search_settings):
        search_settings(llm_thinking_level="low")
        assert LLM(thinking_level="high")._thinking_level == "high"

    def test_none_kwarg_falls_back_to_config_thinking_level(self, search_settings):
        search_settings(llm_thinking_level="low")
        assert LLM()._thinking_level == "low"

    def test_generate_structured_uses_effective_thinking_level(self, monkeypatch):
        # LLM.generate_structured forwards self._thinking_level, the already-resolved
        # instance value, into _thinking_config_for_model.
        llm = LLM(model="gemini-3.5-flash-lite", thinking_level="high")
        assert _thinking_config_for_model(llm._model, llm._thinking_level) == ThinkingConfig(
            thinking_level=ThinkingLevel.HIGH
        )


class TestForStageIdentity:
    """`for_stage` preserves identity when a stage has no level or its input is not an `LLM`."""

    def test_llm_instance_returned_unchanged(self):
        llm = LLM()
        assert for_stage(llm, "extract") is llm

    @pytest.mark.parametrize("stage", ["extract", "bind"])
    def test_fake_llm_returned_unchanged(self, stage):
        fake = FakeLLM()
        assert for_stage(fake, stage) is fake

    def test_unknown_stage_returned_unchanged(self):
        # A stage absent from _STAGE_THINKING_LEVELS behaves the same as a
        # stage mapped to None: .get() returns None either way.
        llm = LLM()
        assert for_stage(llm, "some-future-stage") is llm


class TestForStageClone:
    """A configured stage level produces a clone with the pinned level and unchanged parent."""

    def test_clone_pins_level_without_mutating_original(self, monkeypatch):
        monkeypatch.setitem(_STAGE_THINKING_LEVELS, "bind", "minimal")
        llm = LLM(model="gemini-2.5-flash", seed=99)

        clone = for_stage(llm, "bind")

        assert clone is not llm
        assert clone._thinking_level == "minimal"
        assert llm._thinking_level is None
        assert clone._model == "gemini-2.5-flash"
        assert clone._seed == 99

    def test_explicit_thinking_level_still_outranks_patched_stage(self, monkeypatch):
        # The escape hatch: an LLM built with its own thinking_level kwarg
        # outranks a stage override even when the stage has one configured.
        monkeypatch.setitem(_STAGE_THINKING_LEVELS, "bind", "minimal")
        llm = LLM(thinking_level="high")

        assert for_stage(llm, "bind") is llm


class TestForStageWiring:
    """Extraction and bind calls receive the thinking levels configured for their stages."""

    def test_extract_and_bind_calls_see_configured_thinking_levels(self, search_settings):
        # Force the instance's own thinking_level to None regardless of ambient
        # env, so the assertion below is deterministic.
        search_settings(llm_thinking_level=None)

        class _RecordingLLM(LLM):
            """Records (schema_name, thinking_level) per call, delegating to FakeLLM."""

            def __init__(self) -> None:
                super().__init__()
                self._fake = FakeLLM()
                self.calls: list[tuple[str, str | None]] = []

            async def generate_structured(self, *, prompt, system, schema):
                self.calls.append((schema.__name__, self._thinking_level))
                return await self._fake.generate_structured(
                    prompt=prompt, system=system, schema=schema
                )

        recorder = _RecordingLLM()
        offline_resolve(make_request("total population India"), llm=recorder)
        assert recorder.calls == [
            ("Extraction", None),
            ("_UniversalBindOutput", "minimal"),
            ("QuantityOutput", "minimal"),
        ]


class TestConfigSeedAndThinkingLevelEnvBinding:
    """Tests environment parsing directly with `SearchSettings`.

    Reloading `search.engine.config` leaves a second module that fixture patches cannot see."""

    def test_seed_unset_is_one(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("SEARCH_LLM_SEED", raising=False)
        assert config.SearchSettings().llm_seed == 1

    def test_seed_env_parsed_as_int(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SEARCH_LLM_SEED", "42")
        assert config.SearchSettings().llm_seed == 42

    def test_seed_explicit_empty_string_is_none(self, monkeypatch: pytest.MonkeyPatch):
        """The opt-out: an explicit empty value restores unseeded sampling,
        distinct from an unset variable, which keeps the process default seed.
        """
        monkeypatch.setenv("SEARCH_LLM_SEED", "")
        assert config.SearchSettings().llm_seed is None

    def test_thinking_level_unset_is_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("SEARCH_LLM_THINKING_LEVEL", raising=False)
        assert config.SearchSettings().llm_thinking_level is None

    def test_thinking_level_env_accepts_each_valid_value(self, monkeypatch: pytest.MonkeyPatch):
        for value in ("minimal", "low", "medium", "high"):
            monkeypatch.setenv("SEARCH_LLM_THINKING_LEVEL", value)
            assert config.SearchSettings().llm_thinking_level == value  # must not raise

    def test_thinking_level_invalid_value_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SEARCH_LLM_THINKING_LEVEL", "extreme")
        with pytest.raises(ValueError, match="SEARCH_LLM_THINKING_LEVEL"):
            config.SearchSettings()


class TestEngineModelDefaultIsPinned:
    """The default is a concrete model id, so a `-latest` alias cannot move an eval baseline."""

    def test_default_is_not_a_latest_alias(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("SEARCH_ENGINE_MODEL", raising=False)
        assert not config.SearchSettings().engine_model.endswith("-latest")

    def test_default_is_the_pinned_concrete_id(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("SEARCH_ENGINE_MODEL", raising=False)
        assert config.SearchSettings().engine_model == "gemini-3.5-flash-lite"
