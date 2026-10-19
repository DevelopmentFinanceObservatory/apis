"""Invalid environment values raise ValueError when SearchSettings is constructed.
Tests patch the environment directly because reloading config creates a second module object.
"""

from __future__ import annotations

import pytest

from search.engine import config


class TestNumericBounds:
    @pytest.mark.parametrize(
        ("env_var", "invalid_val"),
        [
            ("SEARCH_RELEVANCE_THRESHOLD", "0"),
            ("SEARCH_RELEVANCE_THRESHOLD", "1.5"),
            ("SEARCH_WEAK_SCORE_THRESHOLD", "0"),
            ("SEARCH_WEAK_SCORE_THRESHOLD", "2.0"),
            ("SEARCH_MAX_CANDIDATES", "0"),
            ("SEARCH_MAX_VARIABLES", "0"),
            ("SEARCH_MAX_ENTITIES", "0"),
            ("SEARCH_MAX_ENTITY_OCCURRENCES", "0"),
            ("SEARCH_BIND_TOP_K", "0"),
            ("SEARCH_OBS_PREVIEW_CONCURRENCY", "0"),
            ("SEARCH_OBS_PREVIEW_TOTAL_CONCURRENCY", "0"),
            ("SEARCH_SDMX_CHUNK_CONCURRENCY", "0"),
            ("SEARCH_GRAPH_TIMEOUT_S", "0"),
            ("SEARCH_GRAPH_TIMEOUT_S", "-5"),
            ("SEARCH_UN_RESOLVE_TIMEOUT_S", "0"),
            ("SEARCH_UN_FIRST_PASS_WAIT_S", "0"),
            ("SEARCH_UN_FIRST_PASS_WAIT_S", "3.5"),
            ("SEARCH_LLM_MAX_CONCURRENCY", "0"),
            ("SEARCH_MAX_QUERY_CHARS", "0"),
            ("SEARCH_REQUEST_TIMEOUT_S", "0"),
            ("SEARCH_REQUEST_TIMEOUT_S", "-1"),
            ("SEARCH_FINALISE_GRACE_S", "0"),
            ("SEARCH_RESUBMIT_FULL_READ_TIMEOUT_S", "30"),
            ("SEARCH_RESUBMIT_SUBJECT_CEILING", "0"),
            ("SEARCH_DETECT_K", "0"),
            ("SEARCH_OBS_PROPERTIES_CACHE_SIZE", "0"),
            ("SEARCH_RESOLVE_TARGET", "everything"),
            ("SEARCH_UNKNOWN_SCORE_THRESHOLD", "0"),
            ("SEARCH_UNKNOWN_SCORE_THRESHOLD", "1.5"),
            ("SEARCH_UN_UNKNOWN_SCORE_THRESHOLD", "0"),
            ("SEARCH_TOPIC_EXPAND_MARGIN", "-0.1"),
            ("SEARCH_TOPIC_EXPAND_MAX_MEMBERS", "0"),
            ("SEARCH_MAX_CHILD_PLACES", "0"),
            ("SEARCH_RECALL_SCORE_FLOOR", "-0.1"),
            ("SEARCH_RECALL_SCORE_FLOOR", "1.0"),
            ("SEARCH_LLM_MEMO_TTL_S", "-1"),
            ("SEARCH_LLM_MEMO_TTL_S", "inf"),
            ("SEARCH_LLM_MEMO_TTL_S", "-inf"),
            ("SEARCH_LLM_MEMO_TTL_S", "nan"),
            ("SEARCH_LLM_MEMO_MAX_BYTES", "-1"),
            ("SEARCH_LLM_MEMO_MAX_ENTRY_BYTES", "0"),
            ("SEARCH_EXTRACT_STORE", "disk"),
            ("SEARCH_EXTRACT_STORE", "file:"),
        ],
    )
    def test_numeric_bounds_out_of_range_raises(
        self, monkeypatch: pytest.MonkeyPatch, env_var: str, invalid_val: str
    ) -> None:
        monkeypatch.setenv(env_var, invalid_val)
        with pytest.raises(ValueError, match=env_var):
            config.SearchSettings()

    def test_memo_entry_ceiling_above_the_byte_budget(self, monkeypatch: pytest.MonkeyPatch):
        # SEARCH_LLM_MEMO_MAX_ENTRY_BYTES must be <= SEARCH_LLM_MEMO_MAX_BYTES,
        # so a memo cannot be built whose per-entry ceiling nothing can reach.
        monkeypatch.setenv("SEARCH_LLM_MEMO_MAX_BYTES", "1024")
        monkeypatch.setenv("SEARCH_LLM_MEMO_MAX_ENTRY_BYTES", "2048")
        with pytest.raises(ValueError, match="SEARCH_LLM_MEMO_MAX_ENTRY_BYTES"):
            config.SearchSettings()

    def test_memo_byte_budget_zero_is_valid(self, monkeypatch: pytest.MonkeyPatch):
        """0 (the explicit opt-out) disables the memo and must not raise, the
        same convention as SEARCH_SDMX_FACET_CACHE_SIZE."""
        monkeypatch.setenv("SEARCH_LLM_MEMO_MAX_BYTES", "0")
        assert config.SearchSettings().llm_memo_max_bytes == 0

    def test_entity_occurrence_cap_below_entity_cap(self, monkeypatch: pytest.MonkeyPatch):
        # SEARCH_MAX_ENTITY_OCCURRENCES must be >= SEARCH_MAX_ENTITIES: an
        # occurrence cap tighter than the distinct-name cap would refuse a
        # query test_conjunction.py::test_entity_dedup_before_cap already
        # pins as legitimate (SEARCH_MAX_ENTITIES + 1 occurrences of ONE name).
        monkeypatch.setenv("SEARCH_MAX_ENTITIES", "96")
        monkeypatch.setenv("SEARCH_MAX_ENTITY_OCCURRENCES", "50")
        with pytest.raises(ValueError, match="SEARCH_MAX_ENTITY_OCCURRENCES"):
            config.SearchSettings()

    def test_variable_concurrency_below_max_variables(self, monkeypatch: pytest.MonkeyPatch):
        # SEARCH_MAX_VARIABLE_CONCURRENCY must be >= SEARCH_MAX_VARIABLES.
        monkeypatch.setenv("SEARCH_MAX_VARIABLES", "6")
        monkeypatch.setenv("SEARCH_MAX_VARIABLE_CONCURRENCY", "3")
        with pytest.raises(ValueError, match="SEARCH_MAX_VARIABLE_CONCURRENCY"):
            config.SearchSettings()

    def test_sdmx_total_concurrency_below_chunk_concurrency(self, monkeypatch: pytest.MonkeyPatch):
        # SEARCH_SDMX_TOTAL_CONCURRENCY must be >= SEARCH_SDMX_CHUNK_CONCURRENCY.
        monkeypatch.setenv("SEARCH_SDMX_CHUNK_CONCURRENCY", "16")
        monkeypatch.setenv("SEARCH_SDMX_TOTAL_CONCURRENCY", "8")
        with pytest.raises(ValueError, match="SEARCH_SDMX_TOTAL_CONCURRENCY"):
            config.SearchSettings()

    @pytest.mark.parametrize("value", ["base", "custom"])
    def test_resolve_target_rejects_short_spellings(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        # The mixer answers target="base" or "custom" with HTTP 400, which
        # detect_svs raises as GraphInfraError on every query, so config
        # refuses both at startup.
        monkeypatch.setenv("SEARCH_RESOLVE_TARGET", value)
        with pytest.raises(ValueError, match="SEARCH_RESOLVE_TARGET"):
            config.SearchSettings()

    def test_resolve_target_accepts_each_mixer_value(self, monkeypatch: pytest.MonkeyPatch):
        for value in ("base_only", "custom_only", "base_and_custom"):
            monkeypatch.setenv("SEARCH_RESOLVE_TARGET", value)
            config.SearchSettings()  # must not raise

    def test_recall_score_floor_zero_is_valid(self, monkeypatch: pytest.MonkeyPatch):
        """0.0 (the explicit opt-out) is the no-op floor and must not raise."""
        monkeypatch.setenv("SEARCH_RECALL_SCORE_FLOOR", "0.0")
        assert config.SearchSettings().recall_score_floor == 0.0


class TestGraphBaseScheme:
    def test_missing_graph_base_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No default: a process with no graph target must not read another environment's graph.
        monkeypatch.delenv("SEARCH_GRAPH_BASE")
        with pytest.raises(ValueError, match="graph_base"):
            config.SearchSettings()

    def test_empty_graph_base_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARCH_GRAPH_BASE", "")
        with pytest.raises(ValueError, match="SEARCH_GRAPH_BASE"):
            config.SearchSettings()

    @pytest.mark.parametrize(
        "invalid_base",
        [
            "ftp://staging.example.com",
            "dc-staging.one.org",
            "http://dc-staging.one.org",
            "http://localhost.evil.com",
            "http://127.0.0.1.attacker",
        ],
    )
    def test_invalid_scheme_or_host_rejected(
        self, monkeypatch: pytest.MonkeyPatch, invalid_base: str
    ) -> None:
        monkeypatch.setenv("SEARCH_GRAPH_BASE", invalid_base)
        with pytest.raises(ValueError, match="SEARCH_GRAPH_BASE"):
            config.SearchSettings()

    @pytest.mark.parametrize(
        "valid_base",
        [
            "https://dc-staging.one.org",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ],
    )
    def test_valid_scheme_accepted(self, monkeypatch: pytest.MonkeyPatch, valid_base: str) -> None:
        monkeypatch.setenv("SEARCH_GRAPH_BASE", valid_base)
        config.SearchSettings()  # must not raise


class TestBroadReadCaps:
    """Pins broad-read caps against measured data.
    The single-pair preview cap is 20,000 rows (#704).
    """

    def test_sdmx_max_rows_admits_the_broadest_measured_query(self):
        # Broadest Health ODA returns 154,839 rows, so 200,000 admits it with margin.
        assert config.settings().sdmx_max_rows == 200_000

    def test_sdmx_max_bytes_covers_the_largest_measured_response(self):
        # The largest measured response was 58.29 MiB; the 64 MiB cap admits it.
        assert config.settings().sdmx_max_bytes == 64 * 1024 * 1024

    def test_region_max_members_defaults_to_700(self):
        # 700 is the known-safe starting bound for candidate facet pre-warming.
        # This coverage read has no SDMX row cap from which to derive a new bound.
        assert config.settings().region_max_members == 700

    def test_obs_preview_max_rows_stays_at_the_old_single_pair_cap(self):
        # The broad-read cap does not raise the separately measured preview cap (#704).
        assert config.settings().obs_preview_max_rows == 20_000
        assert config.settings().obs_preview_max_rows != config.settings().sdmx_max_rows


class TestUnitDefaultCurrency:
    def test_defaults_to_us_dollars(self):
        assert config.SearchSettings().unit_default_currency == "USD"

    @pytest.mark.parametrize("value", ["eur", "XYZ", "US", ""])
    def test_rejects_anything_but_a_registered_iso_code(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("SEARCH_UNIT_DEFAULT_CURRENCY", value)
        with pytest.raises(ValueError, match="SEARCH_UNIT_DEFAULT_CURRENCY"):
            config.SearchSettings()

    def test_a_new_default_currency_mints_a_new_capability_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        default = config.SearchSettings().capability_version
        monkeypatch.setenv("SEARCH_UNIT_DEFAULT_CURRENCY", "EUR")
        assert config.SearchSettings().capability_version != default
