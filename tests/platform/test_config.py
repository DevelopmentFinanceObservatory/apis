"""dfo_platform.config: the platform-wide CORS allowlist parser and the
shared executor's default capacity.
"""

from __future__ import annotations

import pydantic
import pytest

from dfo_platform.config import executor_max_workers, parse_cors_origins


class TestParseCorsOrigins:
    def test_no_env_set_returns_empty(self, monkeypatch):
        monkeypatch.delenv("DFO_CORS_ORIGINS", raising=False)
        assert parse_cors_origins() == []

    def test_dfo_cors_origins_parsed_and_stripped(self, monkeypatch):
        monkeypatch.setenv("DFO_CORS_ORIGINS", " https://a.example.com , https://b.example.com ")
        assert parse_cors_origins() == ["https://a.example.com", "https://b.example.com"]

    def test_explicit_empty_dfo_cors_origins_returns_empty(self, monkeypatch):
        monkeypatch.setenv("DFO_CORS_ORIGINS", "")
        assert parse_cors_origins() == []


class TestExecutorMaxWorkers:
    def test_default_is_32(self, monkeypatch):
        monkeypatch.delenv("DFO_EXECUTOR_MAX_WORKERS", raising=False)
        assert executor_max_workers() == 32

    def test_reads_env_at_call_time(self, monkeypatch):
        monkeypatch.setenv("DFO_EXECUTOR_MAX_WORKERS", "4")
        assert executor_max_workers() == 4

    def test_raises_on_non_positive(self, monkeypatch):
        monkeypatch.setenv("DFO_EXECUTOR_MAX_WORKERS", "0")
        with pytest.raises(pydantic.ValidationError, match="greater than or equal to 1"):
            executor_max_workers()
