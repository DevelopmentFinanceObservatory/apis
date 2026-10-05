"""Fail a passing test whose run missed a recorded FakeLLM fixture.

The engine degrades some failures instead of raising them: a multi-part leg becomes `no_data`,
and a failed gap-fill pass keeps the first pass. A stale fixture bank can therefore leave a test
green on a degraded answer. FakeLLM notes every miss here, and `tests/conftest.py` loads these
hooks for the whole suite. Tests marked `llm_misses_expected` miss on purpose.

The report hook runs inside every other report wrapper, so the harvest outcome plugin records
the converted failure beside the misses logged during it. This module imports only pytest, so
the conftest still loads in installs without the engine extras, and the outcome plugin, which
pytest imports before `tests/conftest.py`, loads without engine settings.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

HARVEST_MISS_PATH_ENV = "SEARCH_LLM_HARVEST_PATH"
"""When set, FakeLLM appends one JSON record for every hit or miss.

Misses include schema and prompt fields. Hits include only the key. Unset, misses write nothing.
"""

_MISSED_KEYS: list[str] = []


def note_miss(key: str) -> None:
    """Record one FakeLLM miss for the running test."""
    _MISSED_KEYS.append(key)


def take_missed_keys() -> list[str]:
    """Return the keys missed since the last call, in call order, and forget them."""
    missed = list(_MISSED_KEYS)
    _MISSED_KEYS.clear()
    return missed


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    # Drop misses left by an earlier test's trailing tasks.
    take_missed_keys()


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Iterator[None]:
    report = yield
    if call.when != "call":
        return report
    missed = take_missed_keys()
    if missed and report.passed and item.get_closest_marker("llm_misses_expected") is None:
        report.outcome = "failed"
        report.longrepr = (
            "FakeLLM missed recorded fixtures and the engine degraded instead of raising; "
            "record these keys in tests/fixtures/llm_responses.json:\n  "
            + "\n  ".join(dict.fromkeys(missed))
        )
    return report
