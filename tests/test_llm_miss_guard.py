"""Tests for `tests/llm_miss_guard.py`, which fails a passing test that missed a FakeLLM fixture."""

from __future__ import annotations

from tests.fixtures import HARVEST_MISS_PATH_ENV

pytest_plugins = ["pytester"]

_GENERATED_TESTS = """
import asyncio

import pytest
from pydantic import BaseModel

from tests.fixtures import FakeLLM, MissingFixtureError


class _S(BaseModel):
    value: str


def _swallow_a_miss():
    try:
        asyncio.run(FakeLLM(responses={}).generate_structured(prompt="p", system="s", schema=_S))
    except MissingFixtureError:
        pass


def test_swallows_a_miss():
    _swallow_a_miss()


@pytest.mark.llm_misses_expected
def test_swallows_a_miss_on_purpose():
    _swallow_a_miss()


def test_misses_nothing():
    assert True
"""


def test_a_swallowed_miss_fails_the_test_unless_it_is_marked(pytester):
    # The generated misses are deliberate; keep them out of a running harvest's miss file.
    pytester._monkeypatch.delenv(HARVEST_MISS_PATH_ENV, raising=False)
    pytester.makepyfile(test_generated=_GENERATED_TESTS)
    pytester.makeini(
        "[pytest]\nmarkers =\n    llm_misses_expected: builds a FakeLLM bank that misses\n"
    )

    result = pytester.runpytest("-q", "-p", "tests.llm_miss_guard")

    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(
        [
            "*test_swallows_a_miss*",
            "*FakeLLM missed recorded fixtures*",
            "  _S@*",
        ]
    )
