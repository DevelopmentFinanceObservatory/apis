"""Pins the stretch of a request that writes a variable phrase."""

from __future__ import annotations

import pytest

from search.engine.written_phrase import written_phrase


@pytest.mark.parametrize(
    ("variable", "query_text", "written"),
    [
        pytest.param(
            "external debt net flows",
            "net flows on external debt for Kenya",
            "net flows on external debt",
            id="a-reordered-phrase",
        ),
        pytest.param(
            "external debt net flows",
            "Kenya external debt net flows",
            "external debt net flows",
            id="the-phrase-as-extracted",
        ),
        pytest.param(
            "external debt net flows",
            "net flows on Kenya's external debt",
            "net flows on kenya s external debt",
            id="another-word-inside",
        ),
        pytest.param(
            "external debt net flows",
            "subnet flows on external debts for Kenya",
            "flows on external",
            id="a-word-inside-a-longer-word",
        ),
        pytest.param(
            "concessional external debt",
            "non-concessional external debt stocks for Kenya",
            "external debt",
            id="a-hyphenated-compound",
        ),
        pytest.param("gdp", "population of Kenya", None, id="no-word-of-the-phrase"),
        pytest.param("gdp", None, None, id="no-query-text"),
    ],
)
def test_the_written_stretch_runs_from_the_first_to_the_last_word_of_the_phrase(
    variable, query_text, written
):
    assert written_phrase(variable, query_text) == written
