"""The search vocabulary's `PartyKind` and `PartyClass` mirror the code index's, one value for
one."""

from __future__ import annotations

from typing import get_args

import pytest

from code_index.types import PartyClass as IndexPartyClass
from code_index.types import PartyKind as IndexPartyKind
from search.models.vocab import PartyClass as SearchPartyClass
from search.models.vocab import PartyKind as SearchPartyKind


@pytest.mark.parametrize(
    ("search", "index"),
    [
        pytest.param(SearchPartyKind, IndexPartyKind, id="party-kind"),
        pytest.param(SearchPartyClass, IndexPartyClass, id="party-class"),
    ],
)
def test_search_vocabulary_equals_the_code_index_vocabulary(search: object, index: object):
    assert set(get_args(search)) == set(get_args(index))
