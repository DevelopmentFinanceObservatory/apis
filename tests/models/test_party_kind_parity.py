"""The search vocabulary's `PartyKind` mirrors the code index's, one value for one."""

from __future__ import annotations

from typing import get_args

from code_index.types import PartyKind as IndexPartyKind
from search.models.vocab import PartyKind as SearchPartyKind


def test_search_party_kinds_equal_the_code_index_party_kinds():
    assert set(get_args(SearchPartyKind)) == set(get_args(IndexPartyKind))
