"""The `value_parents` block loads into `CodeIndex.value_ancestors`, and the loader refuses a
value named twice and a parent chain that cycles."""

from __future__ import annotations

import json

import pytest

from code_index import CyclicParent, DuplicateDcid, parse_index
from tests.code_index.support import INDEX_FIXTURE, load_index

from .conftest import FIXTURE_PATH


def _document(value_parents: list[dict]) -> dict:
    payload = json.loads(FIXTURE_PATH.read_text())
    payload["value_parents"] = value_parents
    return payload


def test_a_value_s_ancestors_follow_its_parent_chain_nearest_first():
    index = parse_index(
        _document([{"dcid": "Leaf", "parent": "Middle"}, {"dcid": "Middle", "parent": "Root"}])
    )

    assert index.value_ancestors("Leaf") == ("Middle", "Root")
    assert index.value_ancestors("Middle") == ("Root",)
    assert index.value_ancestors("Root") == ()
    assert index.value_ancestors("Unknown") == ()


def test_an_index_without_the_block_holds_no_value_parent():
    index = parse_index(json.loads(FIXTURE_PATH.read_text()))

    assert index.value_parents == {}
    assert index.value_ancestors("IDSDebtorTypeGG") == ()


def test_refuses_a_value_named_twice():
    with pytest.raises(DuplicateDcid, match="Leaf"):
        parse_index(
            _document([{"dcid": "Leaf", "parent": "Root"}, {"dcid": "Leaf", "parent": "Other"}])
        )


def test_refuses_a_parent_chain_that_cycles():
    with pytest.raises(CyclicParent, match="value parent"):
        parse_index(_document([{"dcid": "A", "parent": "B"}, {"dcid": "B", "parent": "A"}]))


def test_the_fixture_index_reads_the_general_government_debtor_as_public_sector():
    index = load_index(INDEX_FIXTURE)

    assert index.value_ancestors("IDSDebtorTypeGG") == ("IDSDebtorTypePS",)
    assert index.value_ancestors("IDSDebtorTypePrivateSector") == ()
