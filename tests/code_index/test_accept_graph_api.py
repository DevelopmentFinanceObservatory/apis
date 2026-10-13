"""Acceptance cases the Graph API design commits against the packaged code index: a roll-up of
CRS purposes reads each code once, so parent-coded observations count exactly once, and group
totals are keyed by the design's group ids (P6).

The rows come from `accept-graph.md`."""

from __future__ import annotations

from collections import Counter

import pytest

from code_index.shared import get_index


def test_a_roll_up_of_every_crs_purpose_root_reads_each_code_exactly_once():
    """CRS records carry one purpose each, so a parent's own records sit apart from its
    children's. Rolling up the roots reads every purpose code once: none twice, which would
    count a parent-coded record again, and none missed."""
    index = get_index()
    purposes = [row.dcid for row in index.codes if row.code_system == "crsPurpose"]
    roots = [row.dcid for row in index.codes if row.code_system == "crsPurpose" and not row.parent]

    read = Counter(code for root in roots for code in index.expansion(root))

    assert len(purposes) == 289
    assert set(read) == set(purposes)
    assert set(read.values()) == {1}


def test_a_crs_parent_reads_its_own_code_beside_its_descendants():
    index = get_index()

    expansion = index.expansion("CRSPurposeCode122")

    assert expansion[0] == "CRSPurposeCode122"
    assert "CRSPurposeCode12220" in expansion
    assert len(expansion) == len(set(expansion))


def test_a_dac1_parent_already_totals_its_children_and_stands_alone():
    index = get_index()

    assert index.declarations["dac1AidType"].disjoint_from_children is False
    assert index.expansion("DAC1AidType1015") == ("DAC1AidType1015",)


@pytest.mark.parametrize(("current", "retired"), [("GroupOfSeven", "groups/G7")])
def test_group_totals_are_keyed_by_the_design_group_id(current, retired):
    totals = get_index().totals_by_group

    assert current in totals
    assert retired not in totals


@pytest.mark.parametrize("kept", ["groups/UN.LDC", "groups/OECD.NonDAC", "africa"])
def test_group_totals_keep_ids_the_design_does_not_move(kept):
    assert kept in get_index().totals_by_group
