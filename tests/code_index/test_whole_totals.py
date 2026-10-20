"""A whole total (`PartyRow.whole`), a party's class and a whole total set load and refuse what
they contradict; an index that carries no marker loads with no whole total."""

from __future__ import annotations

import json
from typing import Any

import pytest

from code_index import CodeIndex, IndexFormatError, default_index_path, load_index, parse_index
from code_index.types import (
    PARTY_CLASSES,
    PARTY_KINDS,
    WholeMarker,
    WholeTotalSet,
    covered_class,
)

from .conftest import FIXTURE_PATH

_DAC2A = "provenance/OECD_DAC2A"
_DAC2B = "provenance/OECD_DAC2B"


def _marker(dataset: str, covers: list[str], years: list[int] | None = None) -> dict[str, Any]:
    return {"dataset": dataset, "covers": covers, **({"years": years} if years else {})}


def _total(
    dcid: str,
    whole: list[dict[str, Any]],
    *,
    datasets: list[str] | None = None,
    axes: list[str] | None = None,
    kind: str = "total",
    party_class: str | None = "official",
) -> dict[str, Any]:
    return {
        "dcid": dcid,
        "kind": kind,
        "party_class": party_class,
        "totals": None,
        "datasets": [_DAC2A, _DAC2B] if datasets is None else datasets,
        "axes": ["origin"] if axes is None else axes,
        "whole": whole,
    }


_OFFICIAL = _total(
    "DACOfficialDonors",
    [_marker(_DAC2A, ["official"], [1960, 2023]), _marker(_DAC2B, ["official"])],
)
_PRIVATE = _total(
    "DACPrivateDonors", [_marker(_DAC2A, ["private"])], datasets=[_DAC2A], party_class="private"
)
_RECIPIENTS = _total(
    "DACAllRecipients",
    [_marker(_DAC2A, ["residual", "official"], [1984, 2024])],
    axes=["destination"],
    party_class=None,
)
_ALL_DONORS = {
    "phrase": "All donors including private",
    "axis": "origin",
    "members": ["DACOfficialDonors", "DACPrivateDonors"],
}


def _parse(parties: list[dict[str, Any]], sets: list[dict[str, Any]] | None = None) -> CodeIndex:
    payload = json.loads(FIXTURE_PATH.read_text())
    payload["parties"] = parties
    payload["whole_total_sets"] = sets or []
    return parse_index(payload)


def test_a_whole_total_and_a_set_load_and_answer_the_lookups():
    index = _parse([_OFFICIAL, _PRIVATE, _RECIPIENTS], [_ALL_DONORS])

    official = index.parties["DACOfficialDonors"]
    assert (official.kind, official.party_class) == ("total", "official")
    assert official.marker(_DAC2B) == WholeMarker(dataset=_DAC2B, covers=("official",), years=None)
    assert official.marker(_DAC2A) == WholeMarker(
        dataset=_DAC2A, covers=("official",), years=(1960, 2023)
    )
    assert index.parties["DACPrivateDonors"].party_class == "private"
    assert index.parties["DACPrivateDonors"].marker(_DAC2B) is None
    assert index.parties["DACAllRecipients"].party_class is None
    # A marker's classes are sorted.
    assert index.parties["DACAllRecipients"].marker(_DAC2A) == WholeMarker(
        dataset=_DAC2A, covers=("official", "residual"), years=(1984, 2024)
    )
    assert index.whole_set("  all DONORS   including private ") == WholeTotalSet(
        phrase="All donors including private",
        axis="origin",
        members=("DACOfficialDonors", "DACPrivateDonors"),
    )
    assert index.whole_set("all donors") is None


def test_side_classes_and_population_totals_read_classes_and_covers():
    """Each side holds the classes of its parties, totals aside, and a population total is one
    whose markers cover `official`: the private donors' total is no population total."""
    france = _total("France", [], datasets=[_DAC2A, _DAC2B], axes=["origin", "destination"])
    gates = _total("Gates", [], datasets=[_DAC2A], party_class="private")
    other = _total("DACOtherDonors", [], datasets=[_DAC2A], party_class="residual")
    index = _parse(
        [
            {**france, "kind": "party"},
            {**gates, "kind": "party"},
            {**other, "kind": "residual"},
            _OFFICIAL,
            _PRIVATE,
            _RECIPIENTS,
        ]
    )

    assert index.side_classes == {
        ("origin", _DAC2A): frozenset({"official", "private", "residual"}),
        ("origin", _DAC2B): frozenset({"official"}),
        ("destination", _DAC2A): frozenset({"official"}),
        ("destination", _DAC2B): frozenset({"official"}),
    }
    assert index.population_totals_by_axis == {
        "origin": ("DACOfficialDonors",),
        "destination": ("DACAllRecipients",),
    }


def test_an_index_without_markers_loads_each_party_as_no_whole_total(code_index: CodeIndex):
    index = _parse([{"dcid": "DACAfrica", "kind": "total", "totals": "africa"}])

    assert index.parties["DACAfrica"].whole == ()
    assert index.parties["DACAfrica"].marker(_DAC2A) is None
    assert index.whole_total_sets == ()
    assert code_index.whole_total_sets == ()


def test_the_packaged_index_holds_only_whole_totals_that_are_totals_of_known_classes():
    index = load_index(default_index_path())

    assert {party.kind for party in index.parties.values()} <= PARTY_KINDS
    for party in index.parties.values():
        if party.whole:
            assert party.kind == "total"
            assert all(set(marker.covers) <= PARTY_CLASSES for marker in party.whole)
    # A population total covers the official parties of its side.
    assert {
        dcid
        for dcid, party in index.parties.items()
        if any("official" in marker.covers for marker in party.whole)
    } == {"DACAllRecipients", "DACOfficialDonors"}


@pytest.mark.parametrize(
    ("covers", "expected"),
    [
        (("official", "residual"), None),
        (("private",), "private"),
        (("residual",), "residual"),
        (("official", "private"), None),
    ],
    ids=["official-and-residual", "private", "residual-only", "official-and-private"],
)
def test_covered_class_is_the_one_class_a_marker_covers(covers, expected):
    assert covered_class(covers) == expected


def _party_marked(dcid: str, *markers: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return _total(dcid, list(markers), **extra)


@pytest.mark.parametrize(
    ("parties", "sets", "message"),
    [
        (
            [_party_marked("France", _marker(_DAC2A, ["official"]), kind="party")],
            [],
            "only a total holds",
        ),
        (
            [_party_marked("A", _marker(_DAC2A, ["official"]), axes=[])],
            [],
            "exactly one published axis",
        ),
        (
            [_party_marked("A", _marker(_DAC2A, ["official"]), axes=["destination", "origin"])],
            [],
            "exactly one published axis",
        ),
        (
            [_party_marked("A", _marker(_DAC2B, ["official"]), datasets=[_DAC2A])],
            [],
            "does not publish it",
        ),
        (
            [_party_marked("A", _marker(_DAC2A, ["official"]), _marker(_DAC2A, ["private"]))],
            [],
            "two whole markers",
        ),
        ([_party_marked("A", _marker(_DAC2A, []))], [], "covers no class"),
        ([_party_marked("A", _marker(_DAC2A, ["total"]))], [], "outside"),
        ([_party_marked("A", _marker(_DAC2A, ["official", "official"]))], [], "more than once"),
        ([_party_marked("A", _marker(_DAC2A, ["official"], [2023, 1960]))], [], "out of order"),
        ([_party_marked("A", _marker(_DAC2A, ["official"], [1960]))], [], "first and last year"),
        (
            [
                _party_marked("A", _marker(_DAC2A, ["official"])),
                _party_marked("B", _marker(_DAC2A, ["official", "private"]), party_class=None),
            ],
            [],
            "both cover",
        ),
        ([{"dcid": "France", "kind": "party", "totals": None}], [], "carries class None"),
        (
            [{"dcid": "France", "kind": "party", "party_class": "public", "totals": None}],
            [],
            "a party is official or private",
        ),
        (
            [{"dcid": "France", "kind": "party", "party_class": "residual", "totals": None}],
            [],
            "a party is official or private",
        ),
        (
            [{"dcid": "DACAfricaUnallocated", "kind": "residual", "totals": None}],
            [],
            "a residual is residual",
        ),
        (
            [
                {
                    "dcid": "DACAfricaUnallocated",
                    "kind": "residual",
                    "party_class": "official",
                    "totals": None,
                }
            ],
            [],
            "a residual is residual",
        ),
        (
            [
                _party_marked(
                    "A",
                    _marker(_DAC2A, ["official"]),
                    _marker(_DAC2B, ["private"]),
                    party_class=None,
                )
            ],
            [],
            "name different classes",
        ),
        ([{**_OFFICIAL, "party_class": "private"}], [], "whole markers give"),
        ([{**_RECIPIENTS, "party_class": "official"}], [], "whole markers give"),
        ([_party_marked("Plain")], [], "whole markers give"),
        (
            [_OFFICIAL, _PRIVATE],
            [{**_ALL_DONORS, "members": ["DACOfficialDonors"]}],
            "two or more",
        ),
        (
            [_OFFICIAL, _PRIVATE],
            [_ALL_DONORS, {**_ALL_DONORS, "phrase": "all  donors INCLUDING private"}],
            "twice",
        ),
        (
            [_OFFICIAL, _PRIVATE],
            [{**_ALL_DONORS, "members": ["DACOfficialDonors", "Missing"]}],
            "no whole total",
        ),
        ([_OFFICIAL, _PRIVATE], [{**_ALL_DONORS, "axis": "destination"}], "no whole total"),
        (
            [_OFFICIAL, _party_marked("Plain", datasets=[_DAC2A], party_class=None)],
            [{**_ALL_DONORS, "members": ["DACOfficialDonors", "Plain"]}],
            "no whole total",
        ),
        (
            [
                _party_marked("A", _marker(_DAC2A, ["official"]), datasets=[_DAC2A]),
                _party_marked(
                    "B", _marker(_DAC2B, ["private"]), datasets=[_DAC2B], party_class="private"
                ),
            ],
            [{**_ALL_DONORS, "members": ["A", "B"]}],
            "no dataset marks together",
        ),
    ],
    ids=[
        "non-total",
        "no-axis",
        "two-axes",
        "dataset-not-published",
        "two-markers-for-one-dataset",
        "covers-nothing",
        "unknown-class",
        "repeated-class",
        "years-out-of-order",
        "years-one-year",
        "overlapping-covers",
        "party-without-class",
        "party-with-unknown-class",
        "party-reading-residual",
        "residual-without-class",
        "residual-reading-official",
        "total-markers-name-different-classes",
        "total-class-not-its-markers",
        "total-class-beside-mixed-covers",
        "unmarked-total-with-class",
        "set-of-one",
        "set-phrase-twice",
        "set-member-missing",
        "set-member-on-another-axis",
        "set-member-not-whole",
        "set-members-never-marked-together",
    ],
)
def test_refuses_a_whole_marker_the_row_or_the_set_contradicts(parties, sets, message):
    with pytest.raises(IndexFormatError, match=message):
        _parse(parties, sets)


def test_whole_totals_may_cover_one_class_in_different_datasets_or_on_different_axes():
    index = _parse(
        [
            _party_marked("A", _marker(_DAC2A, ["official"]), datasets=[_DAC2A]),
            _party_marked("B", _marker(_DAC2B, ["official"]), datasets=[_DAC2B]),
            _party_marked(
                "C", _marker(_DAC2A, ["official"]), datasets=[_DAC2A], axes=["destination"]
            ),
        ]
    )

    assert {dcid for dcid, party in index.parties.items() if party.whole} == {"A", "B", "C"}
