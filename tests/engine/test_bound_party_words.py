"""`phase_intake.bound_party_spans` reads the ranges of the variable phrase that name a party
the part binds, from the committed export of its names (`pack.geo_names`), and the readers
mask exactly those ranges."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import pytest

from code_index.tokens import tokenize
from entity_resolution import Entity, Group, Resolved, pack
from search.engine.region.party_aliases import bound_party_spans
from search.engine.region.phase_rank_bind import _entity_spans
from search.engine.region.state import ResolveState


def _place(dcid: str, name: str) -> Resolved:
    return Resolved(entity=Entity(dcid=dcid, name=name, kind="country", graph_types=("Country",)))


_USA = _place("country/USA", "United States")
_AFRICA = _place("africa", "Africa")
_ITALY = _place("country/ITA", "Italy")
_UGANDA = _place("country/UGA", "Uganda")
_CANADA = _place("country/CAN", "Canada")
_EU = _place("EuropeanUnion", "European Union")
_G7 = Group(members=(), entity=None, dcid="GroupOfSeven", label="Group of Seven")


@pytest.mark.parametrize(
    ("variable", "bound", "read"),
    [
        pytest.param("total US ODA", _USA, ("US",), id="country-abbreviation"),
        pytest.param("total UK ODA", _USA, (), id="another-countrys-abbreviation"),
        pytest.param("US-funded health ODA", _USA, (), id="inside-a-hyphenated-word"),
        pytest.param("Africa-wide health spending", _AFRICA, (), id="hyphenated-place"),
        pytest.param("the US's ODA", _USA, ("US's",), id="possessive"),
        pytest.param("U.S. aid", _USA, ("U.S.",), id="dotted-abbreviation"),
        pytest.param("aid to us", _USA, (), id="pronoun-us"),
        pytest.param("ODA to the states of the Sahel", _USA, (), id="ordinary-phrase"),
        pytest.param("what can donors give", _CANADA, (), id="verb-can"),
        pytest.param("G7 ODA", _G7, ("G7",), id="group-abbreviation"),
        pytest.param("Italy's ODA for IT", _ITALY, ("Italy's",), id="fuzzy-code-unread"),
        pytest.param("ODA to +256", _UGANDA, (), id="dial-code-unread"),
        pytest.param(
            "African Development Bank ODA to Africa", _AFRICA, ("Africa",), id="adjective-unread"
        ),
        pytest.param(
            "Africa Finance Corporation lending", _AFRICA, (), id="inside-an-organisation-name"
        ),
        pytest.param("South Africa ODA", _AFRICA, (), id="inside-another-place-name"),
        pytest.param("ODA of dac EU members", _EU, ("EU",), id="enclosing-name-written-otherwise"),
    ],
)
def test_bound_party_spans(variable, bound, read):
    spans = bound_party_spans(variable, [bound])

    assert tuple(variable[start:end] for start, end in spans) == read


def test_a_copy_of_the_alias_inside_another_name_stays_unmasked():
    """ "EU" names the European Union the part binds, and "EU institutions" a stored
    organisation of its own, so the readers mask the first "EU" alone."""
    variable = "EU ODA through EU institutions"
    state = SimpleNamespace(
        variable=variable,
        entity_texts=("European Union",),
        entity_roles={},
        party_alias_spans=bound_party_spans(variable, [_EU]),
    )

    assert _entity_spans(cast(ResolveState, state), tokenize(variable)) == [(0, 2)]


def test_each_listed_abbreviation_names_one_dcid():
    """Each reviewed abbreviation is a name, in the committed export, of the dcid it is
    listed for and of no other."""
    exported = json.loads(pack.geo_names_path().read_text(encoding="utf-8"))
    for abbreviation, dcid in pack.geo_abbreviations().items():
        owners = [
            owner
            for owner, names in exported.items()
            if abbreviation.casefold() in {name.casefold() for name in names}
        ]
        assert owners == [dcid], abbreviation
