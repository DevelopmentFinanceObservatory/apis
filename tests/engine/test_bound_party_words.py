"""`phase_intake.bound_party_words` reads the variable phrase's words that name a party the
part binds, from the committed export of resolvekit's geo names (`pack.geo_names`)."""

from __future__ import annotations

import pytest

from entity_resolution import Entity, Group, Resolved
from search.engine.region.phase_intake import bound_party_words


def _place(dcid: str, name: str) -> Resolved:
    return Resolved(entity=Entity(dcid=dcid, name=name, kind="country", graph_types=("Country",)))


_USA = _place("country/USA", "United States")
_AFRICA = _place("africa", "Africa")
_PERU = _place("country/PER", "Peru")
_CANADA = _place("country/CAN", "Canada")
_SOMALIA = _place("country/SOM", "Somalia")
_G7 = Group(members=(), entity=None, dcid="GroupOfSeven", label="Group of Seven")


@pytest.mark.parametrize(
    ("variable", "bound", "read"),
    [
        pytest.param("total US ODA", _USA, ("US",), id="country-alias"),
        pytest.param("total UK ODA", _USA, (), id="another-countrys-alias"),
        pytest.param("US aid", _USA, ("US",), id="abbreviation-in-capitals"),
        pytest.param("aid to us", _USA, (), id="pronoun-us"),
        pytest.param("aid to the us", _USA, (), id="pronoun-us-after-the"),
        pytest.param("what can donors give", _CANADA, (), id="verb-can"),
        pytest.param("ODA so far", _SOMALIA, (), id="word-so"),
        pytest.param("SO aid", _SOMALIA, ("SO",), id="code-in-capitals"),
        pytest.param("the US government ODA", _USA, ("the US",), id="longest-name-first"),
        pytest.param("G7 ODA", _G7, ("G7",), id="group-alias"),
        pytest.param(
            "African Development Bank ODA to Africa", _AFRICA, ("Africa",), id="adjective-unread"
        ),
        pytest.param(
            "Africa Finance Corporation lending", _AFRICA, (), id="inside-an-organisation-name"
        ),
        pytest.param("South Africa ODA", _AFRICA, (), id="inside-another-place-name"),
        pytest.param("ODA per capita", _PERU, (), id="function-word-name"),
    ],
)
def test_bound_party_words(variable, bound, read):
    assert bound_party_words(variable, [bound]) == read
