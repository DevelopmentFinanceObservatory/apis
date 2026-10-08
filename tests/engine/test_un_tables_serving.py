"""Search reads a UN candidate's facts from the catalogue's UN tables: derivation sends the UN
deployment nothing, a shape holds the variables recall found, and admission follows the
tables."""

from __future__ import annotations

import httpx
import pytest

from origin_catalogue import BASE, DFO, UN_SYSTEM
from search.engine.clients.graph import DetectedSV, DetectLegs, admit_legs
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.facts import read_constraints, read_five_tuple
from tests.origin_catalogue.fake_catalogue import (
    provenance_row,
    restrictive_catalogue,
    un_mirror,
    un_variable,
)

_OURS_HOST = "ours.example.org"
_UN_HOST = "un.example.org"

_CRS = "provenance/OECDCreditorReportingSystem"
_WDI = "dc/base/WorldDevelopmentIndicators"
_BASE_MORTALITY = "Count_Death_Upto5Years_AsFractionOf_Count_BirthEvent_LiveBirth"
_BASE_SDG = "sdg/SH_DYN_MORT"

_SERIES = "undata/who/SERIES-MORT_U5_RATE"
_HEADLINE = "undata/who/MORT_U5_RATE"
_FEMALE = "undata/who/MORT_U5_RATE.SEX--F"
_SEX = "undataSex"
_TOPIC = "undata/topic/who/MORT_U5_RATE"
_AID = "undata/who/ODA_1_3"
_NOT_IN_THE_BUILD = "undata/who/NOT_IN_THE_BUILD"


def _catalogue():
    """The UN tables hold a WHO series of two variables, the headline and its by-sex variant,
    and a topic reaching both, and exclude `_AID` as superseded by the CRS. `_BASE_SDG` holds
    data only under a base provenance the registry excludes."""
    return restrictive_catalogue(
        provenances=[
            provenance_row(_CRS, DFO),
            provenance_row(_WDI, BASE),
            provenance_row("undata/p/WHO", UN_SYSTEM, segment="who"),
        ],
        base_variables={_BASE_MORTALITY: ([_WDI], [])},
        withheld_variables=[_BASE_SDG],
        un_mirror=un_mirror(
            {
                _HEADLINE: un_variable(_SERIES, name="Under-five mortality rate"),
                _FEMALE: un_variable(
                    _SERIES,
                    name="Under-five mortality rate [Sex=Female]",
                    constraints={_SEX: ("Sex", "undata/SEX-F", "Female")},
                ),
            },
            excluded={_AID: "undata/who/SERIES-ODA_1_3"},
            topics={_TOPIC: (_HEADLINE, _FEMALE)},
            superseded_by=_CRS,
        ),
    )


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Every HTTP request the process sends, each answered as an empty graph read."""
    requests: list[httpx.Request] = []

    def send(self: httpx.Client, request: httpx.Request, **_kwargs: object) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": {}}, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    return requests


@pytest.fixture
def client() -> LiveGraphClient:
    return LiveGraphClient(base=f"https://{_OURS_HOST}", un_base_url=f"https://{_UN_HOST}/core/api")


def test_deriving_un_candidates_sends_the_un_deployment_nothing(
    sent: list[httpx.Request], client: LiveGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    (shape,) = derive_shapes(
        confirmed_svs=[_TOPIC, _HEADLINE],
        graph=client,
        sv_scores={_TOPIC: 0.9, _HEADLINE: 0.8},
        sv_types={_TOPIC: "Topic", _HEADLINE: "StatisticalVariable"},
    )

    assert set(shape.sv_arc_facts or {}) == {_HEADLINE, _FEMALE}
    assert _SEX in {slot.property_dcid for slot in shape.slot_keys}
    assert shape.pop_type_name == f"{_SERIES} name"
    assert [request.url.host for request in sent] == []


def test_a_series_members_arcs_come_from_the_tables_with_no_un_request(
    sent: list[httpx.Request], client: LiveGraphClient, bind_catalogue
) -> None:
    """The member walk reads a series' members and their arcs; the tables answer both."""
    catalogue = _catalogue()
    bind_catalogue(catalogue)

    headline = client.node_arcs_batch([_HEADLINE])[_HEADLINE]
    series = read_five_tuple(headline or {}).pop_type_dcid
    members = catalogue.un_series_members(series)
    arcs = client.node_arcs_batch(list(members))

    assert members == (_HEADLINE, _FEMALE)
    assert read_constraints(headline or {}) == {}
    assert {prop for member in members for prop in read_constraints(arcs[member] or {})} == {_SEX}
    assert sent == []


def test_a_un_shape_holds_the_variables_recall_found(
    sent: list[httpx.Request], client: LiveGraphClient, bind_catalogue
) -> None:
    """Recall returns the headline alone. The series also holds a by-sex member, which the
    shape leaves out, as dev's shape of a base series leaves out what recall did not find."""
    bind_catalogue(_catalogue())

    (shape,) = derive_shapes(confirmed_svs=[_HEADLINE], graph=client, sv_scores={_HEADLINE: 0.8})

    assert list(shape.sv_arc_facts or {}) == [_HEADLINE]
    assert shape.slot_taxonomy == {}
    assert sent == []


@pytest.mark.parametrize("refused", [_AID, _NOT_IN_THE_BUILD, _BASE_SDG])
def test_recall_refuses_a_candidate_the_catalogue_does_not_serve_on_either_leg(
    refused: str,
) -> None:
    def scored(*dcids: str) -> tuple[DetectedSV, ...]:
        return tuple(DetectedSV(dcid=d, score=0.9, type_of=None, sentence=None) for d in dcids)

    legs = DetectLegs(ours=scored(refused, _BASE_MORTALITY), un=scored(refused, _HEADLINE))

    detection = admit_legs(legs, catalogue=_catalogue(), detect_k=80)

    assert [c.dcid for c in detection.candidates] == [_BASE_MORTALITY]
    assert [c.dcid for c in detection.un.result() or ()] == [_HEADLINE]
