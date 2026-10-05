"""A graph-declared default facet is honoured whatever the window, and one spec selects the
same facet as a single member, under a unit request, and as a group member.

The declared facet `facetDeclared` covers 1990-2000. `facetRecent` covers 2015-2024 with more
observations, so a 2020-2024 window reaches only `facetRecent`. The two differ only on the
axis the policy declares, so the other axes, ranked in the window, still fit the declared
facet jointly.
"""

from __future__ import annotations

from dataclasses import replace
from urllib.parse import unquote

import httpx
import pytest

from dc_sdmx import SdmxClient
from materialise import MaterialisationRequest, materialise_preview
from materialise.facets import FacetBinding, UnitRequest, declared_default_policies, select_facet
from materialise.generation import _select_member_facets
from materialise.graph import Facet, FacetDefaultPolicy
from materialise.group_facets import select_group_facets
from materialise.window import RequestedWindow

from ._group_support import _STAT_VAR_A, _STAT_VAR_B, _request
from .conftest import FakeGraphReader

_WINDOW = RequestedWindow(kind="range", start_year=2020, end_year=2024)
_DECLARED = Facet(
    facet_id="facetDeclared",
    provenance_id="PROV",
    measurement_method="METH",
    unit="CurrencyGBP",
    obs_count=11,
    earliest_date="1990",
    latest_date="2000",
)
_RECENT = replace(
    _DECLARED, facet_id="facetRecent", obs_count=40, earliest_date="2015", latest_date="2024"
)
_PROVENANCE_CANDIDATES = [
    replace(_DECLARED, provenance_id="PROV_DECLARED"),
    replace(_RECENT, provenance_id="PROV_RECENT"),
]
_UNIT_CANDIDATES = [replace(_DECLARED, unit="CurrencyGBP"), replace(_RECENT, unit="CurrencyEUR")]
# Constant prices skip the native-currency preference, so the declared unit default decides.
_UNIT_REQUEST = UnitRequest(currency="USD", prices="constant", base_year=2020)


def _graph(axis: str, ref: str) -> FakeGraphReader:
    policy = FacetDefaultPolicy(
        resolved_ref=ref,
        policy_node="policy/declared",
        policy_version="v1",
        selection_reason="graph_declared",
    )
    return FakeGraphReader(policies={(_STAT_VAR_A, axis): policy, (_STAT_VAR_B, axis): policy})


def _member_and_group(
    axis: str, binding: FacetBinding, candidates: list[Facet], graph: FakeGraphReader
):
    member = select_facet(
        axis,
        binding,
        candidates=candidates,
        graph=graph,
        stat_var=_STAT_VAR_A,
        window=_WINDOW,
        policies=declared_default_policies({axis: binding}, graph=graph, stat_var=_STAT_VAR_A),
    )
    bindings = {} if binding.state == "unbound" else {axis: binding}
    group = select_group_facets(
        [
            _request(stat_var=_STAT_VAR_A, facet_bindings=bindings, window=_WINDOW),
            _request(stat_var=_STAT_VAR_B, facet_bindings=bindings, window=_WINDOW),
        ],
        [candidates, candidates],
        graph=graph,
    )[axis]
    return member, group


@pytest.mark.parametrize(
    ("axis", "binding", "candidates", "ref"),
    [
        ("provenance", FacetBinding(state="unbound"), _PROVENANCE_CANDIDATES, "PROV_DECLARED"),
        (
            "unit",
            FacetBinding(state="request", request=_UNIT_REQUEST),
            _UNIT_CANDIDATES,
            "CurrencyGBP",
        ),
    ],
    ids=["unbound", "unit_request"],
)
def test_a_declared_default_outside_the_window_serves_as_a_member_and_in_a_group(
    axis, binding, candidates, ref
):
    member, group = _member_and_group(axis, binding, candidates, _graph(axis, ref))

    for selection in (member, group):
        assert selection.basis == "declared_default"
        assert selection.ref == ref
        assert selection.dispatch_facet_id == "facetDeclared"


def test_without_a_declared_default_the_window_elects_the_facet_it_reaches():
    """The control: with no declared policy, the window step picks `facetRecent` on both paths."""
    member, group = _member_and_group(
        "provenance", FacetBinding(state="unbound"), _PROVENANCE_CANDIDATES, FakeGraphReader()
    )

    assert member.dispatch_facet_id == group.dispatch_facet_id == "facetRecent"
    assert member.basis == group.basis == "deployment_fallback"


# The declared provenance's facet is recorded and ends in 2000; another provenance's
# Estimated projection covers 2020-2032. The two differ on method as well as provenance.
_RECORDED = Facet(
    facet_id="facetRecorded",
    provenance_id="PROV_DECLARED",
    measurement_method="Recorded",
    unit="USDollar",
    obs_count=41,
    earliest_date="1960",
    latest_date="2000",
)
_PROJECTED = Facet(
    facet_id="facetProjected",
    provenance_id="PROV_OTHER",
    measurement_method="Estimated",
    unit="USDollar",
    obs_count=13,
    earliest_date="2020",
    latest_date="2032",
)
_YEAR_2027 = RequestedWindow(kind="range", start_year=2027, end_year=2027)
_PLACE = "country/KEN"
_SDMX_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def test_axes_left_to_fallback_follow_the_declared_default_as_a_member_and_in_a_group():
    """The declared provenance wins whatever the window, so the method the fallback picks is
    the declared facet's own: recorded, though only the other provenance's projection reaches
    2027. Taking the method from the window instead pairs the declared provenance with a
    method it never reports, and no facet serves that pair."""
    graph = _graph("provenance", "PROV_DECLARED")
    candidates = [_RECORDED, _PROJECTED]

    member = _select_member_facets(
        _request(stat_var=_STAT_VAR_A, facet_bindings={}, window=_YEAR_2027),
        candidates,
        graph=graph,
    )
    group = select_group_facets(
        [
            _request(stat_var=_STAT_VAR_A, facet_bindings={}, window=_YEAR_2027),
            _request(stat_var=_STAT_VAR_B, facet_bindings={}, window=_YEAR_2027),
        ],
        [candidates, candidates],
        graph=graph,
    )

    for selections in (member, group):
        assert selections["provenance"].basis == "declared_default"
        assert selections["provenance"].ref == "PROV_DECLARED"
        assert selections["measurementMethod"].ref == "Recorded"
        assert selections["measurementMethod"].dispatch_facet_id == "facetRecorded"
        assert selections["unit"].ref == "USDollar"


def test_a_2027_preview_fetches_the_declared_facet_alone():
    """The declared default wins before window narrowing, so a 2027 preview dispatches the
    declared recorded facet though its points end in 2000, and never the other provenance's
    projection: the window then filters the declared facet's points."""
    fetched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetched.append(unquote(str(request.url)))
        return httpx.Response(
            200,
            text=_SDMX_HEADER,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    graph = FakeGraphReader(
        facets={(_STAT_VAR_A.removeprefix("dcid:"), _PLACE): [_RECORDED, _PROJECTED]},
        policies={
            (_STAT_VAR_A, "provenance"): FacetDefaultPolicy(
                resolved_ref="PROV_DECLARED",
                policy_node="policy/declared",
                policy_version="v1",
                selection_reason="graph_declared",
            )
        },
    )
    request = MaterialisationRequest(
        stat_var=_STAT_VAR_A,
        role_bindings={"observationAbout": [_PLACE]},
        declared_roles=("observationAbout",),
        facet_bindings={},
        window=_YEAR_2027,
    )

    outcome = materialise_preview(
        request,
        sdmx=SdmxClient(
            base_url="https://ours.example.org", transport=httpx.MockTransport(handler)
        ),
        graph=graph,
        limit=10,
    )

    assert outcome.selections["provenance"].ref == "PROV_DECLARED"
    assert outcome.selections["measurementMethod"].ref == "Recorded"
    assert fetched and all("facetRecorded" in url for url in fetched)
    assert not any("facetProjected" in url for url in fetched)


def test_a_unit_request_follows_the_other_axes_declared_defaults():
    """The declared provenance reports pounds and ends in 2000; the other provenance's
    projection reports euros into 2032. A current US dollar request has no native match, so
    its unit comes from the ranked pool, which keeps only the declared provenance's facet
    before the 2027 window narrows it: pounds, dispatched from the declared facet."""
    declared = replace(_RECORDED, unit="CurrencyGBP")
    projected = replace(_PROJECTED, unit="CurrencyEUR")
    request = _request(
        stat_var=_STAT_VAR_A,
        facet_bindings={
            "unit": FacetBinding(
                state="request", request=UnitRequest(currency="USD", prices="current")
            )
        },
        window=_YEAR_2027,
    )

    selections = _select_member_facets(
        request, [declared, projected], graph=_graph("provenance", "PROV_DECLARED")
    )

    assert selections["provenance"].ref == "PROV_DECLARED"
    assert selections["unit"].ref == "CurrencyGBP"
    assert selections["unit"].dispatch_facet_id == "facetRecorded"
