"""An answer naming several places is the same whatever order the query names them in.

Places are stated in dcid order (`place_roles.place_order`), and every probe, re-election
and cache warm reads the whole named set, so "in Kenya and Ghana" and "in Ghana and Kenya"
elect, serve and request alike, and an answer holding data at Kenya alone serves Kenya
although Ghana sorts first.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import EngineGraphClient, Facet
from search.engine.region.pipeline import resolve_variable
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from search.models import CoverageExact
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine import test_role_bearing_pipeline_e2e as role_bearing
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    _SDMX_CSV_CONTENT_TYPE,
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    door_roles_for_test_query,
    materialise_double_from_obs,
)
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise.conftest import FakeGraphReader

_KEN, _GHA = "country/KEN", "country/GHA"
_SOURCE = "dc/base/PlaceOrderFixture"
_MORT = "Count_MortalityEventPlaceOrder"
_EST = "Count_MortalityEventEstimatedPlaceOrder"
_PROJ = "Count_MortalityEventProjectedPlaceOrder"
_GRAPH_METHODS = frozenset(name for name in dir(EngineGraphClient) if not name.startswith("_"))


def _arcs(label: str, qualifier: str | None = None) -> dict:
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": "MortalityEvent"}]},
        "measuredProperty": {"nodes": [{"dcid": "count"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if qualifier is not None:
        arcs["measurementQualifier"] = {"nodes": [{"dcid": qualifier}]}
    return arcs


_NODES = {
    _KEN: {"label": "Kenya", "type": "Country"},
    _GHA: {"label": "Ghana", "type": "Country"},
    _MORT: {
        "label": "Mortality event count",
        "type": "StatisticalVariable",
        "arcs": _arcs("Mortality event count"),
    },
    _EST: {
        "label": "Estimated mortality event count",
        "type": "StatisticalVariable",
        "arcs": _arcs("Estimated mortality event count", "Estimated"),
    },
    _PROJ: {
        "label": "Projected mortality event count",
        "type": "StatisticalVariable",
        "arcs": _arcs("Projected mortality event count", "Projected"),
    },
}


def _at_kenya(*stat_vars: str) -> dict[str, list[dict]]:
    """Each of `stat_vars` holds five points at Kenya, 2018 to 2022, and none at Ghana."""
    years = [str(year) for year in range(2018, 2023)]
    return {
        f"{sv}|{_KEN}": [
            {
                "facetId": f"f-{sv}",
                "provenanceId": _SOURCE,
                "earliestDate": years[0],
                "latestDate": years[-1],
                "obsCount": len(years),
                "dates": years,
                "values": [1.0] * len(years),
            }
        ]
        for sv in stat_vars
    }


class _Bind:
    async def generate_structured(self, *, prompt, system, schema):
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=[], ask=None
        ), None


class _RecordingGraph(FakeGraph):
    """A `FakeGraph` logging each graph request by method name, in call order. A request the
    fake serves through another counts once."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requests: list[str] = []
        self._depth = 0

    def __getattribute__(self, name: str):
        attr = super().__getattribute__(name)
        if name not in _GRAPH_METHODS or not callable(attr):
            return attr

        def _request(*args, **kwargs):
            if self._depth == 0:
                self.requests.append(name)
            self._depth += 1
            try:
                return attr(*args, **kwargs)
            finally:
                self._depth -= 1

        return _request


_CASES = {
    # The elected count holds data at Kenya.
    "elected": ((_MORT,), (0.9,), _at_kenya(_MORT), False),
    # The elected count holds none; its estimated sibling holds Kenya's and is re-elected.
    "sibling": ((_MORT, _EST), (0.9, 0.85), _at_kenya(_EST), False),
    # Named as origins, two siblings hold Kenya's data and are offered together.
    "directed_menu": (
        (_MORT, _EST, _PROJ),
        (0.9, 0.85, 0.82),
        _at_kenya(_EST, _PROJ),
        True,
    ),
}


def _answer(case: str, names: tuple[str, str]) -> tuple[RegionResult, list[str]]:
    svs, scores, obs, directed = _CASES[case]
    query = f"mortality count {'from' if directed else 'in'} {names[0]} and {names[1]}"
    graph = _RecordingGraph(
        nodes=_NODES,
        obs=obs,
        detect={query: {"svs": list(svs), "cosine_scores": list(scores)}},
        resolve={"Kenya": _KEN, "Ghana": _GHA},
    )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    entities = list(names)
    result = asyncio.run(
        resolve_variable(
            query,
            entities=entities,
            date_request=None,
            detect_query=query,
            role_query=query,
            entity_roles=door_roles_for_test_query(query, entities) if directed else None,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_Bind(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            base_steps=[],
            base_timing={},
            coverage=FakeCoverageService(build_coverage_index([], role_names={})).snapshot(),
            options=EngineOptions(
                include_observations=True,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
            ),
        )
    )
    return result, graph.requests


def _without_timings(value: Any) -> Any:
    """`value` without the step timings (`ms`) a trace records."""
    if isinstance(value, dict):
        return {k: _without_timings(v) for k, v in value.items() if k != "ms"}
    if isinstance(value, list):
        return [_without_timings(v) for v in value]
    return value


@pytest.mark.parametrize(
    ("case", "status"),
    [("elected", "definite"), ("sibling", "definite"), ("directed_menu", "candidates")],
)
def test_naming_order_does_not_change_the_answer(case: str, status: str) -> None:
    kenya_first, kenya_first_requests = _answer(case, ("Kenya", "Ghana"))
    ghana_first, ghana_first_requests = _answer(case, ("Ghana", "Kenya"))

    assert kenya_first.status == status, kenya_first.no_data_reason
    assert ghana_first.status == status, ghana_first.no_data_reason
    assert [_without_timings(s.model_dump()) for s in ghana_first.specs] == [
        _without_timings(s.model_dump()) for s in kenya_first.specs
    ]
    assert [w.message for w in ghana_first.warnings] == [w.message for w in kenya_first.warnings]
    assert ghana_first_requests == kenya_first_requests
    for spec in kenya_first.specs:
        coverage = spec.resolved.coverage
        assert isinstance(coverage, CoverageExact)
        assert coverage.observation_count == 5
        assert {d.label: d.count for d in coverage.dimensions or []}["countries"] == 1
        if (observations := spec.resolved.observations) is not None:
            assert {e.ref for series in observations.series for e in series.entities} == {_KEN}


# ---------------------------------------------------------------------------
# Role-bearing shapes: the same destinations and origins in any order.
# ---------------------------------------------------------------------------

_NOR, _FRA = "country/NOR", "country/FRA"
_FLOW = role_bearing._SV_DCID
_PLACE_NAMES = {"Kenya": _KEN, "Ghana": _GHA, "Norway": _NOR, "France": _FRA}
_FLOW_YEARS = [2018, 2019, 2020]


def _flow_answer(
    monkeypatch, search_settings, *, destinations, origins, bulk: bool
) -> tuple[RegionResult, list[list[tuple[str, str]]]]:
    """Debt flows to `destinations`, from `origins` when named and from every creditor
    otherwise, with each SDMX request the preview makes."""
    search_settings(obs_preview_max_pairs=1 if bulk else 120)
    pairs = [(creditor, debtor) for creditor in (_NOR, _FRA) for debtor in (_KEN, _GHA)]
    index = patch_coverage(
        monkeypatch,
        [(_FLOW, "f1", creditor, debtor, _FLOW_YEARS) for creditor, debtor in pairs],
        role_names={_FLOW: RoleNames(donor="creditor", recipient="debtor")},
        facet_metadata={"f1": FacetMetadata(provenance_id=_SOURCE)},
    )
    requests: list[list[tuple[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        requests.append(sorted(params.multi_items()))
        debtors = params.get("c[debtor]", "").split(",")
        creditors = params.get("c[creditor]")
        body = _csv_header(["creditor", "debtor"])
        for creditor, debtor in pairs:
            if debtor in debtors and (creditors is None or creditor in creditors.split(",")):
                for year in _FLOW_YEARS:
                    body += _csv_row(
                        stat_var=_FLOW,
                        role_dcids=[creditor, debtor],
                        fixture=ObsFacetFixture(facet_id="f1", provenance_id=_SOURCE),
                        date_str=str(year),
                        value=1.0,
                    )
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    years = [str(year) for year in _FLOW_YEARS]
    facet = Facet(
        facet_id="f1",
        provenance_id=_SOURCE,
        obs_count=len(years),
        earliest_date=years[0],
        latest_date=years[-1],
        dates=tuple(years),
        values=(1.0,) * len(years),
    )
    sdmx = FakeSdmxClient(
        observations={
            _FLOW: {
                (("creditor", creditor), ("debtor", debtor)): [facet] for creditor, debtor in pairs
            }
        },
        availability={},
    )
    query = "debt flows"
    if origins:
        query += f" from {' and '.join(origins)}"
    query += f" to {' and '.join(destinations)}"
    result = role_bearing._run(
        entities=[*origins, *destinations],
        sv_dcid=_FLOW,
        sv_arcs=role_bearing._declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve=_PLACE_NAMES,
        sv_score=0.9,
        role_query=query,
        sdmx=sdmx,
        include_observations=True,
        materialise_graph=FakeGraphReader(),
        materialise_sdmx=MaterialiseSdmxClient(
            base_url=MATERIALISE_BASE_URL, transport=httpx.MockTransport(handler)
        ),
        provenance_labels={_SOURCE: "Place Order Fixture"},
        coverage=FakeCoverageService(index).snapshot(),
    )
    return result, sorted(requests) + [sorted(map(str, call.items())) for call in sdmx.series_calls]


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
@pytest.mark.parametrize(
    "origins", [(), ("Norway", "France")], ids=["open_origin", "named_origins"]
)
def test_naming_order_does_not_change_a_role_bearing_answer(
    monkeypatch, search_settings, origins: tuple[str, ...], bulk: bool
) -> None:
    """Destinations and origins named in either order give one answer: the same entities in
    dcid order on each axis, the same warnings and the same preview requests."""
    answers = [
        _flow_answer(
            monkeypatch, search_settings, destinations=destinations, origins=named, bulk=bulk
        )
        for destinations in (("Kenya", "Ghana"), ("Ghana", "Kenya"))
        for named in dict.fromkeys((origins, origins[::-1]))
    ]

    first, first_requests = answers[0]
    assert first.status == "definite", first.no_data_reason
    (spec,) = first.specs
    assert [(e.role, e.ref) for e in spec.resolved.entities] == [
        *(("destination", dcid) for dcid in (_GHA, _KEN)),
        *(("origin", dcid) for dcid in sorted(_PLACE_NAMES[name] for name in origins)),
    ]
    assert spec.resolved.observations is not None and spec.resolved.observations.series
    for other, other_requests in answers[1:]:
        assert other.status == "definite", other.no_data_reason
        assert [_without_timings(s.model_dump()) for s in other.specs] == [
            _without_timings(s.model_dump()) for s in first.specs
        ]
        assert [w.message for w in other.warnings] == [w.message for w in first.warnings]
        assert other_requests == first_requests
