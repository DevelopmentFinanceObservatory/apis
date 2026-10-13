"""One election under a year range, whoever runs it.

`materialise.facets.rank_candidates` ranks a candidate by the places it holds a point at in
the window, then by the window years its span covers, from `earliest_date` and
`latest_date`, which a dated facet read, an undated probe and a coverage candidate all
carry, then by its count. So the search answer, a mint of its unpinned Spec, a datasets read
of its identity and the unit decision behind a converting preview name the same facet, at
one place and across several.

Zeta holds a long series for 1970 to 2021 (52 points) and a short one for 2020 to 2022 (3
points). Inside 2020 to 2022 the short series covers three years and the long one two.
"""

from __future__ import annotations

import asyncio

import pytest

from code_index.shared import get_index
from datasets_api.adapter import to_materialisation_request
from datasets_api.shape import StatVarShape
from materialise import MaterialisationRequest, RequestedWindow, materialise_preview
from materialise.graph import Facet
from materialise.rate_tables import RateTableCache
from search.engine.bind import _UniversalBindOutput
from search.engine.extract import DateRequest
from search.engine.quantity_read import QuantityOutput
from search.engine.region.convertibility import decide_part_unit
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from search.engine.retrieval.read_plan import facets_served, plan_standard
from search.models import IdentityUnitRequest, Spec, TimeWindow
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import materialise_double_from_obs
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise._rate_table_support import CONVERTIBLE_STAT_VAR_ARCS, conversion_arcs
from tests.materialise.conftest import FakeGraphReader

_PERSON, _ZETA = "Count_Person", "country/ZZZ"
_LONG, _SHORT = "dc/base/WindowElectionLong", "dc/base/WindowElectionShort"
_LONG_YEARS = [str(year) for year in range(1970, 2022)]
_SHORT_YEARS = ["2020", "2021", "2022"]
_WINDOW = TimeWindow(start_year=2020, end_year=2022)


def _obs_facet(facet_id: str, provenance: str, years: list[str]) -> dict:
    return {
        "facetId": facet_id,
        "provenanceId": provenance,
        "earliestDate": years[0],
        "latestDate": years[-1],
        "obsCount": len(years),
        "dates": years,
        "values": [1.0] * len(years),
    }


_OBS = {
    f"{_PERSON}|{_ZETA}": [
        _obs_facet("f-long", _LONG, _LONG_YEARS),
        _obs_facet("f-short", _SHORT, _SHORT_YEARS),
    ]
}


class _Bind:
    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[], also_plausible=[]), None


def _search_answer() -> Spec:
    graph = FakeGraph(
        nodes={
            _PERSON: {
                "label": "Population",
                "type": "StatisticalVariable",
                "arcs": {
                    "populationType": {"nodes": [{"dcid": "Person"}]},
                    "measuredProperty": {"nodes": [{"dcid": "count"}]},
                    "statType": {"nodes": [{"dcid": "measuredValue"}]},
                },
            },
            _ZETA: {"label": "Zeta", "type": "Country"},
            _LONG: {"label": "Long Source"},
            _SHORT: {"label": "Short Source"},
        },
        obs=_OBS,
        detect={"population": {"svs": [_PERSON]}},
        resolve={"Zeta": _ZETA},
    )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_OBS)
    result = asyncio.run(
        resolve_variable(
            "population of Zeta",
            entities=["Zeta"],
            date_request=DateRequest(window=_WINDOW, latest=False),
            detect_query="population",
            role_query="population of Zeta",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_Bind(),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(
                include_observations=True,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                rate_cache=RateTableCache(),
            ),
        )
    )
    assert result.status == "definite", (result.status, result.no_data_reason)
    return result.specs[0]


def test_a_search_answer_and_a_mint_of_its_spec_serve_the_same_source() -> None:
    """The answer cites the short series, and a preview of its unpinned identity, the
    selection a mint runs over an undated probe, elects the short series too."""
    spec = _search_answer()
    identity = spec.definition.identity

    assert identity.facets is None
    assert [ref.ref for ref in spec.resolved.trace.resolved_sources] == [_SHORT]
    assert spec.resolved.facets["provenance"].ref == _SHORT
    observations = spec.resolved.observations
    assert observations is not None
    assert [s.facet["provenance"].ref for s in observations.series] == [_SHORT]

    materialise_graph, materialise_sdmx = materialise_double_from_obs(_OBS)
    probed = materialise_graph.variable_facets_batch(stat_var=_PERSON, entities=[_ZETA])[_ZETA]
    assert all(facet.dates == () for facet in probed)
    preview = materialise_preview(
        MaterialisationRequest(
            stat_var=f"dcid:{_PERSON}",
            role_bindings={"observationAbout": list(identity.entities["observationAbout"])},
            declared_roles=("observationAbout",),
            facet_bindings={},
            window=RequestedWindow(kind="range", start_year=2020, end_year=2022),
        ),
        sdmx=materialise_sdmx,
        graph=materialise_graph,
        limit=100,
    )

    assert preview.selections["provenance"].ref == _SHORT
    assert preview.selections["provenance"].dispatch_facet_id == "f-short"


def test_a_datasets_read_with_a_range_elects_as_the_answer_does() -> None:
    """A datasets read builds its request from the identity alone
    (`datasets_api.adapter.to_materialisation_request`), window included, and its preview
    elects the short series. With no window it elects the long one on count."""
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_OBS)
    shape = StatVarShape(
        stat_var=f"dcid:{_PERSON}",
        declared_roles=("observationAbout",),
        slots=(),
        measured_property="count",
        population_type="Person",
    )

    def elected(requested_window: dict) -> str | None:
        request = to_materialisation_request(
            {
                "identity_version": "spec1",
                "shape": {"kind": "stat_var", "id": _PERSON},
                "slots": [],
                "entities": {"observationAbout": [_ZETA]},
                "requested_window": requested_window,
            },
            shape=shape,
            code_index=get_index(),
        )
        preview = materialise_preview(
            request, sdmx=materialise_sdmx, graph=materialise_graph, limit=100
        )
        return preview.selections["provenance"].ref

    assert elected({"kind": "range", "start_year": 2020, "end_year": 2022}) == _SHORT
    assert elected({"kind": "all"}) == _LONG


_AMOUNT, _KEN = "TESTFIXTURE/WindowElection/Amount", "country/KEN"
_UNIT_ARCS = {
    _AMOUNT: dict(CONVERTIBLE_STAT_VAR_ARCS),
    **{
        node: node_arcs
        for source in (_LONG, _SHORT)
        for node, node_arcs in conversion_arcs(
            recipe_id="ConversionRecipeDAC",
            provenance_id=source,
            currencies=["CurrencyUSD", "CurrencyEUR"],
        ).items()
    },
}


def _dollar_facet(facet_id: str, provenance: str, years: list[str], *, dated: bool) -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id=provenance,
        unit="CurrencyUSD",
        obs_count=len(years),
        earliest_date=years[0],
        latest_date=years[-1],
        dates=tuple(years) if dated else (),
    )


@pytest.mark.parametrize("from_answer", [False, True], ids=["probe", "answer_facets"])
def test_a_converting_unit_decision_dispatches_the_facet_the_plan_cites(from_answer: bool) -> None:
    """Kenya holds the long and the short series in US dollars, and the request asks for
    euros over 2020 to 2022. The plan cites the short series, and the unit decision the
    converting preview dispatches elects it too: over an undated probe, and over the dated
    facets the answer read, where it makes no probe at all."""
    dated = [
        _dollar_facet("f-long", _LONG, _LONG_YEARS, dated=True),
        _dollar_facet("f-short", _SHORT, _SHORT_YEARS, dated=True),
    ]
    plan = plan_standard({(_AMOUNT, _KEN): dated}, members=[_AMOUNT], years=_WINDOW)
    assert [f.provenance_id for f in facets_served(plan)] == [_SHORT]

    materialise_graph = FakeGraphReader(
        facets={
            (_AMOUNT, _KEN): [
                _dollar_facet("f-long", _LONG, _LONG_YEARS, dated=False),
                _dollar_facet("f-short", _SHORT, _SHORT_YEARS, dated=False),
            ]
        },
        arcs=_UNIT_ARCS,
    )
    _unused_graph, materialise_sdmx = materialise_double_from_obs({})
    elected = decide_part_unit(
        [_AMOUNT],
        unit_request=IdentityUnitRequest.model_validate({"currency": "EUR", "prices": "current"}),
        role_bindings={"observationAbout": [_KEN]},
        declared_roles=("observationAbout",),
        years=_WINDOW,
        graph=materialise_graph,
        sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot().index,
        probes=plan.probes if from_answer else None,
    )

    (selections,) = elected[_AMOUNT]
    assert selections["provenance"].ref == _SHORT
    assert selections["unit"].ref == "CurrencyUSD"
    assert materialise_graph.calls["variable_facets_batch"] == (0 if from_answer else 1)


_GHA = "country/GHA"
# What one place holds: each facet's id, source and years. Every range is unbroken, so an
# undated probe and a dated read see the same points (the gap case is the known limit below).
_HELD: list[tuple[tuple[str, str, range], ...]] = [
    (("f-long", _LONG, range(2005, 2023)), ("f-short", _SHORT, range(2019, 2022))),
    (("f-short", _SHORT, range(2000, 2024)), ("f-long", _LONG, range(2018, 2021))),
    (("f-long", _LONG, range(2010, 2021)), ("f-short", _SHORT, range(2015, 2025))),
    (("f-long", _LONG, range(2010, 2021)),),
    (("f-short", _SHORT, range(2015, 2025)),),
    (),
]
_TWO_PLACE_WINDOWS: dict[str, tuple[TimeWindow | str | None, dict]] = {
    "dateless": (None, {"kind": "all"}),
    "latest": ("latest", {"kind": "latest"}),
    "2015_to_2020": (
        TimeWindow(start_year=2015, end_year=2020),
        {"kind": "range", "start_year": 2015, "end_year": 2020},
    ),
    "2019_to_2021": (
        TimeWindow(start_year=2019, end_year=2021),
        {"kind": "range", "start_year": 2019, "end_year": 2021},
    ),
    "2006_to_2017": (
        TimeWindow(start_year=2006, end_year=2017),
        {"kind": "range", "start_year": 2006, "end_year": 2017},
    ),
}


@pytest.mark.parametrize("window", list(_TWO_PLACE_WINDOWS))
@pytest.mark.parametrize("ghana", range(len(_HELD)))
@pytest.mark.parametrize("kenya", range(len(_HELD) - 1))
def test_at_two_places_every_election_names_the_answers_one_group(
    kenya: int, ghana: int, window: str
) -> None:
    """Whatever Kenya and Ghana hold and whatever the window, the answer's plan elects one
    group for the variable across both places, and a mint of its unpinned identity, a
    datasets read of that identity and the unit decision, over an undated probe and over
    the plan's own probes, all select that group."""
    years, requested_window = _TWO_PLACE_WINDOWS[window]
    held = {_KEN: _HELD[kenya], _GHA: _HELD[ghana]}

    def facets(place: str, *, dated: bool) -> list[Facet]:
        return [
            _dollar_facet(facet_id, provenance, [str(y) for y in span], dated=dated)
            for facet_id, provenance, span in held[place]
        ]

    plan = plan_standard(
        {(_AMOUNT, place): facets(place, dated=True) for place in held if held[place]},
        members=[_AMOUNT],
        years=years,
    )
    groups = {s.group_ids for s in plan.series if s.group}
    sources = {f.provenance_id for s in plan.series for f in s.group}
    assert len(sources) == 1, "one group, one source, across both places"
    (source,) = sources
    group_ids = frozenset().union(*groups)

    materialise_graph = FakeGraphReader(
        facets={(_AMOUNT, place): facets(place, dated=False) for place in held},
        arcs=_UNIT_ARCS,
    )
    _unused_graph, materialise_sdmx = materialise_double_from_obs({})
    request = to_materialisation_request(
        {
            "identity_version": "spec1",
            "shape": {"kind": "stat_var", "id": _AMOUNT},
            "slots": [],
            "entities": {"observationAbout": sorted(held)},
            "requested_window": requested_window,
        },
        shape=StatVarShape(
            stat_var=f"dcid:{_AMOUNT}",
            declared_roles=("observationAbout",),
            slots=(),
            measured_property="amount",
            population_type="EconomicActivity",
        ),
        code_index=get_index(),
    )
    minted = materialise_preview(
        request, sdmx=materialise_sdmx, graph=materialise_graph, limit=100
    ).selections["provenance"]
    assert (minted.ref, minted.dispatch_facet_id in group_ids) == (source, True)

    for probes in (None, plan.probes):
        (selections,) = decide_part_unit(
            [_AMOUNT],
            unit_request=IdentityUnitRequest.model_validate(
                {"currency": "EUR", "prices": "current"}
            ),
            role_bindings={"observationAbout": sorted(held)},
            declared_roles=("observationAbout",),
            years=years,
            graph=materialise_graph,
            sdmx=materialise_sdmx,
            coverage=empty_coverage_snapshot().index,
            probes=probes,
        )[_AMOUNT]
        decided = selections["provenance"]
        assert (decided.ref, decided.dispatch_facet_id in group_ids) == (source, True)


_GAP, _RECENT = "dc/base/WindowElectionGap", "dc/base/WindowElectionRecent"
_GAP_YEARS = [str(year) for year in (*range(2010, 2016), *range(2027, 2031))]
_RECENT_YEARS = ["2021", "2022"]
_GAP_OBS = {
    f"{_PERSON}|{_ZETA}": [
        _obs_facet("f-gap", _GAP, _GAP_YEARS),
        _obs_facet("f-recent", _RECENT, _RECENT_YEARS),
    ]
}


def test_an_undated_mint_elects_a_facet_with_a_gap_across_the_window_and_reads_no_rows() -> None:
    """The known limit of the shared election. One facet is recorded to 2015 and projected
    from 2027, another holds 2021 and 2022, and the request asks for 2018 to 2026. Search
    reads dates, sees the gapped facet holds no point there, and serves the recent one. A
    mint's probe carries spans alone, so the gapped facet, whose span covers all nine years,
    is elected and its read returns no row in the window."""
    window = TimeWindow(start_year=2018, end_year=2026)
    dated = [
        Facet(
            facet_id=f["facetId"],
            provenance_id=f["provenanceId"],
            obs_count=f["obsCount"],
            earliest_date=f["earliestDate"],
            latest_date=f["latestDate"],
            dates=tuple(f["dates"]),
        )
        for f in _GAP_OBS[f"{_PERSON}|{_ZETA}"]
    ]
    plan = plan_standard({(_PERSON, _ZETA): dated}, members=[_PERSON], years=window)
    assert [f.provenance_id for f in facets_served(plan)] == [_RECENT]

    materialise_graph, materialise_sdmx = materialise_double_from_obs(_GAP_OBS)
    preview = materialise_preview(
        MaterialisationRequest(
            stat_var=f"dcid:{_PERSON}",
            role_bindings={"observationAbout": [_ZETA]},
            declared_roles=("observationAbout",),
            facet_bindings={},
            window=RequestedWindow(kind="range", start_year=2018, end_year=2026),
        ),
        sdmx=materialise_sdmx,
        graph=materialise_graph,
        limit=100,
    )

    assert preview.selections["provenance"].ref == _GAP
    assert preview.total_rows == 0
