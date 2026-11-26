"""Search states the partly reported and projected years an answer's sources hold, whatever
its window (`resolved.coverage.held_back`), and a window the request's words close opts in to the
held-back years it names (`region.facets.settled_opt_in`). Engine tests pin today to 2026.

The partial fixture is a CRS-sourced series to 2025, which the dev graph marks partial from
2025 (`tests.origin_catalogue.fake_catalogue.dev_provenance_nodes`). The projected fixture
is an IDS-shaped series: reported to 2025 under no method, then `Estimated`, a declared
projection pair (`method_status.projection_methods`), from 2026 to 2030."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from origin_catalogue import bound_catalogue
from origin_catalogue.origins import DFO
from search.engine.departures import DepartureFact, spec_departure_facts
from search.engine.errors import EngineInputError
from search.engine.extract import DateRequest, dates_to_request
from search.engine.region import resolve_variable
from search.engine.region.resubmit_request import (
    _date_request_for_resubmit,
    _requested_window_from_identity,
)
from search.engine.region.result import RegionResult
from search.engine.spec_id import _requested_window_member
from search.engine.time_bind import parse_time_text
from search.models import HeldBack, IdentityWindowLatest, IdentityWindowRange, Spec, TimeWindow
from search.render.departures import departure_text
from search.render.plan import render_sentence
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.origin_catalogue.fake_catalogue import (
    dev_provenance_nodes,
    documenting,
    provenance_row,
)

_PERSON = "Count_Person"
_ZETA = "country/ZZZ"
_CRS = "provenance/OECDCreditorReportingSystem"
_IDS = "provenance/WorldBankInternationalDebtStatistics"


class _BindLLM:
    async def generate_structured(self, *, prompt, system, schema):
        from search.engine.quantity_read import QuantityOutput

        if schema is QuantityOutput:
            return NO_QUANTITY
        return elected_reply(0), None


def _facet(facet_id: str, provenance: str, years: range, method: str | None = None) -> dict:
    facet = {
        "facetId": facet_id,
        "provenanceId": provenance,
        "earliestDate": str(years[0]),
        "latestDate": str(years[-1]),
        "obsCount": len(years),
        "dates": [str(y) for y in years],
        "values": [1.0] * len(years),
    }
    if method:
        facet["measurementMethod"] = method
    return facet


_PARTIAL = [_facet("fCrs", _CRS, range(2000, 2026))]
_PROJECTED = [
    _facet("fIds", _IDS, range(2000, 2026)),
    _facet("fIdsE", _IDS, range(2026, 2031), "Estimated"),
]


def _catalogue():
    nodes = dev_provenance_nodes()
    catalogue = documenting({_CRS: nodes[_CRS], _IDS: {}})
    return replace(
        catalogue,
        provenances={
            _CRS: provenance_row(_CRS, DFO, name="OECD CRS"),
            _IDS: provenance_row(_IDS, DFO, name="World Bank IDS"),
        },
    )


def _answer(facets: list[dict], date_request: DateRequest | None) -> RegionResult:
    """Resolve "population of zeta" over one shape whose series holds `facets`."""
    nodes = {
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
    }
    graph = FakeGraph(
        nodes=nodes,
        obs={f"{_PERSON}|{_ZETA}": facets},
        detect={"population": {"svs": [_PERSON]}},
        resolve={"Zeta": _ZETA},
    )
    with bound_catalogue(_catalogue()):
        return asyncio.run(
            resolve_variable(
                "population of zeta",
                entities=["Zeta"],
                date_request=date_request,
                detect_query="population",
                role_query="population of zeta",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                llm=_BindLLM(),
                base_timing={},
                sdmx=FakeSdmxClient(
                    observations={}, availability={}, scope_availability=graph.scope_availability
                ),
                coverage=empty_coverage_snapshot(),
            )
        )


def _request(text: str) -> DateRequest:
    """The request a time phrase read in 2026 makes (`time_bind`, `dates_to_request`)."""
    dates, _warning = parse_time_text(text, current_year=2026)
    request = dates_to_request(dates)
    assert request is not None
    return request


def _spec(result: RegionResult) -> Spec:
    assert result.status == "definite", result.status
    return result.specs[0]


def _held(spec: Spec) -> list[tuple]:
    return [
        (h.kind, h.served, h.source, h.source_name, h.start_year, h.end_year, h.reported_share)
        for h in spec.resolved.coverage.held_back or []
    ]


def _since(year: int) -> DateRequest:
    return DateRequest(window=TimeWindow(start_year=year), latest=False)


def test_an_open_read_states_the_partial_year_it_holds_back():
    """An open window over a CRS series to 2025 serves 2000 to 2024, counts its 25 points
    there, and states 2025 held back as partly reported."""
    spec = _spec(_answer(_PARTIAL, _since(2000)))

    assert spec.resolved.coverage.window == TimeWindow(start_year=2000, end_year=2024)
    assert spec.resolved.coverage.observation_count == 25
    download = spec.resolved.download
    assert download is not None and download.window is not None
    assert download.window.end_year == 2024
    assert download.rows == 25
    assert _held(spec) == [("partial", False, _CRS, "OECD CRS", 2025, 2025, None)]
    assert spec.definition.identity.requested_window.include is None


def test_a_named_partial_year_opts_in_and_states_it_served():
    """ "2025" names the partial year, so the identity's window opts in to it, the answer
    serves it, and the run is stated served."""
    spec = _spec(_answer(_PARTIAL, _request("2025")))

    window = spec.definition.identity.requested_window
    assert isinstance(window, IdentityWindowRange)
    assert (window.start_year, window.end_year, window.include) == (2025, 2025, ["partial"])
    assert spec.resolved.coverage.window == TimeWindow(start_year=2025, end_year=2025)
    assert _held(spec) == [("partial", True, _CRS, "OECD CRS", 2025, 2025, None)]
    unnamed = _spec(_answer(_PARTIAL, _request("2024")))
    assert unnamed.spec_id != spec.spec_id


@pytest.mark.parametrize("text", ["2024", "2020 to 2024"])
def test_a_named_window_before_the_partial_year_opts_in_to_nothing(text):
    """A window ending at the newest complete year names no partial year, so its identity
    carries no opt-in, and it states the partial year held back."""
    spec = _spec(_answer(_PARTIAL, _request(text)))

    assert spec.definition.identity.requested_window.include is None
    assert _held(spec) == [("partial", False, _CRS, "OECD CRS", 2025, 2025, None)]


def test_a_trailing_span_opts_in_to_nothing():
    """ "the last 2 years" names no year, so it settles on 2023 to 2024 and writes no opt-in."""
    spec = _spec(_answer(_PARTIAL, _request("the last 2 years")))

    window = spec.definition.identity.requested_window
    assert isinstance(window, IdentityWindowRange)
    assert (window.start_year, window.end_year, window.include) == (2023, 2024, None)
    assert _held(spec) == [("partial", False, _CRS, "OECD CRS", 2025, 2025, None)]


def test_an_open_read_states_the_projections_it_holds_back():
    """An open window over an IDS-shaped series serves its reported years to 2025 and states
    2026 to 2030 held back as projections. The answer names no projection."""
    spec = _spec(_answer(_PROJECTED, _since(2000)))

    assert spec.resolved.coverage.window is not None
    assert spec.resolved.coverage.window.end_year == 2025
    assert _held(spec) == [("projected", False, _IDS, "World Bank IDS", 2026, 2030, None)]
    assert "projections" not in render_sentence(spec)


@pytest.mark.parametrize(
    ("text", "window", "projected"),
    [
        pytest.param("2020 to 2030", (2020, 2030), (2026, 2030), id="ending-on-the-run"),
        pytest.param("2020 to 2028", (2020, 2028), (2026, 2028), id="ending-inside-the-run"),
        # The window holds projected years alone, so its read elects the `Estimated` facet,
        # and the part names the projections, not that method's modelled estimates.
        pytest.param("2027 to 2029", (2027, 2029), (2027, 2029), id="starting-inside-the-run"),
    ],
)
def test_a_range_to_a_projected_year_opts_in_and_names_the_projections(text, window, projected):
    """A range reaching the projected years opts the identity in, the answer serves them,
    the run is stated served whole, and the part names the projected years it serves,
    those of the run inside the window ("IDS debt service 2020 to 2028")."""
    spec = _spec(_answer(_PROJECTED, _request(text)))

    identity_window = spec.definition.identity.requested_window
    assert isinstance(identity_window, IdentityWindowRange)
    assert identity_window.include == ["projected"]
    assert spec.resolved.coverage.window == TimeWindow(start_year=window[0], end_year=window[1])
    assert _held(spec) == [("projected", True, _IDS, "World Bank IDS", 2026, 2030, None)]
    modelled = [f for f in spec_departure_facts(spec, operations=()) if f.code == "modelled_values"]
    assert modelled == [
        DepartureFact(
            code="modelled_values",
            binding_ref=None,
            projected_years=projected,
            sources=("World Bank IDS",),
        )
    ]
    assert (
        departure_text(modelled[0], family=None, shows_data=True)
        == f"Values for {projected[0]} to {projected[1]} are World Bank IDS projections."
    )


@pytest.mark.parametrize(
    ("window", "member"),
    [
        (
            IdentityWindowRange(start_year=2020, end_year=2024),
            {"kind": "range", "start_year": 2020, "end_year": 2024},
        ),
        (
            IdentityWindowRange(start_year=2020, end_year=2030, include=["projected", "partial"]),
            {
                "kind": "range",
                "start_year": 2020,
                "end_year": 2030,
                "include": ["partial", "projected"],
            },
        ),
        (
            IdentityWindowRange(start_year=2020, end_year=2030, include=[]),
            {"kind": "range", "start_year": 2020, "end_year": 2030},
        ),
    ],
    ids=["no-opt-in", "opt-in-normal-form", "empty-opt-in-omitted"],
)
def test_an_identity_window_round_trips_its_opt_in(window, member):
    """A resubmitted range keeps its opt-in in the identity core's normal form, and one
    without an opt-in omits the `include` key."""
    assert _requested_window_member(_requested_window_from_identity(window)) == member


def test_a_posted_range_keeps_its_opt_in_whatever_its_window_holds():
    """A resubmitted 2010 to 2015 posting `include: ["partial"]` holds no partial year, and
    its opt-in stands as posted, so Search answers under the posted spec_id, as a stored
    identity does once its partial year completes; the run is stated, not served."""
    posted = IdentityWindowRange(start_year=2010, end_year=2015, include=["partial"])
    inp = SimpleNamespace(identity=SimpleNamespace(requested_window=posted))

    spec = _spec(_answer(_PARTIAL, _date_request_for_resubmit(cast(Any, inp))))

    assert spec.definition.identity.requested_window == posted
    assert _held(spec) == [("partial", False, _CRS, "OECD CRS", 2025, 2025, None)]


def test_a_latest_window_with_an_opt_in_is_refused():
    """Search reads held-back years on a range alone, so a `latest` window carrying an
    opt-in is refused rather than answered under another spec_id."""
    with pytest.raises(EngineInputError) as raised:
        _requested_window_from_identity(IdentityWindowLatest(include=["partial"]))
    assert raised.value.code == "request_invalid"


_LATEST = DateRequest(window=None, latest=True)


@pytest.mark.parametrize(
    ("facets", "named", "kind"),
    [
        pytest.param(_PARTIAL, "2025", "partial", id="partial"),
        pytest.param(_PROJECTED, "2020 to 2030", "projected", id="projected"),
    ],
)
def test_every_window_states_the_same_runs(facets, named, kind):
    """One series read at 2010 to 2015, at its latest, from 2000 and over a window naming its
    held-back years states the same runs. Only the window opting in to them serves them."""
    requests = {
        "2010-2015": _request("2010 to 2015"),
        "latest": _LATEST,
        "open": _since(2000),
        "named": _request(named),
    }
    held = {
        window: _spec(_answer(facets, request)).resolved.coverage.held_back or []
        for window, request in requests.items()
    }
    runs = {
        window: [(h.kind, h.source, h.start_year, h.end_year) for h in one]
        for window, one in held.items()
    }
    served = {window: [h.served for h in one] for window, one in held.items()}

    assert runs["2010-2015"] != []
    assert runs["latest"] == runs["open"] == runs["named"] == runs["2010-2015"]
    assert [one[0] for one in runs["named"]] == [kind]
    assert served == {"2010-2015": [False], "latest": [False], "open": [False], "named": [True]}


def test_held_back_is_absent_from_the_wire_when_nothing_is_held():
    """A coverage block of a source holding no partial or projected year carries no
    `held_back` key."""
    held = HeldBack(
        kind="partial",
        served=False,
        source=_CRS,
        source_name=None,
        start_year=2025,
        end_year=2025,
        reported_share=None,
    )
    spec = _spec(_answer(_PROJECTED[:1], _request("2010 to 2015")))
    assert "held_back" not in spec.resolved.coverage.model_dump(mode="json", exclude_none=True)
    stated = spec.resolved.coverage.model_copy(update={"held_back": [held]})
    assert stated.model_dump(mode="json")["held_back"][0]["kind"] == "partial"


def _role_answer(
    monkeypatch,
    search_settings,
    rows,
    metadata,
    *,
    bulk: bool,
    date_request: DateRequest | None = None,
) -> Spec:
    """Resolve a creditor's flows to Aland over `date_request`, by default an open window
    from 2000, per pair or in bulk, over coverage `rows` of `(facet, years)` described by
    `metadata`."""
    from series_coverage.schema import RoleNames
    from tests.engine import test_role_bearing_pipeline_e2e as role_bearing
    from tests.engine._fake_coverage import FakeCoverageService, patch_coverage

    flow, aland, creditor = role_bearing._SV_DCID, "country/AAA", "country/CRA"
    search_settings(obs_preview_max_pairs=1 if bulk else 120)
    index = patch_coverage(
        monkeypatch,
        [(flow, facet, creditor, aland, list(years)) for facet, years in rows],
        role_names={flow: RoleNames(donor="creditor", recipient="debtor")},
        facet_metadata=metadata,
    )
    with bound_catalogue(_catalogue()):
        result = role_bearing._run(
            entities=["Aland"],
            sv_dcid=flow,
            sv_arcs=role_bearing._declaring_arcs(observation_properties=["creditor", "debtor"]),
            resolve={"Aland": aland},
            sv_score=0.9,
            role_query="debt flows to Aland",
            sdmx=FakeSdmxClient(observations={}, availability={}),
            provenance_labels={_IDS: "IDS", _CRS: "CRS"},
            date_request=date_request or _since(2000),
            coverage=FakeCoverageService(index).snapshot(),
            variable_text="test flow",
        )
    return _spec(result)


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
def test_a_role_answer_states_the_years_it_holds_back(monkeypatch, search_settings, bulk):
    """A creditor's flows to Aland read per pair or in bulk over an open window. An
    IDS-shaped annual series reported to 2025 and projected 2026 to 2030 states the projections
    held back, and a CRS series to 2025 states 2025 held back as partly reported. Only the
    mint, which reads the rows, states the share reported, so Search states none."""
    from series_coverage.schema import FacetMetadata

    projected = _role_answer(
        monkeypatch,
        search_settings,
        [("f1", range(2000, 2026)), ("f3", range(2026, 2031))],
        {
            "f1": FacetMetadata(provenance_id=_IDS, observation_period="P1Y"),
            "f3": FacetMetadata(
                provenance_id=_IDS, measurement_method="Estimated", observation_period="P1Y"
            ),
        },
        bulk=bulk,
    )
    partial = _role_answer(
        monkeypatch,
        search_settings,
        [("f1", range(2000, 2026))],
        {"f1": FacetMetadata(provenance_id=_CRS)},
        bulk=bulk,
    )

    assert _held(projected) == [("projected", False, _IDS, "World Bank IDS", 2026, 2030, None)]
    assert _held(partial) == [("partial", False, _CRS, "OECD CRS", 2025, 2025, None)]


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
def test_a_named_partial_year_counts_its_headline_on_that_year(monkeypatch, search_settings, bulk):
    """ "Aid from France to Kenya in 2025" over a CRS series to 2025. The answer opts in to
    2025, and its headline counts 2025, the year it serves."""
    from series_coverage.schema import FacetMetadata

    spec = _role_answer(
        monkeypatch,
        search_settings,
        [("f1", range(2000, 2026))],
        {"f1": FacetMetadata(provenance_id=_CRS)},
        bulk=bulk,
        date_request=_request("2025"),
    )

    assert spec.definition.identity.requested_window.include == ["partial"]
    assert _held(spec) == [("partial", True, _CRS, "OECD CRS", 2025, 2025, None)]
    headline = spec.resolved.headline
    assert headline is not None
    assert (headline.year, headline.series) == (2025, 1)


def _counting_runs(monkeypatch) -> list[object]:
    """Every plan whose runs are read (`held_back.held_runs`), wherever they are read."""
    import search.engine.region.facets as facets_module
    import search.engine.retrieval.held_back as held_back_module

    calls: list[object] = []
    read = held_back_module.held_runs

    def counted(plan, *, coverage_index):
        calls.append(plan)
        return read(plan, coverage_index=coverage_index)

    monkeypatch.setattr(facets_module, "held_runs", counted)
    monkeypatch.setattr(held_back_module, "held_runs", counted)
    return calls


@pytest.mark.parametrize(
    ("facets", "text"),
    [
        pytest.param(_PARTIAL, "2025", id="named-partial-year"),
        pytest.param(_PROJECTED, "2020 to 2030", id="range-to-projections"),
        pytest.param(_PARTIAL, "2010 to 2015", id="window-holding-none"),
    ],
)
def test_an_answer_reads_its_runs_once(monkeypatch, facets, text):
    """An answer reads its sources' runs once, where it states its plan
    (`facets.plan_answer`): settling the opt-in and stating `held_back` read them from the
    plan."""
    calls = _counting_runs(monkeypatch)

    spec = _spec(_answer(facets, _request(text)))

    assert spec.resolved.coverage.held_back
    assert len(calls) == 1


@pytest.mark.parametrize("bulk", [False, True], ids=["per_pair", "bulk"])
def test_a_role_answer_reads_its_runs_once(monkeypatch, search_settings, bulk):
    """A role-bearing answer, whose runs read each member's coverage probe, reads them once
    per pair and in bulk."""
    from series_coverage.schema import FacetMetadata

    calls = _counting_runs(monkeypatch)
    spec = _role_answer(
        monkeypatch,
        search_settings,
        [("f1", range(2000, 2026))],
        {"f1": FacetMetadata(provenance_id=_CRS)},
        bulk=bulk,
        date_request=_request("2025"),
    )

    assert _held(spec) == [("partial", True, _CRS, "OECD CRS", 2025, 2025, None)]
    assert len(calls) == 1
