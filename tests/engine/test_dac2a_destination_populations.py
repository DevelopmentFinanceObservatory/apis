"""A DAC2a composite that counts bilateral flows reads bilateral-side destinations alone, at
Search's identity resubmit door, at its observations preview, and on a fresh answer.

DAC2a aid type 206 holds a row for each multilateral agency and a `DACAllRecipients` row
spanning both sides, so an open destination that reads every row counts core contributions
beside the countries' rows. The fixture holds France's 206 to Ethiopia, to the European
Commission and to `DACAllRecipients`, and its imputed 106 to Ethiopia.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

import dc_graph
from code_index.composites import CompositeNotServable, destination_population
from code_index.declarations import DAC2A_INCLUSIVE
from code_index.shared import get_index
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from origin_catalogue import DFO
from search.engine.clients.graph import Facet
from search.engine.events import ObservationsEvent
from search.engine.quantity import NamedDenominator
from search.engine.region import (
    RegionResult,
    candidates,
    phase_answer,
    phase_place_materialise,
    pipeline,
    promote_materialise,
    scope_decision,
)
from search.engine.region.state import ResolveState
from search.engine.retrieval import role_resolve
from search.models import (
    BaseIdentity,
    DatasetResubmitInput,
    DefinitePart,
    IdentityBindingValue,
    NoDataPart,
    OptionAvailability,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    Relaxation,
    Resolution,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityEntry,
    Spec,
    Warning,
)
from search.render.heading import LATEST_YEAR_LABEL
from search.render.plan import render_sentence_plan
from search.scope import composite as scope_composite
from search.scope import provider
from search.scope.tables import ScopeFacts
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    ObsFacetFixture,
    build_materialise_double,
    offline_events,
    offline_resolve,
)
from tests.engine.test_aid_dataset_bind_e2e import _ScriptedLLM
from tests.engine.test_whole_total_bind_e2e import (
    _ALL_RECIPIENTS,
    _DAC2A,
    _DAC2A_ATTRIBUTED,
    _DAC2A_MEMBER,
    _DEU,
    _ETH,
    _FRA,
    _VARIABLE,
    _graph,
)
from tests.materialise.conftest import FakeGraphReader
from tests.origin_catalogue.fake_catalogue import FakeCatalogue, provenance_row

_BILATERAL = "provenance/ONEDAC2aBilateral"
_AGENCY = "EuropeanCommission"
_KEN = "country/KEN"
_OFFICIAL = "DACOfficialDonors"
_YEARS = (2019, 2020, 2021)
_NAMES = RoleNames(donor="financingProvider", recipient="financingRecipient")
# What each (stat var, donor, recipient) holds in every year.
_HELD = {
    (_DAC2A_MEMBER, _FRA, _ETH): 100,
    (_DAC2A_MEMBER, _FRA, _AGENCY): 200,
    (_DAC2A_MEMBER, _FRA, _ALL_RECIPIENTS): 500,
    (_DAC2A_ATTRIBUTED, _FRA, _ETH): 7,
    (_DAC2A_MEMBER, _OFFICIAL, _ETH): 900,
    (_DAC2A_MEMBER, _OFFICIAL, _AGENCY): 300,
    # Germany's total ODA in the window went to the agency alone.
    (_DAC2A_MEMBER, _DEU, _AGENCY): 400,
}
_SPURIOUS = {_AGENCY, _ALL_RECIPIENTS}


# The imputed line holds a recipient France's total ODA holds no row to, so an inclusive answer
# lists one recipient its bilateral table lacks.
_IMPUTED_ONLY = {**_HELD, (_DAC2A_ATTRIBUTED, _FRA, _KEN): 11}

# The bilateral population's total (`RolePopulation.total`), which a composite naming no
# destination binds in its place.
_DEVELOPING = "DACDevelopingCountries"
# Rows at the population's total in place of the total spanning both sides, so a part naming
# no destination reads the bilateral composite at that total.
_AT_TOTAL = {
    **{key: value for key, value in _HELD.items() if key[2] != _ALL_RECIPIENTS},
    (_DAC2A_MEMBER, _FRA, _DEVELOPING): 150,
}


@pytest.fixture(autouse=True)
def _coverage(monkeypatch: pytest.MonkeyPatch) -> CoverageIndex:
    return _patched_coverage(monkeypatch, _HELD)


def _patched_coverage(
    monkeypatch: pytest.MonkeyPatch, held: dict[tuple[str, str, str], int]
) -> CoverageIndex:
    return patch_coverage(
        monkeypatch,
        [(sv, "dac2a", donor, recipient, list(_YEARS)) for (sv, donor, recipient) in held],
        role_names=dict.fromkeys((_DAC2A_MEMBER, _DAC2A_ATTRIBUTED), _NAMES),
        facet_metadata={"dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD")},
    )


def _graph_with_parties():
    """The whole-total fixture graph holding every party a population read posts, as the
    resubmit's reference check reads them."""
    graph = _graph(variable=_VARIABLE, members=(_DAC2A_MEMBER, _DAC2A_ATTRIBUTED))
    population = destination_population(get_index(), _DAC2A)
    parties = (*population.members, population.total, _AGENCY, _ALL_RECIPIENTS)
    for dcid in parties:
        graph._nodes.setdefault(dcid, {"label": dcid, "type": "Organization"})
    return graph


def _sdmx(held: dict[tuple[str, str, str], int] = _HELD) -> FakeSdmxClient:
    observations: dict[str, dict] = {}
    for (sv, donor, recipient), value in held.items():
        facet = Facet(
            earliest_date=str(_YEARS[0]),
            latest_date=str(_YEARS[-1]),
            obs_count=len(_YEARS),
            dates=tuple(str(y) for y in _YEARS),
            values=(float(value),) * len(_YEARS),
            provenance_id=_DAC2A,
            unit="USD",
            facet_id="dac2a",
        )
        key = (("financingProvider", donor), ("financingRecipient", recipient))
        observations.setdefault(sv, {})[key] = [facet]
    return FakeSdmxClient(observations=observations, availability={})


def _materialise() -> tuple[FakeGraphReader, MaterialiseSdmxClient]:
    """The materialise graph and SDMX doubles an open destination's preview reads per pair,
    holding the same series as `_sdmx`."""
    graph, sdmx, _requests = build_materialise_double(
        {
            (sv, (("financingProvider", donor), ("financingRecipient", recipient))): [
                ObsFacetFixture(
                    facet_id="dac2a",
                    points=tuple((str(year), float(value)) for year in _YEARS),
                    provenance_id=_DAC2A,
                    unit="USD",
                )
            ]
            for (sv, donor, recipient), value in _HELD.items()
        }
    )
    return graph, sdmx


def _identity(provenance: str) -> BaseIdentity:
    return BaseIdentity.model_validate(
        {
            "identity_version": "spec1",
            "shape": {
                "kind": "statvar_schema",
                "population_type": "FinancialFlow",
                "measured_property": "amount",
                "stat_type": "measuredValue",
                "measurement_qualifier": "Nominal",
            },
            "slots": [
                {"property": "dac2aAidType", "binding": _ref("DAC2aAidType206")},
                {"property": "financialFlowType", "binding": _ref("Disbursement")},
                {"property": "nettingBasis", "binding": _ref("Gross")},
            ],
            "entities": {"origin": [_FRA]},
            "open_roles": ["destination"],
            "requested_window": {"kind": "range", "start_year": 2019, "end_year": 2021},
            "facets": {"provenance": _ref(provenance)},
        }
    )


def _ref(ref: str) -> dict[str, str]:
    return {"state": "value", "ref": ref}


def _resubmit(identity: BaseIdentity, *, observations: bool = False) -> ResolveRequest:
    return ResolveRequest(
        input=DatasetResubmitInput(
            entries=[ResubmitIdentityEntry(variable_text=_VARIABLE, identity=identity)]
        ),
        response_options=ResolveOptions(include_observations=observations),
    )


@dataclass(frozen=True)
class _TableRead:
    """One read of a part's table: the step that made it (`answer` for a fresh part's read,
    `resubmit` for a posted identity's, `candidate` for a reading beside the answer,
    `nearest-real` for a suggestion's relaxed query), whether an open axis was discovered
    within a population's members, and the recipients the read binds. A nearest-real read
    names its places as subjects, not by role (`role_resolve.graph_confirm_resolve`), so its
    recipients are None."""

    step: str
    within: bool
    recipients: tuple[str, ...] | None


# Every function a part's table is read through, by the step that calls it. A nearest-real
# suggestion's relaxed query reads through `graph_confirm_resolve` by its module attribute
# (`grounding._relaxed_binding_suggestion`), which no other step reads it through.
_READ_SITES = {
    "answer": (phase_place_materialise, "run_materialise"),
    "resubmit": (promote_materialise, "run_materialise"),
    "candidate": (candidates, "materialise"),
    "nearest-real": (role_resolve, "graph_confirm_resolve"),
}


def _table_reads(monkeypatch: pytest.MonkeyPatch) -> list[_TableRead]:
    """Every read of a part's table from here on, in order, recorded where each step calls
    the read (`_READ_SITES`)."""
    reads: list[_TableRead] = []

    def counting(step: str, read: Callable[..., Any]) -> Callable[..., Any]:
        def counted(*args: Any, within: Any, **kwargs: Any) -> Any:
            # The role-keyed entities, the third positional argument of every read but the
            # nearest-real one, which names its places as subject dcids.
            recipients = (
                None if step == "nearest-real" else tuple(args[2].get("financingRecipient", ()))
            )
            reads.append(_TableRead(step, bool(within), recipients))
            return read(*args, within=within, **kwargs)

        return counted

    for step, (module, name) in _READ_SITES.items():
        monkeypatch.setattr(module, name, counting(step, getattr(module, name)))
    return reads


def _counted(reads: list[_TableRead]) -> list[tuple[str, bool]]:
    """Each read's step and whether it was made within a population."""
    return [(read.step, read.within) for read in reads]


def _part(response: Any) -> DefinitePart:
    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    return part


def _ended_state(
    monkeypatch: pytest.MonkeyPatch, scenario: str, *, sdmx: FakeSdmxClient | None = None
) -> ResolveState:
    """The state a fresh answer of `scenario` ends its pass with, read over `sdmx`, else the
    test's held rows (`_sdmx`): its shape, bindings and place, and the read that elected its
    reading."""
    captured: list[ResolveState] = []
    preview = pipeline._with_composite_preview

    async def capture(result: RegionResult, state: ResolveState) -> RegionResult:
        captured.append(state)
        return await preview(result, state)

    monkeypatch.setattr(pipeline, "_with_composite_preview", capture)
    offline_resolve(
        _SCENARIOS[scenario](),
        graph=_graph_with_parties(),
        llm=_ScriptedLLM(_VARIABLE),
        sdmx=_sdmx() if sdmx is None else sdmx,
    )
    (state,) = captured
    return state


def _resolve_resubmit(
    identity: BaseIdentity, *, held: dict[tuple[str, str, str], int] = _HELD
) -> DefinitePart:
    return _part(
        offline_resolve(
            _resubmit(identity),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(held),
        )
    )


@pytest.mark.parametrize(
    ("provenance", "series"),
    [
        pytest.param(_BILATERAL, 1, id="bilateral"),
        pytest.param(DAC2A_INCLUSIVE, 2, id="inclusive"),
    ],
)
def test_a_resubmit_counts_no_agency_or_spanning_series_for_an_open_destination(
    provenance: str, series: int
) -> None:
    spec = _resolve_resubmit(_identity(provenance)).spec

    headline = spec.resolved.headline
    assert headline is not None
    assert headline.role_counts["financingRecipient"] == 1
    assert headline.series == series


def _preview(identity: BaseIdentity) -> list[tuple[str, str, float]]:
    """Each series of the streamed preview of `identity` as (stat var, destination, 2021 value)."""
    materialise_graph, materialise_sdmx = _materialise()
    events = offline_events(
        _resubmit(identity, observations=True),
        graph=_graph_with_parties(),
        llm=_ScriptedLLM(_VARIABLE),
        sdmx=_sdmx(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        stream_slot_options=True,
    )
    (event,) = [e for e in events if isinstance(e, ObservationsEvent)]
    read: list[tuple[str, str, float]] = []
    for series in event.observations.series:
        value = next(p.value for p in series.points if p.date == "2021")
        assert value is not None
        destination = next(e.ref for e in series.entities if e.role == "destination")
        read.append((series.stat_var.ref, destination, value))
    return read


@pytest.mark.parametrize(
    ("provenance", "total"),
    [
        pytest.param(_BILATERAL, 100.0, id="bilateral"),
        pytest.param(DAC2A_INCLUSIVE, 100.0 + 7.0, id="inclusive"),
    ],
)
def test_a_composite_preview_holds_the_country_row_and_inclusive_adds_106(
    provenance: str, total: float
) -> None:
    read = _preview(_identity(provenance))

    assert {destination for _sv, destination, _value in read} == {_ETH}
    assert sum(value for _sv, _destination, value in read) == total


def _fresh_request(
    entities: list[tuple[str, str]],
    *,
    message: str,
    open_role: str | None,
    time_text: str = "2019-2021",
) -> ResolveRequest:
    return ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(
                    variable_text=_VARIABLE,
                    entities=[ParsedEntity(text=text, role_text=role) for role, text in entities],
                    open_roles=[open_role] if open_role else [],
                    time_text=time_text,
                )
            ],
            user_message=message,
        )
    )


_FRANCE = ("from", "France")
# H1 to H6 of the headline set that read DAC2a: France with every recipient open, with the
# inclusive scope stated, to all recipients, and every donor with no provider axis.
_SCENARIOS: dict[str, Callable[[], ResolveRequest]] = {
    "H1": lambda: _fresh_request(
        [_FRANCE], message="official development assistance from France", open_role="recipient"
    ),
    "H2": lambda: _fresh_request(
        [_FRANCE],
        message="official development assistance from France including imputed multilateral",
        open_role="recipient",
    ),
    "H5": lambda: _fresh_request(
        [_FRANCE, ("to", "all recipients")],
        message="official development assistance from France to all recipients",
        open_role=None,
    ),
    "H6": lambda: _fresh_request(
        [("from", "All Donors, Total")],
        message="official development assistance from all donors",
        open_role="recipient",
    ),
    # H1 asking for the latest year.
    "H1-latest": lambda: _fresh_request(
        [_FRANCE],
        message="official development assistance from France",
        open_role="recipient",
        time_text="latest",
    ),
    # A part naming no destination, read over `_AT_TOTAL`: France's ODA, and the top donors.
    "D2": lambda: _fresh_request(
        [_FRANCE], message="official development assistance from France", open_role=None
    ),
    "D2-ranking": lambda: _fresh_request(
        [], message="top donors of official development assistance", open_role="donor"
    ),
}


def _answer(
    spec: Spec, part: DefinitePart, *, words: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """What a fresh answer and a resubmit of its definition both state, without the warnings
    whose codes `words` names, which the request's own words raise."""
    resolved = spec.resolved
    dump = lambda model: None if model is None else model.model_dump(mode="json")  # noqa: E731
    return {
        "scope": dump(resolved.scope),
        "headline": dump(resolved.headline),
        "coverage": dump(resolved.coverage),
        "entities": [dump(entity) for entity in resolved.entities],
        "warnings": sorted((w.code, w.message) for w in part.warnings if w.code not in words),
    }


def _resolve_fresh(scenario: str, *, held: dict[tuple[str, str, str], int] = _HELD) -> DefinitePart:
    return _part(
        offline_resolve(
            _SCENARIOS[scenario](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(held),
        )
    )


@pytest.mark.parametrize(
    ("scenario", "held"),
    [
        *((scenario, _HELD) for scenario in ("H1", "H2", "H5", "H6")),
        pytest.param("D2", _AT_TOTAL, id="D2"),
        pytest.param("D2-ranking", _AT_TOTAL, id="D2-ranking"),
    ],
)
def test_a_fresh_answer_equals_a_resubmit_of_its_definition(
    scenario: str, held: dict[tuple[str, str, str], int], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patched_coverage(monkeypatch, held)
    fresh = _resolve_fresh(scenario, held=held)

    again = _resolve_resubmit(fresh.spec.definition.identity, held=held)

    assert again.spec.spec_id == fresh.spec.spec_id
    assert _answer(again.spec, again) == _answer(fresh.spec, fresh)


def _shared_fields(part: DefinitePart) -> dict[str, Any]:
    """What a fresh part and a resubmit of its definition must both state: its definition
    and spec_id, its scope, entities, headline, coverage, StatVar counts, measure label,
    provenance facet, data years and download, its warnings, and its departures without the
    words they quote."""
    spec = part.spec
    resolved = spec.resolved
    dump = lambda model: None if model is None else model.model_dump(mode="json")  # noqa: E731
    return {
        "spec_id": spec.spec_id,
        "definition": dump(spec.definition),
        "scope": dump(resolved.scope),
        "entities": [dump(entity) for entity in resolved.entities],
        "headline": dump(resolved.headline),
        "coverage": dump(resolved.coverage),
        "stat_vars": dump(resolved.stat_vars),
        "measure_label": resolved.measure_label,
        "provenance": dump(resolved.facets.get("provenance")),
        "data_years": resolved.data_years,
        "download": dump(resolved.download),
        "warnings": [dump(warning) for warning in part.warnings],
        "departures": [
            {key: value for key, value in dump(departure).items() if key not in ("asked", "text")}
            for departure in part.departures
        ],
    }


# The headline fields that say whether the request named the years it reads, which the door
# decides. A fresh `latest` request names no year, and a resubmit posts the year elected.
_DOOR_FACTS = ("named_window", "year_named")


@pytest.mark.parametrize("scenario", ["H1", "H2", "H6", "H5", "H1-latest"])
def test_a_fresh_composite_answer_reads_every_field_a_resubmit_reads(
    scenario: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every field of a fresh composite answer the rows it read decide is a resubmit's of
    its definition (`_shared_fields`), as for the plain table answer H5, the control.

    The fields the two doors state differently for the plain table too are left out: the
    trace's `query_qualifiers`, `slot_filters` provenance and `default_reading`, which the
    request's words decide, `available_at_place`, `inapplicable_slots`, and the region's
    `observation_entities`, `served_members`, `preview_read` and `served_filter`, which
    each door builds from its own read.

    The headline's `named_window` and `year_named` are door facts too (`_DOOR_FACTS`). A
    fresh `latest` request names no year, and a resubmit's identity carries the year its
    definition elected, so it names that year. Its figures and window of data are the
    same."""
    fresh = _part(
        offline_resolve(
            _SCENARIOS[scenario](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )
    )

    again = _resolve_resubmit(fresh.spec.definition.identity)

    shared, resubmitted = _shared_fields(fresh), _shared_fields(again)
    if scenario == "H1-latest":
        named = {
            door: {key: shared["headline"].pop(key) for key in _DOOR_FACTS}
            for door, shared in (("fresh", shared), ("resubmit", resubmitted))
        }
        year = fresh.spec.definition.identity.requested_window.end_year
        assert named == {
            "fresh": {"named_window": None, "year_named": False},
            "resubmit": {
                "named_window": {"start_year": year, "end_year": year},
                "year_named": True,
            },
        }
        assert fresh.spec.resolved.trace.requested_window == "latest"
        # The fresh byline states the latest-year label, which the resubmit's lacks (apis#312).
        assert any(token.text == LATEST_YEAR_LABEL for token in render_sentence_plan(fresh.spec))
    assert resubmitted == shared


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DevelopmentFinanceObservatory/apis#312: a composite identity keeps no latest window",
)
def test_a_latest_composite_answer_states_its_label_at_both_doors() -> None:
    """A `latest` request answered as the bilateral composite states "Latest year with data"
    in its byline, fresh and resubmitted. The composite's identity holds the year its lines
    elect in place of `latest` (`provider.elect_latest_window`), since the Datasets mint
    refuses a composite's `latest` window, so only the fresh answer's trace says the request
    asked for the latest year (`heading._asks_latest`), and the resubmit states no label."""
    fresh = _part(
        offline_resolve(
            _SCENARIOS["H1-latest"](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )
    )
    again = _resolve_resubmit(fresh.spec.definition.identity)

    labelled = {
        door: any(token.text == LATEST_YEAR_LABEL for token in render_sentence_plan(spec))
        for door, spec in (("fresh", fresh.spec), ("resubmit", again.spec))
    }
    assert labelled == {"fresh": True, "resubmit": True}


@pytest.mark.parametrize(
    ("scenario", "source"),
    [
        pytest.param("H1", _BILATERAL, id="H1"),
        pytest.param("H2", DAC2A_INCLUSIVE, id="H2"),
        # All recipients spans both sides, so the default reads total ODA there.
        pytest.param("H5", None, id="H5"),
        # A ranking over every donor has no provider axis and still reads the composite.
        pytest.param("H6", _BILATERAL, id="H6"),
    ],
)
def test_a_fresh_dac2a_answer_reads_the_composite_its_destinations_allow(
    scenario: str, source: str | None
) -> None:
    part = _part(
        offline_resolve(
            _SCENARIOS[scenario](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )
    )

    pin = (part.spec.definition.identity.facets or {}).get("provenance")
    assert (pin.ref if isinstance(pin, IdentityBindingValue) else None) == source


def test_a_total_read_as_the_bilateral_composite_keeps_its_table_s_origin_and_source(
    bind_catalogue: Callable[[FakeCatalogue], None],
) -> None:
    bind_catalogue(FakeCatalogue(provenances={_DAC2A: provenance_row(_DAC2A, DFO)}))

    def answer(scenario: str) -> DefinitePart:
        return _part(
            offline_resolve(
                _SCENARIOS[scenario](),
                graph=_graph_with_parties(),
                llm=_ScriptedLLM(_VARIABLE),
                sdmx=_sdmx(),
            )
        )

    plain = answer("H5").spec.resolved.facets["provenance"]
    fresh = answer("H1")
    resubmitted = _resolve_resubmit(fresh.spec.definition.identity)

    assert plain.ref == _DAC2A
    assert plain.data_commons is not None
    assert plain.source is not None
    for part in (fresh, resubmitted):
        facet = part.spec.resolved.facets["provenance"]
        assert facet.ref == _BILATERAL
        assert (facet.data_commons, facet.source) == (plain.data_commons, plain.source)


@pytest.mark.parametrize("scenario", ["H1", "H2"])
def test_a_fresh_bilateral_reading_lists_no_agency_in_its_scope(scenario: str) -> None:
    part = _part(
        offline_resolve(
            _SCENARIOS[scenario](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )
    )

    scope = part.spec.resolved.scope
    assert scope is not None
    assert not {member.ref for member in scope.members} & _SPURIOUS


def test_a_composite_whose_population_holds_no_row_answers_no_data_with_the_request_s_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh answer settled on the bilateral composite whose population holds no row in
    the window answers what a resubmit of its definition answers, no_data, under the
    request's own words, and never the whole table's read of the agency. The coverage the
    decision reads offers the composite only where it serves a year, so the decision is made
    to elect it here, as a coverage snapshot the read disagrees with would."""
    _offer_the_bilateral_composite(monkeypatch)
    request = _fresh_request(
        [("from", "Germany")],
        message="official development assistance from Germany",
        open_role="recipient",
    )

    response = offline_resolve(
        request, graph=_graph_with_parties(), llm=_ScriptedLLM(_VARIABLE), sdmx=_sdmx()
    )

    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, NoDataPart), part
    assert part.variable_text == _VARIABLE
    assert part.no_data.reason == "no_observations"


@pytest.mark.parametrize(
    ("scenario", "within"),
    [
        pytest.param("H1", True, id="within-the-population"),
        # The plain table answer, the control.
        pytest.param("H5", False, id="whole-table"),
    ],
)
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(dc_graph.GraphDeadlineExceededError("deadline"), id="deadline"),
        pytest.param(dc_graph.GraphReadError("lost"), id="lost-read"),
    ],
)
def test_a_read_within_the_population_that_raises_is_handled_as_any_table_read(
    scenario: str, within: bool, error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A part's one read, made within the composite's population, that crosses its deadline
    or loses its read raises the error a whole-table read raises, for the request's door to
    answer as it answers any lost read."""
    read_within: list[bool] = []

    def raising(*_args: Any, within: Any, **_kwargs: Any) -> Any:
        read_within.append(bool(within))
        raise error

    monkeypatch.setattr(phase_place_materialise, "run_materialise", raising)

    with pytest.raises(type(error)):
        offline_resolve(
            _SCENARIOS[scenario](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )

    assert read_within == [within]


def test_a_composite_answer_keeps_the_warnings_raised_before_its_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warning the pass raised before it read the elected shape, here one standing for the
    request's words, stays on a fresh answer read again within its composite's population,
    beside that read's own; the whole-table read's role scope does not."""
    words = Warning(code="query_qualifier_dropped", severity="info", message="the words")
    run_materialise_task = phase_place_materialise._run_materialise_task

    async def raising_first(state: Any) -> Any:
        state.warnings.append(words)
        return await run_materialise_task(state)

    monkeypatch.setattr(phase_place_materialise, "_run_materialise_task", raising_first)

    part = _part(
        offline_resolve(
            _SCENARIOS["H1"](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )
    )

    assert sorted((w.code, w.message) for w in part.warnings) == [
        ("query_qualifier_dropped", "the words"),
        (
            "role_scope_defaulted",
            "No recipient was named, so the answer covers the one recipient with data from France.",
        ),
    ]


def _offer_the_bilateral_composite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the coverage a scope reads offer the bilateral composite whatever years it
    serves, as a snapshot the read disagrees with would."""
    bilateral_option = provider._bilateral_option

    def available(*args: Any) -> Any:
        option = bilateral_option(*args)
        return option.model_copy(update={"availability": OptionAvailability(status="available")})

    monkeypatch.setattr(provider, "_bilateral_option", available)


def test_a_composite_answer_keeps_a_denominator_its_words_name_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`denominator_unverified` holds the base the question's words name against the shape
    the answer reads, and no row decides it, so the warning raised once the read is
    interpreted stands on a fresh answer read again within its composite's population. The
    words are named at that check, so no earlier quantity gate reads them."""
    note = phase_place_materialise.note_denominator_unverified

    def naming_a_base(state: Any, shape: Any, *, spec_id: str | None) -> None:
        state.named_denominator = NamedDenominator(ref=None, base_words="people")
        note(state, shape, spec_id=spec_id)

    monkeypatch.setattr(phase_answer, "note_denominator_unverified", naming_a_base)
    monkeypatch.setattr(phase_place_materialise, "check_denominator", lambda *_: "unverified")

    part = _part(
        offline_resolve(
            _SCENARIOS["H1"](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )
    )

    assert (part.spec.definition.identity.facets or {}).get("provenance") == (
        IdentityBindingValue(ref=_BILATERAL)
    )
    assert [w.spec_id for w in part.warnings if w.code == "denominator_unverified"] == [
        part.spec.spec_id
    ]


def test_an_inclusive_answer_counts_the_recipients_of_both_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh inclusive answer's scope, role axis fact and defaulted-scope warning count the
    recipients of the bilateral and imputed lines together, as its headline and the mint's
    frame do: the table identity it was resolved for reads total ODA alone, which holds no row
    to Kenya."""
    _patched_coverage(monkeypatch, _IMPUTED_ONLY)

    part = _part(
        offline_resolve(
            _SCENARIOS["H2"](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(_IMPUTED_ONLY),
        )
    )

    resolved = part.spec.resolved
    assert resolved.headline is not None
    assert resolved.headline.role_counts["financingRecipient"] == 2
    assert resolved.scope is not None
    assert [member.ref for member in resolved.scope.members] == [_ETH, _KEN]
    assert [(f.role, f.member_count) for f in resolved.role_axis_facts or ()] == [
        ("destination", 2)
    ]
    # The warning carries its count in its words alone, as "both" for two.
    (defaulted,) = [w.message for w in part.warnings if w.code == "role_scope_defaulted"]
    assert "both recipients" in defaulted


_RESIDUALS = ("DACAfricaUnallocated", "DACAsiaUnallocated")
_WITH_RESIDUALS = {**_HELD, **{(_DAC2A_MEMBER, _FRA, residual): 5 for residual in _RESIDUALS}}
"""France's rows beside two regional lines unallocated by country, which the code index
declares `residual` on DAC2a's recipient side."""


@pytest.mark.parametrize("scenario", ["H1", "H2"])
def test_an_answer_counts_its_residual_recipients_apart_from_its_parties(
    scenario: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A table answer and a composite one count three members, Ethiopia and two unallocated
    lines, of which the role axis fact counts the two lines as residual, and the
    defaulted-scope warning names Ethiopia as the one recipient beside the two unallocated
    amounts, and the heading names Ethiopia beside them. The scope lists all three, sorted
    by ref in both answers, as the frame holds them."""
    _patched_coverage(monkeypatch, _WITH_RESIDUALS)

    part = _resolve_fresh(scenario, held=_WITH_RESIDUALS)

    resolved = part.spec.resolved
    assert [(f.role, f.member_count, f.residual_count) for f in resolved.role_axis_facts or ()] == [
        ("destination", 3, 2)
    ]
    assert resolved.scope is not None
    assert [member.ref for member in resolved.scope.members] == sorted((_ETH, *_RESIDUALS))
    (defaulted,) = [w.message for w in part.warnings if w.code == "role_scope_defaulted"]
    assert "covers the one recipient and 2 unallocated amounts with data" in defaulted
    assert resolved.rendered_sentence is not None
    assert " to Ethiopia and 2 unallocated amounts " in resolved.rendered_sentence


def test_an_inclusive_answer_lists_the_parties_its_role_axis_fact_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Where a component's series reach a total on the open axis, here `DACAllRecipients`,
    an inclusive answer's scope lists the recipients its role axis fact counts, so the total
    the count leaves out is no member either."""
    _patched_coverage(monkeypatch, _IMPUTED_ONLY)
    served_parties = scope_composite._served_parties

    def reaching_a_total(served: Any, coverage: Any) -> Any:
        held = served_parties(served, coverage)
        return {**held, "recipient": held["recipient"] | {_ALL_RECIPIENTS}}

    monkeypatch.setattr(scope_composite, "_served_parties", reaching_a_total)

    part = _resolve_fresh("H2", held=_IMPUTED_ONLY)

    resolved = part.spec.resolved
    assert resolved.scope is not None
    assert [member.ref for member in resolved.scope.members] == [_ETH, _KEN]
    assert [(f.role, f.member_count) for f in resolved.role_axis_facts or ()] == [
        ("destination", 2)
    ]


@pytest.mark.parametrize(
    ("origin", "destination"),
    [("financingProvider", "financingRecipient"), ("creditor", "debtor")],
)
def test_a_component_counts_the_places_every_role_key_binds(
    _coverage: CoverageIndex, origin: str, destination: str
) -> None:
    """A component's years count the series under the places its entities bind whatever
    key names the role (`role_key_axis`), as its destinations are bound: Germany's ODA to
    Ethiopia holds no series, however the identity spells its roles."""
    identity = _identity(_DAC2A).model_dump(mode="json", exclude_none=True)
    del identity["facets"]
    del identity["open_roles"]
    facts = ScopeFacts(index=get_index(), coverage=FakeCoverageService(_coverage).snapshot())

    def years(entities: dict[str, list[str]]) -> dict[int, int]:
        return dict(
            scope_composite.component_years({**identity, "entities": entities}, facts=facts).counts
        )

    named = years({"origin": [_DEU], "destination": [_ETH]})
    assert named == years({origin: [_DEU], destination: [_ETH]})
    assert not any(named.values())


def test_an_inclusive_answer_whose_component_parties_cannot_be_listed_says_so(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Where the snapshot cannot list the parties of a component serving a year, an
    inclusive answer keeps the open axes of its table read, which count the bilateral line's
    recipients alone, and logs a WARNING naming the composite and the component."""
    _patched_coverage(monkeypatch, _IMPUTED_ONLY)
    lists_parties = scope_composite._lists_parties

    def unsettled(item: scope_composite.ServedComponent, coverage: CoverageIndex) -> bool:
        return item.component.role != "imputed" and lists_parties(item, coverage)

    monkeypatch.setattr(scope_composite, "_lists_parties", unsettled)

    with caplog.at_level(logging.WARNING, logger=scope_composite.__name__):
        part = _resolve_fresh("H2", held=_IMPUTED_ONLY)

    scope = part.spec.resolved.scope
    assert scope is not None and [member.ref for member in scope.members] == [_ETH]
    (record,) = [r for r in caplog.records if r.name == scope_composite.__name__]
    assert (record.levelno, record.args[0]) == (logging.WARNING, DAC2A_INCLUSIVE)


@pytest.mark.parametrize(
    ("scenario", "axes"),
    [
        pytest.param(
            "D2",
            {
                "measure_basis": "measure_basis:gross",
                "provider": "provider:bilateral",
            },
            id="D2",
        ),
        pytest.param("D2-ranking", {}, id="D2-ranking"),
    ],
)
def test_a_fresh_answer_naming_no_destination_states_the_scope_of_its_own_identity(
    scenario: str, axes: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh answer naming no destination reads the bilateral composite at the population's
    total, and that read never changes the part's identity or the scope decided on it: the
    answer and its scope's base name no destination, and France's part keeps its provider
    switch to total ODA, contributions included. The one read names the population's
    total, `DACDevelopingCountries`, and the composite identity names none."""
    _patched_coverage(monkeypatch, _AT_TOTAL)
    recorded = _table_reads(monkeypatch)

    part = _resolve_fresh(scenario, held=_AT_TOTAL)

    assert recorded == [_TableRead("answer", False, (_DEVELOPING,))]
    identity = part.spec.definition.identity
    assert "destination" not in identity.entities
    assert (identity.facets or {}).get("provenance") == IdentityBindingValue(ref=_BILATERAL)
    choices = part.spec.resolved.scope_choices
    assert choices is not None
    assert "destination" not in choices.base.identity.entities
    assert {choice.axis: choice.active for choice in choices.choices} == axes
    offered = [
        option.option_id
        for choice in choices.choices
        if choice.axis == "provider"
        for option in choice.options
    ]
    assert ("provider:contributions" in offered) == (scenario == "D2")


def test_a_fresh_bilateral_answer_s_component_carries_the_bound_population() -> None:
    """The compiled scope's component identity is the one the mint reads: its destination
    bound to the population's members, the list a client posting the component reads."""
    part = _part(
        offline_resolve(
            _SCENARIOS["H1"](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )
    )

    choices = part.spec.resolved.scope_choices
    assert choices is not None and choices.compiled.components is not None
    (component,) = choices.compiled.components
    population = destination_population(get_index(), _DAC2A)
    assert component.identity.entities["destination"] == list(population.members)
    assert not component.identity.open_roles


# A DAC2a destination the code index holds no row for, so `destination_side` cannot place it and
# the bilateral population is unavailable (`code_index.composites.population_check`).
_UNCLASSIFIED = "DACRecipientOnNoSide"


def _withheld(part: DefinitePart) -> list[tuple[str | None, dict[str, Any] | None]]:
    """The spec_id and params of each `bilateral_reading_unavailable` warning `part` holds."""
    return [
        (w.spec_id, w.params) for w in part.warnings if w.code == "bilateral_reading_unavailable"
    ]


def _years_short(monkeypatch: pytest.MonkeyPatch) -> None:
    """Coverage where the bilateral destinations hold every donor's rows in 2019 and 2020
    alone, while the agency's rows run to 2021, so the composite covers fewer years than the
    table."""
    patch_coverage(
        monkeypatch,
        [
            (sv, "dac2a", donor, recipient, [2019, 2020] if recipient == _ETH else list(_YEARS))
            for (sv, donor, recipient) in _HELD
        ],
        role_names=dict.fromkeys((_DAC2A_MEMBER, _DAC2A_ATTRIBUTED), _NAMES),
        facet_metadata={"dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD")},
    )


def _composite_refuses(pointer: str) -> Callable[[pytest.MonkeyPatch], None]:
    """A drift under which the bilateral composite refuses what the part binds at
    `pointer`."""

    def drift(monkeypatch: pytest.MonkeyPatch) -> None:
        served_components = provider.served_components

        def refusing(document: dict[str, Any], *, facts: Any) -> Any:
            if document.get("facets", {}).get("provenance", {}).get("ref") == _BILATERAL:
                raise CompositeNotServable("refused", ref=_BILATERAL, pointer=pointer)
            return served_components(document, facts=facts)

        monkeypatch.setattr(provider, "served_components", refusing)

    return drift


def _decision_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The part's scope decision raises once the bilateral default has read it."""

    def refusing(*_args: Any, **_kwargs: Any) -> Any:
        raise CompositeNotServable("refused", ref=DAC2A_INCLUSIVE, pointer="/slots/0")

    monkeypatch.setattr(scope_decision, "settle_provider", refusing)


@pytest.mark.parametrize(
    ("drift", "params"),
    [
        pytest.param(
            lambda monkeypatch: _patched_coverage(
                monkeypatch, {**_HELD, (_DAC2A_MEMBER, _OFFICIAL, _UNCLASSIFIED): 50}
            ),
            {"why": "destination_unclassified"},
            id="population-unavailable",
        ),
        pytest.param(_years_short, {"why": "option_not_covered"}, id="years-not-covered"),
        pytest.param(
            _composite_refuses("/slots/0"),
            {"why": "slot_not_served", "slot": "dac2aAidType"},
            id="slot-refused",
        ),
        pytest.param(
            _composite_refuses("/requested_window"),
            {"why": "option_not_covered"},
            id="window-refused",
        ),
        pytest.param(_decision_raises, {"why": "scope_undecided"}, id="decision-raises"),
    ],
)
def test_a_ranking_over_every_donor_says_when_it_cannot_read_bilateral_destinations(
    monkeypatch: pytest.MonkeyPatch,
    drift: Callable[[pytest.MonkeyPatch], Any],
    params: dict[str, str],
) -> None:
    """A fresh ranking over every donor has no provider axis to state why it reads total ODA,
    so where the bilateral default cannot apply, for a population `destination_side` cannot
    fully place, a composite covering fewer years than the answer or none the window reaches, a
    composite refusing a slot the part binds, or a scope decision that raises, it warns
    `bilateral_reading_unavailable` on its own spec_id with the registered reason, and the
    slot's property for a slot it cannot serve."""
    drift(monkeypatch)

    part = _part(
        offline_resolve(
            _SCENARIOS["H6"](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )
    )

    assert (part.spec.definition.identity.facets or {}).get("provenance") is None
    assert _withheld(part) == [(part.spec.spec_id, params)]


def test_a_part_with_a_provider_axis_states_an_unavailable_population_on_its_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A donor's part has a provider axis, whose bilateral option already says why it is not
    offered, so it carries no `bilateral_reading_unavailable` warning; a ranking over every
    donor that reads the composite carries none either."""
    every_donor = _part(
        offline_resolve(
            _SCENARIOS["H6"](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )
    )
    _patched_coverage(monkeypatch, {**_HELD, (_DAC2A_MEMBER, _FRA, _UNCLASSIFIED): 50})

    donor = _part(
        offline_resolve(
            _SCENARIOS["H1"](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )
    )

    choices = donor.spec.resolved.scope_choices
    assert choices is not None
    (provider_choice,) = (c for c in choices.choices if c.axis == "provider")
    bilateral = next(o for o in provider_choice.options if o.option_id == "provider:bilateral")
    assert bilateral.availability.detail == {"why": "destination_unclassified"}
    assert (_withheld(donor), _withheld(every_donor)) == ([], [])


_TOTAL_YEARS = tuple(range(2017, 2022))
# What each series holds when the bilateral destinations' rows end a year before the rows
# of the total spanning both sides, and the agency's, so the composite lacks the table's
# newest year.
_COMPOSITE_BEHIND = {key: _TOTAL_YEARS[:-1] if key[2] == _ETH else _TOTAL_YEARS for key in _HELD}
# The reverse. The total spanning both sides ends a year before the destinations, so the
# composite holds every year the table's newest five do.
_COMPOSITE_AHEAD = {
    key: _TOTAL_YEARS[:-1] if key[2] in (_AGENCY, _ALL_RECIPIENTS) else _TOTAL_YEARS
    for key in _HELD
}


def _ragged(
    monkeypatch: pytest.MonkeyPatch, years: dict[tuple[str, str, str], tuple[int, ...]]
) -> FakeSdmxClient:
    """Coverage and an SDMX double where each series holds the years `years` gives it."""
    patch_coverage(
        monkeypatch,
        [
            (sv, "dac2a", donor, recipient, list(span))
            for (sv, donor, recipient), span in years.items()
        ],
        role_names=dict.fromkeys((_DAC2A_MEMBER, _DAC2A_ATTRIBUTED), _NAMES),
        facet_metadata={"dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD")},
    )
    observations: dict[str, dict] = {}
    for (sv, donor, recipient), span in years.items():
        facet = Facet(
            earliest_date=str(span[0]),
            latest_date=str(span[-1]),
            obs_count=len(span),
            dates=tuple(str(y) for y in span),
            values=(float(_HELD[sv, donor, recipient]),) * len(span),
            provenance_id=_DAC2A,
            unit="USD",
            facet_id="dac2a",
        )
        key = (("financingProvider", donor), ("financingRecipient", recipient))
        observations.setdefault(sv, {})[key] = [facet]
    return FakeSdmxClient(observations=observations, availability={})


def _trailing(scenario: str) -> ResolveRequest:
    (part,) = _SCENARIOS[scenario]().input.parts
    return ResolveRequest(
        input=ParsedInput(
            parts=[part.model_copy(update={"time_text": "the last five years"})],
            user_message=_SCENARIOS[scenario]().input.user_message,
        )
    )


@pytest.mark.parametrize(
    ("years", "composite"),
    [
        # The settled range ends in 2021, a year the bilateral destinations lack.
        pytest.param(_COMPOSITE_BEHIND, False, id="composite-behind"),
        pytest.param(_COMPOSITE_AHEAD, True, id="composite-ahead"),
    ],
)
@pytest.mark.parametrize("scenario", ["H1", "H6"])
def test_a_trailing_request_reads_the_reading_its_settled_range_allows(
    scenario: str,
    years: dict[tuple[str, str, str], tuple[int, ...]],
    composite: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Series ending in different years settle "the last five years" on the newest years
    the table holds, and the answer names the bilateral composite exactly when the
    composite holds every year of that range; otherwise it states
    `bilateral_reading_unavailable`. A resubmit of its definition answers the same."""
    sdmx = _ragged(monkeypatch, years)

    fresh = _part(
        offline_resolve(
            _trailing(scenario),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=sdmx,
        )
    )
    again = _part(
        offline_resolve(
            _resubmit(fresh.spec.definition.identity),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=sdmx,
        )
    )

    pin = (fresh.spec.definition.identity.facets or {}).get("provenance")
    assert (pin == IdentityBindingValue(ref=_BILATERAL)) == composite
    choices = fresh.spec.resolved.scope_choices
    assert choices is not None
    if scenario == "H6":
        # A ranking over every donor has no provider axis to state the reason on.
        assert [params for _spec_id, params in _withheld(fresh)] == (
            [] if composite else [{"why": "option_not_covered"}]
        )
    else:
        (provider_choice,) = (c for c in choices.choices if c.axis == "provider")
        bilateral = next(o for o in provider_choice.options if o.option_id == "provider:bilateral")
        assert bilateral.availability.code == (None if composite else "option_not_covered")
    # The trailing words' stale-year departure, which a resubmit of the definition does not
    # raise, is left out.
    words = frozenset({"date_coverage_partial"})
    assert again.spec.spec_id == fresh.spec.spec_id
    assert _answer(again.spec, again, words=words) == _answer(fresh.spec, fresh, words=words)


# A row of the read-count test: given the test's monkeypatch, the reads its answer needs
# stood in for, it returns the answer's read over an SDMX double, made once the reads are
# counted.
type _AnswerOf = Callable[[pytest.MonkeyPatch], Callable[[FakeSdmxClient], Spec]]


def _answered(request_of: Callable[[], ResolveRequest]) -> _AnswerOf:
    """The Spec a part posted as `request_of` answers."""

    def prepare(_monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeSdmxClient], Spec]:
        def answer(sdmx: FakeSdmxClient) -> Spec:
            return _part(
                offline_resolve(
                    request_of(),
                    graph=_graph_with_parties(),
                    llm=_ScriptedLLM(_VARIABLE),
                    sdmx=sdmx,
                )
            ).spec

        return answer

    return prepare


def _nearest_real(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeSdmxClient], Spec]:
    """The level suggestion of a fresh H1 answer, its elected shape under the same bindings.
    The fake graph resolves no relaxed query, so the relaxed read answers with the part's
    own read of the population, stood in before the reads are counted."""
    answered: list[Any] = []
    monkeypatch.setattr(role_resolve, "graph_confirm_resolve", lambda **_kwargs: answered[0])

    def suggest(sdmx: FakeSdmxClient) -> Spec:
        state = _ended_state(monkeypatch, "H1", sdmx=sdmx)
        answered.append(state.mat_result)
        level = (
            state.shape_draft,
            Relaxation(
                axis="what", property_dcid="measurementDenominator", dropped_value_dcids=["x"]
            ),
        )
        near = asyncio.run(phase_place_materialise._nearest_real_for(state, level=level))
        assert near is not None
        (suggestion,) = near
        return suggestion

    return suggest


@pytest.mark.parametrize(
    ("answer_of", "years", "source", "reads"),
    [
        *(
            pytest.param(
                _answered(_SCENARIOS[scenario]), None, source, [("answer", True)], id=scenario
            )
            for scenario, source in (
                ("H1", _BILATERAL),
                ("H6", _BILATERAL),
            )
        ),
        # A default request resolves the imputed component once, with no preview.
        pytest.param(
            _answered(_SCENARIOS["H2"]),
            None,
            DAC2A_INCLUSIVE,
            [("answer", True), ("resubmit", True)],
            id="H2",
        ),
        pytest.param(
            _answered(lambda: _trailing("H1")),
            _COMPOSITE_AHEAD,
            _BILATERAL,
            [("answer", True)],
            id="trailing",
        ),
        pytest.param(
            _answered(lambda: _resubmit(_identity(_BILATERAL))),
            None,
            _BILATERAL,
            [("resubmit", True)],
            id="resubmit",
        ),
        # The plain table answer, the control, makes one read of the whole table.
        pytest.param(_answered(_SCENARIOS["H5"]), None, None, [("answer", False)], id="H5"),
        # A suggestion beside the answer reads its relaxed query once, within the population.
        pytest.param(
            _nearest_real,
            None,
            _BILATERAL,
            [("answer", True), ("nearest-real", True)],
            id="nearest-real",
        ),
    ],
)
def test_a_part_settling_on_a_population_composite_reads_its_table_once_within_it(
    answer_of: _AnswerOf,
    years: dict[tuple[str, str, str], tuple[int, ...]] | None,
    source: str | None,
    reads: list[tuple[str, bool]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A part settling on a composite that binds a population reads its table once, within
    the population, fresh or resubmitted, as the plain table answer reads its table once
    whole. A nearest-real suggestion settling on the composite reads its own query once,
    within the population, beside the answer's read."""
    sdmx = _sdmx() if years is None else _ragged(monkeypatch, years)
    answer = answer_of(monkeypatch)
    recorded = _table_reads(monkeypatch)

    spec = answer(sdmx)

    pin = (spec.definition.identity.facets or {}).get("provenance")
    assert (pin.ref if isinstance(pin, IdentityBindingValue) else None) == source
    assert _counted(recorded) == reads


def test_a_donor_with_no_total_oda_rows_answers_no_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A donor holding no 206 row has no observation in the table read, so the part answers
    no_data with `no_observations`, as the table read alone would, where the composite's
    population read has no row to hold either. The fixture offers no sibling reading, so
    `_reelect_zero_observation_sibling` has nothing to elect."""
    held = {key: value for key, value in _HELD.items() if key[1] != _FRA}
    held[_DAC2A_ATTRIBUTED, _FRA, _ETH] = 7
    _patched_coverage(monkeypatch, held)
    response = offline_resolve(
        _SCENARIOS["H1"](),
        graph=_graph_with_parties(),
        llm=_ScriptedLLM(_VARIABLE),
        sdmx=_sdmx(held),
    )

    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, NoDataPart)
    assert part.no_data.reason == "no_observations"
