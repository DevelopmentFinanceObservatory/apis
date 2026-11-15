"""A DAC2a composite that counts bilateral flows reads bilateral-side destinations alone, at
Search's identity resubmit door, at its observations preview, and on a fresh answer.

DAC2a aid type 206 holds a row for each multilateral agency and a `DACAllRecipients` row
spanning both sides, so an open destination that reads every row counts core contributions
beside the countries' rows. The fixture holds France's 206 to Ethiopia, to the European
Commission and to `DACAllRecipients`, and its imputed 106 to Ethiopia.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

import dc_graph
from code_index.composites import CompositeNotServable, destination_population
from code_index.declarations import DAC2A_INCLUSIVE
from code_index.shared import get_index
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from origin_catalogue import DFO, current_catalogue
from search.engine.clients.graph import Facet
from search.engine.concurrency import scoped_concurrency
from search.engine.core import ResolveRecord
from search.engine.degraded_reads import note_degraded_read, recording_degraded_reads
from search.engine.departures import measure_not_applied, warning_key
from search.engine.entry import dataset_resubmit
from search.engine.errors import EngineInputError
from search.engine.events import ObservationsEvent
from search.engine.quantity import NamedDenominator
from search.engine.region import (
    RegionResult,
    composite_answer,
    phase_answer,
    phase_place_materialise,
    pipeline,
    result,
    scope_decision,
)
from search.engine.region.composite_answer import (
    ComponentReads,
    as_merged_composite,
    population_table_answer,
)
from search.engine.region.grounding import role_scope_defaulted_warning
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
    Resolution,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityEntry,
    Spec,
    Warning,
)
from search.scope import composite as scope_composite
from search.scope import provider
from search.scope.state import ScopeState
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


def _part(response: Any) -> DefinitePart:
    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    return part


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
    entities: list[tuple[str, str]], *, message: str, open_role: str | None
) -> ResolveRequest:
    return ResolveRequest(
        input=ParsedInput(
            parts=[
                ParsedPart(
                    variable_text=_VARIABLE,
                    entities=[ParsedEntity(text=text, role_text=role) for role, text in entities],
                    open_roles=[open_role] if open_role else [],
                    time_text="2019-2021",
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
    # A part naming no destination, read over `_AT_TOTAL`: France's ODA, and the top donors.
    "D2": lambda: _fresh_request(
        [_FRANCE], message="official development assistance from France", open_role=None
    ),
    "D2-ranking": lambda: _fresh_request(
        [], message="top donors of official development assistance", open_role="donor"
    ),
}


def _answer(spec: Spec, part: DefinitePart) -> dict[str, Any]:
    resolved = spec.resolved
    dump = lambda model: None if model is None else model.model_dump(mode="json")  # noqa: E731
    return {
        "scope": dump(resolved.scope),
        "headline": dump(resolved.headline),
        "coverage": dump(resolved.coverage),
        "warnings": sorted((w.code, w.message) for w in part.warnings),
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


# What a fresh part holds from the request's words or its door, which a resubmit of its
# definition holds none of: the parsed part it restates, the places each text bound, the
# scope its words and binders stated, and its timings. Departures quote the request's words.
_WORD_FIELDS = frozenset({"restatement", "entity_refs", "timing_ms"})


def _read_fields(part: DefinitePart) -> dict[str, Any]:
    """`part`'s dump without the fields its request's words decide, its departures without
    the words they quote."""
    dump = part.model_dump(mode="json", exclude=set(_WORD_FIELDS))
    dump["spec"]["resolved"].pop("scope_choices")
    dump["departures"] = [
        {key: value for key, value in departure.items() if key not in ("asked", "text")}
        for departure in dump["departures"]
    ]
    return dump


# What a fresh region holds from the request's words or its door, which a resubmit region of
# its definition holds otherwise: the part's words and timings, the phrases and places its
# words bound, the scope its binders stated, the relaunch facts and deadline of the door,
# and the closure reading the composite's preview over the door's clients.
_WORD_REGION_FIELDS = frozenset(
    {
        "variable_text",
        "timing_by_step",
        "admitted_qualifier_phrases",
        "scope_choices",
        "entity_words",
        "entity_refs",
        "promote_relaunch",
        "reads_deadline",
        "observations_relaunch",
    }
)


def _region_fields(region: RegionResult) -> dict[str, str]:
    """Each field of `region` the request's words do not decide, as its text with object
    addresses left out, so two closures over the same code compare equal."""
    return {
        field.name: re.sub(r" at 0x[0-9a-f]+", "", repr(getattr(region, field.name)))
        for field in dataclasses.fields(region)
        if field.name not in _WORD_REGION_FIELDS
    }


@pytest.mark.parametrize("scenario", ["H1", "H2", "H6"])
def test_a_fresh_composite_answer_reads_every_field_a_resubmit_reads(
    scenario: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every field of a fresh composite answer the rows it read decide, in its region and in
    its part, its Spec whole, its warnings and its departures, is a resubmit's of its
    definition: the fresh answer is that read with the request's words added, so no read
    field keeps the whole table's."""
    regions: list[RegionResult] = []
    scope_choices, as_composite = pipeline._with_scope_choices, dataset_resubmit._as_composite

    def fresh_region(result: RegionResult, state: Any) -> RegionResult:
        regions.append(scope_choices(result, state))
        return regions[-1]

    def resubmit_region(region: RegionResult, identity: BaseIdentity, **kwargs: Any) -> Any:
        regions.append(as_composite(region, identity, **kwargs))
        return regions[-1]

    monkeypatch.setattr(pipeline, "_with_scope_choices", fresh_region)
    monkeypatch.setattr(dataset_resubmit, "_as_composite", resubmit_region)
    fresh = _part(
        offline_resolve(
            _SCENARIOS[scenario](),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )
    )

    again = _resolve_resubmit(fresh.spec.definition.identity)

    assert _read_fields(again) == _read_fields(fresh)
    fresh_read, resubmitted = regions
    assert _region_fields(resubmitted) == _region_fields(fresh_read)


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


def _reads(coverage: CoverageIndex) -> ComponentReads:
    """The fixture's clients and coverage, as a part's component reads hold the request's."""
    return ComponentReads(
        graph=_graph_with_parties(),
        sdmx=_sdmx(),
        coverage=FakeCoverageService(coverage).snapshot(),
        materialise_graph=None,
        materialise_sdmx=None,
        rate_cache=None,
        deadline=None,
        catalogue=current_catalogue(),
    )


def _in_request_scope[T](call: Callable[[], T]) -> T:
    """`call`'s result, called inside a request's concurrency scope
    (`concurrency.scoped_concurrency`), as a direct call below `core.resolve_events` needs."""

    async def scoped() -> T:
        async with scoped_concurrency():
            return call()

    return asyncio.run(scoped())


def _definite(spec: Spec) -> RegionResult:
    return RegionResult(
        variable_text=_VARIABLE,
        status="definite",
        specs=(spec,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
    )


def test_a_merged_composite_reads_its_table_within_the_population(
    _coverage: CoverageIndex,
) -> None:
    """A same-shape merge of bilateral composite answers rebuilds its table Spec from its
    members' slots and plans; the composite it is restated as reads that table within the
    population, as a resubmit of the merged composite does, so a merged table Spec counting
    the agency leaves no trace. Two donors never fold, since a donor is no slot
    (`same_shape._resolve_diff_key`), so the merge is driven with a merged table Spec read
    over the whole table."""
    whole = _resolve_resubmit(_identity(_DAC2A).model_copy(update={"facets": None})).spec
    scope = whole.resolved.scope
    assert scope is not None and {member.ref for member in scope.members} >= _SPURIOUS
    member = _resolve_resubmit(_identity(_BILATERAL)).spec

    merged = as_merged_composite(_definite(whole), [_definite(member)], reads=_reads(_coverage))

    assert merged.spec.spec_id == member.spec_id
    resolved, again = merged.spec.resolved, member.resolved
    assert (resolved.scope, resolved.headline, resolved.coverage) == (
        again.scope,
        again.headline,
        again.coverage,
    )


def test_a_merged_composite_whose_population_holds_no_row_answers_no_data(
    _coverage: CoverageIndex,
) -> None:
    """A same-shape merge whose merged table read holds no row within the population
    answers that read's no_data, as a reading the read drops does, never the whole table's
    read restated as the composite: Germany's total ODA in the window went to the agency
    alone."""
    table = _identity(_DAC2A).model_copy(update={"facets": None})
    whole = _resolve_resubmit(table.model_copy(update={"entities": {"origin": [_DEU]}})).spec
    member = _resolve_resubmit(_identity(_BILATERAL)).spec

    merged = as_merged_composite(_definite(whole), [_definite(member)], reads=_reads(_coverage))

    assert (merged.status, merged.specs, merged.no_data_reason) == (
        "no_data",
        (),
        "no_observations",
    )


def test_a_candidate_restated_as_the_bilateral_composite_lists_no_agency_in_its_scope(
    _coverage: CoverageIndex,
) -> None:
    """A candidate set's DAC2a total ODA reading, resolved over every party, is read again
    within the bilateral population before it is restated as the composite, as a fresh
    definite answer is, so its scope and headline count the country alone."""
    table = _resolve_resubmit(_identity(_DAC2A).model_copy(update={"facets": None})).spec
    scope = table.resolved.scope
    assert scope is not None and {member.ref for member in scope.members} >= _SPURIOUS
    reads = _reads(_coverage)
    candidates = RegionResult(
        variable_text=_VARIABLE,
        status="candidates",
        specs=(table,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
    )

    (restated,) = _readings(candidates, reads=reads, selection=ScopeState()).specs

    pin = (restated.definition.identity.facets or {}).get("provenance")
    assert isinstance(pin, IdentityBindingValue) and pin.ref == _BILATERAL
    scope = restated.resolved.scope
    assert scope is not None
    assert {member.ref for member in scope.members} == {_ETH}
    headline = restated.resolved.headline
    assert headline is not None and headline.role_counts["financingRecipient"] == 1


@pytest.mark.parametrize("shown", ["candidate", "alternate"])
def test_a_reading_s_population_read_keys_its_warnings_to_the_reading(
    _coverage: CoverageIndex, monkeypatch: pytest.MonkeyPatch, shown: str
) -> None:
    """A reading pinning DAC2a's provenance is read within the bilateral population under the
    composite's table identity, whose spec_id is not the reading's. Every warning that read
    raises, keyed to its own Spec or to none, follows the reading to the composite it is
    restated as, so none names a Spec the response does not hold, whether the reading is the
    set's own candidate or the alternate a refusal offers."""
    table = _resolve_resubmit(_identity(_DAC2A)).spec
    population_read = composite_answer.population_table_answer

    def keyed(spec: Spec, identity: BaseIdentity, *, reads: ComponentReads) -> RegionResult:
        read = population_read(spec, identity, reads=reads)
        assert read is not None and read.status == "definite"
        assert read.spec.spec_id != spec.spec_id
        own = Warning(
            code="date_coverage_partial",
            severity="info",
            message="the read's",
            spec_id=read.spec.spec_id,
        )
        return dataclasses.replace(read, warnings=(*read.warnings, own))

    monkeypatch.setattr(composite_answer, "population_table_answer", keyed)
    region = (
        dataclasses.replace(_definite(table), status="candidates")
        if shown == "candidate"
        else RegionResult(
            variable_text=_VARIABLE,
            status="no_data",
            specs=(),
            no_data_reason="no_observations",
            warnings=(),
            timing_by_step={},
            alternate_match=table,
        )
    )

    read = _readings(region, reads=_reads(_coverage), selection=ScopeState())

    (restated,) = (*read.specs, *([read.alternate_match] if read.alternate_match else ()))
    assert (restated.definition.identity.facets or {}).get("provenance") == (
        IdentityBindingValue(ref=_BILATERAL)
    )
    assert [w.spec_id for w in read.warnings if w.code == "date_coverage_partial"] == [
        restated.spec_id
    ]
    assert {w.spec_id for w in read.warnings} <= {None, restated.spec_id}


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
    ("error", "reason", "degraded"),
    [
        pytest.param(
            EngineInputError(
                "no rows",
                code="facet_not_available",
                status=422,
                pointer="/input/identity/entities/destination",
            ),
            "facet_not_available",
            False,
            id="refused",
        ),
        pytest.param(
            dc_graph.GraphDeadlineExceededError("deadline"), "timed_out", True, id="deadline"
        ),
        pytest.param(dc_graph.GraphReadError("lost"), "no_observations", True, id="lost-read"),
    ],
)
def test_a_population_read_that_raises_answers_no_data_with_the_request_s_words(
    error: Exception, reason: str, degraded: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh answer whose population read raises answers no_data with the read's reason
    under the request's own words, degraded where the read was lost: never the whole
    table's read of every party, and never a refusal pointing at an identity the request
    did not post."""

    def raising(**_kwargs: Any) -> RegionResult:
        raise error

    monkeypatch.setattr(composite_answer, "resolve_spec_resubmit", raising)
    record = ResolveRecord()

    response = offline_resolve(
        _SCENARIOS["H1"](),
        graph=_graph_with_parties(),
        llm=_ScriptedLLM(_VARIABLE),
        sdmx=_sdmx(),
        record=record,
    )

    assert isinstance(response, Resolution), response
    (part,) = response.parts
    assert isinstance(part, NoDataPart), part
    assert (part.variable_text, part.no_data.reason) == (_VARIABLE, reason)
    assert record.done is not None and record.done.degraded_leg is degraded


def test_a_request_warning_follows_the_population_read_s_spec_id() -> None:
    """A warning the pass raised about its whole-table answer before the population read is
    keyed to the read's Spec, the one the part answers, so it never names a spec_id the
    response does not show."""
    whole = _resolve_resubmit(_identity(_DAC2A).model_copy(update={"facets": None})).spec
    read = _resolve_resubmit(
        _identity(_DAC2A).model_copy(update={"facets": None, "entities": {"origin": [_DEU]}})
    ).spec
    assert whole.spec_id != read.spec_id
    words = Warning(
        code="query_qualifier_dropped", severity="info", message="the words", spec_id=whole.spec_id
    )
    fresh = dataclasses.replace(_definite(whole), warnings=(words,))

    grafted = composite_answer.with_population_read(
        fresh, _definite(read), request_warnings=(words,)
    )

    assert [(w.code, w.spec_id) for w in grafted.warnings] == [
        ("query_qualifier_dropped", read.spec_id)
    ]


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


def test_a_candidate_whose_population_holds_no_row_is_dropped_not_restated(
    _coverage: CoverageIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate set's reading settled on the bilateral composite whose population holds no
    row in the window is dropped, never restated over the whole table's read of the agency,
    and a set left with none refuses with the reason that read gives."""
    _offer_the_bilateral_composite(monkeypatch)
    table = _resolve_resubmit(
        _identity(_DAC2A).model_copy(update={"facets": None, "entities": {"origin": [_DEU]}})
    ).spec
    candidates = dataclasses.replace(_definite(table), status="candidates")

    read = _readings(candidates, reads=_reads(_coverage), selection=ScopeState())

    assert (read.status, read.specs, read.no_data_reason) == ("no_data", (), "no_observations")


@pytest.mark.parametrize("with_france", [False, True], ids=["none-left", "one-left"])
def test_a_dropped_reading_takes_its_warnings_and_departures_with_it(
    _coverage: CoverageIndex, monkeypatch: pytest.MonkeyPatch, with_france: bool
) -> None:
    """A candidate whose population holds no row is dropped with the warnings keyed to it
    and the departure facts they disclose, whether the set refuses or is left with one
    reading."""
    _offer_the_bilateral_composite(monkeypatch)
    table = _identity(_DAC2A).model_copy(update={"facets": None})
    germany = _resolve_resubmit(table.model_copy(update={"entities": {"origin": [_DEU]}})).spec
    france = _resolve_resubmit(table).spec
    departed = Warning(
        code="query_qualifier_dropped",
        severity="degraded",
        message="Germany's words",
        spec_id=germany.spec_id,
    )
    candidates = dataclasses.replace(
        _definite(germany),
        status="candidates",
        specs=(france, germany) if with_france else (germany,),
        warnings=(departed,),
        departure_facts={
            warning_key(departed): measure_not_applied(asked="words", property_dcid=None)
        },
    )

    read = _in_request_scope(
        lambda: _readings(candidates, reads=_reads(_coverage), selection=ScopeState())
    )

    assert read.status == ("definite" if with_france else "no_data")
    assert departed.code not in {w.code for w in read.warnings}
    assert warning_key(departed) not in read.departure_facts


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


def test_a_composite_recounts_the_defaulted_scope_warning_on_its_axis_whatever_it_says(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A table answer's `role_scope_defaulted` warning names its axis in `params`, so the
    composite it is restated as states its own count on that axis whatever words the table's
    warning holds, here a label the request spelled its own way; a warning on another axis
    stands."""
    _patched_coverage(monkeypatch, _IMPUTED_ONLY)
    composite = _resolve_fresh("H2", held=_IMPUTED_ONLY).spec
    facts = [
        fact.model_copy(update={"member_count": 1})
        for fact in composite.resolved.role_axis_facts or ()
    ]
    table = composite.model_copy(
        update={
            "spec_id": "table",
            "resolved": composite.resolved.model_copy(update={"role_axis_facts": facts}),
        }
    )
    recipients = role_scope_defaulted_warning(
        "destination",
        "No recipient was named, so the answer covers the one recipient with data from FR.",
    ).model_copy(update={"spec_id": "table"})
    donors = role_scope_defaulted_warning("origin", "No provider was named.").model_copy(
        update={"spec_id": "table"}
    )

    recounted = result.recounted_scope((recipients, donors), table, composite, own=False)

    assert [w.message for w in recounted] == [
        "No recipient was named, so the answer covers both recipients with data from France.",
        donors.message,
    ]


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
    open_axes = scope_composite._open_axes

    def unsettled(stated: Any, served: Any, declaration: Any, *, facts: Any) -> Any:
        *listed, last = served
        return open_axes(
            stated, [*listed, dataclasses.replace(last, members=None)], declaration, facts=facts
        )

    monkeypatch.setattr(scope_composite, "_open_axes", unsettled)

    with caplog.at_level(logging.WARNING, logger=scope_composite.__name__):
        part = _resolve_fresh("H2", held=_IMPUTED_ONLY)

    scope = part.spec.resolved.scope
    assert scope is not None and [member.ref for member in scope.members] == [_ETH]
    (record,) = [r for r in caplog.records if r.name == scope_composite.__name__]
    assert (record.levelno, record.args[0]) == (logging.WARNING, DAC2A_INCLUSIVE)


def test_a_candidate_set_left_with_one_reading_answers_as_its_resubmit(
    _coverage: CoverageIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate set whose other reading holds no row within its population is left with
    one reading, so it answers that reading definite, as a resubmit of its composite does,
    and not as a set of one, stating the scope a definite answer states: compiled to the
    composite, with its components."""
    _offer_the_bilateral_composite(monkeypatch)
    table = _identity(_DAC2A).model_copy(update={"facets": None})
    france = _resolve_resubmit(table).spec
    germany = _resolve_resubmit(table.model_copy(update={"entities": {"origin": [_DEU]}})).spec
    candidates = dataclasses.replace(
        _definite(france), status="candidates", specs=(france, germany)
    )

    read = _in_request_scope(
        lambda: _readings(candidates, reads=_reads(_coverage), selection=ScopeState())
    )

    again = _resolve_resubmit(_identity(_BILATERAL)).spec
    assert (read.status, [spec.spec_id for spec in read.specs]) == ("definite", [again.spec_id])
    assert (read.spec.resolved.scope, read.spec.resolved.headline) == (
        again.resolved.scope,
        again.resolved.headline,
    )
    assert read.scope_choices is not None
    assert read.scope_choices.compiled.spec_id == again.spec_id
    assert read.scope_choices.compiled.components


def test_population_reads_take_the_request_s_limit_and_report_their_lost_reads(
    _coverage: CoverageIndex, monkeypatch: pytest.MonkeyPatch, search_settings: Any
) -> None:
    """Two or more population reads run in worker threads, and two parts' reads together run
    at most the request's concurrency limit at once. A read one of them loses reaches the
    request's record, so no cache stores the answer."""
    search_settings(max_variables=2, max_variable_concurrency=2)
    lock, running, peak = threading.Lock(), [0], [0]

    def reading(spec: Any, identity: Any, *, reads: Any) -> None:
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        note_degraded_read(f"read of {spec}")
        with lock:
            running[0] -= 1

    monkeypatch.setattr(composite_answer, "population_table_answer", reading)
    parts: list[Any] = [
        {name: (name, None) for name in (f"{part}-first", f"{part}-second")} for part in ("a", "b")
    ]
    reads = _reads(_coverage)

    async def read() -> tuple[list[dict[str, Any]], list[str]]:
        async with scoped_concurrency():
            with recording_degraded_reads() as record:
                answered = await asyncio.gather(
                    *(
                        asyncio.to_thread(composite_answer._population_reads, part, reads=reads)
                        for part in parts
                    )
                )
        return answered, record.reads

    answered, lost = asyncio.run(read())

    assert answered == [dict.fromkeys(part) for part in parts]
    assert sorted(lost) == sorted(f"read of {name}" for part in parts for name in part)
    assert peak[0] == 2


def test_a_composite_naming_no_destination_is_read_again_at_its_population_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A composite whose identity names no destination binds the population's total in its
    place (`RolePopulation.total`), which a whole-table answer of the same words never read,
    so a fresh answer settled on it is read again at that total, as a resubmit of it reads
    it."""
    assert destination_population(get_index(), _DAC2A).total == _DEVELOPING
    reads = dataclasses.replace(
        _reads(_patched_coverage(monkeypatch, _AT_TOTAL)), sdmx=_sdmx(_AT_TOTAL)
    )
    composite = _identity(_BILATERAL).model_copy(update={"open_roles": []})
    table = _resolve_resubmit(
        _identity(_DAC2A).model_copy(update={"facets": None}), held=_AT_TOTAL
    ).spec

    read = population_table_answer(table, composite, reads=reads)

    assert read is not None and read.status == "definite"
    assert read.spec.definition.identity.entities["destination"] == [_DEVELOPING]
    resolved = read.spec.resolved
    assert resolved.headline is not None
    assert (resolved.headline.series, resolved.headline.role_counts["financingRecipient"]) == (
        1,
        1,
    )


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
    switch to total ODA, contributions included."""
    _patched_coverage(monkeypatch, _AT_TOTAL)

    part = _resolve_fresh(scenario, held=_AT_TOTAL)

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


def test_a_population_read_the_composite_refuses_answers_no_data(
    _coverage: CoverageIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A composite that refuses the read of its population answers no_data, so the caller
    never states the whole table's read of every party under the composite's spec_id."""

    def refusing(*_args: Any, **_kwargs: Any) -> Any:
        raise EngineInputError("refused", code="requested_option_unavailable", status=422)

    monkeypatch.setattr(composite_answer, "composite_table_identity", refusing)
    table = _resolve_resubmit(_identity(_DAC2A).model_copy(update={"facets": None})).spec

    read = population_table_answer(table, _identity(_BILATERAL), reads=_reads(_coverage))

    assert read is not None
    assert (read.status, read.specs, read.no_data_reason) == ("no_data", (), "facet_not_available")


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


# A DAC2a destination the code index holds no row for, so no side declaration places it and
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


def _slot_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bilateral composite refuses what the part binds at its aid type slot."""
    served_components = provider.served_components

    def refusing(document: dict[str, Any], *, facts: Any) -> Any:
        if document.get("facets", {}).get("provenance", {}).get("ref") == _BILATERAL:
            raise CompositeNotServable("refused", ref=_BILATERAL, pointer="/slots/0")
        return served_components(document, facts=facts)

    monkeypatch.setattr(provider, "served_components", refusing)


def _decision_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The part's scope decision raises once the bilateral default has read it."""

    def refusing(*_args: Any, **_kwargs: Any) -> Any:
        raise CompositeNotServable("refused", ref=DAC2A_INCLUSIVE, pointer="/slots/0")

    monkeypatch.setattr(scope_decision, "settle_provider", refusing)


@pytest.mark.parametrize(
    ("drift", "why"),
    [
        pytest.param(
            lambda monkeypatch: _patched_coverage(
                monkeypatch, {**_HELD, (_DAC2A_MEMBER, _OFFICIAL, _UNCLASSIFIED): 50}
            ),
            "destination_unclassified",
            id="population-unavailable",
        ),
        pytest.param(_years_short, "option_not_covered", id="years-not-covered"),
        pytest.param(_slot_refused, "dac2aAidType", id="slot-refused"),
        pytest.param(_decision_raises, "scope_undecided", id="decision-raises"),
    ],
)
def test_a_ranking_over_every_donor_says_when_it_cannot_read_bilateral_destinations(
    monkeypatch: pytest.MonkeyPatch, drift: Callable[[pytest.MonkeyPatch], Any], why: str
) -> None:
    """A fresh ranking over every donor has no provider axis to state why it reads total ODA,
    so where the bilateral default cannot apply, for a population no side declaration fully
    places, a composite covering fewer years than the answer, a composite refusing a slot
    the part binds, or a scope decision that raises, it warns
    `bilateral_reading_unavailable` on its own spec_id with the reason."""
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
    assert _withheld(part) == [(part.spec.spec_id, {"why": why})]


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


def test_a_candidate_over_every_donor_says_when_it_cannot_read_bilateral_destinations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate set's reading over every donor that keeps total ODA because the bilateral
    population is unavailable warns as a definite answer does, keyed to that reading."""
    coverage = _patched_coverage(
        monkeypatch, {**_HELD, (_DAC2A_MEMBER, _OFFICIAL, _UNCLASSIFIED): 50}
    )
    table = _resolve_resubmit(
        _identity(_DAC2A).model_copy(update={"facets": None, "entities": {"origin": [_OFFICIAL]}})
    ).spec
    candidates = dataclasses.replace(_definite(table), status="candidates")

    read = _readings(candidates, reads=_reads(coverage), selection=ScopeState())

    assert [spec.spec_id for spec in read.specs] == [table.spec_id]
    assert [
        (w.spec_id, w.params) for w in read.warnings if w.code == "bilateral_reading_unavailable"
    ] == [(table.spec_id, {"why": "destination_unclassified"})]


def _readings(
    region: RegionResult, *, reads: ComponentReads, selection: ScopeState
) -> RegionResult:
    """`region` with its readings restated (`with_bilateral_readings`) under scope facts of
    no operation, as a part with no answer phase decision reads them."""
    return composite_answer.with_bilateral_readings(
        region,
        reads=reads,
        selection=selection,
        facts=ScopeFacts(index=get_index(), coverage=reads.coverage),
    )
