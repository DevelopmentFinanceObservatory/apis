"""Search reads a role a DAC2a composite binds to its bilateral population as the open role
the caller posted, within the population's members: at the resubmit door's role-scope
discovery and whole totals, in each component's preview, in the part memo key, and at the
options door. A posted identity binding the composite to an agency or the total spanning
both sides refuses, naming the dataset whose total ODA reads it. Search's SDMX reads key a
population filter (`dc_sdmx.Among`) by its members.

The fixture is `test_dac2a_destination_populations`': France's 206 to Ethiopia, to the
European Commission and to `DACAllRecipients`, and its imputed 106 to Ethiopia. The United
States holds a 206 row to the European Commission alone.
"""

from __future__ import annotations

import time

import pytest

from code_index.composites import (
    PopulationUnavailable,
    UnclassifiedDestinations,
    destination_population,
)
from code_index.declarations import DAC2A_INCLUSIVE
from code_index.shared import get_index
from dc_sdmx import Among
from search.engine import identity_edit
from search.engine.clients.materialise_sdmx_cache import _entities_key
from search.engine.entry import dataset_entries
from search.engine.errors import EngineInputError
from search.engine.options.options_response_cache import options_response_cache_key
from search.engine.options.resolve import resolve_options_target
from search.engine.options.surface import compute_options
from search.engine.region import composite_answer
from search.engine.region.observations.bulk_pair_reads import _bound_key
from search.engine.response_cache_keys import part_memo_key
from search.engine.retrieval import bound_values
from search.engine.retrieval.bound_values import identity_query
from search.heading import pins
from search.heading.pins import identity_pinned_properties
from search.models import (
    BaseIdentity,
    CandidatesPart,
    DatasetResubmitInput,
    OptionsFor,
    OptionsRequest,
    ResolveRequest,
    ResubmitCandidate,
    ResubmitCandidatesEntry,
    ResubmitIdentityPayload,
)
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._harness import offline_resolve
from tests.engine.test_aid_dataset_bind_e2e import _ScriptedLLM
from tests.engine.test_dac2a_destination_populations import (
    _AGENCY,
    _BILATERAL,
    _HELD,
    _NAMES,
    _SPURIOUS,
    _YEARS,
    _graph_with_parties,
    _identity,
    _resolve_resubmit,
    _resubmit,
    _sdmx,
)
from tests.engine.test_whole_total_bind_e2e import (
    _ALL_RECIPIENTS,
    _DAC2A,
    _DAC2A_ATTRIBUTED,
    _DAC2A_MEMBER,
    _ETH,
    _VARIABLE,
)
from tests.materialise.conftest import FakeGraphReader
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

_USA = "country/USA"
_DEVELOPING = "DACDevelopingCountries"
_COMPOSITES = [
    pytest.param(_BILATERAL, id="bilateral"),
    pytest.param(DAC2A_INCLUSIVE, id="inclusive"),
]


def _rows(held: dict[tuple[str, str, str], int]) -> list:
    return [(sv, "dac2a", donor, recipient, list(_YEARS)) for (sv, donor, recipient) in held]


def _patch(monkeypatch: pytest.MonkeyPatch, held: dict[tuple[str, str, str], int]):
    return patch_coverage(
        monkeypatch,
        _rows(held),
        role_names=dict.fromkeys((_DAC2A_MEMBER, _DAC2A_ATTRIBUTED), _NAMES),
        facet_metadata={"dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD")},
    )


@pytest.fixture(autouse=True)
def _coverage(monkeypatch: pytest.MonkeyPatch):
    return _patch(monkeypatch, {**_HELD, (_DAC2A_MEMBER, _USA, _AGENCY): 50})


@pytest.mark.parametrize("provenance", _COMPOSITES)
def test_a_resubmit_discovers_its_open_destination_within_the_population(provenance: str) -> None:
    part = _resolve_resubmit(_identity(provenance))

    scope = part.spec.resolved.scope
    assert scope is not None
    assert {member.ref for member in scope.members} == {_ETH}


@pytest.mark.parametrize("provenance", _COMPOSITES)
def test_a_resubmit_offers_the_population_total_in_place_of_the_spanning_one(
    provenance: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, {**_HELD, (_DAC2A_MEMBER, "country/FRA", _DEVELOPING): 600})

    part = _resolve_resubmit(_identity(provenance))

    totals = {w.params["total"] for w in part.warnings if w.code == "whole_total_available"}
    assert totals == {_DEVELOPING}


@pytest.mark.parametrize("destination", [_AGENCY, _ALL_RECIPIENTS])
@pytest.mark.parametrize("provenance", _COMPOSITES)
def test_a_composite_posted_with_a_destination_off_its_side_refuses(
    provenance: str, destination: str
) -> None:
    identity = BaseIdentity.model_validate(
        _identity(provenance).model_dump(exclude_none=True)
        | {"entities": {"origin": ["country/FRA"], "destination": [destination]}, "open_roles": []}
    )

    with pytest.raises(EngineInputError) as refused:
        offline_resolve(
            _resubmit(identity),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )

    error = refused.value
    assert (error.code, error.status) == ("requested_option_unavailable", 422)
    assert error.pointer == "/input/entries/0/identity/entities/destination"
    assert error.cause is not None
    assert (error.cause.code, error.cause.entities) == ("multilateral_destination", [destination])
    assert [alternative.ref for alternative in error.alternatives or ()] == [_DAC2A]


def test_a_population_read_and_a_named_list_of_its_members_key_apart() -> None:
    population = destination_population(get_index(), _DAC2A)
    identity = BaseIdentity.model_validate(
        _identity(_DAC2A).model_dump(exclude_none=True)
        | {"entities": {"origin": ["country/FRA"], "destination": list(population.members)}}
        | {"open_roles": []}
    )

    def key(payload: ResubmitIdentityPayload) -> str:
        return part_memo_key(
            payload=payload,
            engine_flags={},
            build_commit="b",
            graph_generation="g",
            coverage_stamp="c",
            conversion_catalogue=None,
            origin_catalogue="o",
            un_system_enabled=False,
            today="2026-11-15",
        )

    named = ResubmitIdentityPayload(identity=identity)
    within = ResubmitIdentityPayload(identity=identity, populations={"destination": population})

    assert key(within) != key(named)


@pytest.mark.parametrize("provenance", _COMPOSITES)
def test_each_component_preview_reads_its_destination_within_the_population(
    provenance: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    posted: list[ResubmitIdentityPayload] = []
    resolve = composite_answer.resolve_spec_resubmit

    def spy(*, inp: ResubmitIdentityPayload, **kwargs):
        posted.append(inp)
        return resolve(inp=inp, **kwargs)

    monkeypatch.setattr(composite_answer, "resolve_spec_resubmit", spy)

    offline_resolve(
        _resubmit(_identity(provenance), observations=True),
        graph=_graph_with_parties(),
        llm=_ScriptedLLM(_VARIABLE),
        sdmx=_sdmx(),
    )

    assert posted
    assert all(set(payload.populations) == {"destination"} for payload in posted)


def _target(role: str, provenance: str | None, coverage: CoverageIndex):
    """The options target of `role` for the fixture identity under `provenance`, or under
    none, the table identity a DAC2a composite's option was set on."""
    identity = _identity(provenance or _DAC2A)
    if provenance is None:
        identity = identity.model_copy(update={"facets": None})
    request = OptionsRequest(identity=identity, **{"for": OptionsFor(role=role)})
    graph = _graph_with_parties()
    graph._nodes.setdefault(_USA, {"label": "United States", "type": "Country"})
    return (
        request,
        graph,
        resolve_options_target(
            request,
            graph=graph,
            materialise_graph=FakeGraphReader(),
            coverage=FakeCoverageService(coverage).snapshot(),
        ),
    )


def _options(role: str, provenance: str, coverage) -> set[str]:
    request, graph, target = _target(role, provenance, coverage)
    computed = compute_options(
        request,
        target,
        graph=graph,
        deadline=time.monotonic() + 30.0,
        coverage=FakeCoverageService(coverage).snapshot(),
    )
    return {item.value.ref.ref for item in computed.options.items if item.value.ref is not None}


@pytest.mark.parametrize("provenance", _COMPOSITES)
def test_the_destination_options_offer_no_agency_or_spanning_total(provenance: str, _coverage):
    offered = _options("destination", provenance, _coverage)

    assert _ETH in offered
    assert not offered & _SPURIOUS


@pytest.mark.parametrize("provenance", _COMPOSITES)
def test_the_origin_options_list_the_donors_holding_rows_at_the_population(
    provenance: str, _coverage
):
    offered = _options("origin", provenance, _coverage)

    assert offered == {"country/FRA", "DACOfficialDonors"}


@pytest.mark.parametrize("role", ["destination", "origin"])
def test_a_composite_s_options_key_apart_from_its_table_identity_s(role: str, _coverage) -> None:
    def key(provenance: str | None) -> str:
        request, _graph, target = _target(role, provenance, _coverage)
        return options_response_cache_key(
            request,
            target,
            build_commit="b",
            coverage=FakeCoverageService(_coverage).snapshot(),
            catalogue=FakeCatalogue(),
        )

    assert key(_BILATERAL) != key(None)


def test_a_population_filter_keys_search_s_sdmx_reads_by_its_members() -> None:
    members = ("country/ETH", "country/KEN")
    among = Among(members=members, side="bilateral")

    assert _entities_key(among) == members
    assert _bound_key({"financingRecipient": among}) == (("financingRecipient", members),)


_UNPLACED = "DACRecipientOnNoSide"
"""A DAC2a destination the code index holds no row for, so no side declaration places it."""


def _refused_as_unavailable(
    error: EngineInputError, *, pointer: str, code: str = "requested_option_unavailable"
) -> None:
    assert (error.code, error.status, error.pointer) == (code, 422, pointer)
    assert error.cause is not None
    assert (error.cause.code, error.cause.entities) == ("provider_not_offered", [_UNPLACED])
    assert [alternative.ref for alternative in error.alternatives or ()] == [_DAC2A]


@pytest.mark.parametrize("provenance", _COMPOSITES)
def test_a_composite_whose_coverage_holds_a_destination_on_no_side_refuses_at_resubmit(
    provenance: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coverage snapshot holds a DAC2a recipient no side declaration places, so the
    population would leave it out without a word (`population_check`): a posted composite
    refuses as its scope option reads, not offered, with the destination named and total
    ODA as the alternative, where Search would otherwise answer a scope the mint refuses."""
    _patch(monkeypatch, {**_HELD, (_DAC2A_MEMBER, "country/FRA", _UNPLACED): 5})

    with pytest.raises(EngineInputError) as refused:
        offline_resolve(
            _resubmit(_identity(provenance)),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )

    _refused_as_unavailable(refused.value, pointer="/input/entries/0/identity/facets/provenance")


def _unavailable(provenance: str):
    """A `table_identity` whose code index publishes a destination no side places."""

    def read(_index, _identity):
        raise PopulationUnavailable(
            UnclassifiedDestinations(
                dataset=_DAC2A, destinations=(_UNPLACED,), found_in="code_index"
            ),
            ref=provenance,
        )

    return read


@pytest.mark.parametrize("provenance", _COMPOSITES)
def test_a_composite_whose_code_index_places_no_side_refuses_at_resubmit(
    provenance: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A code index publishing a destination no side declaration places makes the
    population unavailable where the composite derives (`PopulationUnavailable`): the
    resubmit door refuses it as a scope it does not offer, not as a server fault."""

    monkeypatch.setattr(identity_edit, "table_identity", _unavailable(provenance))

    with pytest.raises(EngineInputError) as refused:
        offline_resolve(
            _resubmit(_identity(provenance)),
            graph=_graph_with_parties(),
            llm=_ScriptedLLM(_VARIABLE),
            sdmx=_sdmx(),
        )

    _refused_as_unavailable(refused.value, pointer="/input/entries/0/identity/facets/provenance")


def test_a_composite_whose_coverage_holds_a_destination_on_no_side_refuses_its_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coverage = _patch(monkeypatch, {**_HELD, (_DAC2A_MEMBER, "country/FRA", _UNPLACED): 5})

    with pytest.raises(EngineInputError) as refused:
        _target("destination", _BILATERAL, coverage)

    # The options door registers no scope refusal, so it refuses the facet it cannot serve.
    _refused_as_unavailable(
        refused.value, pointer="/identity/facets/provenance", code="facet_not_available"
    )


@pytest.mark.parametrize("reader", ["pins", "family"])
def test_a_reader_of_a_composite_s_table_refuses_an_unavailable_population(
    reader: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reader of the table a composite reads, its pins or its family query, refuses an
    unavailable population as Search's doors refuse it, not as a server fault."""
    monkeypatch.setattr(pins, "table_identity", _unavailable(_BILATERAL))
    monkeypatch.setattr(bound_values, "table_identity", _unavailable(_BILATERAL))
    read = identity_pinned_properties if reader == "pins" else identity_query

    with pytest.raises(EngineInputError) as refused:
        read(_identity(_BILATERAL))

    _refused_as_unavailable(refused.value, pointer="/input/identity/facets/provenance")


def test_the_candidate_door_reads_each_composite_s_table_identity_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A posted composite candidate that the window election and the served bilateral line
    leave unchanged derives its table identity once, at the check that refuses a composite
    its components cannot serve."""
    read = dataset_entries.composite_table_identity
    derived: list[str | None] = []

    def counted(identity: BaseIdentity, **kwargs: object):
        derived.append(identity.facets["provenance"].ref if identity.facets else None)
        return read(identity, **kwargs)

    monkeypatch.setattr(dataset_entries, "composite_table_identity", counted)

    response = offline_resolve(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitCandidatesEntry(
                        variable_text=_VARIABLE,
                        ordering="intent_rank",
                        truncated=False,
                        candidates=[
                            ResubmitCandidate(identity=_identity(provenance))
                            for provenance in (_BILATERAL, DAC2A_INCLUSIVE)
                        ],
                    )
                ]
            )
        ),
        graph=_graph_with_parties(),
        llm=_ScriptedLLM(_VARIABLE),
        sdmx=_sdmx(),
    )

    (part,) = response.parts
    assert isinstance(part, CandidatesPart), part
    assert sorted(derived) == sorted([_BILATERAL, DAC2A_INCLUSIVE])
