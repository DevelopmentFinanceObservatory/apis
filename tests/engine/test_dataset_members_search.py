"""Search reads a dataset of several member provenances as one dataset.

The synthetic `dataset/Ledger` splits one table's series between two provenances, as a
register filed by publisher does: `provenance/LedgerNorth` holds one recipient and
`provenance/LedgerSouth` another. A source scope allows the dataset by its id whichever of
its references a mention names, and a read under it admits exactly the members the mention
names: the dataset's id both, a member's id that member alone. An unscoped election serves
both members' facets as one group, and the answer cites and labels the dataset. The plain
table `provenance/Plain` is the control throughout.
"""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import cast

import pytest

from code_index import shared as shared_module
from code_index.query import CodeIndex
from code_index.types import DatasetDeclaration, IndexStamp
from datasets_api.adapter import provenance_pin
from graph_release.context import bound_release, current_release
from materialise.errors import DatasetDisagreementError
from materialise.facets import ElectedDataset
from origin_catalogue import bound_catalogue, current_catalogue
from origin_catalogue.catalogue import DatasetRef, ProvenanceRow, SourceRef
from origin_catalogue.context import current_source_scope
from origin_catalogue.origins import BASE, DFO
from search.engine import spec_build
from search.engine.clients.graph import Facet
from search.engine.errors import EngineInputError
from search.engine.extract import DateRequest
from search.engine.facet_refusals import candidate_facet_refusal
from search.engine.region.resubmit_request import pinned_provenance_scope
from search.engine.retrieval import read_plan
from search.engine.retrieval.facet_policy import NO_DECLARED_POLICIES
from search.engine.retrieval.role_resolve import _role_plan
from search.engine.source_scope import SourceScope, admitted_refs, compile_sources
from search.models import (
    BaseIdentity,
    IdentityBindingSet,
    IdentityBindingValue,
    ProvenanceRef,
    SourceMention,
    TimeWindow,
)
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from tests.engine import _answer_invariants_world as world_module
from tests.engine._answer_invariants_world import ALAND, BLAND, World
from tests.engine._fake_coverage import build_coverage_index
from tests.fixtures import FakeGraph
from tests.origin_catalogue.fake_catalogue import FakeCatalogue, provenance_row

_LEDGER = "dataset/Ledger"
_NORTH = "provenance/LedgerNorth"
_SOUTH = "provenance/LedgerSouth"
_PLAIN = "provenance/Plain"
_UNDECLARED = "provenance/Undeclared"
_PUBLISHER = "source/LedgerPublisher"


def _index() -> CodeIndex:
    """An index declaring the Ledger, whose members are North and South, and the plain
    table."""
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="test",
        ),
        declarations=(),
        codes=(),
        statvars=(),
        datasets=tuple(
            DatasetDeclaration(
                id=dataset,
                label=label,
                description="A test table.",
                opt_in=False,
                absent_observation="zero",
                account="provider",
                rank=rank,
            )
            for rank, (dataset, label) in enumerate(
                ((_LEDGER, "Ledger Register"), (_PLAIN, "Plain Table"))
            )
        ),
        dataset_members={_LEDGER: (_NORTH, _SOUTH)},
    )


@pytest.fixture
def index(monkeypatch: pytest.MonkeyPatch) -> CodeIndex:
    """Serve `_index()` wherever a loaded module reads the bound index."""
    index = _index()
    packaged = shared_module.get_index
    for module in list(sys.modules.values()):
        if vars(module).get("get_index") is packaged:
            monkeypatch.setattr(module, "get_index", lambda: index)
    return index


def _row(provenance: str, name: str, *, dataset: DatasetRef | None) -> ProvenanceRow:
    return ProvenanceRow(
        id=provenance,
        data_commons=DFO,
        source=SourceRef(ref=_PUBLISHER, label="Ledger Publisher"),
        name=name,
        url="https://example.org",
        dataset=dataset,
    )


_LEDGER_REF = DatasetRef(ref=_LEDGER, label="Ledger Register")
_CATALOGUE = FakeCatalogue(
    provenances={
        _NORTH: _row(_NORTH, "Ledger North", dataset=_LEDGER_REF),
        _SOUTH: _row(_SOUTH, "Ledger South", dataset=_LEDGER_REF),
        _PLAIN: _row(_PLAIN, "Plain Table", dataset=None),
        _UNDECLARED: _row(_UNDECLARED, "Undeclared Table", dataset=None),
    }
)


def _scope(*mentions: SourceMention) -> SourceScope:
    scope = compile_sources(mentions, catalogue=_CATALOGUE, pointers=["/"] * len(mentions))
    assert isinstance(scope, SourceScope)
    return scope


@pytest.mark.parametrize(
    ("mentions", "allowed", "provenances"),
    [
        pytest.param(
            [SourceMention(text="Ledger Register")], {_LEDGER}, {_NORTH, _SOUTH}, id="the-dataset"
        ),
        pytest.param([SourceMention(text="Ledger North")], {_LEDGER}, {_NORTH}, id="one-member"),
        pytest.param([SourceMention(ref=_SOUTH)], {_LEDGER}, {_SOUTH}, id="one-member-ref"),
        pytest.param(
            [SourceMention(text="Ledger North"), SourceMention(ref=_SOUTH)],
            {_LEDGER},
            {_NORTH, _SOUTH},
            id="both-members",
        ),
        pytest.param(
            [SourceMention(text="Ledger Register"), SourceMention(ref=_NORTH)],
            {_LEDGER},
            {_NORTH, _SOUTH},
            id="the-dataset-and-a-member",
        ),
        pytest.param([SourceMention(text="Plain Table")], {_PLAIN}, {_PLAIN}, id="plain-table"),
        pytest.param(
            [SourceMention(text="Ledger South", polarity="exclude")],
            {_LEDGER, _PLAIN, _UNDECLARED},
            {_NORTH, _PLAIN, _UNDECLARED},
            id="excluding-a-member",
        ),
        pytest.param(
            [
                SourceMention(text="Ledger Register"),
                SourceMention(text="Ledger South", polarity="exclude"),
            ],
            {_LEDGER},
            {_NORTH},
            id="the-dataset-less-a-member",
        ),
        pytest.param(
            [
                SourceMention(text="Ledger North", polarity="exclude"),
                SourceMention(ref=_SOUTH, polarity="exclude"),
            ],
            {_PLAIN, _UNDECLARED},
            {_PLAIN, _UNDECLARED},
            id="excluding-every-member",
        ),
        pytest.param(
            [SourceMention(text="Ledger Register", polarity="exclude")],
            {_PLAIN, _UNDECLARED},
            {_PLAIN, _UNDECLARED},
            id="excluding-the-dataset",
        ),
    ],
)
def test_a_scope_admits_exactly_the_members_its_mentions_name(
    index: CodeIndex, mentions: list[SourceMention], allowed: set[str], provenances: set[str]
) -> None:
    """A shape's datasets are compared with the scope's `allowed`, so the scope holds the
    dataset's id however a mention names it. A read under it admits the members the mentions
    name: the dataset's id every member, a member's id that member alone, and an exclusion
    removes what it names."""
    scope = _scope(*mentions)

    assert scope.allowed == allowed
    assert scope.provenances == provenances


def test_an_exclusion_leaving_no_member_refuses(index: CodeIndex) -> None:
    """The scope empties at the exclusion that removes the last admitted member."""
    with pytest.raises(EngineInputError) as refused:
        compile_sources(
            [SourceMention(text="Ledger North"), SourceMention(ref=_NORTH, polarity="exclude")],
            catalogue=_CATALOGUE,
            pointers=["/0", "/1"],
        )

    assert (refused.value.code, refused.value.pointer) == ("request_invalid", "/1")


@pytest.mark.parametrize(
    ("scope", "refs"),
    [
        pytest.param(None, (_LEDGER, _PLAIN), id="unscoped"),
        pytest.param({_NORTH, _SOUTH, _PLAIN}, (_LEDGER, _PLAIN), id="every-member"),
        pytest.param({_NORTH, _PLAIN}, (_NORTH, _PLAIN), id="one-member"),
    ],
)
def test_an_answer_pins_the_members_its_scope_admits(
    index: CodeIndex, scope: set[str] | None, refs: tuple[str, ...]
) -> None:
    """The identity names a dataset by its id when the read admits every member, so a set
    of every member hashes as the dataset does, and by the admitted members otherwise, so a
    read of one member never shares an answer with the dataset."""
    catalogue = _CATALOGUE if scope is None else _CATALOGUE.scoped(frozenset(scope))

    with bound_catalogue(catalogue):
        assert admitted_refs([_LEDGER, _PLAIN], code_index=index) == refs


# ---------------------------------------------------------------------------
# The election serves both members' facets as one.
# ---------------------------------------------------------------------------

_FLOW = "TEST/LedgerFlow"
_AAA, _BBB = "country/AAA", "country/BBB"
_YEARS = list(range(2015, 2021))


@dataclass(frozen=True)
class _Facts:
    """Member facts reading only what a role plan reads: no constraints, one declared role
    and the elected dataset."""

    dataset: ElectedDataset | None = None
    constraints: dict[str, str] = field(default_factory=dict)

    def constraints_known(self) -> bool:
        return True

    def read_constraints(self) -> dict[str, str]:
        return dict(self.constraints)

    def read_unresolved_constraints(self) -> frozenset[str]:
        return frozenset()

    def read_declared_roles(self) -> tuple[str, ...]:
        return ("debtor",)

    def read_dataset(self) -> ElectedDataset | None:
        return self.dataset


_LEDGER_DATASET = ElectedDataset(
    provenance=_LEDGER, provenances=(_NORTH, _SOUTH), recorded_provenances=(_NORTH, _SOUTH)
)
_PLAIN_DATASET = ElectedDataset(
    provenance=_PLAIN, provenances=(_PLAIN,), recorded_provenances=(_PLAIN, _UNDECLARED)
)


def _plan(
    search_settings, *, bulk: bool, north: str, south: str, dataset: ElectedDataset | None
) -> read_plan.ReadPlan:
    """The role plan of `_FLOW` at AAA and BBB, 2015 to 2020, electing `dataset`: facet
    `fN` of `north` holds AAA with more observations, `fS` of `south` holds BBB, both in
    one unit."""
    search_settings(obs_preview_max_pairs=1 if bulk else 1_000)
    coverage = build_coverage_index(
        [
            (_FLOW, "fN", ABSENT_ENTITY, _AAA, list(range(2010, 2021))),
            (_FLOW, "fS", ABSENT_ENTITY, _BBB, _YEARS),
        ],
        role_names={_FLOW: RoleNames(donor=None, recipient="debtor")},
        facet_metadata={
            "fN": FacetMetadata(provenance_id=north, unit="USDollar", observation_period="P1Y"),
            "fS": FacetMetadata(provenance_id=south, unit="USDollar", observation_period="P1Y"),
        },
    )
    return _role_plan(
        [_FLOW],
        member_facts={_FLOW: _Facts(dataset=dataset)},
        names={"destination": "debtor"},
        subject_dcids=[_AAA, _BBB],
        origin_dcids=[],
        bound_roles={"recipient": frozenset({_AAA, _BBB})},
        years=read_plan.role_years(
            DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False),
            coverage.horizon,
        ),
        coverage_index=coverage,
        declared=NO_DECLARED_POLICIES,
    )


def _served_ids(plan: read_plan.ReadPlan) -> set[str]:
    if isinstance(plan, read_plan.BulkPlan):
        return {facet_id for ids in plan.selected.values() for facet_id in ids}
    return {facet_id for s in plan.served for facet_id in s.group_ids}


@pytest.mark.parametrize("dataset", [_LEDGER_DATASET, None], ids=["elected", "no-election"])
@pytest.mark.parametrize("bulk", [True, False], ids=["bulk", "pair"])
def test_the_election_serves_both_members_facets_as_one(
    search_settings, index: CodeIndex, bulk: bool, dataset: ElectedDataset | None
) -> None:
    """Both members' facets carry the dataset's id on the provenance axis, whether or not
    the StatVar's family elects the dataset, so one group serves AAA from North and BBB from
    South, and the answer cites the dataset."""
    plan = _plan(search_settings, bulk=bulk, north=_NORTH, south=_SOUTH, dataset=dataset)

    assert isinstance(plan, read_plan.BulkPlan) is bulk
    assert _served_ids(plan) == {"fN", "fS"}
    assert read_plan.served_provenances(plan) == (_LEDGER,)


@pytest.mark.parametrize("bulk", [True, False], ids=["bulk", "pair"])
def test_two_plain_tables_elect_one(search_settings, index: CodeIndex, bulk: bool) -> None:
    """The control: one plain table holding both facets serves both; the elected plain
    table beside another one publishes `fN` alone, so `fS` serves nothing."""
    one = _plan(search_settings, bulk=bulk, north=_PLAIN, south=_PLAIN, dataset=_PLAIN_DATASET)
    assert _served_ids(one) == {"fN", "fS"}
    assert read_plan.served_provenances(one) == (_PLAIN,)

    two = _plan(search_settings, bulk=bulk, north=_PLAIN, south=_UNDECLARED, dataset=_PLAIN_DATASET)
    assert _served_ids(two) == {"fN"}
    assert read_plan.served_provenances(two) == (_PLAIN,)


# ---------------------------------------------------------------------------
# The answer names the elected dataset under its bound source slot's label.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("elected", "label"),
    [
        pytest.param(_LEDGER, "Ledger Register", id="dataset-of-several"),
        pytest.param(_PLAIN, "Plain Table", id="plain-table"),
    ],
)
def test_elected_sources_labels_the_elected_dataset(
    monkeypatch: pytest.MonkeyPatch, index: CodeIndex, elected: str, label: str
) -> None:
    """The answer names the elected dataset under its declared label, the label a bound
    source slot carries."""
    monkeypatch.setattr(spec_build, "identity_dataset", lambda _identity: elected)
    sources = [
        ProvenanceRef(ref=elected, label="graph name"),
        ProvenanceRef(ref=_NORTH, label="Ledger North"),
    ]

    with bound_catalogue(_CATALOGUE):
        named = spec_build.elected_sources(sources, cast(BaseIdentity, None))

    assert [(source.ref, source.label) for source in named] == [(elected, label)]


# ---------------------------------------------------------------------------
# A pin of the dataset reads every member, and a pin of one member reads it alone.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pinned", "scope"),
    [
        pytest.param(IdentityBindingValue(ref=_LEDGER), {_NORTH, _SOUTH}, id="the-dataset"),
        pytest.param(IdentityBindingValue(ref=_SOUTH), {_SOUTH}, id="one-member"),
        pytest.param(
            IdentityBindingSet(refs=[_NORTH, _SOUTH]), {_NORTH, _SOUTH}, id="both-members"
        ),
        pytest.param(IdentityBindingValue(ref=_PLAIN), {_PLAIN}, id="plain-table"),
    ],
)
def test_a_provenance_pin_scopes_reads_to_the_members_it_names(
    pinned: IdentityBindingValue | IdentityBindingSet, scope: set[str]
) -> None:
    """Search's resubmit and the Datasets mint scope a pin alike. The index is the caller's,
    so the pin scopes reads with no release bound."""
    facets = {"provenance": pinned}
    identity = {"facets": {"provenance": pinned.model_dump(mode="json")}}

    with bound_catalogue(_CATALOGUE), pinned_provenance_scope(facets, code_index=_index()):
        assert current_source_scope() == scope
    assert provenance_pin(identity, code_index=_index()) == scope


def _member_facet(facet_id: str, provenance: str, *, dataset_id: str | None = None) -> Facet:
    return Facet(
        obs_count=3,
        earliest_date="2018",
        latest_date="2020",
        facet_id=facet_id,
        provenance_id=provenance,
        unit="USDollar",
        dataset_id=dataset_id,
    )


@pytest.mark.parametrize(
    ("pinned", "refused"),
    [
        pytest.param(_LEDGER, False, id="the-dataset"),
        pytest.param(_NORTH, False, id="one-member"),
        pytest.param(_SOUTH, False, id="the-other-member"),
        pytest.param(_PLAIN, True, id="another-table"),
    ],
)
@pytest.mark.parametrize("stamped", [True, False], ids=["stamped", "as-read"])
def test_a_provenance_pin_admits_the_dataset_or_a_member(
    index: CodeIndex, pinned: str, refused: bool, stamped: bool
) -> None:
    """The provenance axis holds dataset ids, so a pin of the dataset is available whether
    or not the facets were stamped, as for a StatVar read with no elected dataset. A pin of
    a member admits that member's facet, and a pin of another table refuses
    `facet_not_available`, offering the dataset's id."""
    dataset_id = _LEDGER if stamped else None
    facets_by_sv = {
        _FLOW: [
            _member_facet("fN", _NORTH, dataset_id=dataset_id),
            _member_facet("fS", _SOUTH, dataset_id=dataset_id),
        ]
    }

    refusal = candidate_facet_refusal(
        FakeGraph(),
        facets_by_sv,
        {"provenance": IdentityBindingValue(ref=pinned)},
        deadline=None,
    )

    if not refused:
        assert refusal is None
        return
    assert refusal is not None
    assert refusal.code == "facet_not_available"
    assert [alternative.ref for alternative in refusal.alternatives or ()] == [_LEDGER]


# ---------------------------------------------------------------------------
# No answer names a member provenance.
# ---------------------------------------------------------------------------

_SOLO = "dataset/Solo"
_SOLO_MEMBER = "provenance/SoloMember"
_MEMBER_DATASETS = {_LEDGER: (_NORTH, _SOUTH), _SOLO: (_SOLO_MEMBER,)}


def _declaring_members(real: CodeIndex) -> CodeIndex:
    """`real` also declaring the Ledger of North and South and the Solo dataset of one
    member, as a release whose export read both Dataset nodes holds them."""
    index = copy.copy(real)
    index.datasets = {
        **real.datasets,
        **{
            dataset: DatasetDeclaration(
                id=dataset,
                label=f"{dataset.rsplit('/', 1)[-1]} Register",
                description="A test table.",
                opt_in=False,
                absent_observation="unknown",
                account="provider",
                rank=200 + rank,
            )
            for rank, dataset in enumerate(_MEMBER_DATASETS)
        },
    }
    index._provenances_by_dataset = {**real._provenances_by_dataset, **_MEMBER_DATASETS}
    index._dataset_by_provenance = {
        **real._dataset_by_provenance,
        **{member: dataset for dataset, members in _MEMBER_DATASETS.items() for member in members},
    }
    return index


@pytest.fixture
def member_world(monkeypatch, search_settings, bind_catalogue, request) -> Iterator[World]:
    """The answer-invariants world on `request.param`'s route, Aland's series held by one
    member and Bland's by another (`north-south`), or both by the one member of a dataset
    (`solo`), under a release declaring both datasets. No family elects a dataset."""
    route, holders = request.param
    world_module.carry_logical_clock(monkeypatch)
    search_settings(obs_preview_max_pairs=1 if route.endswith("bulk") else 120)
    for facet_id, member in (("f-north", _NORTH), ("f-south", _SOUTH), ("f-solo", _SOLO_MEMBER)):
        monkeypatch.setitem(world_module.FACETS, facet_id, (member, "CurrencyUSD"))
    years = tuple(range(2010, 2021))
    aland, bland = holders
    monkeypatch.setitem(
        world_module.ROW_SETS,
        "members",
        {ALAND: ((aland, years),), BLAND: ((bland, years),)},
    )
    ref = {
        _NORTH: _LEDGER_REF,
        _SOUTH: _LEDGER_REF,
        _SOLO_MEMBER: DatasetRef(ref=_SOLO, label="Solo Register"),
    }
    bind_catalogue(
        FakeCatalogue(
            provenances={
                **world_module.CATALOGUE.provenances,
                **{
                    member: provenance_row(
                        member, BASE, source=_PUBLISHER, name=member, dataset=dataset
                    )
                    for member, dataset in ref.items()
                },
            }
        )
    )
    release = current_release()
    views = replace(release.views, code_index=_declaring_members(release.views.code_index))
    with bound_release(replace(release, views=views)):
        yield world_module.world(route, "members", "definite")


_MEMBER_WORLDS = [
    pytest.param((route, holders), id=f"{route}-{name}")
    for route in world_module.ROUTES
    for name, holders in (("north-south", ("f-north", "f-south")), ("solo", ("f-solo", "f-solo")))
]


@pytest.mark.parametrize("member_world", _MEMBER_WORLDS, indirect=True)
def test_no_answer_names_a_member_provenance(member_world: World) -> None:
    """The first answer and its resubmit cite and serve the dataset's id: neither the
    serialised part nor its Spec names a member provenance anywhere but in a series'
    reporting organisation, and every series states
    the dataset under its declared label and the source its members share. The standard
    route's group serves both places, one from each member. A role route with no index
    family reads one facet id, as the mint does, so where each place's series has its own
    member it refuses `DatasetDisagreementError` rather than serve one member's place."""
    world = member_world
    split = world.role and world.rows[0].facet_id != world.rows[1].facet_id
    with world_module.door_loop():
        first = world.first([ALAND, BLAND], None, preview=True, unit_text=None)
        if split:
            assert first.part is None
            assert (first.error or "").startswith("DatasetDisagreementError")
            return
        (spec,) = first.specs
        again = world.resubmit(spec, preview=True)

    dataset = _SOLO if world.rows[0].facet_id == "f-solo" else _LEDGER
    for answer in (first, again):
        assert answer.part is not None, answer.error or answer.refusal
        (stated,) = answer.specs
        observations = stated.resolved.observations
        assert observations is not None
        label = f"{dataset.rsplit('/', 1)[-1]} Register"
        assert {
            (ref.ref, ref.label, ref.source.ref if ref.source else None)
            for ref in (s.facet["provenance"] for s in observations.series)
        } == {(dataset, label, _PUBLISHER)}
        if world.route == "standard":
            assert len(observations.series) == 2
        assert stated.resolved.facets["provenance"].ref == dataset
        serialised = json.dumps(
            _without_reporters(
                [answer.part.model_dump(mode="json"), stated.model_dump(mode="json")]
            )
        )
        assert dataset in serialised
        for member in (_NORTH, _SOUTH, _SOLO_MEMBER):
            assert member not in serialised


def _without_reporters(document: object) -> object:
    """`document` without its `reporting_organisation` members, the one field that names the
    member a series' points were reported under (`test_a_series_names_the_member_that_reported_it`)."""
    if isinstance(document, dict):
        return {
            key: _without_reporters(value)
            for key, value in document.items()
            if key != "reporting_organisation"
        }
    if isinstance(document, list):
        return [_without_reporters(value) for value in document]
    return document


# Each world facet's member provenance, `None` for a plain table's.
_REPORTERS = {"f-north": _NORTH, "f-south": _SOUTH, "f-solo": _SOLO_MEMBER, "f-ids": None}


@pytest.mark.parametrize(
    "member_world",
    [
        pytest.param(("standard", holders), id=name)
        for name, holders in (
            ("north-south", ("f-north", "f-south")),
            ("solo", ("f-solo", "f-solo")),
            ("plain", ("f-ids", "f-ids")),
        )
    ],
    indirect=True,
)
def test_a_series_names_the_member_that_reported_it(member_world: World) -> None:
    """On the standard route, which reads a materialised frame, each series of a declared
    dataset names the member provenance that reported it in its provenance facet, labelled
    as the catalogue lists the member, and a plain table's series names none. The resubmit
    reads the same frame and names the same members."""
    world = member_world
    with world_module.door_loop():
        first = world.first([ALAND, BLAND], None, preview=True, unit_text=None)
        assert first.part is not None, first.error or first.refusal
        (spec,) = first.specs
        again = world.resubmit(spec, preview=True)

    expected = {row.place: _REPORTERS[row.facet_id] for row in world.rows}
    for answer in (first, again):
        assert answer.part is not None, answer.error or answer.refusal
        (stated,) = answer.specs
        observations = stated.resolved.observations
        assert observations is not None
        reported = {}
        for series in observations.series:
            (place,) = [entity.ref for entity in series.entities]
            reporter = series.facet["provenance"].reporting_organisation
            reported[place] = None if reporter is None else (reporter.ref, reporter.label)
        assert reported == {
            place: None if member is None else (member, member)
            for place, member in expected.items()
        }


_ROLE_NORTH_SOUTH_WORLDS = [
    pytest.param((route, ("f-north", "f-south")), id=route)
    for route in world_module.ROUTES
    if route != "standard"
]


@pytest.mark.parametrize("member_world", _ROLE_NORTH_SOUTH_WORLDS, indirect=True)
def test_excluding_a_member_serves_the_other_alone(member_world: World) -> None:
    """A role-bearing part excluding South serves Aland's series from North alone where the
    unscoped part refuses (`test_no_answer_names_a_member_provenance`), and its identity pins
    North, so its resubmit reads North alone too. The role routes read through the coverage
    index, which admits the scope's members; the standard route's facets come from a graph
    double that admits every provenance."""
    world = member_world
    scope = compile_sources(
        [SourceMention(ref=_SOUTH, polarity="exclude")],
        catalogue=current_catalogue(),
        pointers=["/"],
    )
    assert isinstance(scope, SourceScope)
    with world_module.door_loop():
        first = world.first([ALAND, BLAND], None, preview=True, unit_text=None, source_scope=scope)
        assert first.part is not None, first.error or first.refusal
        (spec,) = first.specs
        again = world.resubmit(spec, preview=True)

    pinned = spec.definition.identity.facets or {}
    assert pinned["provenance"] == IdentityBindingValue(ref=_NORTH)
    for answer in (first, again):
        assert answer.part is not None, answer.error or answer.refusal
        (stated,) = answer.specs
        assert stated.spec_id == spec.spec_id
        observations = stated.resolved.observations
        assert observations is not None
        served = {entity.ref for s in observations.series for entity in s.entities}
        assert served & {ALAND, BLAND} == {ALAND}


@pytest.mark.parametrize(
    ("scope", "refused"),
    [
        pytest.param(None, True, id="unscoped"),
        pytest.param({_NORTH, _SOUTH, _PLAIN}, True, id="every-member"),
        pytest.param({_NORTH, _PLAIN}, True, id="one-member"),
        pytest.param({_PLAIN}, False, id="no-member"),
    ],
)
def test_a_scope_leaving_out_every_member_answers_no_data_before_disagreement(
    index: CodeIndex, scope: set[str] | None, refused: bool
) -> None:
    """The Ledger's member holds a facet of the plain table alone. With every member
    admitted, or some, that is drift, and the mint refuses `DatasetDisagreementError`; under
    a scope that leaves out every member the member is unpopulated, so the part answers its
    scope's no data."""
    catalogue = _CATALOGUE if scope is None else _CATALOGUE.scoped(frozenset(scope))
    candidates = [_member_facet("fP", _PLAIN)]

    with bound_catalogue(catalogue):
        if refused:
            with pytest.raises(DatasetDisagreementError):
                _LEDGER_DATASET.published(_FLOW, candidates)
        else:
            assert _LEDGER_DATASET.published(_FLOW, candidates) == []
