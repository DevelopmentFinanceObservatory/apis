"""Test role-bearing confirmation, open-axis discovery, and declared-role refresh."""

from __future__ import annotations

import pytest

from code_index import CodeIndex, IndexStamp, StatVarRow
from dc_graph.client import _declared_role_refresh_deadline
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import Facet
from search.engine.extract import DateRequest
from search.engine.region.grounding import _suggest_nearest_real
from search.engine.region.state import _elected_stat_vars
from search.engine.retrieval import bound_values as bound_values_module
from search.engine.retrieval.retrieve import Materialised, NoDataDraft, materialise
from search.engine.retrieval.role_resolve import graph_confirm_resolve
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.models import CoverageBare, CoverageExact, GraphRef, TimeWindow
from series_coverage.schema import RoleNames
from series_coverage.shared import CoverageSnapshot
from tests.engine._builders import bound_place
from tests.engine._fake_coverage import (
    FIXTURE_PROVENANCE,
    CoverageRow,
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph

# TESTFIXTURE/ is reserved for tests and absent from production StatVar ids.
_SV_DCID = "TESTFIXTURE/RoleBearingResolve/Flow"
_CREDITOR_DEBTOR_ROLE_NAMES = RoleNames(donor="creditor", recipient="debtor")
_YEARS = [2018, 2019, 2020]


def _patch_coverage_snapshot(
    monkeypatch,
    rows: list[CoverageRow],
    *,
    role_names: dict,
    facet_metadata: dict | None = None,
) -> CoverageSnapshot:
    """Patch coverage service bindings and return a snapshot over the same index."""
    index = patch_coverage(monkeypatch, rows, role_names=role_names, facet_metadata=facet_metadata)
    return FakeCoverageService(index).snapshot()


def _facet(obs_count: int) -> Facet:
    """Build an indexable facet with one dated point and caller-supplied summary fields."""
    return Facet(
        earliest_date="2010",
        latest_date="2020",
        obs_count=obs_count,
        dates=["2020"],
        values=[1.0],
        facet_id="f1",
    )


def _role_arcs(declared_roles: tuple[str, ...] | None) -> dict:
    """Build member arcs consistent with the shape's declared roles."""
    arcs: dict = {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if declared_roles is None or declared_roles == ("observationAbout",):
        return arcs
    arcs["observationProperties"] = {"nodes": [{"dcid": r} for r in declared_roles]}
    return arcs


def _shape(*, declared_roles: tuple[str, ...] | None) -> ShapeDraft:
    """Build a synthetic shape whose role-bearing status comes from declared_roles."""
    return ShapeDraft(
        shape_id="test_role_bearing_resolve_shape",
        label="test flow",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={_SV_DCID: _role_arcs(declared_roles)},
        slot_taxonomy={},
        declared_roles=declared_roles,
    )


def test_declared_names_key_the_facet_grouping(monkeypatch):
    """Coverage confirms the bound entities without an SDMX request."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [(_SV_DCID, "f1", "country/FRA", "country/KEN", _YEARS)],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    sdmx = FakeSdmxClient(observations={})

    result = graph_confirm_resolve(
        shape=shape,
        bindings=[],
        subject_dcids=["country/KEN"],
        origin_dcids=["country/FRA"],
        graph=FakeGraph(raise_on_call=True),
        sdmx=sdmx,
        coverage_snapshot=coverage,
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_SV_DCID]
    assert result.destination_confirmed == {_SV_DCID}
    assert sdmx.series_calls == []


def test_materialise_confirms_against_every_named_destination(monkeypatch):
    """Both named destinations confirm and contribute their distinct facet counts."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [
            (_SV_DCID, "f_ken", "country/FRA", "country/KEN", _YEARS[:2]),  # 2 years -> obs 2
            (_SV_DCID, "f_uga", "country/FRA", "country/UGA", _YEARS),  # 3 years -> obs 3
        ],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )

    result = materialise(
        shape,
        [],
        {"debtor": ["country/KEN", "country/UGA"], "creditor": ["country/FRA"]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),  # coverage supplies role facets
        coverage=coverage,
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_SV_DCID]
    assert result.destination_confirmed == {_SV_DCID}
    # Both destinations' facets reached the confirmed answer.
    confirmed_facets = result.facets_by_sv[_SV_DCID] if result.facets_by_sv else []
    assert {f.obs_count for f in confirmed_facets} == {2, 3}
    assert sum(f.obs_count for f in confirmed_facets) == 5
    # Coverage and preview both include the two named destinations.
    assert result.coverage.observation_count == 5


def test_materialise_headline_reflects_both_named_destinations(monkeypatch):
    """The headline uses Uganda's 2020 count."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [
            (_SV_DCID, "f_ken", "country/FRA", "country/KEN", _YEARS[:2]),  # last year 2019
            (_SV_DCID, "f_uga", "country/FRA", "country/UGA", _YEARS),  # last year 2020
        ],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )

    result = materialise(
        shape,
        [],
        {"debtor": ["country/KEN", "country/UGA"], "creditor": ["country/FRA"]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),  # coverage supplies role facets
        coverage=coverage,
    )

    assert isinstance(result, Materialised)
    assert result.headline is not None
    assert result.headline.year == 2020
    assert result.headline.year_named is False  # no date_request passed
    # Only Uganda's series carries an observation in 2020.
    assert result.headline.observations == 1
    assert result.headline.series == 1
    assert result.headline.role_counts == {"creditor": 1, "debtor": 1}


def test_materialise_windowed_to_a_year_with_no_data_reports_no_data(monkeypatch):
    """A 2015 request against a 2020-2024 facet returns the carrier for data outside the
    years read: no point in 2015, data at the place in 2020 to 2024, whose facets the
    carrier keeps so the answer names the span the series runs."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [(_SV_DCID, "f1", "country/FRA", "country/KEN", [2020, 2021, 2022, 2023, 2024])],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )

    result = materialise(
        shape,
        [],
        {"debtor": ["country/KEN"], "creditor": ["country/FRA"]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        date_request=DateRequest(window=TimeWindow(start_year=2015, end_year=2015), latest=False),
        coverage=coverage,
    )

    assert isinstance(result, Materialised)
    assert result.has_data is False
    assert result.available_at_place is True
    assert result.facets_by_sv is not None
    assert [(f.earliest_date, f.latest_date) for f in result.facets_by_sv[_SV_DCID]] == [
        ("2020", "2024")
    ]
    assert isinstance(result.coverage, CoverageBare)
    assert result.coverage.has_data is False


def test_materialise_windowed_to_a_year_with_data_reports_the_exact_count(monkeypatch):
    """A 2022 request reports one windowed observation from the five-point facet."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [(_SV_DCID, "f1", "country/FRA", "country/KEN", [2020, 2021, 2022, 2023, 2024])],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )

    result = materialise(
        shape,
        [],
        {"debtor": ["country/KEN"], "creditor": ["country/FRA"]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        date_request=DateRequest(window=TimeWindow(start_year=2022, end_year=2022), latest=False),
        coverage=coverage,
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_SV_DCID]
    assert isinstance(result.coverage, CoverageExact)
    assert result.coverage.observation_count == 1
    assert result.facets_by_sv is not None
    [facet] = result.facets_by_sv[_SV_DCID]
    assert facet.obs_count == 1
    assert facet.dates == ("2022",)
    assert facet.dates_complete is True


def test_materialise_windowed_count_sums_across_donor_recipient_rows_sharing_a_facet(monkeypatch):
    """Windowed counts sum both destinations' rows sharing facet_id f1."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [
            (_SV_DCID, "f1", "country/FRA", "country/KEN", [2020, 2021, 2022]),
            (_SV_DCID, "f1", "country/FRA", "country/UGA", [2021, 2022, 2023]),
        ],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )

    result = materialise(
        shape,
        [],
        {
            "debtor": ["country/KEN", "country/UGA"],
            "creditor": ["country/FRA"],
        },
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        date_request=DateRequest(window=TimeWindow(start_year=2021, end_year=2022), latest=False),
        coverage=coverage,
    )

    assert isinstance(result, Materialised)
    assert isinstance(result.coverage, CoverageExact)
    # Both rows carry 2021 and 2022, for 4 observations total.
    assert result.coverage.observation_count == 4


def test_materialise_confirms_when_only_the_second_named_origin_has_data(monkeypatch):
    """Both origins reach one coverage lookup. Only the second has data."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    origins = ["country/FRA", "country/USA"]
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [(_SV_DCID, "f1", origins[1], "country/KEN", _YEARS)],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )

    result = materialise(
        shape,
        [],
        {"debtor": ["country/KEN"], "creditor": origins},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),  # coverage supplies role facets
        coverage=coverage,
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_SV_DCID]


def test_no_origin_entity_skips_the_second_pass(monkeypatch):
    """No matching coverage returns an unavailable carrier without an SDMX request."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    coverage = _patch_coverage_snapshot(monkeypatch, [], role_names={})
    sdmx = FakeSdmxClient(observations={})

    result = graph_confirm_resolve(
        shape=shape,
        bindings=[],
        subject_dcids=["country/KEN"],
        origin_dcids=None,
        graph=FakeGraph(raise_on_call=True),
        sdmx=sdmx,
        coverage_snapshot=coverage,
    )

    assert isinstance(result, Materialised)
    assert result.available_at_place is False
    assert result.has_data is False
    assert result.sv_dcids == [_SV_DCID]
    assert sdmx.series_calls == []


def test_a_shape_declaring_no_role_is_refused_before_any_read():
    """Graph confirmation serves role-bearing shapes alone; a declaration-free shape resolves
    through `standard.standard_resolve`, so it is refused before any graph or SDMX read."""
    shape = _shape(declared_roles=None)
    sdmx = FakeSdmxClient(observations={})

    with pytest.raises(ValueError, match="declares no role"):
        graph_confirm_resolve(
            shape=shape,
            bindings=[],
            subject_dcids=["country/KEN"],
            origin_dcids=None,
            graph=FakeGraph(raise_on_call=True),
            sdmx=sdmx,
            coverage_snapshot=empty_coverage_snapshot(),
        )
    assert sdmx.series_calls == []


def test_unplaceable_declared_roles_refuse_role_incompatible():
    """An unplaceable beneficiaryOrganisation role returns role_incompatible without SDMX calls."""
    shape = _shape(declared_roles=("beneficiaryOrganisation",))
    sdmx = FakeSdmxClient(observations={})

    result = graph_confirm_resolve(
        shape=shape,
        bindings=[],
        subject_dcids=["country/KEN"],
        origin_dcids=None,
        graph=FakeGraph(raise_on_call=True),
        sdmx=sdmx,
        coverage_snapshot=empty_coverage_snapshot(),
    )

    assert isinstance(result, NoDataDraft)
    assert result.reason == "role_incompatible"
    assert sdmx.series_calls == []


def test_materialise_forks_on_the_declaration_alone():
    """Declared roles select coverage. A declaration-free shape confirms through graph facets."""
    role_bearing = _shape(declared_roles=("creditor", "debtor"))
    non_role_bearing = _shape(declared_roles=None)
    sdmx = FakeSdmxClient(observations={})

    role_result = materialise(
        role_bearing,
        [],
        {"debtor": ["country/KEN"], "creditor": ["country/FRA"]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert isinstance(role_result, Materialised)
    assert role_result.available_at_place is False

    graph = FakeGraph(
        nodes={}, obs={f"{_SV_DCID}|country/KEN": [{"obsCount": 4}]}, detect={}, resolve={}
    )
    non_role_result = materialise(
        non_role_bearing,
        [],
        {"observationAbout": ["country/KEN"]},
        graph=graph,
        sdmx=FakeSdmxClient(observations={}),
        coverage=empty_coverage_snapshot(),
    )
    assert isinstance(non_role_result, Materialised)


def test_materialise_reads_entities_by_the_declared_names():
    """Declared creditor/debtor keys resolve.

    Provider/recipient keys return variable_not_resolved.
    """
    shape = _shape(declared_roles=("creditor", "debtor"))
    role_tuple: tuple[tuple[str, str], ...] = (
        ("creditor", "country/FRA"),
        ("debtor", "country/KEN"),
    )
    observations: dict[str, dict[tuple[tuple[str, str], ...], list[Facet]]] = {
        _SV_DCID: {role_tuple: [_facet(9)]}
    }

    correctly_keyed = materialise(
        shape,
        [],
        {"debtor": ["country/KEN"], "creditor": ["country/FRA"]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations=observations),
        coverage=empty_coverage_snapshot(),
    )
    assert isinstance(correctly_keyed, Materialised)
    assert correctly_keyed.sv_dcids == [_SV_DCID]

    wrongly_keyed = materialise(
        shape,
        [],
        {"provider": ["country/FRA"], "recipient": ["country/KEN"]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations=observations),
        coverage=empty_coverage_snapshot(),
    )
    assert isinstance(wrongly_keyed, NoDataDraft)
    assert wrongly_keyed.reason == "variable_not_resolved"


# Check filtered members on the zero-data carrier.

_ROLE_PROP = "prop:purpose"
_SV_MATCH = "TESTFIXTURE/RoleBearingResolve/Match"
_SV_MISMATCH = "TESTFIXTURE/RoleBearingResolve/Mismatch"
_SV_SPECIFIC = "TESTFIXTURE/RoleBearingResolve/Specific"
_SV_GENERAL = "TESTFIXTURE/RoleBearingResolve/General"


def _flow_arcs(constraints: dict[str, str]) -> dict:
    """Build role-bearing member arcs with the supplied constraint values."""
    arcs: dict = {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": "creditor"}, {"dcid": "debtor"}]},
    }
    if constraints:
        arcs["constraintProperties"] = {"nodes": [{"dcid": prop} for prop in constraints]}
        for prop, value in constraints.items():
            arcs[prop] = {"nodes": [{"dcid": value}]}
    return arcs


def test_no_data_draft_sv_dcids_excludes_constraint_mismatched_members():
    """The zero-data carrier contains only the member matching the bound purpose."""
    shape = ShapeDraft(
        shape_id="test_role_bearing_resolve_constraint_shape",
        label="aid flow",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            # Put the mismatch first so raw dict order exposes a filtering regression.
            _SV_MISMATCH: _flow_arcs({_ROLE_PROP: "Loan"}),
            _SV_MATCH: _flow_arcs({_ROLE_PROP: "Grant"}),
        },
        slot_taxonomy={},
        declared_roles=("creditor", "debtor"),
    )
    bindings = [
        SlotBindingDraft(axis="what", property_dcid=_ROLE_PROP, kind="value", value_dcids=["Grant"])
    ]
    sdmx = FakeSdmxClient(observations={})

    result = graph_confirm_resolve(
        shape=shape,
        bindings=bindings,
        subject_dcids=["country/KEN"],
        origin_dcids=["country/FRA"],
        graph=FakeGraph(raise_on_call=True),
        sdmx=sdmx,
        coverage_snapshot=empty_coverage_snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.available_at_place is False
    assert result.has_data is False
    assert result.sv_dcids == [_SV_MATCH], (
        f"the bound-constraint filter must exclude {_SV_MISMATCH!r}; got {result.sv_dcids}"
    )
    assert _elected_stat_vars(result, shape) == [GraphRef(ref=_SV_MATCH, label=_SV_MATCH)]


_EXTRA_PROP = "prop:instrument"
_CHANNEL_PROP = "prop:channel"
_MODALITY_PROP = "prop:modality"


def test_no_data_draft_sv_dcids_preserves_specificity_order(monkeypatch):
    """The zero-data carrier represents the member with fewer unbound constraints.

    Neither member's constraint keys contain the other's, so both survive the antichain.
    """
    shape = ShapeDraft(
        shape_id="test_role_bearing_resolve_specificity_shape",
        label="aid flow",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            # Put the less specific member first to test sorted output.
            _SV_GENERAL: _flow_arcs(
                {_ROLE_PROP: "Grant", _EXTRA_PROP: "Concessional", _CHANNEL_PROP: "Bilateral"}
            ),
            _SV_SPECIFIC: _flow_arcs({_ROLE_PROP: "Grant", _MODALITY_PROP: "Project"}),
        },
        slot_taxonomy={},
        declared_roles=("creditor", "debtor"),
    )
    bindings = [
        SlotBindingDraft(axis="what", property_dcid=_ROLE_PROP, kind="value", value_dcids=["Grant"])
    ]
    coverage = _patch_coverage_snapshot(monkeypatch, [], role_names={})

    result = graph_confirm_resolve(
        shape=shape,
        bindings=bindings,
        subject_dcids=["country/KEN"],
        origin_dcids=["country/FRA"],
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage_snapshot=coverage,
    )

    assert isinstance(result, Materialised)
    assert result.available_at_place is False
    assert result.has_data is False
    assert result.sv_dcids == [_SV_SPECIFIC], (
        f"the most specific SV must be the representative; got {result.sv_dcids}"
    )
    assert _elected_stat_vars(result, shape) == [GraphRef(ref=_SV_SPECIFIC, label=_SV_SPECIFIC)]

    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [
            (_SV_SPECIFIC, "f1", "country/FRA", "country/KEN", _YEARS),
            (_SV_GENERAL, "f1", "country/FRA", "country/KEN", _YEARS),
        ],
        role_names={
            _SV_SPECIFIC: _CREDITOR_DEBTOR_ROLE_NAMES,
            _SV_GENERAL: _CREDITOR_DEBTOR_ROLE_NAMES,
        },
    )
    with_data = graph_confirm_resolve(
        shape=shape,
        bindings=bindings,
        subject_dcids=["country/KEN"],
        origin_dcids=["country/FRA"],
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage_snapshot=coverage,
    )

    assert isinstance(with_data, Materialised)
    assert with_data.has_data is True
    assert with_data.sv_dcids == [_SV_SPECIFIC, _SV_GENERAL], (
        "the surviving population keeps its specificity order, most specific first; "
        f"got {with_data.sv_dcids}"
    )
    assert with_data.population is not None
    assert with_data.population.admitted == 2
    assert with_data.population.confirmed == 2


def test_member_refining_another_over_an_unbound_constraint_is_dropped(monkeypatch):
    """Admission applies the mint's antichain: a member whose constraint keys strictly
    contain another's, the extra key read as an aggregate, gives way to it."""
    shape = ShapeDraft(
        shape_id="test_role_bearing_resolve_antichain_shape",
        label="aid flow",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            _SV_GENERAL: _flow_arcs({_ROLE_PROP: "Grant", _EXTRA_PROP: "Concessional"}),
            _SV_SPECIFIC: _flow_arcs({_ROLE_PROP: "Grant"}),
        },
        slot_taxonomy={},
        declared_roles=("creditor", "debtor"),
    )
    bindings = [
        SlotBindingDraft(axis="what", property_dcid=_ROLE_PROP, kind="value", value_dcids=["Grant"])
    ]
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [
            (_SV_SPECIFIC, "f1", "country/FRA", "country/KEN", _YEARS),
            (_SV_GENERAL, "f1", "country/FRA", "country/KEN", _YEARS),
        ],
        role_names={
            _SV_SPECIFIC: _CREDITOR_DEBTOR_ROLE_NAMES,
            _SV_GENERAL: _CREDITOR_DEBTOR_ROLE_NAMES,
        },
    )

    result = graph_confirm_resolve(
        shape=shape,
        bindings=bindings,
        subject_dcids=["country/KEN"],
        origin_dcids=["country/FRA"],
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage_snapshot=coverage,
    )

    assert isinstance(result, Materialised)
    assert result.has_data is True
    assert result.sv_dcids == [_SV_SPECIFIC]


# Count members admitted by bound constraints.

_GENDER_PROP = "prop:gender"
_SV_FEMALE = "TESTFIXTURE/RoleBearingResolve/PopulationFemale"
_SV_MALE = "TESTFIXTURE/RoleBearingResolve/PopulationMale"


def _gendered_shape() -> ShapeDraft:
    """Two role-bearing SVs sharing one five-tuple, Female and Male, each
    asserting a different value of the same constraint property."""
    return ShapeDraft(
        shape_id="test_role_bearing_resolve_admitted_shape",
        label="test population flow",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            _SV_FEMALE: _flow_arcs({_GENDER_PROP: "Female"}),
            _SV_MALE: _flow_arcs({_GENDER_PROP: "Male"}),
        },
        slot_taxonomy={},
        declared_roles=("creditor", "debtor"),
    )


def test_admitted_counts_the_bindings_admitted_population_named_destination(monkeypatch):
    """#895. binding gender=Female admits one member from the Female/Male pair."""
    shape = _gendered_shape()
    bindings = [
        SlotBindingDraft(
            axis="what", property_dcid=_GENDER_PROP, kind="value", value_dcids=["Female"]
        )
    ]
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [(_SV_FEMALE, "f1", "country/FRA", "country/KEN", _YEARS)],
        role_names={_SV_FEMALE: _CREDITOR_DEBTOR_ROLE_NAMES},
    )

    result = graph_confirm_resolve(
        shape=shape,
        bindings=bindings,
        subject_dcids=["country/KEN"],
        origin_dcids=["country/FRA"],
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage_snapshot=coverage,
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_SV_FEMALE]
    assert result.population is not None
    assert result.population.admitted == 1
    assert result.population.confirmed == 1


def test_admitted_counts_the_bindings_admitted_population_discovered_scope(monkeypatch):
    """#895. open-axis discovery also counts only the Female member admitted by the binding."""
    shape = _gendered_shape()
    bindings = [
        SlotBindingDraft(
            axis="what", property_dcid=_GENDER_PROP, kind="value", value_dcids=["Female"]
        )
    ]
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [(_SV_FEMALE, "f1", "TESTFIXTURE/RoleBearingResolve/UnboundDonor", "country/KEN", [2020])],
        role_names={_SV_FEMALE: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    role_tuple = (("debtor", "country/KEN"),)
    sdmx = FakeSdmxClient(observations={_SV_FEMALE: {role_tuple: [_facet(4)]}}, availability={})

    result = graph_confirm_resolve(
        shape=shape,
        bindings=bindings,
        subject_dcids=[],
        origin_dcids=None,
        graph=FakeGraph(raise_on_call=True),
        sdmx=sdmx,
        discover_scope=True,
        coverage_snapshot=coverage,
    )

    assert isinstance(result, Materialised)
    assert result.discovered_scope is not None
    assert result.population is not None
    assert result.population.admitted == 1
    assert result.population.confirmed == 1


# Count confirmed members in the variables dimension.


def test_named_destination_resolve_reports_the_shapes_own_member_count(monkeypatch):
    """The fixture index lacks this shape, so its one confirmed member is counted."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [(_SV_DCID, "f1", "country/FRA", "country/KEN", _YEARS)],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )

    result = graph_confirm_resolve(
        shape=shape,
        bindings=[],
        subject_dcids=["country/KEN"],
        origin_dcids=["country/FRA"],
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage_snapshot=coverage,
    )

    assert isinstance(result, Materialised)
    dims = result.coverage.dimensions
    assert dims is not None
    variables_dim = next(d for d in dims if d.label == "variables")
    assert variables_dim.count == 1


def test_no_break_collects_every_destination_confirmed_member(monkeypatch):
    """#893. retain all 3 destination-confirmed members and count all 3 as confirmed."""
    sv_a, sv_b, sv_c = (
        "TESTFIXTURE/RoleBearingResolve/PopA",
        "TESTFIXTURE/RoleBearingResolve/PopB",
        "TESTFIXTURE/RoleBearingResolve/PopC",
    )
    shape = ShapeDraft(
        shape_id="test_role_bearing_resolve_no_break_shape",
        label="test flow",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={sv: _role_arcs(("creditor", "debtor")) for sv in (sv_a, sv_b, sv_c)},
        slot_taxonomy={},
        declared_roles=("creditor", "debtor"),
    )
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [
            (sv_a, "f1", "country/FRA", "country/KEN", _YEARS),
            (sv_b, "f2", "country/FRA", "country/KEN", _YEARS),
            (sv_c, "f3", "country/FRA", "country/KEN", _YEARS),
        ],
        role_names={
            sv_a: _CREDITOR_DEBTOR_ROLE_NAMES,
            sv_b: _CREDITOR_DEBTOR_ROLE_NAMES,
            sv_c: _CREDITOR_DEBTOR_ROLE_NAMES,
        },
    )

    result = graph_confirm_resolve(
        shape=shape,
        bindings=[],
        subject_dcids=["country/KEN"],
        origin_dcids=["country/FRA"],
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage_snapshot=coverage,
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [sv_a, sv_b, sv_c]
    assert result.destination_confirmed == {sv_a, sv_b, sv_c}
    assert result.population is not None
    assert result.population.confirmed == 3
    assert result.population.confirmed == len(result.sv_dcids)


# Declared-role refresh shares the resolve deadline and reports timeouts.

_SV_INDEX = "TESTFIXTURE/RoleBearingResolve/IndexMember"

_STAMP = IndexStamp(
    graph_endpoint="test",
    exported_at="2026-01-01T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="test-hash",
)


def _index_sourced_shape() -> ShapeDraft:
    """Build a shape whose member population comes from the fixture index."""
    return ShapeDraft(
        shape_id="test_role_bearing_resolve_index_sourced_shape",
        label="test flow",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={_SV_INDEX: _role_arcs(("creditor", "debtor"))},
        sv_arc_facts_complete=False,
        slot_taxonomy={},
        declared_roles=("creditor", "debtor"),
    )


def _index_carrying_the_shape() -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=(),
        codes=(),
        statvars=[
            StatVarRow(
                dcid=_SV_INDEX,
                population_type="EconomicActivity",
                measured_property="amount",
                stat_type="measuredValue",
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=("creditor", "debtor"),
                constraint_properties={},
                provenances=(FIXTURE_PROVENANCE,),
            )
        ],
    )


class _TimingOutGraph(FakeGraph):
    """Raise TimeoutError during the declared-role refresh."""

    def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
        raise TimeoutError("simulated declared-role refresh deadline elapsed")


def test_a_refresh_that_times_out_keeps_frozen_rows_and_reports_unverified(monkeypatch) -> None:
    """A refresh timeout keeps indexed rows and marks role declarations unverified."""
    monkeypatch.setattr(bound_values_module, "get_index", _index_carrying_the_shape)
    shape = _index_sourced_shape()
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [(_SV_INDEX, "f1", "country/FRA", "country/KEN", _YEARS)],
        role_names={_SV_INDEX: _CREDITOR_DEBTOR_ROLE_NAMES},
    )

    result = graph_confirm_resolve(
        shape=shape,
        bindings=[],
        subject_dcids=["country/KEN"],
        origin_dcids=["country/FRA"],
        graph=_TimingOutGraph(nodes={}, obs={}, detect={}, resolve={}),
        sdmx=FakeSdmxClient(observations={}),
        deadline=123.0,
        coverage_snapshot=coverage,
    )

    assert isinstance(result, Materialised)
    assert result.has_data is True
    assert result.sv_dcids == [_SV_INDEX]
    assert result.role_declaration_unverified is True


class _DeadlineRecordingGraph(FakeGraph):
    """Record the deadline scope active during node_observation_properties_batch."""

    def __init__(self, *, nodes: dict) -> None:
        super().__init__(nodes=nodes, obs={}, detect={}, resolve={})
        self.observed_deadlines: list[float | None] = []

    def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
        self.observed_deadlines.append(_declared_role_refresh_deadline.get())
        return super().node_observation_properties_batch(dcids)


def test_the_refresh_opens_a_deadline_scope_matching_the_callers_own_deadline(
    monkeypatch,
) -> None:
    """Declared-role refresh receives graph_confirm_resolve's absolute deadline."""
    monkeypatch.setattr(bound_values_module, "get_index", _index_carrying_the_shape)
    shape = _index_sourced_shape()
    nodes = {
        _SV_INDEX: {
            "label": "index member",
            "type": "StatisticalVariable",
            "arcs": _role_arcs(("creditor", "debtor")),
        }
    }
    role_tuple = (("creditor", "country/FRA"), ("debtor", "country/KEN"))
    sdmx = FakeSdmxClient(observations={_SV_INDEX: {role_tuple: [_facet(3)]}})
    graph = _DeadlineRecordingGraph(nodes=nodes)

    result = graph_confirm_resolve(
        shape=shape,
        bindings=[],
        subject_dcids=["country/KEN"],
        origin_dcids=["country/FRA"],
        graph=graph,
        sdmx=sdmx,
        deadline=555.5,
        coverage_snapshot=empty_coverage_snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.role_declaration_unverified is False
    assert graph.observed_deadlines == [555.5]


# An open axis reaches the index election whatever roles the recalled shape declares.

_SV_INDEX_TOTAL = "TESTFIXTURE/RoleBearingResolve/IndexTotal"
_SV_INDEX_BY_CREDITOR = "TESTFIXTURE/RoleBearingResolve/IndexTotalByCreditor"
_DEBTOR_ONLY_ROLE_NAMES = RoleNames(donor=None, recipient="debtor")


def _index_row(dcid: str, roles: tuple[str, ...]) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="EconomicActivity",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=roles,
        constraint_properties={},
        provenances=(FIXTURE_PROVENANCE,),
    )


def _index_of(*rows: StatVarRow) -> CodeIndex:
    return CodeIndex(stamp=_STAMP, declarations=(), codes=(), statvars=list(rows))


def _recalled_total_shape() -> ShapeDraft:
    """The shape recall gives when it holds only the debtor-only total."""
    return ShapeDraft(
        shape_id="test_role_bearing_resolve_recalled_total_shape",
        label="test flow",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={_SV_INDEX_TOTAL: _role_arcs(("debtor",))},
        sv_arc_facts_complete=False,
        slot_taxonomy={},
        declared_roles=("debtor",),
    )


@pytest.mark.parametrize(
    ("rows", "elected", "open_axes"),
    [
        pytest.param(
            (
                _index_row(_SV_INDEX_TOTAL, ("debtor",)),
                _index_row(_SV_INDEX_BY_CREDITOR, ("creditor", "debtor")),
            ),
            _SV_INDEX_BY_CREDITOR,
            frozenset({"origin"}),
            id="by-creditor-twin-in-the-index",
        ),
        pytest.param(
            (_index_row(_SV_INDEX_TOTAL, ("debtor",)),),
            _SV_INDEX_TOTAL,
            frozenset(),
            id="no-member-declares-the-creditor",
        ),
    ],
)
def test_an_open_creditor_elects_from_the_index_family_beyond_the_recalled_shape(
    monkeypatch, rows, elected, open_axes
) -> None:
    """Recall holds only the debtor-only total, so the recalled shape declares no creditor
    role. The index family decides the members, and the open creditor joins its election.
    A by-creditor twin there displaces the total and the answer serves the creditor axis,
    and a family with no such twin answers the total and serves no open axis."""
    monkeypatch.setattr(bound_values_module, "get_index", lambda: _index_of(*rows))
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [
            (_SV_INDEX_TOTAL, "f_total", "", "country/KEN", _YEARS),
            (_SV_INDEX_BY_CREDITOR, "f_creditor", "country/FRA", "country/KEN", _YEARS),
        ],
        role_names={
            _SV_INDEX_TOTAL: _DEBTOR_ONLY_ROLE_NAMES,
            _SV_INDEX_BY_CREDITOR: _CREDITOR_DEBTOR_ROLE_NAMES,
        },
    )
    nodes = {
        _SV_INDEX_TOTAL: {"label": "total", "arcs": _role_arcs(("debtor",))},
        _SV_INDEX_BY_CREDITOR: {"label": "by creditor", "arcs": _role_arcs(("creditor", "debtor"))},
    }

    result = graph_confirm_resolve(
        shape=_recalled_total_shape(),
        bindings=[],
        subject_dcids=["country/KEN"],
        graph=FakeGraph(nodes=nodes, obs={}, detect={}, resolve={}),
        sdmx=FakeSdmxClient(observations={}),
        open_axes=frozenset({"origin"}),
        coverage_snapshot=coverage,
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [elected]
    assert result.open_axes == open_axes


# Nearest-real relaxation passes its deadline to the declared-role refresh.

_SV_NEAREST_REAL_INDEX = "TESTFIXTURE/RoleBearingResolve/NearestRealIndexMember"
_NEAREST_REAL_PROP = "TestNearestRealConstraintProp"
_NEAREST_REAL_VAL = "TestNearestRealConstraintValue"


def _index_sourced_role_bearing_shape_with_binding() -> ShapeDraft:
    """Add a bound constraint so nearest-real relaxation probes this shape."""
    return ShapeDraft(
        shape_id="test_role_bearing_resolve_nearest_real_index_sourced_shape",
        label="test flow",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(
            SlotKeyDraft(axis="how", property_dcid=_NEAREST_REAL_PROP, label=_NEAREST_REAL_PROP),
        ),
        sv_arc_facts={
            _SV_NEAREST_REAL_INDEX: {
                **_role_arcs(("creditor", "debtor")),
                "constraintProperties": {"nodes": [{"dcid": _NEAREST_REAL_PROP}]},
                _NEAREST_REAL_PROP: {"nodes": [{"dcid": _NEAREST_REAL_VAL}]},
            }
        },
        sv_arc_facts_complete=False,
        slot_taxonomy={},
        declared_roles=("creditor", "debtor"),
    )


def _index_carrying_the_nearest_real_shape() -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=(),
        codes=(),
        statvars=[
            StatVarRow(
                dcid=_SV_NEAREST_REAL_INDEX,
                population_type="EconomicActivity",
                measured_property="amount",
                stat_type="measuredValue",
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=("creditor", "debtor"),
                constraint_properties={_NEAREST_REAL_PROP: _NEAREST_REAL_VAL},
                provenances=(FIXTURE_PROVENANCE,),
            )
        ],
    )


_NEAREST_REAL_BINDINGS = [
    SlotBindingDraft(
        axis="how", property_dcid=_NEAREST_REAL_PROP, kind="value", value_dcids=[_NEAREST_REAL_VAL]
    )
]


def test_the_nearest_real_relaxation_probe_carries_the_caller_own_deadline(monkeypatch) -> None:
    """Nearest-real refresh uses the caller's deadline of 777.5."""
    monkeypatch.setattr(bound_values_module, "get_index", _index_carrying_the_nearest_real_shape)
    shape = _index_sourced_role_bearing_shape_with_binding()
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [(_SV_NEAREST_REAL_INDEX, "f1", "country/FRA", "country/KEN", _YEARS)],
        role_names={_SV_NEAREST_REAL_INDEX: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    sdmx = FakeSdmxClient(observations={})
    nodes = {
        "country/KEN": {"label": "Kenya", "type": "Country"},
        _SV_NEAREST_REAL_INDEX: {
            "label": "index member",
            "type": "StatisticalVariable",
            "arcs": _role_arcs(("creditor", "debtor")),
        },
    }
    graph = _DeadlineRecordingGraph(nodes=nodes)

    suggestions, warnings, _reads, _presence = _suggest_nearest_real(
        graph=graph,
        sdmx=sdmx,
        shape_draft=shape,
        bindings=_NEAREST_REAL_BINDINGS,
        place=bound_place(
            roles={"country/KEN": "observationAbout"},
            entities={"debtor": ["country/KEN"]},
            destination_dcid="country/KEN",
            origin_dcids=["country/FRA"],
        ),
        presence=None,
        date_request=None,
        deadline=777.5,
        coverage=coverage,
    )

    assert warnings == []
    assert len(suggestions) == 1
    assert graph.observed_deadlines == [777.5]
