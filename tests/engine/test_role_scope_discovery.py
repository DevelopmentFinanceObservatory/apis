"""Role-bearing open-axis discovery uses coverage rows for members and facets.
Synthetic fixtures pin filters, windows, dimensions, and no-data outcomes."""

from __future__ import annotations

import dataclasses
import functools

import pytest

from materialise.roles import axis_for_role
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import Facet
from search.engine.extract import DateRequest
from search.engine.retrieval.retrieve import (
    Materialised,
    NoDataDraft,
    UnsupportedDraft,
    materialise,
)
from search.engine.retrieval.role_resolve import graph_confirm_resolve
from search.engine.shape import ShapeDraft
from search.models import TimeWindow
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from tests.engine._fake_coverage import (
    CoverageRow,
    FakeCoverageService,
    build_coverage_index,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph
from tests.series_coverage.support import BUILD_EPOCH_YEAR

_SV = "TESTFIXTURE/RoleScopeDiscovery/Flow"
_SV_A = "TESTFIXTURE/RoleScopeDiscovery/FlowA"
_SV_B = "TESTFIXTURE/RoleScopeDiscovery/FlowB"


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


def _shape(
    *, declared_roles: tuple[str, ...] | None, sv_dcids: tuple[str, ...] = (_SV,)
) -> ShapeDraft:
    """Build a synthetic role-bearing shape."""
    return ShapeDraft(
        shape_id="test_role_scope_discovery_shape",
        label="test flow",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={dcid: _role_arcs(declared_roles) for dcid in sv_dcids},
        slot_taxonomy={},
        declared_roles=declared_roles,
    )


def _facet(obs_count: int, facet_id: str = "f1", provenance_id: str | None = None) -> Facet:
    return Facet(
        earliest_date="2010",
        latest_date="2020",
        obs_count=obs_count,
        dates=["2020"],
        values=[1.0],
        facet_id=facet_id,
        provenance_id=provenance_id,
    )


_UNBOUND_DONOR = "TESTFIXTURE/RoleScopeDiscovery/UnboundDonor"
_UNBOUND_RECIPIENT = "TESTFIXTURE/RoleScopeDiscovery/UnboundRecipient"


def _coverage_seed_for(
    role_name: str,
    members: list[str],
    svs: list[str],
    *,
    filters: dict[str, list[str]] | None = None,
    time_period: list[str] | None = None,
    facet_id: str = "f1",
) -> tuple[list[CoverageRow], dict[str, RoleNames]]:
    """Seed rows for every SV/member pair and optional opposite-role binding.

    Unbound axes use a placeholder dcid; years default to 2020. Facet metadata filters are
    excluded because coverage rows do not carry that metadata."""
    discover_axis = axis_for_role(role_name)
    assert discover_axis is not None, f"{role_name!r} has no flow axis"
    opposite_role_name, opposite_dcids = next(iter(filters.items())) if filters else (None, None)
    opposite_dcids = opposite_dcids or [
        _UNBOUND_DONOR if discover_axis == "destination" else _UNBOUND_RECIPIENT
    ]
    years = [int(y) for y in time_period] if time_period else [2020]
    rows: list[CoverageRow] = [
        (sv, facet_id, *((opp, member) if discover_axis == "destination" else (member, opp)), years)
        for sv in svs
        for member in members
        for opp in opposite_dcids
    ]
    role_names = {
        sv: (
            RoleNames(donor=opposite_role_name or "creditor", recipient=role_name)
            if discover_axis == "destination"
            else RoleNames(donor=role_name, recipient=opposite_role_name or "debtor")
        )
        for sv in svs
    }
    return rows, role_names


def _resolve(shape: ShapeDraft, sdmx: FakeSdmxClient, **kwargs):
    monkeypatch = kwargs.pop("monkeypatch", None)
    coverage_rows = kwargs.pop("coverage_rows", None)
    coverage_role_names = kwargs.pop("coverage_role_names", None)
    coverage_facet_metadata = kwargs.pop("coverage_facet_metadata", None)
    coverage_snapshot = empty_coverage_snapshot()
    if coverage_rows is not None:
        assert monkeypatch is not None, "coverage_rows needs monkeypatch to seed it"
        _coverage_index = patch_coverage(
            monkeypatch,
            coverage_rows,
            role_names=coverage_role_names or {},
            facet_metadata=coverage_facet_metadata,
        )
        coverage_snapshot = FakeCoverageService(_coverage_index).snapshot()
    return graph_confirm_resolve(
        shape=shape,
        bindings=kwargs.pop("bindings", []),
        subject_dcids=kwargs.pop("subject_dcids", []),
        origin_dcids=kwargs.pop("origin_dcids", None),
        graph=FakeGraph(raise_on_call=True),
        sdmx=sdmx,
        discover_scope=True,
        coverage_snapshot=coverage_snapshot,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------


def _read_years(result: Materialised) -> list[str] | None:
    """The years the plan reads a range over, as a year list; None for `latest` and
    dateless reads."""
    years = result.read_plan.years if result.read_plan is not None else None
    if not isinstance(years, TimeWindow):
        return None
    return [str(y) for y in range(years.start_year, years.end_year + 1)]


def test_discover_scope_off_by_default_still_variable_not_resolved():
    shape = _shape(declared_roles=("creditor", "debtor"))
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = graph_confirm_resolve(
        shape=shape,
        bindings=[],
        subject_dcids=[],
        origin_dcids=None,
        graph=FakeGraph(raise_on_call=True),
        sdmx=sdmx,
        coverage_snapshot=empty_coverage_snapshot(),
    )

    assert isinstance(result, NoDataDraft)
    assert result.reason == "variable_not_resolved"
    assert sdmx.availability_calls == []


def test_neither_axis_bound_discovers_destination_with_no_opposite_filter(monkeypatch):
    shape = _shape(declared_roles=("creditor", "debtor"))
    rows, role_names = _coverage_seed_for("debtor", ["country/KEN", "country/UGA"], [_SV])
    observations = {_SV: {(("debtor", "country/KEN"),): [_facet(4)]}}
    sdmx = FakeSdmxClient(observations=observations, availability={})

    result = _resolve(
        shape, sdmx, monkeypatch=monkeypatch, coverage_rows=rows, coverage_role_names=role_names
    )

    assert isinstance(result, Materialised)
    assert result.discovered_scope is not None
    assert result.discovered_scope.role_name == "debtor"
    assert result.discovered_scope.filters == {}
    # No availability_values call exists on this path any more (a local
    # coverage lookup, not a network read) -- see role_scope.py's own
    # docstring.
    assert sdmx.availability_calls == []


def test_discover_role_scope_headline_reflects_open_axis_population(monkeypatch):
    """Headline counts both discovered members in 2020, including the unbound donor role."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    rows, role_names = _coverage_seed_for("debtor", ["country/KEN", "country/UGA"], [_SV])
    observations = {_SV: {(("debtor", "country/KEN"),): [_facet(4)]}}
    sdmx = FakeSdmxClient(observations=observations, availability={})

    result = _resolve(
        shape, sdmx, monkeypatch=monkeypatch, coverage_rows=rows, coverage_role_names=role_names
    )

    assert isinstance(result, Materialised)
    assert result.headline is not None
    assert result.headline.year == 2020
    assert result.headline.year_named is False  # no date_request passed
    assert result.headline.observations == 2
    assert result.headline.series == 2
    assert result.headline.role_counts == {"creditor": 1, "debtor": 2}


def test_origin_bound_carries_the_opposite_role_filter(monkeypatch):
    shape = _shape(declared_roles=("creditor", "debtor"))
    filters = {"creditor": ["country/FRA"]}
    rows, role_names = _coverage_seed_for("debtor", ["country/KEN"], [_SV], filters=filters)
    observations = {_SV: {(("creditor", "country/FRA"), ("debtor", "country/KEN")): [_facet(4)]}}
    sdmx = FakeSdmxClient(observations=observations, availability={})

    result = _resolve(
        shape,
        sdmx,
        origin_dcids=["country/FRA"],
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )

    assert isinstance(result, Materialised)
    # filters is still recorded on DiscoveredScope for echo (see that
    # field's own docstring); coverage's own bound narrowing is asserted
    # via the discovered members/confirmed svs below instead of a captured
    # SDMX call, since there is no more SDMX call on this path.
    assert result.discovered_scope.filters == filters
    assert result.discovered_scope.member_dcids == ["country/KEN"]
    assert sdmx.availability_calls == []


def test_multiple_origins_carry_the_full_opposite_role_filter(monkeypatch):
    """All bound origins filter coverage, regardless of order."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    origins = ["country/FRA", "country/USA"]
    filters = {"creditor": origins}
    rows, role_names = _coverage_seed_for("debtor", ["country/KEN"], [_SV], filters=filters)
    observations = {_SV: {(("creditor", origins[1]), ("debtor", "country/KEN")): [_facet(4)]}}
    sdmx = FakeSdmxClient(observations=observations, availability={})
    _auto_coverage_index_1 = patch_coverage(monkeypatch, rows, role_names=role_names)

    result = materialise(
        shape,
        [],
        {"creditor": origins},
        graph=FakeGraph(raise_on_call=True),
        sdmx=sdmx,
        discover_scope=True,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.discovered_scope is not None
    assert result.discovered_scope.filters == filters
    assert result.discovered_scope.member_dcids == ["country/KEN"]
    assert sdmx.availability_calls == []


def test_origin_only_signature_without_discovery_stays_role_incompatible():
    """An origin-only shape (no destination role at all) refuses through both
    entry points when discover_scope is left at its default. A direct call
    (region/grounding.py's own calling shape) names a subject_dcids entity
    the way _suggest_nearest_real always does; role_incompatible is the
    outcome once a non-empty subject_dcids reaches a shape with no
    destination role to bind it under."""
    shape = _shape(declared_roles=("financingProvider",))

    direct = graph_confirm_resolve(
        shape=shape,
        bindings=[],
        subject_dcids=["country/KEN"],
        origin_dcids=None,
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage_snapshot=empty_coverage_snapshot(),
    )
    assert isinstance(direct, NoDataDraft)
    assert direct.reason == "role_incompatible"

    via_materialise = materialise(
        shape,
        [],
        {},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )
    assert isinstance(via_materialise, NoDataDraft)
    assert via_materialise.reason == "role_not_resolved"


def _provider_only_seed(
    members: list[str], svs: list[str]
) -> tuple[list[CoverageRow], dict[str, RoleNames]]:
    """Rows of a variable declaring `financingProvider` alone: each member in `donor` and
    the recipient empty, as the coverage build stores a one-role origin-axis variable."""
    rows: list[CoverageRow] = [
        (sv, "f1", member, ABSENT_ENTITY, [2020]) for sv in svs for member in members
    ]
    return rows, {sv: RoleNames(donor="financingProvider", recipient=None) for sv in svs}


@pytest.mark.parametrize(
    "seed",
    [
        functools.partial(_coverage_seed_for, "financingProvider"),
        _provider_only_seed,
    ],
    ids=["two-role-coverage", "provider-only-coverage"],
)
def test_origin_only_signature_discovers_the_origin_axis(monkeypatch, seed):
    shape = _shape(declared_roles=("financingProvider",))
    rows, role_names = seed(["country/DEU", "country/FRA"], [_SV])
    observations = {_SV: {(("financingProvider", "country/FRA"),): [_facet(5)]}}
    _auto_coverage_index_1 = patch_coverage(monkeypatch, rows, role_names=role_names)

    direct = graph_confirm_resolve(
        shape=shape,
        bindings=[],
        subject_dcids=[],
        origin_dcids=None,
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        discover_scope=True,
        coverage_snapshot=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert isinstance(direct, Materialised)
    assert direct.discovered_scope.role_name == "financingProvider"
    assert direct.discovered_scope.axis == "origin"
    assert direct.discovered_scope.member_dcids == ["country/DEU", "country/FRA"]

    via_materialise = materialise(
        shape,
        [],
        {},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        discover_scope=True,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert isinstance(via_materialise, Materialised)


@pytest.mark.parametrize(
    "seed",
    [
        functools.partial(_coverage_seed_for, "financingProvider"),
        _provider_only_seed,
    ],
    ids=["two-role-coverage", "provider-only-coverage"],
)
@pytest.mark.parametrize("discover_scope", [False, True])
def test_origin_only_signature_confirms_at_its_bound_origin(seed, discover_scope):
    """An origin-only shape whose origin carries a bound entity is confirmed there, its
    one role read from its axis: the member holding coverage at the bound provider answers,
    with no destination to check. No SDMX call is made."""
    shape = _shape(declared_roles=("financingProvider",))
    sdmx = FakeSdmxClient(observations={}, availability={})
    rows, role_names = seed(["country/FRA"], [_SV])

    result = graph_confirm_resolve(
        shape=shape,
        bindings=[],
        subject_dcids=[],
        origin_dcids=["country/FRA"],
        graph=FakeGraph(raise_on_call=True),
        sdmx=sdmx,
        discover_scope=discover_scope,
        coverage_snapshot=FakeCoverageService(
            build_coverage_index(rows, role_names=role_names)
        ).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.has_data
    assert result.sv_dcids == [_SV]
    assert result.discovered_scope is None
    assert result.destination_confirmed is None
    assert sdmx.availability_calls == []


def _constrained(arcs: dict, flow_type: str) -> dict:
    return {
        **arcs,
        "constraintProperties": {"nodes": [{"dcid": "flowType"}]},
        "flowType": {"nodes": [{"dcid": flow_type}]},
    }


@pytest.mark.parametrize("discover_scope", [False, True])
def test_a_bound_origin_confirms_the_donor_total_beside_an_optional_recipient(discover_scope):
    """A shape holds a donor-only total and, at another constraint value, a flow that also
    declares a recipient, so its recipient role is optional. A request binding the donor
    alone elects the total, whose anchor axis is origin, and confirms it at the bound donor
    whether or not an open axis may be discovered, as a relaxed suggestion's read asks."""
    provider, pair = ("financingProvider",), ("financingProvider", "financingRecipient")
    shape = dataclasses.replace(
        _shape(declared_roles=pair),
        sv_arc_facts={
            _SV_A: _constrained(_role_arcs(provider), "Service"),
            _SV_B: _constrained(_role_arcs(pair), "Arrears"),
        },
        optional_roles=frozenset({"financingRecipient"}),
    )
    coverage = build_coverage_index(
        [
            (_SV_A, "f1", "country/FRA", ABSENT_ENTITY, [2020]),
            (_SV_B, "f1", "country/FRA", "country/KEN", [2020]),
        ],
        role_names={
            _SV_A: RoleNames(donor="financingProvider", recipient=None),
            _SV_B: RoleNames(donor="financingProvider", recipient="financingRecipient"),
        },
    )

    result = graph_confirm_resolve(
        shape=shape,
        bindings=[],
        subject_dcids=[],
        origin_dcids=["country/FRA"],
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        discover_scope=discover_scope,
        coverage_snapshot=FakeCoverageService(coverage).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.has_data
    assert result.sv_dcids == [_SV_A]
    assert result.discovered_scope is None


def test_origin_only_signature_with_named_subject_skips_discovery():
    """A direct caller passing a non-empty subject_dcids on an origin-only
    shape does not trigger origin-axis discovery. subject_dcids names a
    destination-role entity this shape declares no destination role to
    bind, so the non-discovery path answers role_incompatible. A discovered
    scope here would silently drop the named entity and enumerate the whole
    provider axis."""
    shape = _shape(declared_roles=("financingProvider",))
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = graph_confirm_resolve(
        shape=shape,
        bindings=[],
        subject_dcids=["country/KEN"],
        origin_dcids=None,
        graph=FakeGraph(raise_on_call=True),
        sdmx=sdmx,
        discover_scope=True,
        coverage_snapshot=empty_coverage_snapshot(),
    )

    assert isinstance(result, NoDataDraft)
    assert result.reason == "role_incompatible"
    assert sdmx.availability_calls == []


# ---------------------------------------------------------------------------
# Time filter
# ---------------------------------------------------------------------------


def test_dateless_request_carries_no_time_filter(monkeypatch):
    shape = _shape(declared_roles=("creditor", "debtor"))
    rows, role_names = _coverage_seed_for("debtor", ["country/KEN"], [_SV])
    observations = {_SV: {(("debtor", "country/KEN"),): [_facet(3)]}}
    sdmx = FakeSdmxClient(observations=observations, availability={})

    result = _resolve(
        shape,
        sdmx,
        date_request=None,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )

    assert isinstance(result, Materialised)
    assert _read_years(result) is None


def test_exact_year_becomes_a_single_element_time_filter(monkeypatch):
    shape = _shape(declared_roles=("creditor", "debtor"))
    date_request = DateRequest(window=TimeWindow(start_year=2020, end_year=2020), latest=False)
    rows, role_names = _coverage_seed_for("debtor", ["country/KEN"], [_SV], time_period=["2020"])
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _resolve(
        shape,
        sdmx,
        date_request=date_request,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )

    assert isinstance(result, Materialised)
    assert _read_years(result) == ["2020"]
    assert result.coverage.window == TimeWindow(start_year=2020, end_year=2020)


def test_bounded_window_becomes_an_explicit_year_list(monkeypatch):
    shape = _shape(declared_roles=("creditor", "debtor"))
    date_request = DateRequest(window=TimeWindow(start_year=2018, end_year=2020), latest=False)
    rows, role_names = _coverage_seed_for(
        "debtor", ["country/KEN"], [_SV], time_period=["2018", "2019", "2020"]
    )
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _resolve(
        shape,
        sdmx,
        date_request=date_request,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )

    assert isinstance(result, Materialised)
    assert _read_years(result) == ["2018", "2019", "2020"]
    assert result.coverage.window == TimeWindow(start_year=2018, end_year=2020)


def test_open_start_window_floors_at_the_configured_minimum_year(monkeypatch):
    """An open-started window floors at the coverage horizon's epoch,
    BUILD_EPOCH_YEAR (1960), which equals SEARCH_SCOPE_MIN_YEAR's default:
    coverage's own bitmask cannot represent a year before its epoch
    (YearHorizon.masks_from_years' own range check), so a floor below it would
    report a window start this lookup never actually confirmed. No monkeypatch
    of SEARCH_SCOPE_MIN_YEAR here; the real default stays in force."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    date_request = DateRequest(window=TimeWindow(end_year=BUILD_EPOCH_YEAR + 2), latest=False)
    time_period = [str(y) for y in range(BUILD_EPOCH_YEAR, BUILD_EPOCH_YEAR + 3)]
    rows, role_names = _coverage_seed_for("debtor", ["country/KEN"], [_SV], time_period=time_period)
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _resolve(
        shape,
        sdmx,
        date_request=date_request,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )

    assert isinstance(result, Materialised)
    assert _read_years(result) == time_period
    assert result.coverage.window == TimeWindow(
        start_year=BUILD_EPOCH_YEAR, end_year=BUILD_EPOCH_YEAR + 2
    )


def test_explicit_pre_floor_start_reports_the_floored_window(monkeypatch, search_settings):
    """An explicit (not open-started) start below SEARCH_SCOPE_MIN_YEAR floors
    the read the same as an open start does. The reported coverage window
    matches the read actually applied, not the requested one: a member
    whose only data predates the floor would otherwise look silently
    excluded rather than out of range. SEARCH_SCOPE_MIN_YEAR is set to 2015 so
    the floor sits above the requested start."""
    search_settings(scope_min_year=2015)
    shape = _shape(declared_roles=("creditor", "debtor"))
    date_request = DateRequest(window=TimeWindow(start_year=2005, end_year=2020), latest=False)
    time_period = [str(y) for y in range(2015, 2021)]
    rows, role_names = _coverage_seed_for("debtor", ["country/KEN"], [_SV], time_period=time_period)
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _resolve(
        shape,
        sdmx,
        date_request=date_request,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )

    assert isinstance(result, Materialised)
    assert _read_years(result) == time_period
    assert result.coverage.window == TimeWindow(start_year=2015, end_year=2020)


def test_open_end_window_caps_at_the_current_year(monkeypatch):
    """PINNED_DATE (tests.fixtures.PINNED_DATE, 2026-06-23) governs
    search.engine.extract.date.today() for every test in this package, so an
    open-ended window caps at 2026."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    date_request = DateRequest(window=TimeWindow(start_year=2024), latest=False)
    rows, role_names = _coverage_seed_for(
        "debtor", ["country/KEN"], [_SV], time_period=["2024", "2025", "2026"]
    )
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _resolve(
        shape,
        sdmx,
        date_request=date_request,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )

    assert isinstance(result, Materialised)
    assert _read_years(result) == ["2024", "2025", "2026"]
    assert result.coverage.window == TimeWindow(start_year=2024, end_year=2026)


def test_latest_request_carries_no_time_filter_but_resolves_global_coverage_window(monkeypatch):
    shape = _shape(declared_roles=("creditor", "debtor"))
    date_request = DateRequest(window=None, latest=True)
    rows, role_names = _coverage_seed_for("debtor", ["country/KEN"], [_SV])
    observations = {_SV: {(("debtor", "country/KEN"),): [_facet(1)]}}
    sdmx = FakeSdmxClient(observations=observations, availability={})

    result = _resolve(
        shape,
        sdmx,
        date_request=date_request,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )

    assert isinstance(result, Materialised)
    assert _read_years(result) is None
    assert result.coverage.window == TimeWindow(start_year=2020, end_year=2020)


def test_window_too_long_refuses_before_any_read(search_settings):
    search_settings(scope_max_window_years=2)
    shape = _shape(declared_roles=("creditor", "debtor"))
    date_request = DateRequest(window=TimeWindow(start_year=2020, end_year=2026), latest=False)
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _resolve(shape, sdmx, date_request=date_request)

    assert isinstance(result, UnsupportedDraft)
    assert result.code == "scope_too_broad"
    assert result.pointer == "/definition/identity/entities/destination"
    assert "year" in result.detail
    assert sdmx.availability_calls == []


# ---------------------------------------------------------------------------
# Facet-axis filters
# ---------------------------------------------------------------------------


def test_unit_and_source_bound_facets_reach_the_envelope(monkeypatch):
    """A bound facet-axis (source/unit) value is recorded on
    DiscoveredScope.filters for echo, but coverage carries no per-row facet
    metadata to narrow by (see role_scope.py's own docstring), so it never
    reaches the coverage lookup itself -- unlike the opposite-role binding,
    which does (covered by the origin-bound test above). The coverage seed
    below is unbound on the opposite role, matching that."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    bindings = [
        SlotBindingDraft(
            axis="source", property_dcid="provenance", kind="value", value_dcids=["source/OECD"]
        ),
        SlotBindingDraft(axis="what", property_dcid="unit", kind="value", value_dcids=["USDollar"]),
    ]
    filters = {"provenance": ["source/OECD"], "unit": ["USDollar"]}
    rows, role_names = _coverage_seed_for("debtor", ["country/KEN"], [_SV])
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _resolve(
        shape,
        sdmx,
        bindings=bindings,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
    )

    assert isinstance(result, Materialised)
    assert result.discovered_scope.filters == filters
    assert sdmx.availability_calls == []


# ---------------------------------------------------------------------------
# No-data outcomes
# ---------------------------------------------------------------------------


def test_window_entirely_before_the_floor_reports_no_observations():
    """A window that ends before SEARCH_SCOPE_MIN_YEAR floors its start past its own end,
    so `read_plan.role_years` raises `NoYearsToRead`; discovery reports `no_observations`
    before any read, never `role_incompatible`."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    date_request = DateRequest(window=TimeWindow(start_year=1950, end_year=1950), latest=False)
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _resolve(shape, sdmx, date_request=date_request)

    assert isinstance(result, NoDataDraft)
    assert result.reason == "no_observations"
    assert result.sv_dcids == [_SV]
    assert sdmx.availability_calls == []


def test_window_entirely_after_the_current_year_reports_no_observations():
    """PINNED_DATE (2026-06-23) caps an open end at 2026; a start after that
    narrows to no year at all, the same no_observations outcome as a window
    entirely before the floor."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    date_request = DateRequest(window=TimeWindow(start_year=2030), latest=False)
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _resolve(shape, sdmx, date_request=date_request)

    assert isinstance(result, NoDataDraft)
    assert result.reason == "no_observations"
    assert result.sv_dcids == [_SV]
    assert sdmx.availability_calls == []


def test_empty_members_reports_no_observations():
    """No coverage row exists for this synthetic sv at all (no patch_coverage
    call, so the process-wide test artifact tests/conftest.py points at --
    which carries no TESTFIXTURE rows -- answers the lookup): open_axis
    returns no members, producing the no_observations outcome."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _resolve(shape, sdmx)

    assert isinstance(result, NoDataDraft)
    assert result.reason == "no_observations"
    assert result.sv_dcids == [_SV]


def test_empty_statvars_reports_no_observations():
    """No coverage row exists for this synthetic sv either, so both
    open_axis (members) and filter_candidates (confirmed svs) come back
    empty -- the `not members or not confirmed_svs` no-data branch fires on
    the members check first, same outcome as the empty-members test above."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _resolve(shape, sdmx)

    assert isinstance(result, NoDataDraft)
    assert result.reason == "no_observations"
    assert result.sv_dcids == [_SV]


# ---------------------------------------------------------------------------
# Cap trips
# ---------------------------------------------------------------------------


def test_role_index_by_sv_covers_every_confirmed_sv(monkeypatch):
    """role_index_by_sv is built directly from confirmed_coverage now, so it
    always carries an entry for every confirmed_sv -- never partial the way
    an SDMX role_facet_index read could come back."""
    shape = _shape(declared_roles=("creditor", "debtor"), sv_dcids=(_SV_A, _SV_B))
    rows, role_names = _coverage_seed_for("debtor", ["country/KEN"], [_SV_A, _SV_B])
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _resolve(
        shape, sdmx, monkeypatch=monkeypatch, coverage_rows=rows, coverage_role_names=role_names
    )

    assert isinstance(result, Materialised)
    assert result.role_index_by_sv is not None
    assert _SV_A in result.role_index_by_sv
    assert _SV_B in result.role_index_by_sv
    assert [c.facet_id for c in result.role_index_by_sv[_SV_A]] == ["f1"]
    assert [c.facet_id for c in result.role_index_by_sv[_SV_B]] == ["f1"]


# ---------------------------------------------------------------------------
# Confirmed StatVars, sources, member ordering, and the shared envelope
# ---------------------------------------------------------------------------


def test_confirmed_svs_keep_surviving_svs_order(monkeypatch):
    shape = _shape(declared_roles=("creditor", "debtor"), sv_dcids=(_SV_A, _SV_B))
    rows, role_names = _coverage_seed_for("debtor", ["country/KEN"], [_SV_B, _SV_A])
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _resolve(
        shape, sdmx, monkeypatch=monkeypatch, coverage_rows=rows, coverage_role_names=role_names
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_SV_A, _SV_B]


def test_sources_dimension_counts_the_served_facet_and_the_role_index_keeps_every_source(
    monkeypatch,
):
    """The role index keeps every candidate's provenance, read off
    CoverageIndex.facet_metadata, while the sources dimension counts only the facet the
    preview serves: two facets with two provenance ids give two indexed sources and one
    counted source."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    rows: list[CoverageRow] = [
        (_SV, "fA", _UNBOUND_DONOR, "country/KEN", [2020]),
        (_SV, "fB", _UNBOUND_DONOR, "country/KEN", [2020]),
    ]
    role_names = {_SV: RoleNames(donor="creditor", recipient="debtor")}
    facet_metadata = {
        "fA": FacetMetadata(provenance_id="source/A"),
        "fB": FacetMetadata(provenance_id="source/B"),
    }
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _resolve(
        shape,
        sdmx,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
        coverage_facet_metadata=facet_metadata,
    )

    assert isinstance(result, Materialised)
    assert result.coverage.dimensions[1].count == 1
    assert {c.provenance_id for c in result.role_index_by_sv[_SV]} == {"source/A", "source/B"}


def test_members_are_sorted_by_dcid(monkeypatch):
    shape = _shape(declared_roles=("creditor", "debtor"))
    rows, role_names = _coverage_seed_for(
        "debtor", ["country/UGA", "country/FRA", "country/KEN"], [_SV]
    )
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _resolve(
        shape, sdmx, monkeypatch=monkeypatch, coverage_rows=rows, coverage_role_names=role_names
    )

    assert isinstance(result, Materialised)
    assert result.discovered_scope.member_dcids == ["country/FRA", "country/KEN", "country/UGA"]
    assert result.scope_member_dcids == result.discovered_scope.member_dcids


# ---------------------------------------------------------------------------
# discover_role_scope(): the `variables` coverage dimension
# (role-bearing discovered-scope path)
# ---------------------------------------------------------------------------


def test_discovered_scope_reports_the_shapes_own_member_count(monkeypatch):
    """This shape's five-tuple is a self-authored test fixture, absent from
    the fixture CRS index, so member_facts falls back to
    shape.sv_arc_facts unchanged. The resolved population is exactly this
    shape's own one confirmed member, alongside the open-axis (destination)
    and sources dimensions discover_role_scope already builds."""
    shape = _shape(declared_roles=("creditor", "debtor"))
    rows, role_names = _coverage_seed_for("debtor", ["country/KEN", "country/UGA"], [_SV])
    observations = {_SV: {(("debtor", "country/KEN"),): [_facet(4)]}}
    sdmx = FakeSdmxClient(observations=observations, availability={})

    result = _resolve(
        shape, sdmx, monkeypatch=monkeypatch, coverage_rows=rows, coverage_role_names=role_names
    )

    assert isinstance(result, Materialised)
    dims = result.coverage.dimensions
    assert dims is not None
    variables_dim = next(d for d in dims if d.label == "variables")
    assert variables_dim.count == 1


def test_discovered_scope_counts_confirmed_members_not_the_declared_population(monkeypatch):
    """Two candidate StatVars survive into this call (member_facts carries
    both), but coverage confirms only one of them (no row exists for
    _SV_B at all), the same narrowing standard.py/role_resolve.py apply to
    their own `variables` dimension. The dimension must count the narrower,
    confirmed set (Materialised.sv_dcids, also asserted here) rather than
    member_facts' pre-narrowing declared population of two."""
    shape = _shape(declared_roles=("creditor", "debtor"), sv_dcids=(_SV_A, _SV_B))
    # Coverage seeded for _SV_A only: filter_candidates confirms it and
    # narrows _SV_B out.
    rows, role_names = _coverage_seed_for("debtor", ["country/KEN"], [_SV_A])
    observations = {_SV_A: {(("debtor", "country/KEN"),): [_facet(4)]}}
    sdmx = FakeSdmxClient(observations=observations, availability={})

    result = _resolve(
        shape, sdmx, monkeypatch=monkeypatch, coverage_rows=rows, coverage_role_names=role_names
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_SV_A]
    dims = result.coverage.dimensions
    assert dims is not None
    variables_dim = next(d for d in dims if d.label == "variables")
    assert variables_dim.count == 1
