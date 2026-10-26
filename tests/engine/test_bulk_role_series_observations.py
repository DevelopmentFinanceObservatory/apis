"""Tests for role-bearing bulk observations when the role-pair count exceeds the preview cap.
Coverage selects facets. SDMX fetches points for the selected facet. Tests cover truncation,
windowing, ranking, filtering, failures, and the bound-unit pipeline path.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from dc_graph import GraphReadError
from dc_sdmx import SdmxCapExceededError
from materialise.facets import FacetSelection, UnitRequest
from search.engine.clients.graph import Facet
from search.engine.extract import DateRequest
from search.engine.quantity_read import QuantityOutput
from search.engine.region import RegionResult, resolve_variable
from search.engine.region.codes import (
    DEFLATOR_MISSING_OMITTED,
    OBSERVATIONS_BULK_FETCH_FAILED,
    OBSERVATIONS_CLIENTS_UNAVAILABLE,
    OBSERVATIONS_NO_DISPATCHABLE_FACET,
    OBSERVATIONS_NO_MATCHING_SOURCE,
    OBSERVATIONS_NO_POINTS,
    OBSERVATIONS_PROBE_FAILED,
    OBSERVATIONS_SERIES_TRUNCATED,
    OBSERVATIONS_TIMED_OUT,
)
from search.engine.region.facets import facet_admitted
from search.engine.region.observations import bulk_role_series
from search.engine.region.observations import fetch as region_fetch
from search.engine.region.observations.bulk_conversion import ConvertedPairs
from search.engine.region.observations.bulk_role_series import (
    _narrowest_common_role,
    _preview_pairs,
)
from search.engine.region.result import FacetAxisFilter
from search.engine.region.state import EngineOptions
from search.engine.retrieval.facet_election import (
    merge_role_candidates,
    rank_role_candidates,
    role_n_obs,
)
from search.engine.retrieval.read_plan import BulkPlan, PlanRoles
from search.models import IdentityUnitRequest, TimeWindow
from search.observations import (
    OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED,
    OBSERVATIONS_PREVIEW_TOO_LARGE,
    observation_settings,
)
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import (
    FIXTURE_PROVENANCE,
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    build_materialise_double,
    build_observations_planned,
    bulk_plan,
    bulk_role_series_planned,
    door_roles_for_test_query,
    lead_selected,
    sdmx_double,
    unit_check_planned,
)
from tests.engine._quantity import NO_QUANTITY
from tests.engine._role_candidates import role_candidate
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise.conftest import FakeGraphReader
from tests.series_coverage.support import BUILD_EPOCH_YEAR

_SV = "TEST/BulkRoleObs_Flow"
_SV_B = "TEST/BulkRoleObs_FlowB"
_DONOR = "country/BRDonor"
# One recipient over the cap reproduces a large region-bound destination set.
_RECIPIENTS = [f"country/BR{i}" for i in range(1, observation_settings().obs_preview_max_pairs + 2)]
_NODES: dict = {
    _SV: {
        "label": "Bulk Role Flow",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Bulk Role Flow"}]},
            "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "observationProperties": {"nodes": [{"dcid": "provider"}, {"dcid": "recipient"}]},
        },
    },
    _DONOR: {"label": "Donor"},
    **{c: {"label": f"Recipient {c.rsplit('BR', 1)[-1]}"} for c in _RECIPIENTS},
}
_GRAPH = FakeGraph(nodes=_NODES, obs={})

# Coverage rows use the StatVar's provider/recipient role names.
_ROLE_NAMES = {_SV: RoleNames(donor="provider", recipient="recipient")}


def _role_tuple(recipient: str) -> tuple[tuple[str, str], ...]:
    # `role_series` keys are sorted by role name.
    return (("provider", _DONOR), ("recipient", recipient))


def _facet(
    *,
    dates: list[str],
    value: float = 0.0,
    facet_id: str = "F1",
    unit: str | None = None,
    provenance_id: str | None = None,
    measurement_method: str | None = None,
) -> Facet:
    return Facet(
        earliest_date=dates[0],
        latest_date=dates[-1],
        obs_count=len(dates),
        dates=dates,
        values=[value for _ in dates],
        facet_id=facet_id,
        unit=unit,
        provenance_id=provenance_id,
        measurement_method=measurement_method,
    )


def _obs_entities(recipients: list[str]) -> dict[str, list[str]]:
    return {"provider": [_DONOR], "recipient": list(recipients)}


def _patch_role_coverage(
    monkeypatch,
    rows: list[tuple[str, str, str, list[int]]],
    *,
    sv: str = _SV,
    facet_metadata: dict[str, FacetMetadata] | None = None,
):
    """Patch role-index rows and return their snapshot. Each row is `(facet_id, donor, recipient,
    years)`. `sv` defaults to this module's StatVar.
    """
    index = patch_coverage(
        monkeypatch,
        [(sv, facet_id, donor, recipient, years) for facet_id, donor, recipient, years in rows],
        role_names={sv: RoleNames(donor="provider", recipient="recipient")},
        facet_metadata=facet_metadata,
    )
    return FakeCoverageService(index).snapshot()


# ---------------------------------------------------------------------------
# (1) overflow delivers a truncated series
# ---------------------------------------------------------------------------


def test_role_bearing_overflow_delivers_series_instead_of_probe_failed(monkeypatch):
    coverage = _patch_role_coverage(monkeypatch, [("F1", _DONOR, r, [2020]) for r in _RECIPIENTS])
    sdmx = FakeSdmxClient(
        observations={_SV: {_role_tuple(r): [_facet(dates=["2020"])] for r in _RECIPIENTS}}
    )
    observations, warn = build_observations_planned(
        [_SV],
        _obs_entities(_RECIPIENTS),
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        sdmx=sdmx,
        coverage=coverage,
    )
    assert observations is not None
    assert len(observations.series) > 0
    assert warn is None or warn.code != OBSERVATIONS_PROBE_FAILED
    # The served facet's coverage metadata names its provenance.
    assert observations.series[0].facet["provenance"].ref == FIXTURE_PROVENANCE
    # Every confirmed recipient contributes its 2020 point to `total_obs`.
    assert observations.total_obs == len(_RECIPIENTS)
    # The overflow branch dispatches exactly one narrowed series read,
    # regardless of recipient count -- no LATEST-scoped SDMX call precedes
    # it any more, since the role index came from coverage.
    assert len(sdmx.series_calls) == 1


def test_role_bearing_overflow_windowed_takes_the_bulk_path(monkeypatch):
    """A resolved window limits both coverage selection and the SDMX read to its years."""
    dates = ["2019", "2020", "2021", "2022"]
    coverage = _patch_role_coverage(
        monkeypatch, [("F1", _DONOR, r, [int(d) for d in dates]) for r in _RECIPIENTS]
    )
    sdmx = FakeSdmxClient(
        observations={
            _SV: {
                _role_tuple(r): [_facet(dates=dates, value=float(i))]
                for i, r in enumerate(_RECIPIENTS)
            }
        }
    )
    observations, _warn = build_observations_planned(
        [_SV],
        _obs_entities(_RECIPIENTS),
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        sdmx=sdmx,
        window=TimeWindow(start_year=2020, end_year=2021),
        coverage=coverage,
    )
    assert observations.omitted is None
    assert len(observations.series) > 0
    for series in observations.series:
        assert series.points, "every emitted series should carry an in-window point"
        assert {p.date for p in series.points} <= {"2020", "2021"}
        assert series.total_obs <= 2
    assert len(sdmx.series_calls) == 1
    assert sdmx.series_calls[0]["time_period"] == ["2020", "2021"]


def test_role_bearing_overflow_window_before_epoch_year_clamps_to_it(monkeypatch):
    """The explicit year list starts at BUILD_EPOCH_YEAR (1960), the scope floor.

    Coverage cannot represent earlier years.
    """
    dates = [str(y) for y in range(BUILD_EPOCH_YEAR, BUILD_EPOCH_YEAR + 3)]
    coverage = _patch_role_coverage(
        monkeypatch, [("F1", _DONOR, r, [int(d) for d in dates]) for r in _RECIPIENTS]
    )
    sdmx = FakeSdmxClient(
        observations={
            _SV: {
                _role_tuple(r): [_facet(dates=dates, value=float(i))]
                for i, r in enumerate(_RECIPIENTS)
            }
        }
    )
    observations, _warn = build_observations_planned(
        [_SV],
        _obs_entities(_RECIPIENTS),
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        sdmx=sdmx,
        window=TimeWindow(start_year=None, end_year=BUILD_EPOCH_YEAR + 2),
        coverage=coverage,
    )
    assert observations.omitted is None
    assert len(observations.series) > 0
    assert len(sdmx.series_calls) == 1
    assert sdmx.series_calls[0]["time_period"] == dates


def test_windowed_bulk_path_matches_per_pair_path_total_obs_and_points(
    monkeypatch, search_settings
):
    """The per-pair and bulk paths return the same in-window points and `total_obs`."""
    window = TimeWindow(start_year=2020, end_year=2021)
    recipients = [f"country/BRParity{i}" for i in range(1, 4)]
    dates = ["2019", "2020", "2021", "2022"]

    def _make_sdmx() -> FakeSdmxClient:
        return FakeSdmxClient(
            observations={
                _SV: {
                    _role_tuple(r): [
                        _facet(dates=dates, value=float(i), provenance_id="TEST/RoleParitySrc")
                    ]
                    for i, r in enumerate(recipients, start=1)
                }
            }
        )

    materialise_graph, materialise_sdmx, _requests = build_materialise_double(
        {
            (_SV, _role_tuple(r)): [
                ObsFacetFixture(
                    facet_id="F1",
                    points=tuple((d, float(i)) for d in dates),
                    # A provenance id avoids the default-policy failure in the materialise fixture.
                    provenance_id="TEST/RoleParitySrc",
                )
            ]
            for i, r in enumerate(recipients, start=1)
        }
    )

    coverage = _patch_role_coverage(
        monkeypatch,
        [("F1", _DONOR, r, [int(d) for d in dates]) for r in recipients],
        facet_metadata={"F1": FacetMetadata(provenance_id="TEST/RoleParitySrc")},
    )

    per_pair_observations, per_pair_warn = build_observations_planned(
        [_SV],
        _obs_entities(recipients),
        graph=_GRAPH,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        window=window,
        coverage=coverage,
    )
    assert per_pair_warn is None
    assert len(per_pair_observations.series) == len(recipients)

    search_settings(obs_preview_max_pairs=1)
    bulk_observations, bulk_warn = build_observations_planned(
        [_SV],
        _obs_entities(recipients),
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        sdmx=_make_sdmx(),
        window=window,
        coverage=coverage,
    )
    assert bulk_warn is None
    assert len(bulk_observations.series) == len(recipients)

    def _points_by_recipient(observations) -> dict[str, list[tuple[str, float | None]]]:
        # Both paths project the "recipient" role name onto the wire's
        # "destination" EntityRoleKey via _as_entity_role_key.
        return {
            next(e.ref for e in s.entities if e.role == "destination"): sorted(
                (p.date, p.value) for p in s.points
            )
            for s in observations.series
        }

    assert per_pair_observations.total_obs == bulk_observations.total_obs
    assert _points_by_recipient(per_pair_observations) == _points_by_recipient(bulk_observations)


def test_role_bearing_overflow_without_sdmx_still_degrades():
    """A caller without an SDMX client gets the ceiling degrade."""
    observations, warn = build_observations_planned(
        [_SV],
        _obs_entities(_RECIPIENTS),
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        coverage=empty_coverage_snapshot(),
    )
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_PREVIEW_TOO_LARGE
    assert warn is not None
    assert warn.code == OBSERVATIONS_PREVIEW_TOO_LARGE


# Bound-unit requests on role-bearing bulk previews.


def test_role_bearing_overflow_unit_request_without_an_election_withholds_preview(monkeypatch):
    """A bound unit request with no unit election to convert by withholds the preview
    before any SDMX call."""
    coverage = _patch_role_coverage(monkeypatch, [("F1", _DONOR, r, [2020]) for r in _RECIPIENTS])
    sdmx = FakeSdmxClient(
        observations={_SV: {_role_tuple(r): [_facet(dates=["2020"])] for r in _RECIPIENTS}}
    )
    unit_request = IdentityUnitRequest(currency="USD", prices="current")
    observations, warn = build_observations_planned(
        [_SV],
        _obs_entities(_RECIPIENTS),
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        sdmx=sdmx,
        unit_request=unit_request,
        coverage=coverage,
    )

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_CLIENTS_UNAVAILABLE
    assert warn is None
    assert sdmx.series_calls == []

    # Without the unit request, the same fixture reaches SDMX.
    observations, warn = build_observations_planned(
        [_SV],
        _obs_entities(_RECIPIENTS),
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        coverage=coverage,
        sdmx=sdmx,
    )
    assert observations is not None
    assert len(observations.series) > 0
    assert warn is None or warn.code != OBSERVATIONS_PROBE_FAILED
    assert len(sdmx.series_calls) == 1


def _facet_selection(
    *, axis: str, ref: str | None, selection_reason: str | None, request=None
) -> FacetSelection:
    return FacetSelection(
        axis=axis,
        ref=ref,
        basis="deployment_fallback",
        default_policy=None,
        default_policy_version=None,
        selection_reason=selection_reason,
        dispatch_facet_id="F1",
        request=request,
    )


def test_role_bearing_overflow_conversion_without_materialise_clients_calls_no_sdmx():
    """A selected EUR facet needs converting to answer a USD request, which the bulk path
    cannot do without the materialise clients, so it makes no SDMX call."""
    sdmx = FakeSdmxClient(
        observations={_SV: {_role_tuple(r): [_facet(dates=["2020"])] for r in _RECIPIENTS}}
    )
    unit_request = IdentityUnitRequest(currency="USD", prices="current")
    elected = {
        _SV: (
            {
                "provenance": _facet_selection(
                    axis="provenance", ref="dc/base/Src", selection_reason="sole_candidate"
                ),
                "unit": _facet_selection(
                    axis="unit",
                    ref="CurrencyEUR",
                    selection_reason="most_observations",
                    request=UnitRequest(currency="USD", prices="current"),
                ),
            },
        )
    }
    observations, warn = build_observations_planned(
        [_SV],
        _obs_entities(_RECIPIENTS),
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        sdmx=sdmx,
        unit_request=unit_request,
        elected=elected,
        coverage=empty_coverage_snapshot(),
    )

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_CLIENTS_UNAVAILABLE
    assert warn is None
    assert sdmx.series_calls == []


def test_role_bearing_overflow_unit_request_natively_satisfied_falls_through_to_bulk_path(
    monkeypatch,
):
    """A native EUR match narrows the bulk plan to CurrencyEUR before the read, so the
    USD facet that ranks first on coverage is never dispatched."""
    coverage = _patch_role_coverage(
        monkeypatch,
        [("F1", _DONOR, r, [2020]) for r in _RECIPIENTS[:-1]]
        + [("F2", _DONOR, r, [2020]) for r in _RECIPIENTS],
        facet_metadata={
            "F1": FacetMetadata(unit="CurrencyEUR"),
            "F2": FacetMetadata(unit="CurrencyUSD"),
        },
    )
    sdmx = FakeSdmxClient(
        observations={
            _SV: {
                _role_tuple(r): [
                    _facet(dates=["2020"], unit="CurrencyEUR"),
                    _facet(dates=["2020"], facet_id="F2", unit="CurrencyUSD"),
                ]
                for r in _RECIPIENTS
            }
        }
    )
    unit_request = IdentityUnitRequest(currency="EUR", prices="current")
    elected = {
        _SV: (
            {
                "provenance": _facet_selection(
                    axis="provenance", ref="dc/base/Src", selection_reason="sole_candidate"
                ),
                "unit": _facet_selection(
                    axis="unit",
                    ref="CurrencyEUR",
                    selection_reason="native_currency_preferred",
                    request=UnitRequest(currency="EUR", prices="current"),
                ),
            },
        )
    }
    captured_plans: list = []
    real_bulk_role_series = region_fetch._bulk_role_series

    def _capture(*args, **kwargs):
        captured_plans.append(args[0])
        return real_bulk_role_series(*args, **kwargs)

    monkeypatch.setattr(region_fetch, "_bulk_role_series", _capture)

    observations, warn = build_observations_planned(
        [_SV],
        _obs_entities(_RECIPIENTS),
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        sdmx=sdmx,
        unit_request=unit_request,
        elected=elected,
        coverage=coverage,
    )

    assert warn is None or warn.code != OBSERVATIONS_PROBE_FAILED
    assert observations is not None
    assert len(observations.series) > 0
    assert len(sdmx.series_calls) == 1  # the bulk path actually ran
    assert sdmx.series_calls[0]["facet_ids"] == {"F1"}
    (plan,) = captured_plans
    assert {candidate.unit for candidate in plan.candidates.values()} == {"CurrencyEUR"}


# ---------------------------------------------------------------------------
# (2) merge/rank/dispatch logic, called directly
# ---------------------------------------------------------------------------


def test_facet_ranked_by_role_tuple_coverage_not_facet_id(monkeypatch):
    recipients = [f"country/BRSmall{i}" for i in range(1, 6)]  # 5 recipients
    # ZZZLow covers 5/5 role tuples, AAAHigh only 3/5 -- coverage (the count
    # of role tuples a facet_id's own coverage rows name) must win the
    # ranking over facet_id ordering and over recency (AAAHigh's own facet
    # is more recent, per its own SDMX points below). The two hold different
    # sources, so the group ZZZLow leads holds it alone.
    coverage = _patch_role_coverage(
        monkeypatch,
        [("ZZZLow", _DONOR, r, [2019]) for r in recipients]
        + [("AAAHigh", _DONOR, r, [2020]) for r in recipients[:3]],
        facet_metadata={
            "ZZZLow": FacetMetadata(provenance_id="dc/base/CoverageFixture"),
            "AAAHigh": FacetMetadata(provenance_id="dc/base/OtherFixture"),
        },
    )
    obs: dict[tuple[tuple[str, str], ...], list[Facet]] = {}
    for i, r in enumerate(recipients):
        facets = [_facet(dates=["2019"], facet_id="ZZZLow")]
        if i < 3:
            facets.append(_facet(dates=["2020"], facet_id="AAAHigh"))
        obs[_role_tuple(r)] = facets
    sdmx = FakeSdmxClient(observations={_SV: obs})
    observations, warn = bulk_role_series_planned(
        [_SV],
        _obs_entities(recipients),
        sdmx=sdmx,
        graph=_GRAPH,
        facet_filter=None,
        coverage=coverage,
    )
    assert warn is None
    assert observations is not None
    assert sdmx.series_calls[0]["facet_ids"] == {"ZZZLow"}
    assert len(observations.series) == 5


@pytest.mark.parametrize(
    ("index_sourced", "served"),
    [
        pytest.param(True, {_SV, _SV_B}, id="index-sourced"),
        pytest.param(False, {_SV}, id="by-facet-id"),
    ],
)
def test_an_index_sourced_bulk_read_serves_each_member_its_own_method_s_facet(
    monkeypatch, index_sourced: bool, served: set[str]
):
    """`_SV` holds `FM1` and `_SV_B` `FM2`, one source, unit and cadence under two methods.
    An index-resolved family's group is that source and unit, so its one bulk read asks for
    both facets and serves each member its own, as its mint dispatches, and names no member
    left out. Grouped by facet id, the read asks for the lead's facet alone and serves its
    holder."""
    recipients = ["country/BRMethod1", "country/BRMethod2"]
    index = patch_coverage(
        monkeypatch,
        [
            *((_SV, "FM1", _DONOR, r, [2020]) for r in recipients),
            (_SV_B, "FM2", _DONOR, recipients[0], [2020]),
        ],
        role_names={sv: RoleNames(donor="provider", recipient="recipient") for sv in (_SV, _SV_B)},
        facet_metadata={
            facet_id: FacetMetadata(
                provenance_id="dc/base/Shared",
                measurement_method=method,
                unit="CurrencyUSD",
                observation_period="P1Y",
            )
            for facet_id, method in (("FM1", "Commitment"), ("FM2", "Disbursement"))
        },
    )
    sdmx = FakeSdmxClient(
        observations={
            _SV: {_role_tuple(r): [_facet(dates=["2020"], facet_id="FM1")] for r in recipients},
            _SV_B: {_role_tuple(recipients[0]): [_facet(dates=["2020"], facet_id="FM2")]},
        }
    )
    observations, _warn = bulk_role_series_planned(
        [_SV, _SV_B],
        _obs_entities(recipients),
        sdmx=sdmx,
        graph=_GRAPH,
        coverage=FakeCoverageService(index).snapshot(),
        index_sourced=index_sourced,
    )
    assert len(sdmx.series_calls) == 1
    assert sdmx.series_calls[0]["facet_ids"] == ({"FM1", "FM2"} if index_sourced else {"FM1"})
    assert observations is not None
    assert {s.stat_var.ref for s in observations.series} == served


def test_merged_candidates_keep_or_drop_whole_and_preview_pairs_in_sorted_order():
    """Preview role pairs in sorted order.
    Filter whole candidates by metadata, then rank by width.
    """
    role_tuples = [
        _role_tuple("country/BRStable3"),
        _role_tuple("country/BRStable1"),
        _role_tuple("country/BRStable2"),
    ]
    usd_wide = role_candidate(
        facet_id="F-A",
        role_tuples=frozenset(role_tuples),
        latest_date="2020",
        provenance_id="source/a",
        measurement_method="method/a",
        unit="USD",
    )
    usd_narrow = role_candidate(
        facet_id="F-B",
        role_tuples=frozenset(role_tuples[:2]),
        latest_date="2020",
        provenance_id="source/b",
        measurement_method="method/b",
        unit="USD",
    )
    eur = role_candidate(
        facet_id="F-C",
        role_tuples=frozenset(role_tuples),
        latest_date="2020",
        provenance_id="source/c",
        measurement_method="method/c",
        unit="EUR",
    )
    merged = merge_role_candidates({_SV: [usd_narrow, eur, usd_wide]}, [_SV])
    preview = _preview_pairs(
        BulkPlan(
            members=(_SV,),
            years=None,
            facet_id="F-A",
            selected=lead_selected(merged, "F-A"),
            candidates=merged,
        ),
        10,
        member_year=lambda _sv: 2020,
    )
    assert [pair for pair, _n_obs in preview] == [(_SV, rt) for rt in sorted(role_tuples)]

    usd_filter = {"unit": FacetAxisFilter(predicate=lambda facet: facet.unit == "USD")}
    narrowed = {
        facet_id: candidate
        for facet_id, candidate in merged.items()
        if facet_admitted(usd_filter, candidate)
    }
    assert narrowed["F-A"] is merged["F-A"]
    assert "F-C" not in narrowed
    assert [c.facet_id for c in rank_role_candidates(narrowed, stat_vars=[_SV], window=None)] == [
        "F-A",
        "F-B",
    ]


def test_windowed_selection_prefers_the_only_facet_overlapping_the_window():
    """A facet that reaches the requested years beats a wider one that ends before them.

    Without a window the wider facet leads on pair count. A window no facet reaches leaves
    the pool whole, so the wider facet leads again.
    """
    role_tuples = [_role_tuple(f"country/BRWin{i}") for i in range(3)]
    stale_wide = role_candidate(
        facet_id="F-STALE",
        role_tuples=frozenset(role_tuples),
        latest_date="2015",
        provenance_id="source/a",
    )
    current_narrow = role_candidate(
        facet_id="F-CURRENT",
        role_tuples=frozenset(role_tuples[:1]),
        latest_date="2022",
        provenance_id="source/b",
    )
    index = {_SV: [stale_wide, current_narrow]}

    def selected(time_period: list[str] | None) -> str:
        best = bulk_plan(
            [_SV],
            {},
            coverage=empty_coverage_snapshot(),
            discovered_role_index=index,
            read_years=time_period,
        ).elected
        assert best is not None
        return best.facet_id

    assert selected(None) == "F-STALE"
    assert selected(["2020", "2021", "2022"]) == "F-CURRENT"
    assert selected(["2030", "2031"]) == "F-STALE"


def test_facet_filter_narrows_to_zero_returns_no_matching_source(monkeypatch):
    recipients = [f"country/BRFilt{i}" for i in range(1, 4)]
    coverage = _patch_role_coverage(
        monkeypatch,
        [("F1", _DONOR, r, [2020]) for r in recipients],
        facet_metadata={"F1": FacetMetadata(unit="EUR")},
    )
    obs = {_role_tuple(r): [_facet(dates=["2020"], unit="EUR")] for r in recipients}
    sdmx = FakeSdmxClient(observations={_SV: obs})
    facet_filter = {"unit": FacetAxisFilter(predicate=lambda f: f.unit == "USD")}
    observations, warn = bulk_role_series_planned(
        [_SV],
        _obs_entities(recipients),
        sdmx=sdmx,
        graph=_GRAPH,
        facet_filter=facet_filter,
        coverage=coverage,
    )
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_NO_MATCHING_SOURCE
    assert warn is not None
    assert warn.code == OBSERVATIONS_NO_MATCHING_SOURCE
    assert sdmx.series_calls == []


def test_shared_facet_labels_every_role_pair_with_the_coverage_metadata(monkeypatch):
    """Pairs sharing a facet id receive its coverage metadata."""
    recipients = [f"country/BRMeta{i}" for i in range(1, 5)]
    coverage = _patch_role_coverage(
        monkeypatch,
        [("F1", _DONOR, r, [2020]) for r in recipients],
        facet_metadata={
            "F1": FacetMetadata(
                provenance_id="dc/base/UsdSource",
                unit="USD",
                measurement_method="dcAggregate/Usd",
            )
        },
    )
    observations_by_tuple = {
        _role_tuple(recipient): [
            _facet(
                dates=["2020"],
                value=float(index),
                unit="USD",
                provenance_id="dc/base/UsdSource",
                measurement_method="dcAggregate/Usd",
            )
        ]
        for index, recipient in enumerate(recipients)
    }
    graph = FakeGraph(
        nodes={**_NODES, "dc/base/UsdSource": {"label": "USD source"}},
        obs={},
    )
    sdmx = FakeSdmxClient(observations={_SV: observations_by_tuple})

    observations, warn = bulk_role_series_planned(
        [_SV],
        _obs_entities(recipients),
        sdmx=sdmx,
        graph=graph,
        facet_filter=None,
        coverage=coverage,
    )

    assert warn is None
    assert observations is not None
    assert len(observations.series) == len(recipients)
    for series in observations.series:
        assert series.facet["unit"].ref == "USD"
        assert series.facet["measurementMethod"].ref == "dcAggregate/Usd"
        assert series.facet["provenance"].ref == "dc/base/UsdSource"
        assert [source.ref for source in series.source_refs] == ["dc/base/UsdSource"]


def test_facet_filter_excludes_a_whole_facet_whose_coverage_metadata_disagrees(monkeypatch):
    """An EUR facet is excluded entirely when its coverage metadata fails the USD filter."""
    usd_recipients = [f"country/BRUsd{i}" for i in range(1, 3)]
    eur_recipients = [f"country/BREur{i}" for i in range(1, 3)]
    coverage = _patch_role_coverage(
        monkeypatch,
        [("F-USD", _DONOR, r, [2020]) for r in usd_recipients]
        + [("F-EUR", _DONOR, r, [2020]) for r in eur_recipients],
        facet_metadata={
            "F-USD": FacetMetadata(unit="USD", provenance_id="dc/base/UsdSource"),
            "F-EUR": FacetMetadata(unit="EUR", provenance_id="dc/base/EurSource"),
        },
    )
    observations_by_tuple = {
        **{
            _role_tuple(r): [_facet(dates=["2020"], value=1.0, unit="USD", facet_id="F-USD")]
            for r in usd_recipients
        },
        **{
            _role_tuple(r): [_facet(dates=["2020"], value=2.0, unit="EUR", facet_id="F-EUR")]
            for r in eur_recipients
        },
    }
    sdmx = FakeSdmxClient(observations={_SV: observations_by_tuple})
    usd_filter = {"unit": FacetAxisFilter(predicate=lambda f: f.unit == "USD")}

    observations, warn = bulk_role_series_planned(
        [_SV],
        _obs_entities(usd_recipients + eur_recipients),
        sdmx=sdmx,
        graph=_GRAPH,
        facet_filter=usd_filter,
        coverage=coverage,
    )

    assert warn is None
    assert observations is not None
    assert {
        next(e.ref for e in s.entities if e.role == "destination") for s in observations.series
    } == set(usd_recipients)
    assert all(s.facet["unit"].ref == "USD" for s in observations.series)


def test_overflow_facet_filter_keeps_only_the_matching_facets_role_pairs(monkeypatch):
    """Unit filtering precedes truncation, so EUR pairs cannot take display slots."""
    half = len(_RECIPIENTS) // 2
    usd_recipients = _RECIPIENTS[:half]
    eur_recipients = _RECIPIENTS[half:]
    coverage = _patch_role_coverage(
        monkeypatch,
        [("F-USD", _DONOR, r, [2020]) for r in usd_recipients]
        + [("F-EUR", _DONOR, r, [2020]) for r in eur_recipients],
        facet_metadata={
            "F-USD": FacetMetadata(unit="USD", provenance_id="dc/base/UsdSource"),
            "F-EUR": FacetMetadata(unit="EUR", provenance_id="dc/base/EurSource"),
        },
    )
    observations_by_tuple = {
        **{
            _role_tuple(r): [_facet(dates=["2020"], value=float(i), unit="USD", facet_id="F-USD")]
            for i, r in enumerate(usd_recipients)
        },
        **{
            _role_tuple(r): [_facet(dates=["2020"], value=float(i), unit="EUR", facet_id="F-EUR")]
            for i, r in enumerate(eur_recipients)
        },
    }
    graph = FakeGraph(
        nodes={**_NODES, "dc/base/UsdSource": {"label": "USD source"}},
        obs={},
    )
    sdmx = FakeSdmxClient(observations={_SV: observations_by_tuple})
    usd_filter = {"unit": FacetAxisFilter(predicate=lambda f: f.unit == "USD")}

    observations, warn = bulk_role_series_planned(
        [_SV],
        _obs_entities(_RECIPIENTS),
        sdmx=sdmx,
        graph=graph,
        facet_filter=usd_filter,
        coverage=coverage,
    )

    assert observations is not None
    assert warn is not None
    assert warn.code == OBSERVATIONS_SERIES_TRUNCATED
    # USD filtering happens before top-20 truncation.
    assert all(series.facet["unit"].ref == "USD" for series in observations.series)
    assert all(
        [source.ref for source in series.source_refs] == ["dc/base/UsdSource"]
        for series in observations.series
    )
    assert len(sdmx.series_calls) == 1


def test_series_read_failure_returns_bulk_fetch_failed(monkeypatch):
    recipients = [f"country/BRFail2_{i}" for i in range(1, 4)]
    coverage = _patch_role_coverage(monkeypatch, [("F1", _DONOR, r, [2020]) for r in recipients])
    obs = {_role_tuple(r): [_facet(dates=["2020"])] for r in recipients}
    cap_error = SdmxCapExceededError("cap tripped", cap_kind="rows", cap=10, observed=20)
    sdmx = FakeSdmxClient(observations={_SV: obs}, raise_series=cap_error)
    observations, warn = bulk_role_series_planned(
        [_SV],
        _obs_entities(recipients),
        sdmx=sdmx,
        graph=_GRAPH,
        facet_filter=None,
        coverage=coverage,
    )
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_BULK_FETCH_FAILED
    assert warn is not None
    assert warn.code == OBSERVATIONS_BULK_FETCH_FAILED


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        pytest.param(GraphReadError("transient"), OBSERVATIONS_BULK_FETCH_FAILED, id="graph"),
        pytest.param(TimeoutError("deadline"), OBSERVATIONS_TIMED_OUT, id="deadline"),
    ],
)
def test_a_converted_role_read_whose_conversion_read_fails_omits_the_preview(
    monkeypatch, exc: Exception, code: str
):
    """The bulk role read serves its pairs, then a read its conversion makes fails or
    crosses the deadline. The preview is omitted under the failure's code with a matching
    warning, as the scope read's is, and nothing raises."""
    recipients = [f"country/BRConv_{i}" for i in range(1, 4)]
    coverage = _patch_role_coverage(monkeypatch, [("F1", _DONOR, r, [2020]) for r in recipients])
    sdmx = FakeSdmxClient(
        observations={_SV: {_role_tuple(r): [_facet(dates=["2020"])] for r in recipients}}
    )

    def fail(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(bulk_role_series, "convert_pairs", fail)
    observations, warn = bulk_role_series_planned(
        [_SV],
        _obs_entities(recipients),
        sdmx=sdmx,
        graph=_GRAPH,
        facet_filter=None,
        coverage=coverage,
        # The conversion itself never runs: `convert_pairs` fails first.
        conversion=SimpleNamespace(target_unit_ref="CurrencyUSD"),
    )

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == code
    assert warn is not None
    assert warn.code == code


@pytest.mark.parametrize("omitted_count", [1, 3], ids=["one-pair", "every-pair"])
def test_a_role_conversion_names_the_pairs_whose_deflator_it_lacks(monkeypatch, omitted_count: int):
    """The conversion leaves out every row of the first `omitted_count` pairs for a missing
    deflator (`ConvertedPairs.omitted`). Those pairs hold data: the conversion's warning, or
    its omission when nothing converts, names them under `deflator_missing_omitted`, and the
    preview never reports that the read returned no rows."""
    recipients = [f"country/BRDefl_{i}" for i in range(1, 4)]
    omitted = {(_SV, _role_tuple(r)) for r in recipients[:omitted_count]}
    coverage = _patch_role_coverage(monkeypatch, [("F1", _DONOR, r, [2020]) for r in recipients])
    sdmx = FakeSdmxClient(
        observations={_SV: {_role_tuple(r): [_facet(dates=["2020"])] for r in recipients}}
    )

    def convert(pair_points, *, facet, conversion):
        return ConvertedPairs(
            points={pair: p for pair, p in pair_points.items() if pair not in omitted},
            conversions={},
            left_out={},
            omitted=dict.fromkeys(omitted, DEFLATOR_MISSING_OMITTED),
        )

    monkeypatch.setattr(bulk_role_series, "convert_pairs", convert)
    observations, warn = bulk_role_series_planned(
        [_SV],
        _obs_entities(recipients),
        sdmx=sdmx,
        graph=_GRAPH,
        facet_filter=None,
        coverage=coverage,
        conversion=SimpleNamespace(target_unit_ref=None),
    )

    assert len(observations.series) == len(recipients) - omitted_count
    if omitted_count == len(recipients):
        assert observations.omitted is not None
        assert observations.omitted.reason == OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED
    else:
        assert observations.omitted is None
    assert warn is not None
    assert warn.code == OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED
    for _sv, role_tuple in omitted:
        assert f"{_DONOR} / {role_tuple[1][1]} (deflator_missing_omitted)" in warn.message


def test_no_coverage_rows_at_all_returns_no_dispatchable_facet(monkeypatch):
    """No coverage candidate can be dispatched when no facet filter is active."""
    recipients = [f"country/BRNoCand{i}" for i in range(1, 4)]
    coverage = _patch_role_coverage(monkeypatch, [])  # no rows for _SV at all
    sdmx = FakeSdmxClient(observations={_SV: {}})
    observations, warn = bulk_role_series_planned(
        [_SV],
        _obs_entities(recipients),
        sdmx=sdmx,
        graph=_GRAPH,
        facet_filter=None,
        coverage=coverage,
    )
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_NO_DISPATCHABLE_FACET
    assert warn is not None
    assert warn.code == OBSERVATIONS_NO_DISPATCHABLE_FACET
    assert sdmx.series_calls == []


def test_narrowed_series_read_disagreeing_with_the_index_is_no_points():
    """A reused index confirms a facet, but its points fall outside the requested 1900 period.

    A fresh index build is narrowed by the period too and would find no candidate.
    """
    recipients = [f"country/BRDisagree{i}" for i in range(1, 4)]
    obs = {_role_tuple(r): [_facet(dates=["2020"])] for r in recipients}
    sdmx = FakeSdmxClient(observations={_SV: obs})
    role_tuples = frozenset(_role_tuple(r) for r in recipients)
    discovered_index = {
        _SV: [
            role_candidate(
                facet_id="F1",
                role_tuples=role_tuples,
                latest_date="2020",
            )
        ]
    }
    observations, warn = bulk_role_series_planned(
        [_SV],
        _obs_entities(recipients),
        sdmx=sdmx,
        graph=_GRAPH,
        facet_filter=None,
        discovered_role_index=discovered_index,
        time_period=["1900"],
        coverage=empty_coverage_snapshot(),
    )
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_NO_POINTS
    assert warn is None


def test_an_elected_facet_with_no_pair_in_its_years_is_no_points_unread():
    """A facet elected on a role tuple with no observation in the read's years dispatches
    no pair, so nothing is read. SDMX refuses a read with no entity filter."""
    sdmx = FakeSdmxClient(
        observations={_SV: {_role_tuple("country/BR1"): [_facet(dates=["2024"])]}}
    )
    observations, warn = bulk_role_series_planned(
        [_SV],
        _obs_entities(["country/BR1"]),
        sdmx=sdmx,
        graph=_GRAPH,
        facet_filter=None,
        discovered_role_index={
            _SV: [
                role_candidate(
                    facet_id="F1", role_tuples={_role_tuple("country/BR1"): 0}, latest_date="2024"
                )
            ]
        },
        coverage=empty_coverage_snapshot(),
    )
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_NO_POINTS
    assert warn is None
    assert sdmx.series_calls == []


# Under the row gate the read takes every series and ranks them. Over it, or converting
# into each entity's own currency, coverage chooses a fixed selection before any value.


def _identity_conversion(pair_points, *, facet, conversion):
    return ConvertedPairs(points=dict(pair_points), conversions={}, left_out={}, omitted={})


@pytest.mark.parametrize(
    ("gate", "target_unit", "order", "bound"),
    [
        (25, "absent", "ranked", {"donor": frozenset({_DONOR})}),
        (25, "absent", "ranked", {}),
        (25, "CurrencyUSD", "ranked", {"donor": frozenset({_DONOR})}),
        (24, "absent", "fixed", {"donor": frozenset({_DONOR})}),
        (25, None, "fixed", {"donor": frozenset({_DONOR})}),
    ],
    ids=[
        "ranked-at-the-gate",
        "ranked-every-role-open",
        "ranked-converted",
        "fixed-over-the-gate",
        "fixed-role-currency",
    ],
)
def test_series_truncated_to_the_order_the_row_gate_chooses(
    monkeypatch, search_settings, gate: int, target_unit: str | None, order: str, bound
):
    search_settings(role_bulk_rank_max_rows=gate)
    recipients = [f"country/BRTrunc{i}" for i in range(1, 26)]  # 25 recipients, 25 rows
    coverage = _patch_role_coverage(monkeypatch, [("F1", _DONOR, r, [2020]) for r in recipients])
    # Values rise with the recipient's position, so ranking keeps the last 20.
    obs = {
        _role_tuple(r): [_facet(dates=["2020"], value=float(i))] for i, r in enumerate(recipients)
    }
    sdmx = FakeSdmxClient(observations={_SV: obs})
    conversion = None
    if target_unit != "absent":
        monkeypatch.setattr(bulk_role_series, "convert_pairs", _identity_conversion)
        conversion = SimpleNamespace(target_unit_ref=target_unit)
    observations, warn = bulk_role_series_planned(
        [_SV],
        _obs_entities(recipients),
        sdmx=sdmx,
        graph=_GRAPH,
        facet_filter=None,
        coverage=coverage,
        conversion=conversion,
        roles=PlanRoles(bound=bound, role_names={"recipient": "recipient", "donor": "provider"}),
    )
    shown = observation_settings().role_bulk_series_max
    assert observations is not None
    assert len(observations.series) == shown
    # `total_obs` includes all 25 pairs, unshipped pairs counted from coverage.
    assert observations.total_obs == 25
    assert warn is not None
    assert warn.code == OBSERVATIONS_SERIES_TRUNCATED
    assert warn.params == {"shown": shown, "total": 25, "order": order}
    assert observations.order == order
    assert f"showing {shown} of 25" in warn.message
    assert ("not ranked by value" in warn.message) == (order == "fixed")
    (call,) = sdmx.series_calls
    delivered = [
        next(e.ref for e in s.entities if e.role == "destination") for s in observations.series
    ]
    if order == "ranked":
        # One read of every series, filtered on the bound provider. With every role open,
        # the filter is the narrower role, the one provider against 25 recipients.
        assert call["entities"] == {"provider": [_DONOR]}
        assert delivered == recipients[:4:-1]
    else:
        # The first 20 by identity, read pair by pair, shown largest first.
        fixed = sorted(recipients)[:shown]
        assert set(call["entities"]["recipient"]) == set(fixed)
        assert set(delivered) == set(fixed)
        assert delivered == sorted(fixed, key=recipients.index, reverse=True)


def test_a_ranked_read_takes_every_stat_var_with_data_in_its_years():
    """A StatVar whose role tuple holds no observation in the read's years takes no place in
    a ranked read, however new its latest date, so every StatVar with data there is read
    and shown, newest year first."""
    late, newer, older = (f"TEST/BulkRoleObs_{name}" for name in ("Late", "Newer", "Older"))
    # Counts are the observations in the read's years: the late StatVar's one tuple has none.
    members = [
        (late, "country/BR1", "2024", 0),
        (newer, "country/BR2", "2018", 1),
        (older, "country/BR3", "2017", 1),
    ]
    sdmx = FakeSdmxClient(
        observations={
            sv: {_role_tuple(recipient): [_facet(dates=[year])]}
            for sv, recipient, year, _n_obs in members
        }
    )
    observations, warn = bulk_role_series_planned(
        [sv for sv, *_ in members],
        _obs_entities([recipient for _sv, recipient, _year, _n_obs in members]),
        sdmx=sdmx,
        graph=_GRAPH,
        facet_filter=None,
        discovered_role_index={
            sv: [
                role_candidate(
                    facet_id="F1", role_tuples={_role_tuple(recipient): n_obs}, latest_date=year
                )
            ]
            for sv, recipient, year, n_obs in members
        },
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert [s.stat_var.ref for s in observations.series] == [newer, older]
    assert observations.order is None


def test_a_fixed_range_read_takes_only_series_with_data_in_the_range(search_settings):
    """Over the row gate a range read selects only role tuples holding an observation in
    the range, so a tuple sorting first with none there takes no place in the preview."""
    search_settings(role_bulk_rank_max_rows=0)
    shown = observation_settings().role_bulk_series_max
    dataless = "country/BRFixed00"
    with_data = [f"country/BRFixed{i:02d}" for i in range(1, shown + 2)]
    sdmx = FakeSdmxClient(
        observations={
            _SV: {
                _role_tuple(r): [_facet(dates=["2020"], value=float(i))]
                for i, r in enumerate(with_data)
            }
        }
    )
    observations, warn = bulk_role_series_planned(
        [_SV],
        _obs_entities([dataless, *with_data]),
        sdmx=sdmx,
        graph=_GRAPH,
        facet_filter=None,
        discovered_role_index={
            _SV: [
                role_candidate(
                    facet_id="F1",
                    role_tuples={
                        _role_tuple(dataless): 0,
                        **{_role_tuple(r): 1 for r in with_data},
                    },
                    latest_date="2020",
                )
            ]
        },
        time_period=["2019", "2020"],
        coverage=empty_coverage_snapshot(),
    )
    (call,) = sdmx.series_calls
    assert dataless not in call["entities"]["recipient"]
    assert len(observations.series) == shown
    assert warn is not None
    assert warn.params == {"shown": shown, "total": shown + 1, "order": "fixed"}
    assert observations.order == "fixed"


def test_unshipped_pair_total_obs_selects_the_same_facet_as_the_shipped_ones(
    monkeypatch, search_settings
):
    """An unshipped pair adds the selected facet's count to `total_obs`."""
    # Over the row gate the preview reads a fixed selection, which leaves BRTrunc7 out.
    search_settings(role_bulk_rank_max_rows=0)
    recipients = [f"country/BRTrunc{i}" for i in range(1, 26)]  # 25 recipients
    # BRTrunc7 is unshipped. Its F1 row has 2 points and F2 has 5.
    rows = [("F1", _DONOR, r, [2020]) for r in recipients if r != "country/BRTrunc7"]
    rows.append(("F1", _DONOR, "country/BRTrunc7", [2020, 2021]))
    rows.append(("F2", _DONOR, "country/BRTrunc7", [2020, 2021, 2022, 2023, 2024]))
    # F2 holds another source, so the group F1 leads holds F1 alone.
    coverage = _patch_role_coverage(
        monkeypatch,
        rows,
        facet_metadata={
            "F1": FacetMetadata(provenance_id="dc/base/CoverageFixture"),
            "F2": FacetMetadata(provenance_id="dc/base/OtherFixture"),
        },
    )
    obs = {
        _role_tuple(r): [_facet(dates=["2020"], value=float(i))] for i, r in enumerate(recipients)
    }
    sdmx = FakeSdmxClient(observations={_SV: obs})
    observations, warn = bulk_role_series_planned(
        [_SV],
        _obs_entities(recipients),
        sdmx=sdmx,
        graph=_GRAPH,
        facet_filter=None,
        coverage=coverage,
    )
    assert observations is not None
    assert len(observations.series) == observation_settings().role_bulk_series_max
    assert "country/BRTrunc7" not in {
        next(e.ref for e in s.entities if e.role == "destination") for s in observations.series
    }
    # BRTrunc7 contributes F1's count to `total_obs`.
    assert observations.total_obs == 26
    assert warn is not None
    assert warn.code == OBSERVATIONS_SERIES_TRUNCATED


_SINGLE_AXIS_SV = "TEST/BulkRoleObs_SingleAxis"
_SINGLE_AXIS_UNBOUND_DONOR = "country/SAUnbound"
# This shape has one entity role. Coverage rows use fixed donor/recipient columns.
_SINGLE_AXIS_ROLE_NAMES = {_SINGLE_AXIS_SV: RoleNames(donor="provider", recipient="recipient")}


def _single_axis_role_tuple(recipient: str) -> tuple[tuple[str, str], ...]:
    return (("recipient", recipient),)


def _single_axis_obs_entities(recipients: list[str]) -> dict[str, list[str]]:
    # The one-role shape omits the provider key entirely.
    return {"recipient": list(recipients)}


def test_unshipped_pair_total_obs_for_a_one_role_shape(monkeypatch):
    """Unshipped counts must use the one-entry role tuple for a one-role shape."""
    recipients = [f"country/SATrunc{i}" for i in range(1, 26)]  # 25 recipients
    rows = [(_SINGLE_AXIS_SV, "F1", _SINGLE_AXIS_UNBOUND_DONOR, r, [2020]) for r in recipients]
    index = patch_coverage(monkeypatch, rows, role_names=_SINGLE_AXIS_ROLE_NAMES)
    coverage = FakeCoverageService(index).snapshot()
    obs = {
        _single_axis_role_tuple(r): [_facet(dates=["2020"], value=float(i))]
        for i, r in enumerate(recipients)
    }
    sdmx = FakeSdmxClient(observations={_SINGLE_AXIS_SV: obs})
    observations, warn = bulk_role_series_planned(
        [_SINGLE_AXIS_SV],
        _single_axis_obs_entities(recipients),
        sdmx=sdmx,
        graph=_GRAPH,
        facet_filter=None,
        coverage=coverage,
    )
    assert observations is not None
    assert len(observations.series) == observation_settings().role_bulk_series_max
    # Every one of the 25 confirmed pairs carries exactly one in-window
    # `total_obs` includes 20 fetched pairs and 5 counted from coverage.
    assert observations.total_obs == 25
    assert warn is not None
    assert warn.code == OBSERVATIONS_SERIES_TRUNCATED


@pytest.mark.parametrize(
    ("to_fetch", "expected"),
    [
        pytest.param(
            [
                ("sv", (("a_role", "x1"), ("z_role", "y"))),
                ("sv", (("a_role", "x2"), ("z_role", "y"))),
            ],
            {"z_role": ["y"]},
            id="narrower-role-sorts-last-by-name",
        ),
        pytest.param(
            [
                ("sv", (("b_role", "y1"), ("a_role", "x1"))),
                ("sv", (("b_role", "y2"), ("a_role", "x2"))),
            ],
            {"a_role": ["x1", "x2"]},
            id="tie-resolves-by-role-name",
        ),
        pytest.param(
            [
                ("sv", (("a_role", "x1"), ("b_role", "y1"))),
                ("sv", (("a_role", "x2"),)),
            ],
            {"a_role": ["x1", "x2"]},
            id="role-missing-from-a-pair-is-excluded",
        ),
        pytest.param(
            [("sv", (("a_role", "x1"),)), ("sv", (("b_role", "y1"),))],
            ValueError,
            id="no-common-role-raises",
        ),
    ],
)
def test_narrowest_common_role_filters_on_the_role_with_fewest_dcids_every_pair_carries(
    to_fetch, expected
) -> None:
    if expected is ValueError:
        with pytest.raises(ValueError):
            _narrowest_common_role(to_fetch)
    else:
        assert _narrowest_common_role(to_fetch) == expected


def test_preview_pairs_takes_stat_vars_newest_year_first_round_robin_up_to_the_limit() -> None:
    """Preview `(StatVar, role_tuple)` pairs StatVar by StatVar, newest member year first,
    taking each StatVar's next role tuple in turn. StatVars past the limit are not read.
    """

    class _Unread:
        def head(self, limit, *, first_year, last_year, with_data=False):
            raise AssertionError("a StatVar past the preview limit was read")

        def n_obs(self, *, first_year, last_year):
            return 7

    sv_a, sv_b, sv_c = "sv/A", "sv/B", "sv/C"
    index = {
        sv_b: [
            role_candidate(
                facet_id="F1",
                role_tuples={_role_tuple("country/Z"): 4, _role_tuple("country/V"): 1},
                latest_date="2020",
            )
        ],
        sv_a: [
            role_candidate(
                facet_id="F1",
                role_tuples={_role_tuple("country/Y"): 2, _role_tuple("country/X"): 3},
                latest_date="2020",
            )
        ],
        sv_c: [
            replace(
                role_candidate(
                    facet_id="F1", role_tuples=[_role_tuple("country/W")], latest_date="2020"
                ),
                pairs=_Unread(),
            )
        ],
    }
    merged = merge_role_candidates(index, [sv_b, sv_c, sv_a])

    member_years = {sv_b: 2022, sv_a: 2021, sv_c: 2019}
    preview = _preview_pairs(
        BulkPlan(
            members=(sv_b, sv_c, sv_a),
            years=None,
            facet_id="F1",
            selected=lead_selected(merged, "F1"),
            candidates=merged,
        ),
        2,
        member_year=member_years.get,
    )

    # sv_b is newest, so it gives its first tuple. sv_a then gives its first before sv_b
    # gives a second.
    assert preview == [
        ((sv_b, _role_tuple("country/V")), 1),
        ((sv_a, _role_tuple("country/X")), 3),
    ]
    assert merged["F1"].pair_count == 5
    assert role_n_obs(merged["F1"], first_year=None, last_year=None) == 3 + 2 + 4 + 1 + 7


# Pipeline test for a bound-unit request at the preview ceiling.


_PIPE_SV_DCID = "TESTFIXTURE/BulkRolePipeline/Flow"
_NOR = "country/NOR"
_KEN = "country/KEN"
_UGA = "country/UGA"


def _pipe_declaring_arcs() -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": "provider"}, {"dcid": "recipient"}]},
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
    }


class _FixedBindLLM:
    """Elects shape index 0 unconditionally, with no constraint bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, bindings=[]), None


def test_role_bearing_bound_unit_overflow_never_escalates_to_graph_infra_error(
    monkeypatch, search_settings
):
    """Two destinations exceed a one-pair cap, as more than 120 pairs do live.
    The degrade must not become a retryable 503 GraphInfraError; the result is definite.
    """
    search_settings(obs_preview_max_pairs=1)
    resolve = {"Norway": _NOR, "Kenya": _KEN, "Uganda": _UGA}
    role_query = "aid from Norway to Kenya and Uganda"
    graph = FakeGraph(
        nodes={
            **{dcid: {"label": name, "type": "Country"} for name, dcid in resolve.items()},
            _PIPE_SV_DCID: {"label": "Bulk role pipeline flow", "arcs": _pipe_declaring_arcs()},
        },
        detect={role_query: {"svs": [_PIPE_SV_DCID], "cosine_scores": [0.9]}},
        resolve=resolve,
        obs={},
    )
    # Confirmation and bulk indexing share coverage. Its facet F1 must be the id SDMX serves
    # points under, or the narrowed read comes back empty.
    coverage_index = patch_coverage(
        monkeypatch,
        [
            (_PIPE_SV_DCID, "F1", _NOR, _KEN, [2020]),
            (_PIPE_SV_DCID, "F1", _NOR, _UGA, [2020]),
        ],
        role_names={_PIPE_SV_DCID: RoleNames(donor="provider", recipient="recipient")},
        # The shared coverage facet carries native USD, so no recipe is needed.
        facet_metadata={"F1": FacetMetadata(provenance_id="dc/base/BRSrc", unit="CurrencyUSD")},
    )
    coverage = FakeCoverageService(coverage_index).snapshot()
    sdmx = FakeSdmxClient(
        observations={
            _PIPE_SV_DCID: {
                (("provider", _NOR), ("recipient", _KEN)): [
                    _facet(dates=["2020"], value=1.0, unit="CurrencyUSD")
                ],
                (("provider", _NOR), ("recipient", _UGA)): [
                    _facet(dates=["2020"], value=2.0, unit="CurrencyUSD")
                ],
            }
        },
    )
    # Materialise uses a separate graph/client pair to confirm the native USD facet.
    materialise_body = _csv_header(["provider", "recipient"]) + _csv_row(
        stat_var=_PIPE_SV_DCID,
        role_dcids=[_NOR, _KEN],
        fixture=ObsFacetFixture(
            facet_id="f1",
            points=(("2020", 1.0),),
            provenance_id="dc/base/BRSrc",
            unit="CurrencyUSD",
        ),
        date_str="2020",
        value=1.0,
    )
    materialise_graph = FakeGraphReader(
        arcs={
            _PIPE_SV_DCID: {"measurementQualifier": ("Nominal",)},
            "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
        }
    )
    materialise_sdmx = sdmx_double(csv_body=materialise_body)
    result = asyncio.run(
        resolve_variable(
            "aid",
            entities=["Norway", "Kenya", "Uganda"],
            date_request=None,
            detect_query=role_query,
            role_query=role_query,
            entity_roles=door_roles_for_test_query(role_query, ["Norway", "Kenya", "Uganda"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=sdmx,
            base_timing={},
            unit_text="US dollars",
            coverage=coverage,
            options=EngineOptions(
                include_observations=True,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
            ),
        )
    )
    assert result.status == "definite", (
        f"expected definite, got {result.status!r} "
        f"(no_data_reason={getattr(result, 'no_data_reason', None)!r})"
    )
    assert not any(
        w.code in (OBSERVATIONS_PROBE_FAILED, OBSERVATIONS_BULK_FETCH_FAILED)
        for w in result.warnings
    )
    observations = result.specs[0].resolved.observations
    assert observations is not None
    assert len(observations.series) == 2


def test_a_bulk_preview_keeps_the_exact_headline_of_every_confirmed_pair(
    monkeypatch, search_settings
):
    """Norway funds 25 recipients in 2020, read in bulk. The preview ships the first 20
    series (`role_bulk_series_max`), and the headline still counts the 25 series and 25
    recipients the read serves, never the 20 previewed."""
    search_settings(obs_preview_max_pairs=1)
    recipients = [f"country/BRTrunc{i:02d}" for i in range(1, 26)]
    role_query = "aid from Norway"
    graph = FakeGraph(
        nodes={
            _NOR: {"label": "Norway", "type": "Country"},
            _PIPE_SV_DCID: {"label": "Bulk role pipeline flow", "arcs": _pipe_declaring_arcs()},
        },
        detect={role_query: {"svs": [_PIPE_SV_DCID], "cosine_scores": [0.9]}},
        resolve={"Norway": _NOR},
        obs={},
    )
    coverage_index = patch_coverage(
        monkeypatch,
        [(_PIPE_SV_DCID, "F1", _NOR, recipient, [2020]) for recipient in recipients],
        role_names={_PIPE_SV_DCID: RoleNames(donor="provider", recipient="recipient")},
        facet_metadata={"F1": FacetMetadata(provenance_id="dc/base/BRSrc")},
    )
    sdmx = FakeSdmxClient(
        observations={
            _PIPE_SV_DCID: {
                (("provider", _NOR), ("recipient", recipient)): [
                    _facet(dates=["2020"], value=float(i), provenance_id="dc/base/BRSrc")
                ]
                for i, recipient in enumerate(recipients)
            }
        },
    )
    result = asyncio.run(
        resolve_variable(
            "aid",
            entities=["Norway"],
            date_request=None,
            detect_query=role_query,
            role_query=role_query,
            entity_roles=door_roles_for_test_query(role_query, ["Norway"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=sdmx,
            base_timing={},
            coverage=FakeCoverageService(coverage_index).snapshot(),
            options=EngineOptions(include_observations=True),
        )
    )

    assert result.status == "definite", result.no_data_reason
    resolved = result.specs[0].resolved
    assert resolved.observations is not None
    assert len(resolved.observations.series) == observation_settings().role_bulk_series_max == 20
    assert resolved.headline is not None
    assert resolved.headline.series == 25
    assert sorted(resolved.headline.role_counts.values()) == [1, 25]


def _bulk_answer_in(monkeypatch, window: TimeWindow) -> tuple[RegionResult, FakeSdmxClient]:
    """Norway's aid to Kenya and Uganda in 2020, asked for `window` over a one-pair cap,
    so the preview reads in bulk. Returns the region result and the SDMX double."""
    resolve = {"Norway": _NOR, "Kenya": _KEN, "Uganda": _UGA}
    role_query = "aid from Norway to Kenya and Uganda"
    graph = FakeGraph(
        nodes={
            **{dcid: {"label": name, "type": "Country"} for name, dcid in resolve.items()},
            _PIPE_SV_DCID: {"label": "Bulk role pipeline flow", "arcs": _pipe_declaring_arcs()},
        },
        detect={role_query: {"svs": [_PIPE_SV_DCID], "cosine_scores": [0.9]}},
        resolve=resolve,
        obs={},
    )
    coverage_index = patch_coverage(
        monkeypatch,
        [
            (_PIPE_SV_DCID, "F1", _NOR, _KEN, [2020]),
            (_PIPE_SV_DCID, "F1", _NOR, _UGA, [2020]),
        ],
        role_names={_PIPE_SV_DCID: RoleNames(donor="provider", recipient="recipient")},
        facet_metadata={"F1": FacetMetadata(provenance_id="dc/base/BRSrc")},
    )
    sdmx = FakeSdmxClient(
        observations={
            _PIPE_SV_DCID: {
                (("provider", _NOR), ("recipient", recipient)): [
                    _facet(dates=["2020"], value=1.0, provenance_id="dc/base/BRSrc")
                ]
                for recipient in (_KEN, _UGA)
            }
        },
    )
    result = asyncio.run(
        resolve_variable(
            "aid",
            entities=["Norway", "Kenya", "Uganda"],
            date_request=DateRequest(window=window, latest=False),
            detect_query=role_query,
            role_query=role_query,
            entity_roles=door_roles_for_test_query(role_query, ["Norway", "Kenya", "Uganda"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=sdmx,
            base_timing={},
            coverage=FakeCoverageService(coverage_index).snapshot(),
            options=EngineOptions(include_observations=True),
        )
    )
    return result, sdmx


def test_a_bulk_window_in_the_data_years_reads_in_bulk(monkeypatch, search_settings):
    """The control for the pre-epoch window below: 2020 reads both pairs in one bulk read."""
    search_settings(obs_preview_max_pairs=1)
    result, sdmx = _bulk_answer_in(monkeypatch, TimeWindow(start_year=2020, end_year=2020))

    assert result.status == "definite", result.no_data_reason
    assert len(sdmx.series_calls) == 1
    assert sdmx.series_calls[0]["time_period"] == ["2020"]


def test_a_bulk_window_wholly_before_the_epoch_year_answers_no_observations_unread(
    monkeypatch, search_settings
):
    """1800 to 1850 ends before BUILD_EPOCH_YEAR (1960), the first year coverage can
    represent, so the read holds no year (`read_plan.role_years` raises `NoYearsToRead`):
    the answer is `no_observations` and no SDMX read is made."""
    from search.engine.retrieval import read_plan, role_resolve, role_scope

    assert BUILD_EPOCH_YEAR > 1850
    search_settings(obs_preview_max_pairs=1)
    refused: list[object] = []
    real = read_plan.role_years

    def role_years(date_request, horizon):
        try:
            return real(date_request, horizon)
        except read_plan.NoYearsToRead:
            refused.append(date_request)
            raise

    for module in (role_resolve, role_scope):
        monkeypatch.setattr(module, "role_years", role_years)
    result, sdmx = _bulk_answer_in(monkeypatch, TimeWindow(start_year=1800, end_year=1850))

    assert refused, "the read plan found years to read"
    assert result.status == "no_data", result
    assert result.no_data_reason == "no_observations"
    assert sdmx.series_calls == []
    assert sdmx.availability_calls == []
    assert sdmx.subject_observations_calls == []


def test_shared_facet_unit_and_method_carry_their_graph_names(monkeypatch):
    """The unit and measurement method are labelled from the graph, so a native USD answer
    renders "US dollar" in place of the split dcid "Currency USD"."""
    recipients = [f"country/BRLabel{i}" for i in range(1, 3)]
    coverage = _patch_role_coverage(
        monkeypatch,
        [("F1", _DONOR, r, [2020]) for r in recipients],
        facet_metadata={
            "F1": FacetMetadata(unit="CurrencyUSD", measurement_method="dcAggregate/Usd")
        },
    )
    observations_by_tuple = {
        _role_tuple(recipient): [
            _facet(
                dates=["2020"],
                value=float(index),
                unit="CurrencyUSD",
                measurement_method="dcAggregate/Usd",
            )
        ]
        for index, recipient in enumerate(recipients)
    }
    graph = FakeGraph(
        nodes={
            **_NODES,
            "CurrencyUSD": {"label": "US dollar"},
            "dcAggregate/Usd": {"label": "USD aggregate"},
        },
        obs={},
    )
    sdmx = FakeSdmxClient(observations={_SV: observations_by_tuple})

    observations, warn = bulk_role_series_planned(
        [_SV],
        _obs_entities(recipients),
        sdmx=sdmx,
        graph=graph,
        facet_filter=None,
        coverage=coverage,
    )

    assert warn is None
    assert observations is not None
    assert observations.series
    for series in observations.series:
        assert series.facet["unit"].label == "US dollar"
        assert series.facet["measurementMethod"].label == "USD aggregate"


def _native_usd_elected(stat_vars) -> dict:
    native_usd = _facet_selection(
        axis="unit",
        ref="CurrencyUSD",
        selection_reason="native_currency_preferred",
        request=UnitRequest(currency="USD", prices="current"),
    )
    return {sv: ({"unit": native_usd},) for sv in stat_vars}


def test_unit_availability_covers_every_confirmed_statvar_when_emission_truncates(
    monkeypatch, search_settings
):
    """The preview ships `role_bulk_series_max` pairs, all of the first StatVar in identity
    order; the evidence still names every StatVar the plan serves under the elected unit."""
    search_settings(obs_preview_max_pairs=1)
    sv_b = "TEST/BulkRoleObs_FlowB"
    recipients = [f"country/BRTrunc{i}" for i in range(1, 26)]
    index = patch_coverage(
        monkeypatch,
        [(sv, "F1", _DONOR, r, [2020]) for sv in (_SV, sv_b) for r in recipients],
        role_names={sv: RoleNames(donor="provider", recipient="recipient") for sv in (_SV, sv_b)},
        facet_metadata={"F1": FacetMetadata(unit="CurrencyUSD")},
    )
    sdmx = FakeSdmxClient(
        observations={
            sv: {_role_tuple(r): [_facet(dates=["2020"], unit="CurrencyUSD")] for r in recipients}
            for sv in (_SV, sv_b)
        }
    )
    (observations, warn), evidence = unit_check_planned(
        [_SV, sv_b],
        _obs_entities(recipients),
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        sdmx=sdmx,
        unit_request=IdentityUnitRequest(currency="USD", prices="current"),
        elected=_native_usd_elected((_SV, sv_b)),
        coverage=FakeCoverageService(index).snapshot(),
    )

    shown = {s.stat_var.ref for s in observations.series}
    assert len(observations.series) == observation_settings().role_bulk_series_max
    assert shown == {_SV}
    assert warn is not None
    assert warn.code == OBSERVATIONS_SERIES_TRUNCATED
    assert evidence.confirmed == {_SV, sv_b}


def test_unit_availability_counts_only_in_window_coverage(monkeypatch):
    """A StatVar whose coverage rows all fall before the window is not confirmed."""
    sv_b = "TEST/BulkRoleObs_FlowB"
    index = patch_coverage(
        monkeypatch,
        [(_SV, "F1", _DONOR, r, [2020]) for r in _RECIPIENTS]
        + [(sv_b, "F1", _DONOR, r, [2010]) for r in _RECIPIENTS],
        role_names={sv: RoleNames(donor="provider", recipient="recipient") for sv in (_SV, sv_b)},
        facet_metadata={"F1": FacetMetadata(unit="CurrencyUSD")},
    )
    _preview, evidence = unit_check_planned(
        [_SV, sv_b],
        _obs_entities(_RECIPIENTS),
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        window=TimeWindow(start_year=2019, end_year=2021),
        sdmx=FakeSdmxClient(
            observations={
                _SV: {
                    _role_tuple(r): [_facet(dates=["2020"], unit="CurrencyUSD")]
                    for r in _RECIPIENTS
                }
            }
        ),
        unit_request=IdentityUnitRequest(currency="USD", prices="current"),
        elected=_native_usd_elected((_SV, sv_b)),
        coverage=FakeCoverageService(index).snapshot(),
    )
    assert evidence.confirmed == {_SV}


def test_unit_availability_under_latest_confirms_each_member_at_its_own_year(monkeypatch):
    """A `latest` bulk read serves each StatVar its own latest year, as a mint of it does,
    so a StatVar whose pairs end in an earlier year than another's is confirmed there."""
    sv_b = "TEST/BulkRoleObs_FlowB"
    index = patch_coverage(
        monkeypatch,
        [(_SV, "F1", _DONOR, r, [2015, 2020]) for r in _RECIPIENTS]
        + [(sv_b, "F1", _DONOR, r, [2015]) for r in _RECIPIENTS],
        role_names={sv: RoleNames(donor="provider", recipient="recipient") for sv in (_SV, sv_b)},
        facet_metadata={"F1": FacetMetadata(unit="CurrencyUSD")},
    )
    _preview, evidence = unit_check_planned(
        [_SV, sv_b],
        _obs_entities(_RECIPIENTS),
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        window="latest",
        sdmx=FakeSdmxClient(observations={}),
        unit_request=IdentityUnitRequest(currency="USD", prices="current"),
        elected=_native_usd_elected((_SV, sv_b)),
        coverage=FakeCoverageService(index).snapshot(),
    )
    assert evidence.confirmed == {_SV, sv_b}


def test_unit_availability_confirms_only_the_elected_unit(monkeypatch):
    """The StatVar's US dollar facet ends before the window and only its euro facet holds
    in-window pairs. The plan narrowed to the elected US dollar unit serves no pair in the
    window, so the StatVar is unconfirmed although the index holds it in the window."""
    index = patch_coverage(
        monkeypatch,
        [(_SV, "FUSD", _DONOR, r, [2015]) for r in _RECIPIENTS]
        + [(_SV, "FEUR", _DONOR, r, [2020]) for r in _RECIPIENTS],
        role_names={_SV: RoleNames(donor="provider", recipient="recipient")},
        facet_metadata={
            "FUSD": FacetMetadata(provenance_id="dc/base/Src", unit="CurrencyUSD"),
            "FEUR": FacetMetadata(provenance_id="dc/base/Src", unit="CurrencyEUR"),
        },
    )
    _preview, evidence = unit_check_planned(
        [_SV],
        _obs_entities(_RECIPIENTS),
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        window=TimeWindow(start_year=2019, end_year=2021),
        sdmx=FakeSdmxClient(observations={}),
        unit_request=IdentityUnitRequest(currency="USD", prices="current"),
        elected=_native_usd_elected((_SV,)),
        coverage=FakeCoverageService(index).snapshot(),
    )
    assert evidence.confirmed == frozenset()
    assert evidence.serves_any([_SV]) is False
