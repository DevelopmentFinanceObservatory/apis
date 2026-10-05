"""Role-bearing observation previews use coverage-built role indexes and minimal family members.
Tests pin antichain reduction and avoid member enumeration."""

from __future__ import annotations

import search.engine.retrieval.read_plan as read_plan
import statvar_members
from search.engine.clients.graph import Facet
from search.engine.region.codes import (
    OBSERVATIONS_CLIENTS_UNAVAILABLE,
    OBSERVATIONS_PROBE_FAILED,
    OBSERVATIONS_SERIES_TRUNCATED,
)
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.retrieval.read_plan import minimal_constraint_svs
from search.observations import observation_settings
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import FakeCoverageService, empty_coverage_snapshot, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import build_observations_planned
from tests.fixtures import FakeGraph
from tests.statvar_members.conftest import FakeMemberArcReader

# _minimal_constraint_svs reduction tests use fixed, empty absent-reading facts.


def test_minimal_constraint_svs_uses_the_shared_package_reduction():
    # Both modules must expose the same reduction function.
    assert read_plan.antichain_minimal_ids is statvar_members.antichain_minimal_ids


def test_minimal_constraint_svs_drops_the_dominated_part(monkeypatch):
    monkeypatch.setattr(read_plan, "_absent_readings", lambda: {})
    result = minimal_constraint_svs(
        ["Total", "BySectorHealth"],
        {"Total": {}, "BySectorHealth": {"purpose": "Health"}},
    )
    assert result == ["Total"]


def test_minimal_constraint_svs_keeps_disjoint_members(monkeypatch):
    monkeypatch.setattr(read_plan, "_absent_readings", lambda: {})
    # Disjoint constraints remain incomparable.
    result = minimal_constraint_svs(
        ["Male", "Female"], {"Male": {"gender": "Male"}, "Female": {"gender": "Female"}}
    )
    assert result == ["Male", "Female"]


def test_minimal_constraint_svs_respects_not_declared_reading(monkeypatch):
    monkeypatch.setattr(read_plan, "_absent_readings", lambda: {"modality": "not_declared"})
    result = minimal_constraint_svs(
        ["Purpose", "PurposeModality"],
        {
            "Purpose": {"purpose": "Health"},
            "PurposeModality": {"purpose": "Health", "modality": "Grant"},
        },
    )
    assert result == ["Purpose", "PurposeModality"]


def test_minimal_constraint_svs_never_judges_an_sv_with_no_known_constraints(monkeypatch):
    """Unknown constraints must not make an SV dominate a known sibling."""
    monkeypatch.setattr(read_plan, "_absent_readings", lambda: {})
    result = minimal_constraint_svs(["Total", "Unknown"], {"Total": {}})
    assert result == ["Total", "Unknown"]


# Role-bearing _build_observations tests.

_DONOR = "country/RPDonor"
_RECIPIENTS = [f"country/RP{i}" for i in range(1, observation_settings().obs_preview_max_pairs + 2)]
_TOTAL_SV = "TEST/RolePreview_Total"
_HEALTH_SV = "TEST/RolePreview_BySector_Health"


def _role_tuple(recipient: str) -> tuple[tuple[str, str], ...]:
    return (("provider", _DONOR), ("recipient", recipient))


def _facet(*, dates: list[str], value: float = 1.0, facet_id: str = "F1") -> Facet:
    return Facet(
        earliest_date=dates[0],
        latest_date=dates[-1],
        obs_count=len(dates),
        dates=dates,
        values=[value for _ in dates],
        facet_id=facet_id,
    )


def _sv_node(label: str) -> dict:
    return {
        "label": label,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": label}]},
            "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "observationProperties": {"nodes": [{"dcid": "provider"}, {"dcid": "recipient"}]},
        },
    }


_NODES = {
    _TOTAL_SV: _sv_node("Total ODA grants"),
    _HEALTH_SV: _sv_node("ODA grants for health"),
    _DONOR: {"label": "Donor"},
    **{c: {"label": f"Recipient {c.rsplit('RP', 1)[-1]}"} for c in _RECIPIENTS},
}
_GRAPH = FakeGraph(nodes=_NODES, obs={})

# The reverse-arc family enumerate_members would read if it were ever
# called -- populated and answerable, so a test proving it is NOT called
# proves an actual skip, not an incidental failure of a broken double.
_MEMBER_ARC_FAMILY = {
    _TOTAL_SV: {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("EconomicActivity",),
        "measuredProperty": ("amount",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": (),
        "observationProperties": ("provider", "recipient"),
    },
    _HEALTH_SV: {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("EconomicActivity",),
        "measuredProperty": ("amount",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("purpose",),
        "observationProperties": ("provider", "recipient"),
        "purpose": ("Health",),
    },
}


def _member_facts(constraints: dict[str, str]) -> ArcsMemberFacts:
    arcs = {
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraints]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in constraints.items()},
    }
    return ArcsMemberFacts(arcs=arcs)


def _obs_entities(recipients: list[str]) -> dict[str, list[str]]:
    return {"provider": [_DONOR], "recipient": list(recipients)}


def _unreachable_enumerate_members(*args: object, **kwargs: object) -> None:
    """Fail if the role-bearing preview enumerates members."""
    raise AssertionError("enumerate_members must not be called on the role-bearing preview path")


def test_broad_role_bearing_answer_ships_bulk_series_with_warning_and_counts(monkeypatch):
    """Bulk preview caps series and warns when recipients exceed the cap."""
    monkeypatch.setattr(statvar_members, "enumerate_members", _unreachable_enumerate_members)
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_TOTAL_SV, "F1", _DONOR, r, [2020]) for r in _RECIPIENTS],
        role_names={_TOTAL_SV: RoleNames(donor="provider", recipient="recipient")},
    )
    sdmx = FakeSdmxClient(
        observations={_TOTAL_SV: {_role_tuple(r): [_facet(dates=["2020"])] for r in _RECIPIENTS}}
    )
    observations, warn = build_observations_planned(
        [_TOTAL_SV],
        _obs_entities(_RECIPIENTS),
        graph=_GRAPH,
        materialise_graph=FakeMemberArcReader(arcs=_MEMBER_ARC_FAMILY),
        materialise_sdmx=None,
        sdmx=sdmx,
        member_facts={_TOTAL_SV: _member_facts({})},
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert observations is not None
    assert 0 < len(observations.series) <= observation_settings().role_bulk_series_max
    if len(observations.series) < len(_RECIPIENTS):
        assert warn is not None
        assert warn.code == OBSERVATIONS_SERIES_TRUNCATED
        assert str(len(observations.series)) in warn.message
        assert str(len(_RECIPIENTS)) in warn.message


def test_role_bearing_path_never_calls_enumerate_members_below_the_ceiling_too(monkeypatch):
    """Small role-bearing previews also skip member enumeration."""
    monkeypatch.setattr(statvar_members, "enumerate_members", _unreachable_enumerate_members)
    small_recipients = _RECIPIENTS[:3]
    sdmx = FakeSdmxClient(
        observations={
            _TOTAL_SV: {_role_tuple(r): [_facet(dates=["2020"])] for r in small_recipients},
            _HEALTH_SV: {_role_tuple(r): [_facet(dates=["2020"])] for r in small_recipients},
        }
    )
    observations, warn = build_observations_planned(
        [_TOTAL_SV, _HEALTH_SV],
        _obs_entities(small_recipients),
        graph=_GRAPH,
        materialise_graph=FakeMemberArcReader(arcs=_MEMBER_ARC_FAMILY),
        materialise_sdmx=None,
        sdmx=sdmx,
        member_facts={
            _TOTAL_SV: _member_facts({}),
            _HEALTH_SV: _member_facts({"purpose": "Health"}),
        },
        coverage=empty_coverage_snapshot(),
    )
    # Below the ceiling, this is the per-pair path (dispatched through
    # materialise_preview, not _bulk_role_series) -- with no
    # materialise_graph/materialise_sdmx pair supplied to it, it is the
    # honest "nothing to preview" omission.
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_CLIENTS_UNAVAILABLE
    assert warn is None


def test_dominated_candidate_never_appears_beside_its_dominating_sibling(monkeypatch):
    """The total replaces its purpose=Health part before the role index is read."""
    monkeypatch.setattr(read_plan, "_absent_readings", lambda: {})
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_TOTAL_SV, "F1", _DONOR, r, [2020]) for r in _RECIPIENTS]
        + [(_HEALTH_SV, "F1", _DONOR, r, [2020]) for r in _RECIPIENTS],
        role_names={
            _TOTAL_SV: RoleNames(donor="provider", recipient="recipient"),
            _HEALTH_SV: RoleNames(donor="provider", recipient="recipient"),
        },
    )
    sdmx = FakeSdmxClient(
        observations={
            _TOTAL_SV: {_role_tuple(r): [_facet(dates=["2020"], value=1.0)] for r in _RECIPIENTS},
            _HEALTH_SV: {_role_tuple(r): [_facet(dates=["2020"], value=2.0)] for r in _RECIPIENTS},
        }
    )
    observations, warn = build_observations_planned(
        [_TOTAL_SV, _HEALTH_SV],
        _obs_entities(_RECIPIENTS),
        graph=_GRAPH,
        materialise_graph=None,
        materialise_sdmx=None,
        sdmx=sdmx,
        member_facts={
            _TOTAL_SV: _member_facts({}),
            _HEALTH_SV: _member_facts({"purpose": "Health"}),
        },
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert observations is not None
    assert {s.stat_var.ref for s in observations.series} == {_TOTAL_SV}
    # The dominated sv never even reaches the bulk dispatch -- not just the
    # shown series: the narrowed role_series read itself only ever names
    # _TOTAL_SV.
    assert sdmx.series_calls[0]["variables"] == [_TOTAL_SV]
    assert warn is None or warn.code != OBSERVATIONS_PROBE_FAILED


# The standard (observationAbout) branch's own equivalent regression guard --
# proving it, too, never calls enumerate_members -- lives in
# tests/engine/test_group_preview_membership.py, alongside its other tests.
