"""Test open-role facts and coverage-backed origin discovery (#664, #880)."""

from __future__ import annotations

from search.engine.retrieval.retrieve import Materialised, materialise
from search.engine.retrieval.role_resolve import graph_confirm_resolve
from search.engine.retrieval.role_scope import open_role_axis_members
from search.engine.shape import ShapeDraft
from search.engine.spec_build import RoleAxisFactInput, _role_axis_facts
from search.models import CoverageExact, RoleAxisFact
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph

_SV = "TESTFIXTURE/RoleAxisFacts/Flow"
_FINANCING_ROLE_NAMES = RoleNames(donor="financingProvider", recipient="financingRecipient")


def _role_arcs() -> dict:
    """Return the observationProperties arcs required for both declared roles."""
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {
            "nodes": [{"dcid": "financingProvider"}, {"dcid": "financingRecipient"}]
        },
    }


def _financing_shape(*, sv_dcids: tuple[str, ...] = (_SV,)) -> ShapeDraft:
    """Build a neutral-namespace shape declaring both financing roles."""
    return ShapeDraft(
        shape_id="test_role_axis_facts_shape",
        label="test flow",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={dcid: _role_arcs() for dcid in sv_dcids},
        slot_taxonomy={},
        declared_roles=("financingProvider", "financingRecipient"),
    )


# ---------------------------------------------------------------------------
# #664 covers a bound destination and open origin.
# ---------------------------------------------------------------------------


def _bound_destination_open_origin_result(monkeypatch) -> Materialised:
    shape = _financing_shape()
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [
            (_SV, "f1", "country/FRA", "country/KEN", [2018, 2019, 2020, 2021]),
            (_SV, "f2", "country/DEU", "country/KEN", [2018, 2019, 2020, 2021]),
        ],
        role_names={_SV: _FINANCING_ROLE_NAMES},
    )
    result = graph_confirm_resolve(
        shape=shape,
        bindings=[],
        subject_dcids=["country/KEN"],
        origin_dcids=None,
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),  # never read on this branch
        discover_scope=True,
        coverage_snapshot=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert isinstance(result, Materialised)
    return result


def test_bound_destination_open_origin_derives_open_origin_dcids(monkeypatch):
    """Two providers behind one bound recipient, no origin named: the third
    detection arm's derivation reads both provider dcids back out of the
    same coverage lookup that confirms the destination, at zero extra read
    cost."""
    result = _bound_destination_open_origin_result(monkeypatch)

    assert result.open_origin_dcids == frozenset({"country/FRA", "country/DEU"})
    assert result.sv_dcids == [_SV]


def test_bound_destination_open_origin_publishes_a_role_axis_fact(monkeypatch):
    result = _bound_destination_open_origin_result(monkeypatch)

    facts = _role_axis_facts(
        [
            RoleAxisFactInput(
                role="origin",
                member_dcids=result.open_origin_dcids,
                party_dcids=result.open_origin_dcids,
            )
        ]
    )

    assert facts == [
        RoleAxisFact(role="origin", member_count=2, members_available=True),
    ]


def test_bound_destination_open_origin_regression_nothing_else_changes(monkeypatch):
    """Destination-only confirmation returns CoverageExact without scope discovery."""
    result = _bound_destination_open_origin_result(monkeypatch)

    assert result.discovered_scope is None
    assert result.role_index_by_sv is None
    assert result.also_defaulted_axis is None
    assert isinstance(result.coverage, CoverageExact)
    assert result.sv_dcids == [_SV]
    assert result.destination_confirmed == {_SV}


def test_bound_destination_open_origin_makes_no_extra_sdmx_call(monkeypatch):
    shape = _financing_shape()
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV, "f1", "country/FRA", "country/KEN", [2018, 2019, 2020, 2021])],
        role_names={_SV: _FINANCING_ROLE_NAMES},
    )
    sdmx = FakeSdmxClient(observations={})  # never read on this branch

    result = materialise(
        shape,
        [],
        {"financingRecipient": ["country/KEN"]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=sdmx,
        discover_scope=True,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.open_origin_dcids == frozenset({"country/FRA"})
    # Confirmation and origin discovery reuse the same coverage lookup.
    assert sdmx.series_calls == []
    assert sdmx.availability_calls == []


# ---------------------------------------------------------------------------
# Both axes open
# ---------------------------------------------------------------------------


def test_both_axes_open_derives_the_same_fact_from_role_index_by_sv(monkeypatch):
    """With both axes open, coverage supplies members for each role.

    financingRecipient is discovered. Origin is also defaulted.
    """
    shape = _financing_shape()
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [
            (_SV, "f1", "country/FRA", "country/KEN", [2020]),
            (_SV, "f2", "country/USA", "country/KEN", [2020]),
        ],
        role_names={_SV: _FINANCING_ROLE_NAMES},
    )
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = graph_confirm_resolve(
        shape=shape,
        bindings=[],
        subject_dcids=[],
        origin_dcids=[],
        graph=FakeGraph(raise_on_call=True),
        sdmx=sdmx,
        discover_scope=True,
        coverage_snapshot=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.discovered_scope is not None
    assert result.discovered_scope.role_name == "financingRecipient"
    assert result.also_defaulted_axis == "origin"
    assert result.role_index_by_sv is not None

    members = open_role_axis_members(
        result.role_index_by_sv, role_name="financingProvider", svs=result.sv_dcids
    )
    assert members == frozenset({"country/FRA", "country/USA"})

    facts = _role_axis_facts(
        [RoleAxisFactInput(role="origin", member_dcids=members, party_dcids=members)]
    )
    assert facts == [RoleAxisFact(role="origin", member_count=2, members_available=True)]


# Role-axis facts adapter.


def test_role_axis_facts_none_when_empty():
    assert _role_axis_facts([]) is None


def test_role_axis_facts_reports_both_roles_when_both_open():
    facts = _role_axis_facts(
        [
            RoleAxisFactInput(
                role="destination",
                member_dcids=frozenset({"country/KEN"}),
                party_dcids=frozenset({"country/KEN"}),
            ),
            RoleAxisFactInput(role="origin", member_dcids=frozenset(), party_dcids=frozenset()),
        ]
    )
    assert facts == [
        RoleAxisFact(role="destination", member_count=1, members_available=True),
        RoleAxisFact(role="origin", member_count=0, members_available=True),
    ]
