"""Tests the no-data guard for alternate shapes when the elected shape has no observations.

A nominated alternate that is neither an equivalent reading nor a price-basis sibling of the
elected shape rides on the refusal: the part answers `no_data` with `ONLY_ALTERNATE_HAS_DATA`,
the first such alternate with data as its alternate match (status rule 1)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence

import pytest

from search.engine.bind import ShapeRating, SlotBindingDraft, _UniversalBindOutput
from search.engine.binders import Binder
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.region.codes import ELECTED_SHAPE_NO_DATA, ONLY_ALTERNATE_HAS_DATA
from search.engine.region.state import _elected_stat_vars
from search.engine.retrieval.retrieve import NoDataDraft
from search.engine.shape import ShapeDraft
from search.models import GraphRef
from series_coverage.schema import RoleNames
from tests.engine._binders import ReplyBinder, nominee_ratings
from tests.engine._fake_coverage import (
    FakeCoverageService,
    build_coverage_index,
    empty_coverage_snapshot,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_ELECTED_SV = "Count_Person_HighAltitude"
_STD_SV = "HealthcareExpenditureAsFractionOfGDP"
_STD_SV_2 = "Count_MedicalDoctor_PerCapita"


def _std_arcs(label: str, *, pop_type: str, meas_prop: str, denom: str | None = None) -> dict:
    # A plain standard SV with no constraint properties at all.
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": pop_type}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if denom:
        arcs["measurementDenominator"] = {"nodes": [{"dcid": denom}]}
    return arcs


_KEN_NODE = {"country/KEN": {"label": "Kenya", "type": "Country"}}
_ELECTED_NODE = {
    _ELECTED_SV: {
        "label": "Count of people at high altitude",
        "type": "StatisticalVariable",
        "arcs": _std_arcs(
            "Count of people at high altitude", pop_type="Person", meas_prop="countHighAltitude"
        ),
    },
}


class _GuardBindLLM:
    """Elects the data-empty shape, binds `bindings` on it, marks the standard
    alternatives as plausible and rates its pick and each shape in `rated` at 1."""

    def __init__(
        self,
        also_plausible: list[int],
        *,
        chosen_shape_index: int,
        unplaced: Sequence[str] = (),
        bindings: Sequence[SlotBindingDraft] = (),
        rated: Sequence[int] = (),
    ):
        self._also_plausible = also_plausible
        self._rated = list(rated)
        self._chosen_shape_index = chosen_shape_index
        self._unplaced = list(unplaced)
        self._bindings = list(bindings)

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=self._chosen_shape_index,
            bindings=self._bindings,
            also_plausible=self._also_plausible,
            ask=None,
            unplaced=self._unplaced,
            measuring=[
                *(
                    ShapeRating(index=i, rating=1.0)
                    for i in dict.fromkeys([self._chosen_shape_index, *self._rated])
                ),
                *nominee_ratings(
                    self._also_plausible, rated=(self._chosen_shape_index, *self._rated)
                ),
            ],
        ), None


def _run(
    *,
    binder: Binder | None = None,
    nodes: dict,
    detect: dict,
    obs: dict,
    also_plausible: list[int],
    chosen_shape_index: int,
    unplaced: list[str] | None = None,
):
    graph = FakeGraph(nodes=nodes, detect=detect, resolve={"Kenya": "country/KEN"}, obs=obs)
    return asyncio.run(
        resolve_variable(
            "indicators in Kenya",
            entities=["Kenya"],
            date_request=None,
            detect_query="indicators in Kenya",
            role_query="indicators in Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            binder=binder,
            llm=_GuardBindLLM(
                also_plausible, chosen_shape_index=chosen_shape_index, unplaced=unplaced or ()
            ),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


# ---------------------------------------------------------------------------
# One alternate misses CandidateSet's minimum size and yields no_data.
# ---------------------------------------------------------------------------

_ONE_ALT_NODES: dict = {
    **_KEN_NODE,
    **_ELECTED_NODE,
    _STD_SV: {
        "label": "Healthcare expenditure as fraction of GDP",
        "type": "StatisticalVariable",
        "arcs": _std_arcs(
            "Healthcare expenditure as fraction of GDP",
            pop_type="EconomicActivity",
            meas_prop="amount",
            denom="GrossDomesticProduction",
        ),
    },
}
# Standard SV ranks first (higher retrieval score) -- the model must elect
# the data-empty shape despite this.
_ONE_ALT_DETECT = {
    "indicators in Kenya": {
        "svs": [_STD_SV, _ELECTED_SV],
        "cosine_scores": [0.85, 0.82],
    },
}
_ONE_ALT_OBS = {
    # The standard alt shape has real data at Kenya.
    f"{_STD_SV}|country/KEN": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
    # Deliberately NO entry for the elected SV -- confirmed structurally
    # (graph.node_arcs) but zero real observations, the "honest limitation"
    # graph_confirm_resolve's own no_observations relaxation path carries.
}


def _run_one_alt_scenario(*, unplaced: list[str] | None = None):
    # 2 shapes ranked by score: STD_SV(0.85) index 0, ELECTED_SV(0.82) index 1.
    return _run(
        nodes=_ONE_ALT_NODES,
        detect=_ONE_ALT_DETECT,
        obs=_ONE_ALT_OBS,
        also_plausible=[0],
        chosen_shape_index=1,
        unplaced=unplaced,
    )


def test_one_alt_survivor_is_no_data_not_definite():
    result = _run_one_alt_scenario()
    assert result.status == "no_data", (
        f"expected no_data (a lone non-elected alt can't fill CandidateSet's "
        f"min_length=2, and must never silently become definite), got {result.status!r}"
    )


def test_one_alt_survivor_carries_no_diagnostics_warning():
    """The no-data result carries the gate without a diagnostics warning."""
    result = _run_one_alt_scenario()
    assert not any(w.code == ONLY_ALTERNATE_HAS_DATA for w in result.warnings), (
        f"expected no ONLY_ALTERNATE_HAS_DATA warning, got {[w.code for w in result.warnings]}"
    )


def test_one_alt_survivor_carried_as_alternate_match():
    """The plausible alternative is exposed without a Spec for the data-empty elected shape."""
    result = _run_one_alt_scenario()
    assert result.specs == ()
    assert result.flagged_match is None
    assert result.alternate_match is not None
    sv_dcids = [sv.ref.ref for sv in result.alternate_match.resolved.members]
    assert sv_dcids == [_STD_SV]


def test_one_alt_survivor_carries_only_alternate_has_data_gate():
    """Only the alternate has data, so the gate is `ONLY_ALTERNATE_HAS_DATA`."""
    result = _run_one_alt_scenario()
    assert result.gate == ONLY_ALTERNATE_HAS_DATA


@pytest.mark.parametrize(
    "unplaced", [[], ["indicators"]], ids=["every-word-placed", "unplaced-word"]
)
def test_one_alt_survivor_departs_each_word_its_trace_leaves_unmet(unplaced):
    """The refusal names no term, so each word the alternate's trace leaves unmet departs
    once, judged on the alternate itself."""
    result = _run_one_alt_scenario(unplaced=unplaced)
    assert result.gate == ONLY_ALTERNATE_HAS_DATA
    assert result.alternate_match is not None
    trace = result.alternate_match.resolved.trace.query_qualifiers
    assert [q.text for q in trace] == unplaced
    assert [(f.code, f.asked) for f in result.departure_facts.values()] == [
        ("measure_not_applied", word) for word in unplaced
    ]


def test_one_alt_survivor_elected_stat_vars_names_the_elected_shape():
    """`elected_stat_vars` names the structurally confirmed, data-empty elected shape."""
    result = _run_one_alt_scenario()
    assert result.elected_stat_vars == [
        GraphRef(ref=_ELECTED_SV, label="Count of people at high altitude")
    ]


def test_elected_stat_vars_none_when_mat_result_is_no_data_draft():
    """A role-incompatible `NoDataDraft` has no graph-confirmed identity to publish."""
    shape = ShapeDraft(
        shape_id="shape_placeholder",
        label="shape placeholder",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={_ELECTED_SV: _ELECTED_NODE[_ELECTED_SV]["arcs"]},
    )
    assert _elected_stat_vars(NoDataDraft(reason="role_incompatible"), shape) is None


# ---------------------------------------------------------------------------
# Role-bearing elected shapes also publish their identity on the bare no-data path.
# ---------------------------------------------------------------------------

_ROLE_SV = "TESTFIXTURE/OnlyAlternateHasDataGuard/RoleBearingFlow"
# The subject-only alternate shares population and stat type with _ROLE_SV and divides by
# another base, so it is no equivalent reading of it (status rule 1). The query names a
# share, which both honour, so the elected flow passes the materialise quantity gate and the
# guard admits the alternate beside it.
_AID_SV = "AidReceivedAsFractionOfGDP"
_ROLE_NODE = {
    _ROLE_SV: {
        "label": "aid flow",
        "type": "StatisticalVariable",
        "arcs": {
            **_std_arcs(
                "aid flow",
                pop_type="EconomicActivity",
                meas_prop="amount",
                denom="GrossDomesticProduction",
            ),
            "observationProperties": {"nodes": [{"dcid": "creditor"}, {"dcid": "debtor"}]},
        },
    },
    _AID_SV: {
        "label": "Aid received as a fraction of GNI",
        "type": "StatisticalVariable",
        "arcs": _std_arcs(
            "Aid received as a fraction of GNI",
            pop_type="EconomicActivity",
            meas_prop="amount",
            denom="GrossNationalIncome",
        ),
    },
}
_ROLE_GUARD_QUERY = "aid to Kenya as a share of GDP"
_ROLE_GUARD_DETECT = {
    _ROLE_GUARD_QUERY: {
        "svs": [_AID_SV, _ROLE_SV],
        "cosine_scores": [0.85, 0.82],
    },
}
_ROLE_GUARD_OBS = {
    f"{_AID_SV}|country/KEN": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
}


_DENOMINATOR_NOUNS = {"GrossNationalIncome": "GNI", "GrossDomesticProduction": "GDP"}


def _run_role_bearing_one_alt_scenario(
    aid_denominator: str = "GrossNationalIncome", *, rated: Sequence[int] = ()
):
    # AID_SV ranks first at 0.85 and ROLE_SV second at 0.82.
    # ROLE_SV's creditor/debtor roles bind Kenya on the destination axis.
    graph = FakeGraph(
        nodes={
            **_KEN_NODE,
            **_ROLE_NODE,
            _AID_SV: {
                **_ROLE_NODE[_AID_SV],
                "arcs": _std_arcs(
                    f"Aid received as a fraction of {_DENOMINATOR_NOUNS[aid_denominator]}",
                    pop_type="EconomicActivity",
                    meas_prop="amount",
                    denom=aid_denominator,
                ),
            },
        },
        detect=_ROLE_GUARD_DETECT,
        resolve={"Kenya": "country/KEN"},
        obs=_ROLE_GUARD_OBS,
    )
    return asyncio.run(
        resolve_variable(
            _ROLE_GUARD_QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_ROLE_GUARD_QUERY,
            role_query=_ROLE_GUARD_QUERY,
            entity_roles=door_roles_for_test_query(_ROLE_GUARD_QUERY, ["Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_GuardBindLLM([0], chosen_shape_index=1, rated=rated),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}),
            coverage=empty_coverage_snapshot(),
        )
    )


def test_role_bearing_elected_shape_no_observations_elects_stat_vars():
    """The role-bearing no-data path names its elected StatVar after bound-constraint confirmation."""
    result = _run_role_bearing_one_alt_scenario()
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert result.flagged_match is None
    assert result.alternate_match is not None
    assert result.gate == ONLY_ALTERNATE_HAS_DATA
    assert result.elected_stat_vars == [GraphRef(ref=_ROLE_SV, label="aid flow")]


def test_the_house_policy_elects_the_subject_only_twin_with_data(caplog):
    """On a named place, the subject-only twin with data shares the elected flow's reading
    and the reply rates both alike, so the data-at-places policy elects it in the bind
    call. The part answers it as definite, and no elected shape lacked data."""
    caplog.set_level(logging.INFO, logger="search.engine.region.phase_rank_bind")

    result = _run_role_bearing_one_alt_scenario(
        aid_denominator="GrossDomesticProduction", rated=(0, 1)
    )

    (decision,) = [
        json.loads(r.getMessage().removeprefix("bind_decision "))
        for r in caplog.records
        if r.getMessage().startswith("bind_decision ")
    ]
    assert (decision["chosen"], decision["elected"], decision["answered"]) == (1, 0, 0)
    assert decision["fired"] == ["data_at_places"]
    assert result.status == "definite"
    assert [sv.ref.ref for sv in result.specs[0].resolved.members] == [_AID_SV]
    assert ELECTED_SHAPE_NO_DATA not in [w.code for w in result.warnings]


_GRANT_SV = "TESTFIXTURE/OnlyAlternateHasDataGuard/AidGrant"
_LOAN_SV = "TESTFIXTURE/OnlyAlternateHasDataGuard/AidLoan"
_GRANT_FLOW_SV = "TESTFIXTURE/OnlyAlternateHasDataGuard/AidGrantFlow"
_LOAN_FLOW_SV = "TESTFIXTURE/OnlyAlternateHasDataGuard/AidLoanFlow"
_GRANT_QUERY = "grant aid to Kenya"
_FLOW_ROLES = RoleNames(donor="creditor", recipient="debtor")


def _flow_kind_node(label: str, kind: str, *, role_bearing: bool) -> dict:
    arcs = {
        **_std_arcs(label, pop_type="EconomicActivity", meas_prop="amount"),
        "constraintProperties": {"nodes": [{"dcid": "flowKind"}]},
        "flowKind": {"nodes": [{"dcid": kind}]},
    }
    if role_bearing:
        arcs["observationProperties"] = {"nodes": [{"dcid": "creditor"}, {"dcid": "debtor"}]}
    return {"label": label, "type": "StatisticalVariable", "arcs": arcs}


def test_a_flow_twin_answers_when_the_bound_member_has_no_data():
    """Status rule 1 applies past the house policies. Menu availability reads each shape before
    bind, so the subject-only aid shape has data at Kenya (its loan member does) beside its
    by-role twin, and the data-at-places policy keeps the binder's pick. The grant the bind
    draws has no data, so the elected shape holds none, and the twin on the same five-tuple,
    whose grant flow has data, answers in its place."""
    graph = FakeGraph(
        nodes={
            **_KEN_NODE,
            _GRANT_SV: _flow_kind_node("grant aid", "Grant", role_bearing=False),
            _LOAN_SV: _flow_kind_node("loan aid", "Loan", role_bearing=False),
            _GRANT_FLOW_SV: _flow_kind_node("grant aid flow", "Grant", role_bearing=True),
            _LOAN_FLOW_SV: _flow_kind_node("loan aid flow", "Loan", role_bearing=True),
            "Grant": {"label": "Grant", "type": "FlowKind"},
            "Loan": {"label": "Loan", "type": "FlowKind"},
        },
        detect={
            _GRANT_QUERY: {
                "svs": [_GRANT_SV, _LOAN_SV, _GRANT_FLOW_SV, _LOAN_FLOW_SV],
                "cosine_scores": [0.85, 0.84, 0.83, 0.82],
            }
        },
        resolve={"Kenya": "country/KEN"},
        obs={
            f"{_LOAN_SV}|country/KEN": [
                {"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}
            ]
        },
    )
    coverage = build_coverage_index(
        [(_GRANT_FLOW_SV, "facet1", "country/USA", "country/KEN", [2015, 2016])],
        role_names={_GRANT_FLOW_SV: _FLOW_ROLES, _LOAN_FLOW_SV: _FLOW_ROLES},
    )
    grant = SlotBindingDraft(
        axis="how", property_dcid="flowKind", kind="value", value_dcids=["Grant"]
    )
    result = asyncio.run(
        resolve_variable(
            _GRANT_QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_GRANT_QUERY,
            role_query=_GRANT_QUERY,
            entity_roles=door_roles_for_test_query(_GRANT_QUERY, ["Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_GuardBindLLM([], chosen_shape_index=0, bindings=[grant]),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=FakeCoverageService(coverage).snapshot(),
        )
    )
    assert result.status == "definite"
    assert [sv.ref.ref for sv in result.specs[0].resolved.members] == [_GRANT_FLOW_SV]
    assert ELECTED_SHAPE_NO_DATA in [w.code for w in result.warnings]


# ---------------------------------------------------------------------------
# Two nominated alternates with data beside an empty election.
# ---------------------------------------------------------------------------

_TWO_ALT_NODES: dict = {
    **_KEN_NODE,
    **_ELECTED_NODE,
    _STD_SV: {
        "label": "Healthcare expenditure as fraction of GDP",
        "type": "StatisticalVariable",
        "arcs": _std_arcs(
            "Healthcare expenditure as fraction of GDP",
            pop_type="EconomicActivity",
            meas_prop="amount",
            denom="GrossDomesticProduction",
        ),
    },
    # Distinct five-tuples need different populationType/measuredProperty values.
    # Shared five-tuples deduplicate to one shape.
    _STD_SV_2: {
        "label": "Physicians per 1000 people",
        "type": "StatisticalVariable",
        "arcs": _std_arcs(
            "Physicians per 1000 people",
            pop_type="MedicalDoctor",
            meas_prop="count",
        ),
    },
}
_TWO_ALT_DETECT = {
    "indicators in Kenya": {
        "svs": [_STD_SV, _STD_SV_2, _ELECTED_SV],
        "cosine_scores": [0.86, 0.85, 0.82],
    },
}
_TWO_ALT_OBS = {
    f"{_STD_SV}|country/KEN": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
    f"{_STD_SV_2}|country/KEN": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
}


def _run_two_alt_scenario(*, binder: Binder | None = None):
    # 3 shapes ranked by score: STD_SV(0.86) index 0, STD_SV_2(0.85) index 1,
    # ELECTED_SV(0.82) index 2.
    return _run(
        nodes=_TWO_ALT_NODES,
        detect=_TWO_ALT_DETECT,
        obs=_TWO_ALT_OBS,
        also_plausible=[0, 1],
        chosen_shape_index=2,
        binder=binder,
    )


def test_two_alt_survivors_leave_the_part_refused_with_the_first_riding():
    """Status rule 1: neither nominated reading is an equivalent or a price-basis sibling
    of the empty elected shape, so the part refuses, and the first nominated reading with
    data rides on the refusal as its alternate match."""
    result = _run_two_alt_scenario(binder=ReplyBinder())
    assert (result.status, result.no_data_reason) == ("no_data", "no_observations")
    assert result.gate == ONLY_ALTERNATE_HAS_DATA
    assert result.alternate_match is not None
    assert [sv.ref.ref for sv in result.alternate_match.resolved.members] == [_STD_SV]
    assert ELECTED_SHAPE_NO_DATA not in [w.code for w in result.warnings]
