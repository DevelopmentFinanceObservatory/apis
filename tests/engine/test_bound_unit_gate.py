"""Unit tests for the shared bound-unit evidence and the gate verdict, both read from the
answer's read plan alone. Pipeline tests cover the call sites."""

from __future__ import annotations

import pytest

from search.engine.clients.graph import Facet
from search.engine.region.bound_unit_gate import (
    UnitEvidence,
    bound_unit_gate_reason,
    plan_unit_evidence,
)
from search.engine.retrieval.read_plan import PairPlan, PlanSeries

_KEN = {"observationAbout": ["country/KEN"]}


def _facet(facet_id: str) -> Facet:
    return Facet(earliest_date="2020", latest_date="2020", obs_count=1, facet_id=facet_id, unit="X")


def _series(stat_var: str, *, served: bool) -> PlanSeries:
    facet = _facet(f"f-{stat_var}")
    return PlanSeries(
        stat_var=stat_var,
        role_tuple=(("observationAbout", "country/KEN"),),
        candidates=(facet,),
        group=(facet,) if served else (),
    )


def _plan(*, scope: bool = False) -> PairPlan:
    """A and B each serve a series at Kenya; C's series holds no facet of its group."""
    series = (_series("A", served=True), _series("B", served=True), _series("C", served=False))
    return PairPlan(members=("A", "B", "C"), years=None, series=series, scope=scope)


def test_unit_evidence_serves_when_one_served_statvar_is_confirmed():
    evidence = UnitEvidence(confirmed=frozenset({"A"}))
    assert evidence.serves_any(["A"]) is True
    assert evidence.serves_any(["A", "B"]) is True
    assert evidence.serves_any(["B"]) is False
    assert evidence.serves_any([]) is False


@pytest.mark.parametrize("scope", [False, True], ids=["per-series", "scope"])
def test_a_plan_confirms_the_statvars_it_serves(scope: bool) -> None:
    assert plan_unit_evidence(_plan(scope=scope)).confirmed == {"A", "B"}


def test_an_answer_without_a_plan_confirms_nothing():
    assert plan_unit_evidence(None).confirmed == frozenset()


@pytest.mark.parametrize(
    ("plan", "entities", "stat_vars", "reason"),
    [
        pytest.param(_plan(), _KEN, ["A", "B"], None, id="every-statvar-served"),
        # A group serves while one of its served members holds a point, as the mint serves it.
        pytest.param(_plan(), _KEN, ["A", "C"], None, id="one-served-statvar-confirmed"),
        pytest.param(_plan(), _KEN, ["C"], "facet_not_available", id="no-served-statvar-confirmed"),
        pytest.param(_plan(), {}, ["A"], "facet_not_available", id="no-entities"),
        pytest.param(None, _KEN, ["A"], "facet_not_available", id="no-plan"),
        # A plan the elected units narrowed to nothing holds no series in the requested unit.
        pytest.param(
            PairPlan(members=("C",), years=None, series=(_series("C", served=False),)),
            _KEN,
            ["C"],
            "facet_not_available",
            id="plan-serves-nothing",
        ),
    ],
)
def test_gate_reason(
    plan: PairPlan | None, entities: dict[str, list[str]], stat_vars: list[str], reason: str | None
) -> None:
    assert bound_unit_gate_reason(plan, entities, stat_vars) == reason
