"""A series counts when its measured property is `count` or every facet unit at the place is a
count unit.

SDG and UN System series measure `value`, so live "measles incidence rate Kenya" kept the WHO
reported-cases count (`undata/UNIT_MEASURE-COUNT_CASES`) past the explicit-rate prune and
elected it.
"""

from __future__ import annotations

import pytest

from search.engine.quantity import quantity_requirement
from search.engine.region.alternate_guard import inadmissible_alternates, query_pins
from search.engine.region.phase_rank_bind import (
    _is_unnormalised_raw_count,
    _prune_explicit_rate_raw_counts,
)
from search.engine.retrieval.derive import derive_shapes
from search.engine.shape import ShapeDraft, measures_count
from search.engine.unit.count import is_count_unit
from tests.engine._discover_fixtures import _std_sv_arcs
from tests.fixtures import FakeGraph

_KENYA = "country/KEN"
_CASES = "undata/UNIT_MEASURE-COUNT_CASES"
_MORTALITY_RATE = "undata/UNIT_MEASURE-RATE_MORTALITY_PER_1000_COUNT_LIVEBIRTHS"


@pytest.mark.parametrize(
    "unit",
    [
        "undata/UNIT_MEASURE-COUNT",
        _CASES,
        "undata/UNIT_MEASURE-COUNT_DEATHS",
        "undata/UNIT_MEASURE-COUNT_PERSONS_F",
        "SDG_NUMBER",
        "SDG_NUM_TH",
        "SDG_NUM_M",
        "PeopleInThousands",
    ],
)
def test_count_units(unit: str) -> None:
    assert is_count_unit(unit)


@pytest.mark.parametrize(
    "unit",
    [
        None,
        _MORTALITY_RATE,
        "undata/UNIT_MEASURE-RATIO_COUNT_CASES_PER_COUNT_1000_COUNT_POP",
        "undata/UNIT_MEASURE-AVG_COUNT_PERSONS",
        "undata/UNIT_MEASURE-PT_POP",
        "undata/UNIT_MEASURE-COUNTRY",
        "SDG_PERCENT",
        "SDG_PER_100000_POP",
        "Percent",
        "USDollar",
    ],
)
def test_non_count_units(unit: str | None) -> None:
    assert not is_count_unit(unit)


def _value_shape(
    label: str,
    *,
    units: frozenset[str | None] | None,
    meas_prop: str = "value",
    shape_id: str | None = None,
) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id or label,
        label=label,
        pop_type_dcid=f"undata/who/SERIES-{label}",
        meas_prop_dcid=meas_prop,
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        declared_roles=("observationAbout",),
        place_units=units,
    )


class TestMeasuresCount:
    def test_count_property_counts_without_units(self) -> None:
        assert measures_count(_value_shape("people", units=None, meas_prop="count"))

    def test_value_series_with_count_units_counts(self) -> None:
        assert measures_count(_value_shape("cases", units=frozenset({_CASES})))

    def test_value_series_with_rate_units_does_not_count(self) -> None:
        assert not measures_count(_value_shape("rate", units=frozenset({_MORTALITY_RATE})))

    def test_one_non_count_facet_is_not_evidence(self) -> None:
        assert not measures_count(_value_shape("mixed", units=frozenset({_CASES, None})))

    def test_value_series_without_a_probe_does_not_count(self) -> None:
        assert not measures_count(_value_shape("unprobed", units=None))
        assert not measures_count(_value_shape("empty", units=frozenset()))


def _derive(place_dcids: list[str] | None) -> list[ShapeDraft]:
    cases, rate = "undata/who/CASES", "undata/who/RATE"
    graph = FakeGraph(
        nodes={
            cases: {"arcs": _std_sv_arcs("undata/who/SERIES-CASES", "value", name="cases")},
            rate: {"arcs": _std_sv_arcs("undata/who/SERIES-RATE", "value", name="rate")},
        },
        obs={
            f"{cases}|{_KENYA}": [{"obsCount": 50, "unit": _CASES}],
            f"{rate}|{_KENYA}": [{"obsCount": 8, "unit": _MORTALITY_RATE}],
        },
        detect={},
        resolve={},
    )
    return derive_shapes(
        confirmed_svs=[cases, rate],
        graph=graph,
        sv_scores={cases: 0.7411, rate: 0.7141},
        place_dcids=place_dcids,
    )


class TestDerivedShapesCarryPlaceUnits:
    def test_probe_stamps_each_shape_with_its_facet_units(self) -> None:
        units = {s.label: s.place_units for s in _derive([_KENYA])}
        assert units == {"cases": frozenset({_CASES}), "rate": frozenset({_MORTALITY_RATE})}

    def test_no_place_leaves_units_unknown(self) -> None:
        assert all(s.place_units is None for s in _derive(None))

    def test_the_reported_cases_series_is_a_raw_count(self) -> None:
        by_label = {s.label: s for s in _derive([_KENYA])}
        assert _is_unnormalised_raw_count(by_label["cases"])
        assert not _is_unnormalised_raw_count(by_label["rate"])


_REPORTED_CASES = _value_shape(
    "Number of reported cases of vaccine-preventable diseases [Measles]",
    units=frozenset({_CASES}),
)
_MORTALITY = _value_shape(
    "Cause-specific mortality rate among children [Measles]",
    units=frozenset({_MORTALITY_RATE}),
)


def test_explicit_rate_prunes_a_count_unit_series() -> None:
    kept = _prune_explicit_rate_raw_counts([_REPORTED_CASES, _MORTALITY], "measles incidence rate")
    assert kept == [_MORTALITY]


def test_rate_query_rejects_a_count_unit_alternate() -> None:
    rejected = inadmissible_alternates(
        query_pins(
            "measles incidence rate", quantity=quantity_requirement(["measles incidence rate"])
        ),
        _MORTALITY,
        [(1, _REPORTED_CASES)],
    )
    assert rejected[1].startswith("pin:normalisation:")
