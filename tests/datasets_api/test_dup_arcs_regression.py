"""Regression tests for #698 pin deduplication of raw `/v2/node` arcs before
`resolve_shape` reads constraint values and derives StatVar shapes.
"""

from __future__ import annotations

import pytest

from datasets_api.codes_materialise import MemberNotResolvable
from datasets_api.shape import resolve_shape
from materialise.graph import parse_arcs
from tests.materialise._dup_arcs_fixtures import (
    HEALTH_SPENDING_CONTROL_VALUES,
    LIFE_EXPECTANCY_FEMALE_VALUES,
    MORTALITY_INFANT_VALUES,
    doubled_wire_arcs,
)

from ._graph_doubles import FakeMintGraphReader

_HEALTH_SPENDING_DCID = (
    "Amount_EconomicActivity_ExpenditureActivity_HealthcareExpenditure_AsFractionOf_Count_Person"
)
_MORTALITY_DCID = "Count_MortalityEvent_Infant_AsAFractionOf_Count_BirthEvent_LiveBirth"


def test_dual_provenance_constraint_value_resolves_matching_single_provenance_shape():
    # #698 duplicates the same `gender` value across two provenances. Parsing must collapse it.
    parsed = parse_arcs(
        doubled_wire_arcs(LIFE_EXPECTANCY_FEMALE_VALUES), tuple(LIFE_EXPECTANCY_FEMALE_VALUES)
    )
    doubled_graph = FakeMintGraphReader(arcs={"LifeExpectancy_Person_Female": parsed})
    single_provenance_graph = FakeMintGraphReader(
        arcs={"LifeExpectancy_Person_Female": LIFE_EXPECTANCY_FEMALE_VALUES}
    )

    shape = resolve_shape("dcid:LifeExpectancy_Person_Female", graph=doubled_graph)

    assert shape == resolve_shape(
        "dcid:LifeExpectancy_Person_Female", graph=single_provenance_graph
    )
    assert shape.slots == (("gender", "Female"),)
    assert shape.measured_property == "lifeExpectancy"
    assert shape.population_type == "Person"


def test_dual_provenance_constraint_properties_dedup_into_slots_with_no_duplicate_entries():
    # #698 duplicates `ageGroupClassification` in `constraintProperties`. `slots` must stay unique.
    parsed = parse_arcs(doubled_wire_arcs(MORTALITY_INFANT_VALUES), tuple(MORTALITY_INFANT_VALUES))
    doubled_graph = FakeMintGraphReader(arcs={_MORTALITY_DCID: parsed})
    single_provenance_graph = FakeMintGraphReader(arcs={_MORTALITY_DCID: MORTALITY_INFANT_VALUES})

    shape = resolve_shape(f"dcid:{_MORTALITY_DCID}", graph=doubled_graph)

    assert shape.slots == (("ageGroupClassification", "Infant"),)
    assert shape == resolve_shape(f"dcid:{_MORTALITY_DCID}", graph=single_provenance_graph)


def test_dual_provenance_genuinely_conflicting_constraint_value_still_refuses():
    # Distinct values form a genuine conflict.
    arcs = {
        "gender": {
            "nodes": [
                {"dcid": "Male", "provenanceId": "dc/base/India_LifeExpectancy"},
                {"dcid": "Female", "provenanceId": "dc/base/Mongolia_Health"},
            ]
        }
    }
    parsed_gender = parse_arcs(arcs, ["gender"])
    graph = FakeMintGraphReader(
        arcs={"LifeExpectancy_Person_Ambiguous": {**LIFE_EXPECTANCY_FEMALE_VALUES, **parsed_gender}}
    )

    with pytest.raises(MemberNotResolvable) as excinfo:
        resolve_shape("dcid:LifeExpectancy_Person_Ambiguous", graph=graph)

    assert excinfo.value.members == ("dcid:LifeExpectancy_Person_Ambiguous",)
    assert "gender" in str(excinfo.value.__cause__)


def test_single_provenance_health_spending_control_resolves_unaffected():
    # #698 single-provenance control with one entry per arc.
    graph = FakeMintGraphReader(arcs={_HEALTH_SPENDING_DCID: HEALTH_SPENDING_CONTROL_VALUES})

    shape = resolve_shape(f"dcid:{_HEALTH_SPENDING_DCID}", graph=graph)

    assert shape.measured_property == "amount"
    assert shape.population_type == "EconomicActivity"
    assert shape.slots == (
        ("activitySource", "ExpenditureActivity"),
        ("expenditureType", "HealthcareExpenditure"),
    )
