"""Raw `/v2/node` fixtures for duplicate-provenance tests.

The India life-expectancy and Schema provenances return duplicate arc values.
The constants below hold recorded StatVar arcs; `doubled_wire_arcs` emits
one wire entry per provenance for selected values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

DUAL_PROVENANCES = ("dc/base/India_LifeExpectancy", "dc/base/Schema")


def doubled_wire_arcs(
    values_by_property: Mapping[str, Sequence[str]],
    *,
    provenances: tuple[str, str] = DUAL_PROVENANCES,
) -> dict[str, dict]:
    """Build `/v2/node` arcs with each value repeated for every provenance."""
    return {
        prop: {
            "nodes": [
                {"dcid": value, "provenanceId": provenance}
                for value in values
                for provenance in provenances
            ]
        }
        for prop, values in values_by_property.items()
    }


LIFE_EXPECTANCY_FEMALE_VALUES: dict[str, tuple[str, ...]] = {
    "typeOf": ("StatisticalVariable",),
    "measuredProperty": ("lifeExpectancy",),
    "populationType": ("Person",),
    "statType": ("measuredValue",),
    "measurementQualifier": (),
    "measurementDenominator": (),
    "constraintProperties": ("gender",),
    "observationProperties": (),
    "gender": ("Female",),
}

LIFE_EXPECTANCY_MALE_VALUES: dict[str, tuple[str, ...]] = {
    **LIFE_EXPECTANCY_FEMALE_VALUES,
    "gender": ("Male",),
}

MORTALITY_INFANT_VALUES: dict[str, tuple[str, ...]] = {
    "typeOf": ("StatisticalVariable",),
    "measuredProperty": ("count",),
    "populationType": ("MortalityEvent",),
    "statType": ("measuredValue",),
    "measurementQualifier": (),
    "constraintProperties": ("ageGroupClassification",),
    "observationProperties": (),
    "ageGroupClassification": ("Infant",),
    "measurementDenominator": ("Count_BirthEvent_LiveBirth",),
}

# Single-provenance control (`dc/base/Schema` only): every shape property
# reports one entry, never doubled, so the shape must resolve identically
# whether or not the dedup this module's fixtures exercise ever runs.
HEALTH_SPENDING_CONTROL_VALUES: dict[str, tuple[str, ...]] = {
    "typeOf": ("StatisticalVariable",),
    "measuredProperty": ("amount",),
    "populationType": ("EconomicActivity",),
    "statType": ("measuredValue",),
    "measurementQualifier": (),
    "constraintProperties": ("activitySource", "expenditureType"),
    "observationProperties": (),
    "activitySource": ("ExpenditureActivity",),
    "expenditureType": ("HealthcareExpenditure",),
    "measurementDenominator": ("Count_Person",),
}
