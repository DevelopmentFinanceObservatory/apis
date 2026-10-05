"""Shared facet fixtures for observation, option, and facet-filter tests."""

from __future__ import annotations

from tests.engine._harness import materialise_double_from_obs
from tests.fixtures import FakeGraph

_MULTI_AXIS_SV = "Count_Person"
_PER_SERIES_SV_A = "Count_Person_Male"
_PER_SERIES_SV_B = "Count_Person_Female"
_SINGLE_FACET_SV = "Count_Person_Urban"
_TWO_ENTITY_SV = "Count_Person_Rural"

# unit -> a base in-window point count. Combined with the per-measurementMethod
# bonus below, every (unit, measurementMethod) combination gets its own
# distinct point count, so ranking/tie-break logic always has real signal to
# sort on.
_UNIT_BASE_POINTS: dict[str, int] = {
    "USD": 6,
    "EUR": 5,
    "GBP": 4,
}
# GrossDisbursement always beats Commitment always beats GrantEquivalent, for
# every unit -- so the single richest (unit, method) combination (USD +
# GrossDisbursement) is the unconstrained winner across the whole 9-facet
# group.
_METHOD_BONUS: dict[str, int] = {"GrossDisbursement": 2, "Commitment": 1, "GrantEquivalent": 0}


def _multi_axis_group() -> list[dict]:
    """9 candidate facets for one (stat_var, entity) pair: every
    measurementMethod x unit combination in _UNIT_BASE_POINTS."""
    facets = []
    for unit, base in _UNIT_BASE_POINTS.items():
        for method, bonus in _METHOD_BONUS.items():
            n = base + bonus
            dates = [str(2000 + i) for i in range(n)]
            facets.append(
                {
                    "earliestDate": dates[0],
                    "latestDate": dates[-1],
                    "obsCount": n,
                    "provenanceId": f"dc/base/{method}",
                    "measurementMethod": method,
                    "unit": unit,
                    "observations": [{"date": d, "value": float(i)} for i, d in enumerate(dates)],
                }
            )
    return facets


_FACET_AXIS_OBS: dict = {
    f"{_MULTI_AXIS_SV}|country/USA": _multi_axis_group(),
    f"{_PER_SERIES_SV_A}|country/USA": [
        {
            "earliestDate": "2020",
            "latestDate": "2022",
            "obsCount": 3,
            "provenanceId": "dc/base/testA",
            "measurementMethod": "GrossDisbursement",
            "observations": [
                {"date": "2020", "value": 1.0},
                {"date": "2021", "value": 2.0},
                {"date": "2022", "value": 3.0},
            ],
        }
    ],
    f"{_PER_SERIES_SV_B}|country/USA": [
        {
            "earliestDate": "2020",
            "latestDate": "2021",
            "obsCount": 2,
            "provenanceId": "dc/base/testB",
            "measurementMethod": "Commitment",
            "observations": [
                {"date": "2020", "value": 1.0},
                {"date": "2021", "value": 2.0},
            ],
        }
    ],
    # This facet has no provenance for option-axis tests; a variant below supports materialise.
    f"{_SINGLE_FACET_SV}|country/USA": [
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "measurementMethod": "Commitment",
            "observations": [{"date": "2020", "value": 1.0}],
        }
    ],
    f"{_TWO_ENTITY_SV}|country/USA": [
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/Src1",
            "measurementMethod": "Commitment",
            "unit": "EUR",
            "observations": [{"date": "2020", "value": 1.0}],
        }
    ],
    f"{_TWO_ENTITY_SV}|country/GBR": [
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/Src2",
            "measurementMethod": "GrossDisbursement",
            "unit": "USD",
            "observations": [{"date": "2020", "value": 1.0}],
        }
    ],
}

# Materialise preview requires a provenance facet; this variant adds one to the option-test fixture.
_MATERIALISE_SINGLE_FACET_OBS: dict = {
    f"{_SINGLE_FACET_SV}|country/USA": [
        {**_FACET_AXIS_OBS[f"{_SINGLE_FACET_SV}|country/USA"][0], "provenanceId": "dc/base/testC"}
    ],
}


_FACET_AXIS_NODES: dict = {
    "dc/base/GrossDisbursement": {"label": "Gross Disbursement reporting"},
    "dc/base/Commitment": {"label": "Commitment reporting"},
    "dc/base/GrantEquivalent": {"label": "Grant Equivalent reporting"},
    "dc/base/Src1": {"label": "Source 1"},
    "dc/base/Src2": {"label": "Source 2"},
}


def _facet_axis_graph() -> FakeGraph:
    return FakeGraph(nodes=_FACET_AXIS_NODES, obs=_FACET_AXIS_OBS, detect={}, resolve={})


def _facet_axis_materialise_double(obs: dict = _FACET_AXIS_OBS):
    return materialise_double_from_obs(obs)
