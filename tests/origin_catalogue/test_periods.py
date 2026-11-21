"""The last complete year of each family of a DFO provenance's data and the sources behind a
frame's rows (`origin_catalogue.periods`), from the bound release's family counts
(`graph_release.origin.ReleaseOrigin.complete_years`) and the registry's declared years.

The year counts scale down three families: CRS flows, whose preliminary final year holds one
series in twenty, CRS grant equivalents, complete through the same year, and DAC1 ODA as a
share of GNI, whose final year holds nine in ten.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import pytest

import origin_catalogue.periods as periods_module
from graph_release.context import bound_release
from graph_release.service import Release
from origin_catalogue.arc_tables import WireArcs
from origin_catalogue.families import FamilyKey
from origin_catalogue.periods import (
    complete_year,
    counts_periods,
    frame_periods,
    source_periods,
)
from tests.graph_release.support import origin_release

_CRS = "provenance/OECDCreditorReportingSystem"
_DAC1 = "provenance/OECDDAC1"
_CRS_HEALTH = "Amount_FinancialFlow_ODA_Health"
_CRS_EDUCATION = "Amount_FinancialFlow_ODA_Education"
_CRS_GRANTS = "GrantEquivalent_FinancialFlow_ODA"
_DAC1_GNI_SHARE = "Amount_FinancialFlow_ODA_AsFractionOf_GNI"
_COMPOSITE = "provenance/ONEBilateralAndImputedMultilateral"
_AMOUNT = FamilyKey("FinancialFlow", "amount", "measuredValue", None, None, False, True)
_GRANTS = FamilyKey("FinancialFlow", "grantEquivalent", "measuredValue", None, None, False, True)


def _variable(*, measured: str, denominator: str | None = None) -> WireArcs:
    arcs: dict[str, dict[str, list[dict[str, object]]]] = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable", "types": ["Class"]}]},
        "populationType": {
            "nodes": [{"dcid": "FinancialFlow", "name": "FinancialFlow", "types": ["Class"]}]
        },
        "measuredProperty": {"nodes": [{"dcid": measured}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": "financingRecipient"}]},
    }
    if denominator is not None:
        arcs["measurementDenominator"] = {"nodes": [{"dcid": denominator}]}
    return arcs


def _rows(variable: str, facet: str, counts: Mapping[int, int]) -> list[tuple[str, str, list[int]]]:
    """Series of `variable` under `facet` such that `counts[year]` of them hold each year, each
    series holding a run of years up to the last it reaches."""
    rows: list[tuple[str, str, list[int]]] = []
    first = min(counts)
    for n in range(max(counts.values())):
        years = [year for year in sorted(counts) if n < counts[year]]
        rows.append((variable, facet, list(range(first, max(years) + 1))))
    return rows


@pytest.fixture
def release() -> Iterator[Release]:
    released = origin_release(
        {
            _CRS_HEALTH: _variable(measured="amount"),
            _CRS_EDUCATION: _variable(measured="amount"),
            _CRS_GRANTS: _variable(measured="grantEquivalent"),
            _DAC1_GNI_SHARE: _variable(measured="amount", denominator="GNI"),
        },
        series=[
            *_rows(_CRS_HEALTH, "crs", {2023: 20, 2024: 20, 2025: 1}),
            *_rows(_CRS_EDUCATION, "crs", {2023: 10, 2024: 10}),
            *_rows(_CRS_GRANTS, "crs", {2023: 10, 2024: 10, 2025: 10}),
            *_rows(_DAC1_GNI_SHARE, "dac1", {2023: 10, 2024: 10, 2025: 9}),
        ],
        facets={"crs": _CRS, "dac1": _DAC1},
    )
    with bound_release(released) as bound:
        yield bound


@pytest.fixture(autouse=True)
def _no_declared_periods(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(periods_module, "declared_periods", dict)


def test_the_release_holds_the_last_complete_year_of_each_family_under_each_provenance(
    release: Release,
) -> None:
    """The two CRS flow variables share a family, so their series count together: 30 in 2024,
    one in 2025. CRS grant equivalents are a family of their own, complete through 2025."""
    assert release.origin.complete_years == {
        _CRS: {_AMOUNT: 2024, _GRANTS: 2025},
        _DAC1: {release.origin.family_of(_DAC1_GNI_SHARE): 2025},
    }


@pytest.mark.parametrize(
    ("provenance", "family", "expected"),
    [
        pytest.param(_CRS, _AMOUNT, 2024, id="crs-preliminary-year-is-partial"),
        pytest.param(_CRS, _GRANTS, 2025, id="control-sibling-family-complete"),
        pytest.param(_CRS, None, None, id="an-unknown-family-has-no-year"),
        pytest.param("provenance/Other", _AMOUNT, None, id="another-provenance"),
    ],
)
def test_complete_year_reads_the_bound_releases_family(
    release: Release, provenance: str, family: FamilyKey | None, expected: int | None
) -> None:
    assert complete_year(provenance, family) == expected


def test_a_declared_year_wins_for_every_family(
    release: Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(periods_module, "declared_periods", lambda: {_CRS: 2023})

    assert complete_year(_CRS, _GRANTS) == complete_year(_CRS, None) == 2023


def test_each_family_of_a_source_is_judged_against_its_own_year(release: Release) -> None:
    """A frame of CRS flows cut at 2024 and grant equivalents through 2025 is complete: each
    family reaches its own last complete year. One entry over both would take the flows' 2024
    and the grants' 2025 and read the frame as partial."""
    periods = source_periods(
        row_refs=[_CRS],
        sources=[(_CRS, [_CRS_HEALTH, _CRS_GRANTS])],
        ends={(_CRS, _CRS_HEALTH): 2024, (_CRS, _CRS_GRANTS): 2025},
    )

    assert periods == {
        _CRS: [
            {
                "provenance": _CRS,
                "family": list(_AMOUNT),
                "last_complete_year": 2024,
                "last_year": 2024,
            },
            {
                "provenance": _CRS,
                "family": list(_GRANTS),
                "last_complete_year": 2025,
                "last_year": 2025,
            },
        ]
    }
    framed = frame_periods(
        periods, cells=[(f"dcid:{_CRS_HEALTH}", _CRS), (f"dcid:{_CRS_GRANTS}", _CRS)]
    )
    assert [(period.variables, period.partial(2025)) for period in framed] == [
        (frozenset({f"dcid:{_CRS_HEALTH}"}), False),
        (frozenset({f"dcid:{_CRS_GRANTS}"}), False),
    ]


def test_a_partial_family_reaching_past_its_year_is_partial(release: Release) -> None:
    periods = source_periods(
        row_refs=[_CRS], sources=[(_CRS, [_CRS_HEALTH])], ends={(_CRS, _CRS_HEALTH): 2025}
    )

    (framed,) = frame_periods(periods, cells=[(f"dcid:{_CRS_HEALTH}", _CRS)])
    assert framed.partial(2025)
    assert not framed.partial(2024), "rows a card cuts at 2024 are complete"


@pytest.mark.parametrize(
    "row_ref", [pytest.param(_COMPOSITE, id="composite-ref"), pytest.param(_CRS, id="component")]
)
def test_a_composites_entry_holds_rows_carrying_it_or_its_component(
    release: Release, row_ref: str
) -> None:
    """A composite records its component's families under its own provenance. A frame's rows
    carry the composite's, and a derived frame over it may cite the component; both match."""
    recorded = {
        _COMPOSITE: [
            {
                "provenance": _CRS,
                "family": list(_AMOUNT),
                "last_complete_year": 2024,
                "last_year": 2025,
            }
        ]
    }

    (framed,) = frame_periods(recorded, cells=[(f"dcid:{_CRS_HEALTH}", row_ref)])

    assert (framed.row_ref, framed.provenance, framed.partial(2025)) == (_COMPOSITE, _CRS, True)


def test_a_declared_year_is_recorded_under_a_release_that_counts_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(periods_module, "declared_periods", lambda: {_DAC1: 2024})
    with bound_release(origin_release({})):
        assert counts_periods([_DAC1])
        assert not counts_periods([_CRS])
        assert source_periods(
            row_refs=[_DAC1],
            sources=[(_DAC1, [_DAC1_GNI_SHARE])],
            ends={(_DAC1, _DAC1_GNI_SHARE): 2025},
        ) == {
            _DAC1: [
                {
                    "provenance": _DAC1,
                    "family": None,
                    "last_complete_year": 2024,
                    "last_year": 2025,
                }
            ]
        }


def test_without_ends_every_variable_counts_and_no_last_year_is_given(release: Release) -> None:
    assert source_periods(row_refs=[_CRS], sources=[(_CRS, [_CRS_HEALTH])], ends=None) == {
        _CRS: [
            {
                "provenance": _CRS,
                "family": list(_AMOUNT),
                "last_complete_year": 2024,
                "last_year": None,
            }
        ]
    }
