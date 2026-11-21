"""The last complete year of a DFO provenance's data (`origin_catalogue.periods`), from the
bound release's family counts (`graph_release.origin.ReleaseOrigin.complete_years`) and the
registry's declared years.

The year counts scale down two families of the dev graph release read on 2026-11-18: CRS
flows, whose preliminary 2025 holds 4 percent of 2024's series, and DAC1 ODA as a share of
GNI, whose 2025 holds nine tenths of 2024's.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import pytest

import origin_catalogue.periods as periods_module
from description.facts import FixedValue, VariableFacts
from graph_release.context import bound_release
from graph_release.service import Release
from origin_catalogue.arc_tables import WireArcs
from origin_catalogue.periods import last_complete_year, version_last_complete_period
from tests.graph_release.support import origin_release

_CRS = "provenance/OECDCreditorReportingSystem"
_DAC1 = "provenance/OECDDAC1"
_CRS_HEALTH = "Amount_FinancialFlow_ODA_Health"
_CRS_EDUCATION = "Amount_FinancialFlow_ODA_Education"
_DAC1_GNI_SHARE = "Amount_FinancialFlow_ODA_AsFractionOf_GNI"


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
            _DAC1_GNI_SHARE: _variable(measured="amount", denominator="GNI"),
        },
        series=[
            *_rows(_CRS_HEALTH, "crs", {2023: 20, 2024: 20, 2025: 1}),
            *_rows(_CRS_EDUCATION, "crs", {2023: 10, 2024: 10}),
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
    """The two CRS variables share a family, so their series count together: 30 in 2024, one
    in 2025."""
    complete = release.origin.complete_years

    assert {provenance: sorted(years.values()) for provenance, years in complete.items()} == {
        _CRS: [2024],
        _DAC1: [2025],
    }


@pytest.mark.parametrize(
    ("provenance", "variables", "shape", "expected"),
    [
        pytest.param(_CRS, [_CRS_HEALTH], None, 2024, id="crs-preliminary-year-is-partial"),
        pytest.param(_DAC1, [_DAC1_GNI_SHARE], None, 2025, id="control-dac1-share-complete"),
        pytest.param(
            _CRS,
            [],
            {"populationType": "FinancialFlow", "measuredProperty": "amount"},
            2024,
            id="a-shape-reads-its-families",
        ),
        pytest.param(_CRS, [], {"measuredProperty": "grantEquivalent"}, None, id="no-family"),
        pytest.param(_CRS, ["Unknown_Variable"], {}, None, id="unknown-variable-no-shape"),
        pytest.param("provenance/Other", [_CRS_HEALTH], None, None, id="another-provenance"),
    ],
)
def test_last_complete_year_reads_the_bound_releases_families(
    release: Release,
    provenance: str,
    variables: list[str],
    shape: dict[str, str] | None,
    expected: int | None,
) -> None:
    assert last_complete_year(provenance, variables=variables, shape=shape) == expected


def test_a_declared_year_wins_over_the_releases(
    release: Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(periods_module, "declared_periods", lambda: {_DAC1: 2024})

    assert last_complete_year(_DAC1, variables=[_DAC1_GNI_SHARE]) == 2024
    assert last_complete_year(_CRS, variables=[_CRS_HEALTH]) == 2024


def test_a_versions_period_reads_its_resolved_variables_then_its_record() -> None:
    facts = VariableFacts(
        shape=[FixedValue(property="measuredProperty", value="amount")],
        variable=None,
        fixed=[],
        dimensions=None,
        admitted_at_mint=1,
        admitted_from="resolved_variables",
        with_data=1,
    )
    released = origin_release(
        {_CRS_HEALTH: _variable(measured="amount")},
        series=_rows(_CRS_HEALTH, "crs", {2024: 20, 2025: 1}),
        facets={"crs": _CRS},
    )
    with bound_release(released):
        assert version_last_complete_period(_CRS, None, resolved=[_CRS_HEALTH]) == "2024"
        assert version_last_complete_period(_CRS, facts) == "2024"
        assert version_last_complete_period(_CRS, None) is None
