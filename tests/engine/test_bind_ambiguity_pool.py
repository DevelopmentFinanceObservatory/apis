"""Tests two near-neighbour rules through `resolve_variable` with synthetic graph fixtures.

A confirmed tie widens the candidate pool only when another distinct shape has data. Explicit
wording exempts a near-tie only when the query contains a chosen-only token. Naming both concepts
keeps the ambiguity warning. Similarity floor: 0.3.
"""

from __future__ import annotations

import asyncio

import pytest

from search.engine.bind import _UniversalBindOutput
from search.engine.quantity_read import QuantityOutput
from search.engine.region import (
    NEAR_NEIGHBOR_AMBIGUOUS,
    NEAR_NEIGHBOR_POOL_WIDENED,
    RegionResult,
    resolve_variable,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

# The tied, data-less ratio alternate is distinct from the lower-ranked alternate with data.


_SV_INCIDENCE = "SDG_Incidence_Rate"
_SV_CONDITION_COUNT = "Count_MedicalConditionIncident_Condition"
_SV_EXPENDITURE = "Expenditure_On_Condition"

_POOL_NODES: dict = {
    _SV_INCIDENCE: {
        "label": "Number of new infections per 1,000 uninfected population",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {
                "nodes": [{"value": "Number of new infections per 1,000 uninfected population"}]
            },
            "populationType": {"nodes": [{"dcid": "SDG_Series"}]},
            "measuredProperty": {"nodes": [{"dcid": "value"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    _SV_CONDITION_COUNT: {
        "label": "Rate of condition incidents per population",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Rate of condition incidents per population"}]},
            "populationType": {"nodes": [{"dcid": "MedicalConditionIncident"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "measurementDenominator": {"nodes": [{"dcid": "Count_Person"}]},
        },
    },
    _SV_EXPENDITURE: {
        "label": "Expenditure on the condition (percent of health expenditure)",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {
                "nodes": [{"value": "Expenditure on the condition (percent of health expenditure)"}]
            },
            "populationType": {"nodes": [{"dcid": "currentHealthExpenditure"}]},
            "measuredProperty": {"nodes": [{"dcid": "value"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    "country/KEN": {"label": "Kenya", "type": "Country"},
}

_RESOLVE_KENYA: dict = {"Kenya": "country/KEN"}


class _FixedBindLLM:
    """Elects capped_shapes[0] unconditionally, with a caller-supplied also_plausible."""

    def __init__(self, also_plausible: list[int] | None = None):
        self._also_plausible = also_plausible

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[],
            also_plausible=self._also_plausible,
            ask=None,
        ), None


def _run_pool(
    *,
    variable: str,
    svs: list[str],
    scores: list[float],
    svs_with_data: list[str],
    entity: str = "Kenya",
) -> RegionResult:
    """Run `resolve_variable` with the first StatVar elected and no model-reported alternates."""
    detect = {variable: {"svs": svs, "cosine_scores": scores}}
    obs = {
        f"{sv}|{_RESOLVE_KENYA[entity]}": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 10}
        ]
        for sv in svs_with_data
    }
    graph = FakeGraph(nodes=_POOL_NODES, detect=detect, resolve=_RESOLVE_KENYA, obs=obs)
    return asyncio.run(
        resolve_variable(
            variable,
            entities=[entity],
            date_request=None,
            detect_query=variable,
            role_query=f"{variable} in {entity}",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_FixedBindLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )


class TestPoolWidenedWhenPrimaryNeighborLacksData:
    """The tied condition-count shape lacks Kenya data.
    A lower-scoring expenditure alternative has it.
    """

    def _result(self) -> RegionResult:
        return _run_pool(
            variable="incidence rate",
            svs=[_SV_INCIDENCE, _SV_CONDITION_COUNT, _SV_EXPENDITURE],
            scores=[0.831, 0.830, 0.72],
            svs_with_data=[_SV_INCIDENCE, _SV_EXPENDITURE],
        )

    def test_status_is_candidates(self):
        assert self._result().status == "candidates"

    def test_tied_dataless_neighbor_excluded_expenditure_included(self):
        result = self._result()
        sv_dcids = {sv.ref.ref for s in result.specs for sv in s.resolved.members}
        assert len(result.specs) == 2
        assert _SV_INCIDENCE in sv_dcids
        assert _SV_EXPENDITURE in sv_dcids
        assert _SV_CONDITION_COUNT not in sv_dcids

    @pytest.mark.parametrize(
        "check",
        [
            pytest.param("warning_pair", id="warning-pair"),
            pytest.param(
                "near_neighbor_flag_and_exclusion",
                id="near-neighbor-flag-and-exclusion",
            ),
        ],
    )
    def test_widened_pool_warnings_and_dataless_alternate_exclusion(self, check):
        result = self._result()
        if check == "warning_pair":
            codes = [warning.code for warning in result.warnings]
            assert NEAR_NEIGHBOR_AMBIGUOUS in codes
            assert NEAR_NEIGHBOR_POOL_WIDENED in codes
        else:
            assert result.near_neighbor_fired is True
            sv_dcids = {sv.ref.ref for spec in result.specs for sv in spec.resolved.members}
            assert _SV_CONDITION_COUNT not in sv_dcids


class TestPoolStaysDefiniteWhenNoAlternativeHasData:
    """The response stays definite when no alternate has data."""

    def _result(self) -> RegionResult:
        return _run_pool(
            variable="incidence rate",
            svs=[_SV_INCIDENCE, _SV_CONDITION_COUNT, _SV_EXPENDITURE],
            scores=[0.831, 0.830, 0.72],
            svs_with_data=[_SV_INCIDENCE],
        )

    def test_status_is_definite(self):
        assert self._result().status == "definite"

    def test_pool_widened_warning_absent(self):
        codes = [w.code for w in self._result().warnings]
        assert NEAR_NEIGHBOR_AMBIGUOUS not in codes
        assert NEAR_NEIGHBOR_POOL_WIDENED not in codes


class TestCascadeNeverRunsWithoutAConfirmedTie:
    """A lower-scoring alternative widens the pool only after a confirmed tie."""

    def _result(self) -> RegionResult:
        return _run_pool(
            variable="incidence rate",
            svs=[_SV_INCIDENCE, _SV_EXPENDITURE],
            scores=[0.90, 0.70],  # margin 0.20, well above SEARCH_NEAR_MARGIN
            svs_with_data=[_SV_INCIDENCE, _SV_EXPENDITURE],
        )

    def test_status_is_definite(self):
        assert self._result().status == "definite"

    def test_no_near_neighbor_warnings(self):
        codes = [w.code for w in self._result().warnings]
        assert NEAR_NEIGHBOR_AMBIGUOUS not in codes
        assert NEAR_NEIGHBOR_POOL_WIDENED not in codes


class TestCascadeNeverRunsWhenChosenItselfLacksData:
    """The pool-widening cascade stays inactive when the chosen shape lacks data."""

    def _result(self) -> RegionResult:
        return _run_pool(
            variable="incidence rate",
            svs=[_SV_INCIDENCE, _SV_CONDITION_COUNT, _SV_EXPENDITURE],
            scores=[0.831, 0.830, 0.72],
            svs_with_data=[_SV_EXPENDITURE],  # neither chosen nor its tied neighbor has data
        )

    def test_status_is_no_data(self):
        assert self._result().status == "no_data"

    def test_pool_widened_warning_absent(self):
        codes = [w.code for w in self._result().warnings]
        assert NEAR_NEIGHBOR_POOL_WIDENED not in codes


# ---------------------------------------------------------------------------
# Explicit wording can distinguish near-identical labels above the 0.3 similarity floor.
# ---------------------------------------------------------------------------

_SV_NATIONAL_LINE = "Poverty_National_Line"
_SV_INTERNATIONAL_LINE = "Poverty_International_Line"

_LINE_NODES: dict = {
    _SV_NATIONAL_LINE: {
        "label": "Proportion of population living below the national poverty line",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {
                "nodes": [
                    {"value": "Proportion of population living below the national poverty line"}
                ]
            },
            "populationType": {"nodes": [{"dcid": "SDG_National_Poverty"}]},
            "measuredProperty": {"nodes": [{"dcid": "value"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    _SV_INTERNATIONAL_LINE: {
        "label": "Proportion of population below international poverty line",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {
                "nodes": [{"value": "Proportion of population below international poverty line"}]
            },
            "populationType": {"nodes": [{"dcid": "SDG_International_Poverty"}]},
            "measuredProperty": {"nodes": [{"dcid": "value"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    "country/ETH": {"label": "Ethiopia", "type": "Country"},
}

_RESOLVE_ETHIOPIA: dict = {"Ethiopia": "country/ETH"}


def _run_line(*, variable: str, scores: list[float]) -> RegionResult:
    svs = [_SV_NATIONAL_LINE, _SV_INTERNATIONAL_LINE]
    detect = {variable: {"svs": svs, "cosine_scores": scores}}
    obs = {
        f"{sv}|country/ETH": [{"earliestDate": "2015", "latestDate": "2015", "obsCount": 1}]
        for sv in svs
    }
    graph = FakeGraph(nodes=_LINE_NODES, detect=detect, resolve=_RESOLVE_ETHIOPIA, obs=obs)
    return asyncio.run(
        resolve_variable(
            variable,
            entities=["Ethiopia"],
            date_request=None,
            detect_query=variable,
            role_query=f"{variable} in Ethiopia",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_FixedBindLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )


class TestExplicitWordingExemptsAtRecalibratedThreshold:
    """The query's `national` token disambiguates a near-tie above the 0.3 similarity floor."""

    def _result(self) -> RegionResult:
        return _run_line(
            variable="poverty headcount ratio below national poverty line",
            scores=[0.83, 0.81],  # National ranks first in a near tie.
        )

    def test_status_is_definite(self):
        assert self._result().status == "definite"

    def test_no_near_neighbor_warning(self):
        codes = [w.code for w in self._result().warnings]
        assert NEAR_NEIGHBOR_AMBIGUOUS not in codes


class TestLowSimilarityPairStillFiresAtRecalibratedThreshold:
    """A query without either distinguishing word keeps the ambiguity warning."""

    def _result(self) -> RegionResult:
        return _run_line(
            variable="poverty rate statistic",
            scores=[0.81, 0.83],
        )

    def test_status_is_candidates(self):
        assert self._result().status == "candidates"

    def test_near_neighbor_warning_fires(self):
        codes = [w.code for w in self._result().warnings]
        assert NEAR_NEIGHBOR_AMBIGUOUS in codes


# ---------------------------------------------------------------------------
# WHO vaccine labels share a template, so generic wording can score highly against both.
# ---------------------------------------------------------------------------

_SV_DTP3 = "WHO_dptv_Male"
_SV_FULL_IMMUNIZATION = "WHO_fullv_Male"

_WHO_TEMPLATE_NODES: dict = {
    _SV_DTP3: {
        "label": "Dtp3 Immunization Coverage Among One-Year-Olds (%), Male",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {
                "nodes": [{"value": "Dtp3 Immunization Coverage Among One-Year-Olds (%), Male"}]
            },
            "populationType": {"nodes": [{"dcid": "Thing"}]},
            "measuredProperty": {"nodes": [{"dcid": "who/dptv"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    _SV_FULL_IMMUNIZATION: {
        "label": "Full Immunization Coverage Among One-Year-Olds (%) (fullv), Male",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {
                "nodes": [
                    {"value": ("Full Immunization Coverage Among One-Year-Olds (%) (fullv), Male")}
                ]
            },
            "populationType": {"nodes": [{"dcid": "Thing"}]},
            "measuredProperty": {"nodes": [{"dcid": "who/fullv"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    "country/ETH": {"label": "Ethiopia", "type": "Country"},
}

_RESOLVE_ETHIOPIA_WHO: dict = {"Ethiopia": "country/ETH"}


def _run_who_template(*, variable: str, scores: list[float]) -> RegionResult:
    svs = [_SV_DTP3, _SV_FULL_IMMUNIZATION]
    detect = {variable: {"svs": svs, "cosine_scores": scores}}
    obs = {
        f"{sv}|country/ETH": [{"earliestDate": "2019", "latestDate": "2019", "obsCount": 1}]
        for sv in svs
    }
    graph = FakeGraph(
        nodes=_WHO_TEMPLATE_NODES, detect=detect, resolve=_RESOLVE_ETHIOPIA_WHO, obs=obs
    )
    return asyncio.run(
        resolve_variable(
            variable,
            entities=["Ethiopia"],
            date_request=None,
            detect_query=variable,
            role_query=f"{variable} in Ethiopia",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_FixedBindLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )


class TestWhoTemplateEchoClosesDiscriminatingTokenBlindSpot:
    """The generic query scores about 0.44 against the shared label template but names no DTP3-only
    token, so the ambiguity warning stays active.
    """

    def _result(self) -> RegionResult:
        return _run_who_template(
            variable="immunization coverage rate among one-year-olds",
            scores=[0.83, 0.81],
        )

    def test_status_is_candidates(self):
        assert self._result().status == "candidates"

    def test_near_neighbor_warning_fires(self):
        codes = [w.code for w in self._result().warnings]
        assert NEAR_NEIGHBOR_AMBIGUOUS in codes


class TestWhoTemplateNamedVaccineExempts:
    """The query names `dtp3` without naming a token unique to the full-immunisation neighbour."""

    def _result(self) -> RegionResult:
        return _run_who_template(
            variable="dtp3 immunization coverage rate among one-year-olds",
            scores=[0.83, 0.81],
        )

    def test_status_is_definite(self):
        """Positive control (passes under the one-sided AND the symmetric check)."""
        assert self._result().status == "definite"

    def test_no_near_neighbor_warning(self):
        """Positive control (passes under the one-sided AND the symmetric check)."""
        codes = [w.code for w in self._result().warnings]
        assert NEAR_NEIGHBOR_AMBIGUOUS not in codes


# ---------------------------------------------------------------------------
# Naming both concepts' distinguishing words must keep the ambiguity warning.
# These illustrative Pakistan labels have no recorded corpus fixture.
# ---------------------------------------------------------------------------

_SV_ADULT_FERTILITY = "AdultFertilityRate"
_SV_LITERACY = "LiteracyRate"

_LITERACY_FERTILITY_NODES: dict = {
    _SV_ADULT_FERTILITY: {
        "label": "Adult fertility rate",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Adult fertility rate"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "fertilityRate"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    _SV_LITERACY: {
        "label": "Literacy rate",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Literacy rate"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "literacyRate"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    "country/PAK": {"label": "Pakistan", "type": "Country"},
}

_RESOLVE_PAKISTAN: dict = {"Pakistan": "country/PAK"}


def _run_literacy_fertility(*, variable: str, scores: list[float]) -> RegionResult:
    svs = [_SV_ADULT_FERTILITY, _SV_LITERACY]
    detect = {variable: {"svs": svs, "cosine_scores": scores}}
    obs = {
        f"{sv}|country/PAK": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}]
        for sv in svs
    }
    graph = FakeGraph(
        nodes=_LITERACY_FERTILITY_NODES, detect=detect, resolve=_RESOLVE_PAKISTAN, obs=obs
    )
    return asyncio.run(
        resolve_variable(
            variable,
            entities=["Pakistan"],
            date_request=None,
            detect_query=variable,
            role_query=f"{variable} in Pakistan",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_FixedBindLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )


class TestCompoundQueryNamingBothConceptsFires:
    """The query names both `adult` and `literacy`, so it is ambiguous. The higher-scoring
    fertility shape is elected.
    """

    def _result(self) -> RegionResult:
        return _run_literacy_fertility(
            variable="adult fertility literacy rate",
            scores=[0.83, 0.81],
        )

    def test_status_is_candidates(self):
        assert self._result().status == "candidates"

    def test_near_neighbor_warning_fires(self):
        codes = [w.code for w in self._result().warnings]
        assert NEAR_NEIGHBOR_AMBIGUOUS in codes
