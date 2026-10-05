"""Tests the combination and request-level probe ceilings, each of which must reject
oversized work before its first graph probe, and that a converting preview's read count
does not grow with the elected combinations.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

import search.engine.region.observations.fetch as region_fetch
from materialise.facets import FacetSelection
from search.engine.errors import EngineInputError
from search.engine.region.convertibility import (
    _BOUND_FACET_COMBINATIONS_MAX,
    _REQUEST_PROBE_BUDGET_MAX,
    BoundFacetCombinationsCeilingExceededError,
    _bound_facet_combinations,
    _combination_count,
    decide_part_unit,
)
from search.models import IdentityUnitRequest
from tests.engine._fake_coverage import build_coverage_index, empty_coverage_snapshot
from tests.engine._harness import (
    HARNESS_FACET,
    build_observations_planned,
    sdmx_double,
    stub_preview_result,
)
from tests.engine.test_convertibility import _obs_facet, _resubmit
from tests.fixtures import FakeGraph
from tests.materialise.conftest import FakeGraphReader

# These cases raise before reading coverage, so an empty index suffices.
_UNUSED_COVERAGE = build_coverage_index([], role_names={})


def _unit_combo() -> dict[str, FacetSelection]:
    """Build the minimal combination a unit decision returns: its elected unit."""
    return {
        "unit": FacetSelection(
            axis="unit",
            ref="CurrencyUSD",
            basis="bound",
            default_policy=None,
            default_policy_version=None,
            selection_reason=None,
            dispatch_facet_id="f1",
        )
    }


@pytest.mark.parametrize("combos_per_sv", [1, 2, _BOUND_FACET_COMBINATIONS_MAX])
def test_a_converting_preview_makes_one_read_per_series_whatever_the_combinations(
    combos_per_sv: int,
):
    """Elected combinations decide the units the plan is narrowed to. The preview dispatches
    the narrowed plan's group once per series, so its read count never grows with them."""
    entities = ["country/A", "country/B", "country/C"]
    elected = {"Count_Person": tuple(_unit_combo() for _ in range(combos_per_sv))}
    usd_facet = replace(HARNESS_FACET, unit="CurrencyUSD")
    reader = FakeGraphReader(facets={("Count_Person", entity): [usd_facet] for entity in entities})
    calls: list[object] = []

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
    ):
        calls.append(request)
        (entity,) = request.role_bindings["observationAbout"]
        return stub_preview_result(sv="Count_Person", entity=entity)

    with patch.object(region_fetch, "materialise_preview", _spy):
        observations, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": entities},
            graph=FakeGraph(),
            materialise_graph=reader,
            materialise_sdmx=sdmx_double(),
            unit_request=IdentityUnitRequest(currency="USD", prices="current"),
            elected=elected,
            coverage=empty_coverage_snapshot(),
        )

    assert warn is None
    assert len(observations.series) == len(calls) == len(entities)


def test_combination_count_at_ceiling_yields_every_combination():
    """Exactly `_BOUND_FACET_COMBINATIONS_MAX` members fit within the cap."""
    refs = [f"p{i}" for i in range(_BOUND_FACET_COMBINATIONS_MAX)]
    combos = list(_bound_facet_combinations({"provenance": refs}))
    assert len(combos) == _BOUND_FACET_COMBINATIONS_MAX


def test_combination_count_over_ceiling_raises_before_yielding_any_combination():
    """One member over the ceiling raises before yielding a combination."""
    refs = [f"p{i}" for i in range(_BOUND_FACET_COMBINATIONS_MAX + 1)]
    generator = _bound_facet_combinations({"provenance": refs})
    with pytest.raises(BoundFacetCombinationsCeilingExceededError) as exc_info:
        next(generator)
    assert exc_info.value.count == _BOUND_FACET_COMBINATIONS_MAX + 1
    assert exc_info.value.cap == _BOUND_FACET_COMBINATIONS_MAX


def test_32x32_binding_rejected_with_zero_graph_probes():
    """A 32x32 product exceeds the cap and raises before the graph receives a probe."""
    provenance_refs = [f"prov{i}" for i in range(32)]
    measurement_method_refs = [f"mm{i}" for i in range(32)]
    reader = FakeGraphReader()
    with pytest.raises(BoundFacetCombinationsCeilingExceededError) as exc_info:
        decide_part_unit(
            ["Count_Person"],
            unit_request=IdentityUnitRequest(currency="USD", prices="current"),
            role_bindings={"observationAbout": ["country/KEN"]},
            declared_roles=("observationAbout",),
            bound_facets={
                "provenance": provenance_refs,
                "measurementMethod": measurement_method_refs,
            },
            years=None,
            graph=reader,
            sdmx=sdmx_double(),
            coverage=_UNUSED_COVERAGE,
        )
    assert exc_info.value.count == 1024
    assert exc_info.value.cap == _BOUND_FACET_COMBINATIONS_MAX
    assert reader.calls["variable_facets_batch"] == 0


def test_multiple_sv_dcids_never_probe_when_product_over_ceiling():
    """The request-level ceiling rejects the first StatVar before any graph probe."""
    provenance_refs = [f"prov{i}" for i in range(_BOUND_FACET_COMBINATIONS_MAX + 1)]
    reader = FakeGraphReader()
    with pytest.raises(BoundFacetCombinationsCeilingExceededError):
        decide_part_unit(
            ["Count_Person", "Count_Household"],
            unit_request=IdentityUnitRequest(currency="USD", prices="current"),
            role_bindings={"observationAbout": ["country/KEN"]},
            declared_roles=("observationAbout",),
            bound_facets={"provenance": provenance_refs},
            years=None,
            graph=reader,
            sdmx=sdmx_double(),
            coverage=_UNUSED_COVERAGE,
        )
    assert reader.calls["variable_facets_batch"] == 0


def test_combination_count_helper_matches_bound_facet_combinations_length():
    """The combination count matches yielded combinations for string and Set bindings."""
    bound_facets = {"provenance": ["p0", "p1", "p2"], "measurementMethod": "mm0"}
    assert _combination_count(bound_facets) == 3
    assert len(list(_bound_facet_combinations(bound_facets))) == 3


def test_request_level_probe_budget_rejected_with_zero_graph_probes():
    """A 60-wide binding multiplied by the StatVar count exceeds the request budget.
    The request fails before probing.
    """
    combination_width = _BOUND_FACET_COMBINATIONS_MAX
    provenance_refs = [f"prov{i}" for i in range(combination_width)]
    sv_count = (_REQUEST_PROBE_BUDGET_MAX // combination_width) + 1
    sv_dcids = [f"Count_Person_{i}" for i in range(sv_count)]
    reader = FakeGraphReader()
    with pytest.raises(BoundFacetCombinationsCeilingExceededError) as exc_info:
        decide_part_unit(
            sv_dcids,
            unit_request=IdentityUnitRequest(currency="USD", prices="current"),
            role_bindings={"observationAbout": ["country/KEN"]},
            declared_roles=("observationAbout",),
            bound_facets={"provenance": provenance_refs},
            years=None,
            graph=reader,
            sdmx=sdmx_double(),
            coverage=_UNUSED_COVERAGE,
        )
    assert exc_info.value.count == sv_count * combination_width
    assert exc_info.value.cap == _REQUEST_PROBE_BUDGET_MAX
    assert reader.calls["variable_facets_batch"] == 0


def test_promote_door_returns_422_facet_probe_ceiling_over_the_wire():
    """The promote endpoint maps an over-cap binding to `facet_probe_ceiling` with status 422."""
    provenance_refs = [f"prov{i}" for i in range(_BOUND_FACET_COMBINATIONS_MAX + 1)]
    with pytest.raises(EngineInputError) as exc_info:
        _resubmit(
            request=IdentityUnitRequest(currency="USD", prices="current"),
            obs_facets=[_obs_facet(unit="CurrencyUSD")],
            nominal=True,
            provenance_declares_recipe={},
            provenance_refs=provenance_refs,
        )
    assert exc_info.value.code == "facet_probe_ceiling"
    assert exc_info.value.status == 422
