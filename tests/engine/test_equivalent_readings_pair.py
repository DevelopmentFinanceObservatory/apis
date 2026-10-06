"""Checks that the electricity-access twin stays nested under its elected part after multi-variable composition.

The pair uses real registry dcids with synthetic graph arcs.
"""

from __future__ import annotations

import dataclasses

from search.engine.conjoin import combine_regions
from search.engine.envelope import now_ms
from search.engine.region import RegionResult
from tests.engine._builders import conjoin_spec, shape_five_tuple
from tests.engine.test_equivalent_readings_divert import _ELEC_SDG, _ELEC_WB, _facet, _run, _sv_refs


def _population_region() -> RegionResult:
    """An unrelated definite region on a different five-tuple shape, standing
    in for a second variable asked in the same request."""
    shape = shape_five_tuple("Person", "count")
    spec = conjoin_spec(shape, "total", sv_dcid="Count_Person")
    return RegionResult(
        variable_text="population",
        status="definite",
        specs=(spec,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
    )


def test_twin_stays_nested_under_its_part_in_a_two_variable_combine():
    electricity_region = _run(
        svs=[_ELEC_WB, _ELEC_SDG],
        scores=[0.90, 0.85],
        chosen_shape_index=0,
        also_plausible=[1],
        obs={
            f"{_ELEC_WB}|country/KEN": [_facet(provenance_id="source/WorldBank")],
            f"{_ELEC_SDG}|country/KEN": [_facet(provenance_id="source/SDG")],
        },
    )
    assert electricity_region.status == "definite"

    regions = [electricity_region, _population_region()]
    regions = [dataclasses.replace(r, earliest_index=i) for i, r in enumerate(regions)]

    response, _effective = combine_regions(
        regions,
        query="access to electricity and population in Kenya",
        extra_warnings=[],
        start_ms=now_ms(),
        engine_build="test",
        max_candidates=10,
    )

    assert response.status == "definite"
    assert len(response.parts) == 2

    electricity_part, population_part = response.parts
    assert electricity_part.status == "definite"
    assert _sv_refs(electricity_part.spec) == {_ELEC_WB}
    twins = electricity_part.spec.resolved.equivalent_readings
    assert twins is not None and len(twins) == 1
    assert _sv_refs(twins[0]) == {_ELEC_SDG}

    assert population_part.status == "definite"
    assert population_part.variable_text == "population"
    assert _sv_refs(population_part.spec) == {"Count_Person"}

    # The twin never surfaces as its own part: no part in the envelope elects
    # sdg/EG_ACS_ELEC as its leading spec.
    for part in response.parts:
        assert _sv_refs(part.spec) != {_ELEC_SDG}
