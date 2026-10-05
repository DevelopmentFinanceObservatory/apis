"""Reposting a freshly resolved definite spec unchanged preserves its status and identity."""

from __future__ import annotations

from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.models import ResubmitIdentityPayload
from tests.engine._axis_doubles import TOTAL_POPULATION_INDIA_FAMILY, axis_confirming_reader
from tests.engine._base_identity import base_from_spec
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import _FAKE_GRAPH, make_request, offline_resolve


def test_fresh_definite_posted_back_unedited_stays_definite():
    result = offline_resolve(make_request("total population India"))
    inner = result.parts[0]
    assert inner.status == "definite"
    spec = inner.spec

    base_spec_id, base_identity = base_from_spec(spec)
    echo = ResubmitIdentityPayload(
        identity=spec.definition.identity,
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        resolved_stat_var_dcids=[sv.ref for sv in spec.resolved.stat_vars.items],
    )
    assert echo.identity.model_dump(mode="json") == spec.definition.identity.model_dump(mode="json")
    region = resolve_spec_resubmit(
        inp=echo,
        graph=_FAKE_GRAPH,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(
            materialise_graph=axis_confirming_reader(*TOTAL_POPULATION_INDIA_FAMILY)
        ),
    )

    assert region.status == "definite", region.no_data_reason
    assert region.specs[0].spec_id == spec.spec_id


def test_resubmitted_definite_keeps_the_fresh_ones_confidence():
    """A posted identity keeps the fresh definite's high confidence."""
    result = offline_resolve(make_request("total population India"))
    inner = result.parts[0]
    assert inner.status == "definite"
    spec = inner.spec
    assert spec.resolved.confidence == "high"

    base_spec_id, base_identity = base_from_spec(spec)
    echo = ResubmitIdentityPayload(
        identity=spec.definition.identity,
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        resolved_stat_var_dcids=[sv.ref for sv in spec.resolved.stat_vars.items],
    )
    region = resolve_spec_resubmit(
        inp=echo,
        graph=_FAKE_GRAPH,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(
            materialise_graph=axis_confirming_reader(*TOTAL_POPULATION_INDIA_FAMILY)
        ),
    )

    assert region.status == "definite", region.no_data_reason
    assert region.specs[0].resolved.confidence == "high"
