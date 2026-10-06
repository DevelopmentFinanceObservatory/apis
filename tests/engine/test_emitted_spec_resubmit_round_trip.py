"""Reposting a freshly resolved definite spec unchanged preserves its status and identity."""

from __future__ import annotations

from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.models import ResubmitIdentityPayload
from tests.engine._axis_doubles import TOTAL_POPULATION_INDIA_FAMILY, axis_confirming_reader
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import _FAKE_GRAPH, make_request, offline_resolve, walk_options


def test_fresh_definite_posted_back_unedited_stays_definite():
    result = offline_resolve(make_request("total population India"))
    inner = result.parts[0]
    assert inner.status == "definite"
    spec = inner.spec

    echo = ResubmitIdentityPayload(
        identity=spec.definition.identity,
    )
    assert echo.identity.model_dump(mode="json") == spec.definition.identity.model_dump(mode="json")
    region = resolve_spec_resubmit(
        inp=echo,
        graph=_FAKE_GRAPH,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(
            _FAKE_GRAPH,
            EngineOptions(materialise_graph=axis_confirming_reader(*TOTAL_POPULATION_INDIA_FAMILY)),
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

    echo = ResubmitIdentityPayload(
        identity=spec.definition.identity,
    )
    region = resolve_spec_resubmit(
        inp=echo,
        graph=_FAKE_GRAPH,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(
            _FAKE_GRAPH,
            EngineOptions(materialise_graph=axis_confirming_reader(*TOTAL_POPULATION_INDIA_FAMILY)),
        ),
    )

    assert region.status == "definite", region.no_data_reason
    assert region.specs[0].resolved.confidence == "high"
