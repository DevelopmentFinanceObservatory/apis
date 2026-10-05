"""`rate_cache` reaches `_build_observations` through the standard promote
path. Both a supplied cache and `None` preserve the definite result.
"""

from __future__ import annotations

from unittest.mock import patch

from materialise.rate_tables import RateTableCache
from search.engine.region import promote_finalize, resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.models import ResubmitIdentityPayload
from tests.engine._base_identity import GDP_SHAPE, make_base, make_identity, subject_entity
from tests.engine._builders import unbound_slot as _unbound_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import materialise_double_from_obs
from tests.fixtures import FakeGraph

_GDP_SV_DCID = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_GDP_ENTITY_DCID = "country/IND"

_GDP_OBS = {
    f"{_GDP_SV_DCID}|{_GDP_ENTITY_DCID}": [
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/WorldBank",
            "observations": [{"date": "2020", "value": 1.0}],
        }
    ],
}


def _run(*, rate_cache: RateTableCache | None):
    entities = [subject_entity(_GDP_ENTITY_DCID)]
    base_spec_id, base_identity = make_base(GDP_SHAPE, entities=entities)
    inp = ResubmitIdentityPayload(
        identity=make_identity(
            GDP_SHAPE,
            [_unbound_slot("how", "activitySource", "activity source")],
            entities=entities,
        ),
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        resolved_stat_var_dcids=[_GDP_SV_DCID],
    )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_GDP_OBS)
    return resolve_spec_resubmit(
        inp=inp,
        graph=FakeGraph(obs=_GDP_OBS),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(
            include_observations=True,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=rate_cache,
        ),
    )


def test_rate_cache_reaches_build_observations():
    """rate_cache passed to resolve_spec_resubmit lands on the _PromoteState
    _resolve_standard_promote builds, and reaches _build_observations through
    _phase_finalize's include_observations fetch site the same way
    materialise_graph and materialise_sdmx do."""
    sentinel = RateTableCache()
    captured: list[object] = []
    real_build_observations = promote_finalize._build_observations

    def _spy(*args, **kwargs):
        captured.append(kwargs.get("rate_cache"))
        return real_build_observations(*args, **kwargs)

    with patch.object(promote_finalize, "_build_observations", _spy):
        result = _run(rate_cache=sentinel)

    assert result.status == "definite", f"expected definite, got {result.status!r}"
    assert captured, "expected the include_observations fetch to call _build_observations"
    assert all(c is sentinel for c in captured)


def test_standard_promote_with_no_rate_cache_still_serves():
    """None reaches _build_observations as None, and the promote path serves
    the same definite answer."""
    captured: list[object] = []
    real_build_observations = promote_finalize._build_observations

    def _spy(*args, **kwargs):
        captured.append(kwargs.get("rate_cache"))
        return real_build_observations(*args, **kwargs)

    with patch.object(promote_finalize, "_build_observations", _spy):
        result = _run(rate_cache=None)

    assert result.status == "definite", f"expected definite, got {result.status!r}"
    assert captured, "expected the include_observations fetch to call _build_observations"
    assert all(c is None for c in captured)
