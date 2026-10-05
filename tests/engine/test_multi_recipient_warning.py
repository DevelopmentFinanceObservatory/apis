"""In-cap, two-place standard queries emit no `MULTI_RECIPIENT_TRUNCATED`
warning. The configured cap boundary is covered by test_standard_multi_place.py.
"""

from __future__ import annotations

import asyncio

import pytest

from search.engine.bind import _UniversalBindOutput
from search.engine.region import (
    MULTI_RECIPIENT_TRUNCATED,
    resolve_variable,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver

# A single SV resolved via the standard resolve path.
# Arc structure satisfies derive_shapes: populationType non-empty, measuredProperty present.
_SV = "Count_Person"

_NODES: dict = {
    _SV: {
        "label": "Count Person",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Count Person"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
}

# detect_query key used for both test cases; maps to the single standard SV.
_DETECT_QUERY = "exports"
_DETECT: dict = {_DETECT_QUERY: {"svs": [_SV], "entities": []}}
_RESOLVE: dict = {"Kenya": "country/KEN", "Uganda": "country/UGA"}


class _FixedBindLLM:
    """Elects the sole candidate shape with no constraint bindings, unconditionally."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[],
            also_plausible=None,
            ask=None,
        ), None


def _run(variable: str, role_query: str, entities: list[str]):
    """Run resolve_variable with an inline FakeGraph and no obs (forces no_data)."""
    graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve=_RESOLVE, obs={})
    return asyncio.run(
        resolve_variable(
            variable,
            entities=entities,
            date_request=None,
            detect_query=_DETECT_QUERY,
            role_query=role_query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


@pytest.mark.parametrize(
    ("role_query", "entities"),
    [
        pytest.param(
            "exports Kenya Uganda",
            ["Kenya", "Uganda"],
            id="bare-two-places",
        ),
        pytest.param(
            "exports to Kenya and to Uganda",
            ["Kenya", "Uganda"],
            id="directional-two-recipient",
        ),
    ],
)
def test_two_place_query_has_no_truncation_warning(role_query: str, entities: list[str]) -> None:
    """Bare and directional two-place queries stay below the warning cap."""
    result = _run("exports", role_query=role_query, entities=entities)
    assert not any(w.code == MULTI_RECIPIENT_TRUNCATED for w in result.warnings)
