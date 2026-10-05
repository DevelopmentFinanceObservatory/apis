"""Roles that share a flow axis return `no_data(role_signature_incompatible)`.

The pipeline catches `RoleSignatureError` before bind runs.
"""

from __future__ import annotations

import asyncio

from search.engine.core import resolve_async
from search.models import ParsedInput, ParsedPart, ResolveRequest
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph


def _person_arcs(*, observation_properties: list[str] | None = None) -> dict:
    arcs: dict = {
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "count"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if observation_properties is not None:
        arcs["observationProperties"] = {"nodes": [{"dcid": p} for p in observation_properties]}
    return arcs


class _UnreachableLLM:
    """Fails the test loudly if bind ever calls it. The group refusal must
    short-circuit the pipeline before bind runs."""

    async def generate_structured(self, *, prompt, system, schema):
        raise AssertionError("bind must not run when derive_shapes refuses the group")


def _run():
    debtor_sv = "Count_Person_MixedSignatureDebtor"
    role_sv = "Count_Person_MixedSignatureRoles"
    nodes = {
        debtor_sv: {
            "label": "Population (debtor)",
            "arcs": _person_arcs(observation_properties=["debtor"]),
        },
        role_sv: {
            "label": "Population (roles)",
            "arcs": _person_arcs(observation_properties=["provider", "recipient"]),
        },
    }
    detect = {"population count": {"svs": [debtor_sv, role_sv], "cosine_scores": [0.9, 0.9]}}
    graph = FakeGraph(nodes=nodes, detect=detect, resolve={}, obs={})
    part = ParsedPart(variable_text="population count")
    request = ResolveRequest(input=ParsedInput(parts=[part]))
    sdmx = FakeSdmxClient(observations={}, availability={})
    return asyncio.run(resolve_async(request, graph=graph, llm=_UnreachableLLM(), sdmx=sdmx))


def test_roles_sharing_a_flow_axis_yield_no_data_role_signature_incompatible():
    response = _run()

    part = response.parts[0]
    assert part.status == "no_data", f"expected no_data, got {part.status!r}: {part}"
    assert part.no_data is not None
    assert part.no_data.reason == "role_signature_incompatible"


def test_mixed_role_signature_marks_later_pipeline_steps_not_run():
    """Only `shape` ran, so later steps have no timing entries."""
    response = _run()

    by_step = response.parts[0].timing_ms.by_step or {}
    assert "shape" in by_step
    assert "bind" not in by_step
    assert "materialise" not in by_step
    assert "answer" not in by_step
