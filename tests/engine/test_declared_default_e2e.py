"""The declared flow default stamps ``CRSTypeOfFlow10`` as ``declared_default`` (#968).

The fixture uses a packaged-index CRS StatVar.
Omitting "ODA" from the query leaves the flow slot open for the default."""

from __future__ import annotations

import asyncio

from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.extract import ExtractedEntity, Extraction
from search.engine.quantity_read import QuantityOutput
from search.engine.region.pipeline import resolve_variable
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve
from tests.engine._quantity import NO_QUANTITY
from tests.engine.test_qualifier_suppression_and_provenance import _e2e_graph_for
from tests.fixtures import make_resolver

_QUERY = "standard grants"


class _FlowOpenLLM:
    """Elects the sole shape and leaves ``crsTypeOfFlow`` unbound.

    ``crsTypeOfFinance`` is eligible for text recognition."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(
                    axis="how", property_dcid="crsTypeOfFlow", kind="unbound", value_dcids=[]
                )
            ],
            also_plausible=[],
            ask=None,
        ), None


def _filter_for(spec, prop: str):
    return next(
        f
        for f in spec.resolved.trace.slot_filters
        if f.key.property is not None and f.key.property.ref == prop
    )


def test_grants_question_with_no_flow_type_defaults_crs_type_of_flow_to_oda():
    graph = _e2e_graph_for(_QUERY)

    result = asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_FlowOpenLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )

    assert result.status == "definite"
    spec = result.specs[0]

    # The default provenance is carried on the flow filter.
    flow_filter = _filter_for(spec, "crsTypeOfFlow")
    assert flow_filter.binding_kind == "value"
    assert [ref.ref for ref in flow_filter.refs] == ["CRSTypeOfFlow10"]
    assert flow_filter.provenance == "declared_default"

    # The default leaves text-recognised finance provenance unchanged.
    finance_filter = _filter_for(spec, "crsTypeOfFinance")
    assert finance_filter.provenance == "text_recognized"


def test_a_flow_type_named_in_text_keeps_its_own_binding_not_the_default():
    """Naming "ODA" binds the flow through text recognition."""
    query = "ODA standard grants"
    graph = _e2e_graph_for(query)

    result = asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_FlowOpenLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )

    assert result.status == "definite"
    spec = result.specs[0]
    flow_filter = _filter_for(spec, "crsTypeOfFlow")
    assert flow_filter.provenance == "text_recognized"


def test_non_concessional_aid_still_refuses_qualifier_not_bound():
    """Unsupported "non-concessional" text triggers ``qualifier_not_bound`` even when the
    declared flow default applies."""
    query = "non-concessional aid to Kenya"
    graph = _e2e_graph_for(query)

    result = asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_FlowOpenLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )

    assert result.status == "no_data"
    assert result.no_data_reason == "qualifier_not_bound"


class _ExtractFlowOpenLLM(_FlowOpenLLM):
    """Adds scripted raw-text extraction of Kenya and the requested variables."""

    def __init__(self, variables: list[str]) -> None:
        self._variables = variables

    async def generate_structured(self, *, prompt, system, schema):
        if schema.__name__ == "Extraction":
            return Extraction(
                kind="resolution",
                variables=self._variables,
                entities=[ExtractedEntity(text="Kenya")],
                dates=[],
                dropped_qualifiers=[],
            ), None
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


def _flow_filters(response) -> list:
    return [
        _filter_for(part.spec, "crsTypeOfFlow") for part in response.parts if part.spec is not None
    ]


def test_single_variable_raw_query_naming_aid_defaults_the_flow_type():
    """A single extracted variable lets the default match "aid" in the raw query."""
    response = offline_resolve(
        make_request("How much aid did Kenya get?"),
        graph=_e2e_graph_for("funding"),
        llm=_ExtractFlowOpenLLM(["funding"]),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    assert response.status == "definite"
    [flow_filter] = _flow_filters(response)
    assert flow_filter.provenance == "declared_default"
    assert [ref.ref for ref in flow_filter.refs] == ["CRSTypeOfFlow10"]


def test_two_variable_raw_query_naming_aid_leaves_the_flow_type_open():
    """Two extracted variables leave "aid" unattributed, so both flow slots stay open."""
    response = offline_resolve(
        make_request("How much aid did Kenya get?"),
        graph=_e2e_graph_for("funding", "loans"),
        llm=_ExtractFlowOpenLLM(["funding", "loans"]),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    flow_filters = _flow_filters(response)
    assert len(flow_filters) == 2
    for flow_filter in flow_filters:
        assert flow_filter.provenance == "binder"
        assert flow_filter.binding_kind == "unbound"
