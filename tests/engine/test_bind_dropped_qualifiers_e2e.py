"""End-to-end tests for requested-all qualifier bindings.

A verbatim claim suppresses the matching default and warning. An unapplied or invalid claim keeps
its warning. Invalid claims leave the default active. A two-variable query checks that extraction
broadcasts dropped phrases to both bind prompts.

The fixture uses the fixture CRS index and the query "standard grants", which leaves
crsTypeOfFlow open for the binder.
"""

from __future__ import annotations

import asyncio

import pytest

from search.engine.bind import SlotBindingDraft
from search.engine.extract import ExtractedEntity, Extraction
from search.engine.quantity_read import QuantityOutput
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.region.pipeline import resolve_variable
from search.models import UNIT_REF
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve
from tests.engine._quantity import NO_QUANTITY
from tests.engine.test_bind import (
    _RESOLVE_DETECT_QUERY,
    _RESOLVE_NODES,
    _RESOLVE_RESOLVE,
    _RESOLVE_SV,
)
from tests.engine.test_declared_default_e2e import _filter_for
from tests.engine.test_qualifier_suppression_and_provenance import (
    _E2E_KEN_DCID,
    _E2E_NODES,
    _E2E_OBS,
    _E2E_SV_DCID,
    _e2e_graph_for,
)
from tests.fixtures import FakeGraph, make_resolver

_QUERY = "standard grants"
_BREADTH_PHRASE = "across all flows"
_UNRELATED_PHRASE = "an entirely unrelated qualifier"
# Names a price basis beside two currencies, which the unit binder refuses, so it stays dropped.
_PRICE_BASIS_PHRASE = "constant 2022 dollars and euros"


class _FlowRequestedAllLLM:
    """Elect the CRS shape and claim that the dropped breadth phrase applies to its flow slot."""

    def __init__(
        self, requested_all_phrase: str | None, property_dcid: str = "crsTypeOfFlow"
    ) -> None:
        self._phrase = requested_all_phrase
        self._property_dcid = property_dcid

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(
            0,
            bindings=[
                SlotBindingDraft(
                    axis="how",
                    property_dcid=self._property_dcid,
                    kind="unbound",
                    value_dcids=[],
                    evidence="requested_all",
                    requested_all_phrase=self._phrase,
                )
            ],
            also_plausible=[],
        ), None


class _ExtractAndFlowRequestedAllLLM(_FlowRequestedAllLLM):
    """Add a scripted extraction carrying dropped qualifiers."""

    def __init__(
        self,
        dropped_qualifiers: list[str],
        requested_all_phrase: str | None,
        property_dcid: str = "crsTypeOfFlow",
    ) -> None:
        super().__init__(requested_all_phrase, property_dcid)
        self._dropped = dropped_qualifiers

    async def generate_structured(self, *, prompt, system, schema):
        if schema.__name__ == "Extraction":
            return Extraction(
                kind="resolution",
                variables=[_QUERY],
                entities=[ExtractedEntity(text="Kenya")],
                dates=[],
                dropped_qualifiers=self._dropped,
            ), None
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


def test_requested_all_on_the_flow_slot_suppresses_the_declared_default_end_to_end():
    """A validated breadth claim leaves the fixture index's crsTypeOfFlow default unbound (issue #968)."""
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
            llm=_FlowRequestedAllLLM(requested_all_phrase=_BREADTH_PHRASE),
            base_timing={},
            dropped_qualifier_phrases=[_BREADTH_PHRASE],
            coverage=empty_coverage_snapshot(),
        )
    )

    assert result.status == "definite"
    spec = result.specs[0]
    flow_filter = _filter_for(spec, "crsTypeOfFlow")
    assert flow_filter.binding_kind == "unbound"
    assert flow_filter.provenance != "declared_default"

    # Text recognition binds crsTypeOfFinance independently of the breadth phrase.
    finance_filter = _filter_for(spec, "crsTypeOfFinance")
    assert finance_filter.provenance == "text_recognized"

    assert "across all flows" in result.admitted_qualifier_phrases


def test_applied_dropped_qualifier_suppresses_its_warning_unapplied_one_keeps_its_own():
    """Applying one of two dropped qualifiers suppresses only its warning, and only the
    unapplied one reaches the part as a departure."""
    response = offline_resolve(
        make_request(_QUERY),
        graph=_e2e_graph_for(_QUERY),
        llm=_ExtractAndFlowRequestedAllLLM(
            dropped_qualifiers=[_BREADTH_PHRASE, _UNRELATED_PHRASE],
            requested_all_phrase=_BREADTH_PHRASE,
        ),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    assert response.status == "definite"
    envelope_warnings = [
        w for w in response.diagnostics.warnings if w.code == QUERY_QUALIFIER_DROPPED
    ]
    part_warnings = [w for w in response.parts[0].warnings if w.code == QUERY_QUALIFIER_DROPPED]
    all_messages = [w.message for w in envelope_warnings + part_warnings]
    assert all_messages == [f'The requested qualifier "{_UNRELATED_PHRASE}" was not applied.']

    sentence = response.parts[0].spec.resolved.rendered_sentence
    assert "was not applied" not in sentence

    flow_filter = _filter_for(response.parts[0].spec, "crsTypeOfFlow")
    assert flow_filter.binding_kind == "unbound"
    assert [
        (d.code, d.binding_ref, d.asked)
        for d in response.parts[0].departures
        if d.code == "measure_not_applied"
    ] == [("measure_not_applied", None, _UNRELATED_PHRASE)]
    # The fixture's family has no total, so the part shows its closest measure.
    assert [w.code for w in response.parts[0].warnings] == ["family_total_absent"]
    assert [
        (d.code, d.binding_ref, d.asked, d.shown)
        for d in response.parts[0].departures
        if d.code == "closest_match"
    ] == [("closest_match", None, _QUERY, "test grant equivalent flow")]


@pytest.mark.parametrize(("unit", "shown"), [("USDollar", "current US dollars"), (None, None)])
def test_a_dropped_price_basis_phrase_points_its_departure_at_the_unit_token(unit, shown):
    """crs-038: extraction drops "constant 2022 dollars and euros", which names no single unit request. The unit token clears it, and the
    departure shows the dollars the byline states. An answer whose byline states no
    dollars (weo-002) has no prices to show, so the phrase departs not at all. The
    unrelated phrase departs on no slot either way."""
    obs = {key: [{**entry, "unit": unit} for entry in entries] for key, entries in _E2E_OBS.items()}
    graph = FakeGraph(
        nodes=_E2E_NODES,
        detect={_QUERY: {"svs": [_E2E_SV_DCID], "cosine_scores": [0.9]}},
        resolve={"Kenya": _E2E_KEN_DCID},
        obs=obs if unit else _E2E_OBS,
    )
    response = offline_resolve(
        make_request(_QUERY),
        graph=graph,
        llm=_ExtractAndFlowRequestedAllLLM(
            dropped_qualifiers=[_PRICE_BASIS_PHRASE, _UNRELATED_PHRASE], requested_all_phrase=None
        ),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    price_basis = [("measure_not_applied", UNIT_REF, _PRICE_BASIS_PHRASE, shown)] if shown else []
    assert [
        (d.code, d.binding_ref, d.asked, d.shown)
        for d in response.parts[0].departures
        if d.code == "measure_not_applied"
    ] == [
        *price_basis,
        ("measure_not_applied", None, _UNRELATED_PHRASE, None),
    ]


def test_a_hallucinated_requested_all_phrase_not_in_the_input_list_is_discarded():
    """A phrase absent from the extraction is rejected, leaving its warning and default active."""
    response = offline_resolve(
        make_request(_QUERY),
        graph=_e2e_graph_for(_QUERY),
        llm=_ExtractAndFlowRequestedAllLLM(
            dropped_qualifiers=[_BREADTH_PHRASE],
            requested_all_phrase="a phrase never offered to the model",
        ),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    assert response.status == "definite"
    envelope_warnings = [
        w for w in response.diagnostics.warnings if w.code == QUERY_QUALIFIER_DROPPED
    ]
    part_warnings = [w for w in response.parts[0].warnings if w.code == QUERY_QUALIFIER_DROPPED]
    all_messages = [w.message for w in envelope_warnings + part_warnings]
    assert all_messages == [f'The requested qualifier "{_BREADTH_PHRASE}" was not applied.']

    flow_filter = _filter_for(response.parts[0].spec, "crsTypeOfFlow")
    assert flow_filter.binding_kind == "value"
    assert flow_filter.provenance == "declared_default"


# A multi-variable request sends dropped qualifiers to every bind prompt.


class _TwoLegLLM:
    """Record each variable's bind prompt and claim the breadth phrase on the CRS shape."""

    def __init__(self) -> None:
        self.prompts_by_variable: dict[str, str] = {}

    async def generate_structured(self, *, prompt, system, schema):
        if schema.__name__ == "Extraction":
            return Extraction(
                kind="resolution",
                variables=[_QUERY, _RESOLVE_DETECT_QUERY],
                entities=[ExtractedEntity(text="Kenya")],
                dates=[],
                dropped_qualifiers=[_BREADTH_PHRASE],
            ), None
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        variable = next(
            line.removeprefix("Variable phrase: ")
            for line in prompt.splitlines()
            if line.startswith("Variable phrase: ")
        )
        self.prompts_by_variable[variable] = prompt
        if variable == _QUERY:
            return elected_reply(
                0,
                bindings=[
                    SlotBindingDraft(
                        axis="how",
                        property_dcid="crsTypeOfFlow",
                        kind="unbound",
                        value_dcids=[],
                        evidence="requested_all",
                        requested_all_phrase=_BREADTH_PHRASE,
                    )
                ],
                also_plausible=[],
            ), None
        # Count_Person has no constraint slots to bind.
        return elected_reply(0, bindings=[], also_plausible=[]), None


def _two_leg_graph() -> FakeGraph:
    return FakeGraph(
        nodes={**_E2E_NODES, **_RESOLVE_NODES},
        detect={
            _QUERY: {"svs": [_E2E_SV_DCID], "cosine_scores": [0.9]},
            _RESOLVE_DETECT_QUERY: {"svs": [_RESOLVE_SV], "entities": []},
        },
        resolve=_RESOLVE_RESOLVE,
        obs=_E2E_OBS,
    )


def test_two_variable_query_broadcasts_dropped_qualifiers_to_every_leg():
    """Both bind prompts receive the phrase.
    Only the CRS leg applies it, suppressing the warning.
    """
    llm = _TwoLegLLM()
    response = offline_resolve(
        make_request("standard grants and population"),
        graph=_two_leg_graph(),
        llm=llm,
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    # Both legs' own bind prompt carried the block -- the broadcast reaches
    # a leg whose shape has no matching slot exactly the same as it reaches
    # the CRS leg.
    assert set(llm.prompts_by_variable) == {_QUERY, _RESOLVE_DETECT_QUERY}
    for prompt in llm.prompts_by_variable.values():
        assert "DROPPED QUALIFIERS" in prompt
        assert _BREADTH_PHRASE in prompt

    warnings = [w for w in response.diagnostics.warnings if w.code == QUERY_QUALIFIER_DROPPED]
    assert warnings == []

    def _crs_flow_filter(part):
        if part.status != "definite":
            return None
        return next(
            (
                f
                for f in part.spec.resolved.trace.slot_filters
                if f.key.property is not None and f.key.property.ref == "crsTypeOfFlow"
            ),
            None,
        )

    flow_filters = [f for p in response.parts if (f := _crs_flow_filter(p)) is not None]
    assert len(flow_filters) == 1
    assert flow_filters[0].binding_kind == "unbound"
    assert flow_filters[0].provenance != "declared_default"


def test_a_requested_all_claim_on_a_property_the_elected_shape_lacks_keeps_its_warning():
    """A claim on a property absent from the elected shape leaves the warning intact."""
    response = offline_resolve(
        make_request(_QUERY),
        graph=_e2e_graph_for(_QUERY),
        llm=_ExtractAndFlowRequestedAllLLM(
            dropped_qualifiers=[_BREADTH_PHRASE],
            requested_all_phrase=_BREADTH_PHRASE,
            property_dcid="notARealSlotOnThisShape",
        ),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    assert response.status == "definite"
    messages = [
        w.message
        for w in response.diagnostics.warnings + response.parts[0].warnings
        if w.code == QUERY_QUALIFIER_DROPPED
    ]
    assert messages == [f'The requested qualifier "{_BREADTH_PHRASE}" was not applied.']
