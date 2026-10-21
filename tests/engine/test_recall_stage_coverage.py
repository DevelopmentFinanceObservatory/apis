"""Tests coded-qualifier reattachment, CRS slot binding, and warning suppression.

The fixture uses a role-bearing CRS StatVar of the fixture index with finance and flow codes.
"""

from __future__ import annotations

from typing import Literal

from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.extract import ExtractedEntity, Extraction
from search.engine.quantity_read import QuantityOutput
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph

_QUERY = "Gross disbursements type of finance 110 type of flow 10 to Kenya"
# The model's variable phrase before coded qualifiers are reattached.
_UNFIXED_VARIABLE = "gross disbursements"
_QUALIFIER_PHRASE = "type of finance 110 type of flow 10"
_VARIABLE_WITH_CODES = f"{_UNFIXED_VARIABLE} {_QUALIFIER_PHRASE}"

_KEN_DCID = "country/KEN"
# A fixture CRS StatVar with five constraint properties.
_SV_DCID = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSPurposeCode99810_Nominal"
)
_FINANCE_PROPERTY = "crsTypeOfFinance"
_FLOW_PROPERTY = "crsTypeOfFlow"
_FINANCE_CODE = "CRSTypeOfFinance110"
_FLOW_CODE = "CRSTypeOfFlow10"

# Entity text, flow cue, and resolved axis supplied by the door.
_FlowRole = Literal["origin", "destination", "unspecified"]

_RANKING_QUERY = "top 10 countries by population"
_RANKING_QUALIFIER = "top 10 countries by population"


class _UnfixedExtractionLLM:
    """Leaves coded qualifiers dropped and drafts no bindings, isolating admission behaviour."""

    def __init__(
        self,
        variables: list[str],
        dropped_qualifiers: list[str],
        entities: list[tuple[str, str, _FlowRole]],
    ) -> None:
        self._variables = variables
        self._dropped = dropped_qualifiers
        self._entities = entities

    async def generate_structured(self, *, prompt, system, schema):
        if schema.__name__ == "Extraction":
            return Extraction(
                kind="resolution",
                variables=self._variables,
                entities=[
                    ExtractedEntity(
                        text=text, span=text, flow_role=flow_role, role_span=role_span
                    )  # span is the place's own surface, role_span the "to" cue
                    for text, role_span, flow_role in self._entities
                ],
                dates=[],
                dropped_qualifiers=self._dropped,
            ), None
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=[], ask=None
        ), None


class _DetectSpyGraph(FakeGraph):
    """Records the query passed to `detect_svs`."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detect_calls: list[str] = []

    def detect_svs(self, query: str):
        self.detect_calls.append(query)
        return super().detect_svs(query)


def _crs_graph() -> _DetectSpyGraph:
    """Build a graph for the fixture CRS StatVar and composed detect query.

    Kenya's observations use the `financingRecipient` role tuple in the SDMX fixture.
    """
    return _DetectSpyGraph(
        detect={_VARIABLE_WITH_CODES: {"svs": [_SV_DCID], "cosine_scores": [0.9]}},
        resolve={"Kenya": _KEN_DCID},
        obs={},
    )


def _crs_sdmx() -> FakeSdmxClient:
    return FakeSdmxClient(
        observations={
            _SV_DCID: {
                (("financingRecipient", _KEN_DCID),): [
                    Facet(
                        earliest_date="2020",
                        latest_date="2020",
                        obs_count=3,
                        dates=["2020"],
                        values=[1.0],
                        facet_id="F1",
                        provenance_id="provenance/test",
                    )
                ]
            }
        },
        availability={},
    )


def _resolve_coded_qualifier_query(graph: _DetectSpyGraph | None = None):
    return offline_resolve(
        make_request(_QUERY),
        graph=graph if graph is not None else _crs_graph(),
        llm=_UnfixedExtractionLLM(
            variables=[_UNFIXED_VARIABLE],
            dropped_qualifiers=[_QUALIFIER_PHRASE],
            entities=[("Kenya", "to", "destination")],
        ),
        sdmx=_crs_sdmx(),
    )


def _bound_value(response, property_dcid: str) -> str | None:
    part = response.parts[0]
    slot = next(
        s
        for s in part.spec.resolved.slots
        if s.key.property is not None and s.key.property.ref == property_dcid
    )
    assert slot.binding.kind == "value"
    return slot.binding.value.ref.ref


def test_coded_qualifier_query_ships_no_dropped_warning_and_binds_both_codes():
    """Both codes bind and produce no dropped-qualifier warning."""
    response = _resolve_coded_qualifier_query()

    assert response.status == "definite"
    assert QUERY_QUALIFIER_DROPPED not in [w.code for w in response.diagnostics.warnings]
    part = response.parts[0]
    assert part.status == "definite"
    assert "was not applied" not in part.spec.resolved.rendered_sentence

    assert _bound_value(response, _FINANCE_PROPERTY) == _FINANCE_CODE
    assert _bound_value(response, _FLOW_PROPERTY) == _FLOW_CODE


def test_coded_qualifier_codes_are_asserted_on_the_resolved_stat_var():
    """The recalled StatVar's slot values carry both codes."""
    response = _resolve_coded_qualifier_query()
    stat_var = response.parts[0].spec.resolved.members[0]
    assert stat_var.ref.ref == _SV_DCID
    rendered = {sv.key.property.ref: sv.value.ref.ref for sv in stat_var.slot_values}
    assert rendered[_FINANCE_PROPERTY] == _FINANCE_CODE
    assert rendered[_FLOW_PROPERTY] == _FLOW_CODE

    provenance = {
        f.key.property.ref: f.provenance
        for f in response.parts[0].spec.resolved.trace.slot_filters
        if f.key.property is not None
    }
    assert provenance[_FINANCE_PROPERTY] == "text_recognized"
    assert provenance[_FLOW_PROPERTY] == "text_recognized"


def test_ranking_qualifier_still_ships_query_qualifier_dropped():
    """The ranking phrase contains no code token and keeps its dropped-qualifier warning."""
    response = offline_resolve(
        make_request(_RANKING_QUERY),
        graph=FakeGraph(detect={"population": {"svs": ["Count_Person"]}}, obs={}),
        llm=_UnfixedExtractionLLM(
            variables=["population"],
            dropped_qualifiers=[_RANKING_QUALIFIER],
            entities=[],
        ),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )

    warnings = [w for w in response.diagnostics.warnings if w.code == QUERY_QUALIFIER_DROPPED]
    assert [w.message for w in warnings] == [
        f'The requested qualifier "{_RANKING_QUALIFIER}" was not applied.'
    ]


def test_detect_query_is_composed_and_differs_from_the_bare_variable():
    """Recall receives `_VARIABLE_WITH_CODES`; the bind prompt keeps `_UNFIXED_VARIABLE`.

    Reattachment adds codes to `detect_query`, not `Extraction.variables`.
    """
    graph = _crs_graph()
    response = _resolve_coded_qualifier_query(graph)

    assert response.status == "definite"
    assert graph.detect_calls == [_VARIABLE_WITH_CODES]
    assert _VARIABLE_WITH_CODES != _UNFIXED_VARIABLE
