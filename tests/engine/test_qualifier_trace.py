"""Pins query qualifier validation and compatibility with payloads that omit the field."""

from __future__ import annotations

from search.models import GraphRef, ResolutionTrace, ResolvedQualifier


def _minimal_trace(**overrides: object) -> ResolutionTrace:
    fields = {
        "resolved_entities": [],
        "resolved_sources": [],
        "slot_filters": [],
    }
    fields.update(overrides)
    return ResolutionTrace(**fields)  # type: ignore[arg-type]


def test_resolution_trace_defaults_query_qualifiers_to_empty_list():
    # Hand-built fixtures and resubmits omit this field.
    trace = _minimal_trace()
    assert trace.query_qualifiers == []


def test_resolution_trace_round_trips_query_qualifiers():
    qualifiers = [
        ResolvedQualifier(
            text="ODA",
            named=[GraphRef(dcid="CRSTypeOfFlow10", label="ODA")],
            slot="crsTypeOfFlow",
            disposition="bound",
        ),
        ResolvedQualifier(text="untied", named=[], slot=None, disposition="no_vocabulary"),
    ]
    trace = _minimal_trace(query_qualifiers=qualifiers)
    dumped = trace.model_dump(mode="json")
    restored = ResolutionTrace.model_validate(dumped)
    assert restored.query_qualifiers == qualifiers


def test_resolution_trace_validates_from_a_payload_predating_the_field():
    # Mirrors a serialized Spec minted before query_qualifiers existed: the
    # key is simply absent from the payload, not present-and-null.
    payload = {
        "resolved_entities": [],
        "resolved_sources": [],
        "slot_filters": [],
    }
    trace = ResolutionTrace.model_validate(payload)
    assert trace.query_qualifiers == []


def test_resolved_qualifier_defaults_slot_to_none():
    qualifier = ResolvedQualifier(text="grants", named=[], disposition="unbound")
    assert qualifier.slot is None


def test_resolved_qualifier_named_empty_for_no_vocabulary_disposition():
    qualifier = ResolvedQualifier(
        text="non-concessional", named=[], slot=None, disposition="no_vocabulary"
    )
    assert qualifier.named == []
    assert qualifier.disposition == "no_vocabulary"


def test_resolved_qualifier_carries_every_named_row_any_system():
    # "debt relief" names four rows across three systems -- named is not
    # limited to one code system.
    named = [
        GraphRef(dcid="CRSPurposeCode600", label="Action relating to debt"),
        GraphRef(dcid="CRSTypeOfFinance421", label="Debt forgiveness: ODA claims"),
        GraphRef(dcid="CRSTypeOfFinance422", label="Debt rescheduling: ODA claims"),
        GraphRef(dcid="CRSCooperationModalityB03", label="Debt relief"),
    ]
    qualifier = ResolvedQualifier(text="debt relief", named=named, slot=None, disposition="unbound")
    assert len(qualifier.named) == 4
    assert {ref.dcid for ref in qualifier.named} == {ref.dcid for ref in named}
