"""Tests qualifier admission in warnings, filter provenance, and same-shape merges.

Fixtures use fixture CRS data and one synthetic code system.
"""

from __future__ import annotations

import asyncio

import pytest

from code_index.query import CodeIndex
from code_index.types import CodeRow, CodeSystemDeclaration, IndexStamp
from search.engine.assemble import _surviving_qualifier_warnings, build_definite_part
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.extract import ExtractedEntity, Extraction
from search.engine.quantity_read import QuantityOutput
from search.engine.region import RegionResult
from search.engine.region import phase_election as phase_election_module
from search.engine.region import phase_rank_bind as phase_rank_bind_module
from search.engine.region import related_read as related_read_module
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED, SLOT_DEFAULTED
from search.engine.region.pipeline import resolve_variable
from search.engine.same_shape import collapse_same_shape
from search.engine.spec_build import build_spec
from search.models import CoverageBare, StatVar, Warning
from tests.engine._binders import elected_reply
from tests.engine._builders import conjoin_spec as _spec
from tests.engine._builders import gref as _gr
from tests.engine._builders import shape_five_tuple as _shape
from tests.engine._builders import value_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

# ---------------------------------------------------------------------------
# Half one: warning/sentence-note suppression, through the real raw_text door
# ---------------------------------------------------------------------------

_E2E_QUERY = "ODA standard grants"
_E2E_SV_DCID = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode11110_Nominal"
)
_E2E_KEN_DCID = "country/KEN"


def _e2e_sv_arcs() -> dict:
    from code_index.shared import get_index

    row = next(r for r in get_index().statvars if r.dcid == _E2E_SV_DCID)
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": _E2E_SV_DCID}]},
        "populationType": {"nodes": [{"dcid": "FinancialFlow"}]},
        "measuredProperty": {"nodes": [{"dcid": "grantEquivalent"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
    }


_E2E_NODES = {
    _E2E_KEN_DCID: {"label": "Kenya", "type": "Country"},
    _E2E_SV_DCID: {
        "label": "test grant equivalent flow",
        "type": "StatisticalVariable",
        "arcs": _e2e_sv_arcs(),
    },
    "CRSTypeOfFlow10": {"label": "ODA"},
    "CRSTypeOfFinance110": {"label": "standard grant"},
}
_E2E_DETECT = {_E2E_QUERY: {"svs": [_E2E_SV_DCID], "cosine_scores": [0.9]}}
_E2E_OBS = {
    f"{_E2E_SV_DCID}|{_E2E_KEN_DCID}": [
        {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
    ]
}


class _ExtractAndBindLLM:
    """Returns scripted extraction and drafts no CRS-axis bindings."""

    def __init__(
        self, dropped_qualifiers: list[str], variables: tuple[str, ...] = (_E2E_QUERY,)
    ) -> None:
        self._dropped = dropped_qualifiers
        self._variables = list(variables)

    async def generate_structured(self, *, prompt, system, schema):
        if schema.__name__ == "Extraction":
            return Extraction(
                kind="resolution",
                variables=self._variables,
                entities=[ExtractedEntity(text="Kenya")],
                dates=[],
                dropped_qualifiers=self._dropped,
            ), None
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, also_plausible=[]), None


def _e2e_graph() -> FakeGraph:
    return FakeGraph(
        nodes=_E2E_NODES, detect=_E2E_DETECT, resolve={"Kenya": _E2E_KEN_DCID}, obs=_E2E_OBS
    )


def _e2e_graph_for(*detect_queries: str) -> FakeGraph:
    """Build the fixture graph with detect entries for each variable phrase."""
    detect = {q: {"svs": [_E2E_SV_DCID], "cosine_scores": [0.9]} for q in detect_queries}
    return FakeGraph(
        nodes=_E2E_NODES, detect=detect, resolve={"Kenya": _E2E_KEN_DCID}, obs=_E2E_OBS
    )


def test_admitted_dropped_qualifier_suppresses_warning_and_sentence_note():
    """A query whose dropped qualifier the pipeline admitted returns no
    query_qualifier_dropped warning for that phrase, and its sentence
    carries no terminal note for it."""
    response = offline_resolve(
        make_request(_E2E_QUERY),
        graph=_e2e_graph(),
        llm=_ExtractAndBindLLM(dropped_qualifiers=["standard grants"]),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    assert response.status == "definite"
    codes = [w.code for w in response.diagnostics.warnings]
    assert QUERY_QUALIFIER_DROPPED not in codes
    part = response.parts[0]
    assert part.status == "definite"
    finance_slot = next(
        s
        for s in part.spec.resolved.slots
        if s.key.property is not None and s.key.property.ref == "crsTypeOfFinance"
    )
    assert finance_slot.binding.kind == "value"
    assert "was not applied" not in part.spec.resolved.rendered_sentence


def test_two_dropped_qualifiers_keeps_only_the_unadmitted_one():
    """Two dropped qualifiers, one admitted and one not, keeps exactly the
    unadmitted one, as a warning and as a departure on the part."""
    response = offline_resolve(
        make_request(_E2E_QUERY),
        graph=_e2e_graph(),
        llm=_ExtractAndBindLLM(
            dropped_qualifiers=["standard grants", "an entirely unrelated qualifier"]
        ),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    warnings = [w for w in response.diagnostics.warnings if w.code == QUERY_QUALIFIER_DROPPED]
    assert len(warnings) == 1
    assert "an entirely unrelated qualifier" in warnings[0].message
    assert "standard grants" not in warnings[0].message
    assert "was not applied" not in response.parts[0].spec.resolved.rendered_sentence
    assert [
        (d.code, d.asked) for d in response.parts[0].departures if d.code == "measure_not_applied"
    ] == [("measure_not_applied", "an entirely unrelated qualifier")]


def test_unadmitted_dropped_qualifier_survives_unchanged():
    """A dropped qualifier no region admitted survives unchanged -- the
    regression guard that behaviour still works unchanged."""
    response = offline_resolve(
        make_request(_E2E_QUERY),
        graph=_e2e_graph(),
        llm=_ExtractAndBindLLM(dropped_qualifiers=["a phrase naming nothing in the index"]),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    warnings = [w for w in response.diagnostics.warnings if w.code == QUERY_QUALIFIER_DROPPED]
    assert len(warnings) == 1
    assert warnings[0].message == (
        'The requested qualifier "a phrase naming nothing in the index" was not applied.'
    )
    assert warnings[0].message not in response.parts[0].spec.resolved.rendered_sentence
    assert [
        (d.code, d.asked) for d in response.parts[0].departures if d.code == "measure_not_applied"
    ] == [("measure_not_applied", "a phrase naming nothing in the index")]


def test_terminal_before_election_reports_no_admitted_phrases():
    """An `ask` terminal occurs before election, so no qualifier is admitted."""

    class _AskLLM:
        def __init__(self, dropped_qualifiers: list[str]) -> None:
            self._dropped = dropped_qualifiers

        async def generate_structured(self, *, prompt, system, schema):
            if schema.__name__ == "Extraction":
                return Extraction(
                    kind="resolution",
                    variables=[_E2E_QUERY],
                    entities=[ExtractedEntity(text="Kenya")],
                    dates=[],
                    dropped_qualifiers=self._dropped,
                ), None
            if schema is QuantityOutput:
                return NO_QUANTITY
            assert schema.__name__ == "_UniversalBindOutput"
            return _UniversalBindOutput(
                chosen_shape_index=0, bindings=[], also_plausible=[], ask="no shape matches"
            ), None

    response = offline_resolve(
        make_request(_E2E_QUERY),
        graph=_e2e_graph(),
        llm=_AskLLM(dropped_qualifiers=["standard grants"]),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    assert response.parts[0].status == "no_data"
    warnings = [w for w in response.diagnostics.warnings if w.code == QUERY_QUALIFIER_DROPPED]
    assert len(warnings) == 1
    assert warnings[0].message == 'The requested qualifier "standard grants" was not applied.'


# ---------------------------------------------------------------------------
# A measure phrase entry/raw_text.py already owns: the region must not warn
# on it a second time. _E2E_SV_DCID is a grantEquivalent shape and carries no
# nettingBasis (see the real committed index), so "net"/"gross" -- both
# nettingBasis matches -- are always unsatisfied against it.
# ---------------------------------------------------------------------------


def test_dropped_phrase_measure_match_warns_exactly_once_across_both_surfaces():
    """The entry and region warning surfaces produce one warning for `net`, not two."""
    response = offline_resolve(
        make_request(_E2E_QUERY),
        graph=_e2e_graph(),
        llm=_ExtractAndBindLLM(dropped_qualifiers=["net"]),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    envelope_warnings = [
        w for w in response.diagnostics.warnings if w.code == QUERY_QUALIFIER_DROPPED
    ]
    part_warnings = [w for w in response.parts[0].warnings if w.code == QUERY_QUALIFIER_DROPPED]
    assert len(envelope_warnings) + len(part_warnings) == 1
    assert (envelope_warnings + part_warnings)[0].message == (
        'The requested qualifier "net" was not applied.'
    )


def test_two_conflicting_dropped_phrases_each_warn_exactly_once_across_both_surfaces():
    """Conflicting `gross` and `net` phrases each produce one warning across both surfaces."""
    response = offline_resolve(
        make_request(_E2E_QUERY),
        graph=_e2e_graph(),
        llm=_ExtractAndBindLLM(dropped_qualifiers=["gross", "net"]),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    envelope_warnings = [
        w for w in response.diagnostics.warnings if w.code == QUERY_QUALIFIER_DROPPED
    ]
    part_warnings = [w for w in response.parts[0].warnings if w.code == QUERY_QUALIFIER_DROPPED]
    all_messages = [w.message for w in envelope_warnings + part_warnings]
    assert all_messages.count('The requested qualifier "gross" was not applied.') == 1
    assert all_messages.count('The requested qualifier "net" was not applied.') == 1
    assert len(all_messages) == 2


# ---------------------------------------------------------------------------
# A dropped qualifier the variable text never itself carries
# ---------------------------------------------------------------------------


def test_dropped_qualifier_absent_from_variable_text_is_still_admitted():
    """A dropped qualifier can bind a slot even when it is absent from the variable text."""
    response = offline_resolve(
        make_request("ODA at standard grant terms"),
        graph=_e2e_graph_for("ODA"),
        llm=_ExtractAndBindLLM(dropped_qualifiers=["standard grants"], variables=("ODA",)),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    assert response.status == "definite"
    codes = [w.code for w in response.diagnostics.warnings]
    assert QUERY_QUALIFIER_DROPPED not in codes
    part = response.parts[0]
    assert part.status == "definite"
    finance_slot = next(
        s
        for s in part.spec.resolved.slots
        if s.key.property is not None and s.key.property.ref == "crsTypeOfFinance"
    )
    assert finance_slot.binding.kind == "value"
    assert "was not applied" not in part.spec.resolved.rendered_sentence


def test_two_variable_query_attributes_its_dropped_qualifier_to_the_leg_that_can_admit_it():
    """The N>=2 entry broadcasts dropped phrases to each leg; admission by the ODA leg suppresses
    the request-wide warning even though the sibling leg has no data.
    """
    response = offline_resolve(
        make_request("ODA and health spending"),
        graph=_e2e_graph_for("ODA", "health spending"),
        llm=_ExtractAndBindLLM(
            dropped_qualifiers=["standard grants"], variables=("ODA", "health spending")
        ),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    warnings = [w for w in response.diagnostics.warnings if w.code == QUERY_QUALIFIER_DROPPED]
    assert warnings == []

    oda_part = next(p for p in response.parts if p.variable_text == "ODA")
    assert oda_part.status == "definite"
    finance_slot = next(
        s
        for s in oda_part.spec.resolved.slots
        if s.key.property is not None and s.key.property.ref == "crsTypeOfFinance"
    )
    assert finance_slot.binding.kind == "value"


def test_surviving_qualifier_warnings_is_request_wide_across_regions():
    """_surviving_qualifier_warnings drops a phrase once ANY region admitted
    it, not only the first."""
    admitted_region = RegionResult(
        variable_text="a",
        status="no_data",
        specs=(),
        no_data_reason="variable_not_resolved",
        warnings=(),
        timing_by_step={},
        admitted_qualifier_phrases=frozenset({"standard grants"}),
    )
    unadmitted_region = RegionResult(
        variable_text="b",
        status="no_data",
        specs=(),
        no_data_reason="variable_not_resolved",
        warnings=(),
        timing_by_step={},
    )
    warnings = [
        Warning(
            code=QUERY_QUALIFIER_DROPPED,
            severity="degraded",
            message='The requested qualifier "Standard Grants" was not applied.',
        )
    ]
    survivors = _surviving_qualifier_warnings(
        warnings, ["Standard Grants"], [unadmitted_region, admitted_region]
    )
    assert survivors == []


# ---------------------------------------------------------------------------
# Half one: same-shape merge unions admitted_qualifier_phrases
# ---------------------------------------------------------------------------

_PERSON_SHAPE = _shape("Person", "count")


def test_same_shape_merge_over_two_regions_one_admitting_suppresses_the_phrase():
    """A same-shape merge over two regions, one of which admitted the
    phrase, suppresses it."""
    spec_a = _spec(_PERSON_SHAPE, "ValA", sv_dcid="sv")
    spec_b = _spec(_PERSON_SHAPE, "ValB", sv_dcid="sv")
    region_a = RegionResult(
        variable_text="var A",
        status="definite",
        specs=(spec_a,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        earliest_index=0,
        admitted_qualifier_phrases=frozenset({"standard grants"}),
    )
    region_b = RegionResult(
        variable_text="var B",
        status="definite",
        specs=(spec_b,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        earliest_index=1,
    )
    effective, residual = collapse_same_shape([region_a, region_b])
    assert residual == []
    assert len(effective) == 1
    merged = effective[0]
    assert merged.admitted_qualifier_phrases == frozenset({"standard grants"})


def test_same_shape_merge_states_a_shared_slot_default_once():
    """Two merged legs that each default the flow give one answer, so the merged
    region carries one slot_defaulted warning. The warning is the election's and stays
    off the sentence. A warning only one leg raised survives."""
    note = (
        "No flow was named, so the answer covers official development assistance, the default flow."
    )
    slot_warning = Warning(code=SLOT_DEFAULTED, severity="warn", message=note)
    other = Warning(code="date_coverage_partial", severity="info", message="Only B is partial.")
    regions = [
        RegionResult(
            variable_text=f"var {value}",
            status="definite",
            specs=(_spec(_PERSON_SHAPE, f"Val{value}", sv_dcid="sv"),),
            no_data_reason=None,
            warnings=warnings,
            timing_by_step={},
            earliest_index=index,
        )
        for index, (value, warnings) in enumerate(
            [("A", (slot_warning,)), ("B", (slot_warning, other))]
        )
    ]
    effective, residual = collapse_same_shape(regions)
    assert residual == []
    assert len(effective) == 1
    merged = effective[0]
    assert merged.warnings == (slot_warning, other)
    part = build_definite_part(
        merged.spec,
        variable_text=merged.variable_text,
        warnings=list(merged.warnings),
    )
    assert [w.code for w in part.warnings].count(SLOT_DEFAULTED) == 1
    assert note not in (part.spec.resolved.rendered_sentence or "")


# ---------------------------------------------------------------------------
# Half two: ResolvedFilter.provenance
# ---------------------------------------------------------------------------


def test_spec_id_is_byte_identical_with_and_without_provenance_populated():
    """Populating filter provenance does not change `spec_id`."""
    shape = _PERSON_SHAPE
    slots = [_spec(shape, "ValA", sv_dcid="sv").resolved.slots[0]]
    stat_vars = [
        StatVar(ref=_gr("sv"), shape_id=shape.shape_id, slot_values=[], constraint_properties=[])
    ]
    common = {
        "shape": shape,
        "slots": slots,
        "stat_vars": stat_vars,
        "entities": [],
        "coverage": CoverageBare(has_data=True),
        "open_roles": (),
    }
    without = build_spec(measure_elected_properties=frozenset(), **common)
    with_provenance = build_spec(
        measure_elected_properties=frozenset(),
        **common,
        text_recognized_properties=frozenset({"constraintProp"}),
    )
    assert without.spec_id == with_provenance.spec_id
    assert without.definition.identity == with_provenance.definition.identity
    assert without.resolved.trace.slot_filters[0].provenance == "binder"
    assert with_provenance.resolved.trace.slot_filters[0].provenance == "text_recognized"


def test_measure_elected_properties_stamps_forced_slot_and_leaves_the_rest():
    """The three provenances build_spec can stamp: a property in
    measure_elected_properties gets "measure_elected", one in
    text_recognized_properties keeps "text_recognized", and every other
    slot's property gets "binder"."""
    shape = _PERSON_SHAPE
    slots = [
        value_slot("what", "textProp", "textProp", "val1"),
        value_slot("what", "measureProp", "measureProp", "val2"),
        value_slot("what", "binderProp", "binderProp", "val3"),
    ]
    stat_vars = [
        StatVar(ref=_gr("sv"), shape_id=shape.shape_id, slot_values=[], constraint_properties=[])
    ]
    spec = build_spec(
        open_roles=(),
        shape=shape,
        slots=slots,
        stat_vars=stat_vars,
        entities=[],
        coverage=CoverageBare(has_data=True),
        text_recognized_properties=frozenset({"textProp"}),
        measure_elected_properties=frozenset({"measureProp"}),
    )

    def _provenance(prop: str) -> str:
        return next(
            f for f in spec.resolved.trace.slot_filters if f.key.property.ref == prop
        ).provenance

    assert _provenance("textProp") == "text_recognized"
    assert _provenance("measureProp") == "measure_elected"
    assert _provenance("binderProp") == "binder"


def test_measure_elected_and_text_recognized_overlap_raises():
    """A property present in both text_recognized_properties and
    measure_elected_properties is a caller bug -- the sets are disjoint by
    construction (the measure-family election only forces an axis still
    unbound or absent), so build_spec raises ValueError naming it rather
    than silently picking one provenance."""
    shape = _PERSON_SHAPE
    slots = [value_slot("what", "sharedProp", "sharedProp", "val1")]
    stat_vars = [
        StatVar(ref=_gr("sv"), shape_id=shape.shape_id, slot_values=[], constraint_properties=[])
    ]

    with pytest.raises(ValueError, match="sharedProp"):
        build_spec(
            open_roles=(),
            shape=shape,
            slots=slots,
            stat_vars=stat_vars,
            entities=[],
            coverage=CoverageBare(has_data=True),
            text_recognized_properties=frozenset({"sharedProp"}),
            measure_elected_properties=frozenset({"sharedProp"}),
        )


def test_merged_spec_keeps_provenance_through_collapse_same_shape():
    """The same merged spec keeps the provenance through
    collapse_same_shape -- unioned across the group, so a property only ONE
    member forced still reads text_recognized on the merged spec."""
    spec_a = _spec(
        _PERSON_SHAPE,
        "ValA",
        sv_dcid="sv",
        text_recognized_properties=frozenset({"constraintProp"}),
    )
    spec_b = _spec(_PERSON_SHAPE, "ValB", sv_dcid="sv")  # binder (the default)
    region_a = RegionResult(
        variable_text="var A",
        status="definite",
        specs=(spec_a,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        earliest_index=0,
    )
    region_b = RegionResult(
        variable_text="var B",
        status="definite",
        specs=(spec_b,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        earliest_index=1,
    )
    effective, residual = collapse_same_shape([region_a, region_b])
    assert residual == []
    merged_spec = effective[0].spec
    assert merged_spec.resolved.slots[0].binding.kind == "set"
    merged_filter = next(
        f for f in merged_spec.resolved.trace.slot_filters if f.key.property.ref == "constraintProp"
    )
    assert merged_filter.provenance == "text_recognized"


def test_merged_spec_keeps_measure_elected_provenance_through_collapse_same_shape():
    """Same union rule as test_merged_spec_keeps_provenance_through_
    collapse_same_shape above, for measure_elected_properties: a property
    only ONE group member's own measure election forced still reads
    measure_elected on the merged spec."""
    spec_a = _spec(
        _PERSON_SHAPE,
        "ValA",
        sv_dcid="sv",
        measure_elected_properties=frozenset({"constraintProp"}),
    )
    spec_b = _spec(_PERSON_SHAPE, "ValB", sv_dcid="sv")  # binder (the default)
    region_a = RegionResult(
        variable_text="var A",
        status="definite",
        specs=(spec_a,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        earliest_index=0,
    )
    region_b = RegionResult(
        variable_text="var B",
        status="definite",
        specs=(spec_b,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        earliest_index=1,
    )
    effective, residual = collapse_same_shape([region_a, region_b])
    assert residual == []
    merged_spec = effective[0].spec
    merged_filter = next(
        f for f in merged_spec.resolved.trace.slot_filters if f.key.property.ref == "constraintProp"
    )
    assert merged_filter.provenance == "measure_elected"


def test_merged_spec_prefers_text_recognized_over_measure_elected_on_conflict():
    """A same-shape merge only requires the group's one differing slot to
    agree in value (collapse_same_shape's own eligibility gate) -- every
    other slot, including constraintProp here, can disagree in HOW each
    member arrived at the same value. One member's own query text named
    constraintProp (text_recognized); another member's shape left it open
    for its own measure election to force (measure_elected). build_spec's
    own overlap check would refuse a property present in both sets, so
    same_shape.py resolves the conflict before calling it: text_recognized
    wins, because the query named it on at least one member. Proves the
    merge neither raises nor silently reads measure_elected."""
    spec_a = _spec(
        _PERSON_SHAPE,
        "ValA",
        sv_dcid="sv",
        text_recognized_properties=frozenset({"constraintProp"}),
    )
    spec_b = _spec(
        _PERSON_SHAPE,
        "ValB",
        sv_dcid="sv",
        measure_elected_properties=frozenset({"constraintProp"}),
    )
    region_a = RegionResult(
        variable_text="var A",
        status="definite",
        specs=(spec_a,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        earliest_index=0,
    )
    region_b = RegionResult(
        variable_text="var B",
        status="definite",
        specs=(spec_b,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        earliest_index=1,
    )
    effective, residual = collapse_same_shape([region_a, region_b])
    assert residual == []
    merged_spec = effective[0].spec
    merged_filter = next(
        f for f in merged_spec.resolved.trace.slot_filters if f.key.property.ref == "constraintProp"
    )
    assert merged_filter.provenance == "text_recognized"


def test_e2e_forced_binding_carries_text_recognized_provenance_binder_elects_the_rest():
    """A forced binding reaches trace.slot_filters carrying the
    text-recognition provenance; a binding the model elected on the same
    shape carries the binder's."""
    graph = _e2e_graph()

    class _PartlyElectingLLM:
        """Elects the recalled shape and drafts one binding of its OWN, on an
        axis admission never touches (crsCooperationModality), alongside the
        two axes "ODA standard grants" recognises (crsTypeOfFlow,
        crsTypeOfFinance) -- both forced only if admission actually ran."""

        async def generate_structured(self, *, prompt, system, schema):
            if schema is QuantityOutput:
                return NO_QUANTITY
            assert schema.__name__ == "_UniversalBindOutput"
            return elected_reply(
                0,
                bindings=[
                    SlotBindingDraft(
                        axis="how",
                        property_dcid="crsCooperationModality",
                        kind="value",
                        value_dcids=["CRSCooperationModalityA02"],
                    )
                ],
                also_plausible=[],
            ), None

    result = asyncio.run(
        resolve_variable(
            _E2E_QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_E2E_QUERY,
            role_query=_E2E_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_PartlyElectingLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "definite"
    slot_filters = result.specs[0].resolved.trace.slot_filters

    def _provenance(prop: str) -> str:
        return next(
            f for f in slot_filters if f.key.property is not None and f.key.property.ref == prop
        ).provenance

    assert _provenance("crsTypeOfFlow") == "text_recognized"
    assert _provenance("crsTypeOfFinance") == "text_recognized"
    assert _provenance("crsCooperationModality") == "binder"
    assert result.admitted_qualifier_phrases == frozenset({"oda", "standard grants"})


# ---------------------------------------------------------------------------
# Generalisation: a synthetic non-CRS code system, exercising both halves --
# never reaches any committed export.
# ---------------------------------------------------------------------------


def _synthetic_index() -> CodeIndex:
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="synthetic",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={"syntheticAxis": 1},
            statvar_shape_counts={},
            content_hash="synthetic",
        ),
        declarations=(
            CodeSystemDeclaration(
                code_system="syntheticAxis",
                hierarchical=False,
                disjoint_from_children=True,
                summable=True,
                absent_reading="not_declared",
                plain_name="axis",
            ),
        ),
        codes=(
            CodeRow(
                dcid="SyntheticAxisCodeAlpha",
                code_system="syntheticAxis",
                code="alpha",
                label="synthetic widget grade",
                synonyms=(),
                parent=None,
            ),
        ),
        statvars=(),
    )


_SYN_QUERY = "spending on synthetic widget grade equipment"
_SYN_SV_DCID = "SyntheticFlow_syntheticAxisSyntheticAxisCodeAlpha"
_SYN_KEN_DCID = "country/KEN"
_SYN_NODES = {
    _SYN_KEN_DCID: {"label": "Kenya", "type": "Country"},
    _SYN_SV_DCID: {
        "label": "synthetic flow",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": _SYN_SV_DCID}]},
            "populationType": {"nodes": [{"dcid": "SyntheticFlow"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "constraintProperties": {"nodes": [{"dcid": "syntheticAxis"}]},
            "syntheticAxis": {"nodes": [{"dcid": "SyntheticAxisCodeAlpha"}]},
        },
    },
    "SyntheticAxisCodeAlpha": {"label": "synthetic widget grade"},
}
_SYN_DETECT = {_SYN_QUERY: {"svs": [_SYN_SV_DCID], "cosine_scores": [0.9]}}
_SYN_OBS = {
    f"{_SYN_SV_DCID}|{_SYN_KEN_DCID}": [
        {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
    ]
}


class _SyntheticNoBindingLLM:
    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, also_plausible=[]), None


def test_synthetic_code_system_admission_reaches_both_halves(monkeypatch, search_settings):
    """A synthetic non-CRS code system exercises both halves, proving
    neither reads a CRS name. "equipment" beside the synthetic code is a word no coverage
    reader covers, so the log mode keeps the answer definite."""
    search_settings(qualifier_coverage="log")
    index = _synthetic_index()
    monkeypatch.setattr(phase_rank_bind_module, "get_index", lambda: index)
    monkeypatch.setattr(phase_election_module, "get_index", lambda: index)
    monkeypatch.setattr(related_read_module, "get_index", lambda: index)
    graph = FakeGraph(
        nodes=_SYN_NODES, detect=_SYN_DETECT, resolve={"Kenya": _SYN_KEN_DCID}, obs=_SYN_OBS
    )
    result = asyncio.run(
        resolve_variable(
            _SYN_QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_SYN_QUERY,
            role_query=_SYN_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_SyntheticNoBindingLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "definite"
    assert result.admitted_qualifier_phrases == frozenset({"synthetic widget grade"})
    slot_filters = result.specs[0].resolved.trace.slot_filters
    synthetic_filter = next(f for f in slot_filters if f.key.property.ref == "syntheticAxis")
    assert synthetic_filter.provenance == "text_recognized"
    assert synthetic_filter.refs[0].ref == "SyntheticAxisCodeAlpha"
