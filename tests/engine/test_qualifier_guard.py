"""Test `_resolve_query_qualifiers` through `_phase_repair_bindings`.

Cases use the packaged code and measure indexes for real CRS terms, with
synthetic shapes that prevent measure-family election from changing bindings.
"""

from __future__ import annotations

import asyncio
from typing import cast

import pytest
from pydantic import ValidationError

from code_index.shared import get_index, get_measure_index
from entity_resolution import Admission, EntityResolver
from search.engine.bind import DraftBindingKind, SlotBindingDraft, _UniversalBindOutput
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.conjoin import _build_part
from search.engine.interpret import Recall
from search.engine.qualifier_match import recognise_spans
from search.engine.region.phase_election import _phase_repair_bindings
from search.engine.region.phase_rank_bind import _resolve_query_qualifiers
from search.engine.region.state import ResolveState
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.models import NoData, NoDataPart, ResolvedQualifier, Spec
from tests.conftest import minimal_spec
from tests.engine._fake_coverage import empty_coverage_snapshot

_POP_TYPE_DCID = "Widget"
_MEAS_PROP_DCID = "count"
_STAT_TYPE_DCID = "measuredValue"


def _shape(*slot_keys: SlotKeyDraft) -> ShapeDraft:
    return ShapeDraft(
        shape_id="widget",
        label="widget",
        pop_type_dcid=_POP_TYPE_DCID,
        meas_prop_dcid=_MEAS_PROP_DCID,
        stat_type_dcid=_STAT_TYPE_DCID,
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=slot_keys,
    )


def _binding(
    prop: str, *, kind: DraftBindingKind = "unbound", value_dcids: list[str] | None = None
) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how",
        property_dcid=prop,
        kind=kind,
        value_dcids=value_dcids or [],
    )


def _state(
    *,
    variable: str,
    shape_draft: ShapeDraft,
    bindings: list[SlotBindingDraft],
    unplaced: list[str],
    query_text: str | None = None,
) -> ResolveState:
    """Build the hand-crafted state consumed by qualifier repair."""
    index = get_index()
    measure_index = get_measure_index()
    return ResolveState(
        variable=variable,
        entities=["Kenya"],
        date_request=None,
        detect_query=variable,
        role_query=variable,
        query_text=query_text,
        pac=True,
        graph=cast(EngineGraphClient, None),
        resolver=cast(EntityResolver, None),
        admission=cast(Admission, None),
        llm=cast(SupportsLLM, None),
        sdmx=cast(EngineSdmxClient, None),
        deadline=None,
        pre_resolved=None,
        variable_index=0,
        emit=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        unit_text=None,
        source_text=None,
        materialise_graph=None,
        materialise_sdmx=None,
        pipeline_steps=[],
        timing={},
        warnings=[],
        # Recall resolved no place, so the minted identity binds no role key.
        rcl=Recall(candidates=[], resolutions={}, resolved_entity_names={}),
        bind_result=_UniversalBindOutput(
            chosen_shape_index=0,
            bindings=bindings,
            also_plausible=None,
            ask=None,
            unplaced=unplaced,
        ),
        shape_draft=shape_draft,
        variable_qualifier_spans=recognise_spans(variable, index, measure_index),
        coverage=empty_coverage_snapshot(),
    )


def _run(state: ResolveState) -> list[ResolvedQualifier]:
    result = asyncio.run(_phase_repair_bindings(state))
    assert result is None, "expected _phase_repair_bindings to fall through"
    return state.query_qualifiers


def _has_miss(qualifiers: list[ResolvedQualifier]) -> bool:
    """Return whether any qualifier remains unbound."""
    return any(q.disposition != "bound" for q in qualifiers)


# ---------------------------------------------------------------------------
# An unplaced phrase overlapping a bound span is ignored
# ---------------------------------------------------------------------------

_FLOW_SLOT = SlotKeyDraft(axis="what", property_dcid="crsTypeOfFlow", label="flow type")


def test_unplaced_measure_noun_is_dropped_by_the_overlap_rule():
    shape = _shape(_FLOW_SLOT)
    state = _state(
        variable="ODA to Kenya",
        shape_draft=shape,
        bindings=[_binding("crsTypeOfFlow", kind="value", value_dcids=["CRSTypeOfFlow10"])],
        unplaced=["ODA"],
    )
    qualifiers = _run(state)
    assert [q.text for q in qualifiers] == ["ODA"]
    assert qualifiers[0].disposition == "bound"
    assert qualifiers[0].slot == "crsTypeOfFlow"
    assert not _has_miss(qualifiers)


def test_unplaced_measure_noun_guard_fails_without_the_overlap_rule():
    """Without the overlap check, "ODA" becomes no_vocabulary and blocks the answer."""
    shape = _shape(_FLOW_SLOT)
    state = _state(
        variable="ODA to Kenya",
        shape_draft=shape,
        bindings=[_binding("crsTypeOfFlow", kind="value", value_dcids=["CRSTypeOfFlow10"])],
        unplaced=["ODA"],
    )
    # Bypass rule 4 by dropping every recognised span before the resolver
    # ever sees them -- the overlap check has nothing left to compare
    # against, so "ODA" survives into the unplaced/no_vocabulary branch
    # exactly as it would with rule 4 never written.
    state.variable_qualifier_spans = ()
    qualifiers = _resolve_query_qualifiers(state)
    by_text = {q.text: q for q in qualifiers}
    assert by_text["ODA"].disposition == "no_vocabulary"
    assert _has_miss(qualifiers)


# ---------------------------------------------------------------------------
# A per-person rate base the elected shape's denominator carries is placed
# ---------------------------------------------------------------------------

_PRODUCT_SLOT = SlotKeyDraft(axis="how", property_dcid="productType", label="product type")


def _per_person_shape(meas_denom_dcid: str | None) -> ShapeDraft:
    return ShapeDraft(
        shape_id="mobile",
        label="Mobile Phone Subscriptions Per Person in a Population",
        pop_type_dcid="Product",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=meas_denom_dcid,
        slot_keys=(_PRODUCT_SLOT,),
    )


@pytest.mark.parametrize(
    ("variable", "phrase"),
    [
        ("mobile phone subscriptions per 100 people", "per 100 people"),
        ("mobile phone subscriptions per capita", "per capita"),
        ("mobile phone subscriptions per 1,000 inhabitants", "per 1,000 inhabitants"),
    ],
)
def test_unplaced_per_person_rate_base_on_a_per_person_shape_is_not_refused(variable, phrase):
    """mobile-subscriptions-ghana: the bind model listed "per 100 people" as unplaced on the
    `Count_Person`-denominated shape it elected, and the phrase refused a correct answer."""
    state = _state(
        variable=variable,
        shape_draft=_per_person_shape("Count_Person"),
        bindings=[
            _binding("productType", kind="value", value_dcids=["MobileCellularSubscription"])
        ],
        unplaced=[phrase],
    )
    assert _resolve_query_qualifiers(state) == []


def test_unplaced_per_person_rate_base_without_a_per_person_denominator_is_refused():
    state = _state(
        variable="mobile phone subscriptions per 100 people",
        shape_draft=_per_person_shape(None),
        bindings=[
            _binding("productType", kind="value", value_dcids=["MobileCellularSubscription"])
        ],
        unplaced=["per 100 people"],
    )
    qualifiers = _resolve_query_qualifiers(state)
    assert [(q.text, q.disposition) for q in qualifiers] == [("per 100 people", "no_vocabulary")]


def test_unplaced_narrowing_phrase_on_a_per_person_shape_is_refused():
    state = _state(
        variable="prepaid mobile phone subscriptions per 100 people",
        shape_draft=_per_person_shape("Count_Person"),
        bindings=[
            _binding("productType", kind="value", value_dcids=["MobileCellularSubscription"])
        ],
        unplaced=["prepaid", "per 100 people"],
    )
    qualifiers = _resolve_query_qualifiers(state)
    assert [(q.text, q.disposition) for q in qualifiers] == [("prepaid", "no_vocabulary")]


# ---------------------------------------------------------------------------
# A narrowing term without a code row blocks the answer
# ---------------------------------------------------------------------------


def test_unplaced_no_vocabulary_term_trips_the_guard():
    shape = _shape(_FLOW_SLOT)
    state = _state(
        variable="untied support to Vietnam",
        shape_draft=shape,
        bindings=[_binding("crsTypeOfFlow", kind="value", value_dcids=["CRSTypeOfFlow10"])],
        unplaced=["untied"],
    )
    qualifiers = _run(state)
    by_text = {q.text: q for q in qualifiers}
    assert by_text["untied"].disposition == "no_vocabulary"
    assert by_text["untied"].named == []
    assert by_text["untied"].slot is None
    assert _has_miss(qualifiers)


def test_f02_non_concessional_flow_refuses_without_an_llm_unplaced_record():
    """The bounded unsupported term protects f02 even when the binder does
    not independently report it as unplaced."""
    shape = _shape(_FLOW_SLOT)
    state = _state(
        variable="non-concessional aid to Kenya",
        shape_draft=shape,
        bindings=[_binding("crsTypeOfFlow", kind="value", value_dcids=["CRSTypeOfFlow10"])],
        unplaced=[],
    )

    qualifiers = _run(state)

    assert [(q.text, q.disposition) for q in qualifiers] == [("non-concessional", "no_vocabulary")]
    assert _has_miss(qualifiers)


def test_f02_non_concessional_in_raw_query_refuses_when_extraction_omits_it():
    """The raw-query pass keeps the closed detector active after extraction
    omits the unsupported term from the variable phrase."""
    shape = _shape(_FLOW_SLOT)
    state = _state(
        variable="aid to Kenya",
        query_text="non-concessional aid to Kenya",
        shape_draft=shape,
        bindings=[_binding("crsTypeOfFlow", kind="value", value_dcids=["CRSTypeOfFlow10"])],
        unplaced=[],
    )

    qualifiers = _run(state)

    assert [(q.text, q.disposition) for q in qualifiers] == [("non-concessional", "no_vocabulary")]
    assert _has_miss(qualifiers)


def test_non_concessional_dedupes_an_identical_llm_unplaced_record():
    shape = _shape(_FLOW_SLOT)
    state = _state(
        variable="non-concessional aid to Kenya",
        query_text="non-concessional aid to Kenya",
        shape_draft=shape,
        bindings=[_binding("crsTypeOfFlow", kind="value", value_dcids=["CRSTypeOfFlow10"])],
        unplaced=["non-concessional"],
    )

    qualifiers = _run(state)

    assert [q.text for q in qualifiers] == ["non-concessional"]


def test_non_concessional_does_not_affect_non_crs_population_shapes():
    shape = _shape(SlotKeyDraft(axis="how", property_dcid="age", label="age group"))
    state = _state(
        variable="population in Kenya",
        query_text="non-concessional population in Kenya",
        shape_draft=shape,
        bindings=[],
        unplaced=[],
    )

    assert _run(state) == []


# ---------------------------------------------------------------------------
# An ambiguous span needs a binding to one of its declared code systems
# ---------------------------------------------------------------------------

_MODALITY_SLOT = SlotKeyDraft(axis="how", property_dcid="crsCooperationModality", label="modality")
_PURPOSE_SLOT = SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="purpose")


def test_ambiguous_span_with_no_covering_binding_is_unbound():
    shape = _shape(_MODALITY_SLOT, _PURPOSE_SLOT)
    state = _state(
        variable="budget support to Kenya",
        shape_draft=shape,
        bindings=[],
        unplaced=[],
    )
    qualifiers = _run(state)
    assert [q.text for q in qualifiers] == ["budget support"]
    q = qualifiers[0]
    assert q.disposition == "unbound"
    assert q.slot is None
    assert {ref.dcid for ref in q.named} == {"CRSCooperationModalityA", "CRSPurposeCode510"}
    assert _has_miss(qualifiers)


def test_ambiguous_span_with_a_scripted_covering_binding_is_bound():
    shape = _shape(_MODALITY_SLOT, _PURPOSE_SLOT)
    state = _state(
        variable="budget support to Kenya",
        shape_draft=shape,
        bindings=[
            _binding(
                "crsCooperationModality", kind="value", value_dcids=["CRSCooperationModalityA"]
            )
        ],
        unplaced=[],
    )
    qualifiers = _run(state)
    assert [q.text for q in qualifiers] == ["budget support"]
    q = qualifiers[0]
    assert q.disposition == "bound"
    assert q.slot == "crsCooperationModality"
    assert not _has_miss(qualifiers)


# ---------------------------------------------------------------------------
# An ambiguous span within one system can be bound to either row
# ---------------------------------------------------------------------------

_FINANCE_SLOT = SlotKeyDraft(axis="what", property_dcid="crsTypeOfFinance", label="finance type")


def test_ambiguous_single_system_span_with_a_scripted_binding_is_bound():
    shape = _shape(_FINANCE_SLOT)
    state = _state(
        variable="grants to Kenya",
        shape_draft=shape,
        bindings=[_binding("crsTypeOfFinance", kind="value", value_dcids=["CRSTypeOfFinance110"])],
        unplaced=[],
    )
    qualifiers = _run(state)
    assert [q.text for q in qualifiers] == ["grants"]
    q = qualifiers[0]
    assert q.disposition == "bound"
    assert q.slot == "crsTypeOfFinance"
    assert {ref.dcid for ref in q.named} == {"CRSTypeOfFinance100", "CRSTypeOfFinance110"}
    assert not _has_miss(qualifiers)


# ---------------------------------------------------------------------------
# Undeclared code systems do not create qualifier records
# ---------------------------------------------------------------------------


def test_span_naming_an_undeclared_code_system_is_not_recorded():
    # _FINANCE_SLOT declares only crsTypeOfFinance; "budget support" names
    # crsCooperationModality/crsPurpose, neither declared here.
    shape = _shape(_FINANCE_SLOT)
    state = _state(
        variable="budget support to Kenya",
        shape_draft=shape,
        bindings=[],
        unplaced=[],
    )
    qualifiers = _run(state)
    assert qualifiers == []
    assert not _has_miss(qualifiers)


# ---------------------------------------------------------------------------
# Span ranges distinguish repeated terms and substrings within other words
# ---------------------------------------------------------------------------

_PURPOSE_SLOT = SlotKeyDraft(axis="what", property_dcid="crsPurpose", label="purpose")


def test_repeated_term_does_not_defeat_the_overlap_rule():
    """The second "roads" span overlaps "high-speed roads"; the first does not.
    A first-occurrence `str.find` would place both spans at the first "roads"."""
    shape = _shape(_PURPOSE_SLOT)
    state = _state(
        variable="we compared roads spending with high-speed roads improvements in Kenya",
        shape_draft=shape,
        bindings=[_binding("crsPurpose", kind="value", value_dcids=["CRSPurposeCode21020"])],
        unplaced=["high-speed roads"],
    )
    qualifiers = _run(state)
    assert [q.text for q in qualifiers] == ["roads", "roads"]
    assert all(q.disposition == "bound" for q in qualifiers)
    assert not _has_miss(qualifiers)


def test_short_term_embedded_in_a_longer_word_does_not_defeat_the_overlap_rule():
    """The standalone "roads" span is distinct from the substring in "railroads",
    which a plain `str.find` would match first."""
    shape = _shape(_PURPOSE_SLOT)
    state = _state(
        variable="how much aid went towards railroads and roads improvements in Kenya",
        shape_draft=shape,
        bindings=[
            _binding(
                "crsPurpose",
                kind="value",
                value_dcids=["CRSPurposeCode21030", "CRSPurposeCode21020"],
            )
        ],
        unplaced=["roads improvements"],
    )
    qualifiers = _run(state)
    assert [q.text for q in qualifiers] == ["railroads", "roads"]
    assert all(q.disposition == "bound" for q in qualifiers)
    assert not _has_miss(qualifiers)


# ---------------------------------------------------------------------------
# A qualifier_not_bound refusal carries a flagged match without a gate
# ---------------------------------------------------------------------------


def test_qualifier_not_bound_wire_no_data_assembles_with_gate_less_flagged_match():
    """The refusal permits a gate-less flagged match while other reasons still require a gate."""
    # "Untied" has no matching code row.
    shape = _shape(_FLOW_SLOT)
    state = _state(
        variable="untied support to Vietnam",
        shape_draft=shape,
        bindings=[_binding("crsTypeOfFlow", kind="value", value_dcids=["CRSTypeOfFlow10"])],
        unplaced=["untied"],
    )
    qualifiers = _run(state)
    by_text = {q.text: q for q in qualifiers}
    assert by_text["untied"].disposition == "no_vocabulary"

    spec = Spec.model_validate(minimal_spec("s1"))

    # No flagged match needs no gate.
    bare = NoData(reason="qualifier_not_bound", unbound_qualifiers=qualifiers)
    assert bare.flagged_match is None
    assert bare.gate is None

    # A flagged spec still requires a gate for other refusal reasons.
    with pytest.raises(ValidationError):
        NoData(reason="no_observations", flagged_match=spec, gate=None)

    # Qualifier admission refuses before an availability verdict.
    region = state.no_data(
        "qualifier_not_bound",
        flagged_match=(spec,),
        gate=None,
        unbound_qualifiers=tuple(qualifiers),
    )
    part = _build_part(
        region,
        representative=region,
        folded=False,
        include_sentence=False,
        max_candidates=1,
        include_observations=False,
        include_skills=False,
        skills_index=None,
        envelope_warnings=[],
    )
    assert isinstance(part, NoDataPart)
    assert part.status == "no_data"
    no_data = part.no_data
    assert no_data.reason == "qualifier_not_bound"
    assert no_data.gate is None
    assert no_data.flagged_match is not None
    assert no_data.flagged_match.spec_id == "s1"
    assert [q.text for q in no_data.unbound_qualifiers] == ["untied"]
    assert [q.disposition for q in no_data.unbound_qualifiers] == ["no_vocabulary"]
