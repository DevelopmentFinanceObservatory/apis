"""Test `_resolve_query_qualifiers` through `_phase_repair_bindings`.

Cases use the packaged code and measure indexes for real CRS terms, with
synthetic shapes that prevent measure-family election from changing bindings.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import pytest
from pydantic import ValidationError

from code_index.shared import get_index, get_measure_index
from code_index.tokens import tokenize
from entity_resolution import Admission, EntityResolver
from search.engine.bind import DraftBindingKind, SlotBindingDraft, _UniversalBindOutput
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.conjoin import _build_part
from search.engine.function_words import FUNCTION_WORDS
from search.engine.interpret import Recall
from search.engine.qualifier_match import QualifierSpan
from search.engine.region import phase_election
from search.engine.region.phase_election import _phase_repair_bindings
from search.engine.region.phase_rank_bind import (
    _qualifier_disposition,
    _resolve_query_qualifiers,
    _unread_words_beside_codes,
)
from search.engine.region.state import ResolveState
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.models import NoData, NoDataPart, ResolvedQualifier, Spec
from tests.conftest import minimal_spec
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._index_slots import crs_flow_slot_keys, own_variable_spans, read_text

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
    entities: Sequence[str] = ("Kenya",),
) -> ResolveState:
    """Build the hand-crafted state consumed by qualifier repair."""
    return ResolveState(
        variable=variable,
        entities=list(entities),
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
        materialise_graph=None,
        materialise_sdmx=None,
        timing={},
        warnings=[],
        # Recall resolved no place, so the minted identity binds no role key.
        rcl=Recall(candidates=[], resolutions={}),
        bind_result=_UniversalBindOutput(
            chosen_shape_index=0,
            bindings=bindings,
            also_plausible=None,
            ask=None,
            unplaced=unplaced,
        ),
        shape_draft=shape_draft,
        variable_spans_by_vocabulary=own_variable_spans(variable, shape_draft),
        variable_spans_in_every_system=read_text(
            variable, get_index(), get_measure_index()
        ).everywhere.spans,
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


@pytest.mark.parametrize(
    ("unplaced", "records", "dropped"),
    [
        (
            ["multilateral"],
            [("ODA", "bound", "crsTypeOfFlow"), ("multilateral", "no_vocabulary", None)],
            [],
        ),
        ([], [("ODA", "bound", "crsTypeOfFlow")], ["multilateral"]),
    ],
    ids=["model-lists-the-word", "model-omits-the-word"],
)
def test_a_word_the_shapes_own_reading_leaves_unread_is_recorded_or_warned(
    search_settings, unplaced, records, dropped
):
    """In every code system together "multilateral ODA" is DAC1 aid type 2000. A shape with
    the CRS flow slot reads "ODA" alone, so "ODA" records the bound flow type. Under the log
    coverage mode the unread "multilateral" records `no_vocabulary` when the bind model
    lists it as unplaced, and otherwise departs once with the dropped-qualifier warning.
    Exactly one of the two applies."""
    search_settings(qualifier_coverage="log")
    shape = _shape(_FLOW_SLOT)
    state = _state(
        variable="multilateral ODA to Kenya",
        shape_draft=shape,
        bindings=[_binding("crsTypeOfFlow", kind="value", value_dcids=["CRSTypeOfFlow10"])],
        unplaced=unplaced,
    )
    qualifiers = _run(state)
    assert [(q.text, q.disposition, q.slot) for q in qualifiers] == records
    assert [
        fact.asked for fact in state.departure_facts.values() if fact.code == "measure_not_applied"
    ] == dropped
    assert [w.code for w in state.warnings if w.code == "query_qualifier_dropped"] == [
        "query_qualifier_dropped"
    ] * len(dropped)


_CRS_SLOTS = crs_flow_slot_keys()
_DAC2A_SLOT = SlotKeyDraft(axis="what", property_dcid="dac2aAidType", label="aid type")
_DAC1_SLOT = SlotKeyDraft(axis="what", property_dcid="dac1AidType", label="aid type")
_GRANT_EQUIVALENT_SHAPE = replace(_shape(_DAC1_SLOT), meas_prop_dcid="grantEquivalent")
_GNI_SHARE_SHAPE = replace(
    _shape(_DAC1_SLOT),
    meas_denom_dcid="PrimaryIncome_EconomicUnit_GrossOfFixedCapitalConsumption_Nominal",
)


# The part's entity texts.
_ENTITIES = ("Kenya", "France", "Hong Kong SAR, China", "Côte d'Ivoire")


@pytest.mark.parametrize(
    ("variable", "slots", "records"),
    [
        # The label or a synonym of the code beside a word reads it. "multilateral" beside
        # "imputed" is covered by DAC2a aid type 106, "imputed multilateral ODA" (dq-mls-003,
        # dq-mls-026, dataset-imputed-aid-kenya-es, dataset-top-donors-imputed-aid-kenya).
        ("imputed multilateral aid", (_DAC2A_SLOT,), []),
        ("imputed multilateral ODA", (_DAC2A_SLOT,), []),
        # ... and an amount word is read (dataset-imputed-multilateral-flows-kenya,
        # dataset-crs-and-imputed-education-aid-kenya, dq-ext-007), while "grants" past the
        # read words refuses beside 106 (dataset-imputed-multilateral-grants-kenya).
        ("imputed multilateral flows", (_DAC2A_SLOT,), []),
        ("imputed figures for education ODA", (_DAC2A_SLOT, *_CRS_SLOTS), []),
        ("health spending", _CRS_SLOTS, []),
        ("imputed multilateral grants", (_DAC2A_SLOT,), [("grants", ["DAC2aAidType106"])]),
        # A noun naming data itself is read (subj-health-data).
        ("health data", _CRS_SLOTS, []),
        # A word that the names of only some of the span's codes hold is unread. "in-kind" is
        # in standard grant's synonym "in-kind grant" and absent from grant's.
        (
            "ODA grants in-kind",
            _CRS_SLOTS,
            [("in-kind", ["CRSTypeOfFinance100", "CRSTypeOfFinance110"])],
        ),
        # A flow verb is read (dq-dac-019, dq-dac-021, dq-dac-022), and reads no coded
        # restriction beside it.
        ("net ODA received", (_DAC2A_SLOT,), []),
        # ... as are a motion verb (dq-crs-049) and a reporting verb
        # (dataset-total-oda-reported-to-dac-germany).
        ("ODA going to Kenya", (_DAC2A_SLOT,), []),
        ("total ODA reported to the DAC", (_DAC1_SLOT,), []),
        # A restriction beside the span refuses, and a run reaches past the verb to its
        # neighbour.
        ("private ODA going to Kenya", _CRS_SLOTS, [("private", ["CRSTypeOfFlow10"])]),
        ("ODA reported private", _CRS_SLOTS, [("private", ["CRSTypeOfFlow10"])]),
        (
            "private grants disbursed",
            _CRS_SLOTS,
            [("private", ["CRSTypeOfFinance100", "CRSTypeOfFinance110"])],
        ),
        # A shape dividing by GNI reads the ratio words and the denominator's noun
        # (dq-dac-002, dq-dac-017). An amount shape refuses them below.
        ("ODA as a share of GNI", _GNI_SHARE_SHAPE, []),
        ("ODA GNI ratio", _GNI_SHARE_SHAPE, []),
        # Restrictions no reader covers (tip measurement).
        (
            "domestic general government health expenditure",
            _CRS_SLOTS,
            [("domestic general government", ["CRSPurposeCode120"])],
        ),
        (
            "private grants",
            _CRS_SLOTS,
            [("private", ["CRSTypeOfFinance100", "CRSTypeOfFinance110"])],
        ),
        ("HIV prevalence", _CRS_SLOTS, [("prevalence", ["CRSPurposeCode13040"])]),
        ("imputed multilateral ODA", (_DAC1_SLOT,), [("imputed", ["DAC1AidType2000"])]),
        ("ODA GNI ratio", (_DAC1_SLOT,), [("GNI ratio", ["DAC1AidType1010"])]),
        ("grant-equivalent ODA", (_DAC1_SLOT,), [("grant-equivalent", ["DAC1AidType1010"])]),
        ("ODA grant equivalents", (_DAC1_SLOT,), [("grant equivalents", ["DAC1AidType1010"])]),
        # A grant-equivalent shape states "grant" and "equivalent", in a hyphenated word
        # too (dq-dac-004, dq-dac-033), and "grants" still narrows it.
        ("grant-equivalent ODA", _GRANT_EQUIVALENT_SHAPE, []),
        ("ODA grant equivalents", _GRANT_EQUIVALENT_SHAPE, []),
        ("ODA grants", _GRANT_EQUIVALENT_SHAPE, [("grants", ["DAC1AidType1010"])]),
        ("ODA as a share of GNI", (_DAC1_SLOT,), [("as a share of GNI", ["CRSTypeOfFinance2"])]),
        ("reproductive health ODA", _CRS_SLOTS, []),
        ("maternal health", _CRS_SLOTS, [("maternal", ["CRSPurposeCode120"])]),
        ("global health ODA", _CRS_SLOTS, [("global", ["CRSPurposeCode120"])]),
        ("multilateral ODA", _CRS_SLOTS, [("multilateral", ["CRSTypeOfFlow10"])]),
        # A run reaches past the amount and data words beside it to its neighbour, so the
        # restriction they cut off from the code span still refuses.
        ("maternal data ODA", _CRS_SLOTS, [("maternal", ["CRSTypeOfFlow10"])]),
        ("water spending ODA", _CRS_SLOTS, [("water", ["CRSTypeOfFlow10"])]),
        ("education and girls", _CRS_SLOTS, [("girls", ["CRSPurposeCode110"])]),
        ("health and education", _CRS_SLOTS, []),
        ("health ODA plus education ODA", _CRS_SLOTS, []),
        ("malaria control within health", _CRS_SLOTS, []),
        ("gross disbursements to Kenya", _CRS_SLOTS, []),
        ("Kenya's health ODA in 2022", _CRS_SLOTS, []),
        # The possessive ending reads in any case.
        ("KENYA'S HEALTH ODA", _CRS_SLOTS, []),
        ("FRANCE'S education ODA", _CRS_SLOTS, []),
        ("KENYA'S GLOBAL HEALTH ODA", _CRS_SLOTS, [("GLOBAL", ["CRSPurposeCode120"])]),
        # An entity's text reads its words across the separators and either apostrophe it
        # holds.
        ("Hong Kong SAR, China ODA", _CRS_SLOTS, []),
        ("Côte d'Ivoire ODA", _CRS_SLOTS, []),
        ("Côte d\u2019Ivoire ODA", _CRS_SLOTS, []),
        ("health ODA in the last 5 years", _CRS_SLOTS, []),
        ("ODA loans net disbursements", (_DAC2A_SLOT,), []),
    ],
)
def test_an_uncovered_word_beside_a_code_is_a_record(search_settings, variable, slots, records):
    """Under the refuse coverage mode each run of words no reader covers beside a code span
    of the elected shape's reading, or joined to one by a union connective, is an unbound
    record naming that span's rows. Measure values, entity text, dates, function words and
    a connective between two code spans are read."""
    search_settings(qualifier_coverage="refuse")
    shape = slots if isinstance(slots, ShapeDraft) else _shape(*slots)
    state = _state(
        variable=variable, shape_draft=shape, bindings=[], unplaced=[], entities=_ENTITIES
    )
    uncovered = [q for q in _resolve_query_qualifiers(state) if q.evidence == "uncovered"]
    assert [(q.text, [ref.dcid for ref in q.named]) for q in uncovered] == records
    assert all(q.disposition == "unbound" for q in uncovered)
    assert [variable[q.start : q.end] for q in uncovered if q.start is not None] == [
        text for text, _ in records
    ]


_FINANCE_100 = ("crsTypeOfFinance", ["CRSTypeOfFinance100"])
_ODA = ("crsTypeOfFlow", ["CRSTypeOfFlow10"])


@pytest.mark.parametrize(
    ("variable", "bound", "records"),
    [
        # The answer binds flow type 30, "private development finance", which no span of
        # "private grants" names, so it applies "private" (dq-crs-009).
        ("private grants", [_FINANCE_100, ("crsTypeOfFlow", ["CRSTypeOfFlow30"])], []),
        # ODA's label holds no "private", so the restriction refuses.
        (
            "private grants",
            [_FINANCE_100, _ODA],
            [("private", ["CRSTypeOfFinance100", "CRSTypeOfFinance110"])],
        ),
        # A synonym reads too: type of finance 420, "loan instrument", applies "loans".
        (
            "ODA loans for education",
            [
                _ODA,
                ("crsTypeOfFinance", ["CRSTypeOfFinance420"]),
                ("crsPurpose", ["CRSPurposeCode110"]),
            ],
            [],
        ),
        # A code a span names reads only its span: 130, which "reproductive health" names,
        # leaves "policy" beside education unread, though its label holds "policy".
        (
            "reproductive health and education policy ODA",
            [_ODA, ("crsPurpose", ["CRSPurposeCode130", "CRSPurposeCode110"])],
            [("policy", ["CRSPurposeCode110", "CRSTypeOfFlow10"])],
        ),
    ],
    ids=["unnamed-label", "label-lacks-word", "unnamed-synonym", "named-code"],
)
def test_a_word_the_name_of_an_unnamed_bound_code_holds_is_read(
    search_settings, variable, bound, records
):
    """A word the label or a synonym of a code the final bindings apply holds is read when
    no span of the query names that code, since the answer applies it."""
    search_settings(qualifier_coverage="refuse")
    state = _state(
        variable=variable,
        shape_draft=_shape(*_CRS_SLOTS),
        bindings=[
            _binding(prop, kind="value" if len(dcids) == 1 else "set", value_dcids=dcids)
            for prop, dcids in bound
        ],
        unplaced=[],
    )
    uncovered = [q for q in _run(state) if q.evidence == "uncovered"]
    assert [(q.text, [ref.dcid for ref in q.named]) for q in uncovered] == records


@pytest.mark.parametrize("mode", ["refuse", "log"])
def test_coverage_mode_decides_whether_an_uncovered_word_refuses(search_settings, caplog, mode):
    """In refuse mode the uncovered "maternal" enters the part's records and leaves it a
    miss. In log mode the records hold no such word, the would-refuse line names it once,
    and the answer is unaffected."""
    search_settings(qualifier_coverage=mode)
    state = _state(
        variable="maternal health ODA",
        shape_draft=_shape(*_CRS_SLOTS),
        bindings=[
            _binding("crsPurpose", kind="value", value_dcids=["CRSPurposeCode120"]),
            _binding("crsTypeOfFlow", kind="value", value_dcids=["CRSTypeOfFlow10"]),
        ],
        unplaced=[],
    )
    with caplog.at_level(logging.INFO, logger=phase_election.__name__):
        qualifiers = _run(state)
    lines = [r.getMessage() for r in caplog.records if "qualifier_coverage" in r.getMessage()]
    uncovered = [(q.text, q.disposition) for q in qualifiers if q.evidence == "uncovered"]
    if mode == "refuse":
        assert uncovered == [("maternal", "unbound")]
        assert _has_miss(qualifiers)
        assert lines == []
    else:
        assert uncovered == []
        assert not _has_miss(qualifiers)
        assert lines == ["qualifier_coverage would_refuse phrase=maternal codes=CRSPurposeCode120"]


@pytest.mark.parametrize("mode", ["refuse", "log"])
@pytest.mark.parametrize(
    ("variable", "phrase"),
    [
        ("health ODA for maternal care", "maternal care"),
        ("health ODA with a maternal focus", "maternal focus"),
        ("ODA for maternal care", "maternal care"),
    ],
)
def test_coverage_reaches_restrictions_across_safe_function_gaps(
    search_settings, caplog, mode, variable, phrase
):
    """A restriction after a transparent article or preposition still narrows the code
    before it, in both coverage modes."""
    search_settings(qualifier_coverage=mode)
    state = _state(
        variable=variable,
        shape_draft=_shape(*_CRS_SLOTS),
        bindings=[],
        unplaced=[],
    )
    with caplog.at_level(logging.INFO, logger=phase_election.__name__):
        qualifiers = _run(state)

    uncovered = [(q.text, q.disposition) for q in qualifiers if q.evidence == "uncovered"]
    lines = [r.getMessage() for r in caplog.records if "qualifier_coverage" in r.getMessage()]
    if mode == "refuse":
        assert uncovered == [(phrase, "unbound")]
        assert lines == []
    else:
        assert uncovered == []
        assert lines == [f"qualifier_coverage would_refuse phrase={phrase} codes=CRSTypeOfFlow10"]


@pytest.mark.parametrize(
    ("variable", "runs"),
    [
        ("ODA and maternal care", ["maternal care"]),
        ("ODA excluding maternal care", ["excluding maternal care"]),
        ("ODA versus maternal care", ["versus maternal care"]),
        ("ODA; for maternal care", []),
    ],
    ids=["union", "exclusion", "comparison", "punctuation"],
)
def test_neighbour_traversal_keeps_connective_and_punctuation_boundaries(variable, runs):
    """Only the narrow function-word gaps are transparent. Connectives remain in their
    own unread run, and non-connective punctuation prevents a run from reaching a code span."""
    row = get_index().code_by_dcid("CRSTypeOfFlow10")
    assert row is not None
    start = variable.index("ODA")
    tokens = tokenize(variable)
    code_span = QualifierSpan(matched_text="ODA", start=start, end=start + 3, rows=(row,))
    unread = _unread_words_beside_codes(
        variable,
        tokens,
        [code_span],
        [(code_span.start, code_span.end)],
        lambda word: word in FUNCTION_WORDS,
    )
    assert [variable[tokens[run.first].start : tokens[run.last].end] for run in unread] == runs


def test_a_word_the_shapes_own_reading_leaves_unread_refuses_in_refuse_mode(search_settings):
    """In refuse mode "multilateral" in "multilateral ODA to Kenya" is a record, and departs
    nothing."""
    search_settings(qualifier_coverage="refuse")
    state = _state(
        variable="multilateral ODA to Kenya",
        shape_draft=_shape(_FLOW_SLOT),
        bindings=[_binding("crsTypeOfFlow", kind="value", value_dcids=["CRSTypeOfFlow10"])],
        unplaced=[],
    )
    qualifiers = _run(state)
    assert [(q.text, q.disposition, q.evidence) for q in qualifiers] == [
        ("ODA", "bound", None),
        ("multilateral", "unbound", "uncovered"),
    ]
    assert [fact.code for fact in state.departure_facts.values()] == []


@pytest.mark.parametrize(
    ("union", "disposition"),
    [
        ((), "unbound"),
        (("CRSPurposeCode13040", "CRSPurposeCode130"), "bound"),
        (("CRSPurposeCode13040", "CRSPurposeCode120"), "unbound"),
    ],
    ids=["no-union", "union-with-the-ancestor", "union-without-the-ancestor"],
)
def test_an_ancestor_binds_a_code_only_through_a_union_that_selected_both(union, disposition):
    """Sexually transmitted disease control (13040) lies under 130. A binding of 130 binds
    a 13040 record only when the union that record joined selected 130 too."""
    assert (
        _qualifier_disposition(
            [("crsPurpose", "CRSPurposeCode13040")],
            {"crsPurpose": ["CRSPurposeCode130"]},
            get_index(),
            union=union,
        )[0]
        == disposition
    )


def test_a_term_the_shape_reads_none_of_departs_nothing():
    """ "gross national income" is a CRS type of finance in every system together, and the
    series a GNI shape measures. A shape that reads none of its words names no code the
    answer applies in part, so nothing departs."""
    shape = _shape(SlotKeyDraft(axis="how", property_dcid="productType", label="product type"))
    state = _state(
        variable="gross national income of Ghana", shape_draft=shape, bindings=[], unplaced=[]
    )
    _run(state)
    assert [fact.code for fact in state.departure_facts.values()] == []


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
    state.variable_spans_by_vocabulary = {}
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


def test_unplaced_per_person_rate_base_on_a_person_count_denominator_is_not_refused():
    """A StatVar denominator the leg's kinds read as a count of persons carries the phrase
    as `Count_Person` does."""
    state = _state(
        variable="mobile phone subscriptions per 100 people",
        shape_draft=_per_person_shape("Count_Person_15OrMoreYears"),
        bindings=[
            _binding("productType", kind="value", value_dcids=["MobileCellularSubscription"])
        ],
        unplaced=["per 100 people"],
    )
    state.denominator_kinds = {"Count_Person_15OrMoreYears": "person"}
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
# An unplaced phrase naming a declared slot of the elected shape is placed
# ---------------------------------------------------------------------------

_PURPOSE_SLOT = SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="CRS purpose")


@pytest.mark.parametrize(
    ("variable", "slots", "phrase", "refused"),
    [
        pytest.param(
            "ODA by sector", (_FLOW_SLOT, _PURPOSE_SLOT), "sector", [], id="declared-slot-noun"
        ),
        pytest.param(
            "ODA by sectors", (_FLOW_SLOT, _PURPOSE_SLOT), "Sectors", [], id="plural-slot-noun"
        ),
        pytest.param(
            "ODA by sector",
            (_FLOW_SLOT, _PURPOSE_SLOT),
            "by sector",
            [("by sector", "no_vocabulary")],
            id="phrase-holding-the-slot-noun",
        ),
        pytest.param(
            "ODA to the space sector",
            (_FLOW_SLOT, _PURPOSE_SLOT),
            "space sector",
            [("space sector", "no_vocabulary")],
            id="value-the-slot-lacks",
        ),
        pytest.param(
            "ODA by channel",
            (_FLOW_SLOT, _PURPOSE_SLOT),
            "channel",
            [("channel", "no_vocabulary")],
            id="undeclared-noun",
        ),
        pytest.param(
            "ODA by sector",
            (_FLOW_SLOT,),
            "sector",
            [("sector", "no_vocabulary")],
            id="slot-noun-the-shape-does-not-declare",
        ),
    ],
)
def test_unplaced_slot_noun(variable, slots, phrase, refused):
    """dataset-oda-kenya-by-sector: the bind model listed "sector" as unplaced on the CRS
    shape, whose `crsPurpose` slot the index declares under the plain name "sector". Only
    the bare noun is placed; a longer phrase holding it still refuses."""
    state = _state(
        variable=variable,
        shape_draft=_shape(*slots),
        bindings=[
            _binding("crsTypeOfFlow", kind="value", value_dcids=["CRSTypeOfFlow10"]),
            *([_binding("crsPurpose")] if _PURPOSE_SLOT in slots else []),
        ],
        unplaced=[phrase],
    )
    qualifiers = _resolve_query_qualifiers(state)
    assert [(q.text, q.disposition) for q in qualifiers if q.text != "ODA"] == refused


# ---------------------------------------------------------------------------
# An unplaced word of the answering dataset's label is placed
# ---------------------------------------------------------------------------

_IMPUTED = "provenance/ONEImputedMultilateralSectors"
_CRS = "provenance/OECDCreditorReportingSystem"


@pytest.mark.parametrize(
    ("datasets", "named", "phrase", "refused"),
    [
        pytest.param((_IMPUTED,), None, "imputed", [], id="word-of-the-answering-label"),
        pytest.param(
            (_IMPUTED,), None, "Imputed multilateral", [], id="words-of-the-answering-label"
        ),
        pytest.param(
            (_IMPUTED,), None, "grants", [("grants", "no_vocabulary")], id="word-outside-the-label"
        ),
        pytest.param(
            (_IMPUTED,),
            None,
            "imputed multilateral grants",
            [("imputed multilateral grants", "no_vocabulary")],
            id="phrase-reaching-past-the-label",
        ),
        pytest.param(
            (_CRS,), None, "imputed", [("imputed", "no_vocabulary")], id="label-of-another-dataset"
        ),
        pytest.param(
            None, None, "imputed", [("imputed", "no_vocabulary")], id="no-candidacy-datasets"
        ),
        pytest.param(
            (_IMPUTED, _CRS),
            None,
            "imputed",
            [("imputed", "no_vocabulary")],
            id="word-of-one-candidate-label",
        ),
        pytest.param(
            (_IMPUTED, _CRS),
            _IMPUTED,
            "imputed",
            [],
            id="word-of-the-settled-label",
        ),
        pytest.param(
            (_IMPUTED, _CRS),
            _CRS,
            "imputed",
            [("imputed", "no_vocabulary")],
            id="word-of-a-label-the-settled-table-lacks",
        ),
    ],
)
def test_unplaced_dataset_label_words(datasets, named, phrase, refused):
    """dataset-imputed-education-and-health-aid-kenya: the bind model listed "imputed" as
    unplaced on the shape answering from "ONE Imputed Multilateral Sectors", the table the
    word chose, and the phrase refused a correct answer."""
    shape = replace(_shape(_FLOW_SLOT), datasets=datasets, named_dataset=named)
    state = _state(
        variable="imputed multilateral grants of ODA to Kenya",
        shape_draft=shape,
        bindings=[_binding("crsTypeOfFlow", kind="value", value_dcids=["CRSTypeOfFlow10"])],
        unplaced=[phrase],
    )
    qualifiers = _resolve_query_qualifiers(state)
    assert [(q.text, q.disposition) for q in qualifiers if q.text != "ODA"] == refused


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
    # The record names the codes the shape's own vocabulary reads, its two slots' systems;
    # the DAC1 aid types of the same text belong to a system it declares no slot for.
    assert {ref.dcid for ref in q.named} == {
        "CRSCooperationModalityA",
        "CRSPurposeCode510",
    }
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


def test_repeated_term_does_not_defeat_the_overlap_rule(search_settings):
    """The second "roads" span overlaps "high-speed roads"; the first does not.
    A first-occurrence `str.find` would place both spans at the first "roads". The log
    coverage mode keeps the words beside the spans that no reader covers out of the
    records."""
    search_settings(qualifier_coverage="log")
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


def test_short_term_embedded_in_a_longer_word_does_not_defeat_the_overlap_rule(search_settings):
    """The standalone "roads" span is distinct from the substring in "railroads",
    which a plain `str.find` would match first. The log coverage mode keeps the words
    beside the spans that no reader covers out of the records."""
    search_settings(qualifier_coverage="log")
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

    wire = Spec.model_validate(minimal_spec("s1"))
    spec = wire.model_copy(update={"resolved": wire.resolved.stating_members([])})

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
