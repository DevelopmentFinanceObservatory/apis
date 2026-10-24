"""Test `_resolve_query_qualifiers` through `_repair_bindings`.

Cases use the fixture code and measure indexes for real CRS terms, with
synthetic shapes that prevent measure-family election from changing bindings.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Sequence
from dataclasses import replace
from typing import Literal, cast

import pytest
from pydantic import ValidationError

from code_index.shared import get_index, get_measure_index
from code_index.tokens import tokenize
from entity_resolution import Admission, Entity, EntityResolver, Resolved
from origin_catalogue import UnTables
from origin_catalogue.families import FamilyKey
from origin_catalogue.un_mirror import UnVariable
from search.engine.bind import DraftBindingKind, SlotBindingDraft, _UniversalBindOutput
from search.engine.binders import BindRequest, GeminiBinder, JevBinder
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.conjoin import _build_part
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.departures import measure_not_applied, warning_key
from search.engine.errors import LLMInfraError
from search.engine.function_words import FUNCTION_WORDS
from search.engine.interpret import Recall
from search.engine.qualifier_match import QualifierSpan
from search.engine.quantity import NamedDenominator
from search.engine.region.phase_election import _repair_bindings
from search.engine.region.phase_rank_bind import (
    _coverage_reading,
    _nominated_words,
    _qualifier_disposition,
    _resolve_query_qualifiers,
    _unrated_words,
    _unread_runs,
    judge_value_requirements,
    served_member_names,
    value_requirements,
)
from search.engine.region.place_roles import PlaceAssignment
from search.engine.region.requirements import (
    Judgement,
    Outcome,
    answer_terms,
    record_reading,
    refusal_terms,
    settle_answer,
    settle_candidate_set,
)
from search.engine.region.state import (
    Requirement,
    ResolveState,
    ValueRule,
    code_requirement,
    qualifier_dropped_warning,
    unapplied_departure,
)
from search.engine.region.unrealisable import (
    _unapplied,
    _unnamed_departure,
    _value_requirements,
    unbind_unrealisable,
)
from search.engine.retrieval.retrieve import BoundValuesUnrealisable, MemberConflict
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.models import GraphRef, NoData, NoDataPart, ResolvedQualifier, Spec
from tests.conftest import minimal_spec
from tests.engine._binders import FakeAsk, RatingBinder, keep_the_pick
from tests.engine._builders import place_sets
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._index_slots import crs_flow_slot_keys, own_variable_spans, read_text
from tests.fixtures import FakeGraph
from tests.origin_catalogue.fake_catalogue import FakeCatalogue, un_mirror, un_variable

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
    result = asyncio.run(_repair_bindings(state))
    assert result is None, "expected _repair_bindings to fall through"
    return state.query_qualifiers


def _would_refuse(state: ResolveState) -> list[tuple[str, list[str]]]:
    """The phrase and named codes of each term log mode keeps for the would-refuse line."""
    return [(record.text, [ref.ref for ref in record.named]) for record in state.would_refuse]


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
    lists it as unplaced, and otherwise is a term the answer departs once with the
    dropped-qualifier warning.
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
    # The elected answer is the part's final answer here.
    settle_answer(state, state.shape_draft, state.bindings)
    assert state.judgement is not None
    departed = state.judgement.departures()
    assert [fact.asked for _warning, fact in departed if fact.code == "measure_not_applied"] == (
        dropped
    )
    assert [w.code for w, _fact in departed] == ["query_qualifier_dropped"] * len(dropped)


_CRS_SLOTS = crs_flow_slot_keys()
_DAC2A_SLOT = SlotKeyDraft(axis="what", property_dcid="dac2aAidType", label="aid type")
_DAC1_SLOT = SlotKeyDraft(axis="what", property_dcid="dac1AidType", label="aid type")
_DAC2B_SLOT = SlotKeyDraft(axis="what", property_dcid="dac2bAidType", label="aid type")
_GRANT_EQUIVALENT_SHAPE = replace(_shape(_DAC1_SLOT), meas_prop_dcid="grantEquivalent")
_GNI_SHARE_SHAPE = replace(
    _shape(_DAC1_SLOT),
    meas_denom_dcid="PrimaryIncome_EconomicUnit_GrossOfFixedCapitalConsumption_Nominal",
)


def _answering(shape: ShapeDraft, measured_property: str, *datasets: str) -> ShapeDraft:
    """`shape` as the part answers it from `datasets`, measuring `measured_property`."""
    return replace(shape, meas_prop_dcid=measured_property, datasets=datasets)


_CRS_AMOUNT_SHAPE = _answering(
    _shape(*_CRS_SLOTS), "amount", "provenance/OECDCreditorReportingSystem"
)
_DAC1_AMOUNT_SHAPE = _answering(_shape(_DAC1_SLOT), "amount", "provenance/OECDDAC1")
_DAC2A_AMOUNT_SHAPE = _answering(_shape(_DAC2A_SLOT), "amount", "provenance/OECDDAC2A")
_DAC2B_AMOUNT_SHAPE = _answering(_shape(_DAC2B_SLOT), "amount", "provenance/OECDDAC2B")
_DAC2A_IMPUTED_SHAPE = _answering(_shape(_DAC2A_SLOT), "attributedAmount", "provenance/OECDDAC2A")
_IMPUTED_SECTORS_SHAPE = _answering(
    _shape(_DAC2A_SLOT), "attributedAmount", "provenance/ONEImputedMultilateralSectors"
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
        # "ratio" is form wording, always read, so the record names "GNI" alone.
        ("ODA GNI ratio", (_DAC1_SLOT,), [("GNI", ["DAC1AidType1010"])]),
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
        # A table states the word every row it publishes of the measure states. Each `amount`
        # row of the CRS is a flow from one provider to one recipient, so "bilateral"
        # restricts nothing there. DAC1 reads it as aid type 1015. DAC2a states it only where
        # the destination is a country, and this state binds no place.
        ("bilateral ODA", _CRS_AMOUNT_SHAPE, []),
        ("bilateral ODA", _DAC1_AMOUNT_SHAPE, []),
        ("bilateral ODA", _DAC2A_AMOUNT_SHAPE, [("bilateral", ["DAC2aAidType206"])]),
        ("bilateral OOF", _DAC2B_AMOUNT_SHAPE, [("bilateral", ["DAC2bAidType972"])]),
        # ... and no table states "multilateral", nor "bilateral" for imputed multilateral
        # aid, whose measure is the attributed amount.
        ("multilateral ODA", _CRS_AMOUNT_SHAPE, [("multilateral", ["CRSTypeOfFlow10"])]),
        ("bilateral ODA", _DAC2A_IMPUTED_SHAPE, [("bilateral", ["DAC2aAidType206"])]),
        ("bilateral ODA", _IMPUTED_SECTORS_SHAPE, [("bilateral", ["DAC2aAidType206"])]),
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
    assert [(q.text, [ref.ref for ref in q.named]) for q in uncovered] == records
    assert all(q.disposition == "unbound" for q in uncovered)
    assert [variable[q.start : q.end] for q in uncovered if q.start is not None] == [
        text for text, _ in records
    ]


def _reader_shape(
    name: str, *, meas_prop: str = _MEAS_PROP_DCID, pop_type: str = _POP_TYPE_DCID
) -> ShapeDraft:
    """A one-member shape named `name`, with no slot, measuring `meas_prop` of
    `pop_type`."""
    return replace(
        _shape(),
        pop_type_dcid=pop_type,
        meas_prop_dcid=meas_prop,
        sv_arc_facts={"reader_member": {"name": {"nodes": [{"value": name}]}}},
    )


_READER_PERSONS = _reader_shape("Total population", meas_prop="count", pop_type="Person")
_READER_REFUGEES = _reader_shape("Refugee population", meas_prop="count", pop_type="Person")
_READER_TAX = _reader_shape("Tax revenue")
_READER_REFUGEE_COUNT = _reader_shape(
    "Refugees, by host country", meas_prop="count", pop_type="Person"
)
_READER_HOUSEHOLD_COUNT = _reader_shape(
    "Refugee households", meas_prop="count", pop_type="Household"
)
# A UN System series has its own population type and says it counts persons by its unit.
_READER_UN_PERSON_COUNT = replace(
    _reader_shape(
        "Refugees under UNHCR's mandate", meas_prop="value", pop_type="undata/unhcr/REFUGEES"
    ),
    place_units=frozenset({"undata/UNIT_MEASURE-COUNT_PERSONS"}),
)
_READER_REFUGEE_SHARE = _reader_shape("Refugees, by host country", meas_prop="amount")
_READER_DEBT_BY_CREDITOR = replace(
    _reader_shape("External debt stocks"),
    declared_roles=("financingProvider", "financingRecipient"),
)


@pytest.mark.parametrize(
    ("variable", "shape", "unit_text", "base_words", "verdicts"),
    [
        # The count of all persons states the count, not who is counted, so the binder
        # rates "refugee".
        (
            "refugee population",
            _READER_PERSONS,
            None,
            None,
            [("refugee", "unrated"), ("population", "measure")],
        ),
        ("refugees", _READER_REFUGEES, None, None, [("refugees", "measure")]),
        # A decimal number is one word, so the binder rates the poverty line "2.15" whole.
        (
            "population at $2.15 per day",
            _READER_PERSONS,
            None,
            None,
            [
                ("population", "measure"),
                ("at", "conversational"),
                ("2.15", "unrated"),
                ("per", "conversational"),
                ("day", "unrated"),
            ],
        ),
        (
            "number of refugees",
            _READER_REFUGEES,
            None,
            None,
            [("number", "count"), ("of", "conversational"), ("refugees", "measure")],
        ),
        # On a shape measuring a count (`measures_count`), each noun for what it counts.
        *(
            pytest.param(
                f"refugee {noun}",
                _READER_REFUGEE_COUNT,
                None,
                None,
                [("refugee", "measure"), (noun, "count")],
                id=f"count-shape-{noun}",
            )
            for noun in ("number", "population", "populations", "people", "persons")
        ),
        # On a count of anything but persons, only the nouns for a count: the binder rates
        # the nouns for people.
        *(
            pytest.param(
                f"refugee {noun}",
                _READER_HOUSEHOLD_COUNT,
                None,
                None,
                [("refugee", "measure"), (noun, "count" if noun == "number" else "unrated")],
                id=f"count-of-households-{noun}",
            )
            for noun in ("number", "population", "populations", "people", "persons")
        ),
        *(
            pytest.param(
                f"refugee {noun}",
                _READER_UN_PERSON_COUNT,
                None,
                None,
                [("refugee", "measure"), (noun, "count")],
                id=f"un-count-of-persons-{noun}",
            )
            for noun in ("number", "population", "populations", "people", "persons")
        ),
        # The same nouns are unread on a shape measuring no count.
        *(
            pytest.param(
                f"refugee {noun}",
                _READER_REFUGEE_SHARE,
                None,
                None,
                [("refugee", "measure"), (noun, "unrated")],
                id=f"no-count-shape-{noun}",
            )
            for noun in ("number", "population", "people", "persons")
        ),
        (
            "tax revenue rate",
            _READER_TAX,
            None,
            None,
            [("tax", "measure"), ("revenue", "measure"), ("rate", "form")],
        ),
        (
            "tax revenue share",
            _READER_TAX,
            None,
            None,
            [("tax", "measure"), ("revenue", "measure"), ("share", "form")],
        ),
        (
            "tax revenue of GDP",
            _READER_TAX,
            None,
            "GDP",
            [("tax", "measure"), ("revenue", "measure"), ("of", "conversational"), ("GDP", "base")],
        ),
        (
            "tax revenue of GDP",
            _READER_TAX,
            None,
            None,
            [
                ("tax", "measure"),
                ("revenue", "measure"),
                ("of", "conversational"),
                ("GDP", "unrated"),
            ],
        ),
        (
            "total tax revenue",
            _READER_TAX,
            None,
            None,
            [("total", "conversational"), ("tax", "measure"), ("revenue", "measure")],
        ),
        # "current US dollars" is the price basis the reading reads first.
        (
            "tax revenue in current US dollars",
            _READER_TAX,
            "in current US dollars",
            None,
            [
                ("tax", "measure"),
                ("revenue", "measure"),
                ("in", "unit"),
                ("current", "reading"),
                ("US", "reading"),
                ("dollars", "reading"),
            ],
        ),
        (
            "tax revenue in euros",
            _READER_TAX,
            "in euros",
            None,
            [("tax", "measure"), ("revenue", "measure"), ("in", "unit"), ("euros", "unit")],
        ),
        (
            "tax revenue in euros",
            _READER_TAX,
            None,
            None,
            [
                ("tax", "measure"),
                ("revenue", "measure"),
                ("in", "conversational"),
                ("euros", "unrated"),
            ],
        ),
        # A role noun on an axis the shape declares.
        (
            "external debt stocks by creditors",
            _READER_DEBT_BY_CREDITOR,
            None,
            None,
            [
                ("external", "measure"),
                ("debt", "measure"),
                ("stocks", "measure"),
                ("by", "conversational"),
                ("creditors", "role"),
            ],
        ),
        (
            "external debt stocks by creditors",
            _reader_shape("External debt stocks"),
            None,
            None,
            [
                ("external", "measure"),
                ("debt", "measure"),
                ("stocks", "measure"),
                ("by", "conversational"),
                ("creditors", "unrated"),
            ],
        ),
        # Only words the code index folds alike read each other, so another form of a
        # name word is left for the binder to rate.
        (
            "literacy rate",
            _reader_shape("Population: Literate"),
            None,
            None,
            [("literacy", "unrated"), ("rate", "form")],
        ),
        (
            "climate finance",
            _reader_shape("Total climate-specific financial support"),
            None,
            None,
            [("climate", "unrated"), ("finance", "unrated")],
        ),
        (
            "Gini coefficient",
            _reader_shape("Gini index"),
            None,
            None,
            [
                ("Gini", "measure"),
                ("coefficient", "unrated"),
            ],
        ),
        (
            "national index",
            _reader_shape("National income"),
            None,
            None,
            [("national", "measure"), ("index", "unrated")],
        ),
    ],
    ids=[
        "count-of-persons-leaves-refugee",
        "refugee-population",
        "decimal-number",
        "count-noun",
        "count-shape-number",
        "count-shape-population",
        "count-shape-populations",
        "count-shape-people",
        "count-shape-persons",
        *(
            f"count-of-households-{noun}"
            for noun in ("number", "population", "populations", "people", "persons")
        ),
        *(
            f"un-count-of-persons-{noun}"
            for noun in ("number", "population", "populations", "people", "persons")
        ),
        "no-count-shape-number",
        "no-count-shape-population",
        "no-count-shape-people",
        "no-count-shape-persons",
        "rate",
        "share",
        "of-the-named-base",
        "of-an-unnamed-base",
        "total",
        "price-basis",
        "currency-unit",
        "currency-unread",
        "role-noun",
        "role-noun-on-a-subject-only-shape",
        "word-forms",
        "word-forms-running-on",
        "coefficient-on-an-index",
        "index-noun-off-the-name",
    ],
)
def test_each_word_of_the_variable_is_read_or_unread(
    variable, shape, unit_text, base_words, verdicts
):
    """Each word of the variable takes the verdict of the first reader that reads it on the
    elected shape: its members' names (`measure`), the nouns for what a count counts
    (`count`), a role noun on an axis the shape declares (`role`), form wording (`form`),
    the question's words for the base the quantity read names (`base`), the unit text
    (`unit`) and the conversational words. Any other word beside no code is ``unrated``,
    which the binder rates once the bindings settle."""
    state = _state(variable=variable, shape_draft=shape, bindings=[], unplaced=[], entities=())
    state.unit_text = unit_text
    if base_words is not None:
        state.named_denominator = NamedDenominator(ref=None, base_words=base_words)

    assert list(_coverage_reading(state, []).verdicts) == verdicts


# The reader gaps: a word no reader reads on a series that states it, which the binder rates
# stated.
_READER_UNDER_FIVE = _reader_shape("Under-five mortality rate [under 5 years old]")
_READER_CONTRIBUTIONS = _reader_shape("Multilateral contributions")
_READER_POVERTY_LINE = _reader_shape("Proportion of population below international poverty line")
_READER_FOREST = _reader_shape("Forest area")


@pytest.mark.parametrize(
    ("variable", "shape", "ratings", "bar", "verdicts", "records"),
    [
        # No rating reads the word as stated.
        (
            "refugee population",
            _READER_PERSONS,
            {},
            0.5,
            [("refugee", "unrated"), ("population", "measure")],
            [],
        ),
        (
            "refugee population",
            _READER_PERSONS,
            {"refugee": 0.1},
            0.5,
            [("refugee", "unread"), ("population", "measure")],
            ["refugee"],
        ),
        (
            "refugee population",
            _READER_PERSONS,
            {"refugee": 0.5},
            0.5,
            [("refugee", "stated"), ("population", "measure")],
            [],
        ),
        (
            "refugee population",
            _READER_PERSONS,
            {"refugee": 0.5},
            0.6,
            [("refugee", "unread"), ("population", "measure")],
            ["refugee"],
        ),
        # Each stretch rated below the bar is one record.
        (
            "stateless refugee population",
            _READER_PERSONS,
            {"stateless": 0.2, "refugee": 0.3},
            0.5,
            [("stateless", "unread"), ("refugee", "unread"), ("population", "measure")],
            ["stateless refugee"],
        ),
        (
            "stateless refugee population",
            _READER_PERSONS,
            {"stateless": 0.2, "refugee": 0.9},
            0.5,
            [("stateless", "unread"), ("refugee", "stated"), ("population", "measure")],
            ["stateless"],
        ),
        # A word nominated and rated stated: the under-five rate states "child", DAC1's
        # multilateral contributions "organisations", the poverty line series the people
        # "living" describes, and forest area each word of the land use it is a kind of.
        (
            "under-5 child mortality rate",
            _READER_UNDER_FIVE,
            {"child": 0.9},
            0.3,
            [
                ("under-5", "measure"),
                ("child", "stated"),
                ("mortality", "measure"),
                ("rate", "form"),
            ],
            [],
        ),
        (
            "contributions to multilateral organisations",
            _READER_CONTRIBUTIONS,
            {"organisations": 0.9},
            0.3,
            [
                ("contributions", "measure"),
                ("to", "conversational"),
                ("multilateral", "measure"),
                ("organisations", "stated"),
            ],
            [],
        ),
        (
            "population living in extreme poverty",
            _READER_POVERTY_LINE,
            {"living": 0.9, "extreme": 0.9},
            0.3,
            [
                ("population", "measure"),
                ("living", "stated"),
                ("in", "conversational"),
                ("extreme", "stated"),
                ("poverty", "measure"),
            ],
            [],
        ),
        (
            "land use statistic",
            _READER_FOREST,
            {"land": 0.9, "use": 0.9, "statistic": 0.9},
            0.3,
            [("land", "stated"), ("use", "stated"), ("statistic", "stated")],
            [],
        ),
    ],
    ids=[
        "unrated",
        "below-the-bar",
        "at-the-bar",
        "below-a-higher-bar",
        "one-stretch",
        "stretch-split-by-a-stated-word",
        "nominated-and-rated-stated-child",
        "nominated-and-rated-stated-organisations",
        "nominated-and-rated-stated-living",
        "nominated-and-rated-stated-land-use",
    ],
)
def test_a_word_beside_no_code_is_unread_only_when_rated_below_the_bar(
    search_settings, variable, shape, ratings, bar, verdicts, records
):
    """A word beside no code that no reader reads is a ``no_vocabulary`` record only when
    the binder rates the elected shape's series stating it below
    `SearchSettings.stated_word_bar`; at the bar or above it is ``stated``, and with no
    rating it is ``unrated`` and read."""
    search_settings(qualifier_coverage="refuse", stated_word_bar=bar)
    state = _state(variable=variable, shape_draft=shape, bindings=[], unplaced=[])
    state.stated_word_ratings = ratings

    reading = _coverage_reading(state, [])
    assert list(reading.verdicts) == verdicts
    uncovered = [q for q in _resolve_query_qualifiers(state) if q.evidence == "uncovered"]
    assert [(q.text, q.disposition) for q in uncovered] == [
        (text, "no_vocabulary") for text in records
    ]


class _FailingRating(RatingBinder):
    async def rate_words_stated(self, request, *, llm):
        self.requests.append(request)
        raise LLMInfraError("rating failed")


@pytest.mark.parametrize(
    ("variable", "shape", "binder", "asked", "records", "degraded"),
    [
        (
            "refugee population",
            _READER_PERSONS,
            RatingBinder(GeminiBinder(), {"refugee": 0.1}),
            [("refugee",)],
            ["refugee"],
            [],
        ),
        (
            "refugee population",
            _READER_PERSONS,
            RatingBinder(GeminiBinder(), {"refugee": 0.9}),
            [("refugee",)],
            [],
            [],
        ),
        # Every word read: the binder is not asked.
        ("refugee population", _READER_REFUGEES, RatingBinder(GeminiBinder()), [], [], []),
        # A failed read rates no word and is noted as lost.
        (
            "refugee population",
            _READER_PERSONS,
            _FailingRating(GeminiBinder()),
            [("refugee",)],
            [],
            ["stated words read"],
        ),
    ],
    ids=["rated-unread", "rated-stated", "nothing-to-rate", "failed-read"],
)
def test_the_binder_rates_the_words_left_unrated_once_the_bindings_settle(
    search_settings, variable, shape, binder, asked, records, degraded
):
    """Binding repair asks the binder once for the words the coverage rule leaves unrated
    on the elected shape, with the shape's menu index, and the rule refuses on those it
    rates below the bar. A read that fails counts each word as stated and notes the read
    as degraded."""
    search_settings(qualifier_coverage="refuse")
    state = _state(variable=variable, shape_draft=shape, bindings=[], unplaced=[])
    state.capped_shapes = [replace(_READER_TAX, shape_id="tax"), shape]
    state.binder = binder
    with recording_degraded_reads() as lost:
        qualifiers = _run(state)

    assert [request.words for request in binder.requests] == asked
    assert all(request.shape_index == 1 for request in binder.requests)
    assert [q.text for q in qualifiers if q.evidence == "uncovered"] == records
    assert lost.reads == degraded


@pytest.mark.parametrize(
    ("rating", "records"), [(0.1, ["refugee"]), (0.9, [])], ids=["below-the-bar", "above"]
)
def test_jev_rates_the_nominated_words_in_the_bind_and_the_rule_reads_its_rating(
    search_settings, rating, records
):
    """Jev answers the rating of a word left unrated from the ``unread|`` noul its bind
    request asked: "refugee" on the count of all persons, rated below
    `SearchSettings.stated_word_bar`, is a record that leaves the part a miss, and rated
    above it the part answers."""
    search_settings(qualifier_coverage="refuse")
    state = _state(
        variable="refugee population", shape_draft=_READER_PERSONS, bindings=[], unplaced=[]
    )
    state.capped_shapes = [replace(_READER_TAX, shape_id="tax"), _READER_PERSONS]
    binder = JevBinder(ask=FakeAsk(stated={"refugee": rating}))
    request = BindRequest(
        variable=state.variable,
        shapes=state.capped_shapes,
        nominated_words=[_nominated_words(state, shape) for shape in state.capped_shapes],
        query_text=None,
        apply_policies=keep_the_pick,
    )
    state.bind_decision, _usage = asyncio.run(binder.bind(request, llm=None))
    state.binder = binder
    qualifiers = _run(state)

    assert [q.text for q in qualifiers if q.evidence == "uncovered"] == records
    assert _has_miss(qualifiers) == bool(records)


_FEMALE = _binding("gender", kind="value", value_dcids=["Female"])
_MALE = _binding("gender", kind="value", value_dcids=["Male"])
_GENDER_OPENED = _binding("gender").model_copy(
    update={"evidence": "requested_all", "requested_all_phrase": "by gender"}
)


@pytest.mark.parametrize(
    ("total", "bound", "scored", "unrated"),
    [
        (True, (), (), ("refugee", "camps")),
        # The answer serves the total, "Population", whichever member recall scored, so a
        # scored member's name reads no word and the binder rates "refugee". So it does with
        # the gender slot opened, since the family's total answers for every gender.
        (True, (), ("nominating_1",), ("refugee", "camps")),
        (True, (_binding("gender", kind="absent"),), ("nominating_1",), ("refugee", "camps")),
        (True, (_binding("gender"),), ("nominating_1",), ("refugee", "camps")),
        (True, (_GENDER_OPENED,), ("nominating_1",), ("refugee", "camps")),
        # Gender bound to Female serves "Female refugee population" alone, which states
        # "refugee" whichever member recall scored.
        (True, (_FEMALE,), (), ("camps",)),
        (True, (_FEMALE,), ("nominating_1",), ("camps",)),
        (True, (_MALE,), ("nominating_2",), ("refugee",)),
        # A family with no total serves both members, so only the words both names hold
        # read, whichever recall scored.
        (False, (), ("nominating_1",), ("refugee", "camps")),
        (False, (_binding("gender", kind="absent"),), ("nominating_1",), ("refugee", "camps")),
        (
            False,
            (_binding("gender"),),
            ("nominating_1", "nominating_2"),
            ("refugee", "camps"),
        ),
        # With none, the words every member's name holds still read.
        (
            False,
            (_binding("gender", kind="value", value_dcids=["Unknown"]),),
            (),
            ("refugee", "camps"),
        ),
    ],
    ids=[
        "open",
        "scored",
        "absent-and-scored",
        "unbound-and-scored",
        "opened-serves-the-total",
        "bound-serves-the-female-member",
        "bound-and-scored",
        "bound-other",
        "no-total-scored",
        "no-total-absent",
        "no-total-unbound",
        "no-member-served",
    ],
)
def test_every_word_left_unrated_after_the_bind_was_nominated_before_it(
    total, bound, scored, unrated
):
    """The words the coverage rule leaves unrated on the elected shape over its bindings
    (`_unrated_words`) are among those nominated for that shape before the bind
    (`_nominated_words`), so a binder that rates the nominated words in the bind holds a
    rating for each word it is later asked about. Only the names of the members the
    answer serves read words (`_member_measure_words`)."""
    members = [{"gender": "Female"}, {"gender": "Male"}]
    names = ["Female refugee population", "Male population of camps"]
    shape = _value_shape(
        "nominating",
        SlotKeyDraft(axis="how", property_dcid="gender", label="gender"),
        members=[{}, *members] if total else members,
        names=["Population", *names] if total else names,
    )
    if not total:
        shape = replace(
            shape,
            sv_arc_facts={
                f"nominating_{i + 1}": arcs
                for i, arcs in enumerate((shape.sv_arc_facts or {}).values())
            },
        )
    state = _state(
        variable="female refugee population of camps",
        shape_draft=replace(shape, scored_members=frozenset(scored)),
        bindings=list(bound),
        unplaced=[],
        entities=(),
    )
    nominated = _nominated_words(state, state.shape_draft)
    state.bindings = state.bind_result.bindings

    assert nominated == ("female", "refugee", "camps")
    assert _unrated_words(state) == unrated
    assert set(unrated) <= set(nominated)


@pytest.mark.parametrize("kind", ["unbound", "absent"])
def test_a_held_total_recall_did_not_find_reads_its_name(bind_catalogue, kind):
    """ "female unemployment rate" over a family the catalogue holds, of the total and a
    member by each sex, where recall finds the female member alone and the bind leaves the
    sex slot unbound or absent: the answer serves the total, whose name reads
    "unemployment", and "female" is left unrated. The words nominated before the bind are
    read over the whole family too, so "female" is among them."""
    series = "undata/ilo/SERIES-UNE_RATE"
    female = f"{series}.SEX--F"
    sexes = {
        female: ("Unemployment rate: Female", "undata/SEX-F", "Female"),
        f"{series}.SEX--M": ("Unemployment rate: Male", "undata/SEX-M", "Male"),
    }
    bind_catalogue(
        FakeCatalogue(
            un=UnTables(
                mirror=un_mirror(
                    {
                        series: un_variable(series, name="Unemployment rate"),
                        **{
                            sv: un_variable(
                                series, name=name, constraints={"undataSex": ("Sex", value, label)}
                            )
                            for sv, (name, value, label) in sexes.items()
                        },
                    }
                )
            )
        )
    )
    sample = _value_shape(
        "unemployment",
        SlotKeyDraft(axis="how", property_dcid="undataSex", label="sex"),
        members=[{"undataSex": "undata/SEX-F"}],
        names=["Unemployment rate: Female"],
    )
    shape = replace(
        sample,
        pop_type_dcid=series,
        meas_prop_dcid="value",
        stat_type_dcid="measuredValue",
        sv_arc_facts={female: next(iter((sample.sv_arc_facts or {}).values()))},
        sv_arc_facts_complete=False,
        held_family=FamilyKey(series, "value", "measuredValue", None, None, False, False),
    )
    state = _state(
        variable="female unemployment rate",
        shape_draft=shape,
        bindings=[_binding("undataSex", kind=kind)],
        unplaced=[],
        entities=(),
    )
    nominated = _nominated_words(state, state.shape_draft)
    state.bindings = state.bind_result.bindings

    assert nominated == ("female",)
    assert _unrated_words(state) == ("female",)


def test_a_served_member_with_no_name_clears_no_word_the_named_ones_alone_hold():
    """ "GDP" over a family whose bindings serve nominal GDP and a net domestic product
    member whose name the engine cannot read, beside a gross value added member they leave
    out: the words are those every named member holds, so "gdp" goes to the binder although
    the one named member served holds it. The source values are no code the index labels,
    so no bound code reads the word."""
    shape = _value_shape(
        "gdp",
        SlotKeyDraft(axis="what", property_dcid="activitySource", label="activity source"),
        members=[
            {"activitySource": "SourceDomestic"},
            {"activitySource": "SourceValueAdded"},
            {"activitySource": "SourceNetDomestic"},
        ],
        names=["Nominal gross domestic product", "Gross value added", "unnamed"],
    )
    arcs = dict(shape.sv_arc_facts or {})
    arcs["gdp_2"] = {prop: nodes for prop, nodes in arcs["gdp_2"].items() if prop != "name"}
    produced = _binding(
        "activitySource",
        kind="set",
        value_dcids=["SourceDomestic", "SourceNetDomestic"],
    )
    state = _state(
        variable="GDP",
        shape_draft=replace(shape, sv_arc_facts=arcs),
        bindings=[produced.model_copy(update={"axis": "what"})],
        unplaced=[],
        entities=(),
    )
    state.bindings = state.bind_result.bindings

    assert served_member_names(state) == {
        "gdp_0": "Nominal gross domestic product",
        "gdp_2": None,
    }
    assert _unrated_words(state) == ("gdp",)


def test_the_served_members_read_the_direction_place_binding_settled():
    """ "aid to Kenya" named with no direction: binding repair refuses the places
    ``role_not_resolved``, and place binding settles them on the recipient side
    (`phase_place_materialise._settle_direction`). A coverage reading after it, as an
    unbinding at materialise makes, reads the members the recipient class serves, though it
    read both classes under the refusal."""

    def member(role: str, name: str) -> dict:
        return {
            "name": {"nodes": [{"value": name}]},
            "constraintProperties": {"nodes": []},
            "observationProperties": {"nodes": [{"dcid": role}]},
        }

    shape = replace(
        _shape(),
        declared_roles=("financingProvider", "financingRecipient"),
        sv_arc_facts={
            "aid_received": member("financingRecipient", "Aid received"),
            "aid_given": member("financingProvider", "Aid given"),
        },
    )
    state = _state(variable="aid", shape_draft=shape, bindings=[], unplaced=[], entities=())
    state.query_place_roles = "role_not_resolved"

    assert served_member_names(state) == {"aid_received": "Aid received", "aid_given": "Aid given"}

    (kenya,) = place_sets({"Kenya": "country/KEN"}).values()
    state.elected_place_roles = PlaceAssignment(sets={"destination": (kenya,)})

    assert served_member_names(state) == {"aid_received": "Aid received"}


_FLOW_ROLES = ("financingProvider", "financingRecipient")


def _naming_kenya_as_recipient(
    state: ResolveState, kind: Literal["country", "organisation"]
) -> ResolveState:
    """`state` with Kenya named as the destination, resolved to an entity of `kind`."""
    state.place_sets = place_sets({"Kenya": "country/KEN"})
    state.entity_roles = {"Kenya": [DoorRole(axis="destination", span="Kenya")]}
    entity = Entity(dcid="country/KEN", name="Kenya", kind=kind, graph_type="Country")
    state.rcl = Recall(candidates=[], resolutions={"Kenya": Resolved(entity=entity)})
    return state


@pytest.mark.parametrize(
    ("variable", "shape", "recipient", "records"),
    [
        # The CRS states "bilateral" of every recipient.
        ("bilateral ODA to Kenya", _CRS_AMOUNT_SHAPE, "organisation", []),
        # DAC2a and DAC2b state it of a country recipient only.
        ("bilateral ODA to Kenya", _DAC2A_AMOUNT_SHAPE, "country", []),
        (
            "bilateral ODA to Kenya",
            _DAC2A_AMOUNT_SHAPE,
            "organisation",
            [("bilateral", ["DAC2aAidType206"])],
        ),
        ("bilateral OOF to Kenya", _DAC2B_AMOUNT_SHAPE, "country", []),
        (
            "bilateral OOF to Kenya",
            _DAC2B_AMOUNT_SHAPE,
            "organisation",
            [("bilateral", ["DAC2bAidType972"])],
        ),
        # No table states it of an attributed amount, whatever the recipient.
        (
            "bilateral ODA to Kenya",
            _DAC2A_IMPUTED_SHAPE,
            "country",
            [("bilateral", ["DAC2aAidType206"])],
        ),
        (
            "bilateral ODA to Kenya",
            _IMPUTED_SECTORS_SHAPE,
            "country",
            [("bilateral", ["DAC2aAidType206"])],
        ),
    ],
)
def test_a_word_a_table_states_of_country_recipients_is_read_beside_one(
    search_settings, variable, shape, recipient, records
):
    """The recipient the part names decides whether a table that states "bilateral" of its
    country flows states it of the part: a country recipient reads the word, and an
    organisation recipient leaves it as a record naming the aid type code the word reads."""
    search_settings(qualifier_coverage="refuse")
    shape = replace(shape, declared_roles=_FLOW_ROLES)
    state = _naming_kenya_as_recipient(
        _state(variable=variable, shape_draft=shape, bindings=[], unplaced=[]), recipient
    )
    uncovered = [q for q in _resolve_query_qualifiers(state) if q.evidence == "uncovered"]
    assert [(q.text, [ref.ref for ref in q.named]) for q in uncovered] == records


@pytest.mark.parametrize(
    ("shape", "recipient", "with_state", "unmet"),
    [
        (_DAC2A_AMOUNT_SHAPE, "country", True, False),
        (_CRS_AMOUNT_SHAPE, "organisation", True, False),
        (_DAC2A_AMOUNT_SHAPE, "organisation", True, True),
        (_DAC2A_IMPUTED_SHAPE, "country", True, True),
        # A resubmit gives no state, and the part's places are not read.
        (_DAC2A_AMOUNT_SHAPE, "country", False, True),
    ],
)
def test_a_value_word_the_answering_table_states_is_met(shape, recipient, with_state, unmet):
    """A menu shape's constraint value named "bilateral" is met by an answer whose table
    states the word of the rows the part reads, and left unmet otherwise."""
    shape = replace(shape, declared_roles=_FLOW_ROLES)
    state = _naming_kenya_as_recipient(
        _state(variable="bilateral ODA to Kenya", shape_draft=shape, bindings=[], unplaced=[]),
        recipient,
    )
    requirement = Requirement(
        term=ResolvedQualifier(
            text="bilateral",
            named=[GraphRef(ref="Bilateral", label="Bilateral")],
            disposition="unbound",
        ),
        origin="value",
        rule=ValueRule(values=frozenset({("creditorType", "Bilateral")})),
    )

    judged = judge_value_requirements([requirement], shape, [], state if with_state else None)

    assert judged == ((requirement,) if unmet else ())


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
    assert [(q.text, [ref.ref for ref in q.named]) for q in uncovered] == records


@pytest.mark.parametrize("mode", ["refuse", "log"])
def test_coverage_mode_decides_whether_an_uncovered_word_refuses(search_settings, mode):
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
    qualifiers = _run(state)
    uncovered = [(q.text, q.disposition) for q in qualifiers if q.evidence == "uncovered"]
    if mode == "refuse":
        assert uncovered == [("maternal", "unbound")]
        assert _has_miss(qualifiers)
        assert _would_refuse(state) == []
    else:
        assert uncovered == []
        assert not _has_miss(qualifiers)
        assert _would_refuse(state) == [("maternal", ["CRSPurposeCode120"])]


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
    search_settings, mode, variable, phrase
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
    qualifiers = _run(state)

    uncovered = [(q.text, q.disposition) for q in qualifiers if q.evidence == "uncovered"]
    if mode == "refuse":
        assert uncovered == [(phrase, "unbound")]
        assert _would_refuse(state) == []
    else:
        assert uncovered == []
        assert _would_refuse(state) == [(phrase, ["CRSTypeOfFlow10"])]


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
    reading = _unread_runs(
        variable,
        tokens,
        [code_span],
        [("reading", [(code_span.start, code_span.end)])],
        [("function", lambda word: word in FUNCTION_WORDS)],
    )
    beside = [run for run in reading.runs if run.adjacent or run.joined]
    assert [variable[tokens[run.first].start : tokens[run.last].end] for run in beside] == runs


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


@pytest.mark.parametrize(
    ("shape", "phrase", "refused"),
    [
        # Every row of the CRS is bilateral, so the bare word is placed, and "donors" is
        # stated by nothing.
        pytest.param(_CRS_AMOUNT_SHAPE, "bilateral", [], id="word-the-table-states"),
        pytest.param(
            _CRS_AMOUNT_SHAPE,
            "bilateral donors",
            [("bilateral donors", "no_vocabulary")],
            id="phrase-reaching-past-the-stated-word",
        ),
        pytest.param(
            _DAC2A_AMOUNT_SHAPE,
            "bilateral",
            [("bilateral", "no_vocabulary")],
            id="word-the-table-states-of-no-recipient",
        ),
    ],
)
def test_an_unplaced_word_the_answering_dataset_states_is_placed(shape, phrase, refused):
    """The bind model listed "bilateral" as unplaced on a shape answering from the CRS,
    every row of which is a flow from one provider to one recipient."""
    state = _state(
        variable="bilateral donors of ODA to Kenya",
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


@pytest.mark.parametrize(
    ("slot", "variable", "bindings", "term"),
    [
        pytest.param(
            _FLOW_SLOT,
            "non-concessional aid to Kenya",
            [_binding("crsTypeOfFlow", kind="value", value_dcids=["CRSTypeOfFlow10"])],
            "non-concessional",
            id="flow-type",
        ),
        # The CRS tags climate finance with Rio markers the pipeline drops.
        pytest.param(
            SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="sector"),
            "climate finance",
            [],
            "climate",
            id="purpose",
        ),
    ],
)
def test_a_declared_unexpressed_word_refuses_without_an_llm_unplaced_record(
    slot, variable, bindings, term
):
    """A word the shape's code system declares no code expresses refuses even when the
    binder does not report it as unplaced."""
    state = _state(variable=variable, shape_draft=_shape(slot), bindings=bindings, unplaced=[])

    qualifiers = _run(state)

    assert [(q.text, q.disposition) for q in qualifiers] == [(term, "no_vocabulary")]
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
    assert {ref.ref for ref in q.named} == {
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
    assert {ref.ref for ref in q.named} == {"CRSTypeOfFinance100", "CRSTypeOfFinance110"}
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


# ---------------------------------------------------------------------------
# A value the request names by its graph label is judged against the answer
# ---------------------------------------------------------------------------

_VALUE_LABELS = {
    "Female": "Female",
    "Male": "Male",
    "Literate": "Literate",
    "Years10Onwards": "Years 10 Onwards",
    "Student": "Student",
    "German": "German",
    "CRSPurposeCode120": "Health",
    "Infant": "Infant",
    "EducationLevelPrimary": "Primary education",
    "EducationLevelSecondary": "Secondary education",
    "undata/SKILL-LTRCY": "Literacy",
    "undata/SKILL-MATH": "Minimum proficiency in mathematics",
    "undata/SKILL-READ": "Minimum proficiency in reading",
    "undata/POVERTY_STATUS-EXTREME_POVERTY": "In extreme poverty",
    "undata/POVERTY_STATUS-POVERTY": "In poverty",
    "undata/WASH_SERVICE_TYPE-SAN": "Sanitation",
    "undata/WASH_SERVICE_TYPE-HYG": "Hygiene",
    "undata/WASH_SERVICE_TYPE-WAT": "Drinking water",
}
_VALUE_GENDER_SLOT = SlotKeyDraft(axis="how", property_dcid="gender", label="gender")
_VALUE_LITERACY_SLOT = SlotKeyDraft(axis="how", property_dcid="literacyStatus", label="literacy")
_VALUE_LITERATE = ("literacyStatus", ["Literate"])
_VALUE_FEMALE = ("gender", ["Female"])
_VALUE_HEALTH_ODA = [_ODA, ("crsPurpose", ["CRSPurposeCode12220"])]


def _value_member_arcs(constraints: dict[str, str], name: str) -> dict:
    return {
        "name": {"nodes": [{"value": name}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
        **{prop: {"nodes": [{"dcid": value}]} for prop, value in constraints.items()},
    }


def _value_shape(
    shape_id: str,
    *slot_keys: SlotKeyDraft,
    members: Sequence[dict[str, str]],
    names: Sequence[str] = (),
) -> ShapeDraft:
    """A shape whose members hold `members`' constraints under `names`, by default each
    the shape id."""
    return replace(
        _shape(*slot_keys),
        shape_id=shape_id,
        sv_arc_facts={
            f"{shape_id}_{index}": _value_member_arcs(constraints, name)
            for index, (constraints, name) in enumerate(
                zip(members, names or [shape_id] * len(members), strict=True)
            )
        },
        value_labels=_VALUE_LABELS,
    )


_VALUE_INFANT_MORTALITY = _value_shape(
    "infant_mortality",
    SlotKeyDraft(axis="how", property_dcid="ageGroupClassification", label="age group"),
    SlotKeyDraft(axis="how", property_dcid="age", label="age"),
    members=[
        {"ageGroupClassification": "Infant"},
        {"age": "Years0"},
        {"ageGroupClassification": "Newborn"},
    ],
    names=["Infant mortality rate", "Infant Mortality Rate", "Newborn mortality rate"],
)


def _series_shape(shape_id: str, series: str, members: Sequence[dict[str, str]]) -> ShapeDraft:
    """A shape of the series named `series` whose members hold `members`' constraints."""
    return replace(_value_shape(shape_id, members=members), pop_type_name=series)


_UN_YOUTH_AGE = [{"undataAge": "undata/AGE-Y15T24"}]


# A bind menu sibling whose members name the value each row's variable asks for.
_VALUE_FEMALE_SIBLING = _value_shape(
    "female_sibling", members=[{"gender": "Female", "literacyStatus": "Literate"}]
)
_VALUE_LITERACY = _value_shape(
    "literacy",
    _VALUE_LITERACY_SLOT,
    members=[{"literacyStatus": "Literate", "age": "Years10Onwards"}],
)


@pytest.mark.parametrize(
    ("variable", "elected", "menu", "bound", "entity_roles", "records", "ratings"),
    [
        # The elected literacy rate has no gender slot, so "female" is unapplied, whatever
        # the bind model reports (female-literacy-bangladesh).
        (
            "female literacy rate",
            _VALUE_LITERACY,
            [_VALUE_FEMALE_SIBLING],
            [_VALUE_LITERATE],
            {},
            [("female", "no_vocabulary", None)],
            {},
        ),
        # An answer binding Female applies it.
        (
            "female literacy rate",
            _value_shape(
                "literacy_by_gender",
                _VALUE_LITERACY_SLOT,
                _VALUE_GENDER_SLOT,
                members=[{"literacyStatus": "Literate", "gender": "Female"}],
            ),
            [_VALUE_FEMALE_SIBLING],
            [_VALUE_LITERATE, _VALUE_FEMALE],
            {},
            [],
            {},
        ),
        # A gender slot left open answers every gender, so "female" is unapplied, a
        # category the gender slot leaves unbound.
        (
            "female literacy rate",
            _value_shape(
                "literacy_by_gender",
                _VALUE_LITERACY_SLOT,
                _VALUE_GENDER_SLOT,
                members=[
                    {"literacyStatus": "Literate", "gender": "Female"},
                    {"literacyStatus": "Literate", "gender": "Male"},
                ],
            ),
            [_VALUE_FEMALE_SIBLING],
            [_VALUE_LITERATE],
            {},
            [("female", "unbound", "gender")],
            {},
        ),
        # Every member of the elected shape holds gender Female, so its measure states it.
        (
            "female literacy rate",
            _value_shape(
                "female_literacy",
                _VALUE_LITERACY_SLOT,
                members=[{"literacyStatus": "Literate", "gender": "Female"}],
            ),
            [_VALUE_FEMALE_SIBLING],
            [_VALUE_LITERATE],
            {},
            [],
            {},
        ),
        # The member binding age 0 is named "Infant Mortality Rate", so the answer states
        # "infant" though it binds no age group (std-07).
        (
            "infant mortality rate",
            _VALUE_INFANT_MORTALITY,
            [],
            [("age", ["Years0"])],
            {},
            [],
            {},
        ),
        # An age group bound absent leaves the age-0 member alone.
        (
            "infant mortality rate",
            _VALUE_INFANT_MORTALITY,
            [],
            [("ageGroupClassification", [])],
            {},
            [],
            {},
        ),
        # With both slots open the answer holds the newborn rate too, whose name lacks
        # "infant", left unbound on the age group slot that holds Infant
        # (infant-mortality-rate).
        (
            "infant mortality rate",
            _VALUE_INFANT_MORTALITY,
            [],
            [],
            {},
            [("infant", "unbound", "ageGroupClassification")],
            {},
        ),
        # The elected shape's population type is the measure word "student".
        (
            "student literacy rate",
            replace(_VALUE_LITERACY, pop_type_dcid="Student"),
            [_value_shape("student_sibling", members=[{"occupation": "Student"}])],
            [_VALUE_LITERATE],
            {},
            [],
            {},
        ),
        # "German" is the origin role's own words (dq-crs-049), whatever value a sibling
        # labels with it.
        (
            "German literacy rate",
            _VALUE_LITERACY,
            [_value_shape("language_sibling", members=[{"language": "German"}])],
            [_VALUE_LITERATE],
            {"Germany": [DoorRole(axis="origin", span="German")]},
            [],
            {},
        ),
        # A code term is the code reading's: "health" names purpose 120, and the answer
        # applies it through its descendant 12220.
        (
            "health ODA",
            _value_shape("crs", *_CRS_SLOTS, members=[{"crsPurpose": "CRSPurposeCode12220"}]),
            [_value_shape("health_sibling", members=[{"crsPurpose": "CRSPurposeCode120"}])],
            _VALUE_HEALTH_ODA,
            {},
            [],
            {},
        ),
        # A bound parent purpose selects the members holding its descendants, as
        # materialisation reads it, and each of them holds gender Female.
        (
            "female aid",
            _value_shape(
                "aid_by_purpose",
                SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="purpose"),
                members=[
                    {"crsPurpose": "CRSPurposeCode12220", "gender": "Female"},
                    {"crsPurpose": "CRSPurposeCode11220", "gender": "Male"},
                ],
            ),
            [],
            [("crsPurpose", ["CRSPurposeCode120"])],
            {},
            [],
            {},
        ),
        # A code term is the code readers' even where the elected shape holds no slot of
        # its system: they judge "health", purpose 120, and this reader names nothing. The
        # literacy rate does not state "health", so the binder's rating leaves it unread.
        (
            "health literacy rate",
            _VALUE_LITERACY,
            [_value_shape("health_sibling", members=[{"crsPurpose": "CRSPurposeCode120"}])],
            [_VALUE_LITERATE],
            {},
            [("health", "no_vocabulary", None)],
            {"health": 0.0},
        ),
        # "primary education" is also the text of CRS purpose 11220, but the elected
        # completion rate holds no CRS slot, so the code readers judge nothing here and
        # the open education level leaves the value unbound.
        (
            "primary education completion rate",
            _value_shape(
                "completion",
                SlotKeyDraft(axis="how", property_dcid="educationLevel", label="level"),
                members=[
                    {"educationLevel": "EducationLevelPrimary"},
                    {"educationLevel": "EducationLevelSecondary"},
                ],
            ),
            [],
            [],
            {},
            [("primary education", "unbound", "educationLevel")],
            {},
        ),
        # Skill Literacy, the one skill of the UN youth/adult literacy rate, names that
        # series' measure, so "literacy" restricts nothing on the literate share elected
        # (literacy-rate).
        (
            "literacy rate",
            _series_shape(
                "pct_literate", "Percentage of the population that is literate", _UN_YOUTH_AGE
            ),
            [
                _series_shape(
                    "youth_adult_literacy",
                    "Youth/adult literacy rate",
                    [{"undataSkill": "undata/SKILL-LTRCY"}],
                )
            ],
            [],
            {},
            [],
            {},
        ),
        # The ECLAC series states both its poverty statuses and splits on them, so "in
        # extreme poverty" restricts it and another poverty series leaves it unapplied. The
        # poverty line series is defined over the people "living" describes, so the binder
        # rates it stated.
        (
            "population living in extreme poverty",
            _series_shape(
                "poverty_line",
                "Proportion of population below international poverty line",
                _UN_YOUTH_AGE,
            ),
            [
                _series_shape(
                    "eclac_poverty",
                    "Population living in extreme poverty and poverty",
                    [
                        {"undataPovertyStatus": "undata/POVERTY_STATUS-EXTREME_POVERTY"},
                        {"undataPovertyStatus": "undata/POVERTY_STATUS-POVERTY"},
                    ],
                )
            ],
            [],
            {},
            [("in extreme poverty", "no_vocabulary", None)],
            {"living": 1.0},
        ),
        # SE_TOT_PRFL splits on the two skills its series names (SDG proficiency). The
        # completion rate states neither "children" nor "achieving".
        (
            "children achieving minimum proficiency in mathematics",
            _series_shape("completion", "Completion rate", _UN_YOUTH_AGE),
            [
                _series_shape(
                    "se_tot_prfl",
                    "Proportion of children and young people achieving a minimum "
                    "proficiency level in reading and mathematics",
                    [
                        {"undataSkill": "undata/SKILL-MATH"},
                        {"undataSkill": "undata/SKILL-READ"},
                    ],
                )
            ],
            [],
            {},
            [
                ("children achieving", "no_vocabulary", None),
                ("minimum proficiency in mathematics", "no_vocabulary", None),
            ],
            {"children": 0.0, "achieving": 0.0},
        ),
        # The WHO WASH series splits on the three service types its series names. Deaths
        # from diarrhoea state no cause, so "attributable" and "inadequate" are unread.
        (
            "diarrhoea deaths attributable to inadequate sanitation",
            _series_shape("diarrhoea_deaths", "Number of deaths from diarrhoea", _UN_YOUTH_AGE),
            [
                _series_shape(
                    "diarrhoea_wash",
                    "Number of diarrhoea deaths attributable to inadequate water, "
                    "sanitation and hygiene",
                    [
                        {"undataWashServiceType": "undata/WASH_SERVICE_TYPE-SAN"},
                        {"undataWashServiceType": "undata/WASH_SERVICE_TYPE-HYG"},
                        {"undataWashServiceType": "undata/WASH_SERVICE_TYPE-WAT"},
                    ],
                )
            ],
            [],
            {},
            [
                ("attributable", "no_vocabulary", None),
                ("inadequate", "no_vocabulary", None),
                ("sanitation", "no_vocabulary", None),
            ],
            {"attributable": 0.0, "inadequate": 0.0},
        ),
    ],
    ids=[
        "unapplied",
        "applied",
        "slot-left-open",
        "measure-holds-the-value",
        "member-names-state-it",
        "age-group-absent",
        "member-names-differ",
        "measure-word",
        "role-words",
        "descendant-code",
        "parent-code-selects-descendants",
        "code-term-off-the-shape",
        "value-inside-a-code-text-off-the-shape",
        "value-names-its-shape-measure",
        "series-splits-on-poverty-status",
        "series-splits-on-skill",
        "series-splits-on-service-type",
    ],
)
def test_a_value_the_request_names_is_judged_against_the_answer(
    search_settings, variable, elected, menu, bound, entity_roles, records, ratings
):
    """Under the refuse coverage mode a word naming, by its graph label, a value a shape on
    the bind menu holds is a requirement the answer leaves unmet unless it binds that
    value, its measure states it, a place's words hold it, or the word is a code term of
    a system the answer holds. An unmet value is a category left unbound on the answer's
    slot for its property, and a distinction the data does not record (``no_vocabulary``)
    on an answer with no such slot. A value with no codes binds absent. The bind model's
    `unplaced` is empty throughout. A word no reader reads is a distinction the data does
    not record where `ratings`, the binder's, rate the elected series not stating it."""
    search_settings(qualifier_coverage="refuse")
    assert _value_refusal(variable, elected, menu, bound, entity_roles, ratings) == records


def _value_refusal(
    variable: str,
    elected: ShapeDraft,
    menu: Sequence[ShapeDraft],
    bound: Sequence[tuple[str, list[str]]] = (),
    entity_roles: dict[str, list[DoorRole]] | None = None,
    ratings: dict[str, float] | None = None,
) -> list[tuple[str, str, str | None]]:
    """The terms the elected answer refuses on, with their slots, once the variable's value
    requirements, read off `menu` and `elected`, are recorded, with `ratings` as the
    binder's rating of each word the coverage rule leaves unrated."""
    entity_roles = entity_roles or {}
    kinds: dict[int, DraftBindingKind] = {0: "absent", 1: "value"}
    state = _state(
        variable=variable,
        shape_draft=elected,
        bindings=[
            _binding(prop, kind=kinds.get(len(dcids), "set"), value_dcids=dcids)
            for prop, dcids in bound
        ],
        unplaced=[],
        entities=tuple(entity_roles),
    )
    state.bindings = state.bind_result.bindings
    state.entity_roles = entity_roles
    state.capped_shapes = [*menu, elected]
    state.stated_word_ratings = ratings or {}
    record_reading(state, _resolve_query_qualifiers(state))
    state.requirements = (*state.requirements, *value_requirements(state, state.query_qualifiers))
    missed = refusal_terms(state, state.shape_draft, state.bindings)
    return [(q.text, q.disposition, q.slot) for q in missed]


_VALUE_INFANT_TWO_SLOTS = _value_shape(
    "infant_two_slots",
    SlotKeyDraft(axis="how", property_dcid="ageGroupClassification", label="age group"),
    SlotKeyDraft(axis="how", property_dcid="age", label="age"),
    members=[
        {"ageGroupClassification": "Infant"},
        {"age": "Infant"},
        {"ageGroupClassification": "Newborn"},
    ],
    names=["Infant mortality rate", "Infant Mortality Rate", "Newborn mortality rate"],
)


def test_a_value_held_on_two_slots_is_named_on_the_first_property(search_settings):
    """Infant held under both the age and the age group slots of the elected shape, both
    left open: the unmet "infant" names the first property in sorted order, "age", in
    every run, and the refusal names the same term the answer's trace flags."""
    search_settings(qualifier_coverage="refuse")
    state = _state(
        variable="infant mortality rate",
        shape_draft=_VALUE_INFANT_TWO_SLOTS,
        bindings=[],
        unplaced=[],
        entities=(),
    )
    state.bindings = state.bind_result.bindings
    state.capped_shapes = [_VALUE_INFANT_TWO_SLOTS]
    record_reading(state, _resolve_query_qualifiers(state))
    state.requirements = (*state.requirements, *value_requirements(state, state.query_qualifiers))

    refused = list(refusal_terms(state, _VALUE_INFANT_TWO_SLOTS, state.bindings))
    assert [(q.text, q.disposition, q.slot) for q in refused] == [("infant", "unbound", "age")]
    assert [ref.ref for ref in refused[0].named] == ["Infant"]
    trace = answer_terms(state, _VALUE_INFANT_TWO_SLOTS, state.bindings)
    assert refused == [q for q in trace if q.disposition != "bound"]


_LITERACY_SERIES = "undata/SERIES-SE_ADT_LITRAT"
_PROFICIENCY_SERIES = "undata/SERIES-SE_TOT_PRFL"


def _un_member(series: str, skill: str, age: str) -> UnVariable:
    return un_variable(
        series,
        name=f"{series} {skill} {age}",
        constraints={
            "undataSkill": ("Skill", skill, _VALUE_LABELS[skill]),
            "undataAge": ("Age", age, age),
        },
    )


def _un_sample(series: str, name: str, skill: str) -> ShapeDraft:
    """A recalled sample of the UN `series`, one member holding `skill`."""
    return replace(
        _series_shape(series, name, [{"undataSkill": skill}]),
        pop_type_dcid=series,
        meas_prop_dcid="value",
        stat_type_dcid="measuredValue",
        sv_arc_facts_complete=False,
    )


@pytest.mark.parametrize(
    ("variable", "menu", "members", "records", "ratings"),
    [
        # The catalogue lists every member of the youth/adult literacy rate, and each holds
        # Skill Literacy, so "literacy" names the series' measure.
        (
            "literacy rate",
            _un_sample(_LITERACY_SERIES, "Youth/adult literacy rate", "undata/SKILL-LTRCY"),
            [
                _un_member(_LITERACY_SERIES, "undata/SKILL-LTRCY", "undata/AGE-Y15T24"),
                _un_member(_LITERACY_SERIES, "undata/SKILL-LTRCY", "undata/AGE-Y15T99"),
            ],
            [],
            {},
        ),
        # The sample shows mathematics alone, but the catalogue's members split on reading
        # and mathematics, so the skill restricts the series. The literate share states
        # neither "children" nor "achieving".
        (
            "children achieving minimum proficiency in mathematics",
            _un_sample(
                _PROFICIENCY_SERIES,
                "Proportion of children and young people achieving a minimum proficiency "
                "level in reading and mathematics",
                "undata/SKILL-MATH",
            ),
            [
                _un_member(_PROFICIENCY_SERIES, "undata/SKILL-MATH", "undata/AGE-Y15T24"),
                _un_member(_PROFICIENCY_SERIES, "undata/SKILL-READ", "undata/AGE-Y15T24"),
            ],
            [
                ("children achieving", "no_vocabulary", None),
                ("minimum proficiency in mathematics", "no_vocabulary", None),
            ],
            {"children": 0.0, "achieving": 0.0},
        ),
        # No source lists the series' whole family, so its sample cannot show the value
        # restricts nothing.
        (
            "literacy rate",
            _un_sample(_LITERACY_SERIES, "Youth/adult literacy rate", "undata/SKILL-LTRCY"),
            [],
            [("literacy", "no_vocabulary", None)],
            {},
        ),
    ],
    ids=["family-holds-one-value", "sample-hides-a-split", "family-unlisted"],
)
def test_a_value_names_its_series_measure_only_when_its_whole_family_holds_it(
    search_settings, bind_catalogue, variable, menu, members, records, ratings
):
    """Whether a value its holding series states is that series' measure is read off the
    series' whole family, the UN tables of the origin catalogue here, never off the bind
    menu's recalled sample."""
    search_settings(qualifier_coverage="refuse")
    bind_catalogue(
        FakeCatalogue(
            un=UnTables(mirror=un_mirror({f"member_{i}": m for i, m in enumerate(members)}))
        )
    )
    elected = _series_shape(
        "pct_literate", "Percentage of the population that is literate", _UN_YOUTH_AGE
    )

    assert _value_refusal(variable, elected, [menu], ratings=ratings) == records


@pytest.mark.parametrize("mode", ["refuse", "log"])
def test_coverage_mode_decides_whether_an_unapplied_value_refuses(search_settings, mode):
    """In refuse mode the unapplied "female" is a requirement the elected answer leaves
    unmet, so the part is a miss. In log mode no requirement is recorded and the
    would-refuse line names it once."""
    search_settings(qualifier_coverage=mode)
    state = _state(
        variable="female literacy rate",
        shape_draft=_VALUE_LITERACY,
        bindings=[_binding("literacyStatus", kind="value", value_dcids=["Literate"])],
        unplaced=[],
        entities=(),
    )
    state.capped_shapes = [_VALUE_FEMALE_SIBLING, _VALUE_LITERACY]
    _run(state)
    missed = [
        (q.text, q.disposition) for q in refusal_terms(state, state.shape_draft, state.bindings)
    ]
    if mode == "refuse":
        assert missed == [("female", "no_vocabulary")]
        assert _would_refuse(state) == []
    else:
        assert missed == []
        assert _would_refuse(state) == [("female", [])]


_FEMALE_UNBOUND = code_requirement(
    property_dcid="gender",
    codes=("Female",),
    asked="Female",
    named=[GraphRef(ref="Female", label="Female")],
    departure=(
        qualifier_dropped_warning("Female"),
        measure_not_applied(asked="Female", property_dcid="gender"),
    ),
    origin="unbinding",
)


@pytest.mark.parametrize(
    "unbound",
    [(), (_FEMALE_UNBOUND,)],
    ids=["unmet-by-one-answer", "already-departed-by-unbinding"],
)
def test_a_candidate_set_departs_an_unmet_value_once(search_settings, unbound):
    """A candidate set of the literacy rate, which leaves "female" unmet, and the female
    sibling, which applies it, departs "female" once. When unbinding has also dropped Female
    (a requirement on gender, which neither answer's slots hold), the set still departs it
    once, and the literacy rate's trace and a refusal on it name it, as the same term."""
    search_settings(qualifier_coverage="refuse")
    state = _state(
        variable="female literacy rate",
        shape_draft=_VALUE_LITERACY,
        bindings=[_binding("literacyStatus", kind="value", value_dcids=["Literate"])],
        unplaced=[],
        entities=(),
    )
    state.bindings = state.bind_result.bindings
    state.capped_shapes = [_VALUE_FEMALE_SIBLING, _VALUE_LITERACY]
    record_reading(state, _resolve_query_qualifiers(state))
    state.requirements = (
        *state.requirements,
        *value_requirements(state, state.query_qualifiers),
        *unbound,
    )

    settle_candidate_set(state, [(_VALUE_LITERACY, state.bindings), (_VALUE_FEMALE_SIBLING, [])])

    assert state.judgement is not None
    assert [(fact.code, fact.asked) for _warning, fact in state.judgement.departures()] == [
        ("measure_not_applied", "female")
    ]
    trace = answer_terms(state, _VALUE_LITERACY, state.bindings)
    assert [q.text for q in trace if q.disposition == "no_vocabulary"] == ["female"]
    assert list(refusal_terms(state, _VALUE_LITERACY, state.bindings)) == [
        q for q in trace if q.disposition != "bound"
    ]


_VALUE_LITERACY_BY_GENDER = _value_shape(
    "literacy_by_gender",
    _VALUE_GENDER_SLOT,
    _VALUE_LITERACY_SLOT,
    members=[
        {"gender": "Female", "literacyStatus": "Literate"},
        {"gender": "Male", "literacyStatus": "Literate"},
    ],
)


@pytest.mark.parametrize(
    ("answers", "refused", "departed"),
    [
        ([_VALUE_LITERACY_BY_GENDER], [("Female", "unbound", "gender")], []),
        ([_VALUE_LITERACY_BY_GENDER, _VALUE_FEMALE_SIBLING], None, [("Female", "how:gender")]),
        ([_VALUE_FEMALE_SIBLING], [], []),
        ([_VALUE_FEMALE_SIBLING, _VALUE_FEMALE_SIBLING], None, []),
    ],
    ids=[
        "answer-holds-the-property",
        "set-member-holds-the-property",
        "answer-applies-the-value-off-the-property",
        "set-applies-the-value-off-the-property",
    ],
)
def test_a_phrase_a_value_and_unbinding_both_require_is_judged_once(
    search_settings, answers, refused, departed
):
    """ "female" is a value the request names and, as Female on gender, a code unbinding
    dropped (recorded after it). An answer whose shape holds gender and binds nothing on
    it refuses on the code term with its slot, and a set departs that code's departure. An
    answer applying Female without a gender slot meets the phrase, so nothing departs.
    `refused` is `None` for a candidate set, which refuses nothing."""
    search_settings(qualifier_coverage="refuse")
    state = _state(
        variable="female literacy rate",
        shape_draft=_VALUE_LITERACY_BY_GENDER,
        bindings=[_binding("literacyStatus", kind="value", value_dcids=["Literate"])],
        unplaced=[],
        entities=(),
    )
    state.bindings = state.bind_result.bindings
    state.capped_shapes = [_VALUE_FEMALE_SIBLING, _VALUE_LITERACY_BY_GENDER]
    record_reading(state, _resolve_query_qualifiers(state))
    state.requirements = (
        *state.requirements,
        *value_requirements(state, state.query_qualifiers),
        _FEMALE_UNBOUND,
    )
    read = [
        (shape, state.bindings if shape is _VALUE_LITERACY_BY_GENDER else []) for shape in answers
    ]

    if refused is None:
        settle_candidate_set(state, read)
    else:
        terms = settle_answer(state, *read[0])
        assert [(q.text, q.disposition, q.slot) for q in terms] == refused

    assert state.judgement is not None
    assert [
        (fact.asked, fact.binding_ref) for _warning, fact in state.judgement.departures()
    ] == departed


_VALUE_YOUTH_LITERACY = _value_shape(
    "youth_literacy",
    _VALUE_LITERACY_SLOT,
    members=[{"literacyStatus": "Literate", "age": "Years15To24"}],
)


_BOTH = ["Female", "Male"]
_EITHER = ("female literacy rate", "male literacy rate")
# Gender values the graph and the code index hold no name for.
_NAMELESS = "GenderUnnamed"
_NAMELESS_TOO = "GenderUnnamedToo"


@pytest.mark.parametrize(
    ("variable", "values", "answers", "definite", "departed"),
    [
        *(
            (
                variable,
                _BOTH,
                [(_VALUE_LITERACY_BY_GENDER, []), (_VALUE_LITERACY_BY_GENDER, [])],
                False,
                [(other, "how:gender"), (named, "how:gender")],
            )
            for variable, named, other in (
                ("female literacy rate", "Female", "Male"),
                ("male literacy rate", "Male", "Female"),
            )
        ),
        *(
            (
                f"{word} literacy rate",
                values,
                [(_VALUE_LITERACY, []), (_VALUE_YOUTH_LITERACY, [])],
                False,
                [(word, None)]
                + [(value, "how:gender") for value in values if value.casefold() != word],
            )
            for values in (_BOTH, ["Male", "Female"], ["Female", "Male", "Other"])
            for word in ("female", "male")
        ),
        ("female literacy rate", _BOTH, [(_VALUE_LITERACY, [])], True, [("Male", "how:gender")]),
        ("male literacy rate", _BOTH, [(_VALUE_LITERACY, [])], True, [("Female", "how:gender")]),
        *(
            (
                variable,
                _BOTH,
                [
                    (_VALUE_LITERACY_BY_GENDER, ["Female"]),
                    (_VALUE_LITERACY_BY_GENDER, ["Female"]),
                ],
                False,
                [("Male", "how:gender")],
            )
            for variable in _EITHER
        ),
        *(
            (
                "female literacy rate",
                [*_BOTH, _NAMELESS],
                [(_VALUE_LITERACY_BY_GENDER, applied)] * 2,
                False,
                departed,
            )
            for applied, departed in (
                ([], [("Male", "how:gender"), (None, "how:gender"), ("Female", "how:gender")]),
                (_BOTH, [(None, "how:gender")]),
                (["Female"], [("Male", "how:gender"), (None, "how:gender")]),
            )
        ),
        (
            "female literacy rate",
            [*_BOTH, _NAMELESS],
            [(_VALUE_LITERACY, [])],
            True,
            [("Male", "how:gender"), (None, "how:gender")],
        ),
        *(
            (
                "literacy rate",
                ["Female", _NAMELESS, _NAMELESS_TOO],
                [(_VALUE_LITERACY_BY_GENDER, applied)] * 2,
                False,
                departed,
            )
            for applied, departed in (
                ([_NAMELESS], [("Female", "how:gender"), (None, "how:gender")]),
                (["Female", _NAMELESS], [(None, "how:gender")]),
            )
        ),
    ],
    ids=[
        "female-set-holds-the-property",
        "male-set-holds-the-property",
        "female-set-lacks-the-property",
        "male-set-lacks-the-property",
        "female-set-lacks-the-property-reversed",
        "male-set-lacks-the-property-reversed",
        "female-set-lacks-the-property-three-values",
        "male-set-lacks-the-property-three-values",
        "female-definite-lacks-the-property",
        "male-definite-lacks-the-property",
        "female-set-applies-female",
        "male-set-applies-female",
        "nameless-set-applies-none",
        "nameless-set-applies-the-named",
        "nameless-set-applies-female",
        "nameless-definite-lacks-the-property",
        "two-nameless-set-applies-one",
        "two-nameless-set-applies-female-and-one",
    ],
)
def test_a_phrase_sharing_a_departure_departs_once(
    search_settings, variable, values, answers, definite, departed
):
    """Bind sets gender to `values` with no recorded term, and the request's words name
    one of them by its graph label. Unbinding requires the named value and records each
    other value as the bind model's alone (`model_only_values`), which an answer leaving
    it unapplied departs and is never refused on. Each answer binds the gender values its
    row names. Every phrase departs once: the named value under the words' value
    requirement when the answers lack gender, else under its own name, and each
    model-only value under its own name unless every answer applies it. A definite answer
    without gender refuses on the named value and departs the others. A value with no
    name departs under its slot's name, and no departure's text names its dcid. Two such
    values share that departure, which still departs when an answer applies one of them
    and not the other."""
    search_settings(qualifier_coverage="refuse")
    state = _state(
        variable=variable,
        shape_draft=_VALUE_LITERACY_BY_GENDER,
        bindings=[_binding("literacyStatus", kind="value", value_dcids=["Literate"])],
        unplaced=[],
        entities=(),
    )
    state.graph = cast(
        EngineGraphClient,
        FakeGraph(
            nodes={g: {"label": g, "type": "Thing"} for g in ("Female", "Male", "Other")},
            obs={},
            detect={},
            resolve={},
        ),
    )
    state.bindings = state.bind_result.bindings
    state.capped_shapes = [
        _VALUE_FEMALE_SIBLING,
        _VALUE_LITERACY_BY_GENDER,
        _VALUE_LITERACY,
        _VALUE_YOUTH_LITERACY,
    ]
    record_reading(state, _resolve_query_qualifiers(state))
    state.requirements = (*state.requirements, *value_requirements(state, state.query_qualifiers))
    state.bindings = [*state.bindings, _binding("gender", kind="value", value_dcids=values)]
    unbind_unrealisable(
        state,
        BoundValuesUnrealisable(
            conflicts=(MemberConflict(properties=frozenset({"gender"}), fully_bound=True),)
        ),
    )
    read = [
        (shape, [*state.bindings, _binding("gender", kind="value", value_dcids=gender)])
        if gender
        else (shape, state.bindings)
        for shape, gender in answers
    ]

    if definite:
        terms = settle_answer(state, *read[0])
        assert [(q.text, q.disposition) for q in terms] == [(variable.split()[0], "no_vocabulary")]
    else:
        settle_candidate_set(state, read)

    assert state.judgement is not None
    departures = state.judgement.departures()
    assert [(fact.asked, fact.binding_ref) for _warning, fact in departures] == departed
    assert not any(
        dcid in text
        for dcid in (_NAMELESS, _NAMELESS_TOO)
        for warning, fact in departures
        for text in (warning.message, fact.asked or "", *fact.values)
    )


# A phrase's value requirement (`None` when it has none) and unbinding code requirement,
# as one answer judges them.
type _PhraseState = tuple[Outcome | None, Outcome]
_PHRASE_STATES: list[_PhraseState] = [
    (value, code) for value in (None, "met", "unmet") for code in ("met", "unmet", "outside")
]


def _invariant_ledger(
    labels: Sequence[str],
    states: Sequence[_PhraseState],
    key: str,
    nameless: Sequence[Outcome] = (),
) -> list[tuple[Requirement, Outcome]]:
    """Per value in `labels`, its phrase's requirements and their outcomes, values first,
    then a code requirement per value with no name in `nameless`. A named value's code
    departs under its name (`key` "own"), under the request's words, which is the value
    requirement's key with another fact ("words"), or joined with the other named values
    and alone under its name ("joined"), or under words every value shares and alone
    under its name ("shared"), as unbinding records them. A nameless value departs the
    slot's generic departure, beside the shared words under "shared"."""
    joined = " and ".join(labels)
    shared = unapplied_departure("those values", asked="those values", property_dcid="gender")
    values: list[tuple[Requirement, Outcome]] = []
    codes: list[tuple[Requirement, Outcome]] = []
    for label, (value, code) in zip(labels, states, strict=True):
        words = label.lower()
        if value is not None:
            requirement = Requirement(
                term=ResolvedQualifier(text=words, named=[], disposition="no_vocabulary"),
                origin="value",
                rule=ValueRule(values=frozenset({("gender", label)})),
                departure=unapplied_departure(words, asked=words, property_dcid=None),
            )
            values.append((requirement, value))
        own = unapplied_departure(label, asked=label, property_dcid="gender")
        departure = {
            "own": own,
            "words": unapplied_departure(words, asked=words, property_dcid="gender"),
            "joined": unapplied_departure(joined, asked=joined, property_dcid="gender"),
            "shared": shared,
        }[key]
        requirement = code_requirement(
            property_dcid="gender",
            codes=(label,),
            asked=label,
            named=[GraphRef(ref=label, label=label)],
            departure=departure,
            origin="unbinding",
            alone=own if key in ("joined", "shared") else None,
        )
        codes.append((requirement, code))
    for dcid, code in zip(_NAMELESS_DCIDS, nameless, strict=False):
        requirement = code_requirement(
            property_dcid="gender",
            codes=(dcid,),
            asked=None,
            named=(),
            departure=shared if key == "shared" else None,
            origin="unbinding",
            generic=_unnamed_departure("gender", "gender"),
        )
        codes.append((requirement, code))
    return [*values, *codes]


_NAMELESS_DCIDS = ("X1", "X2")


def _invariant_cases():
    for count in (1, 2, 3):
        labels = ("Female", "Male", "Other")[:count]
        for key in ("own", "words", "joined"):
            for states in itertools.product(_PHRASE_STATES, repeat=count):
                for named in itertools.product((False, True), repeat=count):
                    refused = {label for label, name in zip(labels, named, strict=True) if name}
                    yield labels, states, key, refused, True, ()
            # With no answer only whether a phrase has a value requirement matters.
            for value in itertools.product(("met", None), repeat=count):
                yield labels, [(v, "outside") for v in value], key, set(), False, ()
    # Up to two named values beside one or two nameless ones.
    for count, unnamed in itertools.product((0, 1, 2), (1, 2)):
        labels = ("Female", "Male")[:count]
        phrases = (*labels, *_NAMELESS_DCIDS[:unnamed])
        for key in ("own", "words", "joined", "shared"):
            for states in itertools.product(_PHRASE_STATES, repeat=count):
                for nameless in itertools.product(("met", "unmet", "outside"), repeat=unnamed):
                    for named in itertools.product((False, True), repeat=len(phrases)):
                        refused = {p for p, name in zip(phrases, named, strict=True) if name}
                        yield labels, states, key, refused, True, nameless
            for value in itertools.product(("met", None), repeat=count):
                outside = ("outside",) * unnamed
                yield labels, [(v, "outside") for v in value], key, set(), False, outside


def _judged(
    ledger: Sequence[tuple[Requirement, Outcome]], refused: set[str], answered: bool
) -> Judgement:
    return Judgement(
        requirements=tuple(requirement for requirement, _ in ledger),
        outcomes=(tuple(outcome for _, outcome in ledger),) if answered else (),
        refused=tuple(
            ResolvedQualifier(text=label.lower(), named=[], disposition="no_vocabulary")
            for label in refused
        ),
    )


def test_departures_disclose_each_phrase_needing_it_once():
    """For one to three named values of one slot, each phrase met, unmet or outside under
    a value requirement and an unbinding code requirement, and for up to two named values
    beside one or two with no name, each met, unmet or outside, under each kind of
    departure key, with one answer or none, and with the refusal naming any of the
    phrases: each phrase the answer does not meet and the refusal does not name is
    disclosed by exactly one departure, a generic one counting only for the phrases that
    need it and no other departure discloses; no other departure discloses a phrase the
    answer meets or the refusal names; a generic departure departs only when one of its
    phrases needs it; a departure disclosing exactly the phrases that need one departs
    alone; and the ledger departs the same in every order (every order up to two named
    values with none nameless; otherwise values or codes first, each reversed). The
    76,244 ledgers take about 11 s, most of it judging each in several orders."""
    cases = 0
    for labels, states, key, refused, answered, nameless in _invariant_cases():
        cases += 1
        case = (labels, states, key, refused, answered, nameless)
        ledger = _invariant_ledger(labels, states, key, nameless)
        departed = _judged(ledger, refused, answered).departures()

        covers: dict[tuple[str, str], set[str]] = {}
        generic: dict[tuple[str, str], set[str]] = {}
        for requirement, _ in ledger:
            for departure in (requirement.departure, requirement.alone):
                if departure is not None:
                    covers.setdefault(warning_key(departure[0]), set()).add(requirement.phrase)
            if requirement.generic is not None:
                generic.setdefault(warning_key(requirement.generic[0]), set()).add(
                    requirement.phrase
                )
        needing = {
            label.lower()
            for label, state in zip(labels, states, strict=True)
            if label not in refused and not (answered and "unmet" not in state and "met" in state)
        } | {
            dcid.lower()
            for dcid, state in zip(_NAMELESS_DCIDS, nameless, strict=False)
            if dcid not in refused and not (answered and state == "met")
        }
        keys = [warning_key(warning) for warning, _fact in departed]
        emitted = [covers[k] for k in keys if k in covers]
        assert all(phrases <= needing for phrases in emitted), case
        disclosed = set().union(*emitted)
        emitted += [generic[k] & needing - disclosed for k in keys if k not in covers]
        assert all(emitted), case
        for phrase in needing:
            assert sum(phrase in phrases for phrases in emitted) == 1, case
        if needing in covers.values():
            assert len(emitted) == 1, case

        expected = sorted(map(repr, departed))
        size = len(ledger)
        values = size - len(labels) - len(nameless)
        codes_first = [*ledger[values:], *ledger[:values]]
        orders = (
            [[ledger[i] for i in order] for order in itertools.permutations(range(size))]
            if len(labels) < 3 and not nameless
            else [ledger, ledger[::-1], codes_first, codes_first[::-1]]
        )
        for order in orders:
            assert sorted(map(repr, _judged(order, refused, answered).departures())) == expected
    assert cases == 76_244


@pytest.mark.parametrize("met", [0, 1], ids=["all-unmet", "first-met"])
def test_a_joined_departure_of_many_values_departs_once_or_alone(met):
    """Unbinding drops 24 named values of one property under one joined departure. An
    answer leaving every value unmet departs it once; one meeting the first departs each
    other value alone."""
    labels = {f"V{i}": f"Value {i}" for i in range(24)}
    joined = _unapplied(" and ".join(labels.values()), "prop")
    requirements = tuple(
        _value_requirements(
            "prop",
            list(labels),
            labels,
            joined,
            unnamed=_unnamed_departure("prop", "prop"),
            named_only=True,
        )
    )
    judgement = Judgement(
        requirements=requirements, outcomes=(("met",) * met + ("unmet",) * (24 - met),)
    )

    assert [fact.asked for _warning, fact in judgement.departures()] == (
        [" and ".join(labels.values())] if not met else list(labels.values())[1:]
    )
