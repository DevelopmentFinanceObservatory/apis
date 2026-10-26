"""Test qualifier admission against candidate shapes, then binding and warnings.

Real CRS cases use the fixture index; a synthetic index checks that admission
also works for code systems outside the CRS corpus.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from collections.abc import Sequence
from typing import Literal, cast
from unittest.mock import patch

import pytest

from code_index.query import CodeIndex
from code_index.shared import get_index, get_measure_index
from code_index.types import CodeRow, CodeSystemDeclaration, IndexStamp, MeasureValueRow
from entity_resolution import Admission, Entity, EntityResolver, Resolved
from search.engine.bind import (
    CandidateMenu,
    SlotBindingDraft,
    _build_universal_bind_prompt,
)
from search.engine.binders import GeminiBinder
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.departures import warning_key
from search.engine.fanout import _FanOutResult, _run_variables_streaming
from search.engine.interpret import Recall
from search.engine.qualifier_match import (
    QualifierMatch,
    QualifierReadings,
    QualifierTie,
    bindable_codes,
    governing_connectives,
    match_qualifier_ties,
    match_qualifiers,
)
from search.engine.quantity_read import QuantityOutput
from search.engine.region import RegionResult
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.region.phase_election import _vet_election
from search.engine.region.phase_rank_bind import (
    _admit_qualifier_matches,
    _admit_qualifier_matches_against_elected_shape,
    _admit_qualifier_ties,
    _candidate_menus,
    _coverage_records,
    _dataset_candidates,
    _dataset_stated_words,
    _filter_admitted_matches,
    _force_qualifier_bindings,
    _log_admitted_qualifiers,
    _menu_choices,
    _resolve_query_qualifiers,
    _undeclared_qualifiers,
)
from search.engine.region.pipeline import resolve_variable
from search.engine.region.qualifier_readings import match_query_readings
from search.engine.region.state import ResolveState, shape_vocabulary
from search.engine.retrieval.bound_values import held_codes, shape_for_part
from search.engine.retrieval.derive import derive_shapes
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.heading.departure_fact import DepartureFact
from search.models import SourceMention
from search.render.plan import render_sentence
from series_coverage.schema import RoleNames
from tests.engine._binders import elected_reply
from tests.engine._builders import place_sets
from tests.engine._fake_coverage import FakeCoverageService, empty_coverage_snapshot, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.engine._index_slots import (
    crs_flow_slot_keys,
    crs_vocabulary,
    own_variable_spans,
    query_qualifiers,
    read_text,
)
from tests.engine._quantity import NO_QUANTITY, QuantityReadOnlyLLM
from tests.engine._requirements import depart_requirements
from tests.engine._source_scopes import part_scope
from tests.engine.test_two_ended_menu import _CRS_SV as _TWO_ENDED_CRS_SV
from tests.engine.test_two_ended_menu import _DAC2A_ODA_SV as _TWO_ENDED_DAC2A_ODA_SV
from tests.engine.test_two_ended_menu import _SUBJECT_SV as _TWO_ENDED_SUBJECT_SV
from tests.engine.test_two_ended_menu import _index_row_node
from tests.engine.test_two_ended_menu import _resolve as _resolve_two_ended
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_REAL_INDEX = get_index()
_CRS_PROVENANCE = "provenance/OECDCreditorReportingSystem"
_DAC2A_PROVENANCE = "provenance/OECDDAC2A"


def _admit(
    variable: str,
    shapes: list[ShapeDraft],
    dropped_qualifier_phrases: list[str] | None = None,
    query_text: str | None = None,
) -> tuple[QualifierMatch, ...]:
    """Match the text against the candidate shapes, as the prune step does, then admit
    results against them."""
    read = query_qualifiers(
        variable,
        dropped_qualifier_phrases or [],
        query_text=query_text,
        vocabularies=[shape_vocabulary(shape) for shape in shapes],
    )
    return _admit_qualifier_matches(shapes, read.matches)


_CRS = crs_vocabulary()


def _crs_ties(query: str) -> tuple[QualifierTie, ...]:
    """The ties of `query` read in a CRS shape's vocabulary."""
    return read_text(query, _REAL_INDEX, vocabularies=[_CRS]).by_vocabulary[_CRS].ties


# ---------------------------------------------------------------------------
# Shared ShapeDraft builders
# ---------------------------------------------------------------------------


def _shape(shape_id: str, slot_keys: tuple[SlotKeyDraft, ...]) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label="test shape",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="grantEquivalent",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=slot_keys,
        # A recall sample, so its members are the rows of the CRS grant-equivalent family,
        # whose codes tie admission reads (`held_codes`).
        sv_arc_facts_complete=False,
    )


# A shape declaring all four real CRS constraint properties the six-phrase
# regression below exercises -- mirrors a real CRS StatVar group's own
# constraint_properties union (retrieval/derive.py).
_CRS_SLOT_KEYS = (
    SlotKeyDraft(axis="what", property_dcid="crsTypeOfFlow", label="flow type"),
    SlotKeyDraft(axis="what", property_dcid="crsTypeOfFinance", label="finance type"),
    SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="purpose"),
    SlotKeyDraft(axis="how", property_dcid="crsCooperationModality", label="modality"),
)
_CRS_SHAPE = _shape("crs_shape", _CRS_SLOT_KEYS)

# A shape from an entirely different family: no CRS property in sight, so no
# match_qualifiers row for any CRS code system is ever admitted against it,
# regardless of what the raw text happens to name.
_DEMOGRAPHIC_SHAPE = _shape(
    "demographic_shape",
    (SlotKeyDraft(axis="how", property_dcid="age", label="age group"),),
)

# _CRS_SHAPE plus two measure-family constraint properties -- declares
# nettingBasis and financialObligationType so a measure match on either is
# admissible, unlike _CRS_SHAPE alone.
_CRS_SHAPE_WITH_MEASURE = _shape(
    "crs_shape_with_measure",
    (
        *_CRS_SLOT_KEYS,
        SlotKeyDraft(axis="what", property_dcid="nettingBasis", label="netting basis"),
        SlotKeyDraft(axis="what", property_dcid="financialObligationType", label="obligation type"),
    ),
)


def _by_system(matches: tuple[QualifierMatch, ...]) -> dict[str, QualifierMatch]:
    return {m.code_system: m for m in matches}


# ---------------------------------------------------------------------------
# Pre-bind admission: the six-phrase raw-text-door regression, at
# _admit_qualifier_matches, against the real fixture index.
# ---------------------------------------------------------------------------


def test_plain_health_query_admits_the_purpose_qualifier():
    by_system = _by_system(_admit("ODA grants for health", [_CRS_SHAPE]))
    assert by_system["crsPurpose"].code_dcid == "CRSPurposeCode120"
    assert by_system["crsTypeOfFlow"].code_dcid == "CRSTypeOfFlow10"
    assert "crsTypeOfFinance" not in by_system
    assert "crsCooperationModality" not in by_system


def test_plain_malaria_query_admits_the_purpose_qualifier():
    by_system = _by_system(_admit("ODA grants for malaria control", [_CRS_SHAPE]))
    assert by_system["crsPurpose"].code_dcid == "CRSPurposeCode12262"
    assert "crsTypeOfFinance" not in by_system
    assert "crsCooperationModality" not in by_system


def test_standard_grants_admits_the_finance_type_qualifier():
    by_system = _by_system(_admit("ODA standard grants for health", [_CRS_SHAPE]))
    finance = by_system["crsTypeOfFinance"]
    assert finance.code_dcid == "CRSTypeOfFinance110"
    assert finance.label == "standard grant"
    assert by_system["crsPurpose"].code_dcid == "CRSPurposeCode120"


def test_modality_d02_admits_the_modality_qualifier():
    by_system = _by_system(
        _admit("ODA grants for malaria control through modality D02", [_CRS_SHAPE])
    )
    modality = by_system["crsCooperationModality"]
    assert modality.code_dcid == "CRSCooperationModalityD02"
    assert modality.source == "code"
    assert "crsTypeOfFinance" not in by_system


def test_purpose_code_120_admits_the_exact_code_qualifier():
    by_system = _by_system(
        _admit("ODA grants for records coded exactly to purpose code 120", [_CRS_SHAPE])
    )
    purpose = by_system["crsPurpose"]
    assert purpose.code_dcid == "CRSPurposeCode120"
    assert purpose.source == "code"
    assert "crsTypeOfFinance" not in by_system


def test_entities_only_query_admits_only_the_flow_qualifier():
    by_system = _by_system(_admit("ODA grants from France to Togo", [_CRS_SHAPE]))
    assert set(by_system) == {"crsTypeOfFlow"}
    assert by_system["crsTypeOfFlow"].code_dcid == "CRSTypeOfFlow10"


# ---------------------------------------------------------------------------
# These ordinary query terms match CRS rows but are not admitted by a
# demographic shape with no CRS constraint properties.
# ---------------------------------------------------------------------------

_COLLISION_QUERIES = (
    "welfare spending trends",
    "insurance coverage by district",
    "nutrition programs in rural areas",
    "media consumption habits",
    "radio ownership rates",
    "statistics office staffing",
    "roads maintenance budget",
    "election turnout by region",
    "gold reserves held by central banks",
    "coal production totals",
)


def test_collision_words_raw_match_a_real_crs_row():
    """Fixture cross-check, not the behaviour under test: each collision
    query really does resolve a code row by qualifier_match.py's own rule
    (a genuinely unambiguous text) -- if a corpus refresh ever breaks one of
    these, the admission test below would otherwise pass for the wrong
    reason (nothing to refuse)."""
    for query in _COLLISION_QUERIES:
        assert read_text(query, _REAL_INDEX, vocabularies=[_CRS]).by_vocabulary[_CRS].matches, query


def test_collision_words_are_never_admitted_against_a_non_crs_shape():
    for query in _COLLISION_QUERIES:
        assert _admit(query, [_DEMOGRAPHIC_SHAPE]) == (), query


def test_no_capped_shape_at_all_admits_nothing():
    assert _admit("ODA standard grants for health", []) == ()


# ---------------------------------------------------------------------------
# A code no index row holds claims no word in a shape's own reading.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variable", "everywhere", "own"),
    [
        (
            "population policies and reproductive health ODA",
            {"population": ["CRSTypeOfFinance4"]},
            {},
        ),
        (
            "goods donations",
            {"goods donations": ["CRSCooperationModalityJ", "CRSCooperationModalityJ01"]},
            {},
        ),
        ("gross national income", {"gross national income": ["CRSTypeOfFinance1"]}, {}),
        ("ODA", {}, {"ODA": ["CRSTypeOfFlow10"]}),
    ],
    ids=["population", "goods-donations", "gross-national-income", "held-code"],
)
def test_a_code_no_row_holds_claims_no_word_on_a_shape(variable, everywhere, own):
    """`CRSTypeOfFinance4` ("population"), the in-kind modalities and `CRSTypeOfFinance1`
    ("gross national income") are codes no index row holds, itself or through a
    descendant, so a CRS shape's own reading neither matches nor ties them, and
    "population" is free for a purpose reading. The reading in every code system keeps
    them, and with them its measure values. "gross" in "gross national income" is no
    netting basis on any shape. A shape's own reading matches the CRS flow type ODA, which
    rows hold."""
    reading = read_text(variable, _REAL_INDEX, get_measure_index(), vocabularies=[_CRS])

    bindable = bindable_codes(_REAL_INDEX)
    unheld = {
        span.matched_text: sorted(row.dcid for row in span.rows)
        for span in reading.everywhere.spans
        if all(isinstance(row, CodeRow) and row.dcid not in bindable for row in span.rows)
    }
    assert unheld == everywhere
    crs_codes = {
        span.matched_text: sorted(row.dcid for row in span.rows)
        for span in reading.by_vocabulary[_CRS].spans
        if all(isinstance(row, CodeRow) for row in span.rows)
    }
    assert {text: crs_codes[text] for text in own} == own
    assert not set(crs_codes) & set(everywhere)
    assert not [
        span
        for span in reading.by_vocabulary[_CRS].spans
        if any(isinstance(row, MeasureValueRow) for row in span.rows)
        and span.matched_text == "gross"
    ]


# ---------------------------------------------------------------------------
# A part naming the CRS refuses only terms no CRS shape reads and holds.
# ---------------------------------------------------------------------------


_FLOW_ROLES = ("financingProvider", "financingRecipient")


def _flow_family_shape(slots: Literal["crs", "dac2a"], measured_property: str) -> ShapeDraft:
    """A financial flow shape of the CRS flow family or of DAC2a's, by its slot keys, with
    both flow roles and no dataset stamped."""
    flow_row = next(row for row in _REAL_INDEX.statvars if row.population_type == "FinancialFlow")
    slot_keys = (
        crs_flow_slot_keys()
        if slots == "crs"
        else (SlotKeyDraft(axis="what", property_dcid="dac2aAidType", label="aid type"),)
    )
    return ShapeDraft(
        shape_id="amount",
        label="amount",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid=measured_property,
        stat_type_dcid=flow_row.stat_type,
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=slot_keys,
        declared_roles=_FLOW_ROLES,
    )


def _part_state(
    variable: str,
    shapes: list[ShapeDraft],
    recipient: Literal["country", "organisation"] | None = None,
) -> ResolveState:
    """The state of a part reading `variable` on `shapes`. With a `recipient` kind it names
    Kenya as the destination, resolved to an entity of that kind."""
    state = _vet_state(
        variable=variable,
        capped_shapes=shapes,
        chosen_shape_index=0,
        bindings=[],
        admitted_ties=(),
    )
    if recipient is not None:
        entity = Entity(dcid="country/KEN", name="Kenya", kind=recipient, graph_type="Country")
        state.entities = ["Kenya"]
        state.place_sets = place_sets({"Kenya": "country/KEN"})
        state.entity_roles = door_roles_for_test_query("aid to Kenya", ["Kenya"])
        state.rcl = Recall(candidates=[], resolutions={"Kenya": Resolved(entity=entity)})
    return state


@pytest.mark.parametrize(
    ("variable", "refused"),
    [
        ("OOF budget support to Kenya", []),
        ("multilateral ODA to Kenya", ["multilateral ODA"]),
        ("multilateral ODA loans to Kenya", ["multilateral ODA loans"]),
        ("bilateral ODA grants to Kenya", []),
        ("DAC1 aid type 1010", ["1010"]),
    ],
)
def test_a_part_naming_the_crs_refuses_only_terms_its_own_reading_cannot_hold(variable, refused):
    """Read in every code system, each term names a DAC1 aid type no CRS row holds. The CRS
    shape holds "OOF budget support" all the same, since its own reading holds both "OOF"
    (flow type 21) and the "budget support" tie, covering every word. "multilateral ODA"
    and "multilateral ODA loans" each leave a word the CRS reading does not hold, so the
    part refuses `qualifier_not_bound` on them, as a DAC1 code with no CRS reading of its
    words does. "bilateral ODA grants" is held: "bilateral" is a word the CRS states of
    every row, and the CRS reading holds "grants"."""
    flow_row = next(row for row in _REAL_INDEX.statvars if row.population_type == "FinancialFlow")
    shape = ShapeDraft(
        shape_id="crs-amount",
        label="crs-amount",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid=flow_row.stat_type,
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=crs_flow_slot_keys(),
    )
    candidates = _dataset_candidates([shape], part_scope("provenance/OECDCreditorReportingSystem"))

    undeclared = _undeclared_qualifiers(
        variable,
        read_text(variable, _REAL_INDEX, get_measure_index()).everywhere.spans,
        candidates,
        own_variable_spans(variable, *candidates),
        state=_part_state(variable, candidates),
    )

    assert [record.text for record in undeclared] == refused


@pytest.mark.parametrize(
    ("provenance", "slots", "measured_property", "variable", "recipient", "refused"),
    [
        # Every `amount` row of the CRS is a flow from one provider to one recipient, so a
        # part naming it reads "bilateral" without a code, whoever the recipient is.
        (_CRS_PROVENANCE, "crs", "amount", "bilateral ODA grants to Kenya", None, []),
        (_CRS_PROVENANCE, "crs", "amount", "bilateral ODA to Kenya", "organisation", []),
        # DAC2a states "bilateral" only where the destination is a country.
        (_DAC2A_PROVENANCE, "dac2a", "amount", "bilateral ODA to Kenya", "country", []),
        (_DAC2A_PROVENANCE, "dac2a", "amount", "bilateral ODA", None, ["bilateral ODA"]),
        (
            _DAC2A_PROVENANCE,
            "dac2a",
            "amount",
            "bilateral ODA to Kenya",
            "organisation",
            ["bilateral ODA"],
        ),
        # No table states "multilateral", and DAC2a's attributed amount, imputed
        # multilateral aid, states no "bilateral".
        (_CRS_PROVENANCE, "crs", "amount", "multilateral ODA to Kenya", None, ["multilateral ODA"]),
        (
            _DAC2A_PROVENANCE,
            "dac2a",
            "attributedAmount",
            "bilateral ODA to Kenya",
            "country",
            ["bilateral ODA"],
        ),
    ],
)
def test_a_part_naming_a_table_that_states_a_word_does_not_refuse_it(
    provenance, slots, measured_property, variable, recipient, refused
):
    """A word every row of the named table states for the measure restricts nothing there,
    so the part reads it as the table's own reading reads a code. A word the table states
    only where a party is of one kind (`StatedWord.where`) holds when the part names a
    recipient of that kind, and the part refuses it where the recipient is left open or is
    another kind."""
    candidates = _dataset_candidates(
        [_flow_family_shape(slots, measured_property)], part_scope(provenance)
    )

    undeclared = _undeclared_qualifiers(
        variable,
        read_text(variable, _REAL_INDEX, get_measure_index()).everywhere.spans,
        candidates,
        own_variable_spans(variable, *candidates),
        state=_part_state(variable, candidates, recipient),
    )

    assert [record.text for record in undeclared] == refused


@pytest.mark.parametrize(
    ("slots", "recipient", "stated"),
    [
        ("crs", None, {"bilateral"}),
        ("dac2a", "country", {"bilateral"}),
        ("dac2a", None, set()),
    ],
)
def test_a_shape_no_candidacy_stamped_states_what_the_tables_of_its_family_state(
    slots, recipient, stated
):
    """The family election reads each shape before candidacy stamps its datasets, so a
    shape with none reads the tables publishing its family."""
    shape = _flow_family_shape(slots, "amount")
    assert shape.datasets is None

    state = _part_state("bilateral ODA", [shape], recipient)

    assert _dataset_stated_words(shape, _REAL_INDEX, state) == stated


@pytest.mark.parametrize("named_at", ["ref", "words"])
@pytest.mark.parametrize(
    ("variable", "refused"),
    [("multilateral ODA to Kenya", ["multilateral ODA"]), ("ODA to Kenya", [])],
)
def test_a_term_the_elected_crs_shape_reads_in_part_is_unavailable(named_at, variable, refused):
    """A part naming the CRS, by its ref or by words a raw-text query's extraction carries,
    records a term the elected shape's own reading leaves a word of as unbound with
    evidence "unavailable", after the model binds ODA (`CRSTypeOfFlow10`), a code the CRS
    holds, so the part refuses `qualifier_not_bound` with the ODA answer as its near match.
    Otherwise "multilateral" would only warn and the part answer all CRS ODA to Kenya."""
    crs = "provenance/OECDCreditorReportingSystem"
    flow_row = next(row for row in _REAL_INDEX.statvars if row.population_type == "FinancialFlow")
    derived = ShapeDraft(
        shape_id="crs-amount",
        label="crs-amount",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid=flow_row.stat_type,
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=crs_flow_slot_keys(),
    )
    scope = part_scope(crs)
    assert scope is not None
    if named_at == "words":
        scope = dataclasses.replace(scope, mentions=(SourceMention(text="CRS", ref=crs),))
    (shape,) = _dataset_candidates([derived], scope)
    flow = SlotBindingDraft(
        axis="what", property_dcid="crsTypeOfFlow", kind="value", value_dcids=["CRSTypeOfFlow10"]
    )
    state = _vet_state(
        variable=variable,
        capped_shapes=[shape],
        chosen_shape_index=0,
        bindings=[flow],
        admitted_ties=(),
    )
    state.entities = ["Kenya"]
    state.scope = scope
    reading = read_text(
        variable, _REAL_INDEX, get_measure_index(), vocabularies=[shape_vocabulary(shape)]
    )
    state.variable_spans_in_every_system = reading.everywhere.spans
    state.variable_spans_by_vocabulary = {
        vocabulary: own.spans for vocabulary, own in reading.by_vocabulary.items()
    }

    assert _vet_election(state) is None
    assert state.settled_dataset() == crs
    state.bindings = list(state.bind_result.bindings)

    unbound = [q for q in _resolve_query_qualifiers(state) if q.disposition != "bound"]

    assert [(q.text, q.evidence) for q in unbound] == [(text, "unavailable") for text in refused]


# ---------------------------------------------------------------------------
# Post-bind admission: a match admitted against ANY capped_shapes candidate
# does not survive election onto a shape that does not itself declare it.
# ---------------------------------------------------------------------------


def test_match_admitted_against_a_non_elected_candidate_does_not_survive_election():
    pre_bind = _admit("ODA standard grants for health", [_CRS_SHAPE])
    assert {m.code_system for m in pre_bind} == {"crsTypeOfFlow", "crsTypeOfFinance", "crsPurpose"}

    # The elected shape belongs to an entirely different family and
    # declares none of the three -- e.g. bind_universal chose a
    # demographic sibling that also scored well, not the CRS shape that
    # made these matches admissible pre-bind.
    survivors = _admit_qualifier_matches_against_elected_shape(pre_bind, _DEMOGRAPHIC_SHAPE)
    assert survivors == ()


def test_match_admitted_against_the_elected_shape_survives_election():
    pre_bind = _admit("ODA standard grants for health", [_CRS_SHAPE])
    survivors = _admit_qualifier_matches_against_elected_shape(pre_bind, _CRS_SHAPE)
    assert {m.code_system for m in survivors} == {"crsTypeOfFlow", "crsTypeOfFinance", "crsPurpose"}


def test_election_drop_is_per_property_not_all_or_nothing():
    """A shape can declare SOME of the pre-bind admitted properties and not
    others -- election drops only what it does not itself declare."""
    partial_shape = _shape(
        "partial_crs_shape",
        (SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="purpose"),),
    )
    pre_bind = _admit("ODA standard grants for health", [_CRS_SHAPE])
    survivors = _admit_qualifier_matches_against_elected_shape(pre_bind, partial_shape)
    assert {m.code_system for m in survivors} == {"crsPurpose"}


# ---------------------------------------------------------------------------
# Ties are dropped, promoted when one row survives, or passed to the binder.
# ---------------------------------------------------------------------------


def test_admit_qualifier_ties_promotes_when_exactly_one_row_survives_narrowing():
    # "budget support" ties CRSCooperationModalityA (crsCooperationModality)
    # against CRSPurposeCode510 (crsPurpose). A shape declaring only
    # crsPurpose narrows the tie to a single surviving row -- the
    # declared-property tie-break -- promoted straight into
    # admitted_qualifiers, with no tie left for the model to see.
    ties = _crs_ties("budget support")
    assert len(ties) == 1
    purpose_only_shape = _shape(
        "purpose_only_shape",
        (SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="purpose"),),
    )
    promoted, admitted_ties, _unheld = _admit_qualifier_ties(
        [purpose_only_shape], {shape_vocabulary(purpose_only_shape): ties}, holds=held_codes
    )
    assert admitted_ties == ()
    assert len(promoted) == 1
    assert promoted[0].code_system == "crsPurpose"
    assert promoted[0].code_dcid == "CRSPurposeCode510"


def test_admit_qualifier_ties_passes_through_when_two_or_more_rows_survive():
    # _CRS_SHAPE declares both crsCooperationModality and crsPurpose --
    # nothing to promote, the tie is still genuinely live.
    ties = _crs_ties("budget support")
    promoted, admitted_ties, _unheld = _admit_qualifier_ties(
        [_CRS_SHAPE], {shape_vocabulary(_CRS_SHAPE): ties}, holds=held_codes
    )
    assert promoted == ()
    assert len(admitted_ties) == 1
    assert {row.code_system for row in admitted_ties[0].rows} == {
        "crsCooperationModality",
        "crsPurpose",
    }
    assert admitted_ties[0].matched_text == "budget support"


def test_admit_qualifier_ties_drops_when_no_shape_declares_either_property():
    ties = match_qualifier_ties("budget support", _REAL_INDEX)
    promoted, admitted_ties, _unheld = _admit_qualifier_ties(
        [_DEMOGRAPHIC_SHAPE], {shape_vocabulary(_DEMOGRAPHIC_SHAPE): ties}, holds=held_codes
    )
    assert promoted == ()
    assert admitted_ties == ()


def test_admit_qualifier_ties_narrows_a_four_row_tie_per_declaring_shape():
    ties = _crs_ties("debt relief grants")
    debt_relief = next(t for t in ties if t.matched_text == "debt relief")
    assert len(debt_relief.rows) == 4
    modality_only_shape = _shape(
        "modality_only_shape",
        (SlotKeyDraft(axis="how", property_dcid="crsCooperationModality", label="modality"),),
    )
    promoted, _admitted_ties, _unheld = _admit_qualifier_ties(
        [modality_only_shape], {shape_vocabulary(modality_only_shape): ties}, holds=held_codes
    )
    debt_relief_promoted = [m for m in promoted if m.matched_text == "debt relief"]
    assert len(debt_relief_promoted) == 1
    assert debt_relief_promoted[0].code_system == "crsCooperationModality"
    assert debt_relief_promoted[0].code_dcid == "CRSCooperationModalityF01"


def test_admit_qualifier_ties_keeps_only_the_codes_the_shapes_rows_hold():
    """ "ODA" ties CRS's flow type and DAC2a's ODA code in the vocabulary of ONE's imputed
    sectors table, which declares both systems. Its rows hold the flow type alone, so the
    tie promotes to it and bind is never offered DAC2a's ODA code."""
    sectors_sv = (
        "AttributedAmount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_DAC2aAidType106_"
        "CRSPurposeCode11220_Nominal"
    )
    graph = FakeGraph(
        nodes={sectors_sv: _index_row_node(sectors_sv)}, obs={}, detect={}, resolve={}
    )
    (sectors,) = derive_shapes(confirmed_svs=[sectors_sv], graph=graph)
    vocabulary = shape_vocabulary(sectors)
    ties = (
        read_text("ODA to Kenya", _REAL_INDEX, vocabularies=[vocabulary])
        .by_vocabulary[vocabulary]
        .ties
    )
    assert [{row.code_dcid for row in tie.rows} for tie in ties] == [
        {"CRSTypeOfFlow10", "DAC2aAidType206"}
    ]

    promoted, admitted_ties, _unheld = _admit_qualifier_ties(
        [sectors], {vocabulary: ties}, holds=held_codes
    )

    assert admitted_ties == ()
    assert [match.code_dcid for match in promoted] == ["CRSTypeOfFlow10"]


@pytest.mark.parametrize(
    ("query", "entities", "variable", "recalled"),
    [
        pytest.param(
            "deaths attributed to air pollution",
            [],
            "deaths attributed to air pollution",
            (_TWO_ENDED_SUBJECT_SV, 0.9),
            id="health-question",
        ),
        pytest.param(
            "aid attributed to Kenya",
            ["Kenya"],
            "aid attributed",
            (_TWO_ENDED_DAC2A_ODA_SV, 0.9),
            id="aid-question",
        ),
    ],
)
def test_attributed_as_a_plain_verb_names_no_measure(query, entities, variable, recalled):
    """The attributed-amount measure reads on phrases that name it ("attributed to donors",
    "attributed aid"). The verb alone names none, so the answer warns of no dropped
    qualifier."""
    result, _menu = _resolve_two_ended(query, entities, variable=variable, recalled=(recalled,))

    assert not [
        w
        for w in result.warnings
        if w.code == QUERY_QUALIFIER_DROPPED and "attributed" in w.message
    ]


_WOULD_REFUSE = "qualifier_coverage would_refuse "


@pytest.mark.parametrize("mode", ["log", "refuse"])
def test_a_term_naming_codes_no_row_holds_follows_the_coverage_mode(search_settings, caplog, mode):
    """ "Goods donations" names two CRS cooperation modalities, and no index row holds
    either, so bind is never offered them. The part names no dataset, so the record
    follows the coverage mode. In refuse mode the part refuses `qualifier_not_bound` on
    the phrase, with evidence "unavailable", and offers the CRS ODA answer as its near
    match. In log mode the part answers, warns once that the phrase was not applied, and
    logs the would-refuse line once."""
    search_settings(qualifier_coverage=mode)
    with caplog.at_level(logging.INFO, logger="search.engine.region.phase_election"):
        result, menu = _resolve_two_ended(
            "ODA goods donations from France to Kenya",
            ["France", "Kenya"],
            variable="ODA goods donations",
            recalled=((_TWO_ENDED_CRS_SV, 0.9),),
        )

    assert menu == [(_TWO_ENDED_CRS_SV,)]
    would_refuse = [
        (line["phrase"], line["codes"])
        for r in caplog.records
        if (message := r.getMessage()).startswith(_WOULD_REFUSE)
        for line in [json.loads(message.removeprefix(_WOULD_REFUSE))]
    ]
    dropped = [w for w in result.warnings if w.code == QUERY_QUALIFIER_DROPPED]
    if mode == "log":
        assert result.no_data_reason != "qualifier_not_bound"
        assert not result.unbound_qualifiers
        assert len(dropped) == 1 and "goods donations" in dropped[0].message
        assert would_refuse == [
            ("goods donations", ["CRSCooperationModalityJ", "CRSCooperationModalityJ01"])
        ]
        return
    assert result.no_data_reason == "qualifier_not_bound"
    assert [
        (q.text, q.evidence, {ref.ref for ref in q.named}) for q in result.unbound_qualifiers
    ] == [
        (
            "goods donations",
            "unavailable",
            {"CRSCooperationModalityJ", "CRSCooperationModalityJ01"},
        )
    ]
    assert result.flagged_match
    assert would_refuse == []


# ---------------------------------------------------------------------------
# _vet_election verifies the binder's choice for admitted ties.
# ---------------------------------------------------------------------------


def _vet_state(
    *,
    variable: str,
    capped_shapes: list[ShapeDraft],
    chosen_shape_index: int,
    bindings: list[SlotBindingDraft],
    admitted_ties: tuple[QualifierTie, ...],
    variable_matched_qualifier_phrases: frozenset[str] = frozenset(),
    query_qualifier_readings: tuple[QualifierReadings, ...] = (),
) -> ResolveState:
    """Build election state with a fixed score so margin checks stay inactive."""
    return ResolveState(
        variable=variable,
        entities=[],
        date_request=None,
        detect_query=variable,
        role_query=variable,
        pac=True,
        graph=cast(EngineGraphClient, None),
        resolver=cast(EntityResolver, None),
        admission=cast(Admission, None),
        llm=QuantityReadOnlyLLM(),
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
        capped_shapes=capped_shapes,
        query_qualifier_ties={shape_vocabulary(shape): admitted_ties for shape in capped_shapes},
        admitted_ties=admitted_ties,
        variable_matched_qualifier_phrases=variable_matched_qualifier_phrases,
        query_qualifier_readings=query_qualifier_readings,
        bind_result=elected_reply(chosen_shape_index, bindings=bindings, also_plausible=None),
        coverage=empty_coverage_snapshot(),
    )


def _run_vet(state: ResolveState) -> None:
    result = _vet_election(state)
    assert result is None, "expected _vet_election to fall through"
    depart_requirements(state)


_BUDGET_SUPPORT_TIE = _crs_ties("budget support")[0]


def test_vet_election_keeps_a_single_model_elected_tie_row():
    state = _vet_state(
        variable="budget support to Kenya",
        capped_shapes=[_CRS_SHAPE],
        chosen_shape_index=0,
        bindings=[
            SlotBindingDraft(
                axis="how",
                property_dcid="crsPurpose",
                kind="value",
                value_dcids=["CRSPurposeCode510"],
            )
        ],
        admitted_ties=(_BUDGET_SUPPORT_TIE,),
        variable_matched_qualifier_phrases=frozenset({"budget support"}),
    )
    _run_vet(state)
    assert state.elected_tie_phrases == {"budget support"}
    assert state.warnings == []
    # Provenance stays "binder": the model's own draft is untouched.
    by_property = {b.property_dcid: b for b in state.bind_result.bindings}
    assert by_property["crsPurpose"].kind == "value"
    assert by_property["crsPurpose"].value_dcids == ["CRSPurposeCode510"]


def test_vet_election_drops_a_tie_the_model_never_bound():
    state = _vet_state(
        variable="budget support to Kenya",
        capped_shapes=[_CRS_SHAPE],
        chosen_shape_index=0,
        bindings=[],
        admitted_ties=(_BUDGET_SUPPORT_TIE,),
        variable_matched_qualifier_phrases=frozenset({"budget support"}),
    )
    _run_vet(state)
    assert state.elected_tie_phrases == frozenset()
    assert len(state.warnings) == 1
    assert state.warnings[0].code == QUERY_QUALIFIER_DROPPED
    assert "budget support" in state.warnings[0].message
    # The tie spans two code systems, so no one slot edit clears it.
    assert state.departure_facts == {
        warning_key(state.warnings[0]): DepartureFact(
            code="measure_not_applied", binding_ref=None, asked="budget support"
        )
    }


def test_vet_election_zero_count_stays_silent_when_the_tie_is_owned_elsewhere():
    # variable_matched_qualifier_phrases empty here mirrors a tie found only
    # in a dropped_qualifier_phrase -- entry/raw_text.py already warns on
    # that phrase unconditionally, so _vet_election must not warn a
    # second time.
    state = _vet_state(
        variable="ODA to Kenya",
        capped_shapes=[_CRS_SHAPE],
        chosen_shape_index=0,
        bindings=[],
        admitted_ties=(_BUDGET_SUPPORT_TIE,),
        variable_matched_qualifier_phrases=frozenset(),
    )
    _run_vet(state)
    assert state.elected_tie_phrases == frozenset()
    assert state.warnings == []


def test_vet_election_resets_an_over_narrow_conjunction_to_unbound():
    # The model bound BOTH readings of a query that named only one
    # alternative -- narrower than the query asked for.
    bindings = [
        SlotBindingDraft(
            axis="how",
            property_dcid="crsCooperationModality",
            kind="value",
            value_dcids=["CRSCooperationModalityA"],
        ),
        SlotBindingDraft(
            axis="how",
            property_dcid="crsPurpose",
            kind="value",
            value_dcids=["CRSPurposeCode510"],
        ),
    ]
    state = _vet_state(
        variable="budget support to Kenya",
        capped_shapes=[_CRS_SHAPE],
        chosen_shape_index=0,
        bindings=bindings,
        admitted_ties=(_BUDGET_SUPPORT_TIE,),
        variable_matched_qualifier_phrases=frozenset({"budget support"}),
    )
    _run_vet(state)
    assert state.elected_tie_phrases == frozenset()
    assert len(state.warnings) == 1
    assert state.warnings[0].code == QUERY_QUALIFIER_DROPPED
    by_property = {b.property_dcid: b for b in state.bind_result.bindings}
    assert by_property["crsCooperationModality"].kind == "unbound"
    assert by_property["crsCooperationModality"].value_dcids == []
    assert by_property["crsPurpose"].kind == "unbound"
    assert by_property["crsPurpose"].value_dcids == []


# IMF WEO's GDP PPP series: measurementQualifier `Nominal`, with the PPP reading on a
# constraint that no index carries.
_IMF_GDP_PPP_SHAPE = dataclasses.replace(
    _shape(
        "imf_gdp_ppp",
        (
            SlotKeyDraft(
                axis="how", property_dcid="currencyConversionBasis", label="conversion basis"
            ),
        ),
    ),
    pop_type_dcid="EconomicProduction",
    meas_prop_dcid="valueAdded",
    value_labels={"PurchasingPowerParityRate": "purchasing power parity rate"},
)


@pytest.mark.parametrize(
    ("kind", "value_dcids", "dropped"),
    [
        pytest.param("value", ["PurchasingPowerParityRate"], False, id="bound-value-reads-ppp"),
        pytest.param("unbound", [], True, id="unbound-drops-ppp"),
    ],
)
def test_vet_election_counts_ppp_read_by_a_bound_constraint_value(kind, value_dcids, dropped):
    readings, owned = match_query_readings("GDP PPP")
    state = _vet_state(
        variable="GDP PPP",
        capped_shapes=[_IMF_GDP_PPP_SHAPE],
        chosen_shape_index=0,
        bindings=[
            SlotBindingDraft(
                axis="how",
                property_dcid="currencyConversionBasis",
                kind=kind,
                value_dcids=value_dcids,
            )
        ],
        admitted_ties=(),
        variable_matched_qualifier_phrases=owned,
        query_qualifier_readings=readings,
    )
    _run_vet(state)
    assert [w.code for w in state.warnings] == ([QUERY_QUALIFIER_DROPPED] if dropped else [])
    assert ("ppp" in state.satisfied_measure_phrases) is not dropped


# ---------------------------------------------------------------------------
# Forcing: an admitted match becomes a kind="value" SlotBindingDraft, fixed
# to its own graph-sourced code_dcid, overriding whatever the bind call's
# own LLM output drafted for the same axis.
# ---------------------------------------------------------------------------


def test_forced_binding_overrides_an_existing_llm_drafted_binding():
    existing = [
        SlotBindingDraft(
            axis="what", property_dcid="crsTypeOfFinance", kind="unbound", value_dcids=[]
        ),
        SlotBindingDraft(axis="how", property_dcid="crsPurpose", kind="unbound", value_dcids=[]),
    ]
    matches = _admit("ODA standard grants for health", [_CRS_SHAPE])
    forced, properties = _force_qualifier_bindings(
        existing, matches, _CRS_SHAPE, variable="ODA standard grants for health"
    )
    by_property = {b.property_dcid: b for b in forced}
    assert properties == {m.code_system for m in matches}
    assert by_property["crsTypeOfFinance"].kind == "value"
    assert by_property["crsTypeOfFinance"].value_dcids == ["CRSTypeOfFinance110"]
    assert by_property["crsPurpose"].kind == "value"
    assert by_property["crsPurpose"].value_dcids == ["CRSPurposeCode120"]
    assert by_property["crsTypeOfFlow"].value_dcids == ["CRSTypeOfFlow10"]
    # No duplicate entries for a property that was both drafted and forced.
    assert len(forced) == len(by_property)


def test_forced_binding_added_when_the_llm_drafted_no_binding_at_all():
    matches = _admit("ODA grants for malaria control", [_CRS_SHAPE])
    forced, _ = _force_qualifier_bindings(
        [], matches, _CRS_SHAPE, variable="ODA grants for malaria control"
    )
    by_property = {b.property_dcid: b for b in forced}
    assert by_property["crsPurpose"].value_dcids == ["CRSPurposeCode12262"]


def test_private_development_finance_forces_its_flow_value():
    matches = _admit("private development finance to Kenya", [_CRS_SHAPE])
    forced, _ = _force_qualifier_bindings(
        [], matches, _CRS_SHAPE, variable="private development finance to Kenya"
    )
    by_property = {binding.property_dcid: binding for binding in forced}

    assert by_property["crsTypeOfFlow"].kind == "value"
    assert by_property["crsTypeOfFlow"].value_dcids == ["CRSTypeOfFlow30"]


@pytest.mark.parametrize(
    "variable",
    ["refugee costs in donor countries ODA", "ODA refugee costs in donor countries"],
)
def test_two_nested_matches_on_one_property_force_the_descendant(variable: str):
    """ "ODA" and "refugee costs in donor countries" both name DAC1 aid types, and in-donor
    refugee costs (1820) lies under ODA (1010). The narrower code binds in either word
    order."""
    dac1_sv = "Amount_FinancialFlow_Gross_Disbursement_DAC1AidType1010_Nominal"
    graph = FakeGraph(nodes={dac1_sv: _index_row_node(dac1_sv)}, obs={}, detect={}, resolve={})
    (dac1,) = derive_shapes(confirmed_svs=[dac1_sv], graph=graph)
    matches = _admit(variable, [dac1])
    assert {m.code_dcid for m in matches if m.code_system == "dac1AidType"} == {
        "DAC1AidType1010",
        "DAC1AidType1820",
    }

    forced, _ = _force_qualifier_bindings([], matches, dac1, variable=variable)

    assert [b.value_dcids for b in forced if b.property_dcid == "dac1AidType"] == [
        ["DAC1AidType1820"]
    ]


def _crs_amount_shape() -> ShapeDraft:
    """The shape the fixture index derives for nominal CRS financial-flow amounts."""
    flow_row = next(row for row in _REAL_INDEX.statvars if row.population_type == "FinancialFlow")
    return ShapeDraft(
        shape_id="crs-amount",
        label="crs-amount",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid=flow_row.stat_type,
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=crs_flow_slot_keys(),
    )


_HIV_AND_130 = "HIV/AIDS and population policy, programme, and reproductive health"
_UNION = ("union", ["CRSPurposeCode13040", "CRSPurposeCode130"])
_REFINEMENT = ("refinement", ["CRSPurposeCode12262", "CRSPurposeCode120"])
_UNSUPPORTED = ("no_vocabulary", "unsupported", None)
_INCLUDING = ("union", ["CRSPurposeCode120", "CRSPurposeCode12262"])
_AFTER_EXCLUSION = ("refinement", ["CRSPurposeCode120", "CRSPurposeCode12250"])


@pytest.mark.parametrize("mode", ["log", "refuse"])
@pytest.mark.parametrize(
    ("variable", "purpose", "records", "gated"),
    [
        (
            _HIV_AND_130,
            ["CRSPurposeCode130"],
            {
                "HIV": ("bound", None, _UNION),
                "AIDS": ("bound", None, _UNION),
                "population policy, programme, and reproductive health": (
                    "bound",
                    None,
                    _UNION,
                ),
            },
            {},
        ),
        (
            "health and education ODA",
            ["CRSPurposeCode110", "CRSPurposeCode120", "CRSPurposeCode130"],
            {
                "health": ("bound", None, ("union", ["CRSPurposeCode120", "CRSPurposeCode110"])),
                "education": (
                    "bound",
                    None,
                    ("union", ["CRSPurposeCode120", "CRSPurposeCode110"]),
                ),
            },
            {},
        ),
        (
            "malaria control within health ODA",
            ["CRSPurposeCode12262"],
            {
                "malaria control": ("bound", None, _REFINEMENT),
                "health": ("bound", None, _REFINEMENT),
            },
            {},
        ),
        (
            "health excluding malaria control ODA",
            ["CRSPurposeCode120", "CRSPurposeCode130"],
            {
                "health": ("bound", None, None),
                "malaria control": ("unbound", None, None),
            },
            {"excluding malaria control": _UNSUPPORTED},
        ),
        (
            "health versus malaria control ODA",
            ["CRSPurposeCode12262"],
            {
                "health": (
                    "bound",
                    None,
                    ("refinement", ["CRSPurposeCode120", "CRSPurposeCode12262"]),
                ),
                "malaria control": (
                    "bound",
                    None,
                    ("refinement", ["CRSPurposeCode120", "CRSPurposeCode12262"]),
                ),
            },
            {"versus malaria control": _UNSUPPORTED},
        ),
        (
            "compare aid to health and education",
            None,
            {
                "health": ("unbound", None, None),
                "education": ("unbound", None, None),
                "compare aid to health": _UNSUPPORTED,
            },
            {},
        ),
        (
            "health versus education ODA",
            None,
            {
                "health": ("unbound", None, None),
                "education": ("unbound", None, None),
                "versus education": _UNSUPPORTED,
            },
            {},
        ),
        (
            "health & education ODA",
            ["CRSPurposeCode110", "CRSPurposeCode120", "CRSPurposeCode130"],
            {
                "health": ("bound", None, ("union", ["CRSPurposeCode120", "CRSPurposeCode110"])),
                "education": (
                    "bound",
                    None,
                    ("union", ["CRSPurposeCode120", "CRSPurposeCode110"]),
                ),
            },
            {},
        ),
        (
            "health including malaria control ODA",
            ["CRSPurposeCode120", "CRSPurposeCode130"],
            {
                "health": ("bound", None, _INCLUDING),
                "malaria control": ("bound", None, _INCLUDING),
            },
            {},
        ),
        (
            "ODA to Kenya excluding health",
            None,
            {"health": ("unbound", None, None)},
            {"excluding health": _UNSUPPORTED},
        ),
        (
            "health excluding malaria control versus infectious disease control ODA",
            ["CRSPurposeCode12250"],
            {
                "health": ("bound", None, _AFTER_EXCLUSION),
                "malaria control": ("unbound", None, None),
                "infectious disease control": ("bound", None, _AFTER_EXCLUSION),
            },
            {
                "excluding malaria control": _UNSUPPORTED,
                "versus infectious disease control": _UNSUPPORTED,
            },
        ),
    ],
    ids=[
        "union",
        "set",
        "refinement",
        "exclusion",
        "nested-comparison",
        "compare",
        "versus",
        "ampersand",
        "including",
        "excluding-the-only-code",
        "comparison-after-an-exclusion",
    ],
)
@pytest.mark.parametrize("query_text", [False, True], ids=["variable", "with-query-text"])
def test_two_matches_on_one_code_system_bind_by_the_connective_between_them(
    search_settings, mode, query_text, variable, purpose, records, gated
):
    """Admitted matches on one code system bind by the connective between them. A union
    connective keeps the ancestor of two codes that nest, so a "HIV" or "AIDS" record is
    bound through the union once 130 binds, and binds two codes that do not nest as a set.
    Any other joiner refines to the narrower code. An exclusion binds the code before it
    and leaves the excluded code's record unbound. An exclusion or comparison phrase is
    unsupported. A comparison of two codes that do not nest refuses in either coverage
    mode, since a union binding would sum the two. Every other unsupported phrase refuses
    only under `refuse`, and under `log` the would-refuse line names the phrase
    (`_coverage_records`). Codes a comparison sets side by side bind no union, and the
    model's empty draft leaves them unbound. An exclusion of a system's only code forces
    nothing on it. A comparison after an exclusion compares the code the exclusion kept.
    Repeating the variable's codes in the query text changes nothing, because the copies
    do not combine. "health" names CRS 120, and wherever 120 binds as its code it binds as
    the declared health grouping, 120 and 130 (`scope.sector.read_sector`)."""
    search_settings(qualifier_coverage=mode)
    shape = _crs_amount_shape()
    vocabulary = shape_vocabulary(shape)
    reading = read_text(variable, _REAL_INDEX, get_measure_index(), vocabularies=[vocabulary])
    query = f"{variable} to Uganda, 2010 to 2024" if query_text else None
    state = _vet_state(
        variable=variable,
        capped_shapes=[shape],
        chosen_shape_index=0,
        bindings=[],
        admitted_ties=(),
    )
    state.variable_spans_in_every_system = reading.everywhere.spans
    state.variable_spans_by_vocabulary = {
        vocabulary: own.spans for vocabulary, own in reading.by_vocabulary.items()
    }
    state.entities = ["Kenya"]
    state.query_text = query
    state.query_qualifier_matches = query_qualifiers(
        variable, query_text=query, vocabularies=[vocabulary]
    ).matches

    _run_vet(state)
    state.bindings = list(state.bind_result.bindings)
    qualifiers = _resolve_query_qualifiers(state)

    bound = {b.property_dcid: b for b in state.bindings if b.kind != "unbound"}
    if purpose is None:
        assert "crsPurpose" not in bound
    else:
        assert bound["crsPurpose"].kind == ("value" if len(purpose) == 1 else "set")
        assert bound["crsPurpose"].value_dcids == purpose
    expected = {**records, **(gated if mode == "refuse" else {})}

    def summary(q):
        if q.evidence == "unsupported":
            return (q.disposition, q.evidence, q.operation)
        operation = (q.operation, [ref.ref for ref in q.selected]) if q.operation else None
        return (q.disposition, q.evidence, operation)

    assert {q.text: summary(q) for q in qualifiers if q.text != "ODA"} == expected
    if mode == "log":
        assert [q.text for q in _coverage_records(state, qualifiers)] == list(gated)


@pytest.mark.parametrize(
    ("variable", "phrases"),
    [
        ("health excluding malaria control ODA", ["excluding malaria control"]),
        ("ODA without the health", ["without the health"]),
        ("compare aid to health and education", ["compare aid to health"]),
        ("Kenya versus Uganda health ODA", []),
        ("ODA compared to 2010", []),
    ],
)
def test_an_exclusion_or_comparison_connective_governs_the_match_after_it(variable, phrases):
    """A connective governs the first match after it across function words alone. A
    comparing verb governs it at any distance."""
    matches = read_text(variable, _REAL_INDEX, vocabularies=[_CRS]).by_vocabulary[_CRS].matches
    assert [variable[c.start : c.end] for c in governing_connectives(variable, matches)] == phrases


def test_no_admitted_matches_leaves_bindings_untouched():
    existing = [
        SlotBindingDraft(axis="how", property_dcid="crsPurpose", kind="unbound", value_dcids=[])
    ]
    forced, properties = _force_qualifier_bindings(existing, (), _CRS_SHAPE, variable="health")
    assert forced == existing
    assert properties == frozenset()


# ---------------------------------------------------------------------------
# Measure values use the same admission path when a shape declares their axis.
# ---------------------------------------------------------------------------


def test_net_oda_forces_netting_basis_through_admission_and_forcing():
    matches = _admit("net ODA to Kenya", [_CRS_SHAPE_WITH_MEASURE])
    by_system = _by_system(matches)
    assert by_system["nettingBasis"].code_dcid == "Net"
    assert by_system["crsTypeOfFlow"].code_dcid == "CRSTypeOfFlow10"

    forced, _ = _force_qualifier_bindings(
        [], matches, _CRS_SHAPE_WITH_MEASURE, variable="net ODA to Kenya"
    )
    by_property = {b.property_dcid: b for b in forced}
    assert by_property["nettingBasis"].kind == "value"
    assert by_property["nettingBasis"].value_dcids == ["Net"]


def test_netting_basis_match_is_not_admitted_without_a_declaring_shape():
    # _CRS_SHAPE (unlike _CRS_SHAPE_WITH_MEASURE) declares none of the
    # measure family -- the raw text still names "net", but nothing admits
    # it, same as any other undeclared code_system.
    by_system = _by_system(_admit("net ODA to Kenya", [_CRS_SHAPE]))
    assert "nettingBasis" not in by_system


# ---------------------------------------------------------------------------
# dropped_qualifier_phrases: text the extractor set aside from the variable,
# matched on its own alongside it.
# ---------------------------------------------------------------------------


def test_dropped_phrase_naming_a_measure_is_admitted_and_forced():
    # "net" names nothing in "ODA to Kenya" itself -- only a dropped phrase
    # carrying it lets this admit, unlike test_net_oda_forces_netting_basis_
    # through_admission_and_forcing above, where "net" is in the variable.
    assert "nettingBasis" not in _by_system(_admit("ODA to Kenya", [_CRS_SHAPE_WITH_MEASURE]))
    matches = _admit("ODA to Kenya", [_CRS_SHAPE_WITH_MEASURE], ["net"])
    by_system = _by_system(matches)
    assert by_system["nettingBasis"].code_dcid == "Net"

    forced, properties = _force_qualifier_bindings(
        [], matches, _CRS_SHAPE_WITH_MEASURE, variable="ODA to Kenya"
    )
    by_property = {b.property_dcid: b for b in forced}
    assert by_property["nettingBasis"].kind == "value"
    assert by_property["nettingBasis"].value_dcids == ["Net"]
    assert properties == {"crsTypeOfFlow", "nettingBasis"}

    # A rival dropped phrase leaves the two unresolved, so the text forces no basis.
    rivals = _admit("ODA to Kenya", [_CRS_SHAPE_WITH_MEASURE], ["net", "gross"])
    assert {m.code_dcid for m in rivals if m.code_system == "nettingBasis"} == {"Net", "Gross"}
    forced, properties = _force_qualifier_bindings(
        [], rivals, _CRS_SHAPE_WITH_MEASURE, variable="ODA to Kenya"
    )
    assert properties == {"crsTypeOfFlow"}
    assert "nettingBasis" not in {b.property_dcid for b in forced}


def test_dropped_phrase_naming_a_code_system_value_is_admitted_the_same_way():
    # Proves the plumbing is not measure-specific: an ordinary CRS code-row
    # match works identically whether "for health" sits in the variable
    # (test_plain_health_query_admits_the_purpose_qualifier) or is carried
    # in separately as a dropped phrase.
    in_variable = _by_system(_admit("ODA grants for health", [_CRS_SHAPE]))
    dropped = _by_system(_admit("ODA grants", [_CRS_SHAPE], ["for health"]))
    assert dropped["crsPurpose"].code_dcid == in_variable["crsPurpose"].code_dcid
    assert dropped["crsTypeOfFlow"].code_dcid == in_variable["crsTypeOfFlow"].code_dcid


def test_dropped_phrase_matching_nothing_leaves_admission_unchanged():
    baseline = _admit("ODA standard grants for health", [_CRS_SHAPE])
    with_unmatched_phrase = _admit(
        "ODA standard grants for health",
        [_CRS_SHAPE],
        ["a phrase naming nothing in the index"],
    )
    assert with_unmatched_phrase == baseline


def test_measure_word_already_in_the_variable_is_untouched_by_a_dropped_phrase():
    # A query whose measure word is already part of the variable text
    # behaves exactly as it does with no dropped phrase at all -- an
    # unrelated, unmatched dropped phrase changes nothing.
    baseline = _admit("net ODA to Kenya", [_CRS_SHAPE_WITH_MEASURE])
    with_unmatched_phrase = _admit(
        "net ODA to Kenya",
        [_CRS_SHAPE_WITH_MEASURE],
        ["a phrase naming nothing in the index"],
    )
    assert with_unmatched_phrase == baseline


# ---------------------------------------------------------------------------
# Match the original query as well as the extracted variable phrase.
# ---------------------------------------------------------------------------


def test_query_text_admits_a_purpose_code_the_paraphrase_dropped():
    # "agricultural funding" (the paraphrase) matches nothing by itself --
    # this module allows only an exact or plural match, and "agricultural"
    # is neither of "agriculture". query_text carries the user's own word.
    matches = query_qualifiers(
        "agricultural funding",
        (),
        query_text="funding for agriculture in Ethiopia",
        vocabularies=[_CRS],
    ).matches
    by_system = _by_system(matches[_CRS])
    assert by_system["crsPurpose"].code_dcid == "CRSPurposeCode311"


def test_query_text_match_is_admitted_and_forced_the_same_way_a_variable_match_is():
    matches = _admit(
        "agricultural funding",
        [_CRS_SHAPE],
        query_text="funding for agriculture in Ethiopia",
    )
    by_system = _by_system(matches)
    assert by_system["crsPurpose"].code_dcid == "CRSPurposeCode311"

    forced, _ = _force_qualifier_bindings([], matches, _CRS_SHAPE, variable="agricultural funding")
    by_property = {b.property_dcid: b for b in forced}
    assert by_property["crsPurpose"].kind == "value"
    assert by_property["crsPurpose"].value_dcids == ["CRSPurposeCode311"]


def test_query_text_match_is_additive_to_a_variable_match_not_a_replacement():
    # "agricultural funding" alone admits nothing; query_text alone (a
    # different phrasing that also names "ODA") admits crsTypeOfFlow too --
    # both survive together, from two different sources.
    matches = _admit(
        "agricultural funding",
        [_CRS_SHAPE],
        query_text="ODA funding for agriculture in Ethiopia",
    )
    by_system = _by_system(matches)
    assert set(by_system) == {"crsPurpose", "crsTypeOfFlow"}


def test_query_text_none_matches_nothing_the_same_as_before_this_fix():
    # The default (no caller passes query_text) is byte-identical to every
    # existing test above that never mentions it.
    baseline = _admit("ODA standard grants for health", [_CRS_SHAPE])
    explicit_none = _admit("ODA standard grants for health", [_CRS_SHAPE], query_text=None)
    assert explicit_none == baseline


def test_query_text_ownership_set_credits_a_query_text_only_match():
    # The second return value -- the set QUERY_QUALIFIER_DROPPED restricts
    # itself to -- must carry a query_text-only match too: nothing else ever
    # warns on query_text's own phrases the way entry/raw_text.py already
    # warns on dropped_qualifier_phrases.
    owned = query_qualifiers(
        "agricultural funding",
        (),
        query_text="funding for agriculture in Ethiopia",
        vocabularies=[_CRS],
    ).phrases
    assert "agriculture" in owned


def test_n2_fanout_broadcasts_query_text_none_to_every_leg():
    """Multi-variable fan-out passes the default `None` to each leg."""
    seen_query_text: list[object] = []

    async def _stub_resolve_variable(variable: str, **kwargs: object) -> RegionResult:
        seen_query_text.append(kwargs["query_text"])
        return RegionResult(
            variable_text=variable,
            status="no_data",
            specs=(),
            no_data_reason="variable_not_resolved",
            warnings=(),
            timing_by_step={},
        )

    graph = FakeGraph()
    resolver = make_resolver(graph)
    admission = Admission(graph=graph)
    sdmx = FakeSdmxClient(observations={}, availability={})

    async def _scenario() -> None:
        out = _FanOutResult()
        with patch("search.engine.fanout.resolve_variable", new=_stub_resolve_variable):
            async for _ev in _run_variables_streaming(
                ["v0", "v1"],
                entities=[],
                date_request=None,
                query="a raw query naming two variables",
                pac=False,
                graph=graph,
                resolver=resolver,
                admission=admission,
                llm=FakeLLM(responses={}),
                binder=GeminiBinder(),
                sdmx=sdmx,
                base_timing={},
                pre_resolved=None,
                deadline=time.monotonic() + 3600,
                degrade_on_error=True,
                use_semaphore=False,
                out=out,
                coverage=empty_coverage_snapshot(),
            ):
                pass

    asyncio.run(_scenario())
    assert seen_query_text == [None, None]


# ---------------------------------------------------------------------------
# The bind prompt: byte-identical to the no-admission prompt when nothing is
# admitted, and renders only graph-sourced fields (never matched_text) when
# something is.
# ---------------------------------------------------------------------------

_PROMPT_SHAPE = ShapeDraft(
    shape_id="prompt_test_shape",
    label="test shape",
    pop_type_dcid="FinancialFlow",
    meas_prop_dcid="grantEquivalent",
    stat_type_dcid="measuredValue",
    meas_qual_dcid="Nominal",
    meas_denom_dcid=None,
    slot_keys=(),
)


def test_prompt_is_byte_identical_to_the_no_admission_form_when_nothing_is_admitted():
    with_default = _build_universal_bind_prompt("test variable", [_PROMPT_SHAPE])
    with_explicit_empty = _build_universal_bind_prompt(
        "test variable", [_PROMPT_SHAPE], admitted_qualifiers=(), admitted_ties=()
    )
    with_no_menu = _build_universal_bind_prompt(
        "test variable", [_PROMPT_SHAPE], candidate_menus=[()]
    )
    assert with_default == with_explicit_empty == with_no_menu
    assert "FIXED CONTEXT" not in with_default
    assert "QUERY-NAMED ALTERNATIVES" not in with_default
    # This shape carries no score of its own, so no retrieval_score line is
    # rendered at all.
    assert with_default == (
        "Variable phrase: test variable\n\n"
        "CANDIDATE SHAPES (1 total):\n\n"
        "Shape 0:\n"
        "  label: test shape\n"
        "  origin: our Data Commons\n"
        "  family: standard Data Commons\n"
        "  five_tuple: populationType=FinancialFlow, measuredProperty=grantEquivalent, "
        "statType=measuredValue\n"
        "    measurementQualifier=Nominal, measurementDenominator=(none)\n"
        "  constraint_slots: (none)\n"
    )


def test_prompt_renders_admitted_ties_as_query_named_alternatives_without_matched_text():
    # "grants" (plural) is registered on both CRSTypeOfFinance100 ("grant")
    # and CRSTypeOfFinance110 ("standard grant") -- matched_text differs
    # from both labels by the plural allowance's trailing "s", so its
    # absence from the rendered block below cannot be explained by
    # coincidental overlap with a safe field the way "budget support"
    # (a tie whose matched_text equals both rows' own labels) could.
    tie = next(t for t in _crs_ties("debt relief grants") if t.matched_text == "grants")
    prompt = _build_universal_bind_prompt(
        "debt relief grants to Kenya", [_CRS_SHAPE], admitted_ties=[(tie,)]
    )
    assert "QUERY-NAMED ALTERNATIVES" in prompt
    assert "property_dcid=crsTypeOfFinance, value_dcid=CRSTypeOfFinance100 (grant)" in prompt
    assert (
        "property_dcid=crsTypeOfFinance, value_dcid=CRSTypeOfFinance110 (standard grant)" in prompt
    )
    alternatives_block = prompt.split("  query_named_alternatives:")[1]
    assert "grants" not in alternatives_block


def test_prompt_drops_a_tie_left_with_fewer_than_two_rows_after_withholding():
    tie = QualifierTie(
        matched_text="net budget support",
        rows=(
            QualifierMatch(
                code_system="crsPurpose",
                code_dcid="CRSPurposeCode510",
                label="budget support",
                source="label",
                matched_text="net budget support",
            ),
            QualifierMatch(
                code_system="nettingBasis",
                code_dcid="Net",
                label="net",
                source="synonym",
                matched_text="net budget support",
            ),
        ),
    )
    # The shape lists both slots, so withholding the netting basis is what leaves one row.
    shown = _build_universal_bind_prompt(
        "net budget support to Kenya", [_CRS_SHAPE_WITH_MEASURE], admitted_ties=[(tie,)]
    )
    assert "QUERY-NAMED ALTERNATIVES" in shown
    prompt = _build_universal_bind_prompt(
        "net budget support to Kenya",
        [_CRS_SHAPE_WITH_MEASURE],
        admitted_ties=[(tie,)],
        withheld_measure_axes=frozenset({"nettingBasis"}),
    )
    assert "QUERY-NAMED ALTERNATIVES" not in prompt


def test_prompt_renders_admitted_matches_as_fixed_context_without_matched_text():
    matches = _admit("ODA standard grants for health", [_CRS_SHAPE])
    prompt = _build_universal_bind_prompt(
        "ODA standard grants for health", [_CRS_SHAPE], admitted_qualifiers=[matches]
    )
    assert "FIXED CONTEXT" in prompt
    assert (
        "property_dcid=crsTypeOfFinance, value_dcid=CRSTypeOfFinance110 (standard grant)" in prompt
    )
    assert "property_dcid=crsPurpose, value_dcid=CRSPurposeCode120" in prompt
    # matched_text is raw user text and must never reach the FIXED CONTEXT
    # section -- only code_dcid and label do. The finance match's own
    # matched_text ("standard grants", the query's plural span) differs from
    # its label ("standard grant", singular) by exactly the trailing "s"
    # the plural allowance strips (qualifier_match.py's own module
    # docstring), so its absence here cannot be explained by coincidental
    # overlap with a safe field the way a same-spelled label could.
    finance_match = next(m for m in matches if m.code_system == "crsTypeOfFinance")
    assert finance_match.matched_text == "standard grants"
    fixed_context = prompt.split("  fixed_context:")[1]
    assert finance_match.matched_text not in fixed_context


def test_prompt_lists_a_candidate_menu_under_its_phrase_in_place_of_its_terms():
    """A candidate menu lists every code it holds under its phrase, past the 20-value
    taxonomy cap. The term inside the phrase leaves the fixed context on the code system
    the menu names, and a term outside it stays. The menu's code system gets no labels
    line."""
    variable = "global health ODA"
    matches = _admit(variable, [_CRS_SHAPE])
    health = next(m for m in matches if m.code_system == "crsPurpose")
    rows = tuple(
        row for row in _REAL_INDEX.codes if row.code_system == "crsPurpose" and row.parent
    )[:24]
    menu = CandidateMenu(
        text="global health",
        start=0,
        end=health.end or 0,
        candidates=rows,
        hits=(rows[0].dcid,),
    )
    shape = dataclasses.replace(
        _CRS_SHAPE,
        slot_taxonomy={
            "how:crsPurpose": ["CRSPurposeCode12110", "CRSPurposeCode13010"],
            "what:crsTypeOfFinance": ["CRSTypeOfFinance110"],
        },
        slot_taxonomy_index_sourced=True,
    )
    prompt = _build_universal_bind_prompt(
        variable, [shape], admitted_qualifiers=[matches], candidate_menus=[(menu,)]
    )
    assert "CANDIDATE MENUS" in prompt
    listed = prompt.split('    phrase "global health":\n')[1].split("\n\n")[0].splitlines()
    assert [line.strip() for line in listed] == [
        f"property_dcid=crsPurpose, value_dcid={row.dcid} ({row.label})" for row in rows
    ]
    fixed = prompt.split("  fixed_context:\n")[1].split("  candidate_menus:")[0]
    assert "CRSTypeOfFlow10" in fixed
    assert "crsPurpose" not in fixed
    finance_slot = prompt.split("property_dcid=crsTypeOfFinance (")[1].split("    axis=")[0]
    purpose_slot = prompt.split("property_dcid=crsPurpose (")[1].split("    axis=")[0]
    assert "labels:" in finance_slot
    assert "labels:" not in purpose_slot


def test_prompt_shows_each_shape_only_its_own_reading_of_a_word():
    """ "multilateral ODA" reads the CRS flow type on a CRS shape and the DAC1 aid type 2000
    on a DAC1 shape. Each shape's block lists its own fixed value and not the other's."""
    dac1_shape = _shape(
        "dac1_shape", (SlotKeyDraft(axis="what", property_dcid="dac1AidType", label="aid type"),)
    )
    flow_shape = _shape(
        "flow_shape",
        (*_CRS_SLOT_KEYS, SlotKeyDraft(axis="what", property_dcid="dac1AidType", label="aid type")),
    )
    shapes = [flow_shape, dac1_shape]
    matches = query_qualifiers(
        "multilateral ODA",
        vocabularies=[shape_vocabulary(_CRS_SHAPE), shape_vocabulary(dac1_shape)],
    ).matches
    own = [matches[shape_vocabulary(_CRS_SHAPE)], matches[shape_vocabulary(dac1_shape)]]

    prompt = _build_universal_bind_prompt("multilateral ODA", shapes, admitted_qualifiers=own)

    first, second = prompt.split("Shape 1:")
    assert "value_dcid=CRSTypeOfFlow10" in first
    assert "DAC1AidType2000" not in first
    assert "value_dcid=DAC1AidType2000" in second
    assert "CRSTypeOfFlow10" not in second


def test_prompt_renders_an_admitted_measure_match_as_fixed_context_too():
    query = "ODA commitments to Kenya"
    matches = _admit(query, [_CRS_SHAPE_WITH_MEASURE])
    # matched_text ("commitments", the query's own plural span) differs from
    # the rendered label ("commitment", singular) by exactly the trailing
    # "s" the plural allowance strips -- so its absence from FIXED CONTEXT
    # below cannot be explained by coincidental overlap with a safe field.
    obligation_match = next(m for m in matches if m.code_system == "financialObligationType")
    assert obligation_match.matched_text == "commitments"

    prompt = _build_universal_bind_prompt(
        query, [_CRS_SHAPE_WITH_MEASURE], admitted_qualifiers=[matches]
    )
    assert "FIXED CONTEXT" in prompt
    assert "property_dcid=financialObligationType, value_dcid=Commitment (commitment)" in prompt
    fixed_context = prompt.split("  fixed_context:")[1]
    assert obligation_match.matched_text not in fixed_context

    # financialObligationType is one of this deployment's kind="constraint"
    # measure axes (region/phase_rank_bind.py's _phase_bind withholds it). A withheld
    # axis is excluded from an index-sourced shape's constraint_slots menu, so an
    # admitted match on it is also excluded from FIXED CONTEXT (which lists only slots
    # the menu offers). Another admitted match (crsTypeOfFlow, from "ODA") still
    # renders, so FIXED CONTEXT itself survives.
    withheld_prompt = _build_universal_bind_prompt(
        query,
        [dataclasses.replace(_CRS_SHAPE_WITH_MEASURE, slot_taxonomy_index_sourced=True)],
        admitted_qualifiers=[matches],
        withheld_measure_axes=frozenset({"financialObligationType"}),
    )
    assert "FIXED CONTEXT" in withheld_prompt
    assert "financialObligationType" not in withheld_prompt
    assert "property_dcid=crsTypeOfFlow" in withheld_prompt


# ---------------------------------------------------------------------------
# The trace: matched_text is safe in a log line (never a prompt).
# ---------------------------------------------------------------------------


def test_trace_names_each_admitted_match_and_its_source(caplog):
    matches = _admit("ODA standard grants for health", [_CRS_SHAPE])
    with caplog.at_level(logging.INFO, logger="search.engine.region.phase_rank_bind"):
        _log_admitted_qualifiers(matches)
    logged = "\n".join(r.getMessage() for r in caplog.records)
    for match in matches:
        assert match.code_system in logged
        assert match.code_dcid in logged
        assert match.source in logged
        assert match.matched_text in logged


# ---------------------------------------------------------------------------
# Synthetic code systems follow the same admission rules
# ---------------------------------------------------------------------------


def _synthetic_index(*rows: CodeRow, code_system: str = "syntheticAxis") -> CodeIndex:
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="synthetic",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={code_system: len(rows)},
            statvar_shape_counts={},
            content_hash="synthetic",
        ),
        declarations=(
            CodeSystemDeclaration(
                code_system=code_system,
                hierarchical=False,
                disjoint_from_children=True,
                summable=True,
                absent_reading="not_declared",
                plain_name="axis",
            ),
        ),
        codes=rows,
        statvars=(),
    )


def test_synthetic_code_system_admits_the_same_way_a_real_one_does():
    index = _synthetic_index(
        CodeRow(
            dcid="SyntheticAxisCodeAlpha",
            code_system="syntheticAxis",
            code="alpha",
            label="synthetic widget grade",
            synonyms=(),
            parent=None,
        )
    )
    matches = match_qualifiers("spending on synthetic widget grade equipment", index)
    assert len(matches) == 1
    assert matches[0].code_system == "syntheticAxis"

    declaring_shape = _shape(
        "synthetic_shape",
        (SlotKeyDraft(axis="how", property_dcid="syntheticAxis", label="synthetic axis"),),
    )
    admitted = _filter_admitted_matches(matches, {"syntheticAxis"})
    assert admitted == matches

    non_declaring_shape = _shape(
        "unrelated_shape", (SlotKeyDraft(axis="how", property_dcid="otherAxis", label="other"),)
    )
    declared = {
        prop
        for slot in non_declaring_shape.slot_keys
        for prop in (slot.property_dcid,)
        if prop is not None
    }
    assert _filter_admitted_matches(matches, declared) == ()
    # And the declaring shape's own declared set does admit it.
    declared_ok = {
        prop
        for slot in declaring_shape.slot_keys
        for prop in (slot.property_dcid,)
        if prop is not None
    }
    assert _filter_admitted_matches(matches, declared_ok) == matches


# ---------------------------------------------------------------------------
# End-to-end admission forces CRS values through binding and confirmation
# ---------------------------------------------------------------------------

_E2E_QUERY = "ODA standard grants to Kenya"
_E2E_SV_DCID = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode11110_Nominal"
)
_E2E_KEN_DCID = "country/KEN"


def test_e2e_fixture_matches_the_real_fixture_index():
    """Confirm the fixture StatVar declares the matched flow and finance values."""
    row = next(r for r in _REAL_INDEX.statvars if r.dcid == _E2E_SV_DCID)
    assert row.constraint_properties["crsTypeOfFlow"] == "CRSTypeOfFlow10"
    assert row.constraint_properties["crsTypeOfFinance"] == "CRSTypeOfFinance110"
    matches = _by_system(
        read_text(_E2E_QUERY, _REAL_INDEX, vocabularies=[_CRS]).by_vocabulary[_CRS].matches
    )
    assert matches["crsTypeOfFlow"].code_dcid == "CRSTypeOfFlow10"
    assert matches["crsTypeOfFinance"].code_dcid == "CRSTypeOfFinance110"

    # The fixture keeps the real row's observation-role signature.
    synthetic_arcs = _e2e_sv_arcs()
    synthetic_roles = tuple(n["dcid"] for n in synthetic_arcs["observationProperties"]["nodes"])
    assert synthetic_roles == row.observation_properties


def _e2e_sv_arcs() -> dict:
    row = next(r for r in _REAL_INDEX.statvars if r.dcid == _E2E_SV_DCID)
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": _E2E_SV_DCID}]},
        "populationType": {"nodes": [{"dcid": "FinancialFlow"}]},
        "measuredProperty": {"nodes": [{"dcid": "grantEquivalent"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
        "observationProperties": {"nodes": [{"dcid": p} for p in row.observation_properties]},
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

# The destination-end role every fixture CRS row declares; _E2E_QUERY's own
# "to Kenya" places Kenya onto it.
_E2E_ROLE_RECIPIENT = "financingRecipient"


# The origin-end role every fixture CRS row declares alongside
# _E2E_ROLE_RECIPIENT -- unbound by _E2E_QUERY, but still required to key the
# coverage row the destination-bound confirmation reads.
_E2E_ROLE_PROVIDER = "financingProvider"

_E2E_COVERAGE_ROWS = [(_E2E_SV_DCID, "f1", "country/FRA", _E2E_KEN_DCID, [2020])]
_E2E_ROLE_NAMES = {_E2E_SV_DCID: RoleNames(donor=_E2E_ROLE_PROVIDER, recipient=_E2E_ROLE_RECIPIENT)}


class _NoBindingLLM:
    """Elect the shape with no bindings, so admitted values come from forcing."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        # The raw phrase appears only in the echoed variable line, not fixed context.
        fixed_context = prompt.split("FIXED CONTEXT")[1].split("CANDIDATE")[0]
        assert "standard grants" not in fixed_context
        return elected_reply(0, also_plausible=[]), None


def _run_e2e(
    caplog,
    monkeypatch,
    *,
    variable: str = _E2E_QUERY,
    query_text: str | None = None,
    llm: SupportsLLM | None = None,
):
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch, _E2E_COVERAGE_ROWS, role_names=_E2E_ROLE_NAMES
    )
    graph = FakeGraph(
        nodes=_E2E_NODES, detect=_E2E_DETECT, resolve={"Kenya": _E2E_KEN_DCID}, obs={}
    )
    with caplog.at_level(logging.INFO, logger="search.engine.region.phase_rank_bind"):
        result = asyncio.run(
            resolve_variable(
                variable,
                entities=["Kenya"],
                date_request=None,
                detect_query=_E2E_QUERY,
                role_query=_E2E_QUERY,
                entity_roles=door_roles_for_test_query(_E2E_QUERY, ["Kenya"]),
                query_text=query_text,
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=llm if llm is not None else _NoBindingLLM(),
                base_timing={},
                coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
            )
        )
    return result


def test_e2e_admitted_matches_reach_the_wire_as_forced_value_slots(caplog, monkeypatch):
    result = _run_e2e(caplog, monkeypatch)
    assert result.status == "definite"
    spec = result.specs[0]

    def _slot(prop: str):
        return next(
            s
            for s in spec.resolved.slots
            if s.key.property is not None and s.key.property.ref == prop
        )

    flow_slot = _slot("crsTypeOfFlow")
    assert flow_slot.binding.kind == "value"
    assert flow_slot.binding.value.ref.ref == "CRSTypeOfFlow10"

    finance_slot = _slot("crsTypeOfFinance")
    assert finance_slot.binding.kind == "value"
    assert finance_slot.binding.value.ref.ref == "CRSTypeOfFinance110"

    # financialFlowType is one of this deployment's kind="constraint" measure
    # axes. The query does not name it, and admission does not touch it. Once
    # crsTypeOfFlow and crsTypeOfFinance are pinned to the admitted values
    # above, the real fixture index carries no FinancialFlow row without
    # financialFlowType=Disbursement, so the constraint measure axis election
    # in _repair_bindings forces it. This binding is engine-authored,
    # not an admitted match.
    flow_type_slot = _slot("financialFlowType")
    assert flow_type_slot.binding.kind == "value"
    assert flow_type_slot.binding.value.ref.ref == "Disbursement"

    # Every other constraint slot the shape declares was neither matched,
    # forced by admission, nor elected. It grounds as unbound, same as the
    # LLM stub's own empty draft, proving both forcings are scoped to their
    # own axes alone.
    for prop in ("crsPurpose", "crsCooperationModality"):
        assert _slot(prop).binding.kind == "unbound"

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "crsTypeOfFlow" in logged
    assert "CRSTypeOfFlow10" in logged
    assert "crsTypeOfFinance" in logged
    assert "CRSTypeOfFinance110" in logged


def test_e2e_query_text_admits_a_finance_type_qualifier_the_paraphrase_dropped(caplog, monkeypatch):
    """`query_text` supplies "standard grants" omitted from the variable paraphrase.

    The finance slot is forced, while query qualifiers contain no finance entry.
    """
    result = _run_e2e(caplog, monkeypatch, variable="ODA funding", query_text=_E2E_QUERY)
    assert result.status == "definite"
    spec = result.specs[0]

    def _slot(prop: str):
        return next(
            s
            for s in spec.resolved.slots
            if s.key.property is not None and s.key.property.ref == prop
        )

    flow_slot = _slot("crsTypeOfFlow")
    assert flow_slot.binding.kind == "value"
    assert flow_slot.binding.value.ref.ref == "CRSTypeOfFlow10"

    finance_slot = _slot("crsTypeOfFinance")
    assert finance_slot.binding.kind == "value"
    assert finance_slot.binding.value.ref.ref == "CRSTypeOfFinance110"

    assert not any(q.slot == "crsTypeOfFinance" for q in spec.resolved.trace.query_qualifiers)


# ---------------------------------------------------------------------------
# End-to-end: the guard refuses an unbound two-row tie
# ---------------------------------------------------------------------------


class _PermissiveNoBindingLLM:
    """Elect the shape without bindings, accepting a prompt with tie alternatives."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return (
            elected_reply(0, also_plausible=[]),
            None,
        )


def test_e2e_qualifier_not_bound_refuses_while_a_bound_control_still_answers(caplog, monkeypatch):
    control = _run_e2e(caplog, monkeypatch)
    assert control.status == "definite"

    result = _run_e2e(
        caplog, monkeypatch, variable="budget support to Kenya", llm=_PermissiveNoBindingLLM()
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "qualifier_not_bound"
    assert [q.text for q in result.unbound_qualifiers] == ["budget support"]
    assert result.unbound_qualifiers[0].disposition == "unbound"


# ---------------------------------------------------------------------------
# A query-text-only tie reaches the unbound-qualifier guard
# ---------------------------------------------------------------------------


class _TieElectingLLM:
    """Elect the shape and bind the query-text-only "grants" tie to row 110."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return (
            elected_reply(
                0,
                bindings=[
                    SlotBindingDraft(
                        axis="what",
                        property_dcid="crsTypeOfFinance",
                        kind="value",
                        value_dcids=["CRSTypeOfFinance110"],
                    )
                ],
                also_plausible=[],
            ),
            None,
        )


def test_e2e_query_text_only_tie_elects_cleanly_and_answers(caplog, monkeypatch):
    """A resolved query-text-only tie passes the guard."""
    result = _run_e2e(
        caplog,
        monkeypatch,
        variable="ODA to Kenya",
        query_text="ODA grants to Kenya",
        llm=_TieElectingLLM(),
    )
    assert result.status == "definite"
    spec = result.specs[0]
    grants = next(q for q in spec.resolved.trace.query_qualifiers if q.text == "grants")
    assert grants.disposition == "bound"
    assert {ref.ref for ref in grants.named} == {"CRSTypeOfFinance100", "CRSTypeOfFinance110"}


def test_e2e_query_text_only_tie_left_unresolved_refuses(caplog, monkeypatch):
    """An unresolved query-text-only tie triggers `qualifier_not_bound`."""
    result = _run_e2e(
        caplog,
        monkeypatch,
        variable="ODA to Kenya",
        query_text="ODA grants to Kenya",
        llm=_PermissiveNoBindingLLM(),
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "qualifier_not_bound"
    assert [q.text for q in result.unbound_qualifiers] == ["grants"]
    assert result.unbound_qualifiers[0].disposition == "unbound"


_CRS_GROSS_SV = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode11110_Nominal"
)


def _model_choice_state(
    variable: str, chosen: list[str], others: Sequence[str] = ()
) -> ResolveState:
    """Election state for `variable` on the CRS gross disbursements shape, whose bind model
    chose the purposes `chosen` (none when empty), as a set for several. The part also
    recalled the shapes of the index rows `others`, which it may elect after the CRS
    shape."""
    svs = [_CRS_GROSS_SV, *others]
    graph = FakeGraph(nodes={sv: _index_row_node(sv) for sv in svs}, obs={}, detect={}, resolve={})
    shape, *elsewhere = [
        shape_for_part(derived, None, _REAL_INDEX)
        for sv in svs
        for derived in derive_shapes(confirmed_svs=[sv], graph=graph)
    ]
    vocabulary = shape_vocabulary(shape)
    reading = read_text(variable, _REAL_INDEX, get_measure_index(), vocabularies=[vocabulary])
    bindings = (
        [
            SlotBindingDraft(
                axis="how",
                property_dcid="crsPurpose",
                kind="value" if len(chosen) == 1 else "set",
                value_dcids=chosen,
            )
        ]
        if chosen
        else []
    )
    state = _vet_state(
        variable=variable,
        capped_shapes=[shape, *elsewhere],
        chosen_shape_index=0,
        bindings=bindings,
        admitted_ties=(),
    )
    state.shapes_found = [shape, *elsewhere]
    state.variable_spans_in_every_system = reading.everywhere.spans
    state.variable_spans_by_vocabulary = {
        vocabulary: own.spans for vocabulary, own in reading.by_vocabulary.items()
    }
    state.query_qualifier_matches = {
        vocabulary: own.matches for vocabulary, own in reading.by_vocabulary.items()
    }
    return state


_P130_13030 = ("union", ["CRSPurposeCode13030", "CRSPurposeCode130"])


@pytest.mark.parametrize("mode", ["log", "refuse"])
@pytest.mark.parametrize(
    ("variable", "chosen", "purpose", "records", "gated"),
    [
        (
            "contraceptive services and reproductive health ODA",
            ["CRSPurposeCode13030"],
            ["CRSPurposeCode130"],
            {
                "contraceptive services": ("bound", "model", _P130_13030),
                "reproductive health": ("bound", None, _P130_13030),
            },
            {},
        ),
        (
            "population policies and reproductive health ODA",
            ["CRSPurposeCode13010"],
            ["CRSPurposeCode130"],
            {
                "population policies": ("bound", "model", None),
                "reproductive health": ("bound", None, None),
            },
            {},
        ),
        (
            "population policies and reproductive health ODA",
            [],
            ["CRSPurposeCode130"],
            {"reproductive health": ("bound", None, None)},
            {
                "population": ("unbound", "unavailable", None),
                "policies": ("unbound", "uncovered", None),
            },
        ),
        (
            "contraceptive services ODA",
            ["CRSPurposeCode13030"],
            ["CRSPurposeCode13030"],
            {"contraceptive services": ("bound", "model", None)},
            {},
        ),
        (
            "infant health ODA",
            ["CRSPurposeCode120"],
            ["CRSPurposeCode120", "CRSPurposeCode130"],
            {"health": ("bound", None, None)},
            {"infant": ("unbound", "uncovered", None)},
        ),
        (
            "infant health ODA",
            ["CRSPurposeCode121"],
            ["CRSPurposeCode120", "CRSPurposeCode130"],
            {"health": ("bound", None, None)},
            {"infant": ("unbound", "uncovered", None)},
        ),
        (
            "population ODA",
            ["CRSPurposeCode100"],
            ["CRSPurposeCode100"],
            {},
            {"population": ("unbound", "unavailable", None)},
        ),
        (
            "population ODA",
            ["CRSPurposeCode13020"],
            ["CRSPurposeCode13020"],
            {"population": ("bound", "model", None)},
            {},
        ),
        (
            "contraceptive services ODA and population policies ODA",
            ["CRSPurposeCode130"],
            ["CRSPurposeCode130"],
            {},
            {
                "population": ("unbound", "unavailable", None),
                "contraceptive services": ("unbound", "uncovered", None),
            },
        ),
        (
            "maternal and infant health ODA",
            ["CRSPurposeCode12240"],
            ["CRSPurposeCode12240"],
            {"infant health": ("bound", "model", None)},
            {"maternal": ("unbound", "uncovered", None)},
        ),
        (
            "girls education ODA",
            ["CRSPurposeCode15180"],
            ["CRSPurposeCode110"],
            {"education": ("bound", None, None)},
            {"girls": ("unbound", "uncovered", None)},
        ),
        (
            "environment education ODA",
            ["CRSPurposeCode41081"],
            ["CRSPurposeCode41081"],
            {"environment education": ("bound", "model", None)},
            {},
        ),
        (
            "water education ODA",
            ["CRSPurposeCode14081"],
            ["CRSPurposeCode14081"],
            {"water education": ("bound", "model", None)},
            {},
        ),
        (
            "hygiene education ODA",
            ["CRSPurposeCode12261"],
            ["CRSPurposeCode12261"],
            {"hygiene education": ("bound", "model", None)},
            {},
        ),
    ],
    ids=[
        "chosen-in-a-union",
        "chosen-reads-as-its-ancestor",
        "nothing-chosen",
        "chosen-leaf",
        "chosen-seed",
        "chosen-seed-child",
        "chosen-hit-ancestor",
        "chosen-hit-descendant",
        "on-two-menus",
        "unread-beside-a-chosen-phrase",
        "chosen-outside-the-terms-code",
        "chosen-outside-the-terms-code-naming-the-term-environment",
        "chosen-outside-the-terms-code-naming-the-term-water",
        "chosen-outside-the-terms-code-naming-the-term-hygiene",
    ],
)
def test_a_code_chosen_from_a_phrases_menu_reads_the_phrase(
    search_settings, mode, variable, chosen, purpose, records, gated
):
    """A code the bind model chooses from a phrase's candidate menu binds. It reads the
    phrase when it is a lexical hit of the menu or a descendant of one. The phrase then
    records it with evidence "model", combines with the other terms by the connective
    between them, and leaves no uncovered word or unavailable term inside it. 13030 chosen
    for "contraceptive services" unions with 130 into 130. 13010 chosen for "population
    policies" reads as 130, the hit above it that shares "population" too
    (`_broadest_reading`). 13020, a child of the hit 130, reads "population". With no
    choice the terms force as they would with no menu, and "population" and "policies"
    refuse under `refuse`. The term's own code (120 for "infant health", whose menu's hit
    is 12240), its child (121) and a hit's ancestor (100 for "population") read nothing
    and force nothing, so the terms force as with no menu. 121 gives way to the 120
    "health" names, which binds as the health grouping with 130, and 100, which no term contests, stays the model's binding. A code on
    two phrases' menus (130, an ancestor of 13030 and a hit for "population policies") is
    no choice. The model's binding stays and reads neither phrase. No term names 130, so
    its name reads "policies" (`_unnamed_bound_code_words`), and the part refuses on the
    rest. A chosen phrase is a code span to the words beside it, so "maternal" joined to
    "infant health" read as 12240 refuses. A choice outside the code of a term inside the
    phrase reads nothing unless its label or a synonym holds the term's words. 15180
    chosen for "girls education" leaves "education" its 110 and "girls" unread, while
    41081 "environmental education and training", 14081 "education and training in water
    supply and sanitation" and 12261 "health education" read their phrases in place of
    110."""
    search_settings(qualifier_coverage=mode)
    state = _model_choice_state(variable, chosen)

    _run_vet(state)
    state.bindings = list(state.bind_result.bindings)
    qualifiers = _resolve_query_qualifiers(state)

    bound = {b.property_dcid: b.value_dcids for b in state.bindings if b.kind != "unbound"}
    assert bound["crsPurpose"] == purpose

    def summary(q):
        operation = (q.operation, [ref.ref for ref in q.selected]) if q.operation else None
        return (q.disposition, q.evidence, operation)

    expected = {**records, **(gated if mode == "refuse" else {})}
    assert {q.text: summary(q) for q in qualifiers if q.text != "ODA"} == expected


_IMPUTED_SECTORS_SV = (
    "AttributedAmount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_DAC2aAidType106_"
    "CRSPurposeCode11220_Nominal"
)
_DAC1_MULTILATERAL_SV = "Amount_FinancialFlow_Gross_Disbursement_DAC1AidType2000_Nominal"


@pytest.mark.parametrize(
    ("variable", "others", "menus"),
    [
        # The words of a code term that another shape the part recalled binds are read, so
        # the CRS shape offers no menu for them. "imputed" is DAC2a aid type 106 on the
        # imputed sectors shape (dataset-imputed-education-and-health-aid-kenya), and
        # "multilateral ODA" is DAC1 aid type 2000 on the DAC1 multilateral shape
        # (sb-multilateral-oda-kenya).
        ("imputed education aid", [_IMPUTED_SECTORS_SV], []),
        ("multilateral ODA", [_DAC1_MULTILATERAL_SV], []),
        # With no such shape the words keep their menu. "ODA loans" is DAC2a aid type 218,
        # and "loans" is offered type of finance 420.
        ("imputed education aid", [], ["imputed education"]),
        ("ODA loans for education", [], ["loans"]),
        ("ODA loans for education", [_IMPUTED_SECTORS_SV], ["loans"]),
        ("equity investment ODA", [], ["equity investment"]),
        ("technical cooperation ODA", [], ["technical cooperation"]),
        # A word naming data itself is read (subj-health-data), and a run reaches past an
        # amount word to the code span ("water" beside "ODA").
        ("health data", [], []),
        ("water spending ODA", [], ["water"]),
        # "population" is type of finance 4, which binds on no shape, so it keeps its menu.
        ("population ODA", [], ["population"]),
        ("contraceptive services ODA", [], ["contraceptive services"]),
    ],
)
def test_a_candidate_menu_is_offered_for_words_no_code_reads(variable, others, menus):
    """A CRS shape is offered a candidate menu only for a phrase whose words no other
    recalled shape reads as a code it binds."""
    state = _model_choice_state(variable, [], others)
    assert [menu.text for menu in _candidate_menus(state, state.capped_shapes[0])] == menus


@pytest.mark.parametrize(
    ("places", "menus"),
    [
        ({}, ["domestic general government health"]),
        # Two places with no direction take no role on a flow shape (`bind_place_roles`
        # refuses role_not_resolved), as "African countries" in "how much do African
        # countries spend on their own health care?" takes none on the CRS flow shape
        # (health-spending-african-countries).
        ({"Kenya": "country/KEN", "Ghana": "country/GHA"}, []),
    ],
)
def test_a_shape_the_places_do_not_bind_to_is_offered_no_menu(places, menus):
    """A CRS shape the query's places do not bind to is offered no candidate menu, so no
    menu lifts it in the bind model's election over a shape that answers."""
    state = _model_choice_state("domestic general government health expenditure", [])
    state.entities = list(places)
    state.place_sets = place_sets(places)
    assert [menu.text for menu in _candidate_menus(state, state.capped_shapes[0])] == menus


@pytest.mark.parametrize(
    ("variable", "code_system", "chosen", "read"),
    [
        # 420's synonym "loan instrument" shares "loans" as its child 421 "standard loan"
        # does, so 421 chosen reads as 420 (crs-education-loans-france).
        ("education loans", "crsTypeOfFinance", "CRSTypeOfFinance421", "CRSTypeOfFinance420"),
        # "ODA loans" is DAC2a aid type 218, which no shape of the part binds, so "loans"
        # keeps its menu and 420 chosen reads it.
        (
            "ODA loans for education",
            "crsTypeOfFinance",
            "CRSTypeOfFinance420",
            "CRSTypeOfFinance420",
        ),
        # 130 shares "population" as its child 13010 does (sb-population-oda).
        ("population ODA", "crsPurpose", "CRSPurposeCode13010", "CRSPurposeCode130"),
        # 13030 shares "contraceptive", which none of its ancestors holds.
        ("contraceptive services ODA", "crsPurpose", "CRSPurposeCode13030", "CRSPurposeCode13030"),
    ],
)
def test_a_chosen_hit_reads_as_its_broadest_ancestor_sharing_its_words(
    variable, code_system, chosen, read
):
    """A code chosen from a phrase's menu reads the phrase as the furthest ancestor among
    the menu's hits whose label or synonyms share every word of the phrase its own share."""
    state = _model_choice_state(variable, [])
    binding = SlotBindingDraft(
        axis="how", property_dcid=code_system, kind="value", value_dcids=[chosen]
    )
    choices = _menu_choices(state, state.capped_shapes[0], [binding], terms=())
    assert [choice.code_dcid for choice in choices] == [read]


class _MenuChoosingLLM:
    """Elect the shape and bind the purpose the "school governance" menu offers as
    education policy and administrative management, 11110, which the fixture row holds."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        menu = prompt.split('    phrase "school governance":\n')[1].split("\n\n")[0]
        assert "value_dcid=CRSPurposeCode11110 (" in menu
        binding = SlotBindingDraft(
            axis="how",
            property_dcid="crsPurpose",
            kind="value",
            value_dcids=["CRSPurposeCode11110"],
        )
        return (
            elected_reply(0, bindings=[binding], also_plausible=[]),
            None,
        )


def test_e2e_a_menu_choice_binds_and_the_confirmation_sentence_names_it(
    caplog, monkeypatch, search_settings
):
    """No code reads "school governance". Its menu offers 11110 through the synonym
    "education governance", the model chooses it, and the part answers definite with 11110
    bound and named in its sentence, the phrase recorded with evidence "model". The model
    chose the code, so the slot's provenance is the binder's."""
    search_settings(qualifier_coverage="refuse")
    result = _run_e2e(
        caplog, monkeypatch, variable="school governance ODA to Kenya", llm=_MenuChoosingLLM()
    )
    assert result.status == "definite"
    spec = result.specs[0]
    purpose = next(
        s for s in spec.resolved.slots if s.key.property and s.key.property.ref == "crsPurpose"
    )
    assert purpose.binding.value.ref.ref == "CRSPurposeCode11110"
    assert "education policy and administrative management" in render_sentence(spec)
    record = next(q for q in spec.resolved.trace.query_qualifiers if q.evidence == "model")
    assert (record.text, record.disposition) == ("school governance", "bound")
    purpose_filter = next(
        f
        for f in spec.resolved.trace.slot_filters
        if f.key.property is not None and f.key.property.ref == "crsPurpose"
    )
    assert purpose_filter.provenance == "binder"
