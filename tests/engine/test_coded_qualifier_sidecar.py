"""Tests how coded qualifier sidecars affect admission, binding, warnings, and cache keys.
Producer tests compare the real extraction sidecar with appended query text.
"""

from __future__ import annotations

import asyncio
from typing import cast

from code_index.shared import get_index, get_measure_index
from code_index.types import CodeRow
from entity_resolution import Admission, EntityResolver
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.extract import _EXTRACTION_SYSTEM_PROMPT, Extraction, ExtractionResult, extract
from search.engine.qualifier_match import (
    CodedQualifier,
    QualifierSpan,
    match_qualifiers,
    recall_text,
)
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.region.phase_election import _vet_election
from search.engine.region.phase_rank_bind import (
    _admit_qualifier_matches,
    _admit_qualifier_ties,
    _force_qualifier_bindings,
    _resolve_query_qualifiers,
)
from search.engine.region.state import ResolveState, shape_vocabulary
from search.engine.response_cache_keys import layer2_key, scope_document
from search.engine.retrieval.bound_values import held_codes
from search.engine.shape import ShapeDraft, SlotKeyDraft
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._index_slots import crs_vocabulary, own_variable_spans, read_text
from tests.engine._index_slots import query_qualifiers as read_query_qualifiers
from tests.engine._requirements import depart_requirements
from tests.fixtures import FakeLLM
from tests.fixtures.keys import fixture_key

_INDEX = get_index()
_MEASURE_INDEX = get_measure_index()


def _shape(shape_id: str, *slot_keys: SlotKeyDraft) -> ShapeDraft:
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


_FINANCE_SHAPE = _shape(
    "finance_shape",
    SlotKeyDraft(axis="what", property_dcid="crsTypeOfFinance", label="finance type"),
)
# The vocabulary of the DAC1 flow shape, as the committed index's family declares it.
_DAC1_VOCABULARY = frozenset(
    {"dac1AidType", "dacTypeOfFinance", "financialFlowType", "nettingBasis"}
)
_CRS_SHAPE = _shape(
    "crs_shape",
    SlotKeyDraft(axis="how", property_dcid="crsCooperationModality", label="modality"),
    SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="purpose"),
)


def _coded(phrase: str) -> CodedQualifier:
    """Build spans and matches from the phrase, as the producer does."""
    return CodedQualifier(
        phrase=phrase,
        spans=read_text(phrase, _INDEX, _MEASURE_INDEX).everywhere.spans,
        matches=match_qualifiers(phrase, _INDEX, _MEASURE_INDEX),
    )


# ---------------------------------------------------------------------------
# 1. A sidecar match reaches admission and a forced binding.
# ---------------------------------------------------------------------------


def test_sidecar_match_reaches_admitted_qualifiers_and_forced_bindings():
    coded = _coded("type of finance 110")
    vocabulary = shape_vocabulary(_FINANCE_SHAPE)
    read = read_query_qualifiers(
        "aid to Kenya",
        [],
        query_text=None,
        qualifiers=(coded,),
        vocabularies=[vocabulary, _DAC1_VOCABULARY],
    )
    assert {m.code_dcid for m in read.matches[vocabulary]} == {"CRSTypeOfFinance110"}
    # A vocabulary without the CRS type of finance does not receive the sidecar's code.
    assert read.matches[_DAC1_VOCABULARY] == ()
    assert "110" in read.phrases

    admitted = _admit_qualifier_matches([_FINANCE_SHAPE], read.matches)
    assert {m.code_dcid for m in admitted} == {"CRSTypeOfFinance110"}

    forced, _ = _force_qualifier_bindings([], admitted, _FINANCE_SHAPE, variable="aid to Kenya")
    by_property = {b.property_dcid: b for b in forced}
    assert by_property["crsTypeOfFinance"].kind == "value"
    assert by_property["crsTypeOfFinance"].value_dcids == ["CRSTypeOfFinance110"]


# ---------------------------------------------------------------------------
# 2. A sidecar-only tie the elected shape leaves unresolved produces the
# region-level QUERY_QUALIFIER_DROPPED warning -- proves
# variable_matched_qualifier_phrases (the warning's ownership gate) widens
# to include a sidecar-sourced phrase, not just one variable/query_text
# names directly.
# ---------------------------------------------------------------------------


def _vet_state(
    *,
    capped_shapes: list[ShapeDraft],
    bindings: list[SlotBindingDraft],
    admitted_ties,
    variable_matched_qualifier_phrases: frozenset[str],
) -> ResolveState:
    return ResolveState(
        variable="ODA to Kenya",
        entities=[],
        date_request=None,
        detect_query="ODA to Kenya",
        role_query="ODA to Kenya",
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
        capped_shapes=capped_shapes,
        query_qualifier_ties={shape_vocabulary(shape): admitted_ties for shape in capped_shapes},
        admitted_ties=admitted_ties,
        variable_matched_qualifier_phrases=variable_matched_qualifier_phrases,
        bind_result=_UniversalBindOutput(
            chosen_shape_index=0, bindings=bindings, also_plausible=None, ask=None
        ),
        coverage=empty_coverage_snapshot(),
    )


def test_sidecar_only_tie_left_unresolved_produces_query_qualifier_dropped():
    # "budget support" never appears in `variable` here -- the sidecar is
    # its only source. `is_coded_only_phrase` would never fold an ambiguous
    # phrase like this into `variable` in production (the producer requires
    # an unambiguous code), but nothing in the sidecar-consumption
    # machinery assumes that; feeding ResolveState.qualifiers directly
    # exercises that machinery independent of the producer.
    coded = CodedQualifier(
        phrase="budget support",
        spans=read_text("budget support", _INDEX, _MEASURE_INDEX).everywhere.spans,
        matches=(),
    )
    read = read_query_qualifiers(
        "ODA to Kenya",
        [],
        query_text=None,
        qualifiers=(coded,),
        vocabularies=[shape_vocabulary(_CRS_SHAPE), _DAC1_VOCABULARY],
    )
    ties, owned_phrases = read.ties, read.phrases
    assert {t.matched_text for t in ties[shape_vocabulary(_CRS_SHAPE)]} == {"budget support"}
    # A DAC1 vocabulary keeps only the tie rows of its own system.
    assert [{row.code_dcid for row in tie.rows} for tie in ties[_DAC1_VOCABULARY]] == [
        {"DAC1AidType1100", "DAC1AidType2210"}
    ]
    assert "budget support" in owned_phrases

    _promoted, admitted_ties, _unheld = _admit_qualifier_ties([_CRS_SHAPE], ties, holds=held_codes)
    assert len(admitted_ties) == 1, "both rows survive -- _CRS_SHAPE declares both systems"

    state = _vet_state(
        capped_shapes=[_CRS_SHAPE],
        bindings=[],  # the model drafts nothing -- zero rows bound, tie dropped
        admitted_ties=admitted_ties,
        variable_matched_qualifier_phrases=owned_phrases,
    )
    result = _vet_election(state)
    assert result is None, "expected _vet_election to fall through"
    depart_requirements(state)
    assert len(state.warnings) == 1
    assert state.warnings[0].code == QUERY_QUALIFIER_DROPPED
    assert "budget support" in state.warnings[0].message


def test_sidecar_tie_owned_elsewhere_stays_silent():
    """A tie owned by another phrase source produces no dropped-qualifier warning."""
    coded = CodedQualifier(
        phrase="budget support",
        spans=read_text("budget support", _INDEX, _MEASURE_INDEX).everywhere.spans,
        matches=(),
    )
    ties = read_query_qualifiers(
        "ODA to Kenya",
        [],
        query_text=None,
        qualifiers=(coded,),
        vocabularies=[shape_vocabulary(_CRS_SHAPE)],
    ).ties
    _promoted, admitted_ties, _unheld = _admit_qualifier_ties([_CRS_SHAPE], ties, holds=held_codes)

    state = _vet_state(
        capped_shapes=[_CRS_SHAPE],
        bindings=[],
        admitted_ties=admitted_ties,
        variable_matched_qualifier_phrases=frozenset(),
    )
    result = _vet_election(state)
    assert result is None
    depart_requirements(state)
    assert state.warnings == []


# ---------------------------------------------------------------------------
# 3. A sidecar match that duplicates one already derived from `variable`
# appears exactly once, both in the merged QualifierMatch tuple and in the
# ownership phrase set.
# ---------------------------------------------------------------------------


def test_sidecar_duplicate_of_a_variable_match_appears_once():
    variable = "ODA standard grants for health"
    coded = _coded("type of finance 110")  # same CRSTypeOfFinance110 row, different matched_text

    read = read_query_qualifiers(
        variable, [], query_text=None, qualifiers=(coded,), vocabularies=[crs_vocabulary()]
    )
    matches, variable_matched_phrases = read.matches, read.phrases
    finance_matches = [m for m in matches[crs_vocabulary()] if m.code_dcid == "CRSTypeOfFinance110"]
    assert len(finance_matches) == 1
    # The surviving copy is variable's own ("standard grants"), not the
    # sidecar's ("110") -- variable is matched first, so it seeds `seen`.
    assert finance_matches[0].matched_text == "standard grants"
    # The sidecar's own phrase still joins the ownership set even though its
    # match itself was deduplicated away -- it still has no other owner.
    assert "110" in variable_matched_phrases
    assert "standard grants" in variable_matched_phrases


# ---------------------------------------------------------------------------
# 4. A sidecar span's own phrase-local offsets never enter the
# `variable`-scoped occupied_ranges check `_resolve_query_qualifiers` runs
# against bind_result.unplaced.
# ---------------------------------------------------------------------------


def _guard_state(
    *, variable: str, shape_draft: ShapeDraft, unplaced: list[str], qualifiers
) -> ResolveState:
    return ResolveState(
        variable=variable,
        entities=[],
        date_request=None,
        detect_query=variable,
        role_query=variable,
        qualifiers=qualifiers,
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
        bind_result=_UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=None, ask=None, unplaced=unplaced
        ),
        shape_draft=shape_draft,
        # No real CRS text in `variable` itself -- the shape's own spans stay
        # empty, so occupied_ranges (built from them alone) starts empty.
        variable_spans_by_vocabulary=own_variable_spans(variable, shape_draft),
        coverage=empty_coverage_snapshot(),
    )


def test_sidecar_span_offsets_never_enter_occupied_ranges():
    variable = "aid to Kenya"
    assert read_text(variable, _INDEX, _MEASURE_INDEX).everywhere.spans == ()
    kenya_start = variable.casefold().index("kenya")
    kenya_end = kenya_start + len("kenya")

    # A fabricated sidecar span sharing "Kenya"'s own position in `variable`
    # -- its offsets are into a DIFFERENT text (its own phrase), so if
    # _resolve_query_qualifiers ever folded them into occupied_ranges, the
    # unplaced "Kenya" phrase below would be spuriously excluded by the
    # overlap check.
    real_code_row = (
        read_text("type of finance 110", _INDEX, _MEASURE_INDEX).everywhere.spans[0].rows[0]
    )
    assert isinstance(real_code_row, CodeRow)
    fabricated_span = QualifierSpan(
        matched_text="fabricated phrase text",
        start=kenya_start,
        end=kenya_end,
        rows=(real_code_row,),
    )
    coded = CodedQualifier(phrase="fabricated phrase text", spans=(fabricated_span,), matches=())

    # A shape declaring nothing at all: the fabricated span's own code
    # system is never `declared`, so source 5 never records a
    # ResolvedQualifier for it -- only occupied_ranges (built exclusively
    # from the shape's own variable spans) is under test here.
    bare_shape = _shape("bare_shape")
    state = _guard_state(
        variable=variable, shape_draft=bare_shape, unplaced=["Kenya"], qualifiers=(coded,)
    )
    qualifiers = _resolve_query_qualifiers(state)
    by_text = {q.text: q for q in qualifiers}
    assert "Kenya" in by_text, (
        "the sidecar span's offsets leaked into occupied_ranges and "
        "spuriously suppressed the unplaced 'Kenya' phrase"
    )
    assert by_text["Kenya"].disposition == "no_vocabulary"


def test_qualifiers_empty_is_the_exact_baseline_behaviour():
    """None and empty sidecars match the no-sidecar baseline."""
    variable = "ODA standard grants for health"
    crs = [crs_vocabulary()]
    read_none = read_query_qualifiers(variable, [], query_text=None, vocabularies=crs)
    read_default = read_query_qualifiers(
        variable, [], query_text=None, qualifiers=None, vocabularies=crs
    )
    read_empty = read_query_qualifiers(
        variable, [], query_text=None, qualifiers=(), vocabularies=crs
    )
    assert read_none == read_default == read_empty

    ties_none = read_query_qualifiers("budget support", [], query_text=None, vocabularies=crs)
    ties_empty = read_query_qualifiers(
        "budget support", [], query_text=None, qualifiers=(), vocabularies=crs
    )
    assert ties_none == ties_empty
    assert ties_none.ties[crs_vocabulary()]


# ---------------------------------------------------------------------------
# 5. The acceptance test: the sidecar a real extract() call populates
# carries exactly what an appended string would have carried, proven by
# running the SAME real producer output through the consumption machinery
# twice -- once with the sidecar passed through, once with it discarded --
# and comparing every downstream artefact a consumer reads.
# ---------------------------------------------------------------------------


def _stub_extract_llm(query: str, payload: dict) -> FakeLLM:
    """Serve a fixed extraction payload for one query."""
    return FakeLLM(responses={fixture_key(Extraction, _EXTRACTION_SYSTEM_PROMPT, query): payload})


_EQUIVALENCE_SHAPE = _shape(
    "finance_and_flow_shape",
    SlotKeyDraft(axis="what", property_dcid="crsTypeOfFinance", label="finance type"),
    SlotKeyDraft(axis="how", property_dcid="crsTypeOfFlow", label="flow type"),
)


def _run_qualifier_pipeline(
    variable: str, qualifiers: tuple[CodedQualifier, ...]
) -> tuple[
    tuple,
    tuple,
    tuple,
    tuple,
    object,
]:
    """Run admission, election, forced binding, and qualifier resolution without an LLM bind.
    Return each downstream result and the election outcome for equivalence checks.
    """
    read = read_query_qualifiers(
        variable,
        [],
        query_text=None,
        qualifiers=qualifiers,
        vocabularies=[shape_vocabulary(_EQUIVALENCE_SHAPE)],
    )
    matches, variable_matched_phrases = read.matches, read.phrases
    pre_bind_admitted = _admit_qualifier_matches([_EQUIVALENCE_SHAPE], matches)
    state = ResolveState(
        variable=variable,
        entities=[],
        date_request=None,
        detect_query=variable,
        role_query=variable,
        qualifiers=qualifiers,
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
        capped_shapes=[_EQUIVALENCE_SHAPE],
        query_qualifier_matches=matches,
        admitted_qualifiers=pre_bind_admitted,
        variable_matched_qualifier_phrases=variable_matched_phrases,
        variable_spans_by_vocabulary=own_variable_spans(variable, _EQUIVALENCE_SHAPE),
        bind_result=_UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=None, ask=None
        ),
        coverage=empty_coverage_snapshot(),
    )
    fell_through = _vet_election(state)
    depart_requirements(state)
    # Mirrors _repair_bindings' own `state.bindings = list(state.
    # bind_result.bindings)` -- _resolve_query_qualifiers reads state.
    # bindings, not state.bind_result.bindings, for its own bound/unbound
    # disposition call.
    state.bindings = list(state.bind_result.bindings)
    query_qualifiers = _resolve_query_qualifiers(state)
    return (
        state.admitted_qualifiers,
        tuple(state.bind_result.bindings),
        tuple(state.warnings),
        tuple(query_qualifiers),
        fell_through,
    )


def test_bare_variable_plus_sidecar_reproduces_the_old_appended_strings_outcome(search_settings):
    """The acceptance test for the sidecar-only production shape: the
    bare-variable-plus-sidecar production shape reaches the IDENTICAL
    admission/binding/warning/query-qualifier outcome an appended-string,
    no-sidecar shape would, and `detect_query`
    (`qualifier_match.recall_text`) is byte-identical to what the appended
    string would be.

    `_reattach_coded_qualifiers` never rewrites `Extraction.variables`, so
    `result.extraction.variables[0]` is the bare measure text the model
    produced -- this is what proves the sidecar, not a string rewrite,
    carries the reattached codes. The appended-string shape is
    reconstructed here by running the SAME consumption pipeline
    (`_run_qualifier_pipeline`) against the composed `detect_query` text
    with an EMPTY sidecar -- what a caller would see if `variable` itself
    carried the codes and nothing populated `ResolveState.qualifiers`. The
    sidecar shape runs the bare `variable` with the REAL sidecar the
    producer built (not a hand-authored fixture). Both runs share the SAME
    query and the SAME real extract() call.
    """
    # The appended string holds "type of finance" beside the code 110, words no coverage
    # reader covers, so under the refuse mode its records differ from the sidecar's.
    search_settings(qualifier_coverage="log")
    query = "Gross disbursements type of finance 110 type of flow 10 to Kenya"
    llm = _stub_extract_llm(
        query,
        {
            "kind": "resolution",
            "variables": ["gross disbursements"],
            "entities": [{"text": "Kenya", "span": "Kenya"}],
            "dropped_qualifiers": ["type of finance 110 type of flow 10"],
            "dates": [],
        },
    )
    result, _usage = asyncio.run(extract(query, llm=llm))
    assert isinstance(result, ExtractionResult)
    variable = result.extraction.variables[0]
    # `variable` is the bare measure text the model produced, never a
    # phrase-appended string.
    assert variable == "gross disbursements"
    assert result.extraction.dropped_qualifiers == []

    populated = result.qualifiers.get(variable, ())
    assert populated, "the producer must have populated the sidecar for this query"
    assert {m.code_dcid for coded in populated for m in coded.matches} == {
        "CRSTypeOfFinance110",
        "CRSTypeOfFlow10",
    }

    detect_query = recall_text(variable, populated)
    # Byte-identical to what an appended string would carry for this exact
    # query -- recall's own input never moves.
    assert detect_query == "gross disbursements type of finance 110 type of flow 10"

    new_shape_result = _run_qualifier_pipeline(variable, populated)
    # The appended-string shape: `variable` itself carries the composed
    # text, and no sidecar is consulted.
    old_shape_result = _run_qualifier_pipeline(detect_query, ())

    assert new_shape_result == old_shape_result

    # Not vacuously equal -- the shared outcome actually binds both codes,
    # rather than "nothing happened either way".
    admitted, forced, warnings, query_qualifiers, fell_through = new_shape_result
    assert {m.code_dcid for m in admitted} == {"CRSTypeOfFinance110", "CRSTypeOfFlow10"}
    by_property = {b.property_dcid: b for b in forced}
    assert by_property["crsTypeOfFinance"].value_dcids == ["CRSTypeOfFinance110"]
    assert by_property["crsTypeOfFlow"].value_dcids == ["CRSTypeOfFlow10"]
    assert warnings == ()
    assert {q.disposition for q in query_qualifiers} == {"bound"}
    assert fell_through is None


# ---------------------------------------------------------------------------
# 6. Two SEPARATE dropped_qualifiers entries targeting one variable -- the
# case none of section 5's tests exercise (a single entry, however many
# codes it names). Each asserts recall_text's composed detect_query
# matches appending the same two phrases, in the same order.
# ---------------------------------------------------------------------------


def test_recall_text_composes_two_dropped_qualifiers_with_disjoint_codes():
    """Two phrases, two different codes, neither phrase's text overlapping
    the other's or `variable`'s: both phrases are appended, in order."""
    query = "Gross disbursements type of finance 110 to Kenya type of flow 10"
    llm = _stub_extract_llm(
        query,
        {
            "kind": "resolution",
            "variables": ["gross disbursements"],
            "entities": [{"text": "Kenya", "span": "Kenya"}],
            "dropped_qualifiers": ["type of finance 110", "type of flow 10"],
            "dates": [],
        },
    )
    result, _usage = asyncio.run(extract(query, llm=llm))
    variable = result.extraction.variables[0]
    assert result.extraction.dropped_qualifiers == []

    populated = result.qualifiers.get(variable, ())
    # Both phrases survive into the sidecar bucket -- neither's code is a
    # subset of the other's, so nothing here would be dropped even under a
    # code-identity dedup; included for symmetry with the other two cases
    # below.
    assert len(populated) == 2
    assert {m.code_dcid for coded in populated for m in coded.matches} == {
        "CRSTypeOfFinance110",
        "CRSTypeOfFlow10",
    }

    detect_query = recall_text(variable, populated)
    assert detect_query == "gross disbursements type of finance 110 type of flow 10"


def test_recall_text_composes_two_dropped_qualifiers_with_overlapping_codes():
    """Two phrases naming the SAME code under different phrasings ("code" is
    a recognised filler word, so "type of finance code 110" is a distinct
    phrase resolving to the identical CRSTypeOfFinance110 row). recall_text
    appends both, since its idempotence check is a literal text substring
    test, not a code-identity test -- the second phrase's text is not
    contained in the first's. A code-identity dedupe at sidecar
    construction time would drop the second phrase's text entirely; this
    asserts it does not."""
    query = "Gross disbursements type of finance 110 to Kenya type of finance code 110"
    llm = _stub_extract_llm(
        query,
        {
            "kind": "resolution",
            "variables": ["gross disbursements"],
            "entities": [{"text": "Kenya", "span": "Kenya"}],
            "dropped_qualifiers": ["type of finance 110", "type of finance code 110"],
            "dates": [],
        },
    )
    result, _usage = asyncio.run(extract(query, llm=llm))
    variable = result.extraction.variables[0]
    assert result.extraction.dropped_qualifiers == []

    populated = result.qualifiers.get(variable, ())
    # Both phrases survive into the bucket even though they resolve to the
    # same code -- a code-identity dedupe at construction time would keep
    # only the first and silently drop the second phrase's text.
    assert len(populated) == 2
    assert [coded.phrase for coded in populated] == [
        "type of finance 110",
        "type of finance code 110",
    ]

    detect_query = recall_text(variable, populated)
    assert detect_query == "gross disbursements type of finance 110 type of finance code 110"


def test_recall_text_skips_a_dropped_qualifier_whose_text_is_a_substring_of_the_composed_text():
    """The second phrase's own text ("finance 110") is a literal substring
    of the text already composed by the time it is reached (`variable` plus
    the first phrase, "gross disbursements type of finance 110") --
    recall_text's growing-string idempotence check skips it, even though it
    is not a substring of `variable` alone. A check against `variable`
    alone (rather than the growing composition) would append it a second
    time here."""
    query = "Gross disbursements type of finance 110 to Kenya finance 110"
    llm = _stub_extract_llm(
        query,
        {
            "kind": "resolution",
            "variables": ["gross disbursements"],
            "entities": [{"text": "Kenya", "span": "Kenya"}],
            "dropped_qualifiers": ["type of finance 110", "finance 110"],
            "dates": [],
        },
    )
    result, _usage = asyncio.run(extract(query, llm=llm))
    variable = result.extraction.variables[0]
    assert result.extraction.dropped_qualifiers == []

    populated = result.qualifiers.get(variable, ())
    # The sidecar bucket still carries both phrases -- the skip belongs to
    # recall_text's own composition, not to sidecar construction.
    assert len(populated) == 2

    detect_query = recall_text(variable, populated)
    assert detect_query == "gross disbursements type of finance 110"


# ---------------------------------------------------------------------------
# 7. The layer-2 cache key: `recall_text` (section 6 above) is deliberately
# order-SENSITIVE, since it composes exactly what retrieval receives, but
# `response_cache_keys._variable_qualifiers_document` folds each
# `CodedQualifier` tuple into the key sorted by content, not by the order
# `extract._reattach_coded_qualifiers` happened to receive it in (already
# known to vary across identical calls -- see `_coded_qualifier_sort_key`'s
# own docstring). Two requests whose `recall_text` genuinely differs would
# collide on one sorted key unless the key also carries `recall_text`
# itself, verbatim, alongside the sorted set.
# ---------------------------------------------------------------------------


_KEY_SCOPE = scope_document(
    schema_version=1,
    options=None,
    max_candidates=6,
    seam_default=True,
    qualifier_coverage="log",
    build_commit="commit-a",
    binder="gemini",
    model="model-a",
    seed=1,
    today="2026-09-22",
    graph_generation="unknown",
    coverage_stamp="unknown",
    conversion_catalogue=None,
    origin_catalogue="build-a",
    un_system_enabled=True,
)


def _extraction_result(variable: str, qualifiers: tuple[CodedQualifier, ...]) -> ExtractionResult:
    """A hand-built `ExtractionResult` mirroring what `extract()` returns:
    `variable` verbatim in `Extraction.variables`, `qualifiers` keyed on that
    same verbatim text -- `ExtractionResult.qualifiers`' own docstring."""
    return ExtractionResult(
        extraction=Extraction(
            kind="resolution", variables=[variable], entities=[], dates=[], dropped_qualifiers=[]
        ),
        qualifiers={variable: qualifiers},
    )


def _key_for(result: ExtractionResult) -> str:
    variable = result.extraction.variables[0]
    return layer2_key(
        variables=result.extraction.variables,
        entities=result.extraction.entities,
        dates=result.extraction.dates,
        dropped_qualifiers=result.extraction.dropped_qualifiers,
        variable_qualifiers=[result.qualifiers.get(variable, ())],
        operations=[()],
        sources=[()],
        role_contradictions=[],
        query_text_qualifier_matches=[],
        query_text_qualifier_ties=[],
        query_text_qualifier_readings=[],
        query_text_qualifier_occurrences=[],
        unsupported_qualifier_phrases=[],
        declared_default_trigger_words=[],
        declared_default_slots_named=[],
        scope=_KEY_SCOPE,
    )


def test_layer2_key_changes_when_qualifier_order_changes_the_composed_recall_text():
    """`forward` and `backward` carry the IDENTICAL set of `CodedQualifier`s
    for `variable` -- only the tuple order differs, exactly the
    nondeterminism `extract._reattach_coded_qualifiers` cannot promise is
    stable across identical calls. `finance_substring`'s own phrase
    ("finance 110") is a literal substring of what `finance`'s own phrase
    ("type of finance 110") composes onto `variable`, the containment case
    section 6 above already proves is order-sensitive: `recall_text`
    composes genuinely different text for the two orders, so retrieval
    itself would receive different input for the two. The two must not
    share a layer-2 key, or one order's answer would be served to the
    other's request."""
    variable = "gross disbursements"
    finance = _coded("type of finance 110")
    finance_substring = _coded("finance 110")

    forward = _extraction_result(variable, (finance, finance_substring))
    backward = _extraction_result(variable, (finance_substring, finance))

    recall_forward = recall_text(variable, forward.qualifiers[variable])
    recall_backward = recall_text(variable, backward.qualifiers[variable])
    assert recall_forward != recall_backward, (
        "fixture no longer exercises the containment case this test needs -- "
        "recall_text must compose different text for the two orders"
    )

    assert _key_for(forward) != _key_for(backward)


def test_layer2_key_stays_stable_when_qualifier_order_does_not_change_the_composed_recall_text():
    """The companion control: `variable` here already carries both phrases'
    own text, so `recall_text` skips both qualifiers in either order and
    returns `variable` unchanged regardless of which order `forward`/
    `backward` carry them in. The two ExtractionResults must still share a
    key -- extraction's own nondeterministic order must not fragment the
    cache for a query whose composed recall text does not actually move."""
    variable = "gross disbursements type of finance 110 type of flow 10"
    finance = _coded("type of finance 110")
    flow = _coded("type of flow 10")

    forward = _extraction_result(variable, (finance, flow))
    backward = _extraction_result(variable, (flow, finance))

    recall_forward = recall_text(variable, forward.qualifiers[variable])
    recall_backward = recall_text(variable, backward.qualifiers[variable])
    assert recall_forward == recall_backward == variable

    assert _key_for(forward) == _key_for(backward)
