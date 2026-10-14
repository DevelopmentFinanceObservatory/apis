"""Checks bind-menu parsing and truncation against prompts built by the production builder."""

from __future__ import annotations

import pytest

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from promptlab.bindmenu import BindMenuError, parse_menu, truncate_menu
from search.engine import bind as bind_module
from search.engine.bind import _build_universal_bind_prompt
from search.engine.frame import door_frame
from search.engine.interpretation import Interpretation
from search.engine.qualifier_match import QualifierMatch
from search.engine.shape import ShapeDraft, SlotKeyDraft

VARIABLE = "gdp per capita"

# Two entries exercise iteration over every fixed-context line.
ADMITTED_QUALIFIERS = (
    QualifierMatch(
        code_system="p:unit",
        code_dcid="USDollar",
        label="US Dollar",
        source="code",
        matched_text="dollars",
    ),
    QualifierMatch(
        code_system="p:a",
        code_dcid="Annual",
        label="Annual",
        source="label",
        matched_text="annual",
    ),
)

# Two phrases exercise iteration over every dropped-qualifier line.
DROPPED_QUALIFIER_PHRASES = ("across all flows", "regardless of type")

_STAMP = IndexStamp(
    graph_endpoint="test",
    exported_at="2026-01-01T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="test",
)


def _synthetic_flat_index() -> CodeIndex:
    codes = [
        CodeRow(
            dcid="FlatA", code_system="synthFlat", code="A", label="A", synonyms=(), parent=None
        ),
        CodeRow(
            dcid="FlatB", code_system="synthFlat", code="B", label="B", synonyms=(), parent=None
        ),
    ]
    declarations = [
        CodeSystemDeclaration(
            code_system="synthFlat",
            hierarchical=False,
            disjoint_from_children=True,
            summable=True,
            absent_reading="not_declared",
            plain_name="axis",
        )
    ]
    return CodeIndex(stamp=_STAMP, declarations=declarations, codes=codes, statvars=())


# This shape exercises the index-sourced `labels` line in the prompt.
_INDEX_SOURCED_SHAPE = ShapeDraft(
    shape_id="index-sourced",
    label="Index-sourced shape",
    pop_type_dcid="TestPop",
    meas_prop_dcid="testMeasure",
    stat_type_dcid="measuredValue",
    meas_qual_dcid=None,
    meas_denom_dcid=None,
    slot_keys=(SlotKeyDraft(axis="how", property_dcid="synthFlat", label="synth flat"),),
    slot_taxonomy={"how:synthFlat": ["FlatA", "FlatB"]},
    slot_taxonomy_index_sourced=True,
)

SHAPES: list[ShapeDraft] = [
    # 0: denominator-only; 25 taxonomy values exceed the 20-value cap, and 6 samples exceed 5.
    ShapeDraft(
        shape_id="s0",
        label="GDP per capita",
        pop_type_dcid="Country",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid="Count_Person",
        slot_keys=(SlotKeyDraft(axis="what", property_dcid="p:unit", label="unit"),),
        slot_taxonomy={"what:p:unit": [f"unit{i}" for i in range(25)]},
        sv_arc_facts={f"dc/sv{i}": {} for i in range(6)},
        representative_score=0.9123,
    ),
    # 1: unresolved denominator, no slots, and no sample DCIDs.
    ShapeDraft(
        shape_id="s1",
        label="Health ODA",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="oda_qual",
        meas_denom_dcid=None,
        meas_denom_unresolved=True,
        slot_keys=(),
        slot_taxonomy={},
        representative_score=0.5,
    ),
    # 2: no qualifier, denominator, or constraint slots.
    ShapeDraft(
        shape_id="s2",
        label="Plain no-constraint shape",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        slot_taxonomy={},
        representative_score=1.0,
    ),
    # 3: empty DCIDs, two untruncated slots, and two sample DCIDs.
    ShapeDraft(
        shape_id="s3",
        label="Multi-slot shape",
        pop_type_dcid="",
        meas_prop_dcid="",
        stat_type_dcid="",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(
            SlotKeyDraft(axis="what", property_dcid="p:a", label="A slot"),
            SlotKeyDraft(axis="how", property_dcid="p:b", label="B slot"),
        ),
        slot_taxonomy={"what:p:a": ["x", "y"], "how:p:b": ["z"]},
        sv_arc_facts={"dc/sv7": {}, "dc/sv8": {}},
        representative_score=0.1,
    ),
    # 4: qualifier and denominator present, with one slot and no samples.
    ShapeDraft(
        shape_id="s4",
        label="Ratio with named denominator",
        pop_type_dcid="Country",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="rate_qual",
        meas_denom_dcid="GiniIndex",
        slot_keys=(SlotKeyDraft(axis="what", property_dcid="p:c", label="C slot"),),
        slot_taxonomy={"what:p:c": ["only"]},
        representative_score=0.75,
    ),
    # 5: `sv_arc_facts=None` tests the second empty-sample form.
    ShapeDraft(
        shape_id="s5",
        label="No sample dcids, explicit None",
        pop_type_dcid="Country",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid="Count_Household",
        slot_keys=(SlotKeyDraft(axis="what", property_dcid="p:d", label="D slot"),),
        slot_taxonomy={"what:p:d": ["one", "two", "three"]},
        sv_arc_facts=None,
        representative_score=0.333,
    ),
    # 6: three untruncated constraint slots.
    ShapeDraft(
        shape_id="s6",
        label="Three-slot shape",
        pop_type_dcid="Country",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(
            SlotKeyDraft(axis="what", property_dcid="p:e", label="E slot"),
            SlotKeyDraft(axis="how", property_dcid="p:f", label="F slot"),
            SlotKeyDraft(axis="where", property_dcid="p:g", label="G slot"),
        ),
        slot_taxonomy={"what:p:e": ["a"], "how:p:f": ["b"], "where:p:g": ["c"]},
        representative_score=0.001,
    ),
    # 7: qualifier only.
    ShapeDraft(
        shape_id="s7",
        label="Qualifier only",
        pop_type_dcid="Country",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="final_qual",
        meas_denom_dcid=None,
        slot_keys=(),
        slot_taxonomy={},
        representative_score=0.999,
    ),
    # 8: an unresolved denominator with a DCID.
    ShapeDraft(
        shape_id="s8",
        label="Dangling named denominator",
        pop_type_dcid="Country",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid="Count_Person",
        meas_denom_unresolved=True,
        slot_keys=(),
        slot_taxonomy={},
        representative_score=0.7,
    ),
]


# The menu's denominator text per shape: a schema term by its meaning, resolved or not.
_MENU_DENOMINATORS = {
    "s0": "per person",
    "s1": "(present, unresolved)",
    "s4": "GiniIndex",
    "s5": "Count_Household",
    "s8": "per person",
}


def _expected_five_tuple(shape: ShapeDraft) -> dict[str, str | None]:
    """Build the five-tuple text representation emitted by the prompt builder."""
    denominator = _MENU_DENOMINATORS.get(shape.shape_id)
    return {
        "population_type": shape.pop_type_dcid or None,
        "measured_property": shape.meas_prop_dcid or None,
        "stat_type": shape.stat_type_dcid or None,
        "measurement_qualifier": shape.meas_qual_dcid,
        "measurement_denominator": denominator,
    }


def _expected_sample_dcids(shape: ShapeDraft) -> tuple[str, ...]:
    """Return the first five sample DCIDs in insertion order."""
    if not shape.sv_arc_facts:
        return ()
    return tuple(list(shape.sv_arc_facts)[:5])


@pytest.fixture
def prompt() -> str:
    return _build_universal_bind_prompt(VARIABLE, SHAPES)


def _framed_prompt(shapes: list[ShapeDraft]) -> str:
    """A prompt with the frame block, an availability line on each shape, and a
    not-electable list."""
    frame = door_frame(
        door="parsed",
        variable=VARIABLE,
        entities=["Ghana"],
        entity_roles={},
        member_entities=frozenset(),
        pins={},
        date_request=None,
        unit_text=None,
        scope=None,
        operations=(),
        open_axes=frozenset(),
    )
    refused = [
        Interpretation(shape=SHAPES[2], reason="qualifier_not_bound", pinned_failure="pinned_code")
    ]
    availability = [(True, False, None)[i % 3] for i in range(len(shapes))]
    return _build_universal_bind_prompt(
        VARIABLE, shapes, frame=frame, refused=refused, availability=availability
    )


@pytest.fixture
def prompt_with_fixed_context() -> str:
    return _build_universal_bind_prompt(
        VARIABLE, SHAPES, admitted_qualifiers=[ADMITTED_QUALIFIERS] * len(SHAPES)
    )


def test_parse_menu_recovers_the_variable_phrase(prompt: str) -> None:
    variable, _entries = parse_menu(prompt)
    assert variable == VARIABLE


def test_parse_menu_recovers_one_entry_per_shape_in_order(prompt: str) -> None:
    _variable, entries = parse_menu(prompt)
    assert len(entries) == len(SHAPES)
    assert [entry.index for entry in entries] == list(range(len(SHAPES)))
    assert [entry.label for entry in entries] == [shape.label for shape in SHAPES]


def test_parse_menu_recovers_the_five_tuple_for_every_shape(prompt: str) -> None:
    _variable, entries = parse_menu(prompt)
    for entry, shape in zip(entries, SHAPES, strict=True):
        assert entry.five_tuple == _expected_five_tuple(shape), shape.label


def test_parse_menu_recovers_sample_dcids_for_every_shape(prompt: str) -> None:
    _variable, entries = parse_menu(prompt)
    for entry, shape in zip(entries, SHAPES, strict=True):
        assert entry.sample_dcids == _expected_sample_dcids(shape), shape.label


def test_parse_menu_reads_a_retrieval_score_only_from_a_capture_that_lists_one(
    prompt: str,
) -> None:
    _variable, entries = parse_menu(prompt)
    assert [entry.retrieval_score for entry in entries] == [None] * len(SHAPES)
    denominator_line = "    measurementQualifier=(none), measurementDenominator=per person\n"
    captured = prompt.replace(denominator_line, f"{denominator_line}  retrieval_score: 0.912\n", 1)
    _variable, entries = parse_menu(captured)
    assert entries[0].retrieval_score == 0.912


@pytest.mark.parametrize("shapes", [SHAPES, []], ids=["shapes", "empty-menu"])
def test_parse_menu_skips_the_frame_and_not_electable_blocks(shapes: list[ShapeDraft]) -> None:
    """An empty menu puts the NOT ELECTABLE block straight after the count header's blank
    line, as the refused-menu bind sends it."""
    variable, entries = parse_menu(_framed_prompt(shapes))
    assert variable == VARIABLE
    assert [entry.label for entry in entries] == [shape.label for shape in shapes]


@pytest.mark.parametrize("k", range(1, len(SHAPES)))
def test_truncate_menu_keeps_the_frame_and_not_electable_blocks(k: int) -> None:
    assert truncate_menu(_framed_prompt(SHAPES), k) == _framed_prompt(SHAPES[:k])


def test_parse_menu_handles_the_present_unresolved_denominator_branch(prompt: str) -> None:
    _variable, entries = parse_menu(prompt)
    assert entries[1].five_tuple["measurement_denominator"] == "(present, unresolved)"


def test_parse_menu_handles_a_shape_with_no_constraint_slots(prompt: str) -> None:
    assert "  constraint_slots: (none)" in prompt
    _variable, entries = parse_menu(prompt)
    assert entries[2].label == "Plain no-constraint shape"


def test_parse_menu_handles_the_taxonomy_truncation_marker(prompt: str) -> None:
    assert "(+5 more)" in prompt
    _variable, entries = parse_menu(prompt)
    assert entries[0].label == "GDP per capita"


def test_parse_menu_handles_the_qualifier_and_denominator_both_present(prompt: str) -> None:
    _variable, entries = parse_menu(prompt)
    assert entries[4].five_tuple["measurement_qualifier"] == "rate_qual"
    assert entries[4].five_tuple["measurement_denominator"] == "GiniIndex"


@pytest.mark.parametrize("k", range(1, len(SHAPES) + 1))
def test_truncate_menu_equals_rebuilding_from_the_truncated_shape_list(prompt: str, k: int) -> None:
    rebuilt = _build_universal_bind_prompt(VARIABLE, SHAPES[:k])
    assert truncate_menu(prompt, k) == rebuilt


@pytest.mark.parametrize("k", range(1, len(SHAPES) + 1))
def test_truncate_menu_output_itself_parses_back_to_k_entries(prompt: str, k: int) -> None:
    variable, entries = parse_menu(truncate_menu(prompt, k))
    assert variable == VARIABLE
    assert len(entries) == k


def test_truncate_menu_rewrites_the_header_count(prompt: str) -> None:
    truncated = truncate_menu(prompt, 3)
    assert "CANDIDATE SHAPES (3 total):" in truncated
    assert f"CANDIDATE SHAPES ({len(SHAPES)} total):" not in truncated


def test_truncate_menu_rejects_k_below_one(prompt: str) -> None:
    with pytest.raises(BindMenuError):
        truncate_menu(prompt, 0)


def test_truncate_menu_returns_the_prompt_unchanged_when_k_equals_the_shape_count(
    prompt: str,
) -> None:
    assert truncate_menu(prompt, len(SHAPES)) == prompt


def test_truncate_menu_returns_the_prompt_unchanged_when_k_exceeds_the_shape_count(
    prompt: str,
) -> None:
    # A short menu remains unchanged when `k` exceeds its shape count.
    one_shape_prompt = _build_universal_bind_prompt(VARIABLE, SHAPES[:1])
    assert truncate_menu(one_shape_prompt, 8) == one_shape_prompt


def test_parse_menu_requires_exactly_one_trailing_newline(prompt: str) -> None:
    with pytest.raises(BindMenuError):
        parse_menu(prompt.rstrip("\n"))
    with pytest.raises(BindMenuError):
        parse_menu(prompt + "\n")


def test_parse_menu_raises_on_an_unrecognised_line(prompt: str) -> None:
    corrupted = prompt.replace("  constraint_slots: (none)", "  constraint slots: none")
    with pytest.raises(BindMenuError, match="constraint slots"):
        parse_menu(corrupted)


def test_parse_menu_raises_when_the_header_undercounts_the_real_shapes(prompt: str) -> None:
    # Reject a leftover shape block when the header undercounts.
    undercounted = prompt.replace(
        f"CANDIDATE SHAPES ({len(SHAPES)} total):", f"CANDIDATE SHAPES ({len(SHAPES) - 1} total):"
    )
    with pytest.raises(BindMenuError):
        parse_menu(undercounted)


def test_parse_menu_raises_when_the_header_overcounts_the_real_shapes(prompt: str) -> None:
    # Reject a header that names a shape block absent from the prompt.
    overcounted = prompt.replace(
        f"CANDIDATE SHAPES ({len(SHAPES)} total):", f"CANDIDATE SHAPES ({len(SHAPES) + 1} total):"
    )
    with pytest.raises(BindMenuError):
        parse_menu(overcounted)


def test_prompt_with_fixed_context_actually_contains_the_block(
    prompt_with_fixed_context: str,
) -> None:
    # Confirm the builder emits both fixed-context entries before parsing.
    assert "FIXED CONTEXT (" in prompt_with_fixed_context
    assert "property_dcid=p:unit, value_dcid=USDollar (US Dollar)" in prompt_with_fixed_context
    assert "property_dcid=p:a, value_dcid=Annual (Annual)" in prompt_with_fixed_context


def test_parse_menu_round_trips_a_prompt_with_a_fixed_context_block(
    prompt_with_fixed_context: str,
) -> None:
    variable, entries = parse_menu(prompt_with_fixed_context)
    assert variable == VARIABLE
    assert len(entries) == len(SHAPES)
    assert [entry.label for entry in entries] == [shape.label for shape in SHAPES]


@pytest.mark.parametrize("k", range(1, len(SHAPES) + 1))
def test_truncate_menu_with_fixed_context_equals_rebuilding_from_the_truncated_shape_list(
    prompt_with_fixed_context: str, k: int
) -> None:
    rebuilt = _build_universal_bind_prompt(
        VARIABLE, SHAPES[:k], admitted_qualifiers=[ADMITTED_QUALIFIERS] * k
    )
    assert truncate_menu(prompt_with_fixed_context, k) == rebuilt


def test_truncate_menu_preserves_the_fixed_context_block(
    prompt_with_fixed_context: str,
) -> None:
    # Shape 3 is the first to list the `p:a` slot that `Annual` fixes.
    truncated = truncate_menu(prompt_with_fixed_context, 4)
    assert "FIXED CONTEXT (" in truncated
    assert "property_dcid=p:unit, value_dcid=USDollar (US Dollar)" in truncated
    assert "property_dcid=p:a, value_dcid=Annual (Annual)" in truncated


# Dropped-qualifier block parsing and truncation.


@pytest.fixture
def prompt_with_dropped_qualifiers() -> str:
    return _build_universal_bind_prompt(
        VARIABLE, SHAPES, dropped_qualifier_phrases=DROPPED_QUALIFIER_PHRASES
    )


def test_prompt_with_dropped_qualifiers_actually_contains_the_block(
    prompt_with_dropped_qualifiers: str,
) -> None:
    # Confirm the prompt contains the dropped-qualifier block before parsing.
    assert "DROPPED QUALIFIERS (" in prompt_with_dropped_qualifiers
    for phrase in DROPPED_QUALIFIER_PHRASES:
        assert phrase in prompt_with_dropped_qualifiers


def test_parse_menu_round_trips_a_prompt_with_a_dropped_qualifiers_block(
    prompt_with_dropped_qualifiers: str,
) -> None:
    variable, entries = parse_menu(prompt_with_dropped_qualifiers)
    assert variable == VARIABLE
    assert len(entries) == len(SHAPES)
    assert [entry.label for entry in entries] == [shape.label for shape in SHAPES]


@pytest.mark.parametrize("k", range(1, len(SHAPES) + 1))
def test_truncate_menu_with_dropped_qualifiers_equals_rebuilding_from_the_truncated_shape_list(
    prompt_with_dropped_qualifiers: str, k: int
) -> None:
    rebuilt = _build_universal_bind_prompt(
        VARIABLE, SHAPES[:k], dropped_qualifier_phrases=DROPPED_QUALIFIER_PHRASES
    )
    assert truncate_menu(prompt_with_dropped_qualifiers, k) == rebuilt


def test_truncate_menu_preserves_the_dropped_qualifiers_block(
    prompt_with_dropped_qualifiers: str,
) -> None:
    truncated = truncate_menu(prompt_with_dropped_qualifiers, 3)
    assert "DROPPED QUALIFIERS (" in truncated
    for phrase in DROPPED_QUALIFIER_PHRASES:
        assert phrase in truncated


def test_parse_menu_round_trips_a_prompt_with_both_fixed_context_and_dropped_qualifiers() -> None:
    prompt = _build_universal_bind_prompt(
        VARIABLE,
        SHAPES,
        admitted_qualifiers=[ADMITTED_QUALIFIERS] * len(SHAPES),
        dropped_qualifier_phrases=DROPPED_QUALIFIER_PHRASES,
    )
    variable, entries = parse_menu(prompt)
    assert variable == VARIABLE
    assert len(entries) == len(SHAPES)


# Index-sourced label lines in a built prompt.


@pytest.fixture
def prompt_with_index_sourced_labels(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(bind_module, "get_index", _synthetic_flat_index)
    return _build_universal_bind_prompt(VARIABLE, [_INDEX_SOURCED_SHAPE])


def test_prompt_with_index_sourced_shape_actually_contains_a_labels_line(
    prompt_with_index_sourced_labels: str,
) -> None:
    # A control case for the test below: a parse_menu bug that silently skips
    # the labels line cannot pass by vacuity.
    assert "      labels: FlatA = A; FlatB = B" in prompt_with_index_sourced_labels


def test_parse_menu_round_trips_an_index_sourced_shapes_labels_line(
    prompt_with_index_sourced_labels: str,
) -> None:
    variable, entries = parse_menu(prompt_with_index_sourced_labels)
    assert variable == VARIABLE
    assert len(entries) == 1
    assert entries[0].label == _INDEX_SOURCED_SHAPE.label
