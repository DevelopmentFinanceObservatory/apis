"""Pins layer-2 signal values and guards against adding unkeyed state readers."""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import pytest

from search.engine.response_cache_flow import Layer2Signals, compute_layer2_signals
from search.engine.role_text import DoorRole

_REGION_DIR = Path(__file__).parents[2] / "src" / "search" / "engine" / "region"


def _count_state_attribute_reads(attr: str) -> int:
    """Count loaded `state.<attr>` references across the region package."""
    count = 0
    for path in sorted(_REGION_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name)
                and node.value.id == "state"
                and node.attr == attr
            ):
                count += 1
    return count


def test_query_text_and_role_query_reader_counts_are_covered() -> None:
    """Pin the 7 `query_text` readers and 1 `role_query` reader covered by cache signals."""
    query_text_reads = _count_state_attribute_reads("query_text")
    role_query_reads = _count_state_attribute_reads("role_query")
    assert query_text_reads == 7, (
        f"state.query_text has {query_text_reads} readers in src/search/engine/region/, "
        "not the 7 compute_layer2_signals computes signals for. Update "
        "compute_layer2_signals (and this test) to cover the new reader."
    )
    assert role_query_reads == 1, (
        f"state.role_query has {role_query_reads} readers in src/search/engine/region/, "
        "not the 1 compute_layer2_signals computes a signal for. Update "
        "compute_layer2_signals (and this test) to cover the new reader."
    )


def test_no_role_bearing_entities_yields_no_role_contradictions() -> None:
    signals = compute_layer2_signals(query="gdp in kenya", entity_roles={}, query_text=None)
    assert signals.role_contradictions == ()


def test_supporting_preposition_does_not_contradict() -> None:
    """ "aid to Kenya": "to" immediately precedes the destination-axis span,
    supporting the door-supplied axis."""
    entity_roles = {"Kenya": [DoorRole(axis="destination", span="Kenya")]}
    signals = compute_layer2_signals(
        query="aid to Kenya", entity_roles=entity_roles, query_text=None
    )
    assert signals.role_contradictions == (
        {"name": "Kenya", "span": "Kenya", "axis": "destination", "contradicts": False},
    )


def test_opposing_preposition_contradicts() -> None:
    """ "aid from Kenya": "from" contradicts the door-supplied destination
    axis, with no supporting occurrence anywhere in the query."""
    entity_roles = {"Kenya": [DoorRole(axis="destination", span="Kenya")]}
    signals = compute_layer2_signals(
        query="aid from Kenya", entity_roles=entity_roles, query_text=None
    )
    assert signals.role_contradictions == (
        {"name": "Kenya", "span": "Kenya", "axis": "destination", "contradicts": True},
    )


def test_role_contradictions_cover_every_role_bearing_entity() -> None:
    """Return one contradiction verdict per role-bearing entity, in input order."""
    entity_roles = {
        "Kenya": [DoorRole(axis="destination", span="Kenya")],
        "Norway": [DoorRole(axis="origin", span="Norway")],
    }
    signals = compute_layer2_signals(
        query="aid from Norway to Kenya", entity_roles=entity_roles, query_text=None
    )
    assert signals.role_contradictions == (
        {"name": "Kenya", "span": "Kenya", "axis": "destination", "contradicts": False},
        {"name": "Norway", "span": "Norway", "axis": "origin", "contradicts": False},
    )


def test_role_contradictions_cover_every_occurrence_of_one_name() -> None:
    """Keep both role occurrences when one entity appears on both axes."""
    entity_roles = {
        "Kenya": [
            DoorRole(axis="origin", span="Kenya"),
            DoorRole(axis="destination", span="Kenya"),
        ],
    }
    signals = compute_layer2_signals(
        query="aid from Kenya to Kenya", entity_roles=entity_roles, query_text=None
    )
    assert signals.role_contradictions == (
        {"name": "Kenya", "span": "Kenya", "axis": "origin", "contradicts": False},
        {"name": "Kenya", "span": "Kenya", "axis": "destination", "contradicts": False},
    )


def test_query_text_none_yields_no_qualifier_or_unsupported_signals() -> None:
    """A missing raw-text signal yields no qualifier or default-trigger signals."""
    signals = compute_layer2_signals(
        query="funding for agriculture with non-concessional terms",
        entity_roles={},
        query_text=None,
    )
    assert signals.query_text_qualifier_matches == ()
    assert signals.query_text_qualifier_ties == ()
    assert signals.query_text_qualifier_readings == ()
    assert signals.unsupported_qualifier_phrases == ()
    assert signals.declared_default_trigger_words == ()
    assert signals.declared_default_slots_named == ()


def test_query_text_qualifier_match_against_the_real_index() -> None:
    """The index maps the word "agriculture" to `CRSPurposeCode311`."""
    signals = compute_layer2_signals(
        query="funding for agriculture in Ethiopia", entity_roles={}, query_text="agriculture"
    )
    assert len(signals.query_text_qualifier_matches) == 1
    match = signals.query_text_qualifier_matches[0]
    assert match["code_dcid"] == "CRSPurposeCode311"
    assert match["matched_text"] == "agriculture"


def test_query_text_qualifier_tie_against_the_real_index() -> None:
    """The index maps "budget support" to codes in separate systems: two CRS ones and two
    DAC1 aid types. The cache signals read the index before any shape exists."""
    signals = compute_layer2_signals(
        query="budget support", entity_roles={}, query_text="budget support"
    )
    assert len(signals.query_text_qualifier_ties) == 1
    tie = signals.query_text_qualifier_ties[0]
    assert tie["matched_text"] == "budget support"
    assert {row["code_dcid"] for row in tie["rows"]} == {
        "CRSCooperationModalityA",
        "CRSPurposeCode510",
        "DAC1AidType1100",
        "DAC1AidType2210",
    }


def test_query_text_qualifier_reading_against_the_real_index() -> None:
    """ "PPP" names one code row and one measure value, so `match_qualifiers` refuses it and
    only the two-reading signal records it."""
    signals = compute_layer2_signals(
        query="GDP PPP in Kenya", entity_roles={}, query_text="GDP PPP in Kenya"
    )
    assert signals.query_text_qualifier_matches == ()
    # A match of the query text carries no offsets.
    match = {"source": "synonym", "matched_text": "PPP", "start": None, "end": None}
    assert signals.query_text_qualifier_readings == (
        {
            "matched_text": "PPP",
            "code": {
                **match,
                "code_system": "crsPurpose",
                "code_dcid": "CRSPurposeCode25010",
                "label": "business policy and administration",
            },
            "measure": {
                **match,
                "code_system": "measurementQualifier",
                "code_dcid": "PurchasingPowerParity",
                "label": "purchasing power parity",
            },
        },
    )


def test_unsupported_qualifier_phrase_against_the_real_index() -> None:
    """ "non-concessional" is qualifier_match's own closed unsupported-term
    set, found in `query_text`."""
    signals = compute_layer2_signals(
        query="non-concessional loans", entity_roles={}, query_text="non-concessional loans"
    )
    assert signals.unsupported_qualifier_phrases == ("non-concessional",)


def test_query_text_matches_no_vocabulary_yields_empty_signals() -> None:
    """An ordinary query_text with no code-index or unsupported vocabulary
    match yields empty tuples, not an error."""
    signals = compute_layer2_signals(
        query="fertility rate in Kenya", entity_roles={}, query_text="fertility rate in Kenya"
    )
    assert signals.query_text_qualifier_matches == ()
    assert signals.query_text_qualifier_ties == ()
    assert signals.query_text_qualifier_readings == ()
    assert signals.unsupported_qualifier_phrases == ()


def test_declared_default_trigger_word_against_the_real_index() -> None:
    """The real index declares crsTypeOfFlow's CRSTypeOfFlow10 default with
    `when_named=("aid",)`; a raw query naming "aid" carries that word, and
    one naming it only negated ("non-aid") carries nothing, the same
    unnegated whole-word test `apply_declared_defaults` applies."""
    named = compute_layer2_signals(
        query="How much aid did Kenya get?",
        entity_roles={},
        query_text="How much aid did Kenya get?",
    )
    negated = compute_layer2_signals(
        query="non-aid financing to Kenya",
        entity_roles={},
        query_text="non-aid financing to Kenya",
    )
    assert named.declared_default_trigger_words == (
        {"code_system": "crsTypeOfFlow", "word": "aid"},
    )
    assert negated.declared_default_trigger_words == ()


def test_declared_default_slot_named_against_the_real_index() -> None:
    """The real index declares a default on crsTypeOfFlow, whose plain name is "flow";
    a raw query naming "flows" carries it, and one naming no slot carries nothing."""
    named = compute_layer2_signals(
        query="aid flows to Kenya", entity_roles={}, query_text="aid flows to Kenya"
    )
    unnamed = compute_layer2_signals(
        query="aid to Kenya", entity_roles={}, query_text="aid to Kenya"
    )
    assert named.declared_default_slots_named == ("crsTypeOfFlow",)
    assert unnamed.declared_default_slots_named == ()


@pytest.mark.parametrize(
    ("query_text", "expected"),
    [
        ("not bilateral ODA to Kenya", ("bilateral",)),
        ("non-bilateral ODA to Kenya", ("bilateral",)),
        ("bilateral ODA to Kenya", ()),
        (None, ()),
    ],
)
def test_negated_stated_words_against_the_real_index(
    query_text: str | None, expected: tuple[str, ...]
) -> None:
    """The real index has the CRS state "bilateral" of its rows; a raw query negating it
    carries the word, and one stating it, or a request of several variables, carries none."""
    signals = compute_layer2_signals(query=query_text or "", entity_roles={}, query_text=query_text)
    assert signals.query_text_negated_stated_words == expected


def test_layer2_signals_carries_one_field_per_numbered_signal() -> None:
    """The dataclass has one field for each of the ten cache signals."""
    field_names = {f.name for f in fields(Layer2Signals)}
    assert field_names == {
        "role_contradictions",
        "query_text_qualifier_matches",
        "query_text_qualifier_ties",
        "query_text_qualifier_readings",
        "query_text_qualifier_occurrences",
        "unsupported_qualifier_phrases",
        "declared_default_trigger_words",
        "declared_default_slots_named",
        "written_variable_phrases",
        "query_text_negated_stated_words",
    }


def test_each_variable_carries_the_stretch_of_the_query_that_writes_it() -> None:
    """Two wordings that extract alike key apart when they write the phrase differently,
    and a request with no query text carries none."""

    def written(query: str, query_text: str | None) -> tuple[str | None, ...]:
        return compute_layer2_signals(
            query=query,
            entity_roles={},
            query_text=query_text,
            variables=["external debt net flows"],
        ).written_variable_phrases

    assert written(
        "net flows on external debt for Kenya", "net flows on external debt for Kenya"
    ) == ("net flows on external debt",)
    assert written("Kenya external debt net flows", "Kenya external debt net flows") == (
        "external debt net flows",
    )
    assert written("net flows on external debt for Kenya and Ghana", None) == ()
