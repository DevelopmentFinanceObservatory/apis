"""`bound_values_from_bindings` covers empty values, missing properties, excluded kinds, and
collisions where the later binding wins. The family memo (`_resolved_family`) keeps one entry
per deciding query and stays within its row bound.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pytest

from code_index import members
from code_index.members import DatasetScope, FamilyQuery, family_query_digest
from code_index.shared import get_index
from search.engine.bind import DraftBindingEvidence, DraftBindingKind, SlotBindingDraft
from search.engine.retrieval import bound_values
from search.engine.retrieval.bound_values import (
    _resolved_family,
    bound_values_from_bindings,
)
from statvar_members import FiveTuple, SlotBinding


def _binding(
    prop: str | None,
    values: list[str],
    *,
    kind: DraftBindingKind = "value",
    evidence: DraftBindingEvidence = "mentioned",
) -> SlotBindingDraft:
    """A draft, built unvalidated: the bind response refuses a value or set draft naming no
    dcid at parse, and these rows hold the reader to drafts code builds without that check."""
    return SlotBindingDraft.model_construct(
        axis="how",
        property_dcid=prop,
        kind=kind,
        value_dcids=values,
        evidence=evidence,
        requested_all_phrase=None,
    )


def _reference_standard_dict_comprehension(
    bindings: Sequence[SlotBindingDraft],
) -> dict[str, list[str]]:
    """standard.py's earlier implementation, reproduced independently (a dict
    comprehension, later binding wins the key on collision)."""
    return {
        b.property_dcid: list(b.value_dcids)
        for b in bindings
        if b.property_dcid and b.kind in ("value", "set") and b.value_dcids
    }


def _reference_role_resolve_inline_loop(
    bindings: Sequence[SlotBindingDraft],
) -> dict[str, list[str]]:
    """role_resolve.py's earlier implementation, reproduced independently (an
    explicit loop assigning into a dict, later binding wins the key on
    collision)."""
    bound_values: dict[str, list[str]] = {}
    for b in bindings:
        if b.property_dcid and b.kind in ("value", "set") and b.value_dcids:
            bound_values[b.property_dcid] = list(b.value_dcids)
    return bound_values


# ---------------------------------------------------------------------------
# Every case must agree across all three implementations.
# ---------------------------------------------------------------------------


def _assert_all_agree(bindings: list[SlotBindingDraft]) -> dict[str, list[str]]:
    shared = bound_values_from_bindings(bindings)
    assert shared == _reference_standard_dict_comprehension(bindings)
    assert shared == _reference_role_resolve_inline_loop(bindings)
    return shared


def test_empty_bindings_list():
    assert _assert_all_agree([]) == {}


def test_ordinary_value_binding():
    bindings = [_binding("concept", ["X"])]
    assert _assert_all_agree(bindings) == {"concept": ["X"]}


def test_ordinary_set_binding():
    bindings = [_binding("concept", ["X", "Y"], kind="set")]
    assert _assert_all_agree(bindings) == {"concept": ["X", "Y"]}


def test_empty_value_dcids_is_excluded():
    """An empty `value_dcids` names no value, despite being schema-valid."""
    bindings = [_binding("concept", [])]
    assert _assert_all_agree(bindings) == {}


def test_missing_property_dcid_is_excluded():
    """A constraint with `property_dcid=None` contributes no key, regardless of its values."""
    bindings = [_binding(None, ["X"])]
    assert _assert_all_agree(bindings) == {}


def test_kind_unbound_is_excluded():
    bindings = [_binding("concept", ["X"], kind="unbound", evidence="mentioned")]
    # Unbound drafts have empty values. The predicate excludes by kind alone.
    assert _assert_all_agree(bindings) == {}


def test_kind_absent_is_excluded():
    bindings = [_binding("concept", ["X"], kind="absent")]
    assert _assert_all_agree(bindings) == {}


def test_two_bindings_same_property_second_wins():
    """The collision case: two bindings naming the same property_dcid. Both
    reference implementations build the map with an ordinary dict/dict-comp
    keyed assignment, so the later binding's value_dcids silently overwrites
    the earlier one's."""
    bindings = [
        _binding("concept", ["X"]),
        _binding("concept", ["Y"]),
    ]
    assert _assert_all_agree(bindings) == {"concept": ["Y"]}


def test_collision_with_an_excluded_binding_between_keeps_the_last_real_one():
    """An empty binding between two real bindings leaves the later value as the winner."""
    bindings = [
        _binding("concept", ["X"]),
        _binding("concept", []),
        _binding("concept", ["Z"]),
    ]
    assert _assert_all_agree(bindings) == {"concept": ["Z"]}


def test_mixed_realistic_binding_list():
    bindings = [
        _binding("concept", ["X"]),
        _binding(None, ["ignored"]),
        _binding("mode", [], kind="set"),
        _binding("scheme", ["A", "B"], kind="set"),
        _binding("concept", ["Y"], evidence="sole_option"),
    ]
    assert _assert_all_agree(bindings) == {"concept": ["Y"], "scheme": ["A", "B"]}


# ---------------------------------------------------------------------------
# The family memo keys on the deciding query and holds a bounded number of rows.
# ---------------------------------------------------------------------------

_EDU_ODA_SLOTS: dict[str, SlotBinding] = {
    "crsPurpose": ("value", "CRSPurposeCode11110"),
    "crsTypeOfFinance": ("value", "CRSTypeOfFinance110"),
    "crsTypeOfFlow": ("value", "CRSTypeOfFlow10"),
    "financialFlowType": ("value", "Disbursement"),
    "nettingBasis": ("value", "Gross"),
    "crsCooperationModality": ("unbound",),
}
# Education ODA disbursements, modality unbound: 15 rows of the fixture index.
_EDU_ODA = FamilyQuery(
    five_tuple=FiveTuple(
        population_type="FinancialFlow",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier="Nominal",
    ),
    slots=_EDU_ODA_SLOTS,
    source=None,
)
# The same with modality bound to project-type interventions: one row.
_EDU_ODA_PROJECTS = replace(
    _EDU_ODA,
    slots={**_EDU_ODA.slots, "crsCooperationModality": ("value", "CRSCooperationModalityC01")},
)


def test_an_unbound_slot_no_row_declares_shares_the_familys_memo_entry():
    """An options request may leave slots unbound that no row of the five-tuple declares.
    They do not change the family, so both queries read one memoised family."""
    index = get_index()
    widened = replace(_EDU_ODA, slots={**_EDU_ODA.slots, "undeclaredProperty": ("unbound",)})

    family = _resolved_family(index, _EDU_ODA)

    assert family is not None and len(family.rows) == 15
    assert _resolved_family(index, widened) is family


@pytest.mark.parametrize(
    ("headroom", "between", "kept"),
    [
        pytest.param(0, None, True, id="fits"),
        pytest.param(-1, None, False, id="larger-than-the-bound"),
        pytest.param(0, _EDU_ODA_PROJECTS, False, id="evicted-by-a-later-family"),
    ],
)
def test_the_family_memo_holds_at_most_its_row_bound(monkeypatch, headroom, between, kept):
    """With room for 15 rows the education family stays memoised. A bound below its size
    keeps it out, and a later one-row family evicts it."""
    monkeypatch.setattr(bound_values, "_FAMILY_CACHE_ROWS", 15 + headroom)
    index = get_index()

    first = _resolved_family(index, _EDU_ODA)
    if between is not None:
        _resolved_family(index, between)

    assert (_resolved_family(index, _EDU_ODA) is first) is kept


def test_a_memo_key_has_a_fixed_size_and_one_entry_per_query():
    """A key is a digest, so a query that binds thousands of refs costs the same 32 bytes
    as one that binds a single value, and different queries keep separate entries."""
    index = get_index()
    refs = tuple(f"CRSPurposeCode{n}" for n in range(10_000))
    huge = replace(_EDU_ODA, slots={**_EDU_ODA.slots, "crsPurpose": ("set", refs)})

    assert len(family_query_digest(huge)) == len(family_query_digest(_EDU_ODA)) == 32

    first = _resolved_family(index, _EDU_ODA)
    second = _resolved_family(index, _EDU_ODA_PROJECTS)
    memo = bound_values._index_family_cache[id(index)]

    assert first is not second
    assert len(memo.families) == 2
    assert {len(key) for key in memo.families} == {32}


@pytest.mark.parametrize(
    "other",
    [
        pytest.param(replace(_EDU_ODA, source="OECD.CRS"), id="source"),
        pytest.param(replace(_EDU_ODA, role_keys=frozenset({"origin"})), id="role-keys"),
        pytest.param(
            replace(_EDU_ODA, five_tuple=replace(_EDU_ODA.five_tuple, stat_type=None)),
            id="five-tuple",
        ),
        pytest.param(
            replace(_EDU_ODA, slots={**_EDU_ODA.slots, "nettingBasis": ("value", "Net")}),
            id="slot-value",
        ),
        pytest.param(
            replace(_EDU_ODA, slots={**_EDU_ODA.slots, "nettingBasis": ("set", ("Gross",))}),
            id="slot-binding-kind",
        ),
        pytest.param(
            replace(_EDU_ODA, scope=DatasetScope(allowed=frozenset({"OECD.CRS"}), named=False)),
            id="scope",
        ),
        pytest.param(
            replace(_EDU_ODA, scope=DatasetScope(allowed=frozenset({"OECD.CRS"}), named=True)),
            id="named-scope",
        ),
    ],
)
def test_queries_that_differ_in_any_field_get_different_memo_keys(other):
    reordered = replace(_EDU_ODA, slots=dict(reversed(_EDU_ODA.slots.items())))

    assert family_query_digest(reordered) == family_query_digest(_EDU_ODA)
    assert family_query_digest(other) != family_query_digest(_EDU_ODA)


def test_prepared_index_views_answer_a_first_request_without_scanning_the_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The app prepares the code index's views at startup, so the first resolve to read a
    five-tuple's rows or role-class facts, or whether the index carries a five-tuple, builds
    nothing."""
    index = get_index()
    for cache in (
        members._five_tuple_rows_cache,
        members._five_tuple_facts_cache,
        bound_values._index_five_tuples_cache,
    ):
        cache.pop(id(index), None)

    bound_values.prepare_index_views()

    def scan(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a prepared view scanned the index")

    monkeypatch.setattr(members, "row_five_tuple", scan)
    monkeypatch.setattr(members, "role_class_facts", scan)
    monkeypatch.setattr(bound_values, "IndexMemberFacts", scan)
    five_tuples = list(members._five_tuple_rows(index).rows)
    assert five_tuples
    for five_tuple in five_tuples:
        members._five_tuple_facts(index, five_tuple)
    assert bound_values._index_five_tuples(index)
