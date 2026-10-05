"""`bound_values_from_bindings` covers empty values, missing properties, excluded kinds, and
collisions where the later binding wins. The family memo (`_resolved_family`) keeps one entry
per deciding query and stays within its row bound.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pytest

from code_index.members import FamilyQuery
from code_index.shared import get_index
from search.engine.bind import DraftBindingEvidence, DraftBindingKind, SlotBindingDraft
from search.engine.retrieval import bound_values
from search.engine.retrieval.bound_values import _resolved_family, bound_values_from_bindings
from statvar_members import FiveTuple, SlotBinding


def _binding(
    prop: str | None,
    values: list[str],
    *,
    kind: DraftBindingKind = "value",
    evidence: DraftBindingEvidence = "mentioned",
) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how",
        property_dcid=prop,
        kind=kind,
        value_dcids=values,
        evidence=evidence,
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
# Education ODA disbursements, modality unbound: 15 rows of the packaged index.
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
