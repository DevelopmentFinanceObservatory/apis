"""Pins identity-slot translation, including value widening, sets, and exact bindings."""

from __future__ import annotations

from statvar_members.enumerate import slot_bindings_from_identity


def _no_widening(ref: str) -> tuple[str, ...]:
    """Stand-in for `CodeIndex.descendants` over a leaf code, a flat code
    system, or a dcid this index carries no code row for: `descendants(ref)`
    is `(ref,)` alone, a genuine no-op, in every one of those cases."""
    return (ref,)


def test_value_binding_that_does_not_widen_stays_a_bare_value_binding():
    slots_json = [{"property": "crsTypeOfFlow", "binding": {"state": "value", "ref": "Flow10"}}]

    result = slot_bindings_from_identity(slots_json, widen=_no_widening)

    assert result == {"crsTypeOfFlow": ("value", "Flow10")}


def test_value_binding_that_widens_becomes_a_set_binding():
    def widen(ref: str) -> tuple[str, ...]:
        assert ref == "CRSTypeOfFinance100"
        return ("CRSTypeOfFinance100", "CRSTypeOfFinance110", "CRSTypeOfFinance210")

    slots_json = [
        {
            "property": "crsTypeOfFinance",
            "binding": {"state": "value", "ref": "CRSTypeOfFinance100"},
        }
    ]

    result = slot_bindings_from_identity(slots_json, widen=widen)

    assert result == {
        "crsTypeOfFinance": (
            "set",
            ("CRSTypeOfFinance100", "CRSTypeOfFinance110", "CRSTypeOfFinance210"),
        )
    }


def test_set_binding_widens_each_ref_and_dedupes_first_seen_but_stays_a_set():
    # `refs` shares a widened descendant across two of its own members
    # (`CRSFinanceLeafShared`) -- the widened tuple keeps only the first
    # occurrence, in `refs` order.
    widened_by_ref = {
        "CRSTypeOfFinance100": ("CRSTypeOfFinance100", "CRSFinanceLeafShared", "CRSFinanceLeafA"),
        "CRSTypeOfFinance200": ("CRSTypeOfFinance200", "CRSFinanceLeafShared"),
    }
    slots_json = [
        {
            "property": "crsTypeOfFinance",
            "binding": {"state": "set", "refs": ["CRSTypeOfFinance100", "CRSTypeOfFinance200"]},
        }
    ]

    result = slot_bindings_from_identity(slots_json, widen=lambda ref: widened_by_ref[ref])

    assert result == {
        "crsTypeOfFinance": (
            "set",
            (
                "CRSTypeOfFinance100",
                "CRSFinanceLeafShared",
                "CRSFinanceLeafA",
                "CRSTypeOfFinance200",
            ),
        )
    }


def test_exact_binding_maps_to_a_bare_value_binding_and_never_calls_widen():
    """`exact(ref)` denotes `ref` alone, never its subtree -- unlike a
    `"value"` binding, `widen` must never be called for it. This is the one
    place `exact` is read rather than refused (see `check_no_exact_bindings`
    in `search.engine.identity_edit`, which refuses it at every Search
    door)."""
    slots_json = [
        {
            "property": "crsPurpose",
            "binding": {"state": "exact", "ref": "CRSPurposeCode110"},
        }
    ]

    def unreachable_widen(ref: str) -> tuple[str, ...]:
        raise AssertionError("an exact slot must never call widen")

    result = slot_bindings_from_identity(slots_json, widen=unreachable_widen)

    assert result == {"crsPurpose": ("value", "CRSPurposeCode110")}


def test_set_binding_that_does_not_widen_keeps_its_own_refs_unchanged():
    slots_json = [
        {"property": "age", "binding": {"state": "set", "refs": ["Years0", "Years5"]}},
    ]

    result = slot_bindings_from_identity(slots_json, widen=_no_widening)

    assert result == {"age": ("set", ("Years0", "Years5"))}


def test_set_binding_of_one_ref_maps_to_the_binding_its_value_spelling_maps_to():
    """The spec kit hashes a one-ref set and a value to one `spec_id`, so both spellings,
    and a set that repeats one ref, translate to one binding."""

    def widen(ref: str) -> tuple[str, ...]:
        return (ref, f"{ref}Child") if ref == "CRSTypeOfFinance100" else (ref,)

    def translate(binding: dict) -> dict:
        return slot_bindings_from_identity([{"property": "p", "binding": binding}], widen=widen)

    for ref in ("Flow10", "CRSTypeOfFinance100"):
        as_value = translate({"state": "value", "ref": ref})
        assert translate({"state": "set", "refs": [ref]}) == as_value
        assert translate({"state": "set", "refs": [ref, ref]}) == as_value
    assert translate({"state": "set", "refs": ["Flow10"]}) == {"p": ("value", "Flow10")}


def test_unbound_state_passes_through_regardless_of_widen():
    slots_json = [{"property": "residenceStatus", "binding": {"state": "unbound"}}]

    def unreachable_widen(ref: str) -> tuple[str, ...]:
        raise AssertionError("an unbound slot must never call widen")

    result = slot_bindings_from_identity(slots_json, widen=unreachable_widen)

    assert result == {"residenceStatus": ("unbound",)}


def test_missing_binding_state_is_treated_as_unbound():
    slots_json = [{"property": "residenceStatus", "binding": {}}]

    result = slot_bindings_from_identity(slots_json, widen=_no_widening)

    assert result == {"residenceStatus": ("unbound",)}


def test_translates_several_slots_independently_in_one_call():
    def widen(ref: str) -> tuple[str, ...]:
        return (ref, f"{ref}Child") if ref == "CRSTypeOfFinance100" else (ref,)

    slots_json = [
        {
            "property": "crsTypeOfFinance",
            "binding": {"state": "value", "ref": "CRSTypeOfFinance100"},
        },
        {"property": "crsTypeOfFlow", "binding": {"state": "value", "ref": "CRSTypeOfFlow10"}},
        {"property": "age", "binding": {"state": "set", "refs": ["Years0", "Years5"]}},
        {"property": "residenceStatus", "binding": {"state": "unbound"}},
    ]

    result = slot_bindings_from_identity(slots_json, widen=widen)

    assert result == {
        "crsTypeOfFinance": ("set", ("CRSTypeOfFinance100", "CRSTypeOfFinance100Child")),
        "crsTypeOfFlow": ("value", "CRSTypeOfFlow10"),
        "age": ("set", ("Years0", "Years5")),
        "residenceStatus": ("unbound",),
    }
