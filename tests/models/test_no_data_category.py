"""Checks `NoData.category` derivation and the resolution order for every reason code.

The table includes currency conversion's `rate_coverage_gap` and distinguishes unresolved entities
from absent data, including codes whose names contain `_not_`.
"""

from __future__ import annotations

from search.models import NO_DATA_RESOLUTION_ORDER, NoData, NoDataReason

# code -> expected category, covering every NoDataReason value.
_CATEGORY_BY_REASON: dict[NoDataReason, str] = {
    "source_not_resolved": "unresolved",
    "source_ambiguous": "unresolved",
    "source_code_not_held": "unresolved",
    "source_code_ambiguous": "unresolved",
    "source_code_shares_variable": "unresolved",
    "entity_not_resolved": "unresolved",
    "entity_ambiguous": "unresolved",
    "entity_role_not_admitted": "unresolved",
    "entity_set_too_large": "unresolved",  # refused before any place is read
    "variable_not_resolved": "unresolved",
    "role_not_resolved": "unresolved",
    "unit_not_resolved": "unresolved",
    "no_matching_slice": "absent",
    "binding_mismatch": "absent",
    "denominator_not_available": "absent",  # naive-predicate trap: contains "_not_"
    "no_observations": "absent",
    "facet_not_available": "absent",  # naive-predicate trap: contains "_not_"
    "role_incompatible": "absent",
    "role_signature_incompatible": "absent",
    "rate_coverage_gap": "absent",
}


def test_category_derivation_probe_table():
    for reason, expected_category in _CATEGORY_BY_REASON.items():
        assert NoData(reason=reason).category == expected_category, reason


def test_category_is_derived_not_settable():
    """A caller-supplied category is always overwritten by the validator, so no
    constructor call can produce a contradictory pair."""
    no_data = NoData(reason="entity_not_resolved", category="absent")  # type: ignore[arg-type]
    assert no_data.category == "unresolved"

    no_data = NoData(reason="no_observations", category="unresolved")  # type: ignore[arg-type]
    assert no_data.category == "absent"


def test_resolution_order_covers_every_reason_exactly_once():
    assert len(NO_DATA_RESOLUTION_ORDER) == len(set(NO_DATA_RESOLUTION_ORDER))
    assert set(NO_DATA_RESOLUTION_ORDER) == set(NoDataReason.__args__)


def test_resolution_order_role_position():
    """The two role-compatibility codes sit at the role position, per contract."""
    role_index = NO_DATA_RESOLUTION_ORDER.index("role_not_resolved")
    assert NO_DATA_RESOLUTION_ORDER[role_index + 1] == "role_incompatible"
    assert NO_DATA_RESOLUTION_ORDER[role_index + 2] == "role_signature_incompatible"


def test_resolution_order_entity_position():
    """entity_ambiguous, entity_set_too_large and entity_role_not_admitted sit
    directly after entity_not_resolved, in that order, per contract: the four
    run by how far the entity got."""
    entity_index = NO_DATA_RESOLUTION_ORDER.index("entity_not_resolved")
    assert NO_DATA_RESOLUTION_ORDER[entity_index + 1] == "entity_ambiguous"
    assert NO_DATA_RESOLUTION_ORDER[entity_index + 2] == "entity_set_too_large"
    assert NO_DATA_RESOLUTION_ORDER[entity_index + 3] == "entity_role_not_admitted"


def test_resolution_order_slice_position():
    """binding_mismatch sits directly after no_matching_slice, per contract."""
    slice_index = NO_DATA_RESOLUTION_ORDER.index("no_matching_slice")
    assert NO_DATA_RESOLUTION_ORDER[slice_index + 1] == "binding_mismatch"


def test_resolution_order_currency_conversion_position():
    """rate_coverage_gap, currency-conversion's own no_data reason, sits
    directly after unit_not_resolved. recipe_not_declared, the other
    currency-conversion code, maps to unsupported_code rather than no_data_reason
    on the text doors and does not appear in this order."""
    unit_index = NO_DATA_RESOLUTION_ORDER.index("unit_not_resolved")
    assert NO_DATA_RESOLUTION_ORDER[unit_index + 1] == "rate_coverage_gap"
