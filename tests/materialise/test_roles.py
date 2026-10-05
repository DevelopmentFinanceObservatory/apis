"""Tests role names, flow-axis mappings, and ambiguity handling."""

from __future__ import annotations

import pytest

from materialise.roles import (
    KNOWN_ROLE_NAMES,
    axis_ambiguous_for,
    axis_for_role,
    declared_role_for,
    roles_by_axis,
)


def test_origin_role_names_map_to_origin() -> None:
    assert axis_for_role("provider") == "origin"
    assert axis_for_role("creditor") == "origin"
    assert axis_for_role("source") == "origin"
    assert axis_for_role("financingProvider") == "origin"


def test_destination_role_names_map_to_destination() -> None:
    assert axis_for_role("recipient") == "destination"
    assert axis_for_role("debtor") == "destination"
    assert axis_for_role("target") == "destination"
    assert axis_for_role("financingRecipient") == "destination"


def test_observation_about_has_no_axis() -> None:
    assert axis_for_role("observationAbout") is None


def test_unrecognised_role_name_has_no_axis() -> None:
    assert axis_for_role("beneficiaryOrganisation") is None


def test_roles_by_axis_places_both_ends_of_a_declared_flow() -> None:
    assert roles_by_axis(("creditor", "debtor")) == {"origin": "creditor", "destination": "debtor"}


def test_roles_by_axis_places_the_financing_role_names() -> None:
    assert roles_by_axis(("financingProvider", "financingRecipient")) == {
        "origin": "financingProvider",
        "destination": "financingRecipient",
    }


def test_roles_by_axis_omits_an_unplaced_role() -> None:
    assert roles_by_axis(("provider", "beneficiaryOrg")) == {"origin": "provider"}


def test_roles_by_axis_omits_the_subject_role() -> None:
    assert roles_by_axis(("observationAbout",)) == {}


def test_two_names_on_one_axis_raise_value_error() -> None:
    with pytest.raises(ValueError, match="origin"):
        roles_by_axis(("provider", "creditor"))


def test_known_role_names_covers_every_placed_role_and_the_subject_role() -> None:
    assert {
        "provider",
        "creditor",
        "source",
        "financingProvider",
        "recipient",
        "debtor",
        "target",
        "financingRecipient",
        "observationAbout",
    } == KNOWN_ROLE_NAMES


def test_declared_role_for_returns_a_raw_declared_property_itself() -> None:
    assert declared_role_for("financingRecipient", ("financingRecipient",)) == "financingRecipient"


def test_declared_role_for_resolves_an_axis_alias_to_the_declared_role_on_it() -> None:
    assert declared_role_for("destination", ("financingRecipient",)) == "financingRecipient"
    assert declared_role_for("origin", ("financingRecipient", "financingProvider")) == (
        "financingProvider"
    )


def test_declared_role_for_misses_an_axis_alias_reversed_from_the_declared_flow() -> None:
    # `financingProvider` sits on `origin` only, so a `destination` key on a
    # shape that declares no destination role names no declared role.
    assert declared_role_for("destination", ("financingProvider",)) is None


def test_declared_role_for_misses_a_key_in_neither_vocabulary() -> None:
    assert declared_role_for("beneficiaryOrganisation", ("financingRecipient",)) is None


def test_declared_role_for_returns_none_not_a_raise_when_two_declared_roles_share_the_axis() -> (
    None
):
    # `provider` and `creditor` both sit on `origin` (`_ROLE_AXES` above), so
    # an `origin` key names no single declared role. The resolver absorbs
    # `roles_by_axis`'s same-axis ValueError and returns None.
    assert declared_role_for("provider", ("provider", "creditor")) == "provider"
    assert declared_role_for("creditor", ("provider", "creditor")) == "creditor"
    assert declared_role_for("origin", ("provider", "creditor")) is None
    assert declared_role_for("typoRole", ("provider", "creditor")) is None


def test_axis_ambiguous_for_is_true_when_two_declared_roles_share_the_axis() -> None:
    # `provider` and `creditor` both sit on `origin` (`_ROLE_AXES` above),
    # so an `origin` key is ambiguous between the two.
    assert axis_ambiguous_for("origin", ("provider", "creditor"))


def test_axis_ambiguous_for_is_false_when_exactly_one_role_sits_on_the_axis() -> None:
    assert not axis_ambiguous_for("destination", ("financingRecipient",))
    assert not axis_ambiguous_for("origin", ("financingRecipient", "financingProvider"))


def test_axis_ambiguous_for_is_false_for_a_key_that_is_not_an_axis_label() -> None:
    # A role name is not an axis, so no declared-role count makes it
    # ambiguous, even on the two-origin shape above.
    assert not axis_ambiguous_for("beneficiaryOrganisation", ("provider", "creditor"))
    assert not axis_ambiguous_for("creditor", ("provider", "creditor"))


def test_axis_ambiguous_for_is_false_for_a_raw_declared_property() -> None:
    assert not axis_ambiguous_for("financingProvider", ("financingProvider", "financingRecipient"))
