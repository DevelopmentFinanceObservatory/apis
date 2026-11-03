"""Tests the role-text lexicon and its check against door-supplied flow axes."""

from __future__ import annotations

import pytest

from materialise.roles import KNOWN_ROLE_NAMES, axis_for_role
from search.engine.entry.parsed import _role_query_for_part
from search.engine.role_text import (
    Party,
    breakdown_axis,
    contradicts,
    every_member_axis,
    every_member_breakdown,
    every_member_breakdown_axis,
    parsed_directions_contradicted,
    role_axis_for_text,
    span_starts,
)
from search.models import ParsedEntity, ParsedPart, RoleAxis


class TestDebtAndTradeVocabulary:
    @pytest.mark.parametrize("word", ["creditor", "lender", "exporter"])
    def test_from_side_words_resolve_to_origin(self, word: str) -> None:
        assert role_axis_for_text(word) == "origin"
        assert role_axis_for_text(word.upper()) == "origin"

    @pytest.mark.parametrize("word", ["debtor", "borrower", "importer"])
    def test_to_side_words_resolve_to_destination(self, word: str) -> None:
        assert role_axis_for_text(word) == "destination"
        assert role_axis_for_text(word.upper()) == "destination"


@pytest.mark.parametrize("axis", ["origin", "destination"])
def test_each_axis_name_resolves_to_its_own_axis(axis: RoleAxis) -> None:
    assert role_axis_for_text(axis) == axis


@pytest.mark.parametrize(
    ("phrase", "query", "axis", "breakdown"),
    [
        ("all donors", "aid from all donors to Ethiopia in 2022", "origin", False),
        ("from all donors", "aid to Ethiopia from all donors in 2022", "origin", False),
        ("every provider", "aid to Ethiopia from every provider in 2022", "origin", True),
        ("to all creditors", "debt owed to all creditors by Kenya", "origin", False),
        ("each recipient", "aid from Germany to each recipient in 2022", "destination", True),
        ("all the recipients", "aid from Germany to all the recipients", "destination", False),
        ("each beneficiary", "aid from Germany to each beneficiary", "destination", True),
        ("all countries", "aid to Ethiopia from all countries in 2022", "origin", False),
        ("to all countries", "aid from France to all countries", "destination", False),
        ("all donor countries", "aid to Ethiopia from all donor countries", "origin", False),
        ("all countries", "aid from all countries to all countries", None, False),
        ("all countries", "GDP of all countries", None, False),
        ("in all sectors", "aid to Ethiopia in all sectors in 2022", None, False),
        ("all bilateral donors", "aid from all bilateral donors to Kenya", None, False),
        ("all donors and recipients", "aid between all donors and recipients", None, False),
        ("all", "aid to Ethiopia from all", None, False),
        ("donors", "aid to Ethiopia from donors", None, False),
    ],
)
def test_every_member_axis_reads_the_role_a_phrase_asks_every_member_of(
    phrase: str, query: str, axis: RoleAxis | None, breakdown: bool
) -> None:
    assert every_member_axis(phrase, query=query) == axis
    assert every_member_breakdown(phrase) is breakdown


@pytest.mark.parametrize(
    ("phrase", "axis", "every_member"),
    [
        ("by donor", "origin", None),
        ("by donors", "origin", None),
        ("across the recipients", "destination", None),
        ("by creditor", "origin", None),
        ("each donor", "origin", "origin"),
        ("from each donor", "origin", "origin"),
        ("for each donor", "origin", "origin"),
        ("to every recipient", "destination", "destination"),
        ("by each donor", "origin", "origin"),
        ("all donors", None, None),
        ("from all donors", None, None),
        ("by country", None, None),
        ("by sector", None, None),
        ("by donor and recipient", None, None),
        ("top donors", None, None),
        ("by", None, None),
        ("for", None, None),
    ],
)
def test_breakdown_axis_reads_the_role_a_phrase_lists_member_by_member(
    phrase: str, axis: RoleAxis | None, every_member: RoleAxis | None
) -> None:
    """`every_member_breakdown_axis` reads only the each and every phrases, which ask for
    the breakdown under any operation."""
    query = f"aid to Kenya {phrase}"
    assert breakdown_axis(phrase, query=query) == axis
    assert every_member_breakdown_axis(phrase, query=query) == every_member


class TestUnrecognised:
    @pytest.mark.parametrize(
        "word",
        [
            pytest.param(None, id="none"),
            pytest.param("bogus", id="unknown-word"),
            pytest.param("donateur", id="former-french-word"),
            pytest.param("sponsor", id="unrecognised-role-word"),
        ],
    )
    def test_unrecognised_role_text_resolves_to_none(self, word: str | None) -> None:
        assert role_axis_for_text(word) is None


class TestDeclaredRoleNamesDoNotDrift:
    def test_every_axis_placed_role_name_resolves_to_its_own_axis(self) -> None:
        """Every directional role in `materialise.roles` maps to the same axis here."""
        checked = 0
        for role_name in KNOWN_ROLE_NAMES:
            axis = axis_for_role(role_name)
            if axis is None:
                continue
            assert role_axis_for_text(role_name) == axis, role_name
            checked += 1
        assert checked > 0


def test_role_axis_for_text_maps_from_family_to_origin() -> None:
    assert role_axis_for_text("from") == "origin"
    assert role_axis_for_text("donor") == "origin"
    assert role_axis_for_text("provider") == "origin"


def test_role_axis_for_text_maps_to_family_to_destination() -> None:
    assert role_axis_for_text("to") == "destination"
    assert role_axis_for_text("recipient") == "destination"
    assert role_axis_for_text("beneficiary") == "destination"


def test_span_starts_finds_whole_word_runs_and_no_empty_span() -> None:
    assert span_starts(["aid", "to", "south", "sudan", "from", "sudan"], ["sudan"]) == [3, 5]
    assert span_starts(["aid", "to"], []) == []


def test_contradicts_false_when_span_absent_from_query() -> None:
    assert contradicts("population of Kenya", span="Norway", axis="origin", party=None) is False


def test_contradicts_false_when_axis_agrees_with_query_wording() -> None:
    assert (
        contradicts("aid to Kenya from Norway", span="Norway", axis="origin", party=None) is False
    )
    assert (
        contradicts("aid to Kenya from Norway", span="Kenya", axis="destination", party=None)
        is False
    )


def test_contradicts_true_when_opposite_preposition_immediately_precedes_span() -> None:
    assert contradicts("aid to Kenya from Norway", span="Kenya", axis="origin", party=None) is True
    assert (
        contradicts("aid to Kenya from Norway", span="Norway", axis="destination", party=None)
        is True
    )


def test_contradicts_false_at_position_zero_with_nothing_preceding() -> None:
    """Kenya opens the query -- there is no preceding token to check, so no
    contradiction is possible regardless of the asserted axis."""
    assert contradicts("Kenya aid", span="Kenya", axis="origin", party=None) is False


def test_contradicts_does_not_veto_the_substring_collision_pair() -> None:
    """ "aid to South Sudan from Sudan" -- Sudan is the donor (origin), South
    Sudan the recipient (destination). A wider scan would score Sudan
    "destination" by matching the "South Sudan" substring; contradicts must
    not fire for either entity's own (correct) door-supplied axis, since it
    anchors on each entity's own verbatim span, never on a substring match
    that could confuse "Sudan" with "South Sudan"."""
    query = "aid to South Sudan from Sudan"
    assert contradicts(query, span="Sudan", axis="origin", party=None) is False
    assert contradicts(query, span="South Sudan", axis="destination", party=None) is False


def test_contradicts_checks_every_occurrence_of_the_span() -> None:
    """ "Kenya announced a plan; aid flows to Kenya from Norway" -- Kenya's
    first mention has nothing preceding it, but contradicts must still
    catch a genuine conflict at a later occurrence."""
    query = "Kenya announced a plan; aid flows to Kenya from Norway"
    assert contradicts(query, span="Kenya", axis="origin", party=None) is True
    assert contradicts(query, span="Kenya", axis="destination", party=None) is False


def test_contradicts_handles_multi_word_spans() -> None:
    query = "aid to Papua New Guinea from Guinea"
    assert contradicts(query, span="Guinea", axis="origin", party=None) is False
    assert contradicts(query, span="Papua New Guinea", axis="destination", party=None) is False


def test_contradicts_does_not_veto_a_second_unrelated_mention_of_the_same_place() -> None:
    """A scan that returns True whenever ANY occurrence is preceded by an
    opposite-direction preposition would let a second, unrelated mention of
    the same place veto a correct, unambiguous answer. Kenya's first
    occurrence ("from Kenya") supports origin; its second, unrelated
    occurrence ("traveling to Kenya for meetings") would -- read alone --
    contradict origin. A supporting occurrence anywhere means the query does
    state that axis somewhere, so there is nothing to veto."""
    query = "Aid from Kenya to Uganda, with officials traveling to Kenya for meetings next month"
    assert contradicts(query, span="Kenya", axis="origin", party=None) is False


def test_contradicts_does_not_veto_a_second_unrelated_mention_reversed_order() -> None:
    """Same shape as above, with the order of the two mentions reversed:
    India's donor mention comes first, the unrelated "to India" mention
    comes second."""
    query = "Donors from India increased funding, with delegates traveling to India for meetings."
    assert contradicts(query, span="India", axis="origin", party=None) is False


def test_contradicts_still_fires_when_the_query_states_only_the_opposite_axis() -> None:
    """contradicts is not inert: a query that states ONLY the opposite
    direction, with no supporting occurrence anywhere, still vetoes."""
    assert contradicts("aid to Kenya", span="Kenya", axis="origin", party=None) is True
    assert (
        contradicts("aid to Kenya, and also aid to Kenya", span="Kenya", axis="origin", party=None)
        is True
    )


def test_unrecognised_role_text_warning_uses_stable_code() -> None:
    part = ParsedPart(
        variable_text="aid",
        entities=[ParsedEntity(text="Kenya", role_text="bogus")],
    )

    _, warnings = _role_query_for_part(part)

    assert [warning.code for warning in warnings] == ["role_text_not_recognised"]


class TestDebtParties:
    """A creditor or a debtor draws no contradiction; a source, a recipient or a place with
    no party keeps the preposition veto."""

    @pytest.mark.parametrize(
        ("query", "span", "axis", "party"),
        [
            ("Zambia gives to China in debt repayments", "China", "origin", "creditor"),
            ("Kenya's arrears to China", "China", "origin", "creditor"),
            ("debt service paid by Zambia to the IMF", "the IMF", "origin", "creditor"),
            ("repayments from Zambia to China", "Zambia", "destination", "debtor"),
            ("Kenya's debt owed to China", "China", "origin", "creditor"),
        ],
    )
    def test_a_debt_party_draws_no_contradiction(
        self, query: str, span: str, axis: RoleAxis, party: Party
    ) -> None:
        assert contradicts(query, span=span, axis=axis, party=party) is False
        assert contradicts(query, span=span, axis=axis, party=None) is True

    @pytest.mark.parametrize(
        ("query", "span", "axis", "party"),
        [
            ("aid to Kenya", "Kenya", "origin", "source"),
            ("ODA given to Kenya", "Kenya", "origin", None),
            ("debt relief to Kenya", "Kenya", "origin", "source"),
            ("aid from Norway", "Norway", "destination", "recipient"),
            ("Zambia gives to China in debt repayments", "China", "origin", "source"),
        ],
    )
    def test_a_flow_party_still_draws_its_contradiction(
        self, query: str, span: str, axis: RoleAxis, party: Party | None
    ) -> None:
        assert contradicts(query, span=span, axis=axis, party=party) is True

    @pytest.mark.parametrize(
        ("role_text", "contradicted"),
        [("creditor", False), ("lender", False), ("donor", True), ("from", True)],
    )
    def test_the_parsed_door_reads_the_party_from_role_text(
        self, role_text: str, contradicted: bool
    ) -> None:
        """A debt party word ("creditor", or its synonym "lender") exempts a parsed origin
        after "to"; "donor" and "from" name the axis and no debt party."""
        part = ParsedPart(
            variable_text="external debt arrears to",
            entities=[ParsedEntity(text="China", role_text=role_text)],
        )
        assert parsed_directions_contradicted(part) is contradicted
