"""Tests the role-text lexicon and its check against door-supplied flow axes."""

from __future__ import annotations

import pytest

from materialise.roles import KNOWN_ROLE_NAMES, axis_for_role
from search.engine.entry.parsed import _role_query_for_part
from search.engine.role_text import (
    contradicts,
    debt_axes,
    every_member_axis,
    every_member_breakdown,
    role_axis_for_text,
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


def test_contradicts_false_when_span_absent_from_query() -> None:
    assert contradicts("population of Kenya", span="Norway", axis="origin") is False


def test_contradicts_false_when_axis_agrees_with_query_wording() -> None:
    assert contradicts("aid to Kenya from Norway", span="Norway", axis="origin") is False
    assert contradicts("aid to Kenya from Norway", span="Kenya", axis="destination") is False


def test_contradicts_true_when_opposite_preposition_immediately_precedes_span() -> None:
    assert contradicts("aid to Kenya from Norway", span="Kenya", axis="origin") is True
    assert contradicts("aid to Kenya from Norway", span="Norway", axis="destination") is True


def test_contradicts_false_at_position_zero_with_nothing_preceding() -> None:
    """Kenya opens the query -- there is no preceding token to check, so no
    contradiction is possible regardless of the asserted axis."""
    assert contradicts("Kenya aid", span="Kenya", axis="origin") is False


def test_contradicts_does_not_veto_the_substring_collision_pair() -> None:
    """ "aid to South Sudan from Sudan" -- Sudan is the donor (origin), South
    Sudan the recipient (destination). A wider scan would score Sudan
    "destination" by matching the "South Sudan" substring; contradicts must
    not fire for either entity's own (correct) door-supplied axis, since it
    anchors on each entity's own verbatim span, never on a substring match
    that could confuse "Sudan" with "South Sudan"."""
    query = "aid to South Sudan from Sudan"
    assert contradicts(query, span="Sudan", axis="origin") is False
    assert contradicts(query, span="South Sudan", axis="destination") is False


def test_contradicts_checks_every_occurrence_of_the_span() -> None:
    """ "Kenya announced a plan; aid flows to Kenya from Norway" -- Kenya's
    first mention has nothing preceding it, but contradicts must still
    catch a genuine conflict at a later occurrence."""
    query = "Kenya announced a plan; aid flows to Kenya from Norway"
    assert contradicts(query, span="Kenya", axis="origin") is True
    assert contradicts(query, span="Kenya", axis="destination") is False


def test_contradicts_handles_multi_word_spans() -> None:
    query = "aid to Papua New Guinea from Guinea"
    assert contradicts(query, span="Guinea", axis="origin") is False
    assert contradicts(query, span="Papua New Guinea", axis="destination") is False


def test_contradicts_does_not_veto_a_second_unrelated_mention_of_the_same_place() -> None:
    """A scan that returns True whenever ANY occurrence is preceded by an
    opposite-direction preposition would let a second, unrelated mention of
    the same place veto a correct, unambiguous answer. Kenya's first
    occurrence ("from Kenya") supports origin; its second, unrelated
    occurrence ("traveling to Kenya for meetings") would -- read alone --
    contradict origin. A supporting occurrence anywhere means the query does
    state that axis somewhere, so there is nothing to veto."""
    query = "Aid from Kenya to Uganda, with officials traveling to Kenya for meetings next month"
    assert contradicts(query, span="Kenya", axis="origin") is False


def test_contradicts_does_not_veto_a_second_unrelated_mention_reversed_order() -> None:
    """Same shape as above, with the order of the two mentions reversed:
    India's donor mention comes first, the unrelated "to India" mention
    comes second."""
    query = "Donors from India increased funding, with delegates traveling to India for meetings."
    assert contradicts(query, span="India", axis="origin") is False


def test_contradicts_still_fires_when_the_query_states_only_the_opposite_axis() -> None:
    """contradicts is not inert: a query that states ONLY the opposite
    direction, with no supporting occurrence anywhere, still vetoes."""
    assert contradicts("aid to Kenya", span="Kenya", axis="origin") is True
    assert contradicts("aid to Kenya, and also aid to Kenya", span="Kenya", axis="origin") is True


def test_unrecognised_role_text_warning_uses_stable_code() -> None:
    part = ParsedPart(
        variable_text="aid",
        entities=[ParsedEntity(text="Kenya", role_text="bogus")],
    )

    _, warnings = _role_query_for_part(part)

    assert [warning.code for warning in warnings] == ["role_text_not_recognised"]


class TestCreditorCue:
    """A ``to`` that closes a creditor cue leaves an origin claim uncontradicted."""

    @pytest.mark.parametrize(
        ("query", "span", "axis", "expected"),
        [
            ("Kenya's debt owed to China", "China", "origin", False),
            ("Kenya's debt to China", "China", "origin", False),
            ("Zambia's debt service repaid to the IMF", "the IMF", "origin", False),
            ("Zambia's debt service repaid to the IMF", "IMF", "origin", False),
            ("Ghana's liabilities to Norway", "Norway", "origin", False),
            ("Kenya's debt service payments to China", "China", "origin", False),
            ("Kenya's debt owed to China", "Kenya", "destination", False),
            ("loans from China to Angola", "China", "origin", False),
            ("loans from China to Angola", "Angola", "destination", False),
            # "paid by ... to" is no creditor cue, so the bare "to" contradicts.
            ("debt service paid by Zambia to the IMF", "the IMF", "origin", True),
        ],
    )
    def test_debt_cases(self, query: str, span: str, axis: RoleAxis, expected: bool) -> None:
        assert contradicts(query, span=span, axis=axis) is expected

    @pytest.mark.parametrize(
        ("query", "span", "axis"),
        [
            ("aid to Kenya", "Kenya", "origin"),
            ("ODA given to Kenya", "Kenya", "origin"),
            ("debt relief to Kenya", "Kenya", "origin"),
            # A tail word closes a creditor cue only after a debt noun.
            ("aid payments to Kenya", "Kenya", "origin"),
            ("debt relief payments to Kenya", "Kenya", "origin"),
            ("Kenya's debt relative to Uganda", "Uganda", "origin"),
            ("aid from Norway", "Norway", "destination"),
            ("debt owed from China", "China", "destination"),
            # "owing to" means "because of", so it is no creditor cue.
            ("refugees owing to Syria", "Syria", "origin"),
        ],
    )
    def test_other_wording_still_contradicts(self, query: str, span: str, axis: RoleAxis) -> None:
        assert contradicts(query, span=span, axis=axis) is True

    def test_cue_elsewhere_in_the_query_does_not_defer(self) -> None:
        """The cue must end at the place; a cue before another place leaves this ``to`` in force."""
        query = "debt owed to Norway and aid to Kenya"
        assert contradicts(query, span="Kenya", axis="origin") is True

    def test_cued_occurrence_leaves_another_one_in_force(self) -> None:
        query = "debt owed to China, and aid to China"
        assert contradicts(query, span="China", axis="origin") is True


def _places(*names: str) -> dict[str, list[str]]:
    return {name: [name] for name in names}


# (query, places, the places extraction may direct without making either a party)
_COMPARISONS = [
    ("Kenya's external debt compared with China", ("Kenya", "China"), ("Kenya",)),
    ("Kenya's external debt compared with Uganda", ("Kenya", "Uganda"), ("Kenya",)),
    ("Kenya's external debt vs Uganda", ("Kenya", "Uganda"), ("Kenya",)),
    ("Kenya's debt versus Ghana's", ("Kenya", "Ghana"), ("Kenya", "Ghana")),
    ("China and Kenya's external debt", ("China", "Kenya"), ("Kenya",)),
    ("Kenya and Ghana's external debt", ("Kenya", "Ghana"), ("Ghana",)),
    ("Ghana's and Kenya's external debt", ("Ghana", "Kenya"), ("Ghana", "Kenya")),
    ("Kenya's share of Africa's debt", ("Kenya", "Africa"), ("Kenya", "Africa")),
    ("Kenya's external debt in East Africa", ("Kenya", "East Africa"), ("Kenya",)),
    ("Kenya's debt relative to Sub-Saharan Africa", ("Kenya", "Sub-Saharan Africa"), ("Kenya",)),
    ("Nairobi's debt in Kenya", ("Nairobi", "Kenya"), ("Nairobi",)),
    ("Nigeria's debt service compared with Ghana", ("Nigeria", "Ghana"), ("Nigeria",)),
]


class TestDebtAxes:
    """A two-party debt relation puts the creditor on origin and the debtor on destination."""

    @pytest.mark.parametrize(
        ("query", "measure", "places", "expected"),
        [
            ("Kenya's debt to China", "debt", ("Kenya", "China"), ("destination", "origin")),
            ("Kenya's debt owed to China", "debt", ("Kenya", "China"), ("destination", "origin")),
            (
                "Kenya\u2019s debt owed to China",
                "debt",
                ("Kenya", "China"),
                ("destination", "origin"),
            ),
            ("debt owed to China by Kenya", "debt", ("China", "Kenya"), ("origin", "destination")),
            ("DEBT OWED TO CHINA BY KENYA", "debt", ("China", "Kenya"), ("origin", "destination")),
            ("debt owed by Kenya to China", "debt", ("Kenya", "China"), ("destination", "origin")),
            (
                "debt incurred by Kenya to China",
                "debt",
                ("Kenya", "China"),
                ("destination", "origin"),
            ),
            (
                "debt owed to the IMF by the Gambia",
                "debt",
                ("the IMF", "the Gambia"),
                ("origin", "destination"),
            ),
            (
                "Ghana's liabilities to Norway",
                "liabilities",
                ("Ghana", "Norway"),
                ("destination", "origin"),
            ),
            (
                "Kenya's public and publicly guaranteed debt to China",
                "public and publicly guaranteed debt",
                ("Kenya", "China"),
                ("destination", "origin"),
            ),
            (
                "Kenya's debt to South Sudan",
                "debt",
                ("Kenya", "South Sudan"),
                ("destination", "origin"),
            ),
            (
                "debt owed to China out of Kenya's external debt",
                "external debt",
                ("China", "Kenya"),
                ("origin", "destination"),
            ),
            (
                "Kenya's debt service payments to China",
                "debt service payments",
                ("Kenya", "China"),
                ("destination", "origin"),
            ),
            (
                "debt service to China by Kenya",
                "debt service",
                ("China", "Kenya"),
                ("origin", "destination"),
            ),
            (
                "Zambia's debt service repaid to the IMF",
                "debt service",
                ("Zambia", "the IMF"),
                ("destination", "origin"),
            ),
            (
                "Zambia's debt service repaid to the IMF",
                "debt service",
                ("Zambia", "IMF"),
                ("destination", "origin"),
            ),
        ],
    )
    def test_cues_on_both_places_set_both_parties(
        self,
        query: str,
        measure: str,
        places: tuple[str, str],
        expected: tuple[RoleAxis, RoleAxis],
    ) -> None:
        axes = debt_axes(query, measures=[measure], places=_places(*places), directed={})
        assert axes == dict(zip(places, expected, strict=True))

    @pytest.mark.parametrize(
        ("query", "measure", "cued", "uncued", "expected"),
        [
            (
                "how much does Kenya owe to China",
                "debt",
                "China",
                "Kenya",
                {"China": "origin", "Kenya": "destination"},
            ),
            (
                "Kenya debt to China",
                "debt",
                "China",
                "Kenya",
                {"China": "origin", "Kenya": "destination"},
            ),
        ],
    )
    def test_a_place_without_a_cue_keeps_an_extracted_axis_opposite_the_cued_one(
        self, query: str, measure: str, cued: str, uncued: str, expected: dict[str, RoleAxis]
    ) -> None:
        """The uncued place is a party only when extraction put it opposite the cued place.
        On the cued place's own axis it is a second creditor, and with no axis no party."""
        places = _places(cued, uncued)

        def axes(directed: dict[str, RoleAxis]) -> dict[str, RoleAxis]:
            return debt_axes(query, measures=[measure], places=places, directed=directed)

        assert axes({uncued: expected[uncued]}) == expected
        assert axes({uncued: expected[cued]}) == {}
        assert axes({cued: expected[cued]}) == {}
        assert axes({}) == {}

    def test_a_national_adjective_is_matched_by_its_span(self) -> None:
        places = {"Kenya": ["Kenyan"], "China": ["China"]}
        assert debt_axes(
            "Kenyan debt to China",
            measures=["debt"],
            places=places,
            directed={"Kenya": "destination"},
        ) == {"China": "origin", "Kenya": "destination"}

    def test_a_possessive_span_is_read_as_the_possessor(self) -> None:
        places = {"Kenya": ["Kenya's"], "China": ["China"]}
        assert debt_axes(
            "Kenya's debt to China", measures=["debt"], places=places, directed=()
        ) == {"Kenya": "destination", "China": "origin"}

    @pytest.mark.parametrize(("query", "places", "may_direct"), _COMPARISONS)
    @pytest.mark.parametrize("directs", ["none", "cued", "second", "both"])
    def test_a_comparison_or_a_locative_sets_no_axis(
        self, query: str, places: tuple[str, str], may_direct: tuple[str, ...], directs: str
    ) -> None:
        """A comparison place with no cue and no extracted axis is no party, and one that
        extraction put on the cued debtor's own axis is a second debtor, so nothing is set.
        Places that clash or claim set nothing whichever extraction directed."""
        named = {
            "none": (),
            "cued": may_direct,
            "second": places[1:],
            "both": places,
        }[directs]
        directed: dict[str, RoleAxis] = dict.fromkeys(named, "destination")
        axes = debt_axes(
            query, measures=["external debt"], places=_places(*places), directed=directed
        )
        assert axes == {}

    @pytest.mark.parametrize(
        ("query", "measures", "places"),
        [
            # A holder of, or claimant on, someone else's debt is no debtor.
            ("China's holdings of Kenyan debt", ["debt holdings"], ("China", "Kenyan")),
            ("China's holdings of Kenyan debt", ["debt"], ("China", "Kenyan")),
            ("China's share of Zambia debt", ["debt"], ("China", "Zambia")),
            ("China's debt claims on Africa", ["debt claims"], ("China", "Africa")),
            ("China's debt claims on Africa", ["debt"], ("China", "Africa")),
            # "held by" names the holder, not the debtor.
            ("Kenyan debt held by China", ["debt"], ("Kenyan", "China")),
            # More or fewer than one measure.
            ("Japan's ODA and external debt", ["ODA", "external debt"], ("Japan",)),
            (
                "Japan's ODA disbursements and debt to Kenya",
                ["ODA disbursements", "debt"],
                ("Japan", "Kenya"),
            ),
            ("Kenya's GDP and debt to China", ["GDP", "debt"], ("Kenya", "China")),
            ("Kenya's debt to China", [], ("Kenya", "China")),
            # A measure that names no debt, or an action on one.
            ("refugees owing to Syria from Turkey", ["refugees"], ("Syria", "Turkey")),
            ("aid owed to Kenya by Norway", ["aid"], ("Kenya", "Norway")),
            ("total population India", ["total population"], ("India",)),
            ("Norway's aid to Kenya", ["aid"], ("Norway", "Kenya")),
            ("China's debt relief to Kenya", ["debt relief"], ("China", "Kenya")),
            ("France's debt forgiveness to Haiti", ["debt forgiveness"], ("France", "Haiti")),
            (
                "concessional loans to Sri Lanka from Japan",
                ["concessional loans"],
                ("Sri Lanka", "Japan"),
            ),
            # Other than two places.
            ("Kenya's GDP and debt", ["debt"], ("Kenya",)),
            ("Kenya's debt to GDP ratio", ["debt to GDP ratio"], ("Kenya",)),
            ("refugees owing to Syria", ["refugees"], ("Syria",)),
            ("aid owed to Kenya", ["aid"], ("Kenya",)),
            ("Kenya's debt to China and India", ["debt"], ("Kenya", "China", "India")),
            # No cue on either place.
            ("external debt of Kenya and Uganda", ["external debt"], ("Kenya", "Uganda")),
            ("debt Kenya China", ["debt"], ("Kenya", "China")),
            ("indebted to China, Kenya", ["debt"], ("China", "Kenya")),
            ("debt to Chinatown in Kenya", ["debt"], ("China", "Kenya")),
            ("debt service paid by Zambia to the IMF", ["debt service"], ("Zambia", "the IMF")),
            # Cues that clash: both places are debtors, or both creditors.
            ("Kenya's debt and Uganda's debt", ["debt"], ("Kenya", "Uganda")),
            ("debt owed to China and debt owed to Japan", ["debt"], ("China", "Japan")),
            # A place both cues claim.
            ("China's debt owed to China, Kenya", ["debt"], ("China", "Kenya")),
        ],
    )
    @pytest.mark.parametrize("directed_axis", [None, "origin", "destination"])
    def test_other_queries_set_no_axis(
        self,
        query: str,
        measures: list[str],
        places: tuple[str, ...],
        directed_axis: RoleAxis | None,
    ) -> None:
        directed: dict[str, RoleAxis] = (
            dict.fromkeys(places, directed_axis) if directed_axis is not None else {}
        )
        axes = debt_axes(query, measures=measures, places=_places(*places), directed=directed)
        assert axes == {}

    @pytest.mark.parametrize(
        "query",
        ["Kenya's debt to China", "Kenya's debt owed to China", "debt owed to China, Kenya's"],
    )
    def test_debt_axes_draw_no_contradiction(self, query: str) -> None:
        axes = debt_axes(query, measures=["debt"], places=_places("Kenya", "China"), directed={})
        assert axes == {"Kenya": "destination", "China": "origin"}
        for span, axis in axes.items():
            assert contradicts(query, span=span, axis=axis) is False
