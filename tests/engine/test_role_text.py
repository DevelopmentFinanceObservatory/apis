"""Tests the role-text lexicon and its check against door-supplied flow axes."""

from __future__ import annotations

import pytest

from materialise.roles import KNOWN_ROLE_NAMES, axis_for_role
from search.engine.entry.parsed import _role_query_for_part
from search.engine.role_text import (
    breakdown_axis,
    contradicts,
    debt_axes,
    debtor_possessive,
    every_member_axis,
    every_member_breakdown,
    every_member_breakdown_axis,
    receipt_axes,
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
            # An active "owe": the subject owes the object.
            (
                "how much does China owe India",
                "external debt",
                ("China", "India"),
                ("destination", "origin"),
            ),
            (
                "what does Zambia owe the IMF",
                "IMF debt",
                ("Zambia", "IMF"),
                ("destination", "origin"),
            ),
            (
                "how much does Kenya owe China",
                "external debt",
                ("Kenya", "China"),
                ("destination", "origin"),
            ),
            (
                "how much does Kenya owe to China",
                "external debt",
                ("Kenya", "China"),
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
                "Kenya debt owed to China",
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

    def test_an_owe_whose_object_is_not_the_place_reads_only_the_debtor(self) -> None:
        """ "Ghana's creditors" is the object, so Ghana is no party and nothing is set."""
        axes = debt_axes(
            "how much does Kenya owe Ghana's creditors",
            measures=["external debt"],
            places=_places("Kenya", "Ghana"),
            directed={},
        )
        assert axes == {}


# One place a possessive names over its own debt in a query that is the possessive and its
# measure alone.
_ONE_POSSESSIVE_OVER_ITS_DEBT = [
    ("Kenya's debt", "debt", "Kenya"),
    ("Kenya's debt", "total debt", "Kenya"),
    ("Kenya\u2019s debt", "debt", "Kenya"),
    ("the Kenya's debt", "debt", "Kenya"),
    ("Kenya's external debt", "external debt", "Kenya"),
    ("Kenya's gross debt service", "gross debt service", "Kenya"),
    ("Kenya's debt service", "debt service", "Kenya"),
    ("Kenya's debt repayments", "debt repayments", "Kenya"),
    ("Kenya's repayments", "debt repayments", "Kenya"),
    ("Kenya's liabilities", "external liabilities", "Kenya"),
    ("Kenya's debt reduction target", "debt reduction target", "Kenya"),
    ("The World Bank's debt", "debt", "World Bank"),
]


class TestDebtorPossessive:
    """One place a possessive names over its own debt is the debtor."""

    @pytest.mark.parametrize(("query", "measure", "place"), _ONE_POSSESSIVE_OVER_ITS_DEBT)
    @pytest.mark.parametrize("guessed", ["origin", "destination"])
    def test_one_directed_possessive_over_its_own_debt_is_the_debtor(
        self, query: str, measure: str, place: str, guessed: RoleAxis
    ) -> None:
        """One place the extraction directed is read by its possessive alone, whichever
        axis the extraction guessed."""
        axes = debtor_possessive(
            query, measures=[measure], places=_places(place), directed={place: guessed}
        )
        assert axes == {place: "destination"}

    @pytest.mark.parametrize(("query", "measure", "place"), _ONE_POSSESSIVE_OVER_ITS_DEBT)
    def test_one_undirected_possessive_sets_no_axis(
        self, query: str, measure: str, place: str
    ) -> None:
        """One place the extraction left undirected keeps no axis, so a subject-only series
        drops no direction for it."""
        assert (
            debtor_possessive(query, measures=[measure], places=_places(place), directed={}) == {}
        )

    @pytest.mark.parametrize(
        ("query", "measures", "place"),
        [
            # A debt measured against a quantity, or a creditor cue.
            ("Kenya's debt to GDP ratio", ["debt to GDP ratio"], "Kenya"),
            ("debt owed to China", ["debt"], "China"),
            # No possessive over the place.
            ("external debt of Kenya", ["external debt"], "Kenya"),
            # A possessive claim on another party's debt.
            ("Kenya's share of external debt", ["external debt"], "Kenya"),
            # A creditor's claim on the debt.
            ("China's debt claims", ["debt claims"], "China"),
            ("China's debt exposure", ["debt exposure"], "China"),
            ("China's debt exposure", ["debt"], "China"),
            ("China's debt holdings", ["debt holdings"], "China"),
            ("China's debt portfolio", ["debt portfolio"], "China"),
            # A possessive over no debt, or a second measure.
            ("China's lending to Africa", ["lending"], "China"),
            ("Germany's aid", ["aid"], "Germany"),
            ("Kenya's GDP and debt", ["GDP", "debt"], "Kenya"),
            # A creditor or lender word, in any sense: the extraction's axis stands.
            ("Kenya's debt by creditor", ["debt"], "Kenya"),
            ("Kenya's debt by year and creditor", ["debt"], "Kenya"),
            ("Kenya's debt by lenders", ["debt"], "Kenya"),
            ("Kenya's debt by creditor-country", ["debt"], "Kenya"),
            ("Kenya's debt target by creditor", ["debt"], "Kenya"),
            ("Kenya's debt by creditor destination", ["debt"], "Kenya"),
            ("China's debt by creditor", ["debt"], "China"),
            ("China's debt by lender", ["debt"], "China"),
            ("China's debt as a creditor", ["debt"], "China"),
            ("China's debt as the largest bilateral creditor", ["debt"], "China"),
            ("China's outstanding debt as lender", ["outstanding debt"], "China"),
            ("IMF's debt as lender of last resort", ["debt"], "IMF"),
            ("IMF's debt, lender of last resort", ["debt"], "IMF"),
            ("China's creditor debt", ["debt"], "China"),
            ("China's official creditor debt", ["debt"], "China"),
            ("China's debt and creditor countries' share", ["debt"], "China"),
            ("China's debt in creditor countries' share", ["debt"], "China"),
            ("World Bank's debt as a multilateral lender", ["debt"], "World Bank"),
            ("Paris Club's debt, creditor-country totals", ["debt"], "Paris Club"),
            ("China's debt as top lender since 2010", ["debt"], "China"),
            ("China's lender debt stock", ["debt stock"], "China"),
            ("China's debt, top creditors excluded", ["debt"], "China"),
            ("China's debt to recipients by creditor", ["debt"], "China"),
            ("China's debt by destination and creditor", ["debt"], "China"),
            ("China's debt by importer and lender", ["debt"], "China"),
            ("China's debt lent to creditor nations", ["debt"], "China"),
            ("Japan's debt from lenders", ["debt"], "Japan"),
            ("China's debt receivable by creditor", ["debt"], "China"),
            ("Japan's debt in the Creditor Reporting System", ["debt"], "Japan"),
            ("Japan's debt, Creditor Reporting System", ["debt"], "Japan"),
            ("Japan's debt relief in the Creditor Reporting System", ["debt relief"], "Japan"),
            ("Japan's debt rescheduling by creditor", ["debt rescheduling"], "Japan"),
            (
                "Japan's debt forgiveness grants, Creditor Reporting System",
                ["debt forgiveness"],
                "Japan",
            ),
            ("Japan's debt from CRS creditor data", ["debt"], "Japan"),
            ("Paris Club creditors' debt", ["debt"], "Paris Club creditors"),
            ("Kenya's debt in USD", ["debt"], "Kenya"),
            ("Kenya's debt by creditor and Japan", ["debt"], "Kenya"),
            # A role noun of the debtor's end: the extraction's axis stands.
            ("China's debt by debtor country", ["debt"], "China"),
            ("China's debt for each debtor country", ["debt"], "China"),
            ("China's debt, every borrower", ["debt"], "China"),
            ("China's debt by borrower in 2020", ["debt"], "China"),
            ("China's debt by debtor country in 2020", ["debt"], "China"),
            ("China's debt by debtor country and year", ["debt"], "China"),
            ("China's debt by recipient since 2010", ["debt"], "China"),
            ("China's debt by debtor country over time", ["debt"], "China"),
            ("China's debt for each debtor country in Africa", ["debt"], "China"),
            ("China's debt by debtor country between 2010 and 2020", ["debt"], "China"),
            ("China's debt by debtor country during 2015", ["debt"], "China"),
            ("China's debt by debtor country until 2020", ["debt"], "China"),
            ("China's debt by debtor country through 2020", ["debt"], "China"),
            ("China's debt by debtor country after 2015", ["debt"], "China"),
            ("China's debt by debtor country as of 2020", ["debt"], "China"),
            ("China's debt by debtor country per year", ["debt"], "China"),
            ("China's debt by debtor country or region", ["debt"], "China"),
            ("China's debt by debtor country at current prices", ["debt"], "China"),
            ("China's debt by debtor country last decade", ["debt"], "China"),
            ("China's debt by debtor, 2020", ["debt"], "China"),
            ("China's debt by debtor governments in 2020", ["debt"], "China"),
            ("How China's debt by debtor has changed since 2010", ["debt"], "China"),
            ("China's debt by debtor is largest in Angola", ["debt"], "China"),
            ("China's debt by debtor when it peaked", ["debt"], "China"),
            ("China's debt by debtor then sector", ["debt"], "China"),
            ("China's debt by debtor in the most recent year", ["debt"], "China"),
            ("China's debt by year and debtor country", ["debt"], "China"),
            ("China's debt by sector and recipient country", ["debt"], "China"),
            ("China's debt by debtor's region", ["debt"], "China"),
            ("China's debt by debtor country with interest", ["debt"], "China"),
            ("China's debt by debtor country in USD", ["debt"], "China"),
            ("China's debt by debtor country?", ["debt"], "China"),
            ("China's debt by debtor countries in Africa", ["debt"], "China"),
            ("China's debt held by debtor countries", ["debt"], "China"),
            ("China's debt by debtor-country", ["debt"], "China"),
            ("China's debt by creditor and debtor-country", ["debt"], "China"),
            ("Kenya's debt by creditor, owed by borrowers", ["debt"], "Kenya"),
            ("China's debt by borrower-country", ["debt"], "China"),
            ("China's debt by recipient-country", ["debt"], "China"),
            ("China's debt by debtor nation", ["debt"], "China"),
            ("China's debt by debtor economy", ["debt"], "China"),
            ("China's debt by debtor government", ["debt"], "China"),
            ("China's debt service from each debtor", ["debt service"], "China"),
            ("Japan's outstanding debt by recipient", ["outstanding debt"], "Japan"),
            ("Kenya's debt in Africa by debtor", ["debt"], "Kenya"),
            ("Kenya's external debt by debtor type", ["external debt"], "Kenya"),
            ("Kenya's external debt by debtor group", ["external debt"], "Kenya"),
            ("Kenya's external debt by debtor classification", ["external debt"], "Kenya"),
            ("Kenya's external debt by debtor country type", ["external debt"], "Kenya"),
            ("World Bank's debt to recipients", ["debt"], "World Bank"),
            ("Kenya's debt beneficiary", ["debt"], "Kenya"),
            # A debtor named by an owe or a debtor verb and by.
            ("Which countries owe the most of China's debt", ["debt"], "China"),
            (
                "Which African countries owe the most of China's outstanding debt",
                ["outstanding debt"],
                "China",
            ),
            ("rank countries by how much of China's debt they owe", ["debt"], "China"),
            ("China's debt owed by African countries", ["debt"], "China"),
            ("China's debt owed by low-income countries", ["debt"], "China"),
            ("Which countries owe China's debt by creditor", ["debt"], "China"),
            ("China's debt by creditor owed by Africa", ["debt"], "China"),
            # Any other words beside the possessive debt phrase.
            ("China's debt extended to each country", ["debt"], "China"),
            ("China's debt extended to every country", ["debt"], "China"),
            ("China's debt disbursed to each country", ["debt"], "China"),
            (
                "China's outstanding debt going to each country in Africa",
                ["outstanding debt"],
                "China",
            ),
            ("China's debt lent to each country", ["debt"], "China"),
            ("China's debt, to each country", ["debt"], "China"),
            ("China's debt across every country it lends to", ["debt"], "China"),
            ("China's debt by borrowing country", ["debt"], "China"),
            ("China's debt broken down by indebted country", ["debt"], "China"),
            ("China's debt repayments from African countries", ["debt repayments"], "China"),
            ("China's debt repayments from developing countries", ["debt repayments"], "China"),
            ("China's debt repayments received from Africa", ["debt repayments"], "China"),
            ("China's debt service received from developing countries", ["debt service"], "China"),
            ("China's debt service receipts", ["debt service"], "China"),
            ("China's debt investments abroad", ["debt"], "China"),
            ("China's debt loans to Africa", ["debt"], "China"),
            ("China's debt stock held by African governments", ["debt stock"], "China"),
            ("China's debt held abroad", ["debt"], "China"),
            ("China's outstanding debt in low-income economies", ["outstanding debt"], "China"),
            ("China's debt in developing nations", ["debt"], "China"),
            ("China's debt across developing economies", ["debt"], "China"),
            ("China's debt by country", ["debt"], "China"),
            ("China's debt financing", ["debt"], "China"),
            ("China's debt service paid by Zambia", ["debt service"], "China"),
            ("China's debt service payments from Africa", ["debt service"], "China"),
            ("China's debt service by country", ["debt service"], "China"),
            (
                "Which countries does Kenya's gross debt service go to, total",
                ["gross debt service"],
                "Kenya",
            ),
            ("Kenya's debt in 2020", ["debt"], "Kenya"),
            ("Kenya's debt reduction target", ["debt"], "Kenya"),
            ("Kenya's debt by destination of funds", ["debt"], "Kenya"),
            ("Kenya's debt sustainability for importers", ["debt"], "Kenya"),
            ("Kenya's debt and the IMF target", ["debt"], "Kenya"),
            ("Paris Club's debt with Kenya", ["debt"], "Paris Club"),
            ("IMF's debt portfolio in Africa", ["debt"], "IMF"),
            ("The IMF's outstanding debt from Africa", ["debt"], "IMF"),
            ("World Bank's debt outstanding from borrowing countries", ["debt"], "World Bank"),
            # A creditor's claim on the debt.
            ("China's debt receivables", ["debt receivables"], "China"),
            ("China's debt lending", ["debt lending"], "China"),
            ("China's debt assets", ["debt assets"], "China"),
            ("China's debt claim", ["debt claim"], "China"),
            ("China's debt credits", ["debt credits"], "China"),
            ("China's debt claims on Africa", ["debt"], "China"),
        ],
    )
    def test_anything_else_sets_no_axis(self, query: str, measures: list[str], place: str) -> None:
        axes = debtor_possessive(
            query, measures=measures, places=_places(place), directed={place: "origin"}
        )
        assert axes == {}

    def test_two_places_are_debt_axes_alone(self) -> None:
        axes = debtor_possessive(
            "Kenya's debt to China",
            measures=["debt"],
            places=_places("Kenya", "China"),
            directed={"Kenya": "origin", "China": "destination"},
        )
        assert axes == {}


class TestReceiptAxes:
    """A place the query names as receiving takes the destination axis."""

    @pytest.mark.parametrize(
        ("query", "places", "expected"),
        [
            ("net ODA received by Ethiopia in 1965", ("Ethiopia",), {"Ethiopia": "destination"}),
            ("food aid received by the Gambia", ("Gambia",), {"Gambia": "destination"}),
            (
                "By what percentage did the total ODA Ghana received from all donors change",
                ("Ghana",),
                {"Ghana": "destination"},
            ),
            (
                "How much ODA did Tanzania receive from Germany",
                ("Tanzania", "Germany"),
                {"Tanzania": "destination"},
            ),
            # A place a preposition opens heads its own phrase: Norway does not receive.
            (
                "aid from Norway received by Kenya",
                ("Norway", "Kenya"),
                {"Kenya": "destination"},
            ),
            # "received" with no agent "by" names no recipient.
            ("aid received in Kenya", ("Kenya",), {}),
            ("ODA from Japan to Kenya", ("Japan", "Kenya"), {}),
        ],
    )
    def test_receipt_wording_names_the_recipient(
        self, query: str, places: tuple[str, ...], expected: dict[str, RoleAxis]
    ) -> None:
        assert receipt_axes(query, places=_places(*places)) == expected
