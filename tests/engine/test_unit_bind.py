"""Tests unit-text parsing, assembly, completion, and facet installation.

Seeded standard-library randomness keeps fuzz cases repeatable. Recognised partial requests
complete from declared sources. Empty and unrecognised requests refuse.
"""

from __future__ import annotations

import random

import pytest
from dfo_spec_kit import identity

from search.engine.unit.bind import (
    BoundUnitRequest,
    CurrencyAlternative,
    UnitDefaultPolicy,
    UnitDefaultPolicyError,
    UnitNotResolved,
    UnitRequestUnsupported,
    _assemble,
    _classify,
    _complete,
    _normalise,
    _tokenise,
    bind_unit_text,
    names_unit_request,
    unit_text_naming,
    with_unit_lexicon_version,
)
from search.engine.unit.lexicon import (
    AMBIGUOUS_CURRENCY_NAMES,
    AMBIGUOUS_ISO_CODES,
    CONNECTIVE_WORDS,
    CURRENCY_ALIASES,
    LEXICON_VERSION,
    MAX_YEAR,
    MIN_YEAR,
    PRICE_BASIS_PHRASES,
)
from search.models import IdentityCurrencyRole
from tests.engine._assemble_fixtures import _make_spec
from unit_words.currency_codes import ISO_4217_CODES

# The base year the conversion tables declare in these tests.
_DECLARED_BASE_YEAR = 2024


def _policy(currency: str = "USD", base_year: int = _DECLARED_BASE_YEAR) -> UnitDefaultPolicy:
    return UnitDefaultPolicy(currency=currency, base_year=lambda: base_year)


def _base_year_never_read() -> int:
    raise AssertionError("the binder read the declared base year")


# A policy whose base-year source fails the test if the binder calls it.
_NO_BASE_YEAR_READ = UnitDefaultPolicy(currency="USD", base_year=_base_year_never_read)


def _bind(
    text: str,
    *,
    policy: UnitDefaultPolicy | None = None,
    bound_roles: frozenset = frozenset(),
):
    """`bind_unit_text` under the test policy: USD and the 2024 declared base year."""
    return bind_unit_text(text, policy=policy or _policy(), bound_roles=bound_roles)


# ---------------------------------------------------------------------------
# Stage 1: normalise
# ---------------------------------------------------------------------------


class TestNormalise:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param("USD", "usd", id="casefold"),
            # U+FF12 U+FF10 U+FF11 U+FF15 = fullwidth "2015"
            pytest.param("２０１５", "2015", id="nfkc-full-width-digits"),
            pytest.param("＄", "$", id="nfkc-full-width-dollar"),
            pytest.param("U.S. dollars", "us dollars", id="periods-dropped"),
            pytest.param("inflation-adjusted", "inflation adjusted", id="hyphens-spaced"),
            pytest.param("per–capita", "per capita", id="en-dash-spaced"),
            pytest.param("current_prices", "current prices", id="underscores-spaced"),
        ],
    )
    def test_normalises_text(self, text: str, expected: str) -> None:
        assert _normalise(text) == expected


# ---------------------------------------------------------------------------
# Stage 2: tokenise
# ---------------------------------------------------------------------------


class TestTokenise:
    def test_splits_words_and_digits(self) -> None:
        assert _tokenise(_normalise("constant 2015 prices")) == ["constant", "2015", "prices"]

    def test_symbol_is_its_own_token(self) -> None:
        assert _tokenise(_normalise("$100")) == ["$", "100"]

    def test_punctuation_is_dropped_not_tokenised(self) -> None:
        assert _tokenise(_normalise("constant prices, 2015")) == ["constant", "prices", "2015"]

    def test_empty_string_tokenises_to_nothing(self) -> None:
        assert _tokenise(_normalise("")) == []

    def test_whitespace_only_tokenises_to_nothing(self) -> None:
        assert _tokenise(_normalise("   \t\n  ")) == []


# ---------------------------------------------------------------------------
# Unrecognised characters refuse the phrase before policy completion.
# ---------------------------------------------------------------------------


class TestTokeniserErasureRefuses:
    @pytest.mark.parametrize(
        ("text", "random_splices"),
        [
            pytest.param("!!!", False, id="punctuation-only"),
            pytest.param("人民币", False, id="non-latin"),
            pytest.param("руб", False, id="cyrillic"),
            # "current USD" alone would fully bind; the trailing "!!!" must
            # still sink the whole phrase, not just get silently dropped.
            pytest.param("current USD !!!", False, id="mixed-valid-content"),
            pytest.param("", True, id="random-spliced-erasure"),
        ],
    )
    def test_erasure_refuses(self, text: str, random_splices: bool) -> None:
        if not random_splices:
            outcome = _bind(text)
            assert isinstance(outcome, UnitNotResolved)
            return

        # A random insertion checks that token erasure refuses at every position.
        rng = random.Random(4242)
        for _ in range(200):
            phrase = _random_lexicon_only_phrase(rng)
            words = phrase.split()
            position = rng.randint(0, len(words))
            spliced = " ".join((*words[:position], "人", *words[position:]))
            outcome = _bind(spliced)
            assert isinstance(outcome, UnitNotResolved), f"{spliced!r} -> {outcome!r}"

    def test_erasure_refuses_even_with_a_full_policy_declared(self) -> None:
        outcome = _bind("!!!", policy=_NO_BASE_YEAR_READ)
        assert isinstance(outcome, UnitNotResolved)


class TestEmptyInputAlwaysRefuses:
    """Empty and whitespace-only input refuse before policy completion."""

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("", id="empty"),
            pytest.param("   \t\n  ", id="whitespace-only"),
            # "for the" names no currency, price basis, or year -- just
            # connectives -- so it refuses the same way empty text does.
            pytest.param("for the", id="connective-only"),
        ],
    )
    def test_empty_or_connective_only_input_refuses(self, text: str) -> None:
        outcome = _bind(text)
        assert isinstance(outcome, UnitNotResolved)

    def test_empty_string_refuses_even_with_a_full_policy_declared(self) -> None:
        outcome = _bind("", policy=_NO_BASE_YEAR_READ)
        assert isinstance(outcome, UnitNotResolved)


# ---------------------------------------------------------------------------
# Stage 3: classify
# ---------------------------------------------------------------------------


class TestClassify:
    def test_iso_code_classifies_as_currency(self) -> None:
        outcome = _classify(_tokenise(_normalise("EUR")))
        assert outcome.currencies == ("EUR",)
        assert not outcome.unresolved

    def test_curated_alias_classifies_as_currency(self) -> None:
        outcome = _classify(_tokenise(_normalise("US dollars")))
        assert outcome.currencies == ("USD",)

    def test_symbol_classifies_as_currency(self) -> None:
        outcome = _classify(_tokenise(_normalise("$")))
        assert outcome.currencies == ("USD",)

    def test_price_basis_phrase_classifies(self) -> None:
        outcome = _classify(_tokenise(_normalise("constant prices")))
        assert outcome.price_bases == ("constant",)

    def test_longest_match_wins_over_shorter_alias(self) -> None:
        # "us dollars" (2-gram) must win over "dollars" (1-gram) matching alone.
        outcome = _classify(_tokenise(_normalise("us dollars")))
        assert outcome.currencies == ("USD",)
        assert not outcome.unresolved

    def test_four_word_phrase_matches(self) -> None:
        outcome = _classify(_tokenise(_normalise("West African CFA francs")))
        assert outcome.currencies == ("XOF",)

    def test_year_in_range_classifies(self) -> None:
        outcome = _classify(_tokenise(_normalise("2015")))
        assert outcome.years == (2015,)

    def test_year_out_of_range_is_unresolved(self) -> None:
        # A 4-digit run outside [MIN_YEAR, MAX_YEAR] is not a year.
        outcome = _classify(["1000"])
        assert outcome.years == ()
        assert outcome.unresolved == ("1000",)

    def test_connective_words_are_consumed_silently(self) -> None:
        outcome = _classify(_tokenise(_normalise("in constant 2015 prices")))
        assert outcome.price_bases == ("constant",)
        assert outcome.years == (2015,)
        assert not outcome.unresolved

    def test_for_and_years_are_consumed_as_connectives(self) -> None:
        outcome = _classify(_tokenise(_normalise("constant prices for years 2015")))
        assert outcome.price_bases == ("constant",)
        assert outcome.years == (2015,)
        assert not outcome.unresolved

    def test_unrecognised_token_is_unresolved(self) -> None:
        outcome = _classify(_tokenise(_normalise("gobbledygook")))
        assert outcome.unresolved == ("gobbledygook",)

    def test_unsupported_phrase_sets_reason(self) -> None:
        outcome = _classify(_tokenise(_normalise("per capita")))
        assert outcome.unsupported_reason is not None

    def test_first_unsupported_reason_wins_when_several_present(self) -> None:
        outcome = _classify(_tokenise(_normalise("per capita percent")))
        assert outcome.unsupported_reason is not None
        # exactly one reason is kept, not a list/concatenation of both
        assert "per-capita" in outcome.unsupported_reason


# ---------------------------------------------------------------------------
# "base year"/"base years" cue: names the INTENT to set a base year. A cue
# with no year value anywhere in the phrase is structurally incomplete, not
# merely underspecified -- see TestStructurallyIncompletePhrases below for
# the assemble-level refusal this flag drives.
# ---------------------------------------------------------------------------


class TestBaseYearCue:
    def test_base_year_sets_the_cue_flag(self) -> None:
        outcome = _classify(_tokenise(_normalise("base year")))
        assert outcome.base_year_cue is True

    def test_base_years_plural_also_sets_the_cue_flag(self) -> None:
        outcome = _classify(_tokenise(_normalise("base years")))
        assert outcome.base_year_cue is True

    def test_no_cue_when_base_and_year_are_not_adjacent(self) -> None:
        outcome = _classify(_tokenise(_normalise("base 2015 for years")))
        # "base" alone (not immediately followed by year/years) never forms
        # the cue bigram, and "for years" is consumed as ordinary
        # connectives -- neither sets the flag.
        assert outcome.base_year_cue is False
        assert outcome.years == (2015,)


# ---------------------------------------------------------------------------
# Ambiguous codes and names require a curated alias or return alternatives.
# ---------------------------------------------------------------------------


class TestAmbiguousBareCurrencyAlwaysRefuses:
    def test_collision_code_alone_refuses_with_itself_as_the_alternative(self) -> None:
        outcome = _bind("top")
        assert isinstance(outcome, UnitNotResolved)
        # A curated display name makes this a real disambiguation ("the
        # Tongan paʻanga"), not just the input token reflected back.
        assert outcome.alternatives == (
            CurrencyAlternative(currency="TOP", label="Tongan paʻanga"),
        )

    def test_collision_code_uppercase_alone_still_refuses(self) -> None:
        # Capitalisation is not evidence either way.
        outcome = _bind("TOP")
        assert isinstance(outcome, UnitNotResolved)
        assert outcome.alternatives == (
            CurrencyAlternative(currency="TOP", label="Tongan paʻanga"),
        )

    def test_a_year_elsewhere_in_the_phrase_does_not_rescue_a_collision_code(self) -> None:
        # A year alongside the code is not a signal that rescues it: no
        # surrounding context disambiguates an ambiguous mention, only a
        # declared alias does.
        outcome = _bind("top 2015")
        assert isinstance(outcome, UnitNotResolved)
        assert outcome.alternatives == (
            CurrencyAlternative(currency="TOP", label="Tongan paʻanga"),
        )

    def test_price_basis_alongside_a_collision_code_does_not_rescue_it_either(self) -> None:
        outcome = _bind("ALL current prices")
        assert isinstance(outcome, UnitNotResolved)
        assert outcome.alternatives == (CurrencyAlternative(currency="ALL", label="Albanian lek"),)

    def test_a_full_surrounding_phrase_does_not_rescue_a_collision_code(self) -> None:
        outcome = _bind("try constant 2020 prices")
        assert isinstance(outcome, UnitNotResolved)
        assert outcome.alternatives == (CurrencyAlternative(currency="TRY", label="Turkish lira"),)

    def test_non_colliding_code_still_binds_case_insensitively(self) -> None:
        # USD has no English-word collision, so lower case still binds --
        # the ambiguous-code check must not regress the common case.
        outcome = _bind("current usd")
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.currency == "USD"


# ---------------------------------------------------------------------------
# Role-currency phrases: a currency asked for by the spec role of a bound
# entity ("local"/"national"/"own currency" binds to whichever single role
# the query carries an entity on; "recipient's currency"/"donor currency"
# always bind to destination/origin).
# ---------------------------------------------------------------------------


class TestRoleCurrencyPhrases:
    def test_recipients_currency_binds_to_destination(self) -> None:
        outcome = _bind("recipient's currency")
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.currency == IdentityCurrencyRole(role="destination")

    def test_recipients_currency_plain_spelling_also_binds(self) -> None:
        outcome = _bind("recipient currency")
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.currency == IdentityCurrencyRole(role="destination")

    def test_donor_currency_binds_to_origin(self) -> None:
        outcome = _bind("donor currency")
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.currency == IdentityCurrencyRole(role="origin")

    def test_fixed_role_phrase_binds_with_no_bound_roles_at_all(self) -> None:
        # A fixed-role phrase binds to its role whether or not that role has
        # an entity bound on this ask -- the spec's own shape decides
        # legality later.
        outcome = _bind("donor currency", bound_roles=frozenset())
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.currency == IdentityCurrencyRole(role="origin")

    def test_fixed_role_phrase_binds_beside_two_bound_roles(self) -> None:
        outcome = _bind("donor currency", bound_roles=frozenset({"origin", "destination"}))
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.currency == IdentityCurrencyRole(role="origin")

    @pytest.mark.parametrize("phrase", ["local currency", "national currency", "own currency"])
    def test_beside_phrase_binds_to_the_one_bound_role(self, phrase: str) -> None:
        outcome = _bind(phrase, bound_roles=frozenset({"origin"}))
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.currency == IdentityCurrencyRole(role="origin")

        outcome = _bind(phrase, bound_roles=frozenset({"destination"}))
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.currency == IdentityCurrencyRole(role="destination")

    def test_beside_phrase_binds_to_observation_about_for_a_standard_ask(self) -> None:
        outcome = _bind("local currency", bound_roles=frozenset({"observationAbout"}))
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.currency == IdentityCurrencyRole(role="observationAbout")

    def test_beside_phrase_with_no_bound_role_refuses_with_no_alternatives(self) -> None:
        outcome = _bind("local currency", bound_roles=frozenset())
        assert isinstance(outcome, UnitNotResolved)
        assert outcome.alternatives == ()

    def test_beside_phrase_with_two_bound_roles_refuses_with_both_as_alternatives(self) -> None:
        outcome = _bind("local currency", bound_roles=frozenset({"origin", "destination"}))
        assert isinstance(outcome, UnitNotResolved)
        # One role shape per bound role, in sorted role order: the value a
        # caller resubmits as the request's currency.
        assert outcome.alternatives == (
            CurrencyAlternative(
                currency=IdentityCurrencyRole(role="destination"), label="recipient"
            ),
            CurrencyAlternative(currency=IdentityCurrencyRole(role="origin"), label="provider"),
        )

    def test_role_phrase_combined_with_an_iso_currency_refuses(self) -> None:
        # A role phrase and a literal ISO currency in one phrase refuse the
        # same way two ISO currencies do.
        outcome = _bind("donor currency USD")
        assert isinstance(outcome, UnitNotResolved)

    def test_price_basis_and_base_year_still_bind_alongside_a_role_currency(self) -> None:
        outcome = _bind("constant local currency", bound_roles=frozenset({"origin"}))
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.currency == IdentityCurrencyRole(role="origin")
        assert outcome.prices == "constant"

    def test_role_currency_wire_request_carries_the_role_object(self) -> None:
        outcome = _bind("donor currency")
        assert isinstance(outcome, BoundUnitRequest)
        wire = outcome.as_wire_request()
        assert wire["currency"] == {"role": "origin"}
        identity.validate_identity_core(_minimal_identity_doc(wire))


# ---------------------------------------------------------------------------
# `real` names constant prices; ambiguous bare currency names return alternatives.
# ---------------------------------------------------------------------------


class TestDeliberateLexiconExclusions:
    def test_bare_real_resolves_constant_prices_not_a_currency(self) -> None:
        # "real" alone binds prices=constant via text. The currency
        # and the base year complete from the policy.
        policy = _policy(currency="EUR")
        outcome = _bind("real", policy=policy)
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.prices == "constant"
        assert outcome.base_year == _DECLARED_BASE_YEAR
        assert outcome.request_basis["prices"] == "bound"
        assert outcome.request_basis["currency"] == "declared_default"
        assert outcome.request_basis["base_year"] == "declared_default"

    def test_bare_franc_refuses_with_its_currency_alternatives(self) -> None:
        outcome = _bind("franc")
        assert isinstance(outcome, UnitNotResolved)
        assert outcome.alternatives == (
            CurrencyAlternative(currency="CHF", label="Swiss franc"),
            CurrencyAlternative(currency="XAF", label="Central African CFA franc"),
            CurrencyAlternative(currency="XOF", label="West African CFA franc"),
        )

    def test_bare_francs_plural_refuses_the_same_way(self) -> None:
        outcome = _bind("francs")
        assert isinstance(outcome, UnitNotResolved)
        assert outcome.alternatives == (
            CurrencyAlternative(currency="CHF", label="Swiss franc"),
            CurrencyAlternative(currency="XAF", label="Central African CFA franc"),
            CurrencyAlternative(currency="XOF", label="West African CFA franc"),
        )

    def test_bare_kr_refuses_with_its_currency_alternatives(self) -> None:
        outcome = _bind("kr")
        assert isinstance(outcome, UnitNotResolved)
        assert outcome.alternatives == (
            CurrencyAlternative(currency="DKK", label="Danish krone"),
            CurrencyAlternative(currency="NOK", label="Norwegian krone"),
            CurrencyAlternative(currency="SEK", label="Swedish krona"),
        )

    def test_bare_pesos_refuses_with_its_currency_alternatives(self) -> None:
        outcome = _bind("pesos")
        assert isinstance(outcome, UnitNotResolved)
        assert outcome.alternatives == (
            CurrencyAlternative(currency="MXN", label="Mexican peso"),
            CurrencyAlternative(currency="PHP", label="Philippine peso"),
        )

    def test_qualified_franc_phrase_still_resolves_outright(self) -> None:
        # A declared alias resolves a phrase outright, regardless of the
        # bare form's own ambiguity.
        outcome = _bind("swiss francs", policy=_policy())
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.currency == "CHF"

    def test_qualified_peso_phrase_still_resolves_outright(self) -> None:
        outcome = _bind("mexican pesos", policy=_policy())
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.currency == "MXN"


# ---------------------------------------------------------------------------
# Stage 4: assemble. Unsupported grammar asks differ from parse failures.
# ---------------------------------------------------------------------------


class TestAssembleRuleTable:
    @pytest.mark.parametrize(
        ("text", "expected_type"),
        [
            pytest.param(
                "per capita gobbledygook",
                UnitRequestUnsupported,
                id="unsupported-wins-over-unresolved",
            ),
            pytest.param("EUR gobbledygook", UnitNotResolved, id="unresolved-token"),
            pytest.param("USD EUR", UnitNotResolved, id="distinct-currencies"),
            pytest.param("current constant", UnitNotResolved, id="distinct-price-bases"),
            pytest.param("2015 2016", UnitNotResolved, id="distinct-years"),
            pytest.param("current prices 2015", UnitNotResolved, id="current-prices-with-year"),
            # A year with no currency, no price-basis keyword, and no "prices"
            # token names a time window, not a price basis -- even a
            # mis-segmenting extractor cannot turn "2015" alone into constant
            # prices.
            pytest.param("2015", UnitNotResolved, id="bare-year"),
        ],
    )
    def test_refusal_rule_rows(self, text: str, expected_type: type[object]) -> None:
        outcome = _classify(_tokenise(_normalise(text)))
        result = _assemble(outcome)
        assert isinstance(result, expected_type)

    def test_same_currency_named_twice_is_not_a_conflict(self) -> None:
        outcome = _classify(_tokenise(_normalise("USD dollars")))
        result = _assemble(outcome)
        assert result.currency == "USD"  # type: ignore[union-attr]

    def test_year_with_no_basis_keyword_reads_as_constant(self) -> None:
        outcome = _classify(_tokenise(_normalise("2015 EUR")))
        result = _assemble(outcome)
        assert result.prices == "constant"  # type: ignore[union-attr]
        assert result.base_year == 2015  # type: ignore[union-attr]

    def test_year_with_the_literal_prices_token_reads_as_constant(self) -> None:
        # "2015 prices" is the other documented constant-price idiom: the
        # literal "prices" token beside a year, with no currency and no basis
        # keyword, still binds prices=constant/base_year=2015 -- the exact
        # boundary a bare "2015" (above) sits just the other side of.
        outcome = _classify(_tokenise(_normalise("2015 prices")))
        result = _assemble(outcome)
        assert result.currency is None  # type: ignore[union-attr]
        assert result.prices == "constant"  # type: ignore[union-attr]
        assert result.base_year == 2015  # type: ignore[union-attr]

    def test_constant_prices_with_no_year_leaves_base_year_open(self) -> None:
        outcome = _classify(_tokenise(_normalise("constant prices")))
        result = _assemble(outcome)
        assert result.prices == "constant"  # type: ignore[union-attr]
        assert result.base_year is None  # type: ignore[union-attr]

    def test_currency_only_leaves_prices_and_base_year_open(self) -> None:
        outcome = _classify(_tokenise(_normalise("EUR")))
        result = _assemble(outcome)
        assert result.currency == "EUR"  # type: ignore[union-attr]
        assert result.prices is None  # type: ignore[union-attr]
        assert result.base_year is None  # type: ignore[union-attr]

    def test_fully_empty_phrase_refuses(self) -> None:
        # Nothing named at all: no policy or fallback could ever complete
        # this, so it refuses right here at assemble, before _complete is
        # even reached.
        outcome = _classify([])
        result = _assemble(outcome)
        assert isinstance(result, UnitNotResolved)


# ---------------------------------------------------------------------------
# A bare `base year` cue lacks a value and returns `unit_not_resolved`.
# ---------------------------------------------------------------------------


class TestStructurallyIncompletePhrases:
    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("current prices base year", id="current-prices"),
            pytest.param("USD base year", id="currency"),
        ],
    )
    def test_bare_base_year_cue_refuses(self, text: str) -> None:
        outcome = _bind(text)
        assert isinstance(outcome, UnitNotResolved)

    def test_base_year_cue_satisfied_by_a_year_value_still_binds(self) -> None:
        outcome = _bind("base year 2015 USD")
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.base_year == 2015
        assert outcome.currency == "USD"

    def test_constant_prices_for_years_2015_still_assembles(self) -> None:
        # "years" alone (not preceded by "base") stays an ordinary
        # connective, so this phrase's ASSEMBLE result is fully bound: prices
        # and base_year both come from the text, with no base-year-cue
        # refusal.
        result = _assemble(_classify(_tokenise(_normalise("constant prices for years 2015"))))
        assert result.prices == "constant"  # type: ignore[union-attr]
        assert result.base_year == 2015  # type: ignore[union-attr]

        outcome = _bind("constant prices for years 2015 EUR")
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.currency == "EUR"
        assert outcome.prices == "constant"
        assert outcome.base_year == 2015


# ---------------------------------------------------------------------------
# A bare year refuses; a year beside `prices` sets a constant-price base year.
# ---------------------------------------------------------------------------


class TestBareYearVsYearWithPricesToken:
    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("2015", id="bare-year"),
            # A year is never bound alongside current prices, "prices" token
            # or not -- base_year is only valid when prices is constant.
            pytest.param("current 2015", id="year-with-current-basis"),
        ],
    )
    def test_year_boundary_refuses(self, text: str) -> None:
        outcome = _bind(text)
        assert isinstance(outcome, UnitNotResolved)

    def test_year_with_prices_token_binds_with_currency_left_open(self) -> None:
        outcome = _bind("2015 prices")
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.prices == "constant"
        assert outcome.base_year == 2015
        assert outcome.request_basis["currency"] == "declared_default"
        assert outcome.request_basis["prices"] == "bound"
        assert outcome.request_basis["base_year"] == "bound"

    def test_year_with_prices_token_uses_a_declared_currency_policy(self) -> None:
        outcome = _bind("2015 prices", policy=_policy(currency="EUR"))
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.currency == "EUR"
        assert outcome.request_basis["currency"] == "declared_default"

    def test_year_with_a_currency_binds_even_with_no_prices_token(self) -> None:
        # The other documented idiom (a currency, not the "prices" token,
        # beside the year): still binds, still the boundary a bare year sits
        # just the other side of.
        outcome = _bind("2015 EUR")
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.currency == "EUR"
        assert outcome.prices == "constant"
        assert outcome.base_year == 2015


# ---------------------------------------------------------------------------
# Stage 5: complete
# ---------------------------------------------------------------------------


class TestComplete:
    def test_fully_bound_needs_no_completion(self) -> None:
        outcome = _bind("2015 EUR", policy=_NO_BASE_YEAR_READ)
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.request_basis == {
            "currency": "bound",
            "prices": "bound",
            "base_year": "bound",
        }

    def test_missing_currency_completes_from_the_policy(self) -> None:
        outcome = _bind("constant 2015 prices", policy=_policy(currency="EUR"))
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.currency == "EUR"
        assert outcome.request_basis["currency"] == "declared_default"

    def test_missing_prices_complete_to_current(self) -> None:
        outcome = _bind("EUR", policy=_NO_BASE_YEAR_READ)
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.prices == "current"
        assert outcome.base_year is None
        assert outcome.request_basis == {"currency": "bound", "prices": "declared_default"}

    def test_missing_base_year_completes_from_the_declared_tables(self) -> None:
        outcome = _bind("constant EUR", policy=_policy(base_year=1999))
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.base_year == 1999
        assert outcome.request_basis["base_year"] == "declared_default"

    def test_every_missing_member_completes_from_its_declared_source(self) -> None:
        # "constant" alone binds only prices via text.
        outcome = _bind("constant")
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.as_wire_request() == {
            "currency": "USD",
            "prices": "constant",
            "base_year": _DECLARED_BASE_YEAR,
        }
        assert outcome.request_basis == {
            "currency": "declared_default",
            "prices": "bound",
            "base_year": "declared_default",
        }

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("current EUR", id="current-prices"),
            pytest.param("EUR", id="prices-complete-to-current"),
            pytest.param("2015 EUR", id="named-year"),
            pytest.param("constant 2015 prices", id="named-year-with-basis"),
        ],
    )
    def test_the_base_year_source_is_read_only_when_a_constant_ask_names_no_year(
        self, text: str
    ) -> None:
        assert isinstance(_bind(text, policy=_NO_BASE_YEAR_READ), BoundUnitRequest)

    def test_whatever_the_base_year_source_raises_propagates(self) -> None:
        class _Disagree(Exception):
            pass

        def _disagreeing_tables() -> int:
            raise _Disagree

        policy = UnitDefaultPolicy(currency="USD", base_year=_disagreeing_tables)
        with pytest.raises(_Disagree):
            _bind("constant prices", policy=policy)

    def test_direct_complete_call_matches_pipeline(self) -> None:
        assembled = _assemble(_classify(_tokenise(_normalise("current EUR"))))
        direct = _complete(assembled, policy=_NO_BASE_YEAR_READ)  # type: ignore[arg-type]
        via_pipeline = _bind("current EUR")
        assert direct == via_pipeline


# ---------------------------------------------------------------------------
# UnitDefaultPolicy construction-time validation: an invalid declared currency
# is a deployment misconfiguration, caught before any request needs it.
# ---------------------------------------------------------------------------


class TestUnitDefaultPolicyValidation:
    def test_valid_policy_constructs_fine(self) -> None:
        _policy(currency="EUR")

    def test_unregistered_currency_code_raises(self) -> None:
        with pytest.raises(UnitDefaultPolicyError, match="ISO 4217") as exc_info:
            _policy(currency="NOT_A_CODE")
        assert exc_info.value.member == "currency"

    def test_lowercase_currency_code_raises(self) -> None:
        # ISO_4217_CODES only ever holds upper-case entries -- a lower-case
        # spelling never matches, regardless of the code's validity.
        with pytest.raises(UnitDefaultPolicyError, match="ISO 4217") as exc_info:
            _policy(currency="eur")
        assert exc_info.value.member == "currency"


class TestInvalidPolicyNeverMintsAnIdentity:
    def test_a_valid_declared_currency_that_is_not_iso_never_reaches_bind(self) -> None:
        # Constructing the policy itself is where an invalid currency is
        # caught, so bind_unit_text can never be called with one in hand at
        # all.
        with pytest.raises(UnitDefaultPolicyError):
            _policy(currency="NOT_A_CODE")

    def test_every_successful_bind_validates_against_the_real_identity_schema(self) -> None:
        # Belt-and-braces: _complete calls dfo_spec_kit.identity directly
        # before hashing, so a BoundUnitRequest's wire form always already
        # passed the same validation a real spec_id construction would run.
        outcome = _bind("current EUR")
        assert isinstance(outcome, BoundUnitRequest)
        doc = {
            "identity_version": "spec1",
            "shape": {"kind": "statvar_schema", "population_type": "test_shape"},
            "slots": [],
            "entities": {},
            "requested_window": {"kind": "all"},
            "facets": {"unit": {"state": "value", "request": outcome.as_wire_request()}},
        }
        identity.validate_identity_core(doc)  # raises on any schema violation


# ---------------------------------------------------------------------------
# Typed refusals: every refusal code is exactly what the binder documents,
# and every refusal type actually carries a lexicon_version.
# ---------------------------------------------------------------------------


class TestTypedRefusals:
    def test_unresolved_code_is_unit_not_resolved(self) -> None:
        outcome = _bind("gobbledygook")
        assert isinstance(outcome, UnitNotResolved)
        assert outcome.code == "unit_not_resolved"
        assert outcome.lexicon_version == LEXICON_VERSION

    def test_unsupported_code_is_unit_request_unsupported(self) -> None:
        outcome = _bind("per capita")
        assert isinstance(outcome, UnitRequestUnsupported)
        assert outcome.code == "unit_request_unsupported"
        assert outcome.lexicon_version == LEXICON_VERSION

    def test_unit_request_unsupported_is_registered_in_the_platform_registry(self) -> None:
        from search.engine.codes import REFUSAL_CODES

        assert "unit_request_unsupported" in REFUSAL_CODES

    def test_unit_not_resolved_is_a_registered_no_data_reason(self) -> None:
        from typing import get_args

        from search.models import NoDataReason

        assert "unit_not_resolved" in get_args(NoDataReason)

    def test_unit_not_resolved_is_not_a_refusal_code(self) -> None:
        # unit_not_resolved is a no_data reason (see NoDataReason above), not
        # a problem-details refusal code -- it belongs to a different
        # registry namespace, so it must not also appear in REFUSAL_CODES.
        from search.engine.codes import REFUSAL_CODES

        assert "unit_not_resolved" not in REFUSAL_CODES


# ---------------------------------------------------------------------------
# Every BindOutcome variant exposes `code` as its discriminator.
# ---------------------------------------------------------------------------


class TestBindOutcomeDiscriminator:
    def test_bound_request_carries_the_bound_discriminator(self) -> None:
        outcome = _bind("current EUR")
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.code == "bound"

    def test_all_three_variants_expose_the_same_discriminator_attribute(self) -> None:
        bound = _bind("current EUR")
        not_resolved = _bind("gobbledygook")
        unsupported = _bind("per capita")
        assert isinstance(bound, BoundUnitRequest)
        assert isinstance(not_resolved, UnitNotResolved)
        assert isinstance(unsupported, UnitRequestUnsupported)

        codes = {bound.code, not_resolved.code, unsupported.code}
        assert codes == {"bound", "unit_not_resolved", "unit_request_unsupported"}

    def test_switching_on_the_discriminator_is_exhaustive(self) -> None:
        # A caller should be able to dispatch on .code alone, with no
        # isinstance fallback needed for any BindOutcome member.
        for outcome in (
            _bind("current EUR"),
            _bind("gobbledygook"),
            _bind("per capita"),
        ):
            match outcome.code:
                case "bound":
                    handled = "bound"
                case "unit_not_resolved":
                    handled = "unit_not_resolved"
                case "unit_request_unsupported":
                    handled = "unit_request_unsupported"
                case _:  # pragma: no cover - would mean a new, unhandled variant
                    pytest.fail(f"unhandled discriminator: {outcome.code!r}")
            assert handled == outcome.code


# ---------------------------------------------------------------------------
# Property tests check complete input consumption and the real request schema.
# ---------------------------------------------------------------------------


def _minimal_identity_doc(unit_request: dict) -> dict:
    return {
        "identity_version": "spec1",
        "shape": {"kind": "statvar_schema", "population_type": "test_shape"},
        "slots": [],
        "entities": {},
        "requested_window": {"kind": "all"},
        "facets": {"unit": {"state": "value", "request": unit_request}},
    }


_CURRENCY_PHRASES = [
    *CURRENCY_ALIASES,
    # Lower-case ISO codes, excluding AMBIGUOUS_ISO_CODES: a collision code
    # never binds as a currency on its own (see
    # TestAmbiguousBareCurrencyAlwaysRefuses), so this generator -- whose
    # whole contract is "every phrase it builds binds" -- must not pick one.
    *(c.lower() for c in ISO_4217_CODES if c not in AMBIGUOUS_ISO_CODES),
]
_PRICE_BASIS_KEYS = list(PRICE_BASIS_PHRASES)
_CONNECTIVES = list(CONNECTIVE_WORDS)


def _random_lexicon_only_phrase(rng: random.Random) -> str:
    """Build a phrase from lexicon vocabulary alone that always fully binds
    from the text alone, with no policy needed: a currency and a price basis
    are always both named (no member is left for completion to guess at),
    and a year is named whenever the chosen price basis is constant
    (base_year is required once prices is constant)."""
    currency_phrase = rng.choice(_CURRENCY_PHRASES)
    basis_phrase = rng.choice(_PRICE_BASIS_KEYS)
    basis_value = PRICE_BASIS_PHRASES[basis_phrase]
    year = rng.randint(MIN_YEAR, MAX_YEAR) if basis_value == "constant" else None

    tokens = [t for t in (currency_phrase, basis_phrase, str(year) if year else None) if t]
    for _ in range(rng.randint(0, 2)):
        tokens.insert(rng.randint(0, len(tokens)), rng.choice(_CONNECTIVES))
    return " ".join(tokens)


class TestPropertyEveryNonRefusalIsComplete:
    def test_random_lexicon_phrases_never_refuse_and_validate(self) -> None:
        rng = random.Random(20240115)
        for _ in range(500):
            phrase = _random_lexicon_only_phrase(rng)
            outcome = _bind(phrase)
            assert isinstance(outcome, BoundUnitRequest), f"{phrase!r} -> {outcome!r}"

            classified = _classify(_tokenise(_normalise(phrase)))
            assert classified.unresolved == (), phrase
            assert classified.unsupported_reason is None, phrase

            doc = _minimal_identity_doc(outcome.as_wire_request())
            identity.validate_identity_core(doc)  # raises on any schema violation

    def test_random_lexicon_phrases_base_year_presence_matches_prices(self) -> None:
        rng = random.Random(99)
        for _ in range(300):
            phrase = _random_lexicon_only_phrase(rng)
            outcome = _bind(phrase)
            assert isinstance(outcome, BoundUnitRequest)
            if outcome.prices == "constant":
                assert outcome.base_year is not None
                assert "base_year" in outcome.request_basis
            else:
                assert outcome.base_year is None
                assert "base_year" not in outcome.request_basis


# ---------------------------------------------------------------------------
# Connectives form a small allowlist; unknown words remain unresolved.
# ---------------------------------------------------------------------------


class TestConnectiveWordsAreTheShrunkAllowlist:
    def test_an_undocumented_connective_looking_word_is_not_filler(self) -> None:
        # "on" reads like an ordinary connective in English, but nothing in
        # this suite documents a phrase that needs it, so it is not in
        # CONNECTIVE_WORDS and sinks the whole phrase like any other
        # unrecognised token.
        assert "on" not in CONNECTIVE_WORDS
        outcome = _bind("on EUR current")
        assert isinstance(outcome, UnitNotResolved)

    def test_connective_words_is_exactly_the_documented_set(self) -> None:
        assert (
            frozenset({"in", "at", "prices", "terms", "of", "the", "for", "years"})
            == CONNECTIVE_WORDS
        )


# ---------------------------------------------------------------------------
# Identical text yields identical requests and hashes across callers.
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_repeated_calls_with_the_same_text_are_equal(self) -> None:
        a = _bind("constant 2015 euros")
        b = _bind("constant 2015 euros")
        assert a == b
        assert a.request_hash == b.request_hash  # type: ignore[union-attr]

    def test_equivalent_phrasing_converges_on_the_same_hash(self) -> None:
        # An extraction-style phrase and a bare-code phrase both name
        # USD/current explicitly; only request_basis (bound for both here,
        # since both name prices too) may differ in general, though here
        # they happen to agree.
        extracted = _bind("current US dollars")
        coded = _bind("current USD")
        assert isinstance(extracted, BoundUnitRequest)
        assert isinstance(coded, BoundUnitRequest)
        assert extracted.currency == coded.currency == "USD"
        assert extracted.prices == coded.prices == "current"
        assert extracted.request_hash == coded.request_hash

    def test_refusals_are_also_deterministic(self) -> None:
        a = _bind("per capita")
        b = _bind("per capita")
        assert a == b


# ---------------------------------------------------------------------------
# A policy-completed request and the same explicit values share a hash.
# ---------------------------------------------------------------------------


class TestDeliberateIdentityGolden:
    def test_a_declared_policy_matching_the_explicit_values_agrees(self) -> None:
        via_policy = _bind("USD", policy=_policy())
        explicit = _bind("current USD")
        assert isinstance(via_policy, BoundUnitRequest)
        assert isinstance(explicit, BoundUnitRequest)

        assert via_policy.currency == explicit.currency == "USD"
        assert via_policy.prices == explicit.prices == "current"
        assert via_policy.request_hash == explicit.request_hash

        assert via_policy.request_basis == {"currency": "bound", "prices": "declared_default"}
        assert explicit.request_basis == {"currency": "bound", "prices": "bound"}
        assert via_policy.request_basis != explicit.request_basis

    def test_a_declared_base_year_agrees_with_the_same_year_stated_explicitly(self) -> None:
        via_tables = _bind("constant EUR")
        explicit = _bind(f"{_DECLARED_BASE_YEAR} EUR")
        assert isinstance(via_tables, BoundUnitRequest)
        assert isinstance(explicit, BoundUnitRequest)

        assert via_tables.request_hash == explicit.request_hash
        assert via_tables.request_basis["base_year"] == "declared_default"
        assert explicit.request_basis["base_year"] == "bound"

    def test_a_new_declared_base_year_completes_to_a_different_request(self) -> None:
        before = _bind("constant EUR", policy=_policy(base_year=2024))
        after = _bind("constant EUR", policy=_policy(base_year=2025))
        assert isinstance(before, BoundUnitRequest)
        assert isinstance(after, BoundUnitRequest)
        assert before.request_hash != after.request_hash


# ---------------------------------------------------------------------------
# Request hashes cover request values and exclude provenance.
# ---------------------------------------------------------------------------


class TestHashedIdentity:
    def test_hash_format(self) -> None:
        outcome = _bind("2015 EUR")
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.request_hash.startswith("unit_")
        hex_part = outcome.request_hash.removeprefix("unit_")
        assert len(hex_part) == 32
        int(hex_part, 16)

    def test_hash_changes_with_currency(self) -> None:
        a = _bind("current USD")
        b = _bind("current EUR")
        assert a.request_hash != b.request_hash  # type: ignore[union-attr]

    def test_hash_changes_with_prices(self) -> None:
        a = _bind("current EUR")
        b = _bind("constant 2015 EUR")
        assert a.request_hash != b.request_hash  # type: ignore[union-attr]

    def test_hash_ignores_request_basis(self) -> None:
        bound = _bind("current EUR")
        via_default = _bind("EUR", policy=_policy())
        assert isinstance(bound, BoundUnitRequest)
        assert isinstance(via_default, BoundUnitRequest)
        assert bound.request_basis != via_default.request_basis
        assert bound.request_hash == via_default.request_hash


# ---------------------------------------------------------------------------
# Fuzzing checks refusal, completion, and total binding.
# ---------------------------------------------------------------------------

_FUZZ_ASCII = "abcdefghijklmnopqrstuvwxyz0123456789 "
_FUZZ_PUNCTUATION = "$€£¥₹₩%.,-_()[]{}'\";:!?/\\"


def _random_fuzz_string(rng: random.Random) -> str:
    length = rng.randint(0, 40)
    chars: list[str] = []
    for _ in range(length):
        bucket = rng.random()
        if bucket < 0.4:
            chars.append(rng.choice(_FUZZ_ASCII))
        elif bucket < 0.6:
            chars.append(rng.choice(_FUZZ_PUNCTUATION))
        elif bucket < 0.8:
            chars.append(chr(rng.randint(0x00C0, 0x02AF)))  # latin-extended / diacritics
        else:
            chars.append(chr(rng.randint(0x1F300, 0x1F5FF)))  # pictographs
    return "".join(chars)


class TestFuzzing:
    @pytest.mark.parametrize(
        "text",
        [
            "constant 2015 2020 euros",  # two distinct years -> conflict
            "€100 million",  # a scale word the request form cannot represent
            "2015",  # a bare year names a time window, not a price basis
        ],
    )
    def test_documented_adversarial_inputs_refuse(self, text: str) -> None:
        outcome = _bind(text)
        assert outcome.code in ("unit_not_resolved", "unit_request_unsupported")

    def test_random_strings_never_crash_and_never_partially_bind(self) -> None:
        rng = random.Random(7)
        for _ in range(3000):
            text = _random_fuzz_string(rng)
            outcome = _bind(text)
            assert isinstance(outcome, (BoundUnitRequest, UnitNotResolved, UnitRequestUnsupported))
            if isinstance(outcome, BoundUnitRequest):
                assert outcome.prices in ("current", "constant")
                assert (outcome.base_year is not None) == (outcome.prices == "constant")
                assert isinstance(outcome.currency, str) and outcome.currency
            else:
                assert outcome.message
                assert outcome.code in ("unit_not_resolved", "unit_request_unsupported")

    @pytest.mark.parametrize(
        "text",
        [
            "",
            " ",
            "\n\t",
            "a" * 10_000,
            "\x00\x01\x02",
            "😀" * 50,  # emoji surrogate pair, repeated
            "-" * 100,
            "." * 100,
            "2015" * 500,
            "usd eur gbp jpy cny",
            "current constant current constant",
        ],
    )
    def test_pathological_inputs_never_crash(self, text: str) -> None:
        outcome = _bind(text)
        assert isinstance(outcome, (BoundUnitRequest, UnitNotResolved, UnitRequestUnsupported))

    def test_never_produces_a_currency_only_partial_dataclass(self) -> None:
        """ "Never partially bind" is structural, not just behavioural: the
        return type is a closed sum of BoundUnitRequest (always carries a
        currency and a prices value) or one of the two refusal types --
        there is no third shape with only SOME fields populated. This test
        pins that no code path can slip a None currency or None prices past
        _complete."""
        rng = random.Random(11)
        for _ in range(1000):
            text = _random_fuzz_string(rng)
            outcome = _bind(text)
            if isinstance(outcome, BoundUnitRequest):
                assert outcome.currency is not None
                assert outcome.prices is not None


# ---------------------------------------------------------------------------
# Every outcome carries the binder's lexicon version, which a bound Spec states.
# ---------------------------------------------------------------------------


class TestLexiconVersion:
    def test_bound_request_lexicon_version_matches_the_module_constant(self) -> None:
        outcome = _bind("current EUR")
        assert isinstance(outcome, BoundUnitRequest)
        assert outcome.lexicon_version == LEXICON_VERSION

    def test_refusals_also_carry_the_lexicon_version(self) -> None:
        refusal = _bind("gobbledygook")
        assert refusal.lexicon_version == LEXICON_VERSION


def test_a_bound_outcome_states_its_lexicon_version_on_a_copy_of_the_spec() -> None:
    outcome = _bind("current EUR")
    assert isinstance(outcome, BoundUnitRequest)
    spec = _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH")

    stated = with_unit_lexicon_version(spec, outcome)

    assert stated.resolved.unit_lexicon_version == LEXICON_VERSION
    assert spec.resolved.unit_lexicon_version is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("in constant 2024 euros", True),
        ("in euros", True),
        ("constant prices", True),
        ("in 2022", False),
        ("in zorkmids", False),
        ("in francs", False),
        ("in euros and yen", False),
        ("in euros per capita", False),
        ("in local currency", False),
        ("by sector", False),
    ],
)
def test_names_unit_request_is_whether_the_binder_accepts_the_phrase(text, expected):
    assert names_unit_request(text) is expected
    assert isinstance(bind_unit_text(text, policy=_policy()), BoundUnitRequest) is expected


@pytest.mark.parametrize(
    ("text", "currency", "named"),
    [
        ("Constant 2015 Francs", "XOF", "constant 2015 XOF"),
        ("kr", "SEK", "SEK"),
        # A code that spells an English word is replaced by its curated name.
        ("try", "TRY", "turkish lira"),
        ("local currency", IdentityCurrencyRole(role="destination"), "recipient currency"),
        ("own currency", IdentityCurrencyRole(role="origin"), "donor currency"),
        # No ambiguous words to replace.
        ("USD", "USD", None),
    ],
)
def test_unit_text_naming_replaces_the_ambiguous_words_with_ones_naming_the_choice(
    text, currency, named
) -> None:
    assert unit_text_naming(text, currency) == named
    if named is not None:
        rebound = _bind(named)
        assert not isinstance(rebound, UnitNotResolved)
        assert rebound.currency == currency


@pytest.mark.parametrize(
    ("text", "currency"),
    [
        *((f"in {code}", code) for code in sorted(AMBIGUOUS_ISO_CODES)),
        *(
            (f"in {name}", code)
            for name, codes in sorted(AMBIGUOUS_CURRENCY_NAMES.items())
            for code in codes
        ),
    ],
)
def test_every_refusal_alternative_has_words_that_bind_it(text, currency) -> None:
    refused = _bind(text)
    assert isinstance(refused, UnitNotResolved)
    assert currency in {alternative.currency for alternative in refused.alternatives}
    named = unit_text_naming(text, currency)
    assert named is not None
    rebound = _bind(named)
    assert isinstance(rebound, BoundUnitRequest)
    assert rebound.currency == currency
