"""Checks the unit lexicon's ISO codes, normalized keys, and non-overlapping phrases."""

from __future__ import annotations

from search.engine.unit.lexicon import (
    AMBIGUOUS_CURRENCY_NAMES,
    AMBIGUOUS_ISO_CODES,
    CONNECTIVE_WORDS,
    CURRENCY_ALIASES,
    CURRENCY_DISPLAY_NAMES,
    MAX_PHRASE_WORDS,
    PRICE_BASIS_PHRASES,
    ROLE_CURRENCY_PHRASES,
    UNSUPPORTED_PHRASES,
)
from unit_words.currency_codes import ISO_4217_CODES


def test_iso_codes_are_upper_case_three_letters() -> None:
    for code in ISO_4217_CODES:
        assert len(code) == 3, f"{code!r} is not 3 characters"
        assert code.isalpha(), f"{code!r} is not alphabetic"
        assert code == code.upper(), f"{code!r} is not upper-case"


def test_iso_codes_set_is_reasonably_complete() -> None:
    # Not exhaustive by design (see the module docstring), but should cover
    # the common ones a dev-finance/economics corpus will name.
    for code in ("USD", "EUR", "GBP", "JPY", "CNY", "INR", "KES", "NGN", "ETB", "ZAR"):
        assert code in ISO_4217_CODES


def test_currency_alias_keys_are_lower_case() -> None:
    for key in CURRENCY_ALIASES:
        assert key == key.casefold(), f"{key!r} is not casefolded"


def test_currency_alias_values_are_valid_iso_codes() -> None:
    for phrase, code in CURRENCY_ALIASES.items():
        assert code in ISO_4217_CODES, f"{phrase!r} maps to {code!r}, not a registered ISO code"


def test_ambiguous_iso_codes_are_a_subset_of_the_full_code_set() -> None:
    assert AMBIGUOUS_ISO_CODES <= ISO_4217_CODES


def test_ambiguous_iso_codes_do_not_appear_as_curated_alias_keys() -> None:
    # The case guard only applies to the bare-code fallback path; a curated
    # alias entry (none is expected to collide, but this pins it) would need
    # its own reasoning, not this one.
    alias_keys = set(CURRENCY_ALIASES)
    collisions = {code for code in AMBIGUOUS_ISO_CODES if code.lower() in alias_keys}
    assert not collisions, collisions


def test_ambiguous_currency_name_keys_are_lower_case() -> None:
    for key in AMBIGUOUS_CURRENCY_NAMES:
        assert key == key.casefold(), f"{key!r} is not casefolded"


def test_ambiguous_currency_name_alternatives_are_valid_iso_codes() -> None:
    for name, alternatives in AMBIGUOUS_CURRENCY_NAMES.items():
        assert alternatives, f"{name!r} lists no alternatives at all"
        for code in alternatives:
            assert code in ISO_4217_CODES, f"{name!r} lists {code!r}, not a registered ISO code"


def test_ambiguous_currency_names_do_not_appear_as_curated_alias_keys() -> None:
    # A curated alias always wins outright (see unit.bind._phrase_lookup);
    # this pins that the two tables never disagree about the same key.
    assert not (set(AMBIGUOUS_CURRENCY_NAMES) & set(CURRENCY_ALIASES))


def test_price_basis_phrase_keys_are_lower_case() -> None:
    for key in PRICE_BASIS_PHRASES:
        assert key == key.casefold()


def test_price_basis_values_are_current_or_constant() -> None:
    assert set(PRICE_BASIS_PHRASES.values()) <= {"current", "constant"}


def test_unsupported_phrase_keys_are_lower_case() -> None:
    for key in UNSUPPORTED_PHRASES:
        assert key == key.casefold()


def test_connective_words_are_lower_case() -> None:
    for word in CONNECTIVE_WORDS:
        assert word == word.casefold()


# ---------------------------------------------------------------------------
# Phrase tables have no overlapping keys.
# ---------------------------------------------------------------------------


def test_no_phrase_appears_in_two_tables() -> None:
    currency_keys = set(CURRENCY_ALIASES)
    ambiguous_keys = set(AMBIGUOUS_CURRENCY_NAMES)
    price_basis_keys = set(PRICE_BASIS_PHRASES)
    unsupported_keys = set(UNSUPPORTED_PHRASES)
    role_keys = set(ROLE_CURRENCY_PHRASES)

    assert not (currency_keys & price_basis_keys), currency_keys & price_basis_keys
    assert not (currency_keys & unsupported_keys), currency_keys & unsupported_keys
    assert not (currency_keys & ambiguous_keys), currency_keys & ambiguous_keys
    assert not (ambiguous_keys & price_basis_keys), ambiguous_keys & price_basis_keys
    assert not (ambiguous_keys & unsupported_keys), ambiguous_keys & unsupported_keys
    assert not (price_basis_keys & unsupported_keys), price_basis_keys & unsupported_keys
    assert not (role_keys & currency_keys), role_keys & currency_keys
    assert not (role_keys & ambiguous_keys), role_keys & ambiguous_keys
    assert not (role_keys & price_basis_keys), role_keys & price_basis_keys
    assert not (role_keys & unsupported_keys), role_keys & unsupported_keys


def test_no_phrase_key_collides_with_a_bare_iso_code() -> None:
    """A currency-alias or price-basis phrase that happens to spell a valid
    ISO code verbatim (e.g. a hypothetical alias "eur") would be redundant
    but not ambiguous -- _phrase_lookup checks the curated tables before the
    bare ISO-code fallback, so this isn't a correctness bug. Still worth
    pinning: an alias should never DISAGREE with what the code itself means."""
    for phrase, code in CURRENCY_ALIASES.items():
        if phrase.upper() in ISO_4217_CODES:
            assert phrase.upper() == code, (
                f"alias {phrase!r} disagrees with its own ISO code spelling: "
                f"maps to {code!r} but {phrase.upper()!r} is itself a registered code"
            )


def test_connective_words_do_not_collide_with_other_tables() -> None:
    connective = set(CONNECTIVE_WORDS)
    assert not (connective & set(CURRENCY_ALIASES))
    assert not (connective & set(AMBIGUOUS_CURRENCY_NAMES))
    assert not (connective & set(PRICE_BASIS_PHRASES))
    assert not (connective & set(UNSUPPORTED_PHRASES))
    assert not (connective & set(ROLE_CURRENCY_PHRASES))


# ---------------------------------------------------------------------------
# ROLE_CURRENCY_PHRASES: a currency asked for by spec role rather than by
# ISO code or name -- "beside" (local/national/own currency) or a fixed
# role ("origin"/"destination", recipient's/donor currency).
# ---------------------------------------------------------------------------


def test_role_currency_phrase_keys_are_lower_case() -> None:
    for key in ROLE_CURRENCY_PHRASES:
        assert key == key.casefold(), f"{key!r} is not casefolded"


def test_role_currency_phrase_values_are_beside_or_a_fixed_role() -> None:
    assert set(ROLE_CURRENCY_PHRASES.values()) <= {"beside", "origin", "destination"}


def test_role_currency_phrases_do_not_appear_as_curated_alias_keys() -> None:
    assert not (set(ROLE_CURRENCY_PHRASES) & set(CURRENCY_ALIASES))


def test_both_recipient_and_donor_have_a_fixed_role() -> None:
    # The two fixed-role phrases, plain and possessive ("recipient('s)
    # currency", "donor('s) currency") -- unit/bind.py's normalise stage
    # strips the apostrophe, so both spellings reduce to the table entries.
    assert ROLE_CURRENCY_PHRASES["recipient currency"] == "destination"
    assert ROLE_CURRENCY_PHRASES["recipients currency"] == "destination"
    assert ROLE_CURRENCY_PHRASES["donor currency"] == "origin"
    assert ROLE_CURRENCY_PHRASES["donors currency"] == "origin"


def test_beside_phrases_cover_local_national_and_own() -> None:
    for phrase in ("local currency", "national currency", "own currency"):
        assert ROLE_CURRENCY_PHRASES[phrase] == "beside"


# ---------------------------------------------------------------------------
# CURRENCY_DISPLAY_NAMES: labels for the codes a bind_unit_text refusal can
# list in its alternatives. Every code that can actually appear there
# (AMBIGUOUS_ISO_CODES and every AMBIGUOUS_CURRENCY_NAMES candidate) must
# have a curated entry -- a missing one would fall back to the bare code as
# its own label, which is exactly the tautological "did you mean TOP?"
# outcome curating this table exists to avoid.
# ---------------------------------------------------------------------------


def test_currency_display_name_keys_are_valid_iso_codes() -> None:
    for code in CURRENCY_DISPLAY_NAMES:
        assert code in ISO_4217_CODES, f"{code!r} is not a registered ISO code"


def test_currency_display_names_are_non_empty_strings() -> None:
    for code, name in CURRENCY_DISPLAY_NAMES.items():
        assert isinstance(name, str) and name, f"{code!r} has no display name"


def test_every_alternatives_candidate_has_a_curated_display_name() -> None:
    candidates = set(AMBIGUOUS_ISO_CODES)
    for alternatives in AMBIGUOUS_CURRENCY_NAMES.values():
        candidates.update(alternatives)
    missing = candidates - set(CURRENCY_DISPLAY_NAMES)
    assert not missing, f"no curated display name for: {missing}"


def test_max_phrase_words_covers_every_multi_word_entry() -> None:
    longest = max(
        (
            len(phrase.split())
            for phrase in (
                *CURRENCY_ALIASES,
                *AMBIGUOUS_CURRENCY_NAMES,
                *PRICE_BASIS_PHRASES,
                *UNSUPPORTED_PHRASES,
                *ROLE_CURRENCY_PHRASES,
            )
        ),
        default=1,
    )
    assert longest == MAX_PHRASE_WORDS, (
        f"the longest phrase is {longest} words but MAX_PHRASE_WORDS={MAX_PHRASE_WORDS} -- "
        "the classify stage's scan window would silently miss it"
    )
