"""Versioned lookup data for the unit binder: currencies, price phrases, and bounds.

``LEXICON_VERSION`` identifies this snapshot. Lookup keys are lowercase, with
multi-word phrases space-joined to match ``unit.bind`` normalisation. The ISO 4217 code set
the binder matches bare codes against lives in ``unit_words.currency_codes``.
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict

from unit_words.currency_nouns import CURRENCY_NOUNS, CURRENCY_SINGULAR_NOUNS

LEXICON_VERSION = "unit_lexicon_v4"

# ---------------------------------------------------------------------------
# ISO codes that also spell common English words refuse as bare tokens. The
# refusal lists the code as an alternative. The binder ignores case for collisions.
# This curated list favours precision and may miss less common collisions.
# ---------------------------------------------------------------------------

AMBIGUOUS_ISO_CODES: frozenset[str] = frozenset(
    {
        "ALL",  # Albanian lek / "all"
        "BAM",  # Bosnia-Herzegovina mark / "bam"
        "BOB",  # Bolivian boliviano / "bob"
        "COP",  # Colombian peso / "cop"
        "CUP",  # Cuban peso / "cup"
        "GEL",  # Georgian lari / "gel"
        "MAD",  # Moroccan dirham / "mad"
        "MOP",  # Macanese pataca / "mop"
        "PEN",  # Peruvian sol / "pen"
        "RUB",  # Russian ruble / "rub"
        "SOS",  # Somali shilling / "sos"
        "TOP",  # Tongan pa'anga / "top"
        "TRY",  # Turkish lira / "try"
    }
)

# ---------------------------------------------------------------------------
# Currency names. Each code's nouns in `unit_words.currency_nouns`, plural and
# singular ("Malawian kwachas", "Malawian kwacha"), name it alone, and so does
# their last word ("kwachas", "kwacha") when no other code shares it. A last word
# that several codes share names all of them and refuses with each as an
# alternative ("francs", "kwacha"). `_derived_names` builds both tables at import.
# ---------------------------------------------------------------------------

# Names the nouns do not give: symbols, other spellings and short names, and the
# readings that win over a shared last word ("dollars" is the US dollar).
_CURATED_ALIASES: dict[str, str] = {
    # A shared last word with a conventional reading.
    "dollar": "USD",
    "dollars": "USD",
    "pound": "GBP",
    "pounds": "GBP",
    "rupee": "INR",
    "rupees": "INR",
    "riyal": "SAR",
    "riyals": "SAR",
    "dirham": "AED",
    "dirhams": "AED",
    "ruble": "RUB",
    "rubles": "RUB",
    "won": "KRW",
    # Symbols.
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "₩": "KRW",
    # Other spellings and short names.
    "pound sterling": "GBP",
    "sterling": "GBP",
    "renminbi": "CNY",
    "rmb": "CNY",
    "brazilian reals": "BRL",
    "rouble": "RUB",
    "roubles": "RUB",
    "korean won": "KRW",
    "cfa franc": "XOF",
}

_CURATED_AMBIGUOUS: dict[str, tuple[str, ...]] = {
    "kr": ("DKK", "NOK", "SEK"),
}

# ---------------------------------------------------------------------------
# Role currencies map to origin/destination or "beside". Fixed roles bind even
# without an entity. The binder resolves "beside" against bound roles. Apostrophes
# are removed, so possessives match the same keys.
# ---------------------------------------------------------------------------

ROLE_CURRENCY_PHRASES: dict[str, str] = {
    "local currency": "beside",
    "national currency": "beside",
    "own currency": "beside",
    "recipient currency": "destination",
    "recipients currency": "destination",
    "donor currency": "origin",
    "donors currency": "origin",
}

# ---------------------------------------------------------------------------
# Price-basis phrases map to current or constant prices. The binder uses
# longest-match scanning.
# ---------------------------------------------------------------------------

PRICE_BASIS_PHRASES: dict[str, str] = {
    "current prices": "current",
    "current terms": "current",
    "nominal terms": "current",
    "constant prices": "constant",
    "real terms": "constant",
    "inflation adjusted": "constant",
    "current": "current",
    "nominal": "current",
    "constant": "constant",
    "real": "constant",
}

# ---------------------------------------------------------------------------
# Recognised phrases outside the request schema yield unit_request_unsupported.
# ---------------------------------------------------------------------------

UNSUPPORTED_PHRASES: dict[str, str] = {
    "purchasing power parity": "purchasing power parity (PPP) terms are not representable in the currency/prices/base_year request form",
    "per capita": "a per-capita unit is not representable in the currency/prices/base_year request form",
    "percapita": "a per-capita unit is not representable in the currency/prices/base_year request form",
    "ppp": "purchasing power parity (PPP) terms are not representable in the currency/prices/base_year request form",
    "percent": "a percent/share unit is not representable in the currency/prices/base_year request form",
    "%": "a percent/share unit is not representable in the currency/prices/base_year request form",
    "million": "a scale (million/billion/thousand) is not representable in the currency/prices/base_year request form",
    "millions": "a scale (million/billion/thousand) is not representable in the currency/prices/base_year request form",
    "billion": "a scale (million/billion/thousand) is not representable in the currency/prices/base_year request form",
    "billions": "a scale (million/billion/thousand) is not representable in the currency/prices/base_year request form",
    "thousand": "a scale (million/billion/thousand) is not representable in the currency/prices/base_year request form",
    "thousands": "a scale (million/billion/thousand) is not representable in the currency/prices/base_year request form",
}

# ---------------------------------------------------------------------------
# Only these connectives are ignored. Other unknown tokens refuse the phrase.
# The small allowlist prevents meaningful cues such as an empty "base year"
# request from disappearing as stopwords.
# ---------------------------------------------------------------------------

CONNECTIVE_WORDS: frozenset[str] = frozenset(
    {
        "in",
        "at",
        "prices",
        "terms",
        "of",
        "the",
        # In "constant prices for years 2015", neither word forms a base-year
        # cue, so both must be consumed as connectives.
        "for",
        "years",
    }
)

# ---------------------------------------------------------------------------
# Year-token bounds: a bare 4-digit run classifies as a year only inside this
# range (bounds a plain digit run like "1000" or "9999" from misclassifying).
# ---------------------------------------------------------------------------

MIN_YEAR = 1900
MAX_YEAR = 2100


def fold_marks(text: str) -> str:
    """`text` without combining marks, so "krónur" and "kronur" read alike."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def _name_key(name: str) -> str:
    """A currency noun as the binder reads it: no marks, apostrophes or okina, hyphens as
    spaces, casefolded, words space-joined."""
    folded = fold_marks(name).casefold()
    for mark in ("'", "’", "\u02bb", "."):
        folded = folded.replace(mark, "")
    return " ".join(folded.replace("-", " ").split())


def _derived_names() -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """The names `CURRENCY_NOUNS` and `CURRENCY_SINGULAR_NOUNS` give: each full noun names
    its code, and each last word names every code whose singular last word (its lemma)
    it shares a code with. A key another table reads ("real") is left out."""
    taken = {*PRICE_BASIS_PHRASES, *UNSUPPORTED_PHRASES, *ROLE_CURRENCY_PHRASES, *CONNECTIVE_WORDS}
    full: dict[str, str] = {}
    lemma_codes: defaultdict[str, set[str]] = defaultdict(set)
    code_heads: dict[str, set[str]] = {}
    for code, plural in CURRENCY_NOUNS.items():
        keys = (_name_key(plural), _name_key(CURRENCY_SINGULAR_NOUNS[code]))
        full.update(dict.fromkeys(keys, code))
        lemma_codes[keys[1].rsplit(" ", 1)[-1]].add(code)
        code_heads[code] = {key.rsplit(" ", 1)[-1] for key in keys}
    head_codes: defaultdict[str, set[str]] = defaultdict(set)
    for code, heads in code_heads.items():
        sharing = lemma_codes[_name_key(CURRENCY_SINGULAR_NOUNS[code]).rsplit(" ", 1)[-1]]
        for head in heads:
            head_codes[head] |= sharing
    aliases = dict(full)
    ambiguous: dict[str, tuple[str, ...]] = {}
    for head, codes in head_codes.items():
        if len(codes) == 1:
            aliases.setdefault(head, *codes)
        else:
            ambiguous[head] = tuple(sorted(codes))
    return (
        {key: code for key, code in aliases.items() if key not in taken},
        {key: codes for key, codes in ambiguous.items() if key not in taken},
    )


_DERIVED_ALIASES, _DERIVED_AMBIGUOUS = _derived_names()

# Every name that binds one currency. A curated name wins over a shared last word.
CURRENCY_ALIASES: dict[str, str] = {**_DERIVED_ALIASES, **_CURATED_ALIASES}

# Every name that refuses with its codes as alternatives.
AMBIGUOUS_CURRENCY_NAMES: dict[str, tuple[str, ...]] = {
    **{key: codes for key, codes in _DERIVED_AMBIGUOUS.items() if key not in CURRENCY_ALIASES},
    **_CURATED_AMBIGUOUS,
}

# The name `region.facets` labels a currency by when the graph gives none: each code
# that spells an English word ("TOP"), and eight currencies a bare "franc", "kr" or
# "peso" commonly means. Any other code is labelled by itself.
CURRENCY_DISPLAY_NAMES: dict[str, str] = {
    code: CURRENCY_SINGULAR_NOUNS[code]
    for code in sorted(
        AMBIGUOUS_ISO_CODES | {"CHF", "DKK", "MXN", "NOK", "PHP", "SEK", "XAF", "XOF"}
    )
}

# The longest phrase in these tables bounds the binder's longest-match scan window.
MAX_PHRASE_WORDS = max(
    len(phrase.split())
    for phrase in (
        *CURRENCY_ALIASES,
        *AMBIGUOUS_CURRENCY_NAMES,
        *PRICE_BASIS_PHRASES,
        *UNSUPPORTED_PHRASES,
        *ROLE_CURRENCY_PHRASES,
    )
)
