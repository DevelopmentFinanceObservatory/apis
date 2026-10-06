"""Versioned lookup data for the unit binder: currencies, price phrases, and bounds.

``LEXICON_VERSION`` identifies this snapshot. Lookup keys are lowercase, with
multi-word phrases space-joined to match ``unit.bind`` normalisation. The ISO 4217 code set
the binder matches bare codes against lives in ``search.currency_codes``.
"""

from __future__ import annotations

LEXICON_VERSION = "unit_lexicon_v2"

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
# Curated unambiguous currency aliases. Direct ISO codes use the set above.
# Longest-match scan uses lowercase, space-joined phrases. Bare "real" and
# "franc" stay ambiguous. Country-qualified forms are listed below.
# ---------------------------------------------------------------------------

CURRENCY_ALIASES: dict[str, str] = {
    # US dollar
    "dollar": "USD",
    "dollars": "USD",
    "us dollar": "USD",
    "us dollars": "USD",
    "$": "USD",
    # Euro
    "euro": "EUR",
    "euros": "EUR",
    "€": "EUR",
    # British pound
    "pound": "GBP",
    "pounds": "GBP",
    "british pound": "GBP",
    "british pounds": "GBP",
    "pound sterling": "GBP",
    "sterling": "GBP",
    "£": "GBP",
    # Japanese yen
    "yen": "JPY",
    "japanese yen": "JPY",
    "¥": "JPY",
    # Chinese yuan / renminbi
    "yuan": "CNY",
    "renminbi": "CNY",
    "rmb": "CNY",
    "chinese yuan": "CNY",
    # Indian rupee
    "rupee": "INR",
    "rupees": "INR",
    "indian rupee": "INR",
    "indian rupees": "INR",
    "₹": "INR",
    # Swiss franc
    "swiss franc": "CHF",
    "swiss francs": "CHF",
    # Canadian / Australian dollar
    "canadian dollar": "CAD",
    "canadian dollars": "CAD",
    "australian dollar": "AUD",
    "australian dollars": "AUD",
    # Brazilian real
    "brazilian real": "BRL",
    "brazilian reais": "BRL",
    "brazilian reals": "BRL",
    # Southern/East Africa
    "rand": "ZAR",
    "south african rand": "ZAR",
    "kenyan shilling": "KES",
    "kenyan shillings": "KES",
    "naira": "NGN",
    "nigerian naira": "NGN",
    "birr": "ETB",
    "ethiopian birr": "ETB",
    # MENA
    "egyptian pound": "EGP",
    "egyptian pounds": "EGP",
    "saudi riyal": "SAR",
    "riyal": "SAR",
    "riyals": "SAR",
    "uae dirham": "AED",
    "dirham": "AED",
    "dirhams": "AED",
    # Latin America
    "mexican peso": "MXN",
    "mexican pesos": "MXN",
    "philippine peso": "PHP",
    "philippine pesos": "PHP",
    # Europe (non-euro)
    "ruble": "RUB",
    "rubles": "RUB",
    "rouble": "RUB",
    "roubles": "RUB",
    "russian ruble": "RUB",
    "russian rubles": "RUB",
    "swedish krona": "SEK",
    "swedish kronor": "SEK",
    "norwegian krone": "NOK",
    "norwegian kroner": "NOK",
    "danish krone": "DKK",
    "danish kroner": "DKK",
    "turkish lira": "TRY",
    "lira": "TRY",
    # Asia
    "won": "KRW",
    "korean won": "KRW",
    "south korean won": "KRW",
    "₩": "KRW",
    "rupiah": "IDR",
    "indonesian rupiah": "IDR",
    "vietnamese dong": "VND",
    "dong": "VND",
    "thai baht": "THB",
    "baht": "THB",
    "pakistani rupee": "PKR",
    "pakistani rupees": "PKR",
    "bangladeshi taka": "BDT",
    "taka": "BDT",
    # West/Central African CFA francs
    "cfa franc": "XOF",
    "west african cfa franc": "XOF",
    "west african cfa francs": "XOF",
    "central african cfa franc": "XAF",
    "central african cfa francs": "XAF",
}

# ---------------------------------------------------------------------------
# Bare ambiguous names refuse with these candidates. Qualified aliases resolve
# through CURRENCY_ALIASES, which the binder checks first.
# ---------------------------------------------------------------------------

AMBIGUOUS_CURRENCY_NAMES: dict[str, tuple[str, ...]] = {
    "franc": ("CHF", "XAF", "XOF"),
    "francs": ("CHF", "XAF", "XOF"),
    "kr": ("DKK", "NOK", "SEK"),
    "peso": ("MXN", "PHP"),
    "pesos": ("MXN", "PHP"),
}

# ---------------------------------------------------------------------------
# Display names for every code that can appear in an ambiguity refusal.
# Other ISO codes fall back to the code itself.
# ---------------------------------------------------------------------------

CURRENCY_DISPLAY_NAMES: dict[str, str] = {
    # AMBIGUOUS_CURRENCY_NAMES candidates.
    "CHF": "Swiss franc",
    "XAF": "Central African CFA franc",
    "XOF": "West African CFA franc",
    "DKK": "Danish krone",
    "NOK": "Norwegian krone",
    "SEK": "Swedish krona",
    "MXN": "Mexican peso",
    "PHP": "Philippine peso",
    # AMBIGUOUS_ISO_CODES: bare collision-prone codes (see that set's own
    # comment for the English word each collides with).
    "ALL": "Albanian lek",
    "BAM": "Bosnia-Herzegovina convertible mark",
    "BOB": "Bolivian boliviano",
    "COP": "Colombian peso",
    "CUP": "Cuban peso",
    "GEL": "Georgian lari",
    "MAD": "Moroccan dirham",
    "MOP": "Macanese pataca",
    "PEN": "Peruvian sol",
    "RUB": "Russian ruble",
    "SOS": "Somali shilling",
    "TOP": "Tongan paʻanga",
    "TRY": "Turkish lira",
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

# ---------------------------------------------------------------------------
# Four words cover the longest phrases in these tables, including the CFA names.
# ---------------------------------------------------------------------------

MAX_PHRASE_WORDS = 4
