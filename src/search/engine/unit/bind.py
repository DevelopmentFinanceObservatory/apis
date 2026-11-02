"""Map unit text to ``{currency, prices, base_year}`` or a typed refusal.

The pure, deterministic binder runs five stages in order:

1. ``_normalise`` applies NFKC, casefolding, and punctuation flattening.
2. ``_tokenise`` splits symbols, digits, and letters. ``_erased_spans`` sends
   unrecognised characters to classification as unresolved content. Separators
   are whitespace and declared punctuation.
3. ``_classify`` scans longest phrases first.
   Every token must match. Otherwise the phrase refuses. Bare ambiguous
   currency names and collision-prone ISO codes refuse with alternatives.
   Qualified aliases resolve directly. Role currencies resolve against bound
   roles during assembly.
4. ``_assemble`` applies the rules below.
5. ``_complete`` fills unnamed members from declared sources: prices read as
   current, the currency comes from ``UnitDefaultPolicy.currency``, and a
   constant-price base year from ``UnitDefaultPolicy.base_year``.

Assembly rule table (``_assemble``), in the order applied:

- Unsupported phrases take priority over unresolved tokens. Unresolved tokens,
  including erased characters, yield ``unit_not_resolved``.
- Multiple distinct currencies, price bases, or years yield
  ``unit_not_resolved``. An ISO code and role-currency phrase count as separate
  currencies.
- Bare ambiguous currency names and collision-prone ISO codes yield
  ``unit_not_resolved`` with candidate ISO codes in ``alternatives``.
- "Beside" currencies resolve to the sole bound role. Zero roles refuse without
  alternatives, and multiple roles refuse with one role shape per bound role.
  Recipient/donor currencies bind to destination/origin regardless of
  ``bound_roles``.
- A year with current prices, or a "base year" cue without a year value,
  yields ``unit_not_resolved``. A base year requires constant prices.
- A year beside a currency, or beside "prices", implies constant prices. A bare
  year without either cue or a price-basis keyword yields ``unit_not_resolved``.
  A bare year signals a time window.
- Empty or connective-only phrases yield ``unit_not_resolved`` before completion.
  Otherwise, named members bind and the rest are completed. A request that names
  its own year, or asks for current prices, skips the base-year source.

``unit_request_unsupported`` marks a recognised request outside the schema,
such as PPP, per-capita, or percentage units. Parse failures yield
``unit_not_resolved``.

``request_basis`` records each member as ``bound`` (the text named it) or
``declared_default`` (completion filled it). Only request values enter the
hash. ``request_basis`` is metadata.
``request_hash`` is ``unit_`` plus 32 hex digits from
RFC 8785 canonical JSON and SHA-256 over the completed request. Validation runs
before hashing. ``LEXICON_VERSION`` accompanies every outcome, and
``with_unit_lexicon_version`` states it on the Spec the outcome bound.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, cast

from dfo_spec_kit import identity

from code_index.labels import PricesKind
from materialise.roles import axis_noun
from search.models import (
    EntityRoleKey,
    IdentityCurrency,
    IdentityCurrencyRole,
    Spec,
)
from unit_words.currency_codes import ISO_4217_CODES
from unit_words.currency_nouns import CURRENCY_SINGULAR_NOUNS

from .lexicon import (
    AMBIGUOUS_CURRENCY_NAMES,
    AMBIGUOUS_ISO_CODES,
    CONNECTIVE_WORDS,
    CURRENCY_ALIASES,
    LEXICON_VERSION,
    MAX_PHRASE_WORDS,
    MAX_YEAR,
    MIN_YEAR,
    PRICE_BASIS_PHRASES,
    ROLE_CURRENCY_PHRASES,
    UNSUPPORTED_PHRASES,
    fold_marks,
)

__all__ = [
    "LEXICON_VERSION",
    "BindOutcome",
    "BoundUnitRequest",
    "CurrencyAlternative",
    "PricesKind",
    "RequestBasis",
    "UnitDefaultPolicy",
    "UnitDefaultPolicyError",
    "UnitNotResolved",
    "UnitRequestUnsupported",
    "bind_unit_text",
    "names_price_basis",
    "names_unit_request",
    "unit_text_naming",
    "with_unit_lexicon_version",
]

# The binder emits "bound" and "declared_default". "fallback" remains valid because
# a promoted or resubmitted spec carries the request_basis it was minted with.
RequestBasis = Literal["bound", "declared_default", "fallback"]

# A missing price basis reads as current, the contract's completion rule in every
# deployment.
_DEFAULT_PRICES: PricesKind = "current"


class UnitDefaultPolicyError(ValueError):
    """Policy validation error, a ``ValueError``. ``member`` names the invalid member."""

    def __init__(self, member: str, message: str) -> None:
        self.member = member
        super().__init__(message)


@dataclass(frozen=True)
class UnitDefaultPolicy:
    """The declared sources for members the text leaves open.

    ``currency`` is the deployment's configured default ISO 4217 code.
    ``base_year`` returns the base year the live conversion tables declare.
    The binder calls it only for a constant-price ask that names no year and
    lets whatever it raises propagate, so a source that cannot establish the
    year refuses the request.
    """

    currency: str
    base_year: Callable[[], int]

    def __post_init__(self) -> None:
        if self.currency not in ISO_4217_CODES:
            raise UnitDefaultPolicyError(
                "currency",
                f"UnitDefaultPolicy.currency must be a registered ISO 4217 code; "
                f"got {self.currency!r}",
            )


@dataclass(frozen=True)
class BoundUnitRequest:
    """A completed request, its completion provenance, and its hash.

    ``currency`` is an ISO 4217 code or a role-bound currency.
    """

    currency: str | IdentityCurrencyRole
    prices: PricesKind
    base_year: int | None
    request_basis: dict[str, RequestBasis]
    lexicon_version: str
    request_hash: str
    code: Literal["bound"] = "bound"

    def as_wire_request(self) -> dict[str, object]:
        """The `{currency, prices, base_year}` dict, base_year omitted
        unless prices=="constant" -- exactly the member set
        ``dfo_spec_kit.identity``'s unit-request validation expects, and
        what a caller wires into ``facets.unit.request``."""
        return _wire_request_dict(self.currency, self.prices, self.base_year)


@dataclass(frozen=True)
class CurrencyAlternative:
    """A refusal candidate with a display label.

    ``currency`` is the value a caller resubmits as the request's currency. An ISO
    code carries its singular noun, and a role shape carries the role's
    reader-facing axis noun.
    """

    currency: IdentityCurrency
    label: str


@dataclass(frozen=True)
class UnitNotResolved:
    """Refusal for unmatched or conflicting unit text. Text doors return it as ``no_data``.

    ``alternatives`` lists ambiguous currencies or roles when available.
    """

    message: str
    alternatives: tuple[CurrencyAlternative, ...] = ()
    lexicon_version: str = LEXICON_VERSION
    code: Literal["unit_not_resolved"] = "unit_not_resolved"


@dataclass(frozen=True)
class UnitRequestUnsupported:
    """Refusal for a recognised unit outside the request schema.

    Text doors return HTTP 200 with ``unsupported``. Parse failures use ``UnitNotResolved``.
    """

    message: str
    lexicon_version: str = LEXICON_VERSION
    code: Literal["unit_request_unsupported"] = "unit_request_unsupported"


BindOutcome = BoundUnitRequest | UnitNotResolved | UnitRequestUnsupported


# ---------------------------------------------------------------------------
# Stage 1: normalise
# ---------------------------------------------------------------------------

_DASH_TRANSLATION = str.maketrans(dict.fromkeys("‐‑‒–—−", "-"))


def _normalise(text: str, *, casefold: bool = True) -> str:
    # Folding marks makes "córdobas" match the lexicon's names, which carry none.
    normalised = fold_marks(text).translate(_DASH_TRANSLATION)
    if casefold:
        normalised = normalised.casefold()
    normalised = normalised.replace(".", "")
    # Apostrophe removal makes possessives match the lexicon's plain role phrases, and the
    # okina (U+02BB) of the Tongan "pa'anga" display name match its plain spelling.
    normalised = normalised.replace("'", "").replace("’", "").replace("\u02bb", "")
    normalised = normalised.replace("-", " ").replace("_", " ")
    return normalised


# ---------------------------------------------------------------------------
# Stage 2: tokenise
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[$€£¥₹₩%]|[0-9]+|[a-z]+", re.IGNORECASE)

# Declared separators are consumed. _erased_spans refuses other unmatched spans.
_SEPARATOR_CHARACTERS = " \t\n\r\v\f,()[]{}:;/"
_SEPARATOR_RE = re.compile(f"[{re.escape(_SEPARATOR_CHARACTERS)}]+")


def _tokenise(normalised: str) -> list[str]:
    return _TOKEN_RE.findall(normalised)


def _split_erasure(text: str) -> list[str]:
    """Split ``text`` into maximal runs of non-separator characters."""
    return [run for run in _SEPARATOR_RE.split(text) if run]


def _erased_spans(normalised: str) -> list[str]:
    """Return unmatched, non-separator spans for classification as unresolved tokens."""
    erased: list[str] = []
    pos = 0
    for match in _TOKEN_RE.finditer(normalised):
        erased.extend(_split_erasure(normalised[pos : match.start()]))
        pos = match.end()
    erased.extend(_split_erasure(normalised[pos:]))
    return erased


# ---------------------------------------------------------------------------
# Stage 3: classify
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ClassifyOutcome:
    currencies: tuple[str, ...]
    price_bases: tuple[str, ...]
    years: tuple[int, ...]
    unresolved: tuple[str, ...]
    unsupported_reason: str | None
    ambiguous_alternatives: tuple[str, ...] = ()
    base_year_cue: bool = False
    prices_token: bool = False
    # "beside" or fixed role per match. _assemble resolves "beside" against bound roles.
    role_currency_mentions: tuple[str, ...] = ()
    # The (start, length) in tokens of the first ambiguous currency mention, else of the
    # first "beside" role currency mention: the words a refusal's alternatives replace.
    choice_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class _LookupHit:
    kind: Literal["unsupported", "currency", "ambiguous_currency", "price_basis", "role_currency"]
    value: str = ""
    alternatives: tuple[str, ...] = ()


def _phrase_lookup(phrase: str) -> _LookupHit | None:
    """Classify one phrase. Explicit aliases precede ambiguous names."""
    if phrase in UNSUPPORTED_PHRASES:
        return _LookupHit("unsupported", UNSUPPORTED_PHRASES[phrase])
    if phrase in CURRENCY_ALIASES:
        return _LookupHit("currency", CURRENCY_ALIASES[phrase])
    if phrase in AMBIGUOUS_CURRENCY_NAMES:
        return _LookupHit("ambiguous_currency", alternatives=AMBIGUOUS_CURRENCY_NAMES[phrase])
    if phrase in PRICE_BASIS_PHRASES:
        return _LookupHit("price_basis", PRICE_BASIS_PHRASES[phrase])
    if phrase in ROLE_CURRENCY_PHRASES:
        return _LookupHit("role_currency", ROLE_CURRENCY_PHRASES[phrase])
    upper = phrase.upper()
    if upper in ISO_4217_CODES:
        if upper in AMBIGUOUS_ISO_CODES:
            # Refuse collision-prone codes with their curated display label as an alternative.
            return _LookupHit("ambiguous_currency", alternatives=(upper,))
        return _LookupHit("currency", upper)
    return None


def _currency_alternative(code: str) -> CurrencyAlternative:
    """Return an ISO code with its singular noun as label, or the code itself as fallback."""
    return CurrencyAlternative(currency=code, label=CURRENCY_SINGULAR_NOUNS.get(code, code))


def _role_alternative(role: EntityRoleKey) -> CurrencyAlternative:
    """One spec role -> a CurrencyAlternative for a role-bearing
    unit_not_resolved refusal. ``currency`` is the role shape a caller resubmits
    and ``label`` the noun ``materialise.roles.axis_noun`` gives that role."""
    return CurrencyAlternative(currency=IdentityCurrencyRole(role=role), label=axis_noun(role))


def _is_year_token(token: str) -> bool:
    return len(token) == 4 and token.isdigit() and MIN_YEAR <= int(token) <= MAX_YEAR


def _classify(tokens: list[str]) -> _ClassifyOutcome:
    currencies: list[str] = []
    price_bases: list[str] = []
    years: list[int] = []
    unresolved: list[str] = []
    unsupported_reason: str | None = None
    ambiguous_alternatives: tuple[str, ...] = ()
    base_year_cue = False
    prices_token = False
    role_currency_mentions: list[str] = []
    ambiguous_span: tuple[int, int] | None = None
    beside_span: tuple[int, int] | None = None

    i = 0
    n = len(tokens)
    while i < n:
        max_len = min(MAX_PHRASE_WORDS, n - i)
        hit: _LookupHit | None = None
        matched_len = 0
        for length in range(max_len, 0, -1):
            phrase = " ".join(tokens[i : i + length])
            hit = _phrase_lookup(phrase)
            if hit is not None:
                matched_len = length
                break

        if hit is not None:
            if hit.kind == "currency":
                currencies.append(hit.value)
            elif hit.kind == "price_basis":
                price_bases.append(hit.value)
            elif hit.kind == "unsupported":
                if unsupported_reason is None:
                    unsupported_reason = hit.value
            elif hit.kind == "ambiguous_currency":
                # Report alternatives for the first ambiguous mention only.
                if not ambiguous_alternatives:
                    ambiguous_alternatives = hit.alternatives
                    ambiguous_span = (i, matched_len)
            elif hit.kind == "role_currency":
                role_currency_mentions.append(hit.value)
                if hit.value == "beside" and beside_span is None:
                    beside_span = (i, matched_len)
            i += matched_len
            continue

        token = tokens[i]
        if token == "base" and i + 1 < n and tokens[i + 1] in ("year", "years"):
            # The year value may be a separate token. _assemble checks for it.
            base_year_cue = True
            i += 2
            continue
        if token == "prices":
            # _assemble distinguishes "2015 prices" from a bare year.
            prices_token = True
            i += 1
            continue
        if token in CONNECTIVE_WORDS:
            i += 1
            continue
        if _is_year_token(token):
            years.append(int(token))
            i += 1
            continue
        unresolved.append(token)
        i += 1

    return _ClassifyOutcome(
        currencies=tuple(currencies),
        price_bases=tuple(price_bases),
        years=tuple(years),
        unresolved=tuple(unresolved),
        unsupported_reason=unsupported_reason,
        ambiguous_alternatives=ambiguous_alternatives,
        base_year_cue=base_year_cue,
        prices_token=prices_token,
        role_currency_mentions=tuple(role_currency_mentions),
        choice_span=ambiguous_span or beside_span,
    )


# ---------------------------------------------------------------------------
# Stage 4: assemble
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Assembled:
    currency: str | IdentityCurrencyRole | None
    prices: PricesKind | None
    base_year: int | None


def _assemble(
    outcome: _ClassifyOutcome, *, bound_roles: frozenset[EntityRoleKey] = frozenset()
) -> _Assembled | UnitNotResolved | UnitRequestUnsupported:
    """Assemble classified values. ``bound_roles`` resolve "beside" currencies.

    Recipient/donor phrases bind to their fixed role.
    """
    if outcome.unsupported_reason is not None:
        return UnitRequestUnsupported(outcome.unsupported_reason)
    if outcome.unresolved:
        joined = ", ".join(repr(t) for t in outcome.unresolved)
        return UnitNotResolved(f"unrecognised unit token(s): {joined}")

    distinct_currencies = sorted(set(outcome.currencies))
    distinct_price_bases = sorted(set(outcome.price_bases))
    distinct_years = sorted(set(outcome.years))
    distinct_role_mentions = sorted(set(outcome.role_currency_mentions))

    if len(distinct_currencies) + len(distinct_role_mentions) > 1:
        # A role-currency phrase counts as a currency mention here, the same
        # as an ISO code: one bare "USD" plus one "local currency" is two
        # distinct currencies named, refused the same way "USD EUR" is.
        named = ", ".join((*distinct_currencies, *distinct_role_mentions))
        return UnitNotResolved(
            f"multiple distinct currencies named ({named}); "
            "the request form carries exactly one currency"
        )
    if len(distinct_price_bases) > 1:
        return UnitNotResolved(
            "conflicting price basis named (both current and constant); "
            "the request form carries exactly one prices value"
        )
    if len(distinct_years) > 1:
        return UnitNotResolved(
            f"multiple distinct base years named ({', '.join(map(str, distinct_years))}); "
            "the request form carries at most one base_year"
        )
    if outcome.ambiguous_alternatives:
        return UnitNotResolved(
            "an ambiguous currency was named with no declared alias to resolve it",
            alternatives=tuple(
                _currency_alternative(code) for code in outcome.ambiguous_alternatives
            ),
        )

    role_currency: IdentityCurrencyRole | None = None
    if distinct_role_mentions:
        # The conflict check leaves at most one distinct role mention.
        role_mention = distinct_role_mentions[0]
        if role_mention == "beside":
            if len(bound_roles) == 1:
                role_currency = IdentityCurrencyRole(role=next(iter(bound_roles)))
            elif not bound_roles:
                return UnitNotResolved(
                    "a role-relative currency phrase ('local currency'/'national "
                    "currency'/'own currency') was named, but the query binds no "
                    "entity to any role to read it from"
                )
            else:
                return UnitNotResolved(
                    "a role-relative currency phrase was named beside entities "
                    "bound on two roles, with no single role for it to mean",
                    alternatives=tuple(_role_alternative(role) for role in sorted(bound_roles)),
                )
        else:
            # Shape validation checks whether the fixed role is legal for the Spec.
            role_currency = IdentityCurrencyRole(role=cast("EntityRoleKey", role_mention))

    currency: str | IdentityCurrencyRole | None = (
        distinct_currencies[0] if distinct_currencies else role_currency
    )
    # distinct_price_bases only ever holds PRICE_BASIS_PHRASES values, which
    # test_unit_lexicon.py pins to {"current", "constant"} -- cast rather
    # than widen the annotation, since every OTHER member is meant to stay
    # narrowed to PricesKind.
    prices: PricesKind | None = cast(
        "PricesKind | None", distinct_price_bases[0] if distinct_price_bases else None
    )
    year = distinct_years[0] if distinct_years else None

    if prices == "current" and year is not None:
        return UnitNotResolved(
            "a base year was named alongside current prices; base_year is only "
            "valid when prices is constant"
        )

    if outcome.base_year_cue and year is None:
        return UnitNotResolved(
            "a base year was named ('base year'/'base years') without supplying one"
        )

    if prices is None and year is not None:
        if currency is not None or outcome.prices_token:
            # A year with a currency or "prices" implies constant prices.
            prices = "constant"
        else:
            # A bare year carries time intent, so unit binding refuses it.
            return UnitNotResolved(
                "a bare year names a time window, not a price basis; pair it "
                "with a currency, a price-basis keyword, or the word "
                "'prices' to bind a base year"
            )

    base_year = year if prices == "constant" else None

    if currency is None and prices is None and base_year is None:
        # Empty asks refuse before completion.
        return UnitNotResolved("the phrase named no currency, price basis, or base year to bind")

    return _Assembled(currency=currency, prices=prices, base_year=base_year)


# ---------------------------------------------------------------------------
# Stage 5: complete
# ---------------------------------------------------------------------------


def _currency_wire_value(currency: str | IdentityCurrencyRole) -> object:
    """The completed currency member as it belongs on the wire: an ISO code
    string as-is, or ``{"role": ...}`` for a role-bound currency -- the
    shape ``dfo_spec_kit.identity``'s own ``_build_currency`` and
    ``search.models.IdentityCurrency`` both accept."""
    if isinstance(currency, IdentityCurrencyRole):
        return {"role": currency.role}
    return currency


def _wire_request_dict(
    currency: str | IdentityCurrencyRole, prices: PricesKind, base_year: int | None
) -> dict[str, object]:
    request: dict[str, object] = {"currency": _currency_wire_value(currency), "prices": prices}
    if base_year is not None:
        request["base_year"] = base_year
    return request


def _hash_completed_request(
    currency: str | IdentityCurrencyRole, prices: PricesKind, base_year: int | None
) -> str:
    canonical = identity.canonicalize(_wire_request_dict(currency, prices, base_year))
    digest = hashlib.sha256(canonical).hexdigest()
    return f"unit_{digest[:32]}"


def _validate_completed_request(
    currency: str | IdentityCurrencyRole, prices: PricesKind, base_year: int | None
) -> None:
    """Validate before hashing. Schema errors propagate as internal failures."""
    doc = {
        "identity_version": "spec1",
        # A minimal pin-arm shape lets the identity core validate the request.
        "shape": {"kind": "stat_var", "id": "UnitBindValidation"},
        "slots": [],
        "entities": {},
        "requested_window": {"kind": "all"},
        "facets": {
            "unit": {
                "state": "value",
                "request": _wire_request_dict(currency, prices, base_year),
            }
        },
    }
    identity.validate_identity_core(doc)


def _complete(assembled: _Assembled, *, policy: UnitDefaultPolicy) -> BoundUnitRequest:
    """Fill each open member from its declared source and record its basis."""
    basis: dict[str, RequestBasis] = {}

    def _member[T](name: str, bound_value: T | None, complete: Callable[[], T]) -> T:
        if bound_value is not None:
            basis[name] = "bound"
            return bound_value
        basis[name] = "declared_default"
        return complete()

    currency = _member("currency", assembled.currency, lambda: policy.currency)
    prices = _member("prices", assembled.prices, lambda: _DEFAULT_PRICES)
    base_year = (
        _member("base_year", assembled.base_year, policy.base_year)
        if prices == "constant"
        else None
    )

    _validate_completed_request(currency, prices, base_year)
    return BoundUnitRequest(
        currency=currency,
        prices=prices,
        base_year=base_year,
        request_basis=basis,
        lexicon_version=LEXICON_VERSION,
        request_hash=_hash_completed_request(currency, prices, base_year),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _parse(
    text: str, *, bound_roles: frozenset[EntityRoleKey]
) -> _Assembled | UnitNotResolved | UnitRequestUnsupported:
    """Stages 1 to 4: the members `text` names, or the refusal."""
    normalised = _normalise(text)
    outcome = _classify(_tokenise(normalised))
    if erased := _erased_spans(normalised):
        outcome = replace(outcome, unresolved=outcome.unresolved + tuple(erased))
    return _assemble(outcome, bound_roles=bound_roles)


def bind_unit_text(
    text: str,
    *,
    policy: UnitDefaultPolicy,
    bound_roles: frozenset[EntityRoleKey] = frozenset(),
) -> BindOutcome:
    """Parse and complete one unit phrase using ``policy`` and bound entity roles.

    The same arguments, and the same year from ``policy.base_year``, produce the
    same outcome and request hash. A "beside" currency uses the sole bound role.
    Fixed-role currencies keep their named axis. Callers dispatch on ``.code`` and
    state a bound outcome's lexicon version with ``with_unit_lexicon_version``. Whatever
    ``policy.base_year`` raises propagates.
    Applies no length cap; a caller exposing it to request bodies must enforce one.
    """
    assembled = _parse(text, bound_roles=bound_roles)
    if not isinstance(assembled, _Assembled):
        return assembled
    return _complete(assembled, policy=policy)


def _phrases_naming(currency: IdentityCurrency) -> list[str]:
    """The lexicon's words that name `currency` alone: an ISO code that collides with no
    English word, then the code's names (`CURRENCY_ALIASES`); for a role shape, its
    fixed-role phrases."""
    if isinstance(currency, str):
        code = [] if currency in AMBIGUOUS_ISO_CODES else [currency]
        aliases = [alias for alias, named in CURRENCY_ALIASES.items() if named == currency]
        return [*code, *aliases]
    return [phrase for phrase, role in ROLE_CURRENCY_PHRASES.items() if role == currency.role]


def unit_text_naming(text: str, currency: IdentityCurrency) -> str | None:
    """`text` with the words its `unit_not_resolved` alternatives stand for (the first
    ambiguous currency name, else a role-relative currency phrase) replaced by words that
    name `currency` alone, such as "constant 2015 XOF" for "constant 2015 francs". The
    result is read back through the binder, with no role bound, and kept only when it binds
    `currency`. `None` when `text` has no such words or no replacement binds."""
    tokens = _tokenise(_normalise(text))
    span = _classify(tokens).choice_span
    if span is None:
        return None
    start, length = span
    for words in _phrases_naming(currency):
        named = " ".join([*tokens[:start], words, *tokens[start + length :]])
        parsed = _parse(named, bound_roles=frozenset())
        if isinstance(parsed, _Assembled) and parsed.currency == currency:
            return named
    return None


def names_price_basis(text: str) -> bool:
    """Whether `text` names a price basis ("constant 2022 dollars", "in real terms"), read
    with the binder's own lexicon. A dropped phrase that does is cleared by the unit token."""
    return bool(_classify(_tokenise(_normalise(text))).price_bases)


def names_unit_request(text: str) -> bool:
    """Whether `bind_unit_text` binds `text` with no entity roles bound: it names a
    currency, a price basis, or a year beside a currency or "prices", and nothing the
    binder refuses (an unknown token, an ambiguous currency, a conflicting year or
    basis). The raw-text door hands such a phrase on as the part's `unit_text`; any
    other phrase stays a dropped phrase."""
    return isinstance(_parse(text, bound_roles=frozenset()), _Assembled)


def with_unit_lexicon_version(spec: Spec, outcome: BoundUnitRequest) -> Spec:
    """`spec` stating the lexicon version `outcome` was bound under
    (`ResolvedSpec.unit_lexicon_version`), which a same-shape merge requires its members
    to share."""
    return spec.model_copy(
        update={"resolved": spec.resolved.stating_unit_lexicon_version(outcome.lexicon_version)}
    )
