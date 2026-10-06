"""Read the quantity a query names and the quantity a shape serves, and match the two.

The request side is the normalisation a phrase names (`query_normalisation`): per person,
a numbered per-head base, per area, a rate or a share. A leg's `QuantityRequirement` holds
the strong reading of its variable and its sidecar qualifier phrases.

The shape side is the normalisation class of a shape's five-tuple (`normalisation_class`).
A per-person series divides by `Count_Person` or `PerCapita`, a density by `area`. Any
other denominator is a StatVar, and its own arcs say what it counts: a population type of
`Person` measured by `count` is a person count (`Count_Person_15OrMoreYears`), a measured
property of `area` is an area, and anything else (a GNI amount) is other. `denominator_kinds`
reads those arcs for every StatVar denominator in one batch; the schema terms need no read.

`fulfils` says whether a shape serves a requirement, through `honours`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from search.engine.retrieval.facts import _dcids_for
from search.engine.shape import ShapeDraft, measures_count

if TYPE_CHECKING:
    from search.engine.clients.graph import EngineGraphClient

type NormalisationClass = Literal["level", "other", "series", "per_person", "per_area", "unknown"]
type NamedNormalisation = Literal["per_person", "per_head_base", "per_area", "rate", "share"]
type DenominatorKind = Literal["person", "area", "other", "unknown"]

# measurementDenominator dcids of a per-person reading, and of a per-area one.
PER_PERSON_DENOMINATORS = frozenset({"Count_Person", "PerCapita"})
PER_AREA_DENOMINATOR = "area"


def schema_denominator_kind(denominator: str) -> DenominatorKind | None:
    """Return the kind a schema-term denominator names, or None for a StatVar denominator."""
    if denominator in PER_PERSON_DENOMINATORS:
        return "person"
    if denominator == PER_AREA_DENOMINATOR:
        return "area"
    return None


def denominator_kind_from_arcs(arcs: dict | None) -> DenominatorKind:
    """Return the kind of a denominator StatVar from its arcs, `unknown` when none resolved.

    Constraint arcs such as age or sex leave a person count a person count.
    """
    if arcs is None:
        return "unknown"
    measured = _dcids_for(arcs, "measuredProperty")
    if measured == ["area"]:
        return "area"
    if measured == ["count"] and _dcids_for(arcs, "populationType") == ["Person"]:
        return "person"
    return "other"


def denominator_kinds(
    denominators: Iterable[str], *, graph: EngineGraphClient
) -> dict[str, DenominatorKind]:
    """Return the kind of each StatVar denominator, read in one `node_arcs_batch`.

    Schema-term denominators (`schema_denominator_kind`) are left out of the map and the
    read, and no read happens when no StatVar denominator remains.
    """
    unique = [
        denominator
        for denominator in dict.fromkeys(denominators)
        if schema_denominator_kind(denominator) is None
    ]
    if not unique:
        return {}
    arcs_by_dcid = graph.node_arcs_batch(unique)
    return {
        denominator: denominator_kind_from_arcs(arcs_by_dcid.get(denominator))
        for denominator in unique
    }


# Measured properties whose plain values are amounts. A shape on one with no denominator
# arc and a plain stat type is a level.
_LEVEL_PROPERTIES = frozenset({"count", "amount", "valueAdded"})
_LEVEL_STAT_TYPES = frozenset({"measuredValue", ""})
# Add new per-head nouns here. The regex reads this tuple.
_PER_HEAD_NOUNS = ("capita", "person", "head", "inhabitant", "resident", "citizen")

# Applied to lowercase text, this matches `per` plus a singular noun from
# _PER_HEAD_NOUNS, or `capita` as a whole word (including `per-capita`). Bare
# `per`, `per year`, `per cent`, plural nouns, and numbered rate bases such as
# `per 100,000 residents` do not match.
_PER_CAPITA_FRAMING_RE = re.compile(r"\bper\s+(?:" + "|".join(_PER_HEAD_NOUNS) + r")\b|\bcapita\b")

# A whole phrase naming a per-person rate base: "per capita", "per person", "per 100 people",
# "per 1,000 inhabitants". The `count` group holds the number of a numbered base.
PER_HEAD_RATE_BASE_RE = re.compile(
    r"per\s+(?:(?P<count>\d[\d,]*)\s+)?"
    r"(?:capita|head|people|persons?|inhabitants?|residents?|population)",
    re.IGNORECASE,
)

# Terms naming a rate, and terms naming a share of a whole. `STRONG_RATE_REQUEST_RE` is
# the subset the pre-bind raw-count prune reads (`phase_rank_bind._has_strong_rate_intent`).
# `ratio` and `incidence` name a rate here and leave that prune off.
_RATE_TERMS_RE = re.compile(r"\b(?:rates?|ratios?|incidence)\b")
_SHARE_TERMS = r"\b(?:proportions?|percentages?|percent|shares?)\b|%"
_SHARE_TERMS_RE = re.compile(_SHARE_TERMS)
STRONG_RATE_REQUEST_RE = re.compile(r"\brates?\b|" + _SHARE_TERMS, re.IGNORECASE)
# Tokens of a lowercase phrase: alphabetic runs, digit runs, `/`, and `²`.
# Every other character separates tokens, so `km²` reads as `km`, `²`, and
# `km2`, `km^2`, `km**2`, and `km 2` all read as `km`, `2`.
_FRAMING_TOKEN_RE = re.compile(r"[a-z]+|\d+|[/²]")

# Stem shared by dense, denser, densest, densely, density, and densities.
_DENSITY_STEM = "dens"

# Units that name an area on their own. Add new spellings here.
_AREA_UNITS = frozenset({"ha", "hectare", "hectares", "acre", "acres"})

# Length units that name an area once squared. Add new spellings here.
_LENGTH_UNITS = frozenset(
    {
        "km",
        "kms",
        "kilometre",
        "kilometres",
        "kilometer",
        "kilometers",
        "mi",
        "mile",
        "miles",
        "m",
        "metre",
        "metres",
        "meter",
        "meters",
        "ft",
        "foot",
        "feet",
        "yd",
        "yard",
        "yards",
        "cm",
        "centimetre",
        "centimetres",
        "centimeter",
        "centimeters",
    }
)

# Words that square a length unit from either side: `sq km`, `km squared`.
_SQUARE_MARKERS = frozenset({"square", "sq", "squared"})

# A square marker written against its unit with no separator: `sqkm`, `sqm`.
_GLUED_SQUARE_UNITS = frozenset(
    marker + unit for marker in _SQUARE_MARKERS for unit in _LENGTH_UNITS
)

# Exponent tokens that square the length unit before them. The token must equal
# one of these, so `m3` and `km20` stay distances and volumes.
_SQUARE_EXPONENTS = frozenset({"2", "²"})

# Nouns that name area: `per area`, `relative to its area`, `land area`, and also
# `rural areas`, where the noun names a place.
_AREA_NOUNS = frozenset({"area", "areas"})

# Token runs that make the area noun straight after them a division: `per area`,
# `per unit area`, `relative to area`.
_AREA_DIVISORS = (("per",), ("per", "unit"), ("relative", "to"))


def _names_per_area_framing(variable_lower: str, *, strong: bool) -> bool:
    """Return whether the lowercase phrase names density framing.

    A phrase names density framing through the ``dens`` stem, an area unit, a
    length unit squared by a neighbouring marker or a following exponent, or an
    area noun. The test reads tokens and ignores grammar. The loose reading
    (``strong=False``) takes any area noun, so ``population in urban area``,
    ``people living in rural areas``, ``hospitals/ha noi`` and ``bone density``
    match. It decides only whether an alternate honours the framing, where a false
    match admits one alternate. The strong reading takes an area noun only straight
    after ``per``, ``per unit`` or ``relative to``. It decides the normalisation pin,
    where a false match refuses every alternate that is not a density. A bare length
    unit is a distance, so ``per mile driven`` and ``cost per km of road`` match
    neither, and ``crowded`` holds no area token.
    """
    tokens = ["", *_FRAMING_TOKEN_RE.findall(variable_lower), ""]
    for i in range(1, len(tokens) - 1):
        previous, token, following = tokens[i - 1], tokens[i], tokens[i + 1]
        if token.startswith(_DENSITY_STEM) or token in _AREA_UNITS or token in _GLUED_SQUARE_UNITS:
            return True
        if token in _LENGTH_UNITS and (
            previous in _SQUARE_MARKERS
            or following in _SQUARE_MARKERS
            or following in _SQUARE_EXPONENTS
        ):
            return True
        if token in _AREA_NOUNS and (
            not strong or any(tuple(tokens[i - len(run) : i]) == run for run in _AREA_DIVISORS)
        ):
            return True
    return False


def query_normalisation(variable: str, *, strong: bool) -> frozenset[NamedNormalisation]:
    """Return every normalisation class the phrase names.

    Per person is per-head wording (`per capita`, `per person`, `per head`), and a
    numbered per-head base (`per 100,000 people`) is `per_head_base`. Per area is density
    framing (`_names_per_area_framing`, read strong or loose), rate is `rate`, `ratio` or
    `incidence`, and share is `proportion`, `percentage`, `percent` or `%`.
    """
    lower = variable.lower()
    named: set[NamedNormalisation] = set()
    if _PER_CAPITA_FRAMING_RE.search(lower):
        named.add("per_person")
    for base in PER_HEAD_RATE_BASE_RE.finditer(lower):
        named.add("per_head_base" if base.group("count") else "per_person")
    if _names_per_area_framing(lower, strong=strong):
        named.add("per_area")
    if _RATE_TERMS_RE.search(lower):
        named.add("rate")
    if _SHARE_TERMS_RE.search(lower):
        named.add("share")
    return frozenset(named)


_CLASS_OF_DENOMINATOR: Mapping[DenominatorKind, NormalisationClass] = {
    "person": "per_person",
    "area": "per_area",
    "other": "series",
    "unknown": "unknown",
}


def normalisation_class(
    shape: ShapeDraft, *, kinds: Mapping[str, DenominatorKind] | None = None
) -> NormalisationClass:
    """Return the normalisation class of a shape's five-tuple.

    An unresolved denominator arc is `unknown`. A denominator's kind is the schema term's
    (`PerCapita` or `Count_Person` is a person, `area` an area), else its entry in
    `kinds` (`quantity.denominator_kinds`), else other: a person is `per_person`, an area
    `per_area`, other a `series`, and a StatVar whose arcs did not resolve `unknown`. With
    no denominator arc, a plain stat type over a count or a declared amount property
    (`_LEVEL_PROPERTIES`) is a `level`, and anything else, such as an SDG `value` series,
    a fertility rate or a growth rate, is `other`.
    """
    if shape.meas_denom_unresolved:
        return "unknown"
    denominator = shape.meas_denom_dcid
    if denominator is not None:
        kind = schema_denominator_kind(denominator) or (kinds or {}).get(denominator, "other")
        return _CLASS_OF_DENOMINATOR[kind]
    if shape.stat_type_dcid in _LEVEL_STAT_TYPES and (
        measures_count(shape) or shape.meas_prop_dcid in _LEVEL_PROPERTIES
    ):
        return "level"
    return "other"


def honours(shape: ShapeDraft, cls: NormalisationClass, named: NamedNormalisation) -> bool:
    """Whether a shape of class `cls` is a reading of the named normalisation.

    An unresolved arc contradicts nothing. A count of persons per person is a share of
    the population, and an amount or an event count per person is not. A numbered per-head
    base names a rate per person, which an `other` series may hold with its denominator
    unstated (the SDG homicide rate per 100,000 people), while a `series` names a
    denominator that is not a person.
    """
    if cls == "unknown":
        return True
    if cls == "level":
        return False
    if cls == "other":
        return named in ("per_head_base", "rate", "share")
    if cls == "series":
        return named in ("rate", "share")
    if cls == "per_area":
        return named in ("per_area", "rate")
    if named == "share":
        return shape.pop_type_dcid == "Person" and measures_count(shape)
    return named in ("per_person", "per_head_base", "rate")


# The named forms that divide by a count of persons.
PER_PERSON_FORMS: frozenset[NamedNormalisation] = frozenset({"per_person", "per_head_base"})


@dataclass(frozen=True, slots=True)
class QuantityRequirement:
    """The normalisation one leg names, read from its variable and each sidecar qualifier
    phrase, one phrase at a time: `named` from strong signals only, which the materialise
    check and the guard's pin read, and `framing` from loose ones as well, which an
    alternate may honour (`query_normalisation`)."""

    named: frozenset[NamedNormalisation] = frozenset()
    framing: frozenset[NamedNormalisation] = frozenset()

    def narrowed(self, forms: frozenset[NamedNormalisation]) -> QuantityRequirement:
        """The requirement limited to `forms`."""
        return QuantityRequirement(self.named & forms, self.framing & forms)


# A leg that names no normalisation.
NO_REQUIREMENT = QuantityRequirement()


def quantity_requirement(phrases: Iterable[str]) -> QuantityRequirement:
    """Read one leg's requirement from its own phrases."""
    phrases = list(phrases)
    return QuantityRequirement(
        named=frozenset().union(*(query_normalisation(p, strong=True) for p in phrases)),
        framing=frozenset().union(*(query_normalisation(p, strong=False) for p in phrases)),
    )


def fulfils(
    requirement: QuantityRequirement,
    shape: ShapeDraft,
    kinds: Mapping[str, DenominatorKind] | None = None,
) -> bool:
    """Whether `shape` serves `requirement`: nothing is named, or the shape's class
    (`normalisation_class` under `kinds`) honours a named form (`honours`)."""
    if not requirement.named:
        return True
    cls = normalisation_class(shape, kinds=kinds)
    return any(honours(shape, cls, named) for named in requirement.named)
