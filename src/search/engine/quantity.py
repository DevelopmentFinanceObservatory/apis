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

`serves_every_form` says whether a shape honours every form a requirement names, through
`honours`.

A question can also name the denominator itself ("debt to GDP"). The quantity read
(`quantity_read`) picks it from the StatVar denominators the menu's shapes divide by
(`NamedDenominator`), and
`check_denominator` holds the elected shape to it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from search.engine.function_words import FUNCTION_WORDS
from search.engine.region.equivalent_series import CLUSTER_BY_MEMBER
from search.engine.retrieval.facts import _dcids_for, read_name
from search.engine.shape import ShapeDraft, counts_persons, measures_count
from search.models import GraphRef, QuantityForm
from unit_words.unit_class import is_per_head_denominator

if TYPE_CHECKING:
    from search.engine.clients.graph import EngineGraphClient

type NormalisationClass = Literal["level", "other", "series", "per_person", "per_area", "unknown"]
type NamedNormalisation = QuantityForm
type DenominatorKind = Literal["person", "area", "other", "unknown"]

# The measurementDenominator dcid of a per-area reading. A per-person one is a
# `unit_words.unit_class.PER_HEAD_DENOMINATORS` dcid.
PER_AREA_DENOMINATOR = "area"


def schema_denominator_kind(denominator: str) -> DenominatorKind | None:
    """Return the kind a schema-term denominator names, or None for a StatVar denominator."""
    if is_per_head_denominator(denominator):
        return "person"
    if denominator == PER_AREA_DENOMINATOR:
        return "area"
    return None


def shape_denominator_kind(shape: ShapeDraft) -> DenominatorKind | None:
    """Return the kind of the schema term `shape` divides by, or None when it divides by a
    StatVar or by nothing (`schema_denominator_kind`).

    The arc's dcid states the term whether or not its target resolves: base Data Commons
    holds `PerCapita` as a ProvisionalNode, and a series dividing by it divides by the same
    per-person base as one dividing by `Count_Person`."""
    denominator = shape.meas_denom_dcid
    return schema_denominator_kind(denominator) if denominator is not None else None


def denominator_meaning(shape: ShapeDraft) -> DenominatorKind | tuple[str | None, bool]:
    """What `shape` divides by, equal for two shapes exactly when they divide by one base:
    the kind of a schema term (`shape_denominator_kind`), so `Count_Person` and `PerCapita`
    are one per-person base, else the arc itself, its dcid and whether it resolves. The
    status rule (`status_rule._differing_axes`) and the rank election
    (`bind.rank_elected`) both compare denominators by it."""
    kind = shape_denominator_kind(shape)
    if kind is not None:
        return kind
    return shape.meas_denom_dcid, shape.meas_denom_unresolved


def stated_denominator(shape: ShapeDraft) -> str | None:
    """Return the dcid of the base `shape` states it divides by: a resolved arc's, or a
    schema term's whether or not its target resolves (`shape_denominator_kind`). None when
    it divides by nothing, or by an unresolved arc naming no schema term."""
    if shape.meas_denom_unresolved and shape_denominator_kind(shape) is None:
        return None
    return shape.meas_denom_dcid


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


def statvar_denominators(shapes: Iterable[ShapeDraft]) -> list[str]:
    """The distinct StatVar denominators `shapes` divide by, in shape order. Schema-term
    denominators (`schema_denominator_kind`) and unresolved arcs are left out."""
    resolved = (shape.meas_denom_dcid for shape in shapes if not shape.meas_denom_unresolved)
    return [
        denominator
        for denominator in dict.fromkeys(resolved)
        if denominator is not None and schema_denominator_kind(denominator) is None
    ]


@dataclass(frozen=True, slots=True)
class DenominatorRead:
    """What the graph says of one StatVar denominator: its kind and its name, `None` when
    it gives none."""

    kind: DenominatorKind
    name: str | None


def read_denominators(
    denominators: Iterable[str], *, graph: EngineGraphClient
) -> dict[str, DenominatorRead]:
    """Return the kind and name of each StatVar denominator, read in one `node_arcs_batch`.

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
    reads: dict[str, DenominatorRead] = {}
    for denominator in unique:
        arcs = arcs_by_dcid.get(denominator)
        reads[denominator] = DenominatorRead(
            kind=denominator_kind_from_arcs(arcs),
            name=read_name(arcs) if arcs is not None else None,
        )
    return reads


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

# Terms naming a rate, and terms naming a share of a whole.
_RATE_TERMS_RE = re.compile(r"\b(?:rates?|ratios?|incidence)\b")
_SHARE_TERMS = r"\b(?:proportions?|percentages?|percent|shares?)\b|%"
_SHARE_TERMS_RE = re.compile(_SHARE_TERMS)
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


# The wording `query_normalisation` reads as a form through its patterns. Per-area wording
# is read token by token (`_names_per_area_framing`).
_FORM_WORDING_RES = (
    _PER_CAPITA_FRAMING_RE,
    PER_HEAD_RATE_BASE_RE,
    _RATE_TERMS_RE,
    _SHARE_TERMS_RE,
)


def names_only_form(text: str) -> bool:
    """Whether `text` holds no word beyond the wording `query_normalisation` reads as a
    form and function words (`FUNCTION_WORDS`): "ratio", "a share of", "per capita" and
    "density" name a form and nothing it divides by, while "share of GDP" names GDP.
    Bare "capita" is per-head wording (`_PER_CAPITA_FRAMING_RE`); bare "people" and
    "population" are not, since only "per" before them makes a per-head base."""
    lower = text.lower()
    for pattern in _FORM_WORDING_RES:
        lower = pattern.sub(" ", lower)
    return all(
        token in FUNCTION_WORDS or _names_per_area_framing(token, strong=False)
        for token in _FRAMING_TOKEN_RE.findall(lower)
    )


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

    A schema-term denominator's kind is the term's (`shape_denominator_kind`: `PerCapita`
    or `Count_Person` is a person, `area` an area), whether or not its arc resolves. Any
    other unresolved denominator arc is `unknown`. A StatVar denominator's kind is its
    entry in `kinds` (`quantity.denominator_kinds`), else other: a person is `per_person`,
    an area `per_area`, other a `series`, and a StatVar whose arcs did not resolve
    `unknown`. With no denominator arc, a plain stat type over a count or a declared amount
    property (`_LEVEL_PROPERTIES`) is a `level`, and anything else, such as an SDG `value`
    series, a fertility rate or a growth rate, is `other`.
    """
    if (schema_kind := shape_denominator_kind(shape)) is not None:
        return _CLASS_OF_DENOMINATOR[schema_kind]
    if shape.meas_denom_unresolved:
        return "unknown"
    denominator = shape.meas_denom_dcid
    if denominator is not None:
        return _CLASS_OF_DENOMINATOR[(kinds or {}).get(denominator, "other")]
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
        return counts_persons(shape)
    return named in ("per_person", "per_head_base", "rate")


@dataclass(frozen=True, slots=True)
class QuantityRequirement:
    """The normalisation one leg names, read from its variable and each sidecar qualifier
    phrase, one phrase at a time: `named` from strong signals only, which the materialise
    check and the guard's pin read, and `framing` from loose ones as well, which an
    alternate may honour (`query_normalisation`)."""

    named: frozenset[NamedNormalisation] = frozenset()
    framing: frozenset[NamedNormalisation] = frozenset()


# A leg that names no normalisation.
NO_REQUIREMENT = QuantityRequirement()


def quantity_requirement(phrases: Iterable[str]) -> QuantityRequirement:
    """Read one leg's requirement from its own phrases."""
    phrases = list(phrases)
    return QuantityRequirement(
        named=frozenset().union(*(query_normalisation(p, strong=True) for p in phrases)),
        framing=frozenset().union(*(query_normalisation(p, strong=False) for p in phrases)),
    )


def with_stated_form(
    requirement: QuantityRequirement, stated: NamedNormalisation | None
) -> QuantityRequirement:
    """`requirement` with the form the quantity read stated (`stated`) joined, only when no
    wording pattern named a form: a pattern's reading outranks the model's."""
    if stated is None or requirement.named:
        return requirement
    return QuantityRequirement(
        named=frozenset({stated}), framing=requirement.framing | frozenset({stated})
    )


def serves_every_form(
    requirement: QuantityRequirement,
    shape: ShapeDraft,
    kinds: Mapping[str, DenominatorKind] | None = None,
) -> bool:
    """Whether `shape`'s class (`normalisation_class` under `kinds`) honours every form
    `requirement` names (`honours`), as the materialise gate holds the elected shape:
    "incidence per capita" names a rate and a per-person form, and a rate that is not per
    person serves only one of them."""
    if not requirement.named:
        return True
    cls = normalisation_class(shape, kinds=kinds)
    return all(honours(shape, cls, named) for named in requirement.named)


@dataclass(frozen=True, slots=True)
class NamedDenominator:
    """The denominator a question names, as the quantity read chose it from the StatVar
    denominators the admissible shapes divide by: `ref`, or `None` when the read matched
    the base to none of them ("none of these"), and `base_words`, the question's own words
    for the base, which the read quoted and the phrase grounds."""

    ref: GraphRef | None
    base_words: str


type DenominatorCheck = Literal["matches", "unverified", "refused"]


def _same_denominator(named: str, stated: str) -> bool:
    """Whether two denominator dcids are one StatVar or twins in one equivalence cluster."""
    if named == stated:
        return True
    cluster = CLUSTER_BY_MEMBER.get(named)
    return cluster is not None and stated in cluster.members


def check_denominator(shape: ShapeDraft, named: NamedDenominator) -> DenominatorCheck:
    """Hold `shape` to the denominator the question names.

    A shape stating its base (`stated_denominator`: a resolved arc, or a schema term such as
    the ProvisionalNode `PerCapita`) `matches` when that base is the named denominator and
    is `refused` when the read named a different listed one. Under "none of these"
    (`named.ref` is `None`) it is `unverified`: the read matches the question's words to
    terse graph names, and an absent match proves nothing about the arc ("total government
    spending" against "Amount Of Economic Activity: Expenditure Activity, Government" flips
    on one capital letter). A shape whose arc resolves to no stated base, or that has no
    arc and classes as `other`, as UN and WHO ratio series do, states its base only in its
    name, so it passes `unverified`. Any other shape with no arc, such as a level, divides
    by nothing and is `refused`: the question's grounded words name a base.

    Two denominators match when they are one dcid or members of one registered equivalence
    cluster (`CLUSTER_BY_MEMBER`): a ratio is unit-free, so the cluster's currency does not
    matter. Only this check reads the cluster. The elections compare denominators by dcid.
    """
    denominator = stated_denominator(shape)
    if denominator is not None:
        if named.ref is None:
            return "unverified"
        return "matches" if _same_denominator(named.ref.ref, denominator) else "refused"
    if shape.meas_denom_unresolved:
        return "unverified"
    return "unverified" if normalisation_class(shape) == "other" else "refused"


def serves_quantity(
    requirement: QuantityRequirement,
    named: NamedDenominator | None,
    shape: ShapeDraft,
    kinds: Mapping[str, DenominatorKind] | None = None,
) -> bool:
    """Whether `shape` serves a leg's whole quantity: a class that honours every form
    `requirement` names (`serves_every_form`) and, when the question names a denominator
    (`named`), one `check_denominator` does not refuse. The materialise gate holds the
    elected shape to it, and the alternate guard holds every alternate to it."""
    if not serves_every_form(requirement, shape, kinds):
        return False
    return named is None or check_denominator(shape, named) != "refused"
