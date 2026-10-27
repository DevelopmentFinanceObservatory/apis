"""Render what-axis labels, qualifiers, and unit or denominator phrases."""

from __future__ import annotations

import re
import string
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

from code_index.composites import provenance_datasets
from code_index.labels import PRICE_BASIS_AXIS, is_price_basis
from code_index.shared import get_index
from dfo_platform.roles import RoleWording
from search.heading.measure_noun import MeasureNoun as AuthoredNoun
from search.heading.measure_noun import denominator_noun, measure_noun
from search.models import ResolvedFacet, Slot
from unit_words.unit_class import classify_unit

from .how import _is_adjectival
from .labels import (
    _TRAILING_PARENTHETICAL,
    _already_named_in,
    _binding_labels,
    _humanize,
    _join_labels,
)

if TYPE_CHECKING:
    from .heading import ShapeLike


# Generic measured properties use a more specific StatVar or population label when available.
_STOPWORD_MEASURED_PROPS = frozenset({"amount", "count", "value"})


# Match uppercase dcid tails that the fallback cannot split into words.
_ALLCAPS_UNSPLITTABLE_TAIL = re.compile(r"^[A-Z][A-Z0-9]*$")


# Strip symbol-only parentheticals and trailing periods from indicator labels.
_TRAILING_SYMBOL_PAREN = re.compile(r"\s*\([^\w\s]+\)\s*$")


# Convert trailing bracketed slice qualifiers to parentheticals.
_TRAILING_BRACKET = re.compile(r"\[([^\[\]]*)\]\s*$")


def _is_unsplittable_allcaps_dcid(dcid: str, label: str | None) -> bool:
    """Return whether a fallback dcid has an unsplittable uppercase tail."""
    if label and label != dcid:
        return False
    tail = dcid.rsplit("/", 1)[-1]
    return _ALLCAPS_UNSPLITTABLE_TAIL.fullmatch(tail) is not None


def _is_plain_title_case(word: str) -> bool:
    """Return whether a token has an initial capital and lowercase remainder, with no digits.

    Strip edge punctuation first. A single capital letter passes. Acronyms,
    digit-bearing, and mixed-case words fail.
    """
    if any(ch.isdigit() for ch in word):
        return False
    core = word.strip(string.punctuation)
    if not core:
        return False
    return core[0].isupper() and (len(core) == 1 or core[1:].islower())


# Closed function-word list: tokens ignored when (1) checking whether a
# candidate StatVar label adds a real content word over the humanized
# property text it might replace (_content_words, the what-fallback
# promotion gate) and (2) deciding whether a label's head is Title Case
# throughout (_head_is_title_case, the sentence-casing gate). A lowercase
# connective ("of", "for", "the") is expected even in a properly authored
# Title-Case heading and never blocks either check. Compared case-folded.
_WHAT_FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "or",
        "per",
        "by",
        "with",
        "as",
        "at",
        "from",
    }
)


def _recase_words(text: str) -> str:
    """Lower every plain Title-Case word. The plan builder restores sentence capitals.

    Callers gate whole labels first. Bracketed qualifier content is always
    recased because it is subordinate to the label.
    """
    return " ".join(w.lower() if _is_plain_title_case(w) else w for w in text.split(" "))


def _lower_first_word(text: str) -> str:
    """Lower the first word when it is plain Title Case, for a phrase that no longer
    starts the sentence ("Kenya's external debt stocks"). Acronyms stay."""
    first, _, rest = text.partition(" ")
    if not _is_plain_title_case(first):
        return text
    return f"{first.lower()} {rest}" if rest else first.lower()


def breakdown_phrase(slot_labels: Sequence[str]) -> str:
    """The slots an answer breaks a series down by, as a phrase inside a sentence: each
    label's first word lowered (`_lower_first_word`), joined ("sex and age")."""
    return _join_labels([_lower_first_word(label) for label in slot_labels])


def _label_head(text: str) -> str:
    """Return the label head without a trailing parenthetical."""
    match = _TRAILING_PARENTHETICAL.search(text)
    return text[: match.start()] if match else text


def _head_is_title_case(head: str) -> bool:
    """Return whether a label head is Title Case and worth sentence-casing.

    It needs at least two plain Title-Case words. Function words, acronyms,
    digit-bearing words, and mixed-case words do not count. Any other
    lowercase word rejects the whole head to preserve authored sentence case.
    """
    title_case_count = 0
    for word in head.split(" "):
        if not word:
            continue
        if _is_plain_title_case(word):
            title_case_count += 1
            continue
        core = word.strip(string.punctuation)
        if core.lower() in _WHAT_FUNCTION_WORDS:
            continue
        if core.isalpha() and core.islower():
            return False
    return title_case_count >= 2


def _normalize_what_label(text: str) -> str:
    """Clean a catalog label for sentence rendering.

    Repeatedly strip trailing symbol-only parentheses and periods, then turn
    a trailing bracketed qualifier into a recased parenthetical. Lower
    plain Title-Case words only when the head has at least two such words and
    no lowercase content words. Acronyms, digits, and mixed case stay intact.
    Other labels keep their authored casing.
    """
    while True:
        stripped = _TRAILING_SYMBOL_PAREN.sub("", text)
        if stripped.endswith("."):
            stripped = stripped[:-1].rstrip()
        if stripped == text:
            break
        text = stripped
    bracket = _TRAILING_BRACKET.search(text)
    if bracket:
        content = _recase_words(bracket.group(1))
        text = f"{text[: bracket.start()]}({content})"
    if not _head_is_title_case(_label_head(text)):
        return text
    return _recase_words(text)


def _content_words(text: str) -> set[str]:
    """Return case-folded content words after stripping edge punctuation and function words."""
    words: set[str] = set()
    for word in text.lower().split():
        core = word.strip(string.punctuation)
        if core and core not in _WHAT_FUNCTION_WORDS:
            words.add(core)
    return words


def _first_content_word(text: str) -> str | None:
    """Return the first non-function word, case-folded, or None when absent."""
    for word in text.split():
        folded = word.lower()
        if folded not in _WHAT_FUNCTION_WORDS:
            return folded
    return None


def _names_every_adjectival_how_value(sv_text: str, slots: list[Slot]) -> bool:
    """Return whether every bound adjectival how value appears in ``sv_text``."""
    for slot in slots:
        if slot.key.axis != "how" or not _is_adjectival(slot):
            continue
        labels = _binding_labels(slot.binding)
        if labels and not _already_named_in(_join_labels(labels), sv_text):
            return False
    return True


def _names_every_measure_extra(sv_text: str, shape: ShapeLike) -> bool:
    """Return whether ``sv_text`` names each emitted measure extra's first content word."""
    for extra in _measure_extras(shape):
        word = _first_content_word(extra)
        if word and not _already_named_in(word, sv_text):
            return False
    return True


def _fallback_what(
    shape: ShapeLike, sv_labels: list[str] | None, slots: list[Slot]
) -> tuple[str, bool]:
    """Choose fallback what-text and report whether it names one StatVar.

    A bound what slot takes precedence upstream. The engine's ``measure_label``
    also bypasses this ladder when available. Otherwise, prefer a non-generic
    measured property. Promote one StatVar's normalised label only when its
    content words strictly extend the property and it names every adjectival
    how value and measure extra. Generic properties prefer a single StatVar
    label, then population type, the property, and finally the shape label.
    Reject unsplittable uppercase dcids at both property rungs. Normalise all
    returned labels except the final shape label.

    Pre-materialisation callers pass ``sv_labels=None``. They cannot use a
    StatVar label or set ``used_sv_name``, so their how clauses remain visible.
    """
    measured_property = shape.measured_property
    has_property = measured_property is not None and (
        measured_property.ref.lower() not in _STOPWORD_MEASURED_PROPS
        and not _is_unsplittable_allcaps_dcid(measured_property.ref, measured_property.label)
    )
    if has_property:
        property_text = _normalize_what_label(_humanize(measured_property))
        if sv_labels is not None and len(sv_labels) == 1:
            sv_text = _normalize_what_label(sv_labels[0])
            if (
                _content_words(property_text) < _content_words(sv_text)
                and _names_every_adjectival_how_value(sv_text, slots)
                and _names_every_measure_extra(sv_text, shape)
            ):
                return sv_text, True
        return property_text, False
    if sv_labels is not None and len(sv_labels) == 1:
        return _normalize_what_label(sv_labels[0]), True
    if shape.population_type is not None:
        return _normalize_what_label(_humanize(shape.population_type)), False
    if measured_property is not None and not _is_unsplittable_allcaps_dcid(
        measured_property.ref, measured_property.label
    ):
        return _normalize_what_label(_humanize(measured_property)), False
    return shape.label, False


def _measure_extras(shape: ShapeLike) -> list[str]:
    """The stat type's and measurement qualifier's phrases that qualify a graph name.

    Preserve that order and omit defaults. Normalise catalog labels. A heading states
    them only beside a graph name (`MeasureNoun.kind` "graph_name"). A qualifier that
    states a price basis (`is_price_basis`: nominal, real) belongs to the unit and is left
    out.
    A denominator is the unit's (`_unit_heading_phrase`).
    """
    extras: list[str] = []
    stat_type = shape.stat_type
    if stat_type is not None and stat_type.ref != "measuredValue":
        extras.append(_normalize_what_label(_humanize(stat_type)))
    qualifier = shape.measurement_qualifier
    if qualifier is not None and not is_price_basis(PRICE_BASIS_AXIS, qualifier.ref):
        extras.append(_normalize_what_label(_humanize(qualifier)))
    return extras


# The property whose short names modify the measure noun. Other codes' short names read
# as how values (`how._how_value_text`).
_FLOW_TYPE_PROPERTY = "crsTypeOfFlow"


def _flow_short_names(constraints: Mapping[str, str]) -> list[tuple[str, str]]:
    """Return ``(property dcid, short name)`` for the shared CRS flow type's short name
    ("ODA", "OOF"), or an empty list. A value whose code row belongs to another property
    is skipped.
    """
    value_dcid = constraints.get(_FLOW_TYPE_PROPERTY)
    row = get_index().code_by_dcid(value_dcid) if value_dcid is not None else None
    if row is None or not row.short_name or row.code_system != _FLOW_TYPE_PROPERTY:
        return []
    return [(_FLOW_TYPE_PROPERTY, row.short_name)]


def _authored_measure_noun(shape: ShapeLike, constraints: Mapping[str, str]) -> AuthoredNoun | None:
    """Return the authored IDS or WEO noun (`measure_noun`) for the shape, or None."""
    return measure_noun(
        population_type=shape.population_type.ref if shape.population_type else None,
        measured_property=shape.measured_property.ref if shape.measured_property else None,
        stat_type=shape.stat_type.ref if shape.stat_type else None,
        constraints=constraints,
    )


# Leading netting words of a measure noun. A flow short name follows them and precedes
# the head noun: "gross" + "ODA" + "disbursements".
_NETTING_WORDS = frozenset({"gross", "net"})


def _with_short_names(noun: str, names: Sequence[str]) -> str:
    """Insert flow short names before a measure noun's head, after its netting words.

    "gross disbursements" with "ODA" reads "gross ODA disbursements". "commitments"
    reads "ODA commitments".
    """
    if not names:
        return noun
    words = noun.split(" ")
    head = 0
    while head < len(words) - 1 and words[head].lower() in _NETTING_WORDS:
        head += 1
    return " ".join([*words[:head], *names, *words[head:]])


# Ratio terms in the heading text already state a share. Per-terms already state a
# per-head rate. Bare "ratio" is excluded because a ratio measure can still need a unit.
_RATIO_WORDS = frozenset(
    {"%", "percent", "percentage", "prevalence", "share of", "proportion of", "fraction of"}
)
_PER_WORDS = frozenset(
    {"per capita", "per person", "per 100", "per 1,000", "per 1000", "per inhabitant"}
)


def _graph_text_names_any(graph_text: str | None, words: frozenset[str]) -> bool:
    """Return whether any basis term occurs in a graph name's ``graph_text``. A None
    ``graph_text`` (an authored noun) names none.

    Use word boundaries except for ``%``, which uses substring matching.
    """
    if graph_text is None:
        return False
    if "%" in words and "%" in graph_text:
        return True
    return any(_already_named_in(word, graph_text) for word in words)


# The plural phrase for each purchasing-power unit `classify_unit` reads.
_PURCHASING_POWER_PHRASES: Mapping[str, str] = {
    "PPPInternationalDollar": "PPP international dollars",
    "InternationalDollar": "international dollars",
    "PurchasingPowerStandard": "purchasing power standards",
}


_CURRENCY_NAMES: Mapping[str, re.Pattern[str]] = {
    "USD": re.compile(r"(?<!\w)(?:US|United States) dollars?(?!\w)|(?<!\w)US\$", re.IGNORECASE),
}
"""The names a graph measure name gives a currency, by ISO 4217 code: "US dollars",
"United States dollars", "(current US$)"."""


def _names_currency(measure_name: str, currency: str | None) -> bool:
    """Whether a graph measure name names ``currency`` (`_CURRENCY_NAMES`). The graph
    holds no fact saying a name carries its unit, so this matches the name's text."""
    pattern = _CURRENCY_NAMES.get(currency or "")
    return pattern is not None and pattern.search(measure_name) is not None


def facet_currency_units(unit: ResolvedFacet | None) -> dict[str, str]:
    """The unit facet's ref and ISO 4217 code as `classify_unit` reads currencies, or
    empty when the facet names no currency."""
    if unit is None or unit.ref is None or unit.currency is None:
        return {}
    return {unit.ref: unit.currency}


class UnitPhrase(NamedTuple):
    """The unit phrase a heading states (`_unit_heading_phrase`), and whether it names the
    unit itself, so its token states `UNIT_REF`."""

    text: str
    names_unit: bool


def _unit_heading_phrase(
    shape: ShapeLike,
    facets: Mapping[str, ResolvedFacet] | None,
    graph_heading_text: str | None,
    *,
    wording: RoleWording,
) -> UnitPhrase:
    """Return the unit phrase a heading states, empty when the unit leaves it alone.

    A unit stays in the heading when it changes the magnitude and is no currency: "per
    capita" for a per-head denominator (`is_per_head_denominator`, `Count_Person` or
    `PerCapita`), "as a share of exports of goods, services and primary income", "in PPP
    international dollars", or "in percent". A currency, whatever its code, is the
    byline's (`heading._byline_unit`), so a per-capita amount in euros reads "per capita"
    here and "current euros" there. Units outside these classes return empty.
    ``graph_heading_text`` is the heading so far when a graph name states the measure,
    and None when an authored noun or measure label does. A part a graph name already
    states ("CO2 emissions per capita") is left out, so no unit reads twice. An authored
    noun never names a unit, so it is not inspected. Before facets resolve (``facets``
    None) only the per-head reading applies, since the shape alone fixes it. A unit whose
    series disagree (``per_series``) has no class, though its denominator still reads.
    A ratio whose dataset declares whose total it divides by (`_ratio_base_owner`) names
    that party in ``wording``'s noun: DAC2a's "as a share of the recipient's GNI", DAC1's
    "as a share of the provider's GNI".

    The phrase names the unit (`UnitPhrase.names_unit`) when it states a percent ("in
    percent", "as a share of ...") or a purchasing-power unit. "per capita" and "relative
    to ..." name only the denominator.
    """
    unit = (facets or {}).get("unit")
    denominator = shape.measurement_denominator
    unit_class = classify_unit(
        unit=unit.ref if unit is not None else None,
        denominator=denominator.ref if denominator is not None else None,
        currency_units=facet_currency_units(unit),
    )
    phrases: list[str] = []
    if unit_class.per_head and not _graph_text_names_any(graph_heading_text, _PER_WORDS):
        phrases.append("per capita")
    if facets is None or not unit_class.changes_magnitude:
        return UnitPhrase(" ".join(phrases), names_unit=False)
    names_unit = False
    names_ratio = _graph_text_names_any(graph_heading_text, _RATIO_WORDS)
    if denominator is not None and unit_class.ratio_denominator is not None and not names_ratio:
        base = denominator_noun(denominator.ref) or _recase_words(
            _normalize_what_label(_humanize(denominator))
        )
        if (owner := _ratio_base_owner(facets, wording)) is not None:
            base = f"the {owner}'s {base}"
        is_percent = unit_class.kind == "percent"
        phrases.append(f"{'as a share of' if is_percent else 'relative to'} {base}")
        names_unit = is_percent
    elif unit_class.kind == "percent" and not names_ratio:
        phrases.append("in percent")
        names_unit = True
    if unit_class.kind == "purchasing_power":
        unit_text = (unit.label or unit.ref or "") if unit is not None else ""
        phrases.append(f"in {_PURCHASING_POWER_PHRASES.get(unit_class.unit or '', unit_text)}")
        names_unit = True
    return UnitPhrase(" ".join(phrases), names_unit=names_unit)


def _ratio_base_owner(facets: Mapping[str, ResolvedFacet], wording: RoleWording) -> str | None:
    """The noun of the party whose own total the answer's ratio divides by, as the one
    dataset it serves from declares it (`DatasetDeclaration.ratio_base_axis`), else `None`:
    for a dataset that declares none, and for series of several datasets, as a composite's
    are when its components read several (`code_index.composites.provenance_datasets`)."""
    provenance = facets.get("provenance")
    if provenance is None or provenance.ref is None:
        return None
    datasets = provenance_datasets(provenance.ref)
    dataset = get_index().datasets.get(datasets[0]) if len(datasets) == 1 else None
    if dataset is None or dataset.ratio_base_axis is None:
        return None
    return wording.noun(dataset.ratio_base_axis).singular
