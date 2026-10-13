"""Decide whether a nominated alternate is another reading of the quantity the query names.

A shape reads onto declared axes: normalisation, measurement qualifier, stat type,
residence, and each constraint property. Structure refuses an alternate through four
operations on those axes, and where structure is silent the bind model's nomination
stands:

- pin: the query names values on the axis, the alternate honours none of them, and its
  value differs from the elected reading's; on normalisation, the alternate fails the
  materialise gate's rule (`quantity.serves_quantity`): it does not honour every form the
  query names, or divides by another denominator than the one the query names, and the
  elected reading serves the rule or is of another class;
- specialisation: the alternate holds a value on an axis the elected reading leaves open,
  and the query named nothing the alternate honours;
- concept: both hold values on a concept axis and the alternate's lies outside the
  elected reading's;
- default: the alternate's price basis ranks below the elected one's in the declared
  order, the query named none, and the elected shape has data.

`inadmissible_reason` applies them in that order, after the equivalent-series twin
exemption, which a twin failing the normalisation pin does not get, and before the
cross-population check, behind every entry point: nominated alternates
(`inadmissible_alternates`), availability substitutes, re-election siblings and candidate
admission. Residence and constraint properties are judged per member, and an
admitted alternate is offered through its passing members (`offered_alternates`). An
`AlternateJudge` holds one elected reading under one leg's pins, and every caller judging
several sets of alternates under them reuses one.

This module depends on ``ShapeDraft``, the equivalent-series registry, the measure index
and the retrieval layer's bound-value and member-fact readers.
"""

from __future__ import annotations

import functools
import re
import weakref
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from code_index.shared import get_index, get_measure_index
from search.engine.extract import _words
from search.engine.function_words import FUNCTION_WORDS
from search.engine.qualifier_match import (
    QualifierMatch,
    QualifierReadings,
    _normalize,
    match_qualifier_readings,
    match_qualifiers,
)
from search.engine.quantity import (
    DenominatorKind,
    NamedDenominator,
    NamedNormalisation,
    QuantityRequirement,
    check_denominator,
    honours,
    normalisation_class,
    serves_every_form,
    serves_quantity,
)
from search.engine.region.equivalent_series import CLUSTER_BY_MEMBER, EquivalentCluster
from search.engine.region.qualifier_readings import reading_for_shape
from search.engine.retrieval.bound_values import (
    RoleKeys,
    bound_values_from_bindings,
    narrowed_to,
    serving_member_rows,
    shape_five_tuple,
    shape_population_arcs,
    shape_query,
)
from search.engine.retrieval.facts import read_name
from search.engine.retrieval.member_facts import ArcsMemberFacts, IndexMemberFacts
from search.engine.shape import ShapeDraft, _declared_role_names, is_role_bearing
from search.engine.unit.lexicon import AMBIGUOUS_CURRENCY_NAMES, CURRENCY_ALIASES
from unit_words.currency_codes import ISO_4217_CODES

if TYPE_CHECKING:
    from code_index import CodeIndex
    from code_index.members import FamilyQuery
    from code_index.types import MeasureIndex, StatVarRow
    from search.engine.bind import SlotBindingDraft
    from search.engine.retrieval.facts import FiveTuple
    from search.engine.retrieval.member_facts import MemberFacts

type MeasureClass = Literal["prevalence", "incidence", "mortality"]

_QUALIFIER_AXIS = "measurementQualifier"

# The measure axis whose values are the measured properties of one measure family.
_MEASURED_PROPERTY_AXIS = "measuredProperty"

# The words naming each measure class, read whole-word from a lowercase phrase or label.
# Incidence also reads from "new" followed within three tokens by "infection(s)" or
# "case(s)", as in "Number of new HIV infections per 1,000 uninfected population" and
# "Number Of New Reported Cases Of ...". A bare "cases" names no class: most "Number of X
# Cases" series are notified or cumulative counts.
_MEASURE_CLASS_RES: tuple[tuple[MeasureClass, re.Pattern[str]], ...] = (
    ("prevalence", re.compile(r"\bprevalence\b")),
    ("incidence", re.compile(r"\bincidence\b|\bnew(?:\s+\S+){0,3}?\s+(?:infections?|cases?)\b")),
    ("mortality", re.compile(r"\b(?:mortality|deaths|death\s+rates?|dying)\b")),
)
# "Living with" names a population ("people living with HIV"), which is a prevalence only
# when no measure word names another class: "Deaths among people living with HIV" is a
# mortality.
_LIVING_WITH_RE = re.compile(r"\bliving\s+with\b")

# A lower-to-upper case boundary inside a camelCase dcid such as `MortalityEvent`.
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")

# The constraint properties that state a place of residence, in the SDG and Data Commons
# vocabularies, and the residence words a query, a label or a value name holds.
_RESIDENCE_PROPERTIES = ("sdg_urbanisation", "placeOfResidenceClassification")
_RESIDENCE_TOKEN_RE = re.compile(r"\b(?:rural|urban)\b")

# The axis a residence read from a label stands on, for a member with no constraint arcs.
_RESIDENCE_LABEL_AXIS = "residence"

# Operation ranks, in the order `inadmissible_reason` applies them, and axis ranks within
# each operation.
_PIN, _SPECIALISATION, _CONCEPT, _DEFAULT, _CROSS_POPULATION = range(5)
_NORMALISATION, _QUALIFIER, _STAT_TYPE, _RESIDENCE, _CONSTRAINT = range(5)

type _Refusal = tuple[int, int, str]


def measure_class(text: str) -> MeasureClass | None:
    """Return the measure class a query phrase or a label names: prevalence, incidence or
    mortality. None when it names none, or more than one."""
    lower = text.lower()
    named = [cls for cls, pattern in _MEASURE_CLASS_RES if pattern.search(lower)]
    if not named and _LIVING_WITH_RE.search(lower):
        return "prevalence"
    return named[0] if len(named) == 1 else None


def shape_measure_class(shape: ShapeDraft) -> MeasureClass | None:
    """Return the measure class the shape's population type names.

    No graph arc separates prevalence, incidence and mortality, so the class is read from
    the population type's name (`ShapeDraft.pop_type_name`), the one name every member of
    the family shares: "Cause-specific mortality rate among children" reads mortality. A
    member's own name speaks for that member alone, and a family's members are a recall
    sample, so neither the label nor member names decide. A camelCase dcid reads as words,
    so `MortalityEvent` reads mortality and `MedicalConditionIncident` reads None.
    """
    if shape.pop_type_name is None:
        return None
    return measure_class(_CAMEL_BOUNDARY_RE.sub(" ", shape.pop_type_name))


# Currency names from the unit binder's lexicon, as word tuples.
_CURRENCY_NAMES: frozenset[tuple[str, ...]] = frozenset(
    tuple(_words(name))
    for name in (*CURRENCY_ALIASES, *AMBIGUOUS_CURRENCY_NAMES, *ISO_4217_CODES)
    if _words(name)
)
_LONGEST_CURRENCY_NAME = max(len(name) for name in _CURRENCY_NAMES)


def _price_basis_words(text: str) -> set[str]:
    """The content words of a price-basis label or synonym outside the currency names in it.

    A currency names the unit, which the user selects apart from the price basis, so
    "constant US dollars" contributes "constant" alone. Names match longest first.
    """
    words = _words(text)
    kept: set[str] = set()
    i = 0
    while i < len(words):
        span = next(
            (
                length
                for length in range(min(_LONGEST_CURRENCY_NAME, len(words) - i), 0, -1)
                if tuple(words[i : i + length]) in _CURRENCY_NAMES
            ),
            0,
        )
        if span:
            i += span
            continue
        if words[i] not in FUNCTION_WORDS:
            kept.add(words[i])
        i += 1
    return kept


@functools.cache
def _price_basis_vocabulary(measure_index: MeasureIndex) -> frozenset[str]:
    """The price-basis words of the measurement-qualifier axis's value labels and synonyms
    (`_price_basis_words`)."""
    return frozenset(
        word
        for row in measure_index.values
        if row.axis == _QUALIFIER_AXIS
        for text in (row.label, *row.synonyms)
        for word in _price_basis_words(text)
    )


def leaves_price_basis_unread(phrases: Iterable[str], *, operation_phrases: Iterable[str]) -> bool:
    """Whether a dropped qualifier phrase of the leg uses a price-basis word the matcher did
    not read.

    A phrase counts when it is not one of the leg's operation phrases, shares a word with
    the labels and synonyms of the measurement-qualifier axis's values (their content
    words outside currency names, `_price_basis_words`), and neither `match_qualifiers`
    nor a two-reading span (`match_qualifier_readings`, "PPP") finds a
    measurement-qualifier value in it. On such a leg the query may have named a price
    basis that no match records, so no default reading is imposed. "At 2015 base-year
    prices" counts, and "in naira", "in US dollars" and "by sector" do not.
    """
    index, measure_index = get_index(), get_measure_index()
    vocabulary = _price_basis_vocabulary(measure_index)
    operations = {_normalize(phrase) for phrase in operation_phrases}
    return any(
        _normalize(phrase) not in operations
        and not vocabulary.isdisjoint(_words(phrase))
        and not any(
            match.code_system == _QUALIFIER_AXIS
            for match in (
                *match_qualifiers(phrase, index, measure_index),
                *(r.measure for r in match_qualifier_readings(phrase, index, measure_index)),
            )
        )
        for phrase in phrases
    )


@dataclass(frozen=True)
class QueryPins:
    """What the query and the bind step say about the reading an alternate must keep.

    `normalisation` and `residence` are named by the query's words: `normalisation` from
    strong signals only, which a pin reads, and `normalisation_framing` from loose ones as
    well, which an alternate may honour (the leg's `QuantityRequirement`). `qualifiers` is the reading set of every
    measurement qualifier a qualifier match names, and `constraint_matches` the values
    other qualifier matches name, by property, widened as a binding of each reads
    (`CodeIndex.expansion`).
    `qualifier_readings` are spans naming a code and a price basis ("PPP"). A span pins
    the price basis only when the elected shape carries that reading. `bindings` narrow
    the elected reading, and a property in `engine_authored` (a measure election, a
    declared default) narrows it and counts as unnamed. `unread_price_basis` and
    `elected_has_data` gate the default operation. `denominator_kinds` classifies the
    StatVar denominators of the shapes judged (`normalisation_class`), and
    `named_denominator` is the denominator the question names, as the quantity read chose
    it, `None` when it names none or no bind has run.
    """

    normalisation: frozenset[NamedNormalisation] = frozenset()
    normalisation_framing: frozenset[NamedNormalisation] = frozenset()
    residence: frozenset[str] = frozenset()
    qualifiers: frozenset[str] = frozenset()
    constraint_matches: Mapping[str, frozenset[str]] = field(default_factory=dict)
    qualifier_readings: tuple[QualifierReadings, ...] = ()
    bindings: tuple[SlotBindingDraft, ...] = ()
    engine_authored: frozenset[str] = frozenset()
    unread_price_basis: bool = False
    elected_has_data: bool = False
    denominator_kinds: Mapping[str, DenominatorKind] = field(default_factory=dict)
    named_denominator: NamedDenominator | None = None


def query_pins(
    variable: str,
    *,
    bindings: Sequence[SlotBindingDraft] = (),
    engine_authored: frozenset[str] = frozenset(),
    qualifier_matches: Sequence[QualifierMatch] = (),
    qualifier_readings: Sequence[QualifierReadings] = (),
    unread_price_basis: bool = False,
    elected_has_data: bool = False,
    denominator_kinds: Mapping[str, DenominatorKind] | None = None,
    named_denominator: NamedDenominator | None = None,
    quantity: QuantityRequirement,
) -> QueryPins:
    """Read the pins of one leg from its phrase, bindings, qualifier matches and
    two-reading spans. The normalisation pins are the leg's `quantity`
    (`ResolveState.quantity_requirement`), and `named_denominator` the denominator bind
    chose (`ResolveState.named_denominator`)."""
    index, measure_index = get_index(), get_measure_index()
    qualifiers: set[str] = set()
    constraint_matches: dict[str, set[str]] = {}
    for match in qualifier_matches:
        if match.code_system == _QUALIFIER_AXIS:
            qualifiers |= measure_index.reading_set(match.code_dcid)
        else:
            constraint_matches.setdefault(match.code_system, set()).update(
                index.expansion(match.code_dcid)
            )
    return QueryPins(
        normalisation=quantity.named,
        normalisation_framing=quantity.framing,
        residence=frozenset(_RESIDENCE_TOKEN_RE.findall(variable.lower())),
        qualifiers=frozenset(qualifiers),
        constraint_matches={prop: frozenset(values) for prop, values in constraint_matches.items()},
        qualifier_readings=tuple(qualifier_readings),
        bindings=tuple(bindings),
        engine_authored=engine_authored,
        unread_price_basis=unread_price_basis,
        elected_has_data=elected_has_data,
        denominator_kinds=denominator_kinds or {},
        named_denominator=named_denominator,
    )


def _one_measure(first: str, second: str) -> bool:
    """Whether two measured properties measure one quantity: equal, or both values of the
    declared `measuredProperty` measure axis, as an amount and a grant equivalent of one
    flow are (`MeasureIndex.axes`)."""
    if first == second:
        return True
    measure_index = get_measure_index()
    if not any(axis.axis == _MEASURED_PROPERTY_AXIS for axis in measure_index.axes):
        return False
    on_axis = {row.dcid for row in measure_index.values if row.axis == _MEASURED_PROPERTY_AXIS}
    return first in on_axis and second in on_axis


def is_cross_population_false_friend(elected: ShapeDraft, alternate: ShapeDraft) -> bool:
    """Return whether all three structural fields differ.

    The fields are population type, measured property, and role structure.
    Any one difference can mark a valid variant, such as a rate and its count;
    all three leave label wording as the only link. Equal population types
    protect same-population siblings. Alternate admission and the pre-bind flow
    prune both use this predicate. Only admission applies the registry-twin exemption.
    """
    return (
        elected.pop_type_dcid != alternate.pop_type_dcid
        and elected.meas_prop_dcid != alternate.meas_prop_dcid
        and is_role_bearing(elected) != is_role_bearing(alternate)
    )


def _registry_twins(
    served_dcids: Collection[str], alternate: ShapeDraft
) -> dict[str, EquivalentCluster]:
    """Return the alternate's members that share a twin cluster with another member of the
    elected reading, each with that cluster.

    Members are compared one each, because a shape may include members outside the
    cluster. Nominal GDP's shape holds sector value added beside GDP. Both member sets are
    intersected with the registry first, so a reading outside it, such as any CRS flow,
    costs no lookup. The downstream diversion applies its own corroboration gate.
    """
    registered = CLUSTER_BY_MEMBER.keys()
    served: dict[EquivalentCluster, set[str]] = {}
    for dcid in registered & served_dcids:
        served.setdefault(CLUSTER_BY_MEMBER[dcid], set()).add(dcid)
    if not served:
        return {}
    return {
        dcid: cluster
        for dcid in alternate.sv_arc_facts or {}
        if dcid in registered
        and (cluster := CLUSTER_BY_MEMBER[dcid]) in served
        and served[cluster] - {dcid}
    }


def _family(shape: ShapeDraft) -> tuple[object, ...]:
    """The family a shape belongs to: population type, measured property and role
    structure. Constraint values compare only inside one family."""
    return (
        shape.pop_type_dcid,
        shape.meas_prop_dcid,
        _declared_role_names(shape),
        shape.optional_roles,
    )


@dataclass(frozen=True)
class _Member:
    """One member's readings on the per-member axes. `dcid` is `None` for the stand-in of
    a shape with no member facts."""

    dcid: str | None
    constraints: Mapping[str, str]
    unresolved: frozenset[str]
    residence: frozenset[str]
    residence_axis: str

    @property
    def properties(self) -> tuple[str, ...]:
        """The constraint properties the member asserts, resolved or not, residence aside."""
        return tuple(
            prop
            for prop in (*self.constraints, *sorted(self.unresolved - self.constraints.keys()))
            if prop not in _RESIDENCE_PROPERTIES
        )


def _label_residence(label: str) -> frozenset[str]:
    return frozenset(_RESIDENCE_TOKEN_RE.findall(label.lower()))


def _read_member(dcid: str, facts: MemberFacts, label: str) -> _Member:
    """Read a member. A member with no constraint arcs, such as a World Bank code series
    whose code carries the slice, takes its residence from `label`."""
    constraints = facts.read_constraints()
    unresolved = facts.read_unresolved_constraints()
    if not facts.read_declared_constraint_properties():
        return _Member(
            dcid, constraints, unresolved, _label_residence(label), _RESIDENCE_LABEL_AXIS
        )
    names = facts.read_constraint_value_names()
    residence_props = [prop for prop in _RESIDENCE_PROPERTIES if prop in constraints]
    tokens = frozenset(
        token
        for prop in residence_props
        for token in _RESIDENCE_TOKEN_RE.findall(
            names.get(constraints[prop], constraints[prop]).lower()
        )
    )
    axis = residence_props[0] if residence_props else _RESIDENCE_LABEL_AXIS
    return _Member(dcid, constraints, unresolved, tokens, axis)


# Index-sourced member readings by index object id, then by five-tuple and the datasets
# that serve it (`serving_member_rows`). A weakref finalizer removes an index's entry
# before Python can reuse its id.
_index_members_cache: dict[
    int, dict[tuple[FiveTuple, tuple[str, ...] | None], tuple[_Member, ...]]
] = {}


def _index_members(
    shape: ShapeDraft,
    datasets: tuple[str, ...] | None,
    rows: Sequence[StatVarRow],
    index: CodeIndex,
) -> tuple[_Member, ...]:
    """The readings of `rows`, the index rows of `shape`'s five-tuple that `datasets` serve
    (`serving_member_rows`), read once per five-tuple and datasets per index. A row with no
    constraint reads its residence from no label here."""
    key = id(index)
    by_rows = _index_members_cache.get(key)
    if by_rows is None:
        by_rows = _index_members_cache[key] = {}
        weakref.finalize(index, _index_members_cache.pop, key, None)
    rows_key = (shape_five_tuple(shape), datasets)
    if rows_key not in by_rows:
        by_rows[rows_key] = tuple(
            _read_member(row.dcid, IndexMemberFacts(row=row), "") for row in rows
        )
    return by_rows[rows_key]


@dataclass(frozen=True)
class _Verdict:
    """An alternate's first refusal, or `None`, and the members that pass the per-member
    axes, in member order."""

    reason: str | None
    passing: tuple[_Member, ...]


class AlternateJudge:
    """One elected reading under one leg's pins, judged against any number of alternates.

    The elected reading is the elected shape's members that hold every bound value, or
    every member when none does. A shape the index carries reads the rows of the dataset
    that serves it (`serving_member_rows`), elected under the query the answer path builds
    for it (`_serving_query`): the leg's bindings on the slots it declares, its named
    source, and the role keys `role_keys` gives for it, the keys its identity binds the
    query's places under and the axes it leaves open, none by default. Member readings and
    verdicts are memoised on object identity, since distinct shapes can share one `shape_id`.
    Each memo entry holds its shape, so no other shape takes the same identity while the
    judge lives.
    """

    def __init__(
        self,
        pins: QueryPins,
        elected: ShapeDraft,
        *,
        role_keys: Callable[[ShapeDraft], RoleKeys] = lambda _shape: RoleKeys(),
    ) -> None:
        self.pins = pins
        self.elected = elected
        self._role_keys = role_keys
        self._members: dict[int, tuple[ShapeDraft, tuple[_Member, ...], bool]] = {}
        self._verdicts: dict[int, tuple[ShapeDraft, _Verdict]] = {}
        self._reading_properties = frozenset(
            axis.axis
            for axis in get_measure_index().axes
            if axis.kind == "constraint" and axis.role == "free"
        )
        bound = bound_values_from_bindings(pins.bindings)
        members, _ = self.members(elected)
        self.served = (
            tuple(
                member
                for member in members
                if all(member.constraints.get(prop) in values for prop, values in bound.items())
            )
            or members
        )
        self.served_maps = [dict(member.constraints) for member in self.served]
        self.served_dcids = frozenset(
            member.dcid for member in self.served if member.dcid is not None
        )
        self.qualifiers = pins.qualifiers | frozenset(
            dcid
            for readings in pins.qualifier_readings
            if reading_for_shape(readings, elected) is readings.measure
            for dcid in get_measure_index().reading_set(readings.measure.code_dcid)
        )
        self.normalisation = normalisation_class(elected, kinds=pins.denominator_kinds)
        # Whether the elected shape serves the query's quantity by the materialise gate's
        # rule. When it does, an alternate is held to the request alone; when not, an
        # alternate of the elected class is no further from the request than the election,
        # and the gate refuses any answer that rests on such readings alone.
        self.elected_serves = serves_quantity(
            QuantityRequirement(pins.normalisation),
            pins.named_denominator,
            elected,
            pins.denominator_kinds,
        )
        self.residence = frozenset().union(*(member.residence for member in self.served))
        self.holds = frozenset(prop for member in self.served for prop in member.properties)
        index = get_index()
        values: dict[str, set[str]] = {}
        for member in self.served:
            for prop, value in member.constraints.items():
                values.setdefault(prop, set()).add(value)
        self.values = {
            prop: frozenset(found for value in held for found in index.expansion(value))
            for prop, held in values.items()
        }
        self.named = frozenset(
            binding.property_dcid
            for binding in pins.bindings
            if binding.property_dcid
            and binding.property_dcid not in pins.engine_authored
            and (
                (binding.kind in ("value", "set") and binding.evidence == "mentioned")
                or (binding.kind == "unbound" and binding.evidence == "requested_all")
            )
        )

    def _serving_query(self, shape: ShapeDraft) -> FamilyQuery:
        """The family query `shape` is served under (`shape_query`): the leg's bindings on
        the slots `shape` declares, with the source binding, under `shape`'s role keys. For
        the elected shape these are the answer's own bindings."""
        declared = {slot.property_dcid for slot in shape.slot_keys if slot.property_dcid}
        bindings = [
            binding
            for binding in self.pins.bindings
            if binding.property_dcid is None or binding.property_dcid in declared
        ]
        role_keys = self._role_keys(shape)
        return shape_query(shape, bindings, role_keys.bound, open_role_keys=role_keys.open)

    def members(self, shape: ShapeDraft) -> tuple[tuple[_Member, ...], bool]:
        """The shape's members, from the index rows of the dataset that serves it
        (`serving_member_rows`) when the index carries it, else every member it stands for
        (`shape_population_arcs`), with whether they came from the index. A shape with no
        member facts reads as one stand-in.

        A lone member reads its residence label from the shape's label. A member of a
        larger shape reads its own name, since the shape is labelled from whichever member
        recall ranked first and a total must not read as its rural sibling.
        """
        key = id(shape)
        if key not in self._members:
            index = get_index()
            datasets, rows = serving_member_rows(shape, index, self._serving_query(shape))
            arcs = shape_population_arcs(shape)
            count = len(rows) if rows else len(arcs)

            def label(dcid: str) -> str:
                if count == 1:
                    return shape.label
                return read_name(arcs.get(dcid) or {}) or ""

            if rows:
                # A row with no constraint reads its residence from a label, which belongs
                # to this shape, so only the cached readings of constrained rows stand.
                members = tuple(
                    member
                    if member.constraints
                    else _read_member(row.dcid, IndexMemberFacts(row=row), label(row.dcid))
                    for row, member in zip(
                        rows, _index_members(shape, datasets, rows, index), strict=True
                    )
                )
            else:
                members = tuple(
                    _read_member(dcid, ArcsMemberFacts(arcs=a), label(dcid))
                    for dcid, a in arcs.items()
                )
            self._members[key] = (
                shape,
                members
                or (
                    _Member(
                        None, {}, frozenset(), _label_residence(shape.label), _RESIDENCE_LABEL_AXIS
                    ),
                ),
                bool(rows),
            )
        _, members, index_sourced = self._members[key]
        return members, index_sourced

    def verdict(self, alternate: ShapeDraft) -> _Verdict:
        key = id(alternate)
        if key not in self._verdicts:
            self._verdicts[key] = (alternate, self._judge(alternate))
        return self._verdicts[key][1]

    def _judge(self, alternate: ShapeDraft) -> _Verdict:
        members, _ = self.members(alternate)
        quantity = self._quantity_refusal(alternate)
        # A twin that serves the quantity passes for diversion through its twin members
        # alone, so its representative and its offered series are the registered encoding of
        # the elected reading.
        twins = _registry_twins(self.served_dcids, alternate)
        if twins and quantity is None:
            return _Verdict(None, tuple(m for m in members if m.dcid in twins))
        same_family = _family(alternate) == _family(self.elected)
        failures = [self._member_refusal(member, same_family=same_family) for member in members]
        passing = tuple(m for m, failure in zip(members, failures, strict=True) if failure is None)
        refusals = [
            refusal
            for refusal in (
                quantity,
                self._normalisation_refusal(alternate),
                self._qualifier_pin(alternate),
                self._stat_type_concept(alternate),
                self._default_refusal(alternate) if same_family else None,
                self._cross_population(alternate),
                None if passing else min(f for f in failures if f is not None),
            )
            if refusal is not None
        ]
        return _Verdict(min(refusals)[2] if refusals else None, passing)

    def _quantity_refusal(self, alternate: ShapeDraft) -> _Refusal | None:
        """The materialise gate's rule (`quantity.serves_quantity`), as a pin: the
        alternate must honour every form the query names, and divide by the denominator it
        names unless its arc leaves the base unverified (`quantity.check_denominator`).
        Beside an election that fails the rule, an alternate of the elected class passes
        (`elected_serves`)."""
        named = self.pins.normalisation
        kinds = self.pins.denominator_kinds
        cls = normalisation_class(alternate, kinds=kinds)
        if not self.elected_serves and cls == self.normalisation:
            return None
        if not serves_every_form(QuantityRequirement(named), alternate, kinds):
            return (
                _PIN,
                _NORMALISATION,
                f"pin:normalisation: the query names {sorted(named)}, and a {cls} "
                "alternate does not honour every one",
            )
        denominator = self.pins.named_denominator
        if denominator is not None and check_denominator(alternate, denominator) == "refused":
            asked = (
                denominator.ref.ref
                if denominator.ref is not None
                else f'"{denominator.base_words}"'
            )
            return (
                _PIN,
                _NORMALISATION,
                f"pin:normalisation: the query names the denominator {asked}, and the "
                f"alternate divides by {alternate.meas_denom_dcid or 'nothing'}",
            )
        return None

    def _normalisation_refusal(self, alternate: ShapeDraft) -> _Refusal | None:
        cls = normalisation_class(alternate, kinds=self.pins.denominator_kinds)
        honoured = any(honours(alternate, cls, n) for n in self.pins.normalisation_framing)
        # Any denominator arc makes the alternate a quotient of the elected level, and a
        # `series` ratio is one only when its numerator measures that level: the same
        # population type, and a measured property that is the elected one or shares the
        # declared measure axis with it (`_one_measure`). Grant equivalents as a share of GNI
        # divide an amount of aid; education spending as a share of GDP does not divide a
        # count of education facilities. An `other` alternate states no denominator, so
        # nothing shows it divides the level.
        divides = cls in ("per_person", "per_area", "unknown") or (
            cls == "series"
            and alternate.pop_type_dcid == self.elected.pop_type_dcid
            and _one_measure(alternate.meas_prop_dcid, self.elected.meas_prop_dcid)
        )
        if self.normalisation == "level" and divides:
            if honoured:
                return None
            divisor = alternate.meas_denom_dcid or "an unresolved arc"
            return (
                _SPECIALISATION,
                _NORMALISATION,
                f"specialisation:normalisation: the alternate divides the elected level "
                f"({cls}, by {divisor}) and the query names no such framing",
            )
        return None

    def _qualifier_pin(self, alternate: ShapeDraft) -> _Refusal | None:
        named = self.qualifiers
        value = alternate.meas_qual_dcid
        if not named or value is None or value in named or value == self.elected.meas_qual_dcid:
            return None
        return (
            _PIN,
            _QUALIFIER,
            f"pin:qualifier: the query names {sorted(named)} and the alternate measures {value}",
        )

    def _stat_type_concept(self, alternate: ShapeDraft) -> _Refusal | None:
        elected, value = self.elected.stat_type_dcid, alternate.stat_type_dcid
        if not elected or not value or elected == value:
            return None
        return (
            _CONCEPT,
            _STAT_TYPE,
            f"concept:statType: the alternate is a {value} where the elected reading is a "
            f"{elected}",
        )

    def _default_refusal(self, alternate: ShapeDraft) -> _Refusal | None:
        """Refuse a lower-ranked price basis of the elected one's family.

        A two-reading span ("PPP") names a price basis when the elected shape or the
        alternate carries its measure reading, as for the pre-bind collapse
        (`phase_rank_bind._default_reading`), so "GDP PPP" keeps a PPP alternate beside a
        nominal election that carries neither reading.
        """
        pins = self.pins
        if (
            not pins.elected_has_data
            or self.qualifiers
            or pins.unread_price_basis
            or any(
                reading_for_shape(readings, alternate) is readings.measure
                for readings in pins.qualifier_readings
            )
        ):
            return None
        order = get_measure_index().reading_order(_QUALIFIER_AXIS)
        elected, value = self.elected.meas_qual_dcid, alternate.meas_qual_dcid
        if elected not in order or value not in order or order.index(value) <= order.index(elected):
            return None
        return (
            _DEFAULT,
            _QUALIFIER,
            f"default:qualifier: the query names no price basis, and {value} ranks below the "
            f"elected {elected}, which has data",
        )

    def _cross_population(self, alternate: ShapeDraft) -> _Refusal | None:
        if not is_cross_population_false_friend(self.elected, alternate):
            return None
        return (
            _CROSS_POPULATION,
            0,
            "cross_population: the alternate differs in population type, measured property "
            "and role structure, leaving only label wording shared",
        )

    def _member_refusal(self, member: _Member, *, same_family: bool) -> _Refusal | None:
        refusals = [self._residence_refusal(member)]
        if same_family:
            refusals += [self._constraint_refusal(member, prop) for prop in member.properties]
        return min((refusal for refusal in refusals if refusal is not None), default=None)

    def _residence_refusal(self, member: _Member) -> _Refusal | None:
        tokens, axis, named = member.residence, member.residence_axis, self.pins.residence
        if not tokens:
            return None
        if named and tokens.isdisjoint(named) and tokens != self.residence:
            return (
                _PIN,
                _RESIDENCE,
                f"pin:{axis}: the query names {sorted(named)} and a member reads {sorted(tokens)}",
            )
        if not self.residence and tokens.isdisjoint(named):
            return (
                _SPECIALISATION,
                _RESIDENCE,
                f"specialisation:{axis}: a member narrows to {sorted(tokens)}, which neither "
                "the query nor the elected reading names",
            )
        if self.residence and not tokens <= self.residence:
            return (
                _CONCEPT,
                _RESIDENCE,
                f"concept:{axis}: a member reads {sorted(tokens)} where the elected reading "
                f"reads {sorted(self.residence)}",
            )
        return None

    def _constraint_refusal(self, member: _Member, prop: str) -> _Refusal | None:
        value = member.constraints.get(prop)
        elected_values = self.values.get(prop, frozenset())
        if prop in self._reading_properties:
            named = self.pins.constraint_matches.get(prop)
            if named and elected_values <= named and value is not None and value not in named:
                return (
                    _PIN,
                    _CONSTRAINT,
                    f"pin:{prop}: the query names {sorted(named)} and a member holds {value}",
                )
            return None
        if prop in self.holds:
            if value is None or value in elected_values:
                return None
            return (
                _CONCEPT,
                _CONSTRAINT,
                f"concept:{prop}: a member holds {value} where the elected reading holds "
                f"{sorted(elected_values)}",
            )
        if prop in self.named:
            return None
        return (
            _SPECIALISATION,
            _CONSTRAINT,
            f"specialisation:{prop}: a member holds {value or 'an unresolved value'} on a "
            "property neither the query nor the elected reading names",
        )

    def offered(self, alternate: ShapeDraft) -> ShapeDraft:
        """The alternate narrowed to its passing members among every member it stands for
        (`shape_population_arcs`): those whose constraint map equals one a member of the
        elected reading holds first, then member order. An alternate the index carries is
        judged on its index rows and offered whole, as is one with no member facts. A
        narrowed alternate stands for its passing members alone (`narrowed_to`)."""
        _, index_sourced = self.members(alternate)
        facts = shape_population_arcs(alternate)
        if index_sourced or not facts:
            return alternate
        passing = self.verdict(alternate).passing
        counterparts = [m for m in passing if dict(m.constraints) in self.served_maps]
        ordered = [*counterparts, *(m for m in passing if m not in counterparts)]
        narrowed = {m.dcid: facts[m.dcid] for m in ordered if m.dcid is not None}
        if list(narrowed) == list(facts):
            return alternate
        return narrowed_to(alternate, narrowed)

    def inadmissible(self, alternates: Sequence[tuple[int, ShapeDraft]]) -> dict[int, str]:
        """Refused alternate indices mapped to their reasons (`inadmissible_reason`)."""
        return {
            index: reason
            for index, alternate in alternates
            if (reason := self.verdict(alternate).reason) is not None
        }

    def twin_alternates(self, alternates: Sequence[tuple[int, ShapeDraft]]) -> list[int]:
        """The indices of the alternates carrying a registry twin of the elected reading
        that the elected reading does not serve, the cluster's preferred member
        (`EquivalentCluster.preferred`) first, then in `alternates` order."""
        found: list[tuple[bool, int]] = []
        for index, alternate in alternates:
            twins = {
                dcid: cluster
                for dcid, cluster in _registry_twins(self.served_dcids, alternate).items()
                if dcid not in self.served_dcids
            }
            if twins:
                preferred = any(cluster.preferred == dcid for dcid, cluster in twins.items())
                found.append((not preferred, index))
        return [index for _, index in sorted(found, key=lambda item: item[0])]


def inadmissible_reason(pins: QueryPins, elected: ShapeDraft, alternate: ShapeDraft) -> str | None:
    """Return why `alternate` is not a reading of `elected` under `pins`, or `None`.

    The reason opens with its code and a colon: `pin:<axis>`, `specialisation:<axis>`,
    `concept:<axis>`, `default:qualifier` or `cross_population`, where `<axis>` is
    `normalisation`, `qualifier`, `statType`, `residence` (read from a label), or a
    constraint property dcid. A registered equivalent-series twin passes for diversion.
    """
    return AlternateJudge(pins, elected).verdict(alternate).reason


def inadmissible_alternates(
    pins: QueryPins, elected: ShapeDraft, alternates: Sequence[tuple[int, ShapeDraft]]
) -> dict[int, str]:
    """Return refused alternate indices mapped to their reasons (`inadmissible_reason`).

    Alternate indices refer to the caller's ``capped_shapes``.
    """
    return AlternateJudge(pins, elected).inadmissible(alternates)


def twin_alternates(
    pins: QueryPins, elected: ShapeDraft, alternates: Sequence[tuple[int, ShapeDraft]]
) -> list[int]:
    """Return the indices of the alternates carrying a registry twin of the elected
    reading that it does not serve (`AlternateJudge.twin_alternates`).

    Candidate admission adds them whether or not bind nominated them, so an answer lists
    every registered twin on the menu and the registry's preference decides the served
    one. Alternate indices refer to the caller's ``capped_shapes``.
    """
    return AlternateJudge(pins, elected).twin_alternates(alternates)


def offered_alternates(
    pins: QueryPins, elected: ShapeDraft, alternates: Mapping[int, ShapeDraft]
) -> dict[int, ShapeDraft]:
    """Return each alternate narrowed to the members it is offered through, by index.

    Passing members whose constraint map equals one the elected reading holds come first,
    so an alternate is represented only by a member the guard passes. Nominal GDP, not a
    sector's value added, stands for a 12-member nominal shape beside an
    inflation-adjusted GDP election.
    """
    judge = AlternateJudge(pins, elected)
    return {index: judge.offered(alternate) for index, alternate in alternates.items()}
