"""The frame: what one part of a question asks, each field with who stated it.

A frame is built in two passes. The door-time fields come from the request: the variable
phrase, the places, the window, the unit request, the sources named or excluded, the
operations and the open roles (`door_frame`). The post-recall fields
need the recalled shapes' vocabularies, since a phrase's codes are read per vocabulary:
the measure, the normalisation, the price basis read from the text, and what the frame
could not settle (`Frame.unresolved`). `region.phase_rank_bind` completes them once the
eligible shapes' vocabularies have read the text.

The two text doors build one frame and differ in who states each field (`StatedBy`). On
`raw_text`, extraction states the variable, places, directions, members, dates,
operations, open roles and dropped phrases, and code readers state the unit phrase and the
sources. On `parsed`, the caller states all of them. The `dataset_resubmit` door resolves a
posted identity without recall or bind and builds no frame.

A field the question leaves unstated carries its default with `stated_by="default"`: the
window reads every year, the statType is `measuredValue` unless a measure word states
another, the form is a level, the price basis is the dataset's declared default reading,
and each constraint slot takes its declared default (`FrameMeasure.slot_defaults`).

The frame is built from the `ResolveState` fields the stages read, and it is complete before
bind. The bind prompt shows it to the model as fixed context (`bind._frame_lines`). The status
rule reads its stat type and price basis to decide which axes the question closes
(`phase_place_materialise._open_axes`), and a refusal that flags the reading bind named as
nearest states its open roles (`phase_place_materialise._declined_open_roles`). The form and
the denominator the quantity read's model states before bind (`ResolveState.stated_form`,
`ResolveState.named_denominator`) enter the normalisation stated by `model`.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from code_index.shared import get_index
from search.engine.extract import DateRequest
from search.engine.qualifier_match import QualifierMatch, QualifierReadings, QualifierTie
from search.engine.quantity import NamedDenominator, NamedNormalisation, QuantityRequirement
from search.engine.role_text import DoorRole
from search.engine.source_scope import SourceScope
from search.models import (
    EntryPath,
    IdentityUnitRequest,
    Operation,
    RoleAxis,
    SourceMention,
    TimeWindow,
)

if TYPE_CHECKING:
    # The region package imports this module through `events`.
    from search.engine.region.place_set import PlaceSet

type StatedBy = Literal["caller", "extraction", "reader", "model", "default"]
"""Who stated a frame field: the caller, the extraction model, a code reader over the
request's words, the quantity read's model over the variable phrase
(`quantity_read.read_stated_quantity`), or the default the question left in place."""

type Direction = Literal["origin", "destination", "both", "undirected"]
"""The flow axis a place is named on. `both` is a place the door put on both axes, as a
self-flow; `undirected` is a place named with no axis."""

type FrameReading = Literal["default", "members"]
"""The reading asked of a place: its members each as their own place, or the default,
which is the set's node where it has one and its members otherwise
(`region.place_set.PlaceSet.dcids`)."""

type FrameWindow = Literal["all", "latest"] | TimeWindow
"""Every year the data holds, the latest year, or a window of years."""

type UnresolvedKind = Literal["direction", "span", "group_reading"]

_DEFAULT_STAT_TYPE = "measuredValue"
_DEFAULT_FORM = "level"
_PRICE_BASIS_PROPERTY = "measurementQualifier"


class _FrameModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Stated[T](_FrameModel):
    """A frame value and who stated it."""

    value: T
    stated_by: StatedBy


class FramePlace(_FrameModel):
    """One place the part names, keyed by the request's text for it."""

    name: str = Field(
        description="the request's text for the place; the dcid on the dataset_resubmit door"
    )
    stated_by: StatedBy = Field(description="who named the place")
    direction: Direction = Field(description="the flow axis the place is named on")
    direction_stated: bool = Field(
        description=(
            "whether the request's words give the place its direction; a role read off "
            "the measure, as in a possessive, states none"
        )
    )
    reading: FrameReading = Field(description="the reading asked of the place")
    pin: str | None = Field(
        default=None, description="the dcid the caller pinned for the place, when it gave one"
    )
    dcids: tuple[str, ...] = Field(
        default=(),
        description=(
            "the places the name stands for under its reading, once resolved; empty before "
            "place resolution and for a name that did not resolve"
        ),
    )


class VocabularyCodes(_FrameModel):
    """The codes the part's words match in one recalled vocabulary."""

    vocabulary: tuple[str, ...] = Field(
        description="the vocabulary's constraint properties, sorted"
    )
    codes: tuple[str, ...] = Field(description="the code dcids matched, in match order")


class SlotDefault(_FrameModel):
    """The value a constraint slot takes where the question names none."""

    property: str
    value: str
    stated_by: Literal["default"] = "default"


class FrameMeasure(_FrameModel):
    """What the part measures, as the recalled vocabularies read its words."""

    codes: Stated[tuple[VocabularyCodes, ...]] = Field(
        description="the codes the words match, per recalled vocabulary"
    )
    measure_class: Stated[str] | None = Field(
        default=None,
        description="the measure class a word names (prevalence, incidence, mortality)",
    )
    stat_type: Stated[str] = Field(
        description="the statType: measuredValue unless a measure word states another"
    )
    dropped: Stated[tuple[str, ...]] = Field(
        description="the qualifier phrases the door dropped from the variable"
    )
    slot_defaults: tuple[SlotDefault, ...] = Field(
        default=(),
        description="the declared default of each constraint slot the recalled shapes hold",
    )


class FrameNormalisation(_FrameModel):
    """The form the part asks its measure in."""

    form: Stated[tuple[str, ...]] = Field(
        description=(
            "the normalisations the words name (per_person, per_head_base, per_area, rate, "
            "share), sorted; level when they name none"
        )
    )
    denominator: Stated[str] | None = Field(
        default=None,
        description=(
            "the denominator the words name, as the quantity read chose it: a base the "
            "eligible shapes divide by under its graph name, else the words that name it; "
            "None when the words name none"
        ),
    )


class Unresolved(_FrameModel):
    """Something the frame could not settle, with its possible readings."""

    kind: UnresolvedKind = Field(
        description=(
            "direction: a place named with no axis on a question a recalled shape gives "
            "roles; span: words that match two or more codes, or a code and a measure "
            "value; group_reading: a set holding both a node and members, read by default"
        )
    )
    subject: str = Field(description="the place name or the words")
    readings: tuple[str, ...] = Field(description="the readings the subject admits")


class Frame(_FrameModel):
    """What one part asks.

    `variable`, `places`, `window`, `unit`, `sources`, `operations`, `open_roles` and
    `price_basis` are door-time. `measure`, `normalisation` and `unresolved` are set after
    recall (`with_recall`), and stay empty on the dataset_resubmit door, which runs no
    reader. A place's `dcids` and resolved reading are set then too.
    """

    variable: Stated[str]
    places: tuple[FramePlace, ...] = ()
    window: Stated[FrameWindow]
    unit: Stated[str | IdentityUnitRequest] | None = Field(
        default=None,
        description=(
            "the unit request: the phrase on the raw_text and parsed doors, the completed "
            "request an identity carries on dataset_resubmit"
        ),
    )
    sources: Stated[tuple[SourceMention, ...]] | None = Field(
        default=None, description="the sources the part names or excludes, when it names any"
    )
    operations: Stated[tuple[Operation, ...]]
    open_roles: Stated[tuple[RoleAxis, ...]] = Field(
        description="the flow axes asked member by member, sorted"
    )
    price_basis: Stated[str | None] = Field(
        description=(
            "the measurementQualifier the words name; None stated by default reads the "
            "dataset's declared default reading"
        )
    )
    measure: FrameMeasure | None = None
    normalisation: FrameNormalisation | None = None
    unresolved: tuple[Unresolved, ...] = ()

    @property
    def window_defaulted(self) -> bool:
        """Whether the question named no window, so the frame reads every year."""
        return self.window.stated_by == "default"


def _language(door: EntryPath) -> StatedBy:
    """Who states the fields a door reads from the question's words."""
    return "extraction" if door == "raw_text" else "caller"


def _reader(door: EntryPath) -> StatedBy:
    """Who states the fields code reads from the raw_text door's words, and the caller
    elsewhere."""
    return "reader" if door == "raw_text" else "caller"


def _direction(roles: Sequence[DoorRole]) -> Direction:
    axes = {role.axis for role in roles}
    if not axes:
        return "undirected"
    if len(axes) == 2:
        return "both"
    return next(iter(axes))


def _window(date_request: DateRequest | None, stated_by: StatedBy) -> Stated[FrameWindow]:
    if date_request is None:
        return Stated[FrameWindow](value="all", stated_by="default")
    if date_request.latest:
        return Stated[FrameWindow](value="latest", stated_by=stated_by)
    window = date_request.asked_window
    if window is None:
        return Stated[FrameWindow](value="all", stated_by="default")
    return Stated[FrameWindow](value=window, stated_by=stated_by)


def _stated_operations(
    operations: Iterable[Operation], stated_by: StatedBy
) -> Stated[tuple[Operation, ...]]:
    return Stated[tuple[Operation, ...]](value=tuple(operations), stated_by=stated_by)


def _stated_open_roles(
    open_roles: Iterable[RoleAxis], stated_by: StatedBy
) -> Stated[tuple[RoleAxis, ...]]:
    return Stated[tuple[RoleAxis, ...]](value=tuple(sorted(set(open_roles))), stated_by=stated_by)


def _default_price_basis() -> Stated[str | None]:
    return Stated[str | None](value=None, stated_by="default")


def door_frame(
    *,
    door: EntryPath,
    variable: str,
    entities: Sequence[str],
    entity_roles: Mapping[str, Sequence[DoorRole]],
    member_entities: Collection[str],
    pins: Mapping[str, str],
    date_request: DateRequest | None,
    unit_text: str | None,
    scope: SourceScope | None,
    operations: Sequence[Operation],
    open_axes: Collection[RoleAxis],
) -> Frame:
    """The door-time frame of a part posted on the raw_text or parsed door.

    `entities` are the part's place names, `entity_roles` the axes the door gave each,
    `member_entities` the names asked for as their members, and `pins` the dcid the caller
    pinned for a name. `scope` is the part's compiled source mentions. `open_axes` are the
    axes the part asks every member of, its operations' among them.
    """
    language = _language(door)
    places = tuple(
        FramePlace(
            name=name,
            stated_by=language,
            direction=_direction(entity_roles.get(name, ())),
            direction_stated=any(role.stated for role in entity_roles.get(name, ())),
            reading="members" if name in member_entities else "default",
            pin=pins.get(name),
        )
        for name in dict.fromkeys(entities)
    )
    return Frame(
        variable=Stated[str](value=variable, stated_by=language),
        places=places,
        window=_window(date_request, language),
        unit=(
            None
            if unit_text is None
            else Stated[str | IdentityUnitRequest](value=unit_text, stated_by=_reader(door))
        ),
        sources=(
            None
            if scope is None
            else Stated[tuple[SourceMention, ...]](value=scope.mentions, stated_by=_reader(door))
        ),
        operations=_stated_operations(operations, language),
        open_roles=_stated_open_roles(open_axes, language),
        price_basis=_default_price_basis(),
    )


def _stat_type(variable: str) -> Stated[str]:
    """The statType the variable's words state, by the first pattern the code index declares
    that matches them (`CodeIndex.stat_type_words`), else `measuredValue` by default."""
    text = variable.lower()
    for declaration in get_index().stat_type_words:
        if declaration.pattern.search(text):
            return Stated[str](value=declaration.stat_type, stated_by="reader")
    return Stated[str](value=_DEFAULT_STAT_TYPE, stated_by="default")


def _normalisation(
    requirement: QuantityRequirement,
    stated_form: NamedNormalisation | None,
    denominator: NamedDenominator | None,
) -> FrameNormalisation:
    """The forms the leg's quantity requirement names, else a level by default; and
    `denominator`, the base the quantity read chose, by its graph name, else by the words
    that name it. A form a wording pattern names is stated by the reader, and
    `stated_form`, the form the quantity read joined when none did
    (`quantity.with_stated_form`), by the model."""
    named = tuple(sorted(requirement.named))
    if not named:
        form = Stated[tuple[str, ...]](value=(_DEFAULT_FORM,), stated_by="default")
    else:
        stated_by: StatedBy = "reader" if stated_form is None else "model"
        form = Stated[tuple[str, ...]](value=named, stated_by=stated_by)
    base = (
        None
        if denominator is None
        else Stated[str](
            value=denominator.base_words if denominator.ref is None else denominator.ref.label,
            stated_by="model",
        )
    )
    return FrameNormalisation(form=form, denominator=base)


def _price_basis(matches: Iterable[QualifierMatch]) -> Stated[str | None]:
    """The one measurementQualifier the words match, else the declared default reading."""
    named = {match.code_dcid for match in matches if match.code_system == _PRICE_BASIS_PROPERTY}
    if len(named) == 1:
        return Stated[str | None](value=named.pop(), stated_by="reader")
    return _default_price_basis()


def _unresolved(
    places: Sequence[FramePlace],
    place_sets: Mapping[str, PlaceSet],
    *,
    role_bearing: bool,
    ties: Iterable[QualifierTie],
    readings: Iterable[QualifierReadings],
) -> tuple[Unresolved, ...]:
    """A place named with no axis on a question a recalled shape gives roles, words
    matching two or more codes or a code and a measure value, and a set holding a node
    and members read by default."""
    unresolved: list[Unresolved] = []
    if role_bearing:
        unresolved.extend(
            Unresolved(kind="direction", subject=place.name, readings=("origin", "destination"))
            for place in places
            if place.direction == "undirected" and place.name in place_sets
        )
    unresolved.extend(
        Unresolved(
            kind="span",
            subject=tie.matched_text,
            readings=tuple(dict.fromkeys(row.code_dcid for row in tie.rows)),
        )
        for tie in ties
    )
    unresolved.extend(
        Unresolved(
            kind="span",
            subject=reading.matched_text,
            readings=(reading.code.code_dcid, reading.measure.code_dcid),
        )
        for reading in readings
    )
    unresolved.extend(
        Unresolved(kind="group_reading", subject=name, readings=("node", "members"))
        for name, place_set in place_sets.items()
        if place_set.node is not None and place_set.members and place_set.reading == "default"
    )
    return tuple(unresolved)


def with_recall(
    frame: Frame,
    *,
    dropped_phrases: Sequence[str],
    matches: Mapping[frozenset[str], Sequence[QualifierMatch]],
    ties: Iterable[QualifierTie],
    readings: Iterable[QualifierReadings],
    requirement: QuantityRequirement,
    stated_form: NamedNormalisation | None = None,
    named_denominator: NamedDenominator | None = None,
    measure_class: str | None,
    place_sets: Mapping[str, PlaceSet],
    role_bearing: bool,
    slot_defaults: Iterable[SlotDefault],
) -> Frame:
    """`frame` completed after recall.

    Each resolved place takes its set's reading and places (`PlaceSet.dcids`).
    `dropped_phrases` are the leg's dropped phrases, its operations' phrases among them,
    which the frame leaves out. `matches`, `ties` and `readings` are the codes, tied codes
    and two-reading spans the eligible shapes' vocabularies read in the words, by
    vocabulary for `matches`. `requirement` is the leg's quantity requirement,
    `stated_form` and `named_denominator` the form and base the quantity read stated
    (`ResolveState.stated_form`, `ResolveState.named_denominator`), `measure_class` the
    class a word names, `role_bearing` whether a role-bearing shape outranks the
    subject-only ones (`ResolveState.role_bearing_outranks`), and `slot_defaults` the
    declared defaults of the recalled shapes' slots.
    """
    places = tuple(
        place
        if (place_set := place_sets.get(place.name)) is None
        else place.model_copy(update={"reading": place_set.reading, "dcids": place_set.dcids()})
        for place in frame.places
    )
    operation_phrases = {operation.phrase for operation in frame.operations.value}
    measure = FrameMeasure(
        codes=Stated[tuple[VocabularyCodes, ...]](
            value=tuple(
                VocabularyCodes(
                    vocabulary=tuple(sorted(vocabulary)),
                    codes=tuple(dict.fromkeys(match.code_dcid for match in vocabulary_matches)),
                )
                for vocabulary, vocabulary_matches in matches.items()
            ),
            stated_by="reader",
        ),
        measure_class=(
            None if measure_class is None else Stated[str](value=measure_class, stated_by="reader")
        ),
        stat_type=_stat_type(frame.variable.value),
        dropped=Stated[tuple[str, ...]](
            value=tuple(phrase for phrase in dropped_phrases if phrase not in operation_phrases),
            stated_by=frame.variable.stated_by,
        ),
        slot_defaults=tuple(slot_defaults),
    )
    return frame.model_copy(
        update={
            "places": places,
            "measure": measure,
            "normalisation": _normalisation(requirement, stated_form, named_denominator),
            "price_basis": _price_basis(
                match for vocabulary_matches in matches.values() for match in vocabulary_matches
            ),
            "unresolved": _unresolved(
                places, place_sets, role_bearing=role_bearing, ties=ties, readings=readings
            ),
        }
    )
