"""The status rule: which readings of a part code offers beside the elected one, and which
answers when the elected one holds no data.

Let P be the elected reading, the alternatives the rest of the admissible menu after the
measure-family collapse, and D(x) whether x has data where the question asks
(`region.availability.has_data_where_asked`). An axis is closed when the frame names a value
for it, or when P holds the axis's stated default: `measuredValue` for the stat type, no
denominator for the form, the declared default reading for the price basis. Otherwise it is
open (`OpenAxes`).

1. When D(P) is false, an equivalent reading of P with data answers: a registry twin
   (`region.equivalent_series`, `StatusPlan.twins`) or another reading of P's five-tuple,
   such as the by-creditor twin of an IDS total (`StatusPlan.equivalents`). Else a
   sibling with data that differs from P only on the measurement qualifier, with no
   role-bearing difference (`StatusPlan.stand_ins`, `shares_role_class`), answers when it
   is the only one, and several are offered as candidates without P. Otherwise the part
   answers no data, and a nominated reading with data rides on the refusal as its
   alternate match (`StatusPlan.riders`).
2. Otherwise the rivals R (`StatusPlan.rivals`) are the alternatives x with D(x) that
   meet (a) or (b), after two exclusions. An alternative equivalent to P never enters R:
   a registry twin is listed on P as an equivalent reading, and another reading of P's
   five-tuple, its denominator read by meaning (`quantity.denominator_meaning`), stands
   in for P only under rule 1. An alternative that differs from P only on
   closed axes never enters R, even when the model nominates it, except that a nominated
   x differing on the denominator meets (b) when the question states a base the quantity
   read matched to no listed denominator, since the question then picked no reading's
   base (`base_unmatched`). (a) x shares P's population type and measured property, and
   differs from it only on the stat type, the measurement qualifier or the denominator,
   at least one of them open (`structural`). (b) the model nominated x (`nominated`).
3. R empty gives `definite`; else `candidates`, P first and R in menu order.
4. A pinned part never produces candidates. A reading that fails the form the question
   asks for or a code it names is inadmissible before bind, so it is never on the menu. A
   reading on another denominator than the one bind reads in the question stays on the
   menu, since the denominator is judged on the bind output, and the alternate guard drops
   it from the menu before this rule reads it (`phase_place_materialise._plan_status`). The
   guard reads the same open axes as this rule (`alternate_guard.QueryPins.open_axes`): it
   refuses a reading for a difference on an axis only when that axis is closed.
5. Scores set no status. A retrieval score orders the menu and reaches confidence only
   (`grounding._grade_confidence`).

D(x) unknown is no evidence either way: such a reading is planned as though it had data,
and the read of it that materialise makes decides. Only a confirmed absence removes a
reading.

The rule reads one pass's menu, so every reading it weighs comes from the pool that pass
elects over (`ShapeDraft.pool`): our candidates, or the UN System Data Commons' on the pass
that runs when ours ends no data (`pipeline._fill_gap`).

Pure module: the caller reads D and the nominations and passes them in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from code_index.labels import PRICE_BASIS_AXIS
from code_index.shared import get_measure_index
from search.engine.quantity import denominator_meaning
from search.engine.shape import ShapeDraft, is_role_bearing, is_un_shape, role_class
from search.models import CandidateBasis

_DEFAULT_STAT_TYPE = "measuredValue"


@dataclass(frozen=True, slots=True)
class OpenAxes:
    """Which of the three axes two readings of one measure can differ on the question
    leaves open for P: neither named by the frame nor held by P at its stated default."""

    stat_type: bool
    denominator: bool
    price_basis: bool


CLOSED_AXES = OpenAxes(stat_type=False, denominator=False, price_basis=False)


@dataclass(frozen=True, slots=True)
class MenuReading:
    """One admissible reading beside P, as the status rule reads it: its menu index, its
    shape, D (`None` when no read settled it), whether the bind model nominated it, and
    whether the equivalent-series registry pairs it with P."""

    index: int
    shape: ShapeDraft
    has_data: bool | None
    nominated: bool = False
    registry_twin: bool = False


@dataclass(frozen=True, slots=True)
class Alternative:
    """A reading the status rule offers beside P, by menu index, with why."""

    index: int
    basis: CandidateBasis


@dataclass(frozen=True, slots=True)
class StatusPlan:
    """What the status rule makes of P's menu, before P's own read settles D(P).

    `rivals` is R under rule 2, which the part offers when P has data. The rest serve rule
    1 and hold readings without a confirmed absence: `twins`, the registry twins of P, which
    the equivalent-series diversion serves or lists on P whether or not P has data;
    `equivalents`, the other readings of P's five-tuple; `stand_ins`, the siblings that
    differ from P only on the measurement qualifier and may share its role class once
    walked (`_may_stand_in`); and `riders`, the nominated readings that ride on a refusal
    as its alternate match. Each is in menu order. `sole` holds when every other admissible
    reading has a confirmed absence, so P, given data, is the only admissible reading with
    data where the question asks.
    """

    rivals: tuple[Alternative, ...] = ()
    twins: tuple[int, ...] = ()
    equivalents: tuple[int, ...] = ()
    stand_ins: tuple[int, ...] = ()
    riders: tuple[int, ...] = ()
    sole: bool = False


def holds_default_price_basis(shape: ShapeDraft) -> bool:
    """Whether `shape` reads at the declared default price basis: it states no
    measurement qualifier, or one spelling the first reading the measure index declares
    for the axis (`MeasureIndex.reading_order`)."""
    qualifier = shape.meas_qual_dcid
    if qualifier is None:
        return True
    index = get_measure_index()
    order = index.reading_order(PRICE_BASIS_AXIS)
    return bool(order) and qualifier in index.reading_set(order[0])


def open_axes(
    elected: ShapeDraft, *, stat_type_named: bool, form_named: bool, price_basis_named: bool
) -> OpenAxes:
    """The axes left open for `elected`: an axis is closed when the question names a value
    for it (`*_named`) or when `elected` holds its stated default."""
    return OpenAxes(
        stat_type=not stat_type_named and elected.stat_type_dcid != _DEFAULT_STAT_TYPE,
        denominator=not form_named
        and (elected.meas_denom_dcid is not None or elected.meas_denom_unresolved),
        price_basis=not price_basis_named and not holds_default_price_basis(elected),
    )


def _differing_axes(elected: ShapeDraft, other: ShapeDraft) -> frozenset[str]:
    """The axes among stat type, price basis and denominator on which the two differ. Two
    denominators differ when they mean another base (`quantity.denominator_meaning`), so a
    series dividing by `Count_Person` and one dividing by the ProvisionalNode `PerCapita`
    hold one five-tuple."""
    differing = set()
    if elected.stat_type_dcid != other.stat_type_dcid:
        differing.add("stat_type")
    if elected.meas_qual_dcid != other.meas_qual_dcid:
        differing.add("price_basis")
    if denominator_meaning(elected) != denominator_meaning(other):
        differing.add("denominator")
    return frozenset(differing)


def _one_measure(elected: ShapeDraft, other: ShapeDraft) -> bool:
    """Whether the two share population type and measured property, and come from one pool."""
    return (
        elected.pop_type_dcid == other.pop_type_dcid
        and elected.meas_prop_dcid == other.meas_prop_dcid
        and is_un_shape(elected) == is_un_shape(other)
    )


def _one_family(elected: ShapeDraft, other: ShapeDraft) -> bool:
    """Whether the two hold one five-tuple, so they read one measure, whatever their role
    class or dataset."""
    return _one_measure(elected, other) and not _differing_axes(elected, other)


def shares_role_class(elected: ShapeDraft, other: ShapeDraft) -> bool:
    """Whether the two declare the same roles, in any arc order, so neither reads a flow the
    other does not."""
    return role_class(elected) == role_class(other)


def _may_stand_in(elected: ShapeDraft, other: ShapeDraft) -> bool:
    """Whether `other` may stand in for `elected` under rule 1: it differs from it only on
    the measurement qualifier, stat type and denominator equal, and shares its role class
    (`shares_role_class`) or is a role-bearing shape beside a subject-only `elected`. The
    role-class walk replaces a recalled role-bearing shape with the subject-only family of
    its five-tuple where the question binds none of its roles, so the walked shape decides:
    rule 1 takes a stand-in only when the shape its walk elects shares the elected shape's
    role class."""
    return (
        _one_measure(elected, other)
        and _differing_axes(elected, other) == {"price_basis"}
        and (
            shares_role_class(elected, other)
            or (is_role_bearing(other) and not is_role_bearing(elected))
        )
    )


def _rival_basis(
    elected: ShapeDraft,
    reading: MenuReading,
    *,
    axes: OpenAxes,
    base_unmatched: bool,
) -> CandidateBasis | None:
    """Why `reading` enters R beside `elected`, or `None` when it does not (rule 2)."""
    other = reading.shape
    if reading.registry_twin or _one_family(elected, other):
        return None
    if _one_measure(elected, other):
        open_ = {
            axis
            for axis, is_open in (
                ("stat_type", axes.stat_type),
                ("price_basis", axes.price_basis),
                ("denominator", axes.denominator),
            )
            if is_open
        }
        differing = _differing_axes(elected, other)
        if differing & open_:
            return "structural"
        if reading.nominated and base_unmatched and "denominator" in differing:
            return "nominated"
        return None
    return "nominated" if reading.nominated else None


def plan_status(
    elected: ShapeDraft,
    menu: Sequence[MenuReading],
    *,
    axes: OpenAxes,
    base_unmatched: bool = False,
) -> StatusPlan:
    """The status rule over `elected`'s admissible alternatives `menu`, in menu order.

    A reading with a confirmed absence (`MenuReading.has_data` false) takes no part. R
    holds every rival; the candidate budget applies to what the part offers
    (`phase_place_materialise._plan_admission`). `base_unmatched` holds when the question
    states a base the quantity read matched to no listed denominator (rule 2)."""
    present = [reading for reading in menu if reading.has_data is not False]
    rivals: list[Alternative] = []
    for reading in present:
        basis = _rival_basis(elected, reading, axes=axes, base_unmatched=base_unmatched)
        if basis is not None:
            rivals.append(Alternative(index=reading.index, basis=basis))
    return StatusPlan(
        rivals=tuple(rivals),
        twins=tuple(reading.index for reading in present if reading.registry_twin),
        equivalents=tuple(
            reading.index
            for reading in present
            if not reading.registry_twin and _one_family(elected, reading.shape)
        ),
        stand_ins=tuple(
            reading.index for reading in present if _may_stand_in(elected, reading.shape)
        ),
        riders=tuple(reading.index for reading in present if reading.nominated),
        sole=not present,
    )
