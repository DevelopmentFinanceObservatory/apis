"""The measure-basis axis: what a part's amounts are counted on (gross or net disbursements,
grant equivalents or commitments), and the bases its dataset could count them on instead.

The basis is a declared product default, after the OECD's own reporting
(`code_index.scope_conventions.MEASURE_BASIS_CONVENTIONS`): it is read from the part's shape
and the years it asks for, never from how the request words its measure. A donor's total is
the OECD headline, grant equivalents for years from 2018 and net flows for years before it or
on both sides of it (`HEADLINE_SERVED_AS`); a recipient's total over every sector is its net
receipts; a part that binds a sector, an instrument or a modality is counted in gross CRS
disbursements. An explicit or carried entry on the axis wins
over the convention (`elect_basis`), and the house policies elect the shape and dataset that
publish the elected basis (`binders.house_policy`).

`basis_rule` is the axis rule. It reads the base identity alone, so a dataset_resubmit,
where no binder runs, offers what the parsed door offered: every declared basis, available
where the identity's dataset publishes it at a covered series and unavailable otherwise,
with the basis the identity holds first in the preference, so a part's derived default is
the basis it resolved and a switch is the only thing that moves it.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from code_index.labels import PRICES_BY_QUALIFIER
from code_index.members import identity_family_query, query_family
from code_index.query import CodeIndex
from code_index.scope_conventions import (
    BASIS_MEASURES,
    BASIS_RESTRICTIONS,
    MEASURE_BASIS_CONVENTIONS,
    OPTION_LABELS,
    BasisConvention,
    BasisMeasure,
    MeasureBasis,
    PartShape,
    WindowClass,
)
from search.models import (
    BaseIdentity,
    Entry,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityShapeSchema,
    Option,
    OptionAvailability,
    ShapeBinding,
    TimeWindow,
)
from search.scope.edits import ScopeTableError, apply_binding
from search.scope.state import AxisRecord
from search.scope.tables import AxisTable, ScopeFacts
from search.scope.vocabulary import measure_basis_option_id

BASIS_RULE = "measure_basis_convention"
"""The declared rule the measure-basis axis applies."""

GRANT_EQUIVALENT_HEADLINE_FROM = 2018
"""The first year the OECD's headline counts official development assistance in grant
equivalents."""

HEADLINE_SERVED_AS: MeasureBasis = "net"
"""The basis a part whose convention is the OECD headline per year (`oecd_headline`) is
counted on. One identity holds one basis, so a window crossing 2018 is counted on net flows
throughout, the headline's basis for its earlier years."""

BASIS_AXES: frozenset[str] = frozenset(
    {"populationType", "measuredProperty", "nettingBasis"}
    | {row.flow_type[0] for row in BASIS_MEASURES}
)
"""The measure axes a basis sets (`code_index.types.MeasureAxis.axis`). Measures that differ
on one differ in basis, which the axis owns, so no other list offers them."""

CONVENTION_DATASETS: frozenset[str] = frozenset(row.dataset for row in MEASURE_BASIS_CONVENTIONS)
"""The datasets the conventions read. A part served from another has no measure-basis
axis."""

_MEASURES: Mapping[MeasureBasis, BasisMeasure] = {row.basis: row for row in BASIS_MEASURES}
_NETTING = "nettingBasis"
_RECIPIENT_DIRECTIONS = ("destination", "both", "undirected")


def window_class(window: TimeWindow | Literal["all", "latest"]) -> WindowClass:
    """Where `window` lies against `GRANT_EQUIVALENT_HEADLINE_FROM`. A window open at its
    start, and one naming no years ("all"), reach back before it; the latest year lies
    after it."""
    if window == "latest":
        return "from_2018"
    if window == "all":
        return "crossing_2018"
    start, end = window.start_year, window.end_year
    if start is not None and start >= GRANT_EQUIVALENT_HEADLINE_FROM:
        return "from_2018"
    if end is not None and end < GRANT_EQUIVALENT_HEADLINE_FROM:
        return "before_2018"
    return "crossing_2018"


def part_shape(
    *,
    directions: Iterable[str],
    admitted_systems: Collection[str],
    sector_systems: Collection[str],
) -> PartShape | None:
    """The shape the conventions read of a part whose places take `directions` and whose
    admitted qualifier matches name codes of `admitted_systems`: a sector or restriction
    part when one of them is a sector system (`sector_systems`, the code systems the code
    index declares sector questions of) or a restriction (`BASIS_RESTRICTIONS`), else a
    recipient part when a place is a recipient or undirected, else a donor part when it
    names places, and None when it names none."""
    restricting = frozenset(sector_systems) | frozenset(BASIS_RESTRICTIONS)
    if restricting.intersection(admitted_systems):
        return "sector_or_restriction"
    directions = tuple(directions)
    if any(direction in _RECIPIENT_DIRECTIONS for direction in directions):
        return "recipient_all_sectors"
    return "donor_only" if directions else None


def sector_systems(index: CodeIndex) -> frozenset[str]:
    """The code systems `index` declares sector questions of (`CodeIndex.house_case_systems`):
    an admitted code of one makes a part a sector part."""
    return frozenset(
        system for system, case in index.house_case_systems().items() if case == "sector"
    )


def convention(shape: PartShape, window: WindowClass) -> BasisConvention:
    """The declared convention of a part of `shape` asking for `window`. Raises
    `LookupError` when the table declares none, which is a gap in the table."""
    for row in MEASURE_BASIS_CONVENTIONS:
        if row.shape == shape and row.window in (window, "any"):
            return row
    raise LookupError(f"no basis convention is declared for {shape!r} over {window!r}")


@dataclass(frozen=True, kw_only=True)
class BasisElection:
    """The basis one part is counted on: the declared `convention` for its shape and window,
    and `basis`, the convention's servable basis or the one `entry`, an explicit or carried
    entry, asks for."""

    convention: BasisConvention
    basis: MeasureBasis
    entry: Entry | None = None

    @property
    def measure(self) -> BasisMeasure:
        """What the elected basis binds."""
        return _MEASURES[self.basis]

    def by_convention(self) -> BasisElection:
        """The basis the convention alone elects, for a part no shape of whose menu
        measures the basis its entry asks for."""
        return BasisElection(convention=self.convention, basis=_served(self.convention))

    def measured_by(
        self,
        population_type: str | None,
        measured_property: str | None,
        netting: Collection[str] | None,
    ) -> bool:
        """Whether a shape of `population_type` measuring `measured_property`, whose
        netting-basis slot offers `netting` (None for a shape with no such slot), measures
        the elected basis."""
        measure = self.measure
        if (population_type, measured_property) != (
            measure.population_type,
            measure.measured_property,
        ):
            return False
        return measure.netting_basis is None or measure.netting_basis in (netting or ())


def elect_basis(
    *, shape: PartShape, window: WindowClass, entries: Sequence[Entry]
) -> BasisElection:
    """The basis a part of `shape` asking for `window` is counted on. The first of `entries`
    (`ScopeState.entries_on`, explicit before carried) asking for a basis one identity can
    hold wins; otherwise the convention's basis, with the per-year headline counted as
    `HEADLINE_SERVED_AS`."""
    row = convention(shape, window)
    for entry in entries:
        measure = next(
            (m for m in BASIS_MEASURES if measure_basis_option_id(m.basis) == entry.option_id),
            None,
        )
        if measure is not None:
            return BasisElection(convention=row, basis=measure.basis, entry=entry)
    return BasisElection(convention=row, basis=_served(row))


def _served(row: BasisConvention) -> MeasureBasis:
    """The basis one identity counts `row`'s basis on: the basis itself, or
    `HEADLINE_SERVED_AS` for the per-year headline."""
    return HEADLINE_SERVED_AS if row.basis == "oecd_headline" else row.basis


def measure_of(
    population_type: str | None, measured_property: str | None, netting_basis: str | None
) -> BasisMeasure | None:
    """The declared basis a shape of `population_type` measuring `measured_property` with
    `netting_basis` bound (None when it binds none) counts on, None when it is none."""
    return next(
        (
            row
            for row in BASIS_MEASURES
            if (row.population_type, row.measured_property, row.netting_basis)
            == (population_type, measured_property, netting_basis)
        ),
        None,
    )


def names_a_basis(population_type: str | None, measured_property: str | None) -> bool:
    """Whether a declared basis measures `measured_property` of `population_type`."""
    return any(
        (row.population_type, row.measured_property) == (population_type, measured_property)
        for row in BASIS_MEASURES
    )


def held_measure(identity: BaseIdentity) -> BasisMeasure | None:
    """The declared basis `identity` is counted on, None when it holds none: a pinned
    shape, a measure no basis names, or an amount whose netting basis it leaves open."""
    shape = identity.shape
    if not isinstance(shape, IdentityShapeSchema):
        return None
    netting = next(
        (
            slot.binding.ref
            for slot in identity.slots
            if slot.property == _NETTING and isinstance(slot.binding, IdentityBindingValue)
        ),
        None,
    )
    return measure_of(shape.population_type, shape.measured_property, netting)


def identity_part_shape(identity: BaseIdentity, index: CodeIndex) -> PartShape | None:
    """The shape the conventions read of `identity`: a sector or restriction part when it
    binds a slot of a sector system (`sector_systems`) or a restriction, a recipient part
    when it binds a recipient, a donor part when it binds only donors, and None when it
    binds none."""
    bound = {
        slot.property
        for slot in identity.slots
        if isinstance(slot.binding, IdentityBindingValue | IdentityBindingSet)
    }
    if bound & (sector_systems(index) | frozenset(BASIS_RESTRICTIONS)):
        return "sector_or_restriction"
    if identity.entities.get("destination"):
        return "recipient_all_sectors"
    return "donor_only" if identity.entities.get("origin") else None


def basis_dataset(identity: BaseIdentity, index: CodeIndex) -> str | None:
    """The convention dataset that serves `identity` when the measure-basis axis applies to
    it, else None: `identity` holds a declared basis (`held_measure`), the conventions read
    its shape (`identity_part_shape`), and its family is published by a dataset the
    conventions read (`CONVENTION_DATASETS`)."""
    if held_measure(identity) is None or identity_part_shape(identity, index) is None:
        return None
    family = query_family(
        index, identity_family_query(index, identity.model_dump(mode="json", exclude_none=True))
    )
    if family is None:
        return None
    return next((dataset for dataset in family.datasets if dataset in CONVENTION_DATASETS), None)


def oda_flows(basis: BasisMeasure, held: Collection[str]) -> tuple[str, ...]:
    """The CRS types of flow an ODA flow binding reads as on `basis`, of those `held`, the
    codes a shape's flow slot offers."""
    return tuple(code for code in basis.oda_flows if code in held)


def basis_binding(row: BasisMeasure) -> ShapeBinding:
    """The binding of the option counting on `row`'s basis."""
    return ShapeBinding(measured_property=row.measured_property, netting_basis=row.netting_basis)


def basis_rule(
    base: BaseIdentity, record: AxisRecord | None, facts: ScopeFacts
) -> AxisTable | None:
    """The measure-basis table of a part whose identity is `base`, None where the axis does
    not apply to it (`basis_dataset`).

    Read from `base` alone, whether a binder recorded the axis (`record`) or not, so a
    resubmit of a served identity offers what its search offered. Each declared basis is an
    option. The one `base` holds is available. Another is available when its identity
    (`base` with the option's binding) reads a member of the dataset holding every other
    binding of `base` with a series in the coverage index, `option_not_covered` when the
    dataset publishes such a member with no series there, and `basis_not_published`
    otherwise."""
    dataset = basis_dataset(base, facts.index)
    held = held_measure(base)
    if dataset is None or held is None:
        return None
    bindings = [basis_binding(row) for row in BASIS_MEASURES]
    options = tuple(
        Option(
            option_id=measure_basis_option_id(row.basis),
            label=basis_label(row, base),
            binding=binding,
            availability=(
                OptionAvailability(status="available")
                if row is held
                else _availability(base, binding, bindings, dataset=dataset, facts=facts)
            ),
        )
        for row, binding in zip(BASIS_MEASURES, bindings, strict=True)
    )
    held_id = measure_basis_option_id(held.basis)
    return AxisTable(
        axis="measure_basis",
        rule=BASIS_RULE,
        options=options,
        preference=(
            held_id,
            *(option.option_id for option in options if option.option_id != held_id),
        ),
        held=held_id,
    )


def basis_label(row: BasisMeasure, identity: BaseIdentity) -> str:
    """The reader's name of `row`'s basis on `identity` (`OPTION_LABELS`): its words and the
    prices of the identity's price basis, "gross disbursements, current USD", or its words
    alone when the identity states no price basis."""
    shape = identity.shape
    qualifier = shape.measurement_qualifier if isinstance(shape, IdentityShapeSchema) else None
    prices = PRICES_BY_QUALIFIER.get(qualifier) if qualifier is not None else None
    if prices is None:
        return OPTION_LABELS.measure_basis_unpriced.format(words=row.words)
    return OPTION_LABELS.measure_basis.format(words=row.words, prices=prices)


def _availability(
    base: BaseIdentity,
    binding: ShapeBinding,
    bindings: Sequence[ShapeBinding],
    *,
    dataset: str,
    facts: ScopeFacts,
) -> OptionAvailability:
    """Whether `base` switched to `binding` reads a member of `dataset` with a series in
    the coverage index."""
    unpublished = OptionAvailability(
        status="unavailable", code="basis_not_published", detail={"dataset": dataset}
    )
    try:
        switched = apply_binding(base, binding, bindings)
    except ScopeTableError:
        return unpublished
    index = facts.index
    query = identity_family_query(index, switched.model_dump(mode="json", exclude_none=True))
    family = query_family(index, query)
    if family is None or dataset not in family.datasets:
        return unpublished
    allowed = _allowed(query.slots)
    rows = [
        row
        for row in family.rows
        if all(row.constraint_properties.get(p) in refs for p, refs in allowed.items())
    ]
    if not rows:
        return unpublished
    coverage = facts.coverage.index
    if not any(coverage.covers(row.dcid) for row in rows):
        return OptionAvailability(status="unavailable", code="option_not_covered")
    return OptionAvailability(status="available")


def _allowed(slots: Mapping[str, tuple]) -> dict[str, frozenset[str]]:
    """The values each value or set binding of `slots` allows, by property."""
    allowed: dict[str, frozenset[str]] = {}
    for prop, binding in slots.items():
        match binding:
            case ("value", ref):
                allowed[prop] = frozenset((ref,))
            case ("set", refs):
                allowed[prop] = frozenset(refs)
    return allowed
