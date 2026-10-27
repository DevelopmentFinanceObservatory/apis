"""The measure-basis axis: what a part's amounts are counted on (gross or net disbursements,
grant equivalents or commitments), and the bases its dataset could count them on instead.

The basis is a declared product default, after the OECD's own reporting
(`code_index.scope_conventions.MEASURE_BASIS_CONVENTIONS`): it is read from the part's shape
and the years it asks for. A donor's total is the OECD headline: grant equivalents for years
from 2018, net flows for years before it, and for a window on both sides of it the headline
per year, net flows to 2017 and grant equivalents from 2018, served by the headline
composite (`HEADLINE_BINDING`); a recipient's total over every sector is its net
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

import itertools
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from code_index.labels import PRICES_BY_QUALIFIER
from code_index.members import identity_family_query, query_family
from code_index.query import CodeIndex
from code_index.scope_conventions import (
    BASIS_MEASURES,
    BASIS_RESTRICTIONS,
    CONVENTION_CURRENCY,
    MEASURE_BASIS_CONVENTIONS,
    OPTION_LABELS,
    BasisConvention,
    BasisMeasure,
    MeasureBasis,
    PartShape,
    WindowClass,
)
from code_index.shared import per_index
from search.models import (
    BaseIdentity,
    Entry,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityCurrencyRole,
    IdentityFacetValueRequest,
    IdentityShapeSchema,
    IdentityWindowRange,
    Option,
    OptionAvailability,
    PartScope,
    ShapeBinding,
    TimeWindow,
)
from search.scope.composite import table_document
from search.scope.edits import HEADLINE_PROVENANCE, ScopeTableError, apply_binding
from search.scope.state import AxisRecord
from search.scope.tables import AxisTable, ScopeFacts, allowed_values
from search.scope.vocabulary import measure_basis_option_id

BASIS_RULE = "measure_basis_convention"
"""The declared rule the measure-basis axis applies."""

SERVED_DIFFERS_RULE = "served_differs_from_convention"
"""The rule of a part whose answer holds another basis than its convention's, which no
entry asked for: no shape of its pool measures the convention's basis."""

GRANT_EQUIVALENT_HEADLINE_FROM = 2018
"""The first year the OECD's headline counts official development assistance in grant
equivalents."""

HEADLINE_SERVED_AS: MeasureBasis = "net"
"""The basis whose shape a part counted on the OECD headline per year (`oecd_headline`)
elects. Its identity binds that shape and the headline composite as its source
(`HEADLINE_BINDING`), which serves net flows to 2017 and grant equivalents from 2018 as two
period components (`code_index.composites`)."""

HEADLINE_OPTION = measure_basis_option_id("oecd_headline")
"""The measure-basis option of the OECD headline per year."""

HEADLINE_BINDING = ShapeBinding(
    measured_property="amount", netting_basis="Net", provenance=HEADLINE_PROVENANCE
)
"""The binding of the OECD headline option: the net shape the part elects
(`HEADLINE_SERVED_AS`) and the headline composite as its source."""

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


@per_index
def sector_systems(index: CodeIndex) -> frozenset[str]:
    """The code systems `index` declares sector questions of (`CodeIndex.house_case_systems`):
    an admitted code of one makes a part a sector part. Built once per index."""
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


def elect_part_basis(
    *,
    datasets: Collection[str],
    directions: Iterable[str],
    admitted_systems: Collection[str],
    sector_systems: Collection[str],
    window: TimeWindow | Literal["all", "latest"],
    entries: Sequence[Entry],
) -> BasisElection | None:
    """The basis a part is counted on, from the shape its places (`directions`) and
    admitted codes give it (`part_shape`) and its `window` (`elect_basis`), with `entries`
    winning over the convention. None when no dataset of `datasets`, those the part could
    be answered from, is one the conventions read (`CONVENTION_DATASETS`), and for a part
    the conventions read no shape of."""
    if CONVENTION_DATASETS.isdisjoint(datasets):
        return None
    shape = part_shape(
        directions=directions, admitted_systems=admitted_systems, sector_systems=sector_systems
    )
    if shape is None:
        return None
    return elect_basis(shape=shape, window=window_class(window), entries=entries)


def _served(row: BasisConvention) -> MeasureBasis:
    """The basis one identity counts `row`'s basis on: the basis itself, or
    `HEADLINE_SERVED_AS` for the per-year headline."""
    return HEADLINE_SERVED_AS if row.basis == "oecd_headline" else row.basis


def convention_measures(dataset: str) -> tuple[BasisMeasure, ...]:
    """The bases the conventions reading `dataset` count a part on, each once, in table
    order: the DAC1 headline's grant equivalents and net flows (the per-year headline is
    served as `HEADLINE_SERVED_AS`), DAC2a's net receipts, the CRS's gross disbursements.
    Empty for a dataset no convention reads."""
    return tuple(
        dict.fromkeys(
            _MEASURES[_served(row)] for row in MEASURE_BASIS_CONVENTIONS if row.dataset == dataset
        )
    )


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


def held_option(identity: BaseIdentity) -> str | None:
    """The measure-basis option `identity` holds: the OECD headline when its source binds the
    headline composite, else its declared basis (`held_measure`), None when it holds none."""
    if _source(identity) == HEADLINE_PROVENANCE:
        return HEADLINE_OPTION
    measure = held_measure(identity)
    return measure_basis_option_id(measure.basis) if measure is not None else None


def _source(identity: BaseIdentity) -> str | None:
    """The provenance `identity`'s source binds as a value, None for none or a set."""
    binding = (identity.facets or {}).get("provenance")
    return binding.ref if isinstance(binding, IdentityBindingValue) else None


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


def identity_basis_record(
    identity: BaseIdentity, index: CodeIndex, entries: Sequence[Entry]
) -> AxisRecord | None:
    """The measure-basis record a search of `identity` wrote when no binder runs (a
    dataset_resubmit, a same-shape merge), read from the identity as `_record_basis` in
    `phase_election` reads the search: `SERVED_DIFFERS_RULE` where `identity` holds a basis
    that is not the convention's for its shape (`identity_part_shape`) and window
    (`identity.requested_window`) and no entry of `entries` settles the axis. None
    otherwise, so `basis_rule` names `BASIS_RULE`."""
    held = held_measure(identity)
    shape = identity_part_shape(identity, index)
    if held is None or shape is None:
        return None
    active = measure_basis_option_id(held.basis)
    if any(entry.axis == "measure_basis" for entry in entries):
        return None
    window = identity.requested_window
    row = convention(
        shape,
        window_class(
            TimeWindow(start_year=window.start_year, end_year=window.end_year)
            if window.kind == "range"
            else window.kind
        ),
    )
    if _MEASURES[_served(row)] == held:
        return None
    return AxisRecord(
        axis="measure_basis", rule=SERVED_DIFFERS_RULE, active=active, provenance="default"
    )


def basis_dataset(identity: BaseIdentity, index: CodeIndex) -> str | None:
    """The convention dataset that serves `identity` when the measure-basis axis applies to
    it, else None: `identity` holds a declared basis (`held_measure`), the conventions read
    its shape (`identity_part_shape`), and its family is published by a dataset the
    conventions read (`CONVENTION_DATASETS`)."""
    if held_measure(identity) is None or identity_part_shape(identity, index) is None:
        return None
    document = table_document(identity, index)
    if document is None:
        return None
    family = query_family(index, identity_family_query(index, document))
    if family is None:
        return None
    return next((dataset for dataset in family.datasets if dataset in CONVENTION_DATASETS), None)


def offers_basis_of(scope: PartScope, identity: BaseIdentity) -> bool:
    """Whether `scope`'s measure-basis choice offers the basis `identity` holds
    (`held_measure`) as an option other than its active one: a measure the choice switches
    to, which no other list of the part need offer."""
    held = held_measure(identity)
    if held is None:
        return False
    option = measure_basis_option_id(held.basis)
    return any(
        choice.axis == "measure_basis"
        and choice.active != option
        and any(o.option_id == option for o in choice.options)
        for choice in scope.choices
    )


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
    resubmit of a served identity offers what its search offered. The rule is the record's
    when it holds the option `base` holds (`SERVED_DIFFERS_RULE` for a basis the convention
    did not settle, written by the binder or, with none, `identity_basis_record`), else
    `BASIS_RULE`. Each declared basis is an option, and so is the OECD headline per year where
    `_offers_headline` holds. The one `base` holds (`held_option`) is available. Another is
    available when its identity (`base` with the option's binding) reads a member of the
    dataset holding every other binding of `base` with a series in the coverage index,
    `option_not_covered` when the dataset publishes such a member with no series there, and
    `basis_not_published` otherwise. The headline is available when both bases it splices
    are (`_headline_availability`). A composite base is read through the identity its option was
    set on (`table_document`)."""
    dataset = basis_dataset(base, facts.index)
    held_id = held_option(base)
    if dataset is None or held_id is None:
        return None
    headline = _offers_headline(base, dataset)
    measures = [basis_binding(row) for row in BASIS_MEASURES]
    bindings = [*measures, *((HEADLINE_BINDING,) if headline else ())]
    options = [
        Option(
            option_id=(option_id := measure_basis_option_id(row.basis)),
            label=basis_label(row, base),
            binding=binding,
            availability=(
                OptionAvailability(status="available")
                if option_id == held_id
                else _availability(base, binding, bindings, dataset=dataset, facts=facts)
            ),
        )
        for row, binding in zip(BASIS_MEASURES, measures, strict=True)
    ]
    if headline:
        options.append(
            Option(
                option_id=HEADLINE_OPTION,
                label=OPTION_LABELS.oecd_headline,
                binding=HEADLINE_BINDING,
                availability=(
                    OptionAvailability(status="available")
                    if held_id == HEADLINE_OPTION
                    else _headline_availability(base, bindings, dataset=dataset, facts=facts)
                ),
            )
        )
    return AxisTable(
        axis="measure_basis",
        rule=BASIS_RULE if record is None or record.active != held_id else record.rule,
        options=tuple(options),
        preference=(
            held_id,
            *(option.option_id for option in options if option.option_id != held_id),
        ),
        held=held_id,
    )


def _offers_headline(identity: BaseIdentity, dataset: str) -> bool:
    """Whether the OECD headline per year is an option of a part served from `dataset` whose
    identity is `identity`. It is when a convention counts that dataset's parts on it and the
    identity's window crosses 2018 (`window_class`), where the headline changes basis."""
    reads_headline = any(
        row.basis == "oecd_headline" and row.dataset == dataset for row in MEASURE_BASIS_CONVENTIONS
    )
    return reads_headline and window_class(identity_window(identity)) == "crossing_2018"


def identity_window(identity: BaseIdentity) -> TimeWindow | Literal["all", "latest"]:
    """`identity`'s requested window in the form `window_class` reads."""
    window = identity.requested_window
    if isinstance(window, IdentityWindowRange):
        return TimeWindow(start_year=window.start_year, end_year=window.end_year)
    return window.kind


def _headline_availability(
    base: BaseIdentity,
    bindings: Sequence[ShapeBinding],
    *,
    dataset: str,
    facts: ScopeFacts,
) -> OptionAvailability:
    """Whether `dataset` publishes, for `base`, both bases the headline splices: net flows
    and grant equivalents (`_availability`). Unavailable with the first missing basis's
    reason otherwise, naming it."""
    for basis in ("net", "grant_equivalent"):
        found = _availability(
            base, basis_binding(_MEASURES[basis]), bindings, dataset=dataset, facts=facts
        )
        if found.status != "available":
            return found.model_copy(update={"detail": {**found.detail, "basis": basis}})
    return OptionAvailability(status="available")


def basis_label(row: BasisMeasure, identity: BaseIdentity) -> str:
    """The reader's name of `row`'s basis on `identity` (`OPTION_LABELS`): its words and the
    identity's prices and currency, "gross disbursements, current USD". A unit request
    (`identity.facets["unit"]`) states both, so a part converted to constant prices reads
    "net disbursements, constant 2024 USD" and one converted to a role's currency "net
    disbursements, current donor currency". Without one, the prices are the shape's price
    basis (`PRICES_BY_QUALIFIER`) in the conventions' currency (`CONVENTION_CURRENCY`), and
    the words stand alone when the shape states no price basis."""
    unit = (identity.facets or {}).get("unit")
    if isinstance(unit, IdentityFacetValueRequest):
        request = unit.request
        prices = "current" if request.prices == "current" else f"constant {request.base_year}"
        currency = (
            _ROLE_CURRENCIES[request.currency.role]
            if isinstance(request.currency, IdentityCurrencyRole)
            else request.currency
        )
        return OPTION_LABELS.measure_basis.format(words=row.words, prices=prices, currency=currency)
    shape = identity.shape
    qualifier = shape.measurement_qualifier if isinstance(shape, IdentityShapeSchema) else None
    qualified = PRICES_BY_QUALIFIER.get(qualifier) if qualifier is not None else None
    if qualified is None:
        return OPTION_LABELS.measure_basis_unpriced.format(words=row.words)
    return OPTION_LABELS.measure_basis.format(
        words=row.words, prices=qualified, currency=CONVENTION_CURRENCY
    )


_ROLE_CURRENCIES: Mapping[str, str] = {
    "origin": "donor currency",
    "destination": "recipient currency",
    "observationAbout": "national currency",
}
"""The reader's name of the currency a unit request reads from a role's entity."""


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
    document = table_document(switched, index)
    if document is None:
        return unpublished
    query = identity_family_query(index, document)
    family = query_family(index, query)
    if family is None or dataset not in family.datasets:
        return unpublished
    allowed = allowed_values(query.slots)
    properties = tuple(sorted(allowed))
    values = facts.family_values(family, properties)
    wanted = set(itertools.product(*(allowed[p] for p in properties)))
    if wanted.isdisjoint(values.published):
        return unpublished
    if wanted.isdisjoint(values.covered):
        return OptionAvailability(status="unavailable", code="option_not_covered")
    return OptionAvailability(status="available")
