"""Derives a composite identity's component identities.

An identity is a composite when its `facets.provenance` binds a declared composite provenance
(`code_index.declarations.COMPOSITES`) as a value. No reader takes such an identity's
provenance pin as a dataset: `component_identities` is the one derivation of what each
component reads, and Search, its resubmit and the Datasets mint all call it before any read.
`code_index.members.pinned_refs` refuses a composite ref (`CompositeProvenanceRead`), so a
reader that skips the derivation fails rather than reading the node nobody publishes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from code_index.declarations import COMPOSITES
from code_index.measure_basis import edit_basis, with_slot
from code_index.scope_conventions import (
    BASIS_MEASURES,
    BILATERAL_SOURCES,
    ODA_FLOW,
    ODA_FLOW_SYSTEM,
    RolePopulation,
    destination_side,
)
from code_index.shared import per_index
from code_index.types import (
    ComponentRecipe,
    ComponentRole,
    CompositeDeclaration,
    StatVarRow,
    YearPeriod,
)
from dfo_platform.roles import role_key_axis

if TYPE_CHECKING:
    from code_index.query import CodeIndex
    from dfo_platform.roles import FlowAccount, RoleAxis
    from series_coverage.index import CoverageIndex

_PROVENANCE = "provenance"
_PINNED_SHAPE = "stat_var"
_DESTINATION: RoleAxis = "destination"

DECLARED_COMPOSITES: Mapping[str, CompositeDeclaration] = {
    composite.provenance: composite for composite in COMPOSITES
}
"""Every declared composite by its provenance."""

STACKED_YEARS_RULE = 2
"""The version of the rule a stacked composite's served years follow
(`materialise.composite.stacked_years` and the held years `compose_components` reads from its
rows). `composite_digest` digests it for a stacked composite, so a head composed under another
version fails its fingerprint check and re-mints. A change to which years or rows a stacked
composite serves for unchanged inputs bumps it, since an import-keyed head otherwise serves the
old rows until `DATASETS_HEAD_IMPORTED_DATA_MAX_AGE_SECONDS`."""


def components_sum(composite: CompositeDeclaration) -> bool:
    """Whether a Datasets groupby whose `by` keeps no `stat_var` adds the members of
    `composite`'s different components into one series per cell: true for a stacked
    composite, whose components are parts of one total by declaration, and false for a
    spliced one. The mint sums by it (`datasets_api.groupby.sums_components`) and Search
    serves it (`ResolvedStatVars.sums_components`), so the two cannot disagree. A cell
    holding one component's line serves that line."""
    return composite.kind == "stacked"


def components_splice(composite: CompositeDeclaration) -> bool:
    """Whether a Datasets groupby whose `by` keeps no `stat_var` splices the members of
    `composite`'s different components into one series per cell: true for a spliced
    composite, whose components each serve their own years, so each cell takes the members of
    the one component holding it and no two components' values are added. The mint refuses a
    cell holding rows of two components (`datasets_api.groupby.SpliceCheck`). The mint
    splices by it (`datasets_api.groupby.splices_components`) and Search serves it
    (`ResolvedStatVars.splices_components`), so the two cannot disagree."""
    return composite.kind == "spliced"


def _jsonable(value: object) -> object:
    if isinstance(value, frozenset | set):
        return sorted(str(item) for item in value)
    raise TypeError(f"cannot digest {type(value).__name__}")


def composite_digest(composite: CompositeDeclaration, *, index: CodeIndex) -> str:
    """`sha256:` and the hex digest of what a run composes `composite` from, over sorted-key
    JSON of:

    - its declaration, every field of it and of each component's recipe
      (`ComponentRecipe`), with `destinations` left out of a recipe where it is None, so a
      composite whose components bind no destinations digests as its other fields alone;
    - for a stacked composite, the served-years rule version (`STACKED_YEARS_RULE`);
    - where any component sets a basis, the basis measures and ODA flow codes
      `measure_basis.edit_basis` reads;
    - where any component binds its destinations, for each distinct (dataset, side) bound,
      the dataset, the side, and the members and total of the population it reads in
      `index` (`destination_population`), the code index the caller reads the composite
      under, so a release that moves a party across sides moves the digest.

    Raises `PopulationUnavailable` as `destination_population` does."""
    declaration = dataclasses.asdict(composite)
    for component in cast("list[dict[str, object]]", declaration["components"]):
        if component["destinations"] is None:
            del component["destinations"]
    payload: dict[str, object] = {"composite": declaration}
    if composite.kind == "stacked":
        payload["stacked_years_rule"] = STACKED_YEARS_RULE
    if any(component.basis is not None for component in composite.components):
        payload["basis_measures"] = [dataclasses.asdict(row) for row in BASIS_MEASURES]
        payload["oda_flow"] = [ODA_FLOW_SYSTEM, ODA_FLOW]
    bound = sorted(
        {
            (recipe.dataset, recipe.destinations)
            for recipe in composite.components
            if recipe.destinations is not None
        }
    )
    if bound:
        populations = [
            destination_population(index, dataset, composite=composite.provenance)
            for dataset, _side in bound
        ]
        payload["destination_population"] = [
            [dataset, population.side, list(population.members), population.total]
            for (dataset, _side), population in zip(bound, populations, strict=True)
        ]
    encoded = json.dumps(payload, sort_keys=True, default=_jsonable).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def declared_composite_digest(provenance: str, *, index: CodeIndex) -> str | None:
    """`composite_digest` under `index` of the composite this deployment declares for
    `provenance` (`DECLARED_COMPOSITES`), or `None` where it declares none. Raises
    `PopulationUnavailable` as `composite_digest` does."""
    composite = DECLARED_COMPOSITES.get(provenance)
    return composite_digest(composite, index=index) if composite is not None else None


class CompositeNotServable(ValueError):
    """A composite identity whose components cannot be derived, because it binds a slot a
    component's rows do not declare, pins a StatVar, names a composite in a provenance set, or
    names one the index serves no component table of, or binds a component reading one side's
    destinations to destinations off that side. `ref` is the composite's provenance and
    `pointer` the identity member at fault, relative to the identity. `refs` holds the refs at
    fault in that member, in the identity's order, where the refusal names some: the
    destinations off the side for the last case, and empty otherwise. `key` is the `entities`
    key of the destination role those destinations bind, for the last case alone, and None
    otherwise."""

    def __init__(
        self,
        message: str,
        *,
        ref: str,
        pointer: str,
        refs: tuple[str, ...] = (),
        key: str | None = None,
    ) -> None:
        self.ref = ref
        self.pointer = pointer
        self.refs = refs
        self.key = key
        super().__init__(message)


UnclassifiedIn = Literal["code_index", "coverage"]
"""Where a destination `destination_side` cannot place was found, among the parties the code
index publishes in the dataset or among the recipients coverage holds for the population's
variables (`population_check`)."""


@dataclass(frozen=True)
class UnclassifiedDestinations:
    """Destinations of `dataset`, a dataset a composite component binds the destinations of
    (`ComponentRecipe.destinations`), which `destination_side` cannot place: its graph types or
    the declarations it reads have drifted from the graph. `found_in` says where they were
    found. While any is unclassified the population is unavailable, since a list of members
    would leave the drifted destination out without a word.

    `unlisted` names the population's variables whose recipients coverage cannot list: it
    holds rows of the variable, but no facet of it carries `dataset` as its provenance. A
    destination there could sit off every side unseen, so the population is unavailable too;
    `destinations` is then empty and `found_in` is `coverage`."""

    dataset: str
    destinations: tuple[str, ...]
    found_in: UnclassifiedIn
    unlisted: tuple[str, ...] = ()

    def message(self) -> str:
        if self.unlisted:
            return (
                f"coverage lists no recipients of {self.dataset} for the population variables "
                f"{list(self.unlisted)!r}"
            )
        where = "publishes" if self.found_in == "code_index" else "holds coverage of"
        return f"{self.dataset} {where} destinations on no side: {list(self.destinations)!r}"


class PopulationUnavailable(CompositeNotServable):
    """A composite, provenance `ref`, whose component reads a population of destinations
    (`destination_population`) that is unavailable, for the reason `unclassified` gives: the
    code index cannot place a destination on a side, a server fault. `pointer` is the
    composite's provenance facet, since no edit of the identity serves it. A handler of
    `CompositeNotServable` that reads every refusal as the caller's fault catches this first."""

    def __init__(self, unclassified: UnclassifiedDestinations, *, ref: str) -> None:
        self.unclassified = unclassified
        super().__init__(unclassified.message(), ref=ref, pointer=f"/facets/{_PROVENANCE}")


class CompositeProvenanceRead(ValueError):
    """A provenance pin read as a dataset when it binds a composite. The caller reads the
    composite's components (`component_identities`) instead."""


@dataclass(frozen=True)
class ComponentIdentity:
    """One component of a composite identity: the role it plays, the recipe it was derived
    by, and its own identity document, which reads the component's table alone.

    `populations` maps each `entities` key of `identity` whose binding is a population
    (`RolePopulation.members`) in place of the destinations the caller named, the key an
    open role took. It is empty for a component that binds no destinations, and for one
    whose identity named its destinations or left them out. A reader of rows reads
    `identity` alone; a reader of the names the caller asked for reads the key as open within
    its population."""

    role: ComponentRole
    recipe: ComponentRecipe
    identity: Mapping[str, object]
    populations: Mapping[str, RolePopulation] = field(default_factory=dict)


@dataclass(frozen=True)
class TableIdentity:
    """The identity a reader of one table's family reads (`table_identity`), and the
    populations its `entities` bind in place of names (`ComponentIdentity.populations`)."""

    identity: Mapping[str, object]
    populations: Mapping[str, RolePopulation] = field(default_factory=dict)


def composite_ref(identity: Mapping[str, object]) -> str | None:
    """The declared composite `identity`'s provenance facet binds, or `None` when it binds
    none.

    Raises `CompositeNotServable` for a composite named in a provenance set, which would read
    as an election among the set's tables."""
    facets = cast("Mapping[str, Mapping[str, object]]", identity.get("facets") or {})
    binding = facets.get(_PROVENANCE, {})
    ref = binding.get("ref")
    if isinstance(ref, str):
        return ref if ref in DECLARED_COMPOSITES else None
    refs = cast("Sequence[str]", binding.get("refs", ()))
    named = [ref for ref in refs if ref in DECLARED_COMPOSITES]
    if named:
        raise CompositeNotServable(
            f"provenance set {list(refs)!r} names the composite "
            f"{named[0]!r}, which is bound as a value alone",
            ref=named[0],
            pointer=f"/facets/{_PROVENANCE}",
        )
    return None


def binds_composite(identity: Mapping[str, object]) -> bool:
    """Whether `identity`'s provenance facet binds a declared composite, or names one in a
    provenance set, which `composite_ref` refuses and every reader treats as a composite:
    deriving its components refuses it there."""
    try:
        return composite_ref(identity) is not None
    except CompositeNotServable:
        return True


def provenance_datasets(ref: str) -> tuple[str, ...]:
    """The datasets a provenance `ref` serves from: a declared composite's component tables,
    each once in declaration order, else `ref` itself."""
    composite = DECLARED_COMPOSITES.get(ref)
    if composite is None:
        return (ref,)
    return tuple(dict.fromkeys(recipe.dataset for recipe in composite.components))


def provenance_account(index: CodeIndex, ref: str) -> FlowAccount | None:
    """The flow account the datasets of `ref` declare (`DatasetDeclaration.account`,
    `provenance_datasets`), or `None` where `index` declares none of them or they declare
    different accounts."""
    accounts = {
        index.datasets[dataset].account
        for dataset in provenance_datasets(ref)
        if dataset in index.datasets
    }
    return accounts.pop() if len(accounts) == 1 else None


def refuse_composite_refs(refs: Sequence[str]) -> None:
    """Raise `CompositeProvenanceRead` when `refs`, a provenance pin, names a declared
    composite."""
    named = [ref for ref in refs if ref in DECLARED_COMPOSITES]
    if named:
        raise CompositeProvenanceRead(
            f"provenance {named[0]!r} is a composite; read its components "
            "(code_index.composites.component_identities) instead of the pin"
        )


def component_identities(
    index: CodeIndex, identity: Mapping[str, object]
) -> tuple[ComponentIdentity, ...] | None:
    """The component identities of a composite `identity`, in declaration order, or `None`
    for an identity that binds no composite.

    Each component's identity is `identity` with the composite's provenance binding replaced
    by the recipe's own (`ComponentRecipe.provenance`), its measure edited as the recipe says,
    the recipe's slots bound and its undeclared slots removed, and its window clipped to the
    recipe's periods. A spliced component whose periods the window does not reach is left out.
    Slots stay sorted by property.

    A component that binds its destinations (`ComponentRecipe.destinations`) reads its
    dataset's population (`destination_population`): an open destination binds the
    population's members and is listed in `ComponentIdentity.populations`, a named one passes
    when the dataset publishes no ref of it off the population's side, and an identity that
    leaves the destination out binds the population's total.

    Raises `CompositeNotServable` when `identity` pins a StatVar, binds a slot a component
    drops (`ComponentRecipe.drop_unbound`), names a composite in a provenance set, names one
    whose component tables `index` does not serve (`CodeIndex.composite`), names a
    destination off a component's population (at `/entities/<key>`), or reaches none of its
    components' periods, and `PopulationUnavailable`, a `CompositeNotServable`, as
    `destination_population` does."""
    ref = composite_ref(identity)
    if ref is None:
        return None
    declaration = index.composite(ref)
    if declaration is None:
        raise CompositeNotServable(
            f"the code index serves no component table of composite {ref!r}",
            ref=ref,
            pointer=f"/facets/{_PROVENANCE}",
        )
    shape = cast("Mapping[str, object]", identity.get("shape", {}))
    if shape.get("kind") == _PINNED_SHAPE:
        raise CompositeNotServable(
            f"composite {ref!r} composes described identities; a pinned StatVar has no components",
            ref=ref,
            pointer="/shape",
        )
    window = cast("Mapping[str, object]", identity.get("requested_window", {"kind": "all"}))
    components: list[ComponentIdentity] = []
    for recipe in declaration.components:
        clipped = _clipped_window(window, recipe.periods)
        if clipped is None:
            continue
        derived, populations = _derived_identity(index, identity, recipe, ref=ref)
        components.append(
            ComponentIdentity(
                role=recipe.role,
                recipe=recipe,
                identity={**derived, "requested_window": clipped},
                populations=populations,
            )
        )
    if not components:
        raise CompositeNotServable(
            f"the requested window reaches none of composite {ref!r}'s component periods",
            ref=ref,
            pointer="/requested_window",
        )
    return tuple(components)


def table_identity(index: CodeIndex, identity: Mapping[str, object]) -> TableIdentity:
    """The identity a reader of one table's family reads for `identity`. For a composite that
    is the identity its composite option was set on (`without_composite`), once its components
    derive (`component_identities`), and otherwise `identity` itself, with no populations. The
    result is the bilateral component of a provider-inclusive composite, and the net identity
    over the whole window of the OECD headline, whose shape the part elected. Search reads a
    composite's family, pins, dataset and provenance scope here, so its readers elect over the
    same rows the composite's option edits.

    A composite whose first component binds its destinations (`ComponentRecipe.destinations`)
    binds them here as that component does, so Search elects within the population the mint
    reads, and `TableIdentity.populations` names the keys bound to a population. The read
    holds the first component's rows alone: a party only another component holds, such as a
    recipient of the imputed line with no total ODA row, is not in it, so a reader counting
    what the composite serves reads every component (`component_identities`).

    The first component is the first `component_identities` derives, whose `entities` and
    `open_roles` the read takes as they were bound there.

    Raises `CompositeNotServable` and `PopulationUnavailable` as `component_identities`
    does."""
    components = component_identities(index, identity)
    if components is None:
        return TableIdentity(identity=identity)
    read = without_composite(identity)
    first = components[0]
    if first.recipe.destinations is None:
        return TableIdentity(identity=read)
    roles = ("entities", "open_roles")
    bound = {key: value for key, value in read.items() if key not in roles}
    bound.update({key: first.identity[key] for key in roles if key in first.identity})
    return TableIdentity(identity=bound, populations=first.populations)


def component_of(composite: CompositeDeclaration, row: StatVarRow) -> ComponentRole | None:
    """The role of the one component of `composite` that serves `row`: its table publishes
    `row` (`StatVarRow.datasets`), its recipe sets no measure or the measure `row` has, and
    `row` holds every code the recipe binds (`ComponentRecipe.bind`). `None` when no component
    or more than one serves it, so a row two components could each have read is no member of
    either. DAC2a's inclusive composite reads both components from DAC2a, and only the aid type
    each binds tells its rows apart."""
    roles = [
        recipe.role
        for recipe in composite.components
        if recipe.dataset in row.datasets
        and _recipe_measure(recipe) in (None, row.measured_property)
        and all(row.constraint_properties.get(prop) == code for prop, code in recipe.bind.items())
    ]
    return roles[0] if len(roles) == 1 else None


def _recipe_measure(recipe: ComponentRecipe) -> str | None:
    """The measured property `recipe` sets, by its basis or alone, or `None`."""
    return recipe.basis[0] if recipe.basis is not None else recipe.measured_property


def without_composite(identity: Mapping[str, object]) -> Mapping[str, object]:
    """`identity` with a composite provenance binding removed: its own shape, slots, entities
    and window, read as an identity that pins no provenance. `identity` itself when it binds
    no composite."""
    if composite_ref(identity) is None:
        return identity
    facets = {
        axis: binding
        for axis, binding in cast("Mapping[str, object]", identity["facets"]).items()
        if axis != _PROVENANCE
    }
    stripped = {key: value for key, value in identity.items() if key != "facets"}
    return {**stripped, "facets": facets} if facets else stripped


def _derived_identity(
    index: CodeIndex, identity: Mapping[str, object], recipe: ComponentRecipe, *, ref: str
) -> tuple[dict[str, object], Mapping[str, RolePopulation]]:
    """`identity` derived by `recipe`, its window unchanged, and the `entities` keys it binds
    to a population (`_bind_destinations`)."""
    derived = dict(without_composite(identity))
    populations = _bind_destinations(index, derived, recipe=recipe, ref=ref)
    if recipe.provenance is not None:
        facets = dict(cast("Mapping[str, object]", derived.get("facets", {})))
        facets[_PROVENANCE] = {"state": "value", "ref": recipe.provenance}
        derived["facets"] = facets
    if recipe.basis is not None:
        measured_property, netting_basis = recipe.basis
        derived = edit_basis(
            derived, measured_property=measured_property, netting_basis=netting_basis
        )
    elif recipe.measured_property is not None:
        shape = cast("Mapping[str, object]", derived["shape"])
        derived["shape"] = {**shape, "measured_property": recipe.measured_property}
    slots = list(cast("Sequence[Mapping[str, object]]", derived.get("slots", ())))
    for position, slot in enumerate(slots):
        binding = cast("Mapping[str, object]", slot["binding"])
        if slot["property"] in recipe.drop_unbound and binding.get("state") != "unbound":
            raise CompositeNotServable(
                f"composite {ref!r} serves no {slot['property']!r} binding: its "
                f"{recipe.role} component's rows do not declare it",
                ref=ref,
                pointer=f"/slots/{position}",
            )
    slots = [slot for slot in slots if slot["property"] not in recipe.drop_unbound]
    for prop, code in sorted(recipe.bind.items()):
        slots = with_slot(slots, {"property": prop, "binding": {"state": "value", "ref": code}})
    derived["slots"] = sorted(slots, key=lambda slot: cast(str, slot["property"]))
    return derived, populations


def _bound_datasets() -> frozenset[str]:
    """The datasets a declared composite component binds the destinations of."""
    return frozenset(
        recipe.dataset
        for composite in COMPOSITES
        for recipe in composite.components
        if recipe.destinations is not None
    )


@dataclass(frozen=True)
class _Classified:
    """One dataset's destinations as `index` publishes them: its population, or the
    destinations `destination_side` cannot place, with no population."""

    population: RolePopulation | None
    unclassified: tuple[str, ...]


@per_index
def _classified(index: CodeIndex) -> Mapping[str, _Classified]:
    """`_Classified` for each dataset a declared component binds the destinations of."""
    sources = {source.dataset: source for source in BILATERAL_SOURCES}
    classified: dict[str, _Classified] = {}
    for dataset in sorted(_bound_datasets()):
        source = sources.get(dataset)
        if source is None:
            raise LookupError(
                f"a composite component binds the destinations of {dataset!r}, which no "
                "BILATERAL_SOURCES row declares"
            )
        members: list[str] = []
        named: set[str] = set()
        unclassified: list[str] = []
        for dcid, row in index.parties.items():
            if not row.publishes(_DESTINATION, (dataset,)):
                continue
            side = destination_side(row, datasets=(dataset,), index=index)
            if side is None:
                unclassified.append(dcid)
            elif side == "bilateral":
                named.add(dcid)
                if row.kind_over(_DESTINATION, (dataset,)) != "total":
                    members.append(dcid)
        classified[dataset] = (
            _Classified(population=None, unclassified=tuple(sorted(unclassified)))
            if unclassified
            else _Classified(
                population=RolePopulation(
                    side="bilateral",
                    members=tuple(sorted(members)),
                    named=frozenset(named),
                    source=source,
                ),
                unclassified=(),
            )
        )
    return classified


def destination_population(
    index: CodeIndex, dataset: str, *, composite: str | None = None
) -> RolePopulation:
    """The population of destinations a component reading `dataset` binds its destinations
    to (`ComponentRecipe.destinations`), from the parties `index` publishes there as
    destinations, built once per index.

    Raises `PopulationUnavailable` (`found_in="code_index"`) when `destination_side` places
    none of a destination `dataset` publishes, whatever its kind. It names `composite` as the
    composite refused, or by default the first declared composite whose component binds the
    destinations of `dataset`. Raises `KeyError` for a dataset no declared component binds
    the destinations of."""
    classified = _classified(index)[dataset]
    if classified.population is None:
        raise PopulationUnavailable(
            UnclassifiedDestinations(
                dataset=dataset, destinations=classified.unclassified, found_in="code_index"
            ),
            ref=composite if composite is not None else _first_binding(dataset),
        )
    return classified.population


def _first_binding(dataset: str) -> str:
    """The provenance of the first declared composite whose component binds the destinations
    of `dataset`."""
    return next(
        composite.provenance
        for composite in COMPOSITES
        if any(
            recipe.dataset == dataset and recipe.destinations is not None
            for recipe in composite.components
        )
    )


def _bind_destinations(
    index: CodeIndex, derived: dict[str, object], *, recipe: ComponentRecipe, ref: str
) -> Mapping[str, RolePopulation]:
    """Bind `derived`'s destination role, in place, as `recipe` reads it, and return the
    `entities` key bound to a population, empty where none is. A recipe that binds no
    destinations leaves `derived` as it is.

    An open destination binds the population's members (`destination_population`). A named
    one keeps its refs when the dataset publishes none of them as a destination off the
    population (`RolePopulation.named`), and refuses with `CompositeNotServable` otherwise; a
    ref the dataset does not publish as a destination passes, and reads no rows. An identity
    that neither names nor opens the destination binds the population's total
    (`RolePopulation.total`), since the dataset's total over every destination spans both
    sides."""
    if recipe.destinations is None:
        return {}
    population = destination_population(index, recipe.dataset, composite=ref)
    entities = dict(cast("Mapping[str, Sequence[str]]", derived.get("entities", {})))
    open_roles = list(cast("Sequence[str]", derived.get("open_roles", ())))
    bound_key = next((key for key in entities if role_key_axis(key) == _DESTINATION), None)
    open_key = next((key for key in open_roles if role_key_axis(key) == _DESTINATION), None)
    if bound_key is not None:
        refused = [
            dcid
            for dcid in entities[bound_key]
            if dcid not in population.named
            and (row := index.parties.get(dcid)) is not None
            and row.publishes(_DESTINATION, (recipe.dataset,))
        ]
        if refused:
            raise CompositeNotServable(
                f"composite {ref!r} reads {population.side} destinations alone, and "
                f"{refused!r} {'is' if len(refused) == 1 else 'are'} not on that side",
                ref=ref,
                pointer=f"/entities/{bound_key}",
                refs=tuple(refused),
                key=bound_key,
            )
        return {}
    populations: dict[str, RolePopulation] = {}
    if open_key is not None:
        open_roles.remove(open_key)
        entities[open_key] = list(population.members)
        populations[open_key] = population
    else:
        entities[_DESTINATION] = [population.total]
    derived["entities"] = dict(sorted(entities.items()))
    if open_roles:
        derived["open_roles"] = open_roles
    else:
        derived.pop("open_roles", None)
    return populations


@dataclass(frozen=True)
class PopulationCheck:
    """Whether the populations of one code index (`destination_population`) hold every
    destination one coverage index serves for their variables. `unclassified` maps each
    dataset whose population is unavailable to why: the destinations `destination_side`
    cannot place, found in the code index or, where it places every party it publishes, in
    coverage, or the variables whose recipients coverage cannot list (`population_check`).
    An empty map means every population is available."""

    unclassified: Mapping[str, UnclassifiedDestinations]

    def refusal(self, composite: str) -> UnclassifiedDestinations | None:
        """Why the composite with provenance `composite` cannot be read, for the first of its
        components whose population is unavailable in declaration order, or `None` where
        every population it reads is available or it binds none."""
        declaration = DECLARED_COMPOSITES.get(composite)
        if declaration is None:
            return None
        return next(
            (
                self.unclassified[recipe.dataset]
                for recipe in declaration.components
                if recipe.destinations is not None and recipe.dataset in self.unclassified
            ),
            None,
        )


class _Checks(dict[int, PopulationCheck]):
    """The population checks of one code index built so far, by the `id` of the coverage index
    each was built against (`population_check`). A subclass, since a plain `dict` cannot be
    weakly referenced."""


@per_index
def _checks(index: CodeIndex) -> _Checks:
    """The population checks of `index` built so far (`_Checks`)."""
    return _Checks()


def _forget(checks: weakref.ReferenceType[_Checks], key: int) -> None:
    """Drop the entry `key` from `checks`, unless its code index has gone with it."""
    if (held := checks()) is not None:
        held.pop(key, None)


def population_check(index: CodeIndex, coverage: CoverageIndex) -> PopulationCheck:
    """Check every population `index` declares (`destination_population`) against the
    recipients `coverage` holds for the population's variables: the rows of the dataset that
    hold every code a component binding the population binds (`ComponentRecipe.bind`). A
    recipient the code index does not publish there as a destination is one
    `destination_side` cannot place, and the population's members would leave it out without
    a word, so the population is unavailable until the code index places it. A population the code
    index itself cannot place (`destination_population`) is unavailable too, and so is one
    with a variable whose recipients coverage cannot list (`UnclassifiedDestinations
    .unlisted`): the check fails closed.

    Built once per pair of indexes: memoised per code index (`per_index`), then per coverage
    index within it by `id`. Each coverage index carries one finalizer per code index it was
    checked against, which drops its entry when the coverage index is collected; the code
    index's own memo goes with the code index, which the finalizer holds weakly. The release
    service builds it when a release opens (`graph_release.service.build_release_views`), so
    a request reads it from the memo. The mint and Search refuse a composite reading an
    unavailable population (`PopulationCheck.refusal`)."""
    checks = _checks(index)
    key = id(coverage)
    found = checks.get(key)
    if found is None:
        found = _population_check(index, coverage)
        checks[key] = found
        weakref.finalize(coverage, _forget, weakref.ref(checks), key)
    return found


def _population_check(index: CodeIndex, coverage: CoverageIndex) -> PopulationCheck:
    unclassified: dict[str, UnclassifiedDestinations] = {}
    for dataset, classified in _classified(index).items():
        if classified.population is None:
            unclassified[dataset] = UnclassifiedDestinations(
                dataset=dataset, destinations=classified.unclassified, found_in="code_index"
            )
            continue
        held: set[str] = set()
        unlisted: list[str] = []
        for variable in _population_variables(index, dataset):
            recipients = _covered_recipients(coverage, variable, dataset)
            if recipients is None:
                unlisted.append(variable)
            else:
                held |= recipients
        if unlisted:
            unclassified[dataset] = UnclassifiedDestinations(
                dataset=dataset, destinations=(), found_in="coverage", unlisted=tuple(unlisted)
            )
            continue
        drifted = sorted(held - _published_destinations(index, dataset))
        if drifted:
            unclassified[dataset] = UnclassifiedDestinations(
                dataset=dataset, destinations=tuple(drifted), found_in="coverage"
            )
    return PopulationCheck(unclassified=unclassified)


def _covered_recipients(
    coverage: CoverageIndex, variable: str, dataset: str
) -> frozenset[str] | None:
    """The recipients `coverage` holds for `variable` in the facets whose provenance is
    `dataset`, or None when it holds rows of `variable` but no facet of it names `dataset`:
    rows with no provenance, or another one, whose recipients coverage cannot attribute.

    A variable coverage holds no row of contributes no recipients. The check reads the
    snapshot, so it is only as fresh as the snapshot: where the source publishes rows of a
    variable the snapshot does not yet hold, a destination there goes unchecked until the
    next coverage build."""
    summary = coverage.facet_summary.variable(variable)
    if summary is None:
        return frozenset()
    facets = [facet for facet in summary.facets if facet.metadata.provenance_id == dataset]
    if not facets:
        return None
    return frozenset().union(*(facet.parties.get("recipient", frozenset()) for facet in facets))


def _published_destinations(index: CodeIndex, dataset: str) -> frozenset[str]:
    """The parties `index` publishes as destinations of `dataset`, each of which a population
    `index` builds places on a side."""
    return frozenset(
        dcid for dcid, row in index.parties.items() if row.publishes(_DESTINATION, (dataset,))
    )


def _population_variables(index: CodeIndex, dataset: str) -> tuple[str, ...]:
    """The StatVars of `dataset` a component binding its destinations reads: those holding
    every code such a component binds (`ComponentRecipe.bind`)."""
    binds = [
        recipe.bind
        for composite in COMPOSITES
        for recipe in composite.components
        if recipe.destinations is not None and recipe.dataset == dataset
    ]
    return tuple(
        row.dcid
        for row in index.statvars
        if dataset in row.datasets
        and any(
            all(row.constraint_properties.get(prop) == code for prop, code in bind.items())
            for bind in binds
        )
    )


def _clipped_window(
    window: Mapping[str, object], periods: YearPeriod | None
) -> Mapping[str, object] | None:
    """`window` clipped to `periods`, or `None` when the two share no year. A `latest`
    window passes unclipped, as does any window when `periods` is `None`. The clipped
    window keeps `window`'s `include`."""
    if periods is None or window.get("kind") == "latest":
        return window
    start = cast("int | None", window.get("start_year")) if window.get("kind") == "range" else None
    end = cast("int | None", window.get("end_year")) if window.get("kind") == "range" else None
    first = _later(start, periods.first)
    last = _earlier(end, periods.last)
    if first is not None and last is not None and first > last:
        return None
    clipped: dict[str, object] = {"kind": "range"}
    if first is None and last is None:
        clipped = {"kind": "all"}
    if first is not None:
        clipped["start_year"] = first
    if last is not None:
        clipped["end_year"] = last
    if "include" in window:
        clipped["include"] = window["include"]
    return clipped


def _later(a: int | None, b: int | None) -> int | None:
    return b if a is None else a if b is None else max(a, b)


def _earlier(a: int | None, b: int | None) -> int | None:
    return b if a is None else a if b is None else min(a, b)
