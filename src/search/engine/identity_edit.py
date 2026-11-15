"""Validate a resubmitted identity.

A posted identity passes the identity core before any door reads it, and Search doors
refuse the `exact` binding, the pinned-StatVar arm and an identity binding more places than
a Spec holds. A composite provenance is read as the identity its option was set on
(`composite_table_identity`). The resubmit and options doors share these checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from dfo_spec_kit import identity as _identity

from code_index.composites import (
    DECLARED_COMPOSITES,
    CompositeNotServable,
    PopulationUnavailable,
    UnclassifiedDestinations,
    composite_ref,
    destination_population,
    population_check,
    table_identity,
)
from code_index.query import CodeIndex
from code_index.scope_conventions import RolePopulation
from code_index.shared import get_index
from dfo_platform import Alternative, Cause
from dfo_platform.named import cut
from dfo_platform.roles import role_key_axis
from search.engine.config import settings
from search.engine.errors import EngineInputError
from search.models import (
    BaseIdentity,
    IdentityBindingExact,
    IdentityShape,
    IdentityShapePin,
    RoleAxis,
    ScopeReasonCode,
)
from series_coverage.shared import CoverageSnapshot, get_coverage_service

_MULTILATERAL_DESTINATION: ScopeReasonCode = "multilateral_destination"
"""The cause a composite posted with a destination off its population's side refuses
under (`composite_table_identity`)."""
_PROVIDER_NOT_OFFERED: ScopeReasonCode = "provider_not_offered"
"""The cause a composite whose population is unavailable refuses under
(`composite_table_identity`), as its scope option reads (`scope.provider._bilateral_option`,
why `destination_unclassified`)."""


def check_no_exact_bindings(identity: BaseIdentity, *, pointer: str) -> None:
    """Refuse the first `exact` slot binding at its JSON pointer.

    Search readers do not handle this identity-core binding; only dataset
    readers do. Run this before core validation, which accepts `exact`.
    """
    for i, slot in enumerate(identity.slots):
        if isinstance(slot.binding, IdentityBindingExact):
            raise EngineInputError(
                f"slot {cut(slot.property)!r} carries an 'exact' binding; no Search "
                "door serves it -- only the datasets doors read it",
                code="exact_binding_unsupported",
                status=501,
                pointer=f"{pointer}/slots/{i}/binding",
            )


def validate_edited_identity(identity: BaseIdentity, *, pointer: str = "/input/identity") -> None:
    """Validate a posted identity through the identity core.

    `BaseIdentity` mirrors most of the core schema but deliberately does not
    duplicate each cross-member invariant -- duplicate slot properties
    (collapsed silently by a naive dict build rather than refused) chief
    among them. The posted identity must therefore pass the authoritative
    core before any door reads its shape, slots, entities, window, or
    facets. `pointer` defaults to the resubmit path's own nested identity field;
    a caller whose posted identity lives under a different wire shape (e.g.
    options/resolve.py's `OptionsRequest.identity`) passes its own pointer,
    the same convention `check_described_arm` already follows.

    Refuses `exact_binding_unsupported` (501) first, ahead of the core
    check, for any slot bound `exact` (`check_no_exact_bindings`).
    """
    check_no_exact_bindings(identity, pointer=pointer)
    try:
        _identity.validate_identity_core(identity.model_dump(exclude_none=True))
    except ValueError as exc:
        raise EngineInputError(
            f"identity is not a well-formed identity document: {cut(str(exc))}",
            code="request_invalid",
            pointer=pointer,
        ) from exc


def open_requested_role(identity: BaseIdentity, *, role: str) -> BaseIdentity:
    """Return `identity` with the flow axis `role` names in `open_roles`.

    An options request for a flow axis asks for that axis member by member, so the
    population it reads is the one an edit binding that axis elects, such as the by-creditor
    twin of an IDS total. `identity` comes back unchanged when `role` names no flow axis
    (`observationAbout`) or when an `entities` key or an open role already names the axis.
    Keys match by axis (`dfo_platform.roles.role_key_axis`), since they keep the caller's
    spelling. A population whose StatVars declare no role on the axis drops the open role
    at election, so the door's role-target check still refuses it.
    """
    axis = role_key_axis(role)
    if axis is None:
        return identity
    named = {role_key_axis(key) for key in identity.entities} | set(identity.open_roles or ())
    if axis in named:
        return identity
    return BaseIdentity.model_validate(
        identity.model_dump(exclude_none=True)
        | {"open_roles": [*(identity.open_roles or ()), axis]}
    )


def population_members(
    populations: Mapping[str, RolePopulation],
) -> dict[RoleAxis, frozenset[str]]:
    """The members each role `populations` binds is open within, by the flow axis its key
    names (`RolePopulation.members`)."""
    return {
        axis: frozenset(population.members)
        for key, population in populations.items()
        if (axis := role_key_axis(key)) is not None
    }


def open_within(
    identity: BaseIdentity, populations: Mapping[str, RolePopulation]
) -> tuple[BaseIdentity, dict[RoleAxis, frozenset[str]]]:
    """`identity` with each `entities` key `populations` lists moved to `open_roles`, and the
    members each such role is open within, by flow axis: a role bound to a population
    (`TableRead.populations`) read as the open role the caller posted, over the parties an
    open read of the composite reads. `identity` itself, and no members, when `populations`
    lists none of its keys."""
    keys = [key for key in identity.entities if key in populations]
    if not keys:
        return identity, {}
    within = population_members({key: populations[key] for key in keys})
    document = identity.model_dump(exclude_none=True)
    document["entities"] = {
        key: dcids for key, dcids in identity.entities.items() if key not in populations
    }
    document["open_roles"] = sorted({*(identity.open_roles or ()), *within})
    return BaseIdentity.model_validate(document), within


def refuse_over_place_bound(
    entities: Mapping[str, Sequence[str]], *, pointer: str = "/input/identity/entities"
) -> None:
    """Refuse with `request_invalid` an identity that binds more places than a Spec holds
    (`SearchSettings.max_child_places`). Places count as distinct dcids per role, summed
    over the roles.

    `pointer` is the door's own path to the identity's entity map. The refusal names the
    first role that exceeds the bound alone, and the whole entity map when only the sum
    does."""
    bound = settings().max_child_places
    counts = {role: len(set(dcids)) for role, dcids in entities.items()}
    for role, count in counts.items():
        if count > bound:
            raise EngineInputError(
                f"{count} places on {role} exceed the {bound}-place limit a dataset binds; "
                "re-post with fewer places",
                code="request_invalid",
                pointer=f"{pointer}/{role}",
            )
    if (total := sum(counts.values())) > bound:
        raise EngineInputError(
            f"{total} places across {len(counts)} roles exceed the {bound}-place limit a "
            "dataset binds; re-post with fewer places",
            code="request_invalid",
            pointer=pointer,
        )


@dataclass(frozen=True)
class TableRead:
    """The identity a Search door reads for a posted identity (`composite_table_identity`).

    `populations` maps each `entities` key of `identity` that binds a population in place of
    names the caller posted (`code_index.composites.TableIdentity.populations`): a reader of
    the caller's names reads such a role as open within the population's members. `sides`
    holds the population of each flow axis the composite reads on one side alone
    (`ComponentRecipe.destinations`), whether the posted identity names that axis, opens it
    or leaves it out, so an options read of the axis offers that side's destinations alone.
    Both are empty for an identity that binds no such composite."""

    identity: BaseIdentity
    populations: Mapping[str, RolePopulation] = field(default_factory=dict)
    sides: Mapping[RoleAxis, RolePopulation] = field(default_factory=dict)


def composite_table_identity(
    identity: BaseIdentity, *, pointer: str, coverage: CoverageSnapshot | None
) -> TableRead:
    """The identity a Search door reads for `identity`. For a composite
    (`code_index.composites`) that is the identity its option was set on
    (`code_index.composites.table_identity`), whose one table every graph and index read
    names, with the populations its roles bind, and otherwise `identity` itself.

    Refuses a composite that reads a population of destinations that is unavailable, in the
    code index (`code_index.composites.PopulationUnavailable`) or in `coverage`, the
    request's coverage snapshot, or the release's where the door holds none
    (`code_index.composites.population_check`), with 422 `requested_option_unavailable`,
    cause `provider_not_offered`, the destinations no side declaration places, at `pointer`
    joined with the provenance facet; the alternative names the dataset whose total ODA reads
    every destination (`_population_unavailable`), as a fresh search reads it then. Refuses a
    composite that reads one side of its destinations and is posted naming a destination off
    that side, an agency or the total spanning both sides, with 422
    `requested_option_unavailable`, cause `multilateral_destination`, at `pointer`, the posted
    identity, joined with the role at fault; the alternative names the dataset whose total
    ODA reads the destination (`_multilateral_destination`). Refuses any other composite whose
    components cannot be derived (`CompositeNotServable`) as `request_invalid`, at `pointer`
    joined with the member at fault."""
    document = identity.model_dump(mode="json", exclude_none=True)
    index = get_index()
    try:
        table = table_identity(index, document)
    except PopulationUnavailable as exc:
        raise _population_unavailable(index, exc.ref, exc.unclassified, pointer=pointer) from exc
    except CompositeNotServable as exc:
        refused = _multilateral_destination(index, document, exc, pointer=pointer)
        raise (refused or _request_invalid(exc, pointer=pointer)) from exc
    if table.identity is document:
        return TableRead(identity=identity)
    ref = cast(str, composite_ref(document))
    if any(recipe.destinations is not None for recipe in DECLARED_COMPOSITES[ref].components):
        held = (coverage or get_coverage_service().snapshot()).index
        if (unclassified := population_check(index, held).refusal(ref)) is not None:
            raise _population_unavailable(index, ref, unclassified, pointer=pointer)
    return TableRead(
        identity=BaseIdentity.model_validate(table.identity),
        populations=table.populations,
        sides=_sides(index, document),
    )


def population_refusal(exc: PopulationUnavailable, *, pointer: str) -> EngineInputError:
    """The refusal of a posted composite whose population is unavailable in the code index
    (`exc`), at `pointer` joined with the provenance facet, as `composite_table_identity`
    refuses it, for a reader of the composite's table that runs before or without that
    check."""
    return _population_unavailable(get_index(), exc.ref, exc.unclassified, pointer=pointer)


def _population_unavailable(
    index: CodeIndex, ref: str, unclassified: UnclassifiedDestinations, *, pointer: str
) -> EngineInputError:
    """The refusal of composite `ref`, whose population of `unclassified.dataset`'s
    destinations is unavailable: the side declarations have drifted from the graph, so the
    composite is not offered (`provider_not_offered`, why `destination_unclassified`), and
    the dataset's own total ODA, which reads every destination, is the alternative."""
    dataset = unclassified.dataset
    label = index.datasets[dataset].label if dataset in index.datasets else dataset
    destinations = list(unclassified.destinations)
    return EngineInputError(
        f"composite {ref!r} is not offered (provider_not_offered, why "
        f"destination_unclassified): {unclassified.message()}; total ODA to every "
        f"destination reads from {label}: post this identity with facets.provenance bound to "
        f"{dataset!r} in place of {ref!r}",
        code="requested_option_unavailable",
        status=422,
        pointer=f"{pointer}/facets/provenance",
        alternatives=[Alternative(ref=dataset, label=label)],
        cause=Cause(
            code=_PROVIDER_NOT_OFFERED,
            entities=destinations[:10],
            truncated=len(destinations) > 10,
        ),
    )


def _request_invalid(exc: CompositeNotServable, *, pointer: str) -> EngineInputError:
    return EngineInputError(str(exc), code="request_invalid", pointer=f"{pointer}{exc.pointer}")


def _sides(index: CodeIndex, document: Mapping[str, object]) -> dict[RoleAxis, RolePopulation]:
    """The population of each flow axis the composite `document` binds reads on one side
    alone: its first component's destinations (`ComponentRecipe.destinations`), the
    component `table_identity` reads. Empty for a document binding no such composite."""
    ref = composite_ref(document)
    if ref is None:
        return {}
    first = DECLARED_COMPOSITES[ref].components[0]
    if first.destinations is None:
        return {}
    return {"destination": destination_population(index, first.dataset, composite=ref)}


def _multilateral_destination(
    index: CodeIndex, document: Mapping[str, object], exc: CompositeNotServable, *, pointer: str
) -> EngineInputError | None:
    """The refusal of composite `document` naming destinations off the side its population
    reads (`RolePopulation.named`): agencies, or the total spanning both sides. Its message
    and alternative name the dataset whose total ODA reads them, the population's own
    (`BilateralSource.dataset`), which the identity can bind in place of the composite.
    `None` when `exc` refuses anything else."""
    key = exc.key
    if key is None:
        return None
    population = _sides(index, document).get("destination")
    if population is None:
        return None
    dataset = population.source.dataset
    refused = list(exc.refs)
    label = index.datasets[dataset].label if dataset in index.datasets else dataset
    return EngineInputError(
        f"composite {exc.ref!r} reads {population.side} destinations alone, and "
        f"{', '.join(refused)} {'is' if len(refused) == 1 else 'are'} not on that side; total ODA "
        f"to {'it' if len(refused) == 1 else 'them'} reads from {label}: post this identity "
        f"with facets.provenance bound to {dataset!r} in place of {exc.ref!r}",
        code="requested_option_unavailable",
        status=422,
        pointer=f"{pointer}/entities/{key}",
        alternatives=[Alternative(ref=dataset, label=label)],
        cause=Cause(
            code=_MULTILATERAL_DESTINATION,
            entities=refused[:10],
            truncated=len(refused) > 10,
        ),
    )


def check_not_composite(identity: BaseIdentity, *, pointer: str) -> None:
    """Refuse an identity whose `facets.provenance` names a declared composite
    (`code_index.composites`), at `pointer`, the posted identity's provenance facet.

    No table publishes a composite, so its pin scopes every read to nothing. A Search door
    resolves the identity its option was set on (`composite_table_identity`) in its place,
    and this guards the resolve that reads it."""
    try:
        ref = composite_ref(identity.model_dump(mode="json", include={"facets"}))
    except CompositeNotServable as exc:
        raise EngineInputError(str(exc), code="request_invalid", pointer=pointer) from exc
    if ref is not None:
        raise EngineInputError(
            f"provenance {ref!r} is a composite of other tables' series; post each of its "
            "component identities instead",
            code="request_invalid",
            pointer=pointer,
        )


def check_described_arm(shape: IdentityShape, *, pointer: str = "/input/shape") -> None:
    """Refuse a pinned-ask shape (`kind="stat_var"`) posted to a Search door.

    Search doors (a dataset_resubmit identity entry, options) resolve described
    asks only -- the pin arm is Datasets' statvar_set identity, never a Search
    input. `pointer` defaults to the resubmit path's identity shape
    (`ResubmitIdentityPayload.identity.shape`); a caller whose posted value lives under a
    different wire shape (e.g. options/resolve.py's `OptionsRequest.identity.shape`)
    passes its own pointer so the refusal lands where the client can act on it.
    """
    if isinstance(shape, IdentityShapePin):
        raise EngineInputError(
            "a pinned StatVar identity (kind='stat_var') was posted to a Search "
            "door; Search doors resolve described asks only",
            code="request_invalid",
            pointer=pointer,
        )
