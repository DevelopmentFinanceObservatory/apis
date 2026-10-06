"""Resolve conversion recipes, validate convertibility, and map currencies and roles.

A recipe id comes from a Provenance's `conversionRecipe` arc or, when it declares none,
from the first of its Datasets that does, in `isPartOf` order (`elect_recipe`, which the
search engine's conversion catalogue also calls); `RECIPE_BODIES` supplies the
deployment's body. Missing declarations raise
`RecipeNotDeclaredError`; multiple ids or an id without a body raise
`UpstreamResponseError`.

Conversion requires a `CurrencyUnit` facet unit, a `Nominal` StatVar, a
recipe, and a StatVar population type in the recipe's `flow_population_types`.
These checks run over node arcs the caller read once with
`CONVERSION_NODE_PROPERTIES`; only a provenance that declares no recipe sends
`resolve_recipe` back to the graph, for its Dataset. `resolve_deflator_role`
requires exactly one declared role in the recipe's deflator-role set.
`graph_role_for_spec_role` maps a request's role key to the StatVar's declared graph
role. An ISO code names `Currency<ISO>`. An entity
resolves to the one currency whose `nationalCurrencyOf` names it; an entity with
anything other than exactly one is unmapped, and the caller decides whether its rows
are omitted or the request refuses (`currency_unmapped_error`).

Facet selection, table loading, arithmetic, tracing, and content-version
minting belong to their callers.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from dfo_platform.executor import submit_with_context
from dfo_platform.problem import Cause
from materialise.errors import MaterialisationError, UpstreamResponseError
from materialise.graph import GraphReader, SubjectPage
from materialise.recipe_bodies import RECIPE_BODIES, ConversionRecipe
from materialise.roles import declared_roles, roles_by_axis

logger = logging.getLogger(__name__)

# Mirrors `search.models.identity._IDENTITY_ENTITY_ROLE_KEYS` (a spec's own
# unit-request role keys), duplicated here rather than imported: materialise
# stays independent of `search.*` (`tests/materialise/test_independence.py`).
type SpecRole = Literal["observationAbout", "origin", "destination"]

_MAX_NAMED_ENTITIES = 10
"""Caps how many entity/currency dcids `RateCoverageGapError.entities`
carries on the wire, matching `materialise.facets`' own
`_MAX_ALTERNATIVES_PER_ISSUE` truncation convention."""

_CURRENCY_UNIT_TYPE = "CurrencyUnit"
_NOMINAL_QUALIFIER = "Nominal"

DATASET_RECIPE_PROPERTIES: tuple[str, ...] = ("conversionRecipe",)
"""The properties `resolve_recipe` reads from a Provenance's Datasets."""
_CURRENCY_NODE_PROPERTIES: tuple[str, ...] = ("typeOf",)
_OBSERVATION_PROPERTIES: tuple[str, ...] = ("observationProperties",)

_NATIONAL_CURRENCY_OF = "nationalCurrencyOf"
# A place with more than one currency refuses, so the reverse read needs only
# enough room to see a second currency.
_MAX_ENTITY_CURRENCIES = 2
_ENTITY_CURRENCY_READS_MAX = 16
"""The most entity-currency reverse reads in flight at once. A role ask over every
recipient of one donor names about 150 entities, so 16 lanes bring its serial depth to
ten reads. It matches `dc_graph.GraphClient`'s default per-call arc concurrency, well
inside that client's connection pool."""

CONVERSION_NODE_PROPERTIES: tuple[str, ...] = (
    "measurementQualifier",
    "populationType",
    "conversionRecipe",
    "isPartOf",
    "typeOf",
)
"""The properties a conversion decision reads from its StatVar, provenance, and currency
nodes, all in one node read. `typeOf` also keeps every live node from coming back with
no arcs, which a multi-node graph read answers with a retry and a probe per node."""

CONVERSION_DECISION_PROPERTIES: frozenset[str] = frozenset(
    (*CONVERSION_NODE_PROPERTIES, _NATIONAL_CURRENCY_OF)
)
"""Every property a conversion decision or an entity-currency lookup reads.
None of them changes within one request, so a request may memoise them."""

type NodeArcs = Mapping[str, tuple[str, ...]] | None
"""One node's arcs from a `node_arcs_many` read, `None` for a node the graph lacks."""


def _values(arcs: NodeArcs, prop: str) -> tuple[str, ...]:
    return arcs.get(prop, ()) if arcs is not None else ()


class RecipeNotDeclaredError(MaterialisationError):
    """422 `recipe_not_declared`; causes are `nominal_undeclared`,
    `recipe_undeclared`, `role_undeclared`, `stock_unsupported`, and
    `recipes_conflict`. Role-axis errors subclass this error.
    """

    def __init__(
        self, *args: object, cause: str, pointer: str | None = None, detail: str | None = None
    ) -> None:
        super().__init__(
            *args, code="recipe_not_declared", status=422, pointer=pointer, detail=detail
        )
        self.cause = cause


class RateCoverageGapError(MaterialisationError):
    """422 `rate_coverage_gap` for `currency_unmapped`, `rate_missing`,
    `deflator_missing`, or `base_year_projected`. `entities` preserves the
    affected dcids/keys in order, capped at `_MAX_NAMED_ENTITIES`; truncation
    sets `entities_truncated`.
    """

    def __init__(
        self,
        *args: object,
        cause: str,
        entities: Sequence[str] = (),
        pointer: str | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(
            *args, code="rate_coverage_gap", status=422, pointer=pointer, detail=detail
        )
        self.cause = cause
        capped = tuple(entities)
        self.entities_truncated = len(capped) > _MAX_NAMED_ENTITIES
        self.entities = capped[:_MAX_NAMED_ENTITIES]


class NonCurrencyUnitError(MaterialisationError):
    """422 `facet_binding_unsupported` for a noncurrency unit. Only
    `recipe_not_declared` and `rate_coverage_gap` carry a cause.
    """

    def __init__(self, *args: object, unit: str, pointer: str | None = None) -> None:
        super().__init__(*args, code="facet_binding_unsupported", status=422, pointer=pointer)
        self.unit = unit


def cause_from_recipe_error(exc: RecipeNotDeclaredError | RateCoverageGapError) -> Cause:
    """Map `exc.cause` to `Cause.code`. Rate-gap errors also copy their
    ordered, capped entity list and truncation flag; recipe errors carry no
    entities and leave `truncated` false.
    """
    if isinstance(exc, RateCoverageGapError):
        return Cause(code=exc.cause, entities=list(exc.entities), truncated=exc.entities_truncated)
    return Cause(code=exc.cause)


def resolve_recipe(
    graph: GraphReader,
    *,
    provenance_id: str,
    provenance_arcs: NodeArcs,
    deadline: float | None = None,
) -> ConversionRecipe:
    """`elect_recipe` over the Provenance's `provenance_arcs` and, when it declares no
    `conversionRecipe`, its `isPartOf` Datasets' arcs, the one case that reads `graph`.
    """
    dataset_ids = recipe_datasets(provenance_arcs)
    dataset_arcs = (
        graph.node_arcs_many(dataset_ids, properties=DATASET_RECIPE_PROPERTIES, deadline=deadline)
        if dataset_ids
        else {}
    )
    return elect_recipe(
        provenance_id=provenance_id, provenance_arcs=provenance_arcs, dataset_arcs=dataset_arcs
    )


def recipe_datasets(provenance_arcs: NodeArcs) -> tuple[str, ...]:
    """The Datasets whose arcs `elect_recipe` reads for a Provenance: its `isPartOf`
    Datasets in the graph's order, or none when it declares its own `conversionRecipe`."""
    if _values(provenance_arcs, "conversionRecipe"):
        return ()
    return _values(provenance_arcs, "isPartOf")


def elect_recipe(
    *,
    provenance_id: str,
    provenance_arcs: NodeArcs,
    dataset_arcs: Mapping[str, NodeArcs],
) -> ConversionRecipe:
    """The recipe a Provenance converts under: its own `conversionRecipe`, else that of the
    first of its `recipe_datasets` that declares one, read from `dataset_arcs` (each read
    with at least `DATASET_RECIPE_PROPERTIES`). No declaration raises
    `RecipeNotDeclaredError(cause="recipe_undeclared")`; the chosen node naming several ids,
    or an id missing from `RECIPE_BODIES`, raises `UpstreamResponseError`.
    """
    recipe_ids = _values(provenance_arcs, "conversionRecipe") or next(
        (
            found
            for dataset_id in recipe_datasets(provenance_arcs)
            if (found := _values(dataset_arcs.get(dataset_id), "conversionRecipe"))
        ),
        (),
    )
    if not recipe_ids:
        raise RecipeNotDeclaredError(
            f"no conversionRecipe declared on provenance {provenance_id!r} or its dataset",
            cause="recipe_undeclared",
        )
    if len(recipe_ids) > 1:
        raise UpstreamResponseError(
            f"provenance {provenance_id!r} resolves to more than one conversionRecipe: "
            f"{recipe_ids!r}"
        )
    recipe_id = recipe_ids[0]
    recipe = RECIPE_BODIES.get(recipe_id)
    if recipe is None:
        raise UpstreamResponseError(
            f"conversionRecipe {recipe_id!r} is declared but this deployment carries no body for it"
        )
    return recipe


def resolve_convertible_recipe(
    graph: GraphReader,
    *,
    stat_var: str,
    stat_var_arcs: NodeArcs,
    provenance_id: str,
    provenance_arcs: NodeArcs,
    deadline: float | None = None,
) -> ConversionRecipe:
    """Return the recipe after checking for a `Nominal` StatVar, a declared recipe, and a
    population type the recipe converts, over arcs read with
    `CONVERSION_NODE_PROPERTIES`. A missing qualifier, recipe, or convertible population
    type raises `RecipeNotDeclaredError`. `graph` serves only the dataset-level recipe
    fallback (`resolve_recipe`).
    """
    check_nominal(stat_var_arcs, stat_var=stat_var)
    recipe = resolve_recipe(
        graph, provenance_id=provenance_id, provenance_arcs=provenance_arcs, deadline=deadline
    )
    check_flow_population_type(stat_var_arcs, stat_var=stat_var, recipe=recipe)
    return recipe


def check_nominal(stat_var_arcs: NodeArcs, *, stat_var: str) -> None:
    """The converting path requires a positive `Nominal` qualifier on the
    StatVar. Any other qualifier, or none, raises
    `RecipeNotDeclaredError(cause="nominal_undeclared")`. Native satisfaction
    uses `check_nominal_natively_satisfied` instead.
    """
    qualifiers = _values(stat_var_arcs, "measurementQualifier")
    if _NOMINAL_QUALIFIER in qualifiers:
        return
    if qualifiers:
        raise RecipeNotDeclaredError(
            f"StatVar {stat_var!r} carries measurementQualifier {qualifiers!r}, never convertible",
            cause="nominal_undeclared",
        )
    raise RecipeNotDeclaredError(
        f"StatVar {stat_var!r} declares no measurementQualifier", cause="nominal_undeclared"
    )


def check_nominal_natively_satisfied(stat_var_arcs: NodeArcs, *, stat_var: str) -> None:
    """Native satisfaction permits `Nominal` or an absent qualifier because
    no arithmetic runs. Any other declared qualifier raises
    `RecipeNotDeclaredError(cause="nominal_undeclared")`; no provenance-level
    nominality declaration applies on this path.
    """
    qualifiers = _values(stat_var_arcs, "measurementQualifier")
    if not qualifiers or _NOMINAL_QUALIFIER in qualifiers:
        return
    raise RecipeNotDeclaredError(
        f"StatVar {stat_var!r} carries measurementQualifier {qualifiers!r}, never convertible",
        cause="nominal_undeclared",
    )


def check_flow_population_type(
    stat_var_arcs: NodeArcs, *, stat_var: str, recipe: ConversionRecipe
) -> None:
    """Refuse unless `stat_var`'s `populationType` is one the recipe converts.
    A StatVar with no population type refuses too, since period-average
    rates are wrong for a stock and the graph gives no other way to tell.
    """
    population_types = _values(stat_var_arcs, "populationType")
    if any(population_type in recipe.flow_population_types for population_type in population_types):
        return
    raise RecipeNotDeclaredError(
        f"StatVar {stat_var!r} has populationType {population_types!r}; conversionRecipe "
        f"{recipe.recipe_id!r} converts only {sorted(recipe.flow_population_types)!r}",
        cause="stock_unsupported",
    )


def read_declared_roles(graph: GraphReader, stat_var: str) -> tuple[str, ...]:
    """Read declared roles from `observationProperties`, using
    `declared_roles`'s `observationAbout` default and degenerate-shape guard.
    Pass the result to both role resolvers to avoid another graph read.
    """
    arcs_by_id = graph.node_arcs_many([stat_var], properties=_OBSERVATION_PROPERTIES)
    stat_var_arcs = arcs_by_id.get(stat_var)
    observation_properties = (
        stat_var_arcs.get("observationProperties", ()) if stat_var_arcs is not None else ()
    )
    return declared_roles(observation_properties)


def resolve_deflator_role(recipe: ConversionRecipe, *, declared_roles: tuple[str, ...]) -> str:
    """Return the sole declared role in the recipe's deflator-role set.
    Zero or multiple matches raise `RecipeNotDeclaredError(cause="role_undeclared")`.
    The role name comes from the graph; the observation row supplies its entity.
    """
    matches = [role for role in declared_roles if role in recipe.deflator_entity_roles]
    if len(matches) != 1:
        raise RecipeNotDeclaredError(
            f"declared roles {declared_roles!r} match {len(matches)} of conversionRecipe "
            f"{recipe.recipe_id!r}'s deflatorEntityRole set "
            f"{sorted(recipe.deflator_entity_roles)!r}; exactly one match is required",
            cause="role_undeclared",
        )
    return matches[0]


class RoleAxisNotDeclaredError(RecipeNotDeclaredError):
    """Raised when declared roles contain no role on the requested axis.
    This subclass preserves adapter handling as `request_invalid`; callers
    can catch it to distinguish it from other `role_undeclared` errors.
    """

    def __init__(
        self, *args: object, pointer: str | None = None, detail: str | None = None
    ) -> None:
        super().__init__(*args, cause="role_undeclared", pointer=pointer, detail=detail)


class RoleAxisConflictError(RecipeNotDeclaredError):
    """Raised when multiple declared roles map to one requested flow axis.
    This is distinct from `RoleAxisNotDeclaredError`, though both share the
    wire cause `role_undeclared`. It remains a `RecipeNotDeclaredError` so
    adapters map the refusal to `request_invalid`; callers distinguish it
    by class.
    """

    def __init__(
        self, *args: object, pointer: str | None = None, detail: str | None = None
    ) -> None:
        super().__init__(*args, cause="role_undeclared", pointer=pointer, detail=detail)


def graph_role_for_spec_role(*, spec_role: SpecRole, declared_roles: tuple[str, ...]) -> str:
    """Map a request role to the StatVar's declared graph role. Check
    `observationAbout` directly and flow axes through `roles_by_axis`.
    This checks each StatVar even when the whole spec passes role-shape validation.

    Missing axes raise `RoleAxisNotDeclaredError`; ambiguous axes raise
    `RoleAxisConflictError`. Both share cause `role_undeclared` and remain
    `RecipeNotDeclaredError` subclasses for adapter handling. An invalid raw
    `spec_role` raises `ValueError`.
    """
    if spec_role == "observationAbout":
        if "observationAbout" not in declared_roles:
            raise RoleAxisNotDeclaredError(
                f"declared roles {declared_roles!r} do not include observationAbout"
            )
        return "observationAbout"
    if spec_role not in ("origin", "destination"):
        raise ValueError(f"spec_role {spec_role!r} is outside the closed SpecRole vocabulary")
    try:
        graph_role = roles_by_axis(declared_roles).get(spec_role)
    except ValueError as exc:
        raise RoleAxisConflictError(str(exc)) from exc
    if graph_role is None:
        raise RoleAxisNotDeclaredError(
            f"declared roles {declared_roles!r} name no role on the {spec_role!r} axis"
        )
    return graph_role


def require_currency_unit(arcs: NodeArcs, *, currency: str) -> str:
    """Return `currency` when its `arcs` type it `CurrencyUnit`, else raise
    `NonCurrencyUnitError`. `Currency{ISO}` names an ISO 4217 code's node, and the type
    check fences out base Data Commons nodes such as `USDollar`. Whether the currency
    converts depends on its rate rows.
    """
    if _CURRENCY_UNIT_TYPE not in _values(arcs, "typeOf"):
        raise NonCurrencyUnitError(f"{currency!r} is not a CurrencyUnit node", unit=currency)
    return currency


@dataclass(frozen=True)
class EntityCurrencies:
    """Each mapped entity's one national currency, and the unmapped entities in input
    order: those no currency names through `nationalCurrencyOf`, those several name, and
    those whose reverse read was truncated."""

    currency_by_entity: Mapping[str, str]
    unmapped: tuple[str, ...]


def resolve_entity_currencies(
    graph: GraphReader, *, entities: Sequence[str], deadline: float | None = None
) -> EntityCurrencies:
    """Map each of `entities` to the one currency whose `nationalCurrencyOf` names it.

    Each entity's reverse read runs on a pool of at most `_ENTITY_CURRENCY_READS_MAX`
    threads under the caller's context and the one absolute `deadline`, so a wide role
    ask costs its reads divided by the pool width in serial time. One node read then types
    every distinct currency found. An entity named by no currency or by more than one, or
    whose reverse read was truncated, is unmapped: the ontology compiler holds each place
    to one national currency, and the served graph enforces no cardinality, so the read
    checks it again. An entity several currencies name, or whose read was truncated, breaks
    that rule: each is logged at WARNING with the entity, the currencies read and whether the
    read was truncated, so the graph fault behind its omitted rows can be found. A mapped
    entity's currency the graph does not type `CurrencyUnit` raises `NonCurrencyUnitError`.
    """
    unique = list(dict.fromkeys(entities))
    pages = _entity_currency_pages(graph, unique, deadline=deadline)
    currency_by_entity: dict[str, str] = {}
    unmapped: list[str] = []
    for entity, page in zip(unique, pages, strict=True):
        if len(page.dcids) == 1 and not page.truncated:
            currency_by_entity[entity] = page.dcids[0]
            continue
        unmapped.append(entity)
        if len(page.dcids) > 1 or page.truncated:
            logger.warning(
                "entity %r is named by currencies %r through nationalCurrencyOf "
                "(read truncated: %r); it is unmapped",
                entity,
                page.dcids,
                page.truncated,
            )
    _require_currency_units(
        graph, list(dict.fromkeys(currency_by_entity.values())), deadline=deadline
    )
    return EntityCurrencies(currency_by_entity=currency_by_entity, unmapped=tuple(unmapped))


def currency_unmapped_error(entities: Iterable[str]) -> RateCoverageGapError:
    """The `rate_coverage_gap`/`currency_unmapped` refusal of a conversion into each
    entity's own currency that leaves no row to convert, naming the unmapped `entities`
    in dcid order, capped as `RateCoverageGapError` caps them."""
    named = sorted(set(entities))
    return RateCoverageGapError(
        f"{len(named)} entities are named by no single currency through nationalCurrencyOf, "
        "and no row of the request names a mapped entity",
        cause="currency_unmapped",
        entities=named,
    )


def deflator_missing_error(omitted: Iterable[tuple[str, int]]) -> RateCoverageGapError:
    """The `rate_coverage_gap`/`deflator_missing` refusal of a constant-price conversion
    that leaves no row to convert because the deflator table lacks every row's value,
    naming each `(entity, year)` as `entity@year`, sorted and capped as
    `RateCoverageGapError` caps them."""
    keys = sorted({f"{entity}@{year}" for entity, year in omitted})
    return RateCoverageGapError(
        f"the deflator table holds no value for {len(keys)} entity-year key(s), "
        "and no row of the request converts",
        cause="deflator_missing",
        entities=keys,
    )


def _entity_currency_pages(
    graph: GraphReader, entities: Sequence[str], *, deadline: float | None
) -> list[SubjectPage]:
    """Each entity's `nationalCurrencyOf` subjects, in `entities` order. One entity reads
    on the calling thread; more fan out over a bounded pool, and the first failed read
    cancels the reads not yet started and raises."""

    def read(entity: str) -> SubjectPage:
        return graph.node_subjects(
            entity,
            property=_NATIONAL_CURRENCY_OF,
            max_results=_MAX_ENTITY_CURRENCIES,
            deadline=deadline,
        )

    if len(entities) <= 1:
        return [read(entity) for entity in entities]
    with ThreadPoolExecutor(max_workers=min(_ENTITY_CURRENCY_READS_MAX, len(entities))) as pool:
        futures = [submit_with_context(pool, read, entity) for entity in entities]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def _require_currency_units(
    graph: GraphReader, currencies: Sequence[str], *, deadline: float | None
) -> None:
    """Raise `NonCurrencyUnitError` for the first of `currencies` the graph does not type
    `CurrencyUnit`, reading every one in one node read."""
    if not currencies:
        return
    arcs_by_id = graph.node_arcs_many(
        currencies, properties=_CURRENCY_NODE_PROPERTIES, deadline=deadline
    )
    for currency in currencies:
        require_currency_unit(arcs_by_id.get(currency), currency=currency)
