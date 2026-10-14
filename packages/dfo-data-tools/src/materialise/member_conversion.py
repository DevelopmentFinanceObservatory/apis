"""Resolve, plan, and apply each member's requested unit conversion.

`decide_member_conversion` validates the native currency, then resolves a
recipe and target unless the current-price request already matches the native
currency. That native path checks nominality only. Constant-price requests
also resolve the deflator role.

`plan_member_conversion` runs before observation fetches. It loads the
recipe's whole tables immediately for uniform targets. Role-shaped targets
load them after fetch, once the served entities' currencies resolve.
A role-shaped target first omits the rows of entities no single currency names
(`omit_unmapped_rows`), before a `latest` window picks its period, so that period is one a
served row holds. `apply_member_conversion` collapses duplicate identity facets before
changing `unit`, then converts or relabels as the plan requires, and records the omitted
entities (`ConversionRecord.omitted_entities`) and the rows a constant-price conversion left
out for a missing deflator (`ConversionRecord.deflator_omitted`). A missing plan leaves the
collapsed table unchanged. A request left with no row refuses at its entry point, once every
member has run, with `currency_unmapped` or `deflator_missing`
(`refuse_if_every_row_omitted`).

A converting decision reads its StatVar, provenance, native currency, and any ISO target
currency in one node read (`CONVERSION_NODE_PROPERTIES`). A provenance that declares no
recipe reads its Dataset in a second. `conversion_graph_reader` wraps a request's graph so
that deciding many members folds every StatVar's arcs into the first decision's read and
reads each provenance, dataset, and currency node once.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.compute as pc

from dc_sdmx import SdmxClient
from materialise.arc_memo import MemoGraphReader
from materialise.convert import (
    ConversionRecord,
    ResolvedUnitRequest,
    RoleShapedTarget,
    RowMarkers,
    UniformTarget,
    convert_member_marked,
    require_conversion_coverage,
    split_record,
)
from materialise.errors import MaterialisationUsageError
from materialise.facet_collapse import collapse_duplicate_identity_facets_table
from materialise.facets import FacetSelection, UnitRequest, native_currency_iso
from materialise.graph import GraphReader
from materialise.rate_tables import RateTable, RateTableCache, load_table
from materialise.recipes import (
    CONVERSION_DECISION_PROPERTIES,
    CONVERSION_NODE_PROPERTIES,
    ConversionRecipe,
    EntityCurrencies,
    SpecRole,
    check_nominal_natively_satisfied,
    currency_unmapped_error,
    deflator_missing_error,
    graph_role_for_spec_role,
    require_currency_unit,
    resolve_convertible_recipe,
    resolve_deflator_role,
    resolve_entity_currencies,
)

if TYPE_CHECKING:
    # `pipeline.py` imports this module through `generation.py`; a runtime
    # import would cycle. Future annotations allow a type-checking import.
    from materialise.pipeline import MaterialisationRequest

_UNIT_COLUMN = "unit"

# Local schema copy keeps this module independent of `convert`'s private constant.
_UNIT_STRUCT_TYPE = pa.struct(
    [("currency", pa.string()), ("prices", pa.string()), ("base_year", pa.int64())]
)

# Requests use `dcid:` refs; graph reads and SDMX use the bare dcid.
_STAT_VAR_REF_PREFIX = "dcid:"


def _is_iso_currency(currency: str) -> bool:
    """Independent copy of `facets._is_iso_currency`: whether `currency` is
    an upper-case ISO 4217 alpha code, as opposed to a spec role key.
    """
    return len(currency) == 3 and currency.isalpha() and currency.isupper()


def _natively_satisfied(unit_request: UnitRequest, *, native_iso: str | None) -> bool:
    """Whether a current-price request names the native currency itself, which needs no
    conversion."""
    return (
        unit_request.prices == "current"
        and _is_iso_currency(unit_request.currency)
        and unit_request.currency == native_iso
    )


def request_conversion_reader(graph: GraphReader) -> GraphReader:
    """A `MemoGraphReader` over `graph` remembering every conversion-decision read
    (`CONVERSION_DECISION_PROPERTIES`) for the life of one request, so the request's unit
    decisions, its coverage check and its previews read each distinct node once. Build one
    per request."""
    return MemoGraphReader(graph, properties=CONVERSION_DECISION_PROPERTIES)


def conversion_graph_reader(graph: GraphReader, *, stat_vars: Sequence[str]) -> GraphReader:
    """A `MemoGraphReader` over `graph` for one request's conversion decisions.

    The first decision read naming a StatVar in `stat_vars` (bare or `dcid:`-prefixed)
    fetches `CONVERSION_NODE_PROPERTIES` for every one of them and for whatever other
    nodes that read names, in one read, and a request whose decisions read no StatVar
    makes none. A converting decision names its provenance and currencies, so members
    sharing that provenance then decide with no further read. A natively satisfied
    decision names only its StatVar and native currency, so a later non-native member
    reads its provenance through the memo's ordinary miss path. Each later read of `CONVERSION_DECISION_PROPERTIES` is served once per distinct node,
    which means one recipe read per provenance, one type read per currency, and one
    reverse read per entity. Rate-table refresh-date reads pass through.
    """
    return MemoGraphReader(
        graph,
        properties=CONVERSION_DECISION_PROPERTIES,
        batch=[stat_var.removeprefix(_STAT_VAR_REF_PREFIX) for stat_var in stat_vars],
        batch_properties=CONVERSION_NODE_PROPERTIES,
    )


@dataclass(frozen=True)
class MemberConversionDecision:
    """A member's conversion decision before tables load or observations are fetched.
    Exactly one of `natively_satisfied`, `uniform_target`, or `spec_role` is set;
    `recipe` is absent only for native satisfaction. Role legality and any
    constant-price deflator role are resolved here for later use.
    """

    unit_request: UnitRequest
    recipe: ConversionRecipe | None
    native_currency: str
    native_unit_ref: str
    declared_roles: tuple[str, ...]
    natively_satisfied: bool
    uniform_target: UniformTarget | None
    spec_role: SpecRole | None
    graph_role: str | None
    deflator_role: str | None

    def __post_init__(self) -> None:
        shapes = (
            self.natively_satisfied,
            self.uniform_target is not None,
            self.spec_role is not None,
        )
        if sum(shapes) != 1:
            raise MaterialisationUsageError(
                "MemberConversionDecision must carry exactly one of natively_satisfied, "
                "uniform_target, or spec_role"
            )
        if (self.recipe is None) != self.natively_satisfied:
            raise MaterialisationUsageError(
                "MemberConversionDecision.recipe must be None exactly for a natively "
                "satisfied decision"
            )


def decide_member_conversion(
    *,
    request: MaterialisationRequest,
    selections: Mapping[str, FacetSelection],
    graph: GraphReader,
    deadline: float | None = None,
) -> MemberConversionDecision | None:
    """Resolve a member's unit request without loading tables or fetching observations.
    Return `None` when the unit binding is a ref or unbound. Mint planning
    calls this before table loads; `decide_member_unit` uses the same decision.
    The native unit resolves through its ISO code (`native_currency_iso`) to
    `Currency<ISO>`, so a `USDollar` facet satisfies and converts as `CurrencyUSD`.
    The decision reads the StatVar, the provenance, and both `Currency<ISO>` nodes in one
    `graph.node_arcs_many` call under `deadline`. A natively satisfied request reads only the
    StatVar and the native currency. A provenance that declares no `conversionRecipe`
    costs a second read, of its Dataset.

    Raises `NonCurrencyUnitError` for a native or target currency the graph
    does not type `CurrencyUnit`, or `RecipeNotDeclaredError` for
    `nominal_undeclared`, `recipe_undeclared`, `stock_unsupported`, or
    `role_undeclared`. A missing requested role raises its
    `RoleAxisNotDeclaredError` subclass. A currency the rate table does not
    cover refuses at conversion with `rate_missing`.
    """
    return _decide(
        stat_var=request.stat_var,
        declared_roles=request.declared_roles,
        selections=selections,
        graph=graph,
        deadline=deadline,
    )


def _decide(
    *,
    stat_var: str,
    declared_roles: tuple[str, ...],
    selections: Mapping[str, FacetSelection],
    graph: GraphReader,
    deadline: float | None = None,
) -> MemberConversionDecision | None:
    """`decide_member_conversion` over the two request fields it reads."""
    unit_selection = selections["unit"]
    unit_request = unit_selection.request
    if unit_request is None:
        return None

    native_unit_ref = unit_selection.ref
    native_iso = native_currency_iso(native_unit_ref)
    # `select_facet` raises `NonCurrencyUnitError` for any other unit request result.
    assert native_unit_ref is not None and native_iso is not None
    provenance_id = selections["provenance"].ref
    assert provenance_id is not None  # `select_facet` never resolves provenance to a null ref.
    bare_stat_var = stat_var.removeprefix(_STAT_VAR_REF_PREFIX)
    # Name the native currency by its ISO code: the unit ref `USDollar` is a base Data
    # Commons node, and its rates key on `CurrencyUSD`.
    native_node = f"Currency{native_iso}"
    target_node = (
        f"Currency{unit_request.currency}" if _is_iso_currency(unit_request.currency) else None
    )
    satisfied = _natively_satisfied(unit_request, native_iso=native_iso)
    # A natively satisfied request needs only the StatVar's qualifier and the native currency.
    nodes = [bare_stat_var, native_node]
    if not satisfied:
        nodes += [provenance_id, *([target_node] if target_node else [])]
    arcs = graph.node_arcs_many(
        list(dict.fromkeys(nodes)), properties=CONVERSION_NODE_PROPERTIES, deadline=deadline
    )

    native_currency = require_currency_unit(arcs.get(native_node), currency=native_node)

    if satisfied:
        # A matching current-price currency runs only the nominal check. Bare
        # StatVars pass here because this path performs no arithmetic.
        check_nominal_natively_satisfied(arcs.get(bare_stat_var), stat_var=bare_stat_var)
        return MemberConversionDecision(
            unit_request=unit_request,
            recipe=None,
            native_currency=native_currency,
            native_unit_ref=native_unit_ref,
            declared_roles=declared_roles,
            natively_satisfied=True,
            uniform_target=None,
            spec_role=None,
            graph_role=None,
            deflator_role=None,
        )

    # A current- or constant-price conversion resolves the recipe (bundling
    # the nominal and population-type checks).
    recipe = resolve_convertible_recipe(
        graph,
        stat_var=bare_stat_var,
        stat_var_arcs=arcs.get(bare_stat_var),
        provenance_id=provenance_id,
        provenance_arcs=arcs.get(provenance_id),
        deadline=deadline,
    )

    if target_node is None:
        spec_role = cast(SpecRole, unit_request.currency)
        # Resolve role legality before fetching rows; apply reads the stored role.
        graph_role = graph_role_for_spec_role(spec_role=spec_role, declared_roles=declared_roles)
        deflator_role: str | None = None
        if unit_request.prices == "constant":
            # Resolve deflator legality pre-fetch and store the role for apply.
            deflator_role = resolve_deflator_role(recipe, declared_roles=declared_roles)
        return MemberConversionDecision(
            unit_request=unit_request,
            recipe=recipe,
            native_currency=native_currency,
            native_unit_ref=native_unit_ref,
            declared_roles=declared_roles,
            natively_satisfied=False,
            uniform_target=None,
            spec_role=spec_role,
            graph_role=graph_role,
            deflator_role=deflator_role,
        )

    target_currency = require_currency_unit(arcs.get(target_node), currency=target_node)
    deflator_role = None
    if unit_request.prices == "constant":
        deflator_role = resolve_deflator_role(recipe, declared_roles=declared_roles)
    return MemberConversionDecision(
        unit_request=unit_request,
        recipe=recipe,
        native_currency=native_currency,
        native_unit_ref=native_unit_ref,
        declared_roles=declared_roles,
        natively_satisfied=False,
        uniform_target=UniformTarget(target_currency),
        spec_role=None,
        graph_role=None,
        deflator_role=deflator_role,
    )


@dataclass(frozen=True)
class MemberConversionPlan:
    """A member's conversion plan, resolved before observation fetch.
    Uniform targets carry preloaded tables. Role-shaped targets load tables
    in `apply_member_conversion`, using this plan's graph, SDMX client, cache,
    and deadline. `__post_init__` enforces the decision invariant.
    """

    unit_request: UnitRequest
    recipe: ConversionRecipe | None
    native_currency: str
    native_unit_ref: str
    declared_roles: tuple[str, ...]
    natively_satisfied: bool
    uniform_target: UniformTarget | None
    spec_role: SpecRole | None
    graph_role: str | None
    deflator_role: str | None
    exchange_rate_table: RateTable | None
    deflator_table: RateTable | None
    graph: GraphReader
    sdmx: SdmxClient
    rate_cache: RateTableCache
    deadline: float | None = None

    def __post_init__(self) -> None:
        shapes = (
            self.natively_satisfied,
            self.uniform_target is not None,
            self.spec_role is not None,
        )
        if sum(shapes) != 1:
            raise MaterialisationUsageError(
                "MemberConversionPlan must carry exactly one of natively_satisfied, "
                "uniform_target, or spec_role"
            )
        if (self.recipe is None) != self.natively_satisfied:
            raise MaterialisationUsageError(
                "MemberConversionPlan.recipe must be None exactly for a natively satisfied plan"
            )


def plan_member_conversion(
    *,
    request: MaterialisationRequest,
    selections: Mapping[str, FacetSelection],
    graph: GraphReader,
    sdmx: SdmxClient,
    rate_cache: RateTableCache,
    deadline: float | None = None,
) -> MemberConversionPlan | None:
    """Plan conversion before fetch, or return `None` for a ref or unbound
    unit binding. Uniform targets load tables here; role-shaped targets load
    them in `apply_member_conversion` after their currencies resolve. Table
    loading may raise `RateTableInvalidError` or `UpstreamVersionChangedError`.
    """
    decision = decide_member_conversion(
        request=request, selections=selections, graph=graph, deadline=deadline
    )
    if decision is None:
        return None

    exchange_rate_table: RateTable | None = None
    deflator_table: RateTable | None = None
    if decision.uniform_target is not None:
        assert decision.recipe is not None  # non-native: `MemberConversionDecision` guarantees it
        exchange_rate_table, deflator_table = _load_tables(
            recipe=decision.recipe,
            prices=decision.unit_request.prices,
            graph=graph,
            sdmx=sdmx,
            rate_cache=rate_cache,
            deadline=deadline,
        )

    return MemberConversionPlan(
        unit_request=decision.unit_request,
        recipe=decision.recipe,
        native_currency=decision.native_currency,
        native_unit_ref=decision.native_unit_ref,
        declared_roles=decision.declared_roles,
        natively_satisfied=decision.natively_satisfied,
        uniform_target=decision.uniform_target,
        spec_role=decision.spec_role,
        graph_role=decision.graph_role,
        deflator_role=decision.deflator_role,
        exchange_rate_table=exchange_rate_table,
        deflator_table=deflator_table,
        graph=graph,
        sdmx=sdmx,
        rate_cache=rate_cache,
        deadline=deadline,
    )


def _load_tables(
    *,
    recipe: ConversionRecipe,
    prices: str,
    graph: GraphReader,
    sdmx: SdmxClient,
    rate_cache: RateTableCache,
    deadline: float | None,
) -> tuple[RateTable, RateTable | None]:
    """Load the recipe's whole exchange-rate table, and its deflator table
    for constant prices. Each table holds every entity its StatVar serves,
    so one load covers any native, target, deflator, or fallback entity.
    """
    exchange_rate_table = load_table(
        kind="exchange_rate",
        stat_var=recipe.exchange_rate_variable,
        provenance_id=recipe.table_provenance,
        sdmx=sdmx,
        graph=graph,
        cache=rate_cache,
        deadline=deadline,
    )
    if prices != "constant":
        return exchange_rate_table, None
    deflator_table = load_table(
        kind="deflator",
        stat_var=recipe.deflator_variable,
        provenance_id=recipe.table_provenance,
        sdmx=sdmx,
        graph=graph,
        cache=rate_cache,
        deadline=deadline,
    )
    return exchange_rate_table, deflator_table


def _require_string_unit_column(table: pa.Table) -> None:
    """Require string-typed `unit` before facet collapse; struct-typed input
    raises `MaterialisationUsageError` because collapse reads strings.
    """
    if pa.types.is_struct(table.schema.field(_UNIT_COLUMN).type):
        raise MaterialisationUsageError(
            "apply_member_conversion: table's own unit column is already struct-typed; "
            "collapse_duplicate_identity_facets_table reads it as a string"
        )


def omit_unmapped_rows(
    plan: MemberConversionPlan | None,
    table: pa.Table,
    *,
    served: pa.Table,
    entities: Iterable[str] = (),
) -> tuple[pa.Table, EntityCurrencies | None]:
    """Drop `table`'s rows binding, on a role-shaped target's role, an entity no single
    currency names, and return the kept rows with the currencies resolved.

    Currencies resolve, in one read, for the entities `served` binds on that role, the rows
    of `table` the read can serve, and `entities` (`_role_currencies`). A bound entity with
    no served row needs no mapping, and a row of an entity `served` lacks stays, for the
    caller's own filters to drop. Every other plan returns `table` and None.
    """
    if plan is None or plan.spec_role is None:
        return table, None
    graph_role = plan.graph_role
    assert graph_role is not None  # `MemberConversionDecision`: set with `spec_role`
    currencies = _role_currencies(
        plan.graph,
        entities=[*served[graph_role].to_pylist(), *entities],
        deadline=plan.deadline,
    )
    return without_unmapped(plan, table, currencies), currencies


def without_unmapped(
    plan: MemberConversionPlan | None, table: pa.Table, currencies: EntityCurrencies | None
) -> pa.Table:
    """`table` less the rows binding, on a role-shaped target's role, an entity `currencies`
    holds as unmapped (`omit_unmapped_rows`). `table` itself when `currencies` names none."""
    if plan is None or plan.graph_role is None or currencies is None or not currencies.unmapped:
        return table
    unmapped = pa.array(currencies.unmapped, type=pa.string())
    return table.filter(pc.invert(pc.is_in(table[plan.graph_role], value_set=unmapped)))


def apply_member_conversion(
    plan: MemberConversionPlan | None,
    table: pa.Table,
    *,
    declared_roles: Sequence[str],
    currencies: EntityCurrencies | None = None,
) -> tuple[pa.Table, ConversionRecord | None]:
    """Collapse duplicate identity facets before conversion; the collapse expects
    `unit` as a string. Return the frame-ready table and a conversion record,
    or `None` for no plan or native satisfaction. Row order stays unchanged, and a
    planned result has struct-typed `unit`, even when empty. Every row stays.

    A role-shaped target converts rows `omit_unmapped_rows` already kept, into the
    `currencies` it returned, and its record names their unmapped entities in
    `omitted_entities`. A role-shaped plan without `currencies`, or a plan whose
    `declared_roles` differ from this call's, raises `MaterialisationUsageError`.
    """
    converted, record, _markers = _apply(
        plan,
        table,
        declared_roles=declared_roles,
        currencies=currencies,
        rank_within_series=False,
    )
    return converted, record


def apply_member_conversion_by_series(
    plan: MemberConversionPlan | None,
    table: pa.Table,
    *,
    declared_roles: Sequence[str],
    currencies: EntityCurrencies | None = None,
) -> tuple[pa.Table, ConversionRecord | None, dict[tuple[str, ...], ConversionRecord]]:
    """`apply_member_conversion` as if each series, one dcid on every declared role, were
    applied on its own, in one call: facets rank within a series
    (`collapse_duplicate_identity_facets_table`'s `rank_within_series`), and each series'
    record, keyed by its dcids in `declared_roles` order, counts and names the fallback and
    projection markers of its own rows (`convert.split_record`). The member's record is the
    whole conversion's, which `merged_record` of the series' records equals. No plan, a
    natively satisfied one, or no declared role records no series."""
    converted, record, markers = _apply(
        plan,
        table,
        declared_roles=declared_roles,
        currencies=currencies,
        rank_within_series=True,
    )
    if record is None or not declared_roles:
        return converted, record, {}
    assert markers is not None  # `_apply`: `None` only beside a `None` record
    row_keys = list(zip(*(converted[role].to_pylist() for role in declared_roles), strict=True))
    return converted, record, split_record(record, markers, row_keys=row_keys)


def _apply(
    plan: MemberConversionPlan | None,
    table: pa.Table,
    *,
    declared_roles: Sequence[str],
    currencies: EntityCurrencies | None,
    rank_within_series: bool,
) -> tuple[pa.Table, ConversionRecord | None, RowMarkers | None]:
    """`apply_member_conversion` with the conversion's row markers, or `None` beside a
    `None` record."""
    _require_string_unit_column(table)
    if plan is not None and tuple(declared_roles) != plan.declared_roles:
        raise MaterialisationUsageError(
            f"apply_member_conversion: declared_roles {tuple(declared_roles)!r} disagrees "
            f"with plan.declared_roles {plan.declared_roles!r}"
        )
    collapsed = collapse_duplicate_identity_facets_table(
        table, declared_roles=declared_roles, rank_within_series=rank_within_series
    )
    if plan is None:
        return collapsed, None, None

    if plan.natively_satisfied:
        return _apply_native_satisfaction(plan, collapsed), None, None

    if plan.uniform_target is not None:
        assert plan.exchange_rate_table is not None
        assert plan.recipe is not None  # `__post_init__`: `None` only when natively satisfied
        return convert_member_marked(
            rows=collapsed,
            unit_request=ResolvedUnitRequest(
                prices=plan.unit_request.prices,
                target=plan.uniform_target,
                base_year=plan.unit_request.base_year,
            ),
            recipe=plan.recipe,
            native_currency=plan.native_currency,
            native_unit_ref=plan.native_unit_ref,
            declared_roles=plan.declared_roles,
            exchange_rate_table=plan.exchange_rate_table,
            deflator_table=plan.deflator_table,
        )

    assert plan.spec_role is not None
    if currencies is None:
        raise MaterialisationUsageError(
            "apply_member_conversion: a role-shaped plan converts the rows and currencies "
            "omit_unmapped_rows returned"
        )
    return _apply_role_shaped(plan, collapsed, currencies)


def _apply_native_satisfaction(plan: MemberConversionPlan, table: pa.Table) -> pa.Table:
    """Relabel `unit` uniformly without changing values. The output column
    is struct-typed even when the input has zero rows.
    """
    cell = {
        "currency": plan.unit_request.currency,
        "prices": plan.unit_request.prices,
        "base_year": plan.unit_request.base_year,
    }
    unit_array = pa.array([cell] * table.num_rows, type=_UNIT_STRUCT_TYPE)
    return table.set_column(table.schema.get_field_index(_UNIT_COLUMN), _UNIT_COLUMN, unit_array)


def _apply_role_shaped(
    plan: MemberConversionPlan, table: pa.Table, currencies: EntityCurrencies
) -> tuple[pa.Table, ConversionRecord, RowMarkers]:
    """Load or reuse the recipe's tables from `plan.rate_cache` and convert the collapsed
    rows into `currencies`, whose unmapped entities' rows `omit_unmapped_rows` dropped. The
    record names those entities. The plan stores role legality and the deflator role before
    fetch.
    """
    spec_role = plan.spec_role
    assert spec_role is not None
    recipe = plan.recipe
    assert recipe is not None  # `__post_init__`: `None` only when natively satisfied
    target = RoleShapedTarget(
        spec_role=spec_role, currency_by_entity=dict(currencies.currency_by_entity)
    )

    exchange_rate_table, deflator_table = _load_tables(
        recipe=recipe,
        prices=plan.unit_request.prices,
        graph=plan.graph,
        sdmx=plan.sdmx,
        rate_cache=plan.rate_cache,
        deadline=plan.deadline,
    )

    converted, record, markers = convert_member_marked(
        rows=table,
        unit_request=ResolvedUnitRequest(
            prices=plan.unit_request.prices, target=target, base_year=plan.unit_request.base_year
        ),
        recipe=recipe,
        native_currency=plan.native_currency,
        native_unit_ref=plan.native_unit_ref,
        declared_roles=plan.declared_roles,
        exchange_rate_table=exchange_rate_table,
        deflator_table=deflator_table,
    )
    return converted, replace(record, omitted_entities=currencies.unmapped), markers


def _role_currencies(
    graph: GraphReader, *, entities: Iterable[str | None], deadline: float | None
) -> EntityCurrencies:
    """Each distinct entity's one national currency, read for all of them at once
    (`resolve_entity_currencies`), with the unmapped entities in dcid order."""
    served = sorted({entity for entity in entities if entity is not None})
    return resolve_entity_currencies(graph, entities=served, deadline=deadline)


def refuse_if_every_row_omitted(
    *, served_rows: int, records: Iterable[ConversionRecord | None]
) -> None:
    """Raise `RateCoverageGapError` when a request serves no row and its conversion omitted
    some rows: `currency_unmapped` naming every entity no single currency names
    (`ConversionRecord.omitted_entities`), else `deflator_missing` naming each entity and
    year the deflator table lacks (`ConversionRecord.deflator_omitted`). A request that
    serves a row keeps it and warns `currency_unmapped_omitted` or
    `deflator_missing_omitted` instead; one that omitted nothing returns."""
    if served_rows:
        return
    converted = [record for record in records if record is not None]
    omitted = {entity for record in converted for entity in record.omitted_entities}
    if omitted:
        raise currency_unmapped_error(omitted)
    deflated = [
        (entity, year)
        for record in converted
        for entity, years in record.deflator_omitted
        for year in years
    ]
    if deflated:
        raise deflator_missing_error(deflated)


@dataclass(frozen=True)
class ServedBlock:
    """Rows a read will serve in `year`, as conversion sees them: each row binds, on every
    graph role in `entities`, one of that role's dcids. A block with one dcid per role is one
    row. A block with several stands for rows pairing them in ways the caller does not record,
    so it holds at most their product. Build one with `served_block`."""

    year: int
    entities: tuple[tuple[str, frozenset[str]], ...]


def served_block(year: int, entities: Mapping[str, Iterable[str]]) -> ServedBlock:
    """The `ServedBlock` of rows in `year` binding `entities`, by graph role. A role with no
    dcid is left out, so its rows bind no entity there."""
    by_role = {role: frozenset(dcids) for role, dcids in entities.items()}
    return ServedBlock(
        year=year, entities=tuple(sorted((role, dcids) for role, dcids in by_role.items() if dcids))
    )


def coverage_decision(
    *,
    stat_var: str,
    declared_roles: tuple[str, ...],
    selections: Mapping[str, FacetSelection],
    graph: GraphReader,
    deadline: float | None = None,
) -> MemberConversionDecision | None:
    """The conversion decision a coverage check of rows served under `selections` runs,
    or None when those rows need no table: an unbound or ref-form unit, or a natively
    satisfied one, which its unit decision already checked. Members whose decisions are
    equal read the same tables through the same recipe, so one check covers all their
    rows (`require_decision_coverage`). The decision's graph reads run under `deadline`.
    Raises the decision's `RecipeNotDeclaredError` or `NonCurrencyUnitError`.
    """
    unit = selections["unit"]
    if unit.request is None or _natively_satisfied(
        unit.request, native_iso=native_currency_iso(unit.ref)
    ):
        return None
    decision = _decide(
        stat_var=stat_var,
        declared_roles=declared_roles,
        selections=selections,
        graph=graph,
        deadline=deadline,
    )
    if decision is None or decision.natively_satisfied:
        return None
    return decision


def _block_terms(entities: Mapping[str, Sequence[str]]) -> Iterator[Mapping[str, str]]:
    """One term per dcid of a block's widest role, pairing every role's dcids cyclically, so
    each role's dcid meets the block's year without listing the roles' product."""
    width = max(len(dcids) for dcids in entities.values())
    for i in range(width):
        yield {role: dcids[i % len(dcids)] for role, dcids in entities.items()}


@dataclass(frozen=True)
class DecisionCoverage:
    """What converting served rows under one decision keeps: whether any served row
    survives, the entities, sorted, whose rows a role-shaped target omits because no
    single currency names them, and the `(entity, year)` pairs, sorted, whose rows a
    constant-price conversion omits because the deflator table lacks their value."""

    survives: bool
    omitted_entities: tuple[str, ...] = ()
    deflator_omitted: tuple[tuple[str, int], ...] = ()


def require_decision_coverage(
    decision: MemberConversionDecision,
    *,
    served: Iterable[ServedBlock],
    graph: GraphReader,
    sdmx: SdmxClient,
    rate_cache: RateTableCache | None,
    deadline: float | None = None,
) -> DecisionCoverage:
    """Raise the refusal converting the rows `served` holds under `decision` raises, before
    any row is read, or return what the conversion keeps.

    A role-shaped target resolves the currency of every entity the blocks bind on its role
    (`_role_currencies`) and drops the unmapped ones from each block, as
    `omit_unmapped_rows` drops their rows. The recipe's tables then load through
    `rate_cache` whenever any block was served, even when every row was dropped, as a read
    loads them for any role-shaped member, so a table that fails to load refuses here as it
    refuses the read. The remaining terms go to `convert.require_conversion_coverage`, the
    lookup `convert_member_marked` itself runs, and a term missing its deflator keeps no
    row there. A block left with no entity on that role keeps no
    row. Dropping an entity is exact for a block of one row; in a block of several, an
    entity on another role keeps its year even when its rows paired it only with dropped
    entities, so such a block can over-check.
    A block that binds no entity on a role the conversion reads cannot be checked, is
    skipped, and survives. Raises `RateCoverageGapError` with the cause, entities and
    truncation a conversion raises, or `NonCurrencyUnitError` for an entity currency the
    graph does not type `CurrencyUnit`. Whether a request with no surviving row refuses
    `currency_unmapped` or `deflator_missing` is the caller's decision across all its decisions
    (`refuse_if_every_row_omitted` for a read). A conversion without `rate_cache` raises
    `MaterialisationUsageError`, as a read without one does.
    """
    recipe = decision.recipe
    assert recipe is not None  # non-native: `MemberConversionDecision` guarantees it
    if rate_cache is None:
        raise MaterialisationUsageError(
            "require_decision_coverage: converting a member needs a rate_cache"
        )
    needed = [role for role in (decision.graph_role, decision.deflator_role) if role is not None]
    blocks: list[tuple[int, dict[str, frozenset[str]]]] = []
    unchecked = 0
    for block in served:
        entities = dict(block.entities)
        if all(role in entities for role in needed):
            blocks.append((block.year, entities))
        else:
            unchecked += 1
    served_any = bool(blocks) or unchecked > 0
    omitted: tuple[str, ...] = ()
    mapped: Mapping[str, str] = {}
    graph_role = decision.graph_role
    if decision.uniform_target is None and blocks:
        assert graph_role is not None
        currencies = _role_currencies(
            graph,
            entities=(entity for _year, entities in blocks for entity in entities[graph_role]),
            deadline=deadline,
        )
        omitted = currencies.unmapped
        mapped = currencies.currency_by_entity
        blocks = [
            (year, {**entities, graph_role: kept})
            for year, entities in blocks
            if (kept := entities[graph_role] & mapped.keys())
        ]
    coverage = DecisionCoverage(survives=bool(blocks) or unchecked > 0, omitted_entities=omitted)
    if not served_any:
        return coverage
    exchange_rate_table, deflator_table = _load_tables(
        recipe=recipe,
        prices=decision.unit_request.prices,
        graph=graph,
        sdmx=sdmx,
        rate_cache=rate_cache,
        deadline=deadline,
    )
    if not blocks:
        return coverage
    terms = [
        (year, term)
        for year, entities in blocks
        for term in _block_terms({role: sorted(dcids) for role, dcids in entities.items()})
    ]
    if decision.uniform_target is not None:
        targets = [decision.uniform_target.currency] * len(terms)
    else:
        assert graph_role is not None
        targets = [mapped[term[graph_role]] for _year, term in terms]
    deflator_role = decision.deflator_role
    dropped = require_conversion_coverage(
        prices=decision.unit_request.prices,
        base_year=decision.unit_request.base_year,
        recipe=recipe,
        native_currency=decision.native_currency,
        target_currencies=targets,
        deflator_entities=(
            [term[deflator_role] for _year, term in terms] if deflator_role is not None else None
        ),
        years=[year for year, _term in terms],
        exchange_rate_table=exchange_rate_table,
        deflator_table=deflator_table,
    )
    if not dropped:
        return coverage
    assert deflator_role is not None  # a missing deflator reads the deflator role
    return replace(
        coverage,
        survives=unchecked > 0 or len(dropped) < len(terms),
        deflator_omitted=tuple(
            sorted({(terms[i][1][deflator_role], terms[i][0]) for i in dropped})
        ),
    )
