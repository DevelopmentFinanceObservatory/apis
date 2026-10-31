"""Converts member rows using a resolved `ConversionRecipe` and loaded rate tables.
Callers resolve graph facts and pass them as plain data.

`X` is an exchange rate in units of a currency per US dollar, keyed by
currency, and `D` a US-dollar deflator, keyed by entity. One formula serves
every recipe, in left-to-right binary64 elementwise operations with no fused
multiply-add:

* Current prices use ``V x X[T,t] / X[S,t]``.
* Constant prices use ``V x (D[R,b] / D[R,t]) x X[T,b] / X[S,t]``.

`V` is `OBS_VALUE`. Null values stay null and skip gap handling. `t` is the row year, `b` the
base year, `S` the native currency, and `T` the uniform target or the currency bound to a
spec role. `R` is the entity in the recipe's deflator role, or `fallback_entity` when it has
no row in the deflator table. It appears only in the deflator terms. A fallback applies
only when the entity is absent for every year (`deflator_key`). A missing year for a present
entity is a gap. `row_terms` lists the terms a row reads at each price basis and
`converted_facets` the facets a recipe converts with; the search engine's conversion offers
read both, so they state only years a conversion accepts.

Missing rates raise `RateCoverageGapError` with sorted, capped `key@year` keys. A missing
deflator leaves out the row that reads it: an entity whose deflator the table lacks in a
row's year loses that row, and one whose deflator it lacks in the base year loses every
row, and the record names each entity with the years it lost
(`ConversionRecord.deflator_omitted`). A row with no entity on the deflator role names
nothing the record could list, so it refuses with `deflator_missing` (`None@year`). Whether
a request left with no row refuses is the caller's decision across its members
(`member_conversion.refuse_if_every_row_omitted`).
Every projected term, the base-year terms included, converts under `FlagProjected` and
records its key. Under `RefuseProjected` a projected base-year term refuses with
`base_year_projected` and any other projected term counts as a gap. Error priority is
`base_year_projected`, then `rate_missing`. An unknown forecast policy raises
`UpstreamResponseError`.

The output preserves columns and the order of the rows it keeps, changing only `OBS_VALUE`
and `unit`. Values are
stored as strings using `repr()`, which round-trips exactly through `observed_values`. Unit
is a per-row `{currency, prices, base_year}` struct, allowing
role-shaped targets. `convert_scalar` replays the formula with Python
floats in the same order."""

from __future__ import annotations

from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, cast

import pyarrow as pa
import pyarrow.compute as pc

from dc_sdmx.results import VALUE_COLUMN
from materialise.columnar import leading_years, observed_values
from materialise.errors import MaterialisationUsageError, UpstreamResponseError
from materialise.rate_tables import RateTable
from materialise.recipes import (
    _MAX_NAMED_ENTITIES,
    ConversionRecipe,
    RateCoverageGapError,
    SpecRole,
    graph_role_for_spec_role,
    resolve_deflator_role,
)

_UNIT_COLUMN = "unit"

# Mirrors `dfo_frame_grammar.grammar.UnitRequest`'s field set, defined here
# to keep this module independent of `dfo_frame_grammar`.
_UNIT_STRUCT_TYPE = pa.struct(
    [("currency", pa.string()), ("prices", pa.string()), ("base_year", pa.int64())]
)

_DEFLATOR_UNKEYED = "deflator_unkeyed"
"""A deflator term with no entity or no year to key it: a row whose deflator-role cell is
null names no entity a record could list, so it refuses as `deflator_missing`."""
_GAP_PRIORITY: tuple[str, ...] = (
    "base_year_projected",
    "rate_missing",
    _DEFLATOR_UNKEYED,
    "deflator_missing",
)
_REFUSING_GAPS: tuple[str, ...] = _GAP_PRIORITY[:3]
"""The gaps that refuse a conversion. A `deflator_missing` gap leaves its rows out."""
_RAISED_CAUSE: Mapping[str, str] = {_DEFLATOR_UNKEYED: "deflator_missing"}
"""The `RateCoverageGapError` cause a gap raises under, where it differs from its name."""

type TermTable = Literal["exchange_rate", "deflator"]
"""The table a term is read from."""
type TermKey = Literal["native", "target", "deflator_entity"]
"""What keys a term in its table: the row's native currency, its target currency, or its
entity on the recipe's deflator role (`deflator_key`)."""


@dataclass(frozen=True, slots=True)
class RowTerm:
    """One term of the formula: the value `table` holds for the row's `key`, in the base
    year when `at_base_year`, else in the row's year. `name` is the formula's symbol."""

    name: Literal["x_source_t", "x_target_t", "d_r_b", "d_r_t", "x_target_b"]
    table: TermTable
    key: TermKey
    at_base_year: bool


_X_SOURCE_T = RowTerm("x_source_t", "exchange_rate", "native", at_base_year=False)
_ROW_TERMS: Mapping[str, tuple[RowTerm, ...]] = {
    "current": (_X_SOURCE_T, RowTerm("x_target_t", "exchange_rate", "target", at_base_year=False)),
    "constant": (
        _X_SOURCE_T,
        RowTerm("d_r_b", "deflator", "deflator_entity", at_base_year=True),
        RowTerm("d_r_t", "deflator", "deflator_entity", at_base_year=False),
        RowTerm("x_target_b", "exchange_rate", "target", at_base_year=True),
    ),
}


def row_terms(prices: Literal["current", "constant"]) -> tuple[RowTerm, ...]:
    """Every term a row converted at `prices` reads, in the order a conversion looks them
    up. A row converts exactly when every term is found under `converted_facets`. The
    conversion (`_gather_terms`) and the search engine's conversion offers and periods
    (`search.engine.unit.conversion.ConversionAvailability`) both read the terms here."""
    return _ROW_TERMS[prices]


def converted_facets(forecast_policy: str) -> frozenset[str]:
    """The table facets a recipe under `forecast_policy` converts with: actual values, and
    projected ones under `FlagProjected`. A term under any other facet is a gap."""
    return frozenset({"actual", "projected"} if forecast_policy == "FlagProjected" else {"actual"})


def deflator_key(entity: str, *, fallback_entity: str | None, held: Container[str]) -> str:
    """The deflator table key a row of `entity` reads: `entity` itself, or `fallback_entity`
    when the table, whose keys are `held`, holds no row for `entity` in any year. A missing
    year for a held entity is a gap, never a fallback."""
    if fallback_entity is not None and entity not in held:
        return fallback_entity
    return entity


_RECOGNIZED_FORECAST_POLICIES: frozenset[str] = frozenset({"FlagProjected", "RefuseProjected"})

# `CurrencyUnit` refs use `Currency<ISO>`. Frame unit cells use the bare ISO code.
_CURRENCY_DCID_PREFIX = "Currency"


def _iso_currency(currency_dcid: str) -> str:
    return currency_dcid.removeprefix(_CURRENCY_DCID_PREFIX)


@dataclass(frozen=True)
class UniformTarget:
    """`T` is one `CurrencyUnit` dcid for every row of the member."""

    currency: str


@dataclass(frozen=True)
class RoleShapedTarget:
    """Uses the `CurrencyUnit` dcid mapped to each row's entity in `spec_role`.
    `convert_member_marked` raises `MaterialisationUsageError` when the map lacks a bound entity."""

    spec_role: SpecRole
    currency_by_entity: Mapping[str, str]


@dataclass(frozen=True)
class ResolvedUnitRequest:
    """The target unit. `current` prices omit `base_year`. `constant` prices require it."""

    prices: Literal["current", "constant"]
    target: UniformTarget | RoleShapedTarget
    base_year: int | None = None


@dataclass(frozen=True)
class ConversionRecord:
    """Recipe/table versions and fallback/projection counts for one member. `table_provenance_id`
    is `recipe.table_provenance`. `table_refresh_date` is the `lastDataRefreshDate` the tables
    were admitted under, the head-reuse change signal. It is `None` when the exchange-rate and
    deflator tables were admitted under different dates, so the next head recheck rebuilds.
    `projected_count` counts rows priced from at least one projected term. Sorted examples are
    capped at `_MAX_NAMED_ENTITIES`. Fallback pairs are per entity, and projected keys are
    `currency@year` or `entity@year`. Markers use keys because frame sorting changes row
    positions. `native_currency` is the `Currency<ISO>` rate key the rows convert from.
    `native_unit_ref` is the member's native facet unit, which can be a twin of that key
    (`USDollar` for `CurrencyUSD`); manifest provenance names this ref. `omitted_entities`
    holds every entity, sorted and uncapped, whose rows a conversion into each entity's own
    currency left out because no single currency names it (`currency_unmapped_omitted`).
    `deflator_omitted` holds every entity, sorted, whose rows a constant-price conversion
    left out because the deflator table lacks its value, each with the years of those rows,
    sorted and uncapped (`deflator_missing_omitted`). No other conversion omits a row."""

    recipe_id: str
    exchange_rate_table_version: str
    deflator_table_version: str | None
    native_currency: str
    native_unit_ref: str
    table_provenance_id: str
    table_refresh_date: str | None
    fallback_count: int
    fallback_examples: tuple[tuple[str, str], ...]
    projected_count: int
    projected_examples: tuple[str, ...]
    omitted_entities: tuple[str, ...] = ()
    deflator_omitted: tuple[tuple[str, tuple[int, ...]], ...] = ()


def _omitted_years(
    pairs: Iterable[tuple[str, int]],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """`(entity, year)` pairs as each entity, sorted, with its distinct years, sorted."""
    by_entity: dict[str, set[int]] = {}
    for entity, year in pairs:
        by_entity.setdefault(entity, set()).add(year)
    return tuple((entity, tuple(sorted(years))) for entity, years in sorted(by_entity.items()))


@dataclass(frozen=True)
class RowMarkers:
    """Which rows of one `convert_member_marked` output read a fallback deflator entity or a
    projected term, by row index. `fallback` holds one `(row, (entity, fallback entity))` per
    substituted row, and `projected` one `(row, key)` per projected term a row read."""

    fallback: tuple[tuple[int, tuple[str, str]], ...]
    projected: tuple[tuple[int, str], ...]


def _marked_record(
    record: ConversionRecord,
    *,
    fallback: Sequence[tuple[int, tuple[str, str]]],
    projected: Sequence[tuple[int, str]],
) -> ConversionRecord:
    """`record` with the fallback and projection fields these markers give: counts of rows,
    and distinct examples sorted and capped at `_MAX_NAMED_ENTITIES`."""
    return replace(
        record,
        fallback_count=len(fallback),
        fallback_examples=tuple(sorted({pair for _row, pair in fallback})[:_MAX_NAMED_ENTITIES]),
        projected_count=len({row for row, _key in projected}),
        projected_examples=tuple(sorted({key for _row, key in projected})[:_MAX_NAMED_ENTITIES]),
    )


def split_record[K](
    record: ConversionRecord, markers: RowMarkers, *, row_keys: Sequence[K]
) -> dict[K, ConversionRecord]:
    """`record` split over the series `row_keys` names for each converted row: one record per
    distinct key, in first-seen order, counting and naming that series' markers alone. Every
    other field is the member's, so `merged_record` of the parts is `record`."""
    fallback: dict[K, list[tuple[int, tuple[str, str]]]] = {key: [] for key in row_keys}
    projected: dict[K, list[tuple[int, str]]] = {key: [] for key in row_keys}
    for marker in markers.fallback:
        fallback[row_keys[marker[0]]].append(marker)
    for marker in markers.projected:
        projected[row_keys[marker[0]]].append(marker)
    return {
        key: _marked_record(record, fallback=fallback[key], projected=projected[key])
        for key in fallback
    }


def _trace_key(record: ConversionRecord) -> tuple[str, str, str | None]:
    """The (recipe, rate-table version, deflator-table version) a trace entry is keyed on."""
    return (
        record.recipe_id,
        record.exchange_rate_table_version,
        record.deflator_table_version,
    )


def merged_record(records: Sequence[ConversionRecord]) -> ConversionRecord | None:
    """One record for rows converted in several parts under one recipe and tables: the
    parts' counts summed, their distinct examples sorted and capped at
    `_MAX_NAMED_ENTITIES`, every part's omitted entities, sorted, and every part's
    deflator omissions, each entity's years joined. None for no part.

    Every part must share the recipe and both table versions, the key of one
    `ManifestTrace.conversion` entry. Parts that differ raise `MaterialisationUsageError`,
    because summing them would attribute rows to tables they did not convert under.
    """
    if not records:
        return None
    keys = {_trace_key(record) for record in records}
    if len(keys) > 1:
        raise MaterialisationUsageError(
            "merged_record needs parts converted under one recipe and table-version pair, got "
            f"{sorted(keys, key=repr)!r}"
        )
    return replace(
        records[0],
        fallback_count=sum(record.fallback_count for record in records),
        fallback_examples=tuple(
            sorted({example for record in records for example in record.fallback_examples})[
                :_MAX_NAMED_ENTITIES
            ]
        ),
        projected_count=sum(record.projected_count for record in records),
        projected_examples=tuple(
            sorted({example for record in records for example in record.projected_examples})[
                :_MAX_NAMED_ENTITIES
            ]
        ),
        omitted_entities=tuple(
            sorted({entity for record in records for entity in record.omitted_entities})
        ),
        deflator_omitted=_omitted_years(
            (entity, year)
            for record in records
            for entity, years in record.deflator_omitted
            for year in years
        ),
    )


def convert_scalar(
    v: float,
    *,
    prices: Literal["current", "constant"],
    x_target_t: float | None = None,
    x_source_t: float,
    x_target_b: float | None = None,
    d_r_b: float | None = None,
    d_r_t: float | None = None,
) -> float:
    """Replays the vectorised formula with Python floats in the same left-to-right binary64
    order. Current prices read `x_target_t`. Constant prices read the base-year terms."""
    if prices == "current":
        assert x_target_t is not None
        return v * x_target_t / x_source_t
    assert d_r_b is not None and d_r_t is not None and x_target_b is not None
    return v * (d_r_b / d_r_t) * x_target_b / x_source_t


def _resolve_targets(
    rows: pa.Table,
    target: UniformTarget | RoleShapedTarget,
    *,
    declared_roles: tuple[str, ...],
    skip: Sequence[bool],
) -> list[str]:
    """Returns one target `CurrencyUnit` dcid per row. Uniform targets repeat one dcid.
    Role-shaped targets resolve the row's bound entity, raising `MaterialisationUsageError`
    if the map lacks it. Null-value rows are skipped and receive placeholder empty strings.
    Frame construction filters those rows before reading their unit."""
    if isinstance(target, UniformTarget):
        return [target.currency] * rows.num_rows

    graph_role = graph_role_for_spec_role(spec_role=target.spec_role, declared_roles=declared_roles)
    entities = rows[graph_role].to_pylist()
    currencies: list[str] = []
    for entity, is_skipped in zip(entities, skip, strict=True):
        if is_skipped:
            currencies.append("")
            continue
        currency = target.currency_by_entity.get(entity) if entity is not None else None
        if currency is None:
            raise MaterialisationUsageError(
                f"convert_member_marked's role-shaped target carries no currency for "
                f"entity {entity!r} bound in role {graph_role!r}"
            )
        currencies.append(currency)
    return currencies


def _apply_fallback(
    r_raw: Sequence[str | None],
    *,
    fallback_entity: str | None,
    deflator_entities: frozenset[str],
    skip: Sequence[bool],
) -> tuple[list[str | None], list[tuple[int, tuple[str, str]]]]:
    """Each non-skipped row's deflator key (`deflator_key`): `fallback_entity` only when the
    row's entity is absent from the entire deflator table. Returns effective entities and one `(row, (entity, fallback
    entity))` per substituted row."""
    effective: list[str | None] = []
    substituted: list[tuple[int, tuple[str, str]]] = []
    for row, (entity, is_skipped) in enumerate(zip(r_raw, skip, strict=True)):
        if is_skipped or entity is None:
            effective.append(entity)
            continue
        key = deflator_key(entity, fallback_entity=fallback_entity, held=deflator_entities)
        effective.append(key)
        if key != entity:
            substituted.append((row, (entity, key)))
    return effective, substituted


@dataclass
class _Terms:
    """Gap and projection bookkeeping shared by every term lookup of one member."""

    skip: Sequence[bool]
    converted: frozenset[str]
    gaps: dict[str, set[str]]
    projected: list[tuple[int, str]]
    deflator_gap_rows: set[int] = field(default_factory=set)


def _term_values(
    lookup: Mapping[tuple[str, int], tuple[float, str]],
    terms: _Terms,
    *,
    keys: Sequence[str | None],
    years: Sequence[int | None],
    at_base_year: bool,
    kind: Literal["exchange_rate", "deflator"],
) -> list[float | None]:
    """Looks up one term per row. Skipped null-value rows return `None`. Missing terms and
    terms under a facet outside `terms.converted` (`converted_facets`) are added to
    `terms.gaps`. The caller raises after checking every term. A row missing the deflator of
    a named entity in a named year joins `terms.deflator_gap_rows`, the rows a conversion
    leaves out; a deflator term with no entity or year refuses (`_DEFLATOR_UNKEYED`), since
    no record could name the row it would leave out. A converted projected
    term records its row and key in `terms.projected`."""
    missing_cause = "rate_missing" if kind == "exchange_rate" else "deflator_missing"
    refused_cause = "base_year_projected" if at_base_year else missing_cause
    # Rows repeat a few (entity, year) keys, so each distinct key is looked up once.
    outcomes: dict[tuple[str | None, int | None], _TermOutcome] = {}
    values: list[float | None] = []
    for index, (key_entity, year, is_skipped) in enumerate(
        zip(keys, years, terms.skip, strict=True)
    ):
        if is_skipped:
            values.append(None)
            continue
        outcome = outcomes.get((key_entity, year))
        if outcome is None:
            outcome = outcomes[key_entity, year] = _term_outcome(
                lookup,
                terms,
                key_entity=key_entity,
                year=year,
                kind=kind,
                missing_cause=missing_cause,
                refused_cause=refused_cause,
            )
        value, leaves_row_out, projected_key = outcome
        if leaves_row_out:
            terms.deflator_gap_rows.add(index)
        if projected_key is not None:
            terms.projected.append((index, projected_key))
        values.append(value)
    return values


type _TermOutcome = tuple[float | None, bool, str | None]


def _term_outcome(
    lookup: Mapping[tuple[str, int], tuple[float, str]],
    terms: _Terms,
    *,
    key_entity: str | None,
    year: int | None,
    kind: Literal["exchange_rate", "deflator"],
    missing_cause: str,
    refused_cause: str,
) -> _TermOutcome:
    """The term `_term_values` reads at `key_entity` and `year`, adding its gap to
    `terms.gaps`: its value, whether a row reading it is left out for a missing deflator, and
    its key when the term is projected."""
    key = f"{key_entity}@{year}"
    found = lookup.get((key_entity, year)) if key_entity is not None and year is not None else None
    if found is None:
        if kind == "exchange_rate":
            terms.gaps[missing_cause].add(key)
            return None, False, None
        if key_entity is None or year is None:
            terms.gaps[_DEFLATOR_UNKEYED].add(key)
            return None, False, None
        terms.gaps[missing_cause].add(key)
        return None, True, None
    value, facet = found
    if facet not in terms.converted:
        terms.gaps[refused_cause].add(key)
        return None, refused_cause == "deflator_missing", None
    return value, False, key if facet == "projected" else None


def _raise_if_gaps(gaps: Mapping[str, set[str]], *, causes: Sequence[str] = _GAP_PRIORITY) -> None:
    """Raise the gap of the first of `causes`, in priority order, that `gaps` holds."""
    for cause in causes:
        entities = gaps.get(cause)
        if entities:
            raised = _RAISED_CAUSE.get(cause, cause)
            raise RateCoverageGapError(
                f"conversion refuses: {raised} for {len(entities)} entity-year key(s)",
                cause=raised,
                entities=tuple(sorted(entities)),
            )


def _as_float_array(values: Sequence[float | None]) -> pa.Array:
    return pa.array(values, type=pa.float64())


def _pc_multiply(left: pa.Array, right: pa.Array) -> pa.Array:
    """Narrows PyArrow's broad stub return to `pa.Array` for arithmetic chaining."""
    return cast(pa.Array, pc.multiply(left, right))


def _pc_divide(left: pa.Array, right: pa.Array) -> pa.Array:
    """Narrows PyArrow's result as described by `_pc_multiply`."""
    return cast(pa.Array, pc.divide(left, right))


@dataclass(frozen=True)
class _GatheredTerms:
    """Every term one member's rows read, with the fallback and gap bookkeeping. Current
    prices fill `x_target_t`; constant prices fill `d_r_b`, `d_r_t` and `x_target_b`."""

    terms: _Terms
    target_currencies: Sequence[str]
    x_source_t: list[float | None]
    x_target_t: list[float | None] | None = None
    d_r_b: list[float | None] | None = None
    d_r_t: list[float | None] | None = None
    x_target_b: list[float | None] | None = None
    fallback: list[tuple[int, tuple[str, str]]] = field(default_factory=list)


def _gather_terms(
    *,
    prices: Literal["current", "constant"],
    base_year: int | None,
    recipe: ConversionRecipe,
    native_currency: str,
    target_currencies: Sequence[str],
    deflator_entities: Sequence[str | None] | None,
    years: Sequence[int | None],
    skip: Sequence[bool],
    exchange_rate_table: RateTable,
    deflator_table: RateTable | None,
) -> _GatheredTerms:
    """Look up every term the formula reads for rows given as their target currency,
    deflator-role entity (constant prices only) and year, adding missing and refused terms
    to the gaps rather than raising. `convert_member_marked` and `require_conversion_coverage`
    share it, so a coverage check and a conversion of the same terms find the same gaps."""
    if recipe.forecast_policy not in _RECOGNIZED_FORECAST_POLICIES:
        raise UpstreamResponseError(
            f"conversionRecipe {recipe.recipe_id!r} declares unrecognized forecastPolicy "
            f"{recipe.forecast_policy!r}"
        )
    n = len(years)
    terms = _Terms(
        skip=skip,
        converted=converted_facets(recipe.forecast_policy),
        gaps={cause: set() for cause in _GAP_PRIORITY},
        projected=[],
    )
    keys: dict[TermKey, Sequence[str | None]] = {
        "native": [native_currency] * n,
        "target": target_currencies,
    }
    lookups = {"exchange_rate": exchange_rate_table.lookup}
    fallback: list[tuple[int, tuple[str, str]]] = []
    if prices == "constant":
        if deflator_table is None or base_year is None or deflator_entities is None:
            raise MaterialisationUsageError(
                "convert_member_marked: prices='constant' requires both deflator_table and "
                "unit_request.base_year"
            )
        keys["deflator_entity"], fallback = _apply_fallback(
            deflator_entities,
            fallback_entity=recipe.fallback_entity,
            deflator_entities=deflator_table.entities,
            skip=skip,
        )
        lookups["deflator"] = deflator_table.lookup
    values = {
        term.name: _term_values(
            lookups[term.table],
            terms,
            keys=keys[term.key],
            years=[base_year] * n if term.at_base_year else years,
            at_base_year=term.at_base_year,
            kind=term.table,
        )
        for term in row_terms(prices)
    }
    return _GatheredTerms(
        terms=terms,
        target_currencies=target_currencies,
        x_source_t=values["x_source_t"],
        x_target_t=values.get("x_target_t"),
        d_r_b=values.get("d_r_b"),
        d_r_t=values.get("d_r_t"),
        x_target_b=values.get("x_target_b"),
        fallback=fallback,
    )


def require_conversion_coverage(
    *,
    prices: Literal["current", "constant"],
    base_year: int | None,
    recipe: ConversionRecipe,
    native_currency: str,
    target_currencies: Sequence[str],
    deflator_entities: Sequence[str | None] | None,
    years: Sequence[int],
    exchange_rate_table: RateTable,
    deflator_table: RateTable | None = None,
) -> frozenset[int]:
    """Raise the `RateCoverageGapError` `convert_member_marked` raises for rows holding these
    terms, without rows, or return the rows it leaves out for a missing deflator: row `i`
    converts from `native_currency` to `target_currencies[i]` in `years[i]`, deflated
    through `deflator_entities[i]` at constant prices. A caller that knows the years and
    entities a read will serve learns before the read whether its conversion succeeds and
    which rows it keeps."""
    gathered = _gather_terms(
        prices=prices,
        base_year=base_year,
        recipe=recipe,
        native_currency=native_currency,
        target_currencies=target_currencies,
        deflator_entities=deflator_entities,
        years=years,
        skip=[False] * len(years),
        exchange_rate_table=exchange_rate_table,
        deflator_table=deflator_table,
    )
    _raise_if_gaps(gathered.terms.gaps, causes=_REFUSING_GAPS)
    return frozenset(gathered.terms.deflator_gap_rows)


@dataclass(frozen=True)
class _MemberInputs:
    """What converting one member's rows reads besides the rows."""

    unit_request: ResolvedUnitRequest
    recipe: ConversionRecipe
    native_currency: str
    declared_roles: tuple[str, ...]
    exchange_rate_table: RateTable
    deflator_table: RateTable | None


def _gather_rows(
    rows: pa.Table, inputs: _MemberInputs
) -> tuple[_GatheredTerms, list[str | None] | None, list[int | None]]:
    """Every term `rows` read (`_gather_terms`), with each row's entity on the recipe's
    deflator role at constant prices and each row's year."""
    nulls = pc.is_null(observed_values(rows[VALUE_COLUMN])).to_pylist()
    skip: list[bool] = [bool(value) for value in nulls]
    years: list[int | None] = leading_years(rows).to_pylist()
    request = inputs.unit_request
    target_currencies = _resolve_targets(
        rows, request.target, declared_roles=inputs.declared_roles, skip=skip
    )
    deflator_entities = None
    if request.prices == "constant" and inputs.deflator_table is not None:
        r_role = resolve_deflator_role(inputs.recipe, declared_roles=inputs.declared_roles)
        deflator_entities = rows[r_role].to_pylist()
    gathered = _gather_terms(
        prices=request.prices,
        base_year=request.base_year,
        recipe=inputs.recipe,
        native_currency=inputs.native_currency,
        target_currencies=target_currencies,
        deflator_entities=deflator_entities,
        years=years,
        skip=skip,
        exchange_rate_table=inputs.exchange_rate_table,
        deflator_table=inputs.deflator_table,
    )
    return gathered, deflator_entities, years


def _keyed[T](value: T | None) -> T:
    """`value` of a row `_term_values` left out, which only a keyed term can be."""
    assert value is not None  # an unkeyed deflator term refuses (`_DEFLATOR_UNKEYED`)
    return value


def _gather_kept(
    rows: pa.Table, inputs: _MemberInputs
) -> tuple[pa.Table, _GatheredTerms, tuple[tuple[str, tuple[int, ...]], ...]]:
    """The rows a conversion keeps, their terms, and the rows it leaves out for a missing
    deflator, by entity and year. Raises the refusing gaps (`_REFUSING_GAPS`) of every row
    first, so a rate gap refuses even in a row a missing deflator leaves out. The kept rows'
    terms are gathered again, so markers index the kept rows."""
    gathered, deflator_entities, years = _gather_rows(rows, inputs)
    _raise_if_gaps(gathered.terms.gaps, causes=_REFUSING_GAPS)
    dropped = gathered.terms.deflator_gap_rows
    if not dropped:
        return rows, gathered, ()
    assert deflator_entities is not None  # a deflator gap reads a deflator entity
    omitted = _omitted_years(
        (_keyed(deflator_entities[row]), _keyed(years[row])) for row in sorted(dropped)
    )
    kept = rows.filter(pa.array([row not in dropped for row in range(rows.num_rows)]))
    gathered, _entities, _years = _gather_rows(kept, inputs)
    assert not gathered.terms.deflator_gap_rows
    return kept, gathered, omitted


def convert_member_marked(
    *,
    rows: pa.Table,
    unit_request: ResolvedUnitRequest,
    recipe: ConversionRecipe,
    native_currency: str,
    native_unit_ref: str,
    declared_roles: tuple[str, ...],
    exchange_rate_table: RateTable,
    deflator_table: RateTable | None = None,
) -> tuple[pa.Table, ConversionRecord, RowMarkers]:
    """Converts one member's observation table using supplied rate tables, and returns the
    rows each fallback and projection marker came from (`RowMarkers`), from which the
    record's counts and examples derive (`split_record` for each series' share).

    `native_currency` is the `Currency<ISO>` rate key of the member's native facet unit, and
    `native_unit_ref` is that facet unit's own dcid, recorded for provenance.
    Raises `MaterialisationUsageError` when constant prices lack a deflator table or base
    year, or a role-shaped target omits a bound entity. An unknown forecast policy raises
    `UpstreamResponseError`. Missing or refused rates, and a projected base-year term under
    `RefuseProjected`, raise `RateCoverageGapError`. A row missing a deflator is left out
    and named in the record's `deflator_omitted`; markers index the rows kept."""
    rows, gathered, omitted = _gather_kept(
        rows,
        _MemberInputs(
            unit_request=unit_request,
            recipe=recipe,
            native_currency=native_currency,
            declared_roles=declared_roles,
            exchange_rate_table=exchange_rate_table,
            deflator_table=deflator_table,
        ),
    )
    v = observed_values(rows[VALUE_COLUMN])
    target_currencies = gathered.target_currencies
    terms = gathered.terms
    x_source_t = _as_float_array(gathered.x_source_t)
    if unit_request.prices == "current":
        assert gathered.x_target_t is not None
        result = _pc_divide(_pc_multiply(v, _as_float_array(gathered.x_target_t)), x_source_t)
        deflator_table_version = None
        table_refresh_date: str | None = exchange_rate_table.refresh_date
    else:
        assert deflator_table is not None
        assert gathered.d_r_b is not None and gathered.d_r_t is not None
        assert gathered.x_target_b is not None
        term1 = _pc_divide(_as_float_array(gathered.d_r_b), _as_float_array(gathered.d_r_t))
        step = _pc_multiply(_pc_multiply(v, term1), _as_float_array(gathered.x_target_b))
        result = _pc_divide(step, x_source_t)
        deflator_table_version = deflator_table.version
        table_refresh_date = (
            exchange_rate_table.refresh_date
            if deflator_table.refresh_date == exchange_rate_table.refresh_date
            else None
        )

    converted = _apply_conversion(
        rows,
        result=result,
        prices=unit_request.prices,
        base_year=unit_request.base_year,
        target_currencies=[_iso_currency(currency) for currency in target_currencies],
    )
    record = ConversionRecord(
        recipe_id=recipe.recipe_id,
        exchange_rate_table_version=exchange_rate_table.version,
        deflator_table_version=deflator_table_version,
        native_currency=native_currency,
        native_unit_ref=native_unit_ref,
        table_provenance_id=recipe.table_provenance,
        table_refresh_date=table_refresh_date,
        fallback_count=0,
        fallback_examples=(),
        projected_count=0,
        projected_examples=(),
        deflator_omitted=omitted,
    )
    markers = RowMarkers(fallback=tuple(gathered.fallback), projected=tuple(terms.projected))
    return (
        converted,
        _marked_record(record, fallback=markers.fallback, projected=markers.projected),
        markers,
    )


def _apply_conversion(
    rows: pa.Table,
    *,
    result: pa.Array,
    prices: str,
    base_year: int | None,
    target_currencies: Sequence[str],
) -> pa.Table:
    """Replaces `OBS_VALUE` with the result's `repr()` (`""` for null) and `unit` with a
    per-row `{currency, prices, base_year}` struct, null when the value is null. Other columns
    and row order stay unchanged."""
    values = result.to_pylist()
    value_strings = pa.array(
        ["" if value is None else repr(value) for value in values], type=pa.string()
    )
    unit_cells = [
        None if value is None else {"currency": currency, "prices": prices, "base_year": base_year}
        for value, currency in zip(values, target_currencies, strict=True)
    ]
    unit_array = pa.array(unit_cells, type=_UNIT_STRUCT_TYPE)

    converted = rows.set_column(
        rows.schema.get_field_index(VALUE_COLUMN), VALUE_COLUMN, value_strings
    )
    return converted.set_column(
        converted.schema.get_field_index(_UNIT_COLUMN), _UNIT_COLUMN, unit_array
    )
