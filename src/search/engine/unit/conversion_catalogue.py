"""The conversion catalogue: the facts every answer's unit conversion offer and the period
a unit request shows (`unit.conversion`) read, loaded once.

`ResolvedSpec.unit_conversion` needs four facts that live in the graph and the conversion
tables: which recipe each provenance converts under (`conversionRecipe`, on the provenance
or the first of its Datasets declaring one, as `materialise.recipes.elect_recipe` elects
it), the years each currency's exchange rate, each entity's deflator, and the deflator's actual values cover in each
recipe's tables, the base year the tables declare, and each place's one national currency
(`nationalCurrencyOf`). Reading them per answer would add network reads to every answer,
so the app loads them once at startup (`load_conversion_catalogue`) and each request reads
the loaded catalogue from memory.

`ConversionCatalogueService` holds the loaded catalogue. The app's lifespan starts its load
beside its other startup work, waits for it at most a bounded time, and requests a reload
when the graph generation changes, since every fact here lives in the graph. One daemon
thread runs the loads in order, retrying a failed one with backoff until a load succeeds.
A failed load logs an ERROR and leaves the last loaded catalogue in place, or none. The
app's request scope binds the service's catalogue for each request
(`bound_conversion_catalogue`), None while none has loaded, and the engine reads it with
`current_conversion_catalogue`. `conversion_catalogue_unavailable` is true for a request
whose scope bound no catalogue, whose answers then state that offers are unavailable, and
false for code that runs outside any request scope, as tests that bind none do.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType

from dc_sdmx import SdmxClient
from dfo_platform.executor import submit_with_context
from materialise.convert import converted_facets, deflator_key
from materialise.errors import UpstreamResponseError
from materialise.facets import native_currency_iso
from materialise.graph import GraphReader
from materialise.rate_tables import (
    RateTable,
    RateTableCache,
    RateTableInvalidError,
    RateTableKind,
    load_table,
)
from materialise.recipe_bodies import RECIPE_BODIES, ConversionRecipe
from materialise.recipes import (
    DATASET_RECIPE_PROPERTIES,
    NodeArcs,
    RecipeNotDeclaredError,
    elect_recipe,
    recipe_datasets,
)
from search.engine.clients.graph import EngineGraphClient
from search.engine.unit.base_year import declared_base_year
from search.engine.unit.lexicon import ISO_4217_CODES

logger = logging.getLogger(__name__)

type _TableKey = tuple[RateTableKind, str, str]
"""A conversion table: its kind, StatVar, and table provenance."""

_RECIPE_PROPERTY = "conversionRecipe"
_NATIONAL_CURRENCY_OF = "nationalCurrencyOf"
_CURRENCY_UNIT = "CurrencyUnit"
_DATASET = "Dataset"
_RECIPE_NODE_PROPERTIES = (*DATASET_RECIPE_PROPERTIES, "isPartOf", "typeOf")
"""What the catalogue reads of each node a recipe election consults. `typeOf` tells a
Dataset from a provenance and keeps every live node from coming back with no arcs."""
_MAX_SUBJECTS = 5000
"""The most subjects one reverse read lists: the provenances or Datasets naming a recipe,
the provenances of one Dataset, or the currency units. A truncated read fails the load."""


@dataclass(frozen=True, slots=True)
class RecipeTables:
    """What one recipe's tables hold: the years its exchange-rate table holds a rate the
    recipe converts with for each ISO code a unit request accepts, the years its deflator
    table holds a value the recipe converts with for each entity it holds any row for (none
    for an entity whose every row is under a facet the recipe refuses), and the first and
    last years that table holds an actual value for. A rate or deflator the recipe converts
    with is under one of `materialise.convert.converted_facets`."""

    rate_years: Mapping[str, frozenset[int]]
    deflator_years: Mapping[str, frozenset[int]]
    base_year_min: int
    base_year_max: int

    @property
    def currencies(self) -> tuple[str, ...]:
        """The ISO codes the exchange-rate table holds a rate for, sorted."""
        return tuple(sorted(self.rate_years))

    @property
    def years(self) -> frozenset[int]:
        """Every year either table holds a value the recipe converts with in."""
        return frozenset().union(*self.rate_years.values(), *self.deflator_years.values())

    def deflator_years_of(self, entity: str, *, fallback: str | None) -> frozenset[int]:
        """The years a conversion deflates `entity`'s rows in: those of the key a conversion
        reads for it (`materialise.convert.deflator_key`), its own or `fallback`'s when the
        table holds no row for it."""
        key = deflator_key(entity, fallback_entity=fallback, held=self.deflator_years)
        return self.deflator_years.get(key, frozenset())


@dataclass(frozen=True, slots=True)
class ConversionCatalogue:
    """The recipe of each provenance that names one, each recipe's `RecipeTables`, the ISO
    code of each place's one national currency, the tables' declared base year, and the
    graph name of each entity a deflator table holds and of each place with a national
    currency, by which an answer names the entities whose rows a missing deflator leaves
    out. A served entity outside both, such as a multilateral donor or a regional
    aggregate, has no name here, and the answer counts it among the others.

    `revision` is a SHA-256 digest of those facts, so two catalogues share a revision
    exactly when they hold the same facts. The response cache keys an answer on the
    revision its request bound (`search.engine.response_cache_flow.CacheScope`)."""

    recipes: Mapping[str, str]
    tables: Mapping[str, RecipeTables]
    national_currencies: Mapping[str, str]
    default_base_year: int
    entity_names: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    revision: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "revision", _revision(self))

    def recipe(self, provenance: str) -> ConversionRecipe | None:
        """The recipe `provenance` names, None when it names none."""
        recipe_id = self.recipes.get(provenance)
        return RECIPE_BODIES[recipe_id] if recipe_id is not None else None


def _revision(catalogue: ConversionCatalogue) -> str:
    """The SHA-256 digest of `catalogue`'s facts, each mapping and year set sorted."""

    def years(by_key: Mapping[str, frozenset[int]]) -> list[tuple[str, list[int]]]:
        return [(key, sorted(found)) for key, found in sorted(by_key.items())]

    facts = (
        sorted(catalogue.recipes.items()),
        [
            (
                recipe_id,
                years(tables.rate_years),
                years(tables.deflator_years),
                tables.base_year_min,
                tables.base_year_max,
            )
            for recipe_id, tables in sorted(catalogue.tables.items())
        ],
        sorted(catalogue.national_currencies.items()),
        catalogue.default_base_year,
        sorted(catalogue.entity_names.items()),
    )
    return hashlib.sha256(repr(facts).encode()).hexdigest()


def load_conversion_catalogue(
    *,
    graph: EngineGraphClient,
    reader: GraphReader,
    sdmx: SdmxClient,
    cache: RateTableCache,
    deadline: float | None = None,
) -> ConversionCatalogue:
    """Read the catalogue, its independent reads at once on a pool under the caller's
    context: each distinct table through `cache`, one reverse read per recipe (and per
    Dataset naming one) with up to three node reads for the recipe election, one reverse and one node read for the currency units, and the
    declared base year (`declared_base_year`, read through `graph`), then one name read for
    every entity a deflator table holds and every place with a national currency. Every
    other read goes through `reader`. Raises the first failed read's error,
    `RateTableInvalidError` for a deflator table with no actual value, and
    `UpstreamResponseError` for a truncated listing."""
    table_keys = sorted({key for recipe in RECIPE_BODIES.values() for key in _table_keys(recipe)})

    def table(key: _TableKey) -> RateTable:
        kind, stat_var, provenance = key
        return load_table(
            kind=kind,
            stat_var=stat_var,
            provenance_id=provenance,
            sdmx=sdmx,
            graph=reader,
            cache=cache,
            deadline=deadline,
        )

    with ThreadPoolExecutor(max_workers=len(table_keys) + 3) as pool:
        table_futures = {key: submit_with_context(pool, table, key) for key in table_keys}
        recipes = submit_with_context(pool, _provenance_recipes, reader, deadline)
        currencies = submit_with_context(pool, _national_currencies, reader, deadline)
        base_year = submit_with_context(pool, declared_base_year, graph, deadline=deadline)
        loaded = {key: future.result() for key, future in table_futures.items()}
        national = currencies.result()
        named = sorted(
            {
                entity
                for (kind, _stat_var, _provenance), table in loaded.items()
                if kind == "deflator"
                for entity in table.entities
            }
            | national.keys()
        )
        names = reader.node_names_many(named, deadline=deadline) if named else {}
        return ConversionCatalogue(
            recipes=MappingProxyType(recipes.result()),
            tables=MappingProxyType(
                {
                    recipe.recipe_id: _recipe_tables(
                        recipe, *(loaded[key] for key in _table_keys(recipe))
                    )
                    for recipe in RECIPE_BODIES.values()
                }
            ),
            national_currencies=MappingProxyType(national),
            default_base_year=base_year.result(),
            entity_names=MappingProxyType(
                {dcid: found[0] for dcid, found in names.items() if found}
            ),
        )


def _table_keys(recipe: ConversionRecipe) -> tuple[_TableKey, _TableKey]:
    """The keys of `recipe`'s exchange-rate and deflator tables."""
    return (
        ("exchange_rate", recipe.exchange_rate_variable, recipe.table_provenance),
        ("deflator", recipe.deflator_variable, recipe.table_provenance),
    )


def _subjects(reader: GraphReader, dcid: str, prop: str, deadline: float | None) -> tuple[str, ...]:
    page = reader.node_subjects(dcid, property=prop, max_results=_MAX_SUBJECTS, deadline=deadline)
    if page.truncated:
        raise UpstreamResponseError(f"more than {_MAX_SUBJECTS} nodes name {dcid!r} by {prop}")
    return page.dcids


def _provenance_recipes(reader: GraphReader, deadline: float | None) -> dict[str, str]:
    """Each provenance's recipe, as the materialiser elects it (`elect_recipe`) from the
    same arcs `resolve_recipe` reads. A provenance can elect a recipe only when it or one
    of its Datasets names one with a body, so the candidates are the nodes naming such a
    recipe and the provenances of every Dataset among them. A candidate the materialiser
    would refuse, such as one naming two recipes, has none."""
    naming = sorted(
        {
            subject
            for recipe_id in RECIPE_BODIES
            for subject in _subjects(reader, recipe_id, _RECIPE_PROPERTY, deadline)
        }
    )
    arcs = _read_missing(reader, {}, naming, deadline)
    candidates = {subject for subject in naming if _DATASET not in _types(arcs[subject])}
    for dataset in sorted(set(naming) - candidates):
        candidates.update(_subjects(reader, dataset, "isPartOf", deadline))
    arcs = _read_missing(reader, arcs, sorted(candidates), deadline)
    datasets = {dataset for node in candidates for dataset in recipe_datasets(arcs[node])}
    arcs = _read_missing(reader, arcs, sorted(datasets), deadline)
    recipes: dict[str, str] = {}
    for provenance in sorted(candidates):
        try:
            recipe = elect_recipe(
                provenance_id=provenance, provenance_arcs=arcs[provenance], dataset_arcs=arcs
            )
        except (RecipeNotDeclaredError, UpstreamResponseError) as exc:
            logger.error("%s elects no conversion recipe: %s", provenance, exc)
            continue
        recipes[provenance] = recipe.recipe_id
    return recipes


def _read_missing(
    reader: GraphReader,
    arcs: dict[str, NodeArcs],
    dcids: Sequence[str],
    deadline: float | None,
) -> dict[str, NodeArcs]:
    """`arcs` with those of each of `dcids` it lacks, read in one node read."""
    missing = [dcid for dcid in dcids if dcid not in arcs]
    if not missing:
        return arcs
    read = reader.node_arcs_many(missing, properties=_RECIPE_NODE_PROPERTIES, deadline=deadline)
    return arcs | {dcid: read.get(dcid) for dcid in missing}


def _types(arcs: NodeArcs) -> tuple[str, ...]:
    return arcs.get("typeOf", ()) if arcs is not None else ()


def _recipe_tables(
    recipe: ConversionRecipe, exchange_rates: RateTable, deflators: RateTable
) -> RecipeTables:
    years = [year for (_, year), (_, facet) in deflators.lookup.items() if facet == "actual"]
    if not years:
        raise RateTableInvalidError(
            f"deflator table {recipe.deflator_variable!r} holds no actual value"
        )
    converted = converted_facets(recipe.forecast_policy)
    rate_years: dict[str, set[int]] = {}
    for (entity, year), (_, facet) in exchange_rates.lookup.items():
        code = native_currency_iso(entity)
        if code in ISO_4217_CODES and facet in converted:
            rate_years.setdefault(code, set()).add(year)
    deflator_years: dict[str, set[int]] = {}
    for (entity, year), (_, facet) in deflators.lookup.items():
        held = deflator_years.setdefault(entity, set())
        if facet in converted:
            held.add(year)
    return RecipeTables(
        rate_years=MappingProxyType({code: frozenset(ys) for code, ys in rate_years.items()}),
        deflator_years=MappingProxyType(
            {entity: frozenset(ys) for entity, ys in deflator_years.items()}
        ),
        base_year_min=min(years),
        base_year_max=max(years),
    )


def _national_currencies(reader: GraphReader, deadline: float | None) -> dict[str, str]:
    """Each place's one national currency, by ISO code: the `CurrencyUnit` nodes'
    `nationalCurrencyOf` arcs, inverted. A place two currencies name has none, as
    `materialise.recipes.resolve_entity_currencies` leaves it unmapped."""
    units = _subjects(reader, _CURRENCY_UNIT, "typeOf", deadline)
    # `typeOf` keeps every node from coming back with no arcs, which the read would retry.
    arcs = reader.node_arcs_many(
        list(units), properties=(_NATIONAL_CURRENCY_OF, "typeOf"), deadline=deadline
    )
    named: dict[str, set[str]] = {}
    for unit in units:
        code = native_currency_iso(unit)
        if code is None:
            continue
        for place in (arcs.get(unit) or {}).get(_NATIONAL_CURRENCY_OF, ()):
            named.setdefault(place, set()).add(code)
    return {place: next(iter(codes)) for place, codes in named.items() if len(codes) == 1}


_UNBOUND = object()
"""The scope of code no request scope wraps, as in tests that bind no catalogue."""

_BOUND: ContextVar[ConversionCatalogue | None | object] = ContextVar(
    "conversion_catalogue", default=_UNBOUND
)


@contextmanager
def bound_conversion_catalogue(
    catalogue: ConversionCatalogue | None,
) -> Iterator[ConversionCatalogue | None]:
    """Binds `catalogue` for the body of the `with` block; None binds a scope whose service
    has loaded no catalogue."""
    token = _BOUND.set(catalogue)
    try:
        yield catalogue
    finally:
        _BOUND.reset(token)


def current_conversion_catalogue() -> ConversionCatalogue | None:
    """The catalogue bound to this request, None where none is bound or none loaded."""
    bound = _BOUND.get()
    return bound if isinstance(bound, ConversionCatalogue) else None


def conversion_catalogue_unavailable() -> bool:
    """Whether this request's scope bound no catalogue because its service has loaded none:
    the load is still running or has failed. False outside any request scope."""
    return _BOUND.get() is None


_RETRY_DELAYS_S: tuple[float, ...] = (5.0, 15.0, 60.0, 300.0)
"""The waits before each retry of a failed load, the last repeating for every later retry."""


class ConversionCatalogueService:
    """The process's loaded `ConversionCatalogue`.

    `start` keeps a loader and requests a load; `reload_in_background` requests another, as
    a graph generation change does. One daemon thread runs the loads in order: a request
    made while a load runs sets a pending flag, and the thread runs one more load once the
    current one ends, so loads run one at a time and the last load requested is the last to
    publish. A failed load logs an ERROR, keeps the catalogue already loaded, or none, and
    is retried after `retry_delays_s` (then every last delay), or at once when a new request
    arrives, until a load succeeds. `wait_first_load` blocks until the first load ends, and
    `close` stops the thread after its current load.
    """

    def __init__(self, *, retry_delays_s: Sequence[float] = _RETRY_DELAYS_S) -> None:
        if not retry_delays_s:
            raise ValueError("retry_delays_s needs at least one delay")
        self._retry_delays_s = tuple(retry_delays_s)
        self._state = threading.Condition()
        self._attempted = threading.Event()
        self._catalogue: ConversionCatalogue | None = None
        self._loader: Callable[[], ConversionCatalogue] | None = None
        self._pending = False
        self._running = False
        self._closed = False

    def snapshot(self) -> ConversionCatalogue | None:
        return self._catalogue

    def stats(self) -> Mapping[str, int]:
        return {"loaded": int(self._catalogue is not None)}

    def start(self, loader: Callable[[], ConversionCatalogue]) -> None:
        """Keep `loader` for every later load and request the first."""
        with self._state:
            self._loader = loader
        self.reload_in_background()

    def reload_in_background(self) -> None:
        """Request a load with the kept loader. It runs on the loading thread, at once when
        no load runs, else after the running one; a retry waiting out its delay runs now."""
        with self._state:
            if self._loader is None or self._closed:
                return
            self._pending = True
            self._state.notify_all()
            if self._running:
                return
            self._running = True
        threading.Thread(target=self._run, name="conversion-catalogue", daemon=True).start()

    def wait_first_load(self, timeout_s: float) -> bool:
        """Wait at most `timeout_s` for the first load to succeed or fail, and return
        whether a catalogue is loaded."""
        self._attempted.wait(timeout_s)
        return self._catalogue is not None

    def close(self) -> None:
        """Stop the loading thread once its current load ends; later requests do nothing."""
        with self._state:
            self._closed = True
            self._state.notify_all()

    def _run(self) -> None:
        failures = 0
        while True:
            with self._state:
                if failures and not self._pending:
                    delay = self._retry_delays_s[min(failures, len(self._retry_delays_s)) - 1]
                    self._state.wait_for(lambda: self._pending or self._closed, timeout=delay)
                if self._closed or not (failures or self._pending):
                    self._running = False
                    return
                self._pending = False
                loader = self._loader
            assert loader is not None  # `reload_in_background` starts no thread without one
            failures = 0 if self._load(loader) else failures + 1
            self._attempted.set()

    def _load(self, loader: Callable[[], ConversionCatalogue]) -> bool:
        started = time.monotonic()
        try:
            catalogue = loader()
        except Exception:
            # A process boundary. Conversion offers are an enrichment, so a failed load
            # leaves answers without one and the app running.
            logger.exception(
                "conversion catalogue load failed; answers state unit_conversion_unavailable "
                "until a load succeeds, and the load is retried"
            )
            return False
        self._catalogue = catalogue
        logger.info(
            "conversion catalogue loaded in %.2f s: %d provenances, %d places",
            time.monotonic() - started,
            len(catalogue.recipes),
            len(catalogue.national_currencies),
        )
        return True
