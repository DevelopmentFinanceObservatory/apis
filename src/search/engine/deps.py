"""Dependency bundle built by the Search lifespan and read by routes through `SearchDepsDep`.

`search_deps` applies the same fallbacks to lifespan and test-provided values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, NamedTuple

from fastapi import Depends

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dfo_platform.state import StateSlot
from dfo_platform.static_response import FrozenJSON
from entity_resolution import Admission, EntityResolver
from graph_generation import UNKNOWN_GENERATION, GenerationProvider
from materialise import GraphReader
from materialise.facets import VariableBreadthReader
from materialise.rate_tables import RateTableCache
from search.catalogue.service import SearchCatalogueService
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.errors import EngineInfraError
from search.engine.options.options_response_cache import OptionsResponseCache
from search.engine.options.population_memo import OptionsPopulationMemo
from search.engine.region.place_options import PlaceOptionsPool
from search.engine.response_cache import ResponseCache
from search.engine.response_cache_memo import PartMemo
from skills_index import SharedSkillsIndex


class SearchDepsNotSet(EngineInfraError):
    """Retryable 503 `service_unavailable` when a request arrives before the Search lifespan sets
    `SEARCH_DEPS_SLOT`.
    """

    def __init__(self) -> None:
        super().__init__("Search dependencies are not set: the surface's lifespan has not run")


class SearchDocuments(NamedTuple):
    """The two process-scoped public documents, frozen at surface construction."""

    schema: FrozenJSON
    capabilities: FrozenJSON


@dataclass(frozen=True, slots=True)
class SearchDeps:
    """Optional dependencies are absent only where consumers define the outcome: observation
    previews fail open, the options and catalogue routes return 503, the resolver builds
    `Admission` locally, and missing caches, pools or skills index disable optimisation or
    enrichment.
    """

    graph: EngineGraphClient
    resolver: EntityResolver
    llm: SupportsLLM
    sdmx: EngineSdmxClient
    generation: GenerationProvider
    documents: SearchDocuments
    admission: Admission | None
    materialise_graph: GraphReader | None
    materialise_sdmx: MaterialiseSdmxClient | None
    rate_cache: RateTableCache | None
    skills_index: SharedSkillsIndex | None
    response_cache: ResponseCache | None
    part_memo: PartMemo | None
    options_response_cache: OptionsResponseCache | None
    population_memo: OptionsPopulationMemo | None
    place_options_pool: PlaceOptionsPool | None
    search_catalogue: SearchCatalogueService | None
    # Bound around each request, so an answer naming one place breaks a facet tie on each
    # facet's places across the whole variable. None leaves the tie to the unit's name.
    variable_breadth: VariableBreadthReader | None = None


def search_deps(
    *,
    graph: EngineGraphClient,
    resolver: EntityResolver,
    llm: SupportsLLM,
    sdmx: EngineSdmxClient,
    documents: SearchDocuments,
    live_graph: GraphReader | None = None,
    generation: GenerationProvider | None = None,
    admission: Admission | None = None,
    materialise_graph: GraphReader | None = None,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
    rate_cache: RateTableCache | None = None,
    skills_index: SharedSkillsIndex | None = None,
    response_cache: ResponseCache | None = None,
    part_memo: PartMemo | None = None,
    options_response_cache: OptionsResponseCache | None = None,
    population_memo: OptionsPopulationMemo | None = None,
    place_options_pool: PlaceOptionsPool | None = None,
    search_catalogue: SearchCatalogueService | None = None,
    variable_breadth: VariableBreadthReader | None = None,
) -> SearchDeps:
    """Assemble dependencies, using `live_graph` for `materialise_graph` and
    `UNKNOWN_GENERATION` for an unset generation provider. Other values pass through unchanged.
    """
    return SearchDeps(
        graph=graph,
        resolver=resolver,
        llm=llm,
        sdmx=sdmx,
        generation=generation if generation is not None else UNKNOWN_GENERATION,
        documents=documents,
        admission=admission,
        materialise_graph=materialise_graph if materialise_graph is not None else live_graph,
        materialise_sdmx=materialise_sdmx,
        rate_cache=rate_cache,
        skills_index=skills_index,
        response_cache=response_cache,
        part_memo=part_memo,
        options_response_cache=options_response_cache,
        population_memo=population_memo,
        place_options_pool=place_options_pool,
        search_catalogue=search_catalogue,
        variable_breadth=variable_breadth,
    )


# Each surface uses a distinct `app.state` attribute when composed on one app.
SEARCH_DEPS_SLOT: StateSlot[SearchDeps] = StateSlot("search_deps", unset_exception=SearchDepsNotSet)

# The `Annotated` form keeps the call out of an argument default (ruff B008).
SearchDepsDep = Annotated[SearchDeps, Depends(SEARCH_DEPS_SLOT)]
