"""Build a Search app and dependency slot for tests without entering lifespan."""

from __future__ import annotations

from fastapi import FastAPI

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from entity_resolution import Admission, EntityResolver
from graph_generation import GenerationProvider
from materialise import GraphReader
from materialise.rate_tables import RateTableCache
from search.engine.app import create_app, search_documents
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.deps import SEARCH_DEPS_SLOT, search_deps
from search.engine.options.options_response_cache import OptionsResponseCache
from search.engine.options.population_memo import OptionsPopulationMemo
from search.engine.region.place_options import PlaceOptionsPool
from search.engine.response_cache import ResponseCache
from skills_index import SharedSkillsIndex
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM, make_resolver


def set_search_deps(
    app: FastAPI,
    *,
    graph: EngineGraphClient | None = None,
    resolver: EntityResolver | None = None,
    admission: Admission | None = None,
    llm: SupportsLLM | None = None,
    sdmx: EngineSdmxClient | None = None,
    materialise_graph: GraphReader | None = None,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
    rate_cache: RateTableCache | None = None,
    skills_index: SharedSkillsIndex | None = None,
    response_cache: ResponseCache | None = None,
    options_response_cache: OptionsResponseCache | None = None,
    population_memo: OptionsPopulationMemo | None = None,
    place_options_pool: PlaceOptionsPool | None = None,
    generation: GenerationProvider | None = None,
) -> None:
    if graph is None or resolver is None or llm is None or sdmx is None:
        return
    SEARCH_DEPS_SLOT.set(
        app,
        search_deps(
            graph=graph,
            resolver=resolver,
            llm=llm,
            sdmx=sdmx,
            documents=search_documents(),
            generation=generation,
            admission=admission,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=rate_cache,
            skills_index=skills_index,
            response_cache=response_cache,
            options_response_cache=options_response_cache,
            population_memo=population_memo,
            place_options_pool=place_options_pool,
        ),
    )


def search_app(
    *,
    graph: EngineGraphClient | None = None,
    resolver: EntityResolver | None = None,
    admission: Admission | None = None,
    llm: SupportsLLM | None = None,
    sdmx: EngineSdmxClient | None = None,
    materialise_graph: GraphReader | None = None,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
    rate_cache: RateTableCache | None = None,
    skills_index: SharedSkillsIndex | None = None,
    response_cache: ResponseCache | None = None,
    options_response_cache: OptionsResponseCache | None = None,
    population_memo: OptionsPopulationMemo | None = None,
    place_options_pool: PlaceOptionsPool | None = None,
    generation: GenerationProvider | None = None,
) -> FastAPI:
    app = create_app(
        graph=graph,
        resolver=resolver,
        admission=admission,
        llm=llm,
        sdmx=sdmx,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        rate_cache=rate_cache,
        skills_index=skills_index,
        generation=generation,
    )
    set_search_deps(
        app,
        graph=graph,
        resolver=resolver,
        admission=admission,
        llm=llm,
        sdmx=sdmx,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        rate_cache=rate_cache,
        skills_index=skills_index,
        response_cache=response_cache,
        options_response_cache=options_response_cache,
        population_memo=population_memo,
        place_options_pool=place_options_pool,
        generation=generation,
    )
    return app


def with_fake_search_deps(app: FastAPI) -> FastAPI:
    graph = FakeGraph()
    set_search_deps(
        app, graph=graph, resolver=make_resolver(graph), llm=FakeLLM(), sdmx=FakeSdmxClient()
    )
    return app
