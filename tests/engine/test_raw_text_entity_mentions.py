"""Source-grounded raw-text entity identities and member readings."""

from __future__ import annotations

import asyncio
import time
from typing import cast
from unittest.mock import patch

import pytest

from entity_resolution import (
    Admission,
    Ambiguous,
    Entity,
    EntityResolver,
    Group,
    NotFound,
    NotHeld,
    Resolution,
    Resolved,
)
from search.engine.binders import GeminiBinder
from search.engine.concurrency import scoped_concurrency
from search.engine.entry import raw_text
from search.engine.entry.params import EntryParams
from search.engine.entry.raw_text import (
    _entity_surfaces,
    _ExtractionPrep,
    _prepare_extraction,
    _recover_entity_surfaces,
)
from search.engine.events import InterpretationEvent
from search.engine.extract import (
    ExtractedEntity,
    Extraction,
    ExtractionResult,
)
from search.engine.surface_resolution import resolve_names
from search.models import RawTextInput
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver


class _Resolver:
    """Small resolver double that records the one batched lookup."""

    def __init__(self, answers: dict[str, object], *, names: tuple[str, ...] = ()) -> None:
        self.answers = answers
        self.names = names
        self.calls: list[list[str]] = []

    def is_exact_registered_name(self, text: str) -> bool:
        return text.casefold() in {name.casefold() for name in self.names}

    async def resolve(self, texts: list[str]) -> list[object]:
        self.calls.append(texts)
        return [self.answers[text] for text in texts]


_ASIA = Entity(dcid="asia", name="Asia", kind="continent", graph_type="Continent")
_DAC_ASIA = Entity(
    dcid="DACAsiaUnallocated",
    name="Asia, unallocated by country (DAC)",
    kind="aggregate",
    graph_type="StatisticalPopulation",
)


def _lookup(
    entities: list[ExtractedEntity], *, query: str, resolver: _Resolver
) -> dict[str, Resolution]:
    """Resolve the entities' names as recall does, through their grounded spans."""
    names = list(dict.fromkeys(entity.text.strip() for entity in entities))
    resolutions = asyncio.run(
        resolve_names(
            names,
            resolver=cast(EntityResolver, resolver),
            surfaces=_entity_surfaces(entities, query=query),
        )
    )
    return dict(zip(names, resolutions, strict=True))


def test_registered_parenthetical_label_recovers_the_full_source_surface() -> None:
    resolver = _Resolver({}, names=("Asia (DAC)",))
    entities = _recover_entity_surfaces(
        [ExtractedEntity(text="Asia", span="Asia")],
        query="France ODA to Asia (DAC)",
        resolver=cast(EntityResolver, resolver),
    )

    assert entities == [ExtractedEntity(text="Asia (DAC)", span="Asia (DAC)")]


def test_registered_full_span_promotes_the_qualified_canonical_identity() -> None:
    resolver = _Resolver({}, names=("Asia (DAC)",))
    entities = _recover_entity_surfaces(
        [ExtractedEntity(text="Asia", span="Asia (DAC)")],
        query="France ODA to Asia (DAC)",
        resolver=cast(EntityResolver, resolver),
    )

    assert entities == [ExtractedEntity(text="Asia (DAC)", span="Asia (DAC)")]


@pytest.mark.parametrize("span", ["Asie", "Asie (DAC)"])
def test_translated_span_carries_its_qualifier_onto_the_english_name(span: str) -> None:
    resolver = _Resolver({}, names=("Asia (DAC)",))
    entities = _recover_entity_surfaces(
        [ExtractedEntity(text="Asia", span=span)],
        query="aide de la France à l'Asie (DAC)",
        resolver=cast(EntityResolver, resolver),
    )

    assert entities == [ExtractedEntity(text="Asia (DAC)", span="Asie (DAC)")]


@pytest.mark.parametrize(
    ("query", "entities", "expected"),
    [
        pytest.param(
            "ODA from France to Africa (DAC) in 2016",
            [
                ExtractedEntity(text="France", span="France"),
                ExtractedEntity(text="Africa", span="Africa"),
                ExtractedEntity(text="DAC", span="DAC"),
            ],
            [
                ExtractedEntity(text="France", span="France"),
                ExtractedEntity(text="Africa (DAC)", span="Africa (DAC)"),
            ],
            id="qualifier_read_as_its_own_entity",
        ),
        pytest.param(
            "ODA from the DAC to Africa (DAC)",
            [
                ExtractedEntity(text="DAC", span="DAC"),
                ExtractedEntity(text="Africa", span="Africa"),
            ],
            [
                ExtractedEntity(text="DAC", span="DAC"),
                ExtractedEntity(text="Africa (DAC)", span="Africa (DAC)"),
            ],
            id="qualifier_also_named_on_its_own",
        ),
        pytest.param(
            "DAC2a gross ODA to Africa (DAC)",
            [
                ExtractedEntity(text="Africa", span="Africa"),
                ExtractedEntity(text="DAC", span="DAC"),
            ],
            [ExtractedEntity(text="Africa (DAC)", span="Africa (DAC)")],
            id="qualifier_inside_a_dataset_code",
        ),
    ],
)
def test_a_qualifier_inside_a_recovered_name_is_no_entity_of_its_own(
    query: str, entities: list[ExtractedEntity], expected: list[ExtractedEntity]
) -> None:
    resolver = _Resolver({}, names=("Africa (DAC)",))

    recovered = _recover_entity_surfaces(
        entities, query=query, resolver=cast(EntityResolver, resolver)
    )

    assert recovered == expected


def test_unknown_parenthetical_does_not_change_a_bare_place_identity() -> None:
    resolver = _Resolver({})
    entities = _recover_entity_surfaces(
        [ExtractedEntity(text="Asia", span="Asia")],
        query="aid to Asia (unknown grouping)",
        resolver=cast(EntityResolver, resolver),
    )

    assert entities == [ExtractedEntity(text="Asia", span="Asia")]


def test_explicit_qualified_and_unqualified_mentions_stay_distinct() -> None:
    resolver = _Resolver({}, names=("Asia (DAC)",))
    entities = _recover_entity_surfaces(
        [
            ExtractedEntity(text="Asia", span="Asia"),
            ExtractedEntity(text="Asia (DAC)", span="Asia (DAC)"),
        ],
        query="aid to Asia and Asia (DAC)",
        resolver=cast(EntityResolver, resolver),
    )

    assert [entity.span for entity in entities] == ["Asia", "Asia (DAC)"]


def test_lookup_uses_the_source_surface_before_english_fallback() -> None:
    resolver = _Resolver(
        {
            "Asia (DAC)": Resolved(entity=_DAC_ASIA),
            "Asia": Resolved(entity=_ASIA),
        }
    )

    found = _lookup(
        [ExtractedEntity(text="Asia (DAC)", span="Asia (DAC)")],
        query="aid to Asia (DAC)",
        resolver=resolver,
    )

    assert found == {"Asia (DAC)": Resolved(entity=_DAC_ASIA)}
    assert resolver.calls == [["Asia (DAC)"]]


def test_lookup_never_falls_back_from_a_qualified_surface_to_a_broader_label() -> None:
    resolver = _Resolver({"Asia (DAC)": NotFound(), "Asia": Resolved(entity=_ASIA)})

    found = _lookup(
        [ExtractedEntity(text="Asia (DAC)", span="Asia (DAC)")],
        query="aid to Asia (DAC)",
        resolver=resolver,
    )

    assert found == {"Asia (DAC)": NotFound()}
    assert resolver.calls == [["Asia (DAC)"]]


def test_lookup_keeps_an_unregistered_qualified_source_out_of_bare_fallback() -> None:
    resolver = _Resolver({"Asia (unknown)": NotFound(), "Asia": Resolved(entity=_ASIA)})

    found = _lookup(
        [ExtractedEntity(text="Asia", span="Asia (unknown)")],
        query="aid to Asia (unknown)",
        resolver=resolver,
    )

    assert found == {"Asia": NotFound()}
    assert resolver.calls == [["Asia (unknown)"]]


def test_lookup_keeps_a_registered_qualified_source_out_of_bare_fallback() -> None:
    resolver = _Resolver({"Asia (DAC)": NotFound(), "Asia": Resolved(entity=_ASIA)})

    found = _lookup(
        [ExtractedEntity(text="Asia", span="Asia (DAC)")],
        query="aid to Asia (DAC)",
        resolver=resolver,
    )

    assert found == {"Asia": NotFound()}
    assert resolver.calls == [["Asia (DAC)"]]


def test_lookup_falls_back_from_a_translated_qualified_source_to_its_qualified_english_name() -> (
    None
):
    resolver = _Resolver({"Asie (DAC)": NotFound(), "Asia (DAC)": Resolved(entity=_DAC_ASIA)})

    found = _lookup(
        [ExtractedEntity(text="Asia (DAC)", span="Asie (DAC)")],
        query="aide à l'Asie (DAC)",
        resolver=resolver,
    )

    assert found == {"Asia (DAC)": Resolved(entity=_DAC_ASIA)}
    assert resolver.calls == [["Asie (DAC)", "Asia (DAC)"]]


_COD = Entity(
    dcid="country/COD",
    name="Democratic Republic of the Congo",
    kind="country",
    graph_type="Country",
)
_COG = Entity(
    dcid="country/COG", name="Republic of the Congo", kind="country", graph_type="Country"
)
_KOR = Entity(dcid="country/KOR", name="South Korea", kind="country", graph_type="Country")
_PRK = Entity(dcid="country/PRK", name="North Korea", kind="country", graph_type="Country")
_CAN = Entity(dcid="country/CAN", name="Canada", kind="country", graph_type="Country")
_NAIROBI = Entity(dcid="wikidataId/Q3870", name="Nairobi", kind="place_other", graph_type="City")
_NAIROBI_COUNTY = Entity(
    dcid="wikidataId/Q3335223", name="Nairobi County", kind="place_other", graph_type="County"
)
# Two towns named Canadian, in Oklahoma and Texas, which the graph pack holds.
_CANADIAN_OK = Entity(dcid="geoId/4011450", name="Canadian", kind="place_other", graph_type="City")
_CANADIAN_TX = Entity(dcid="geoId/4812412", name="Canadian", kind="place_other", graph_type="City")


@pytest.mark.parametrize(
    ("query", "entity", "answers", "expected"),
    [
        pytest.param(
            "aide à l'Asie (DAC)",
            ExtractedEntity(text="Asia (DAC)", span="Asie (DAC)"),
            {
                "Asie (DAC)": Ambiguous(candidates=(_DAC_ASIA, _ASIA)),
                "Asia (DAC)": Resolved(entity=_DAC_ASIA),
            },
            Resolved(entity=_DAC_ASIA),
            id="qualified_translation_ambiguous",
        ),
        pytest.param(
            "aide à l'Asie (DAC)",
            ExtractedEntity(text="Asia (DAC)", span="Asie (DAC)"),
            {
                "Asie (DAC)": NotHeld(dcid="DACAsiaUnallocated", name="Asia (DAC)"),
                "Asia (DAC)": Resolved(entity=_DAC_ASIA),
            },
            Resolved(entity=_DAC_ASIA),
            id="qualified_translation_not_held",
        ),
        pytest.param(
            "aid to Congo",
            ExtractedEntity(text="Democratic Republic of the Congo", span="Congo"),
            {
                "Congo": Ambiguous(candidates=(_COD, _COG)),
                "Democratic Republic of the Congo": Resolved(entity=_COD),
            },
            Resolved(entity=_COD),
            id="congo",
        ),
        pytest.param(
            "aid to Korea",
            ExtractedEntity(text="South Korea", span="Korea"),
            {"Korea": Ambiguous(candidates=(_KOR, _PRK)), "South Korea": Resolved(entity=_KOR)},
            Resolved(entity=_KOR),
            id="korea",
        ),
        pytest.param(
            "how fast has Canadian aid to Haiti grown",
            ExtractedEntity(text="Canada", span="Canadian"),
            {
                "Canadian": Ambiguous(candidates=(_CANADIAN_OK, _CANADIAN_TX)),
                "Canada": Resolved(entity=_CAN),
            },
            Resolved(entity=_CAN),
            id="demonym_naming_towns",
        ),
        pytest.param(
            "population of Nairobi county",
            ExtractedEntity(text="Nairobi", span="Nairobi county"),
            {
                "Nairobi county": Resolved(entity=_NAIROBI_COUNTY),
                "Nairobi": Resolved(entity=_NAIROBI),
            },
            Resolved(entity=_NAIROBI_COUNTY),
            id="surface_adds_words_to_the_name",
        ),
    ],
)
def test_a_surface_and_its_name_settle_on_the_more_specific_text(
    query: str, entity: ExtractedEntity, answers: dict[str, object], expected: Resolution
) -> None:
    resolver = _Resolver(answers)

    found = _lookup([entity], query=query, resolver=resolver)

    assert found == {entity.text: expected}
    assert resolver.calls == [[entity.span, entity.text]]


@pytest.mark.parametrize(
    ("query", "entity", "registered", "answers", "expected"),
    [
        pytest.param(
            "ODA to Asia",
            ExtractedEntity(text="Asia", span="Asia (DAC)"),
            ("Asia (DAC)",),
            {"Asia": Resolved(entity=_ASIA)},
            {"Asia": Resolved(entity=_ASIA)},
            id="qualified_span_absent_from_query",
        ),
        pytest.param(
            "ODA to Kenya",
            ExtractedEntity(text="Kenya", span="Uganda"),
            (),
            {"Kenya": Resolved(entity=_COD)},
            {"Kenya": Resolved(entity=_COD)},
            id="other_place_span",
        ),
    ],
)
def test_an_ungrounded_span_neither_promotes_nor_resolves(
    query: str,
    entity: ExtractedEntity,
    registered: tuple[str, ...],
    answers: dict[str, object],
    expected: dict[str, Resolution],
) -> None:
    resolver = _Resolver(answers, names=registered)

    recovered = _recover_entity_surfaces(
        [entity], query=query, resolver=cast(EntityResolver, resolver)
    )
    found = _lookup(recovered, query=query, resolver=resolver)

    assert recovered == [entity]
    assert found == expected
    assert resolver.calls == [[entity.text]]


@pytest.mark.parametrize(
    "first",
    [
        Ambiguous(candidates=(_DAC_ASIA, _ASIA)),
        NotHeld(dcid="DACAsiaUnallocated", name="Asia (DAC)"),
    ],
    ids=["ambiguous", "not_held"],
)
def test_lookup_does_not_overwrite_a_non_missing_qualified_surface(first: object) -> None:
    resolver = _Resolver({"Asia (DAC)": first, "Asia": Resolved(entity=_ASIA)})

    found = _lookup(
        [ExtractedEntity(text="Asia (DAC)", span="Asia (DAC)")],
        query="aid to Asia (DAC)",
        resolver=resolver,
    )

    assert found == {"Asia (DAC)": first}
    assert resolver.calls == [["Asia (DAC)"]]


def test_lookup_falls_back_to_translated_english_only_after_not_found() -> None:
    resolver = _Resolver({"Asie": NotFound(), "Asia": Resolved(entity=_ASIA)})

    found = _lookup(
        [ExtractedEntity(text="Asia", span="Asie")],
        query="aide à l'Asie",
        resolver=resolver,
    )

    assert found == {"Asia": Resolved(entity=_ASIA)}
    assert resolver.calls == [["Asie", "Asia"]]


def test_adjective_and_translated_surfaces_resolve_source_first_under_canonical_names() -> None:
    resolver = _Resolver(
        {
            "French": NotFound(),
            "France": Resolved(
                entity=Entity(
                    dcid="country/FRA", name="France", kind="country", graph_type="Country"
                )
            ),
            "Asie": NotFound(),
            "Asia": Resolved(entity=_ASIA),
        }
    )
    entities = [
        ExtractedEntity(text="France", span="French"),
        ExtractedEntity(text="Asia", span="Asie"),
    ]

    found = _lookup(
        entities,
        query="French aid to Asie",
        resolver=resolver,
    )

    assert set(found) == {"France", "Asia"}
    assert resolver.calls == [["French", "France", "Asie", "Asia"]]


def test_preparation_keeps_canonical_keys_and_query_wording_separate() -> None:
    entity = ExtractedEntity(
        text="Asia",
        span="Asie",
        flow_role="destination",
        role_span="to",
    )
    extraction = Extraction(kind="resolution", variables=["aid"], entities=[entity])

    prep = _prepare_extraction(
        extraction,
        ExtractionResult(extraction=extraction, qualifiers={}),
        query="aid to Asie",
    )

    assert prep.entities == ["Asia"]
    assert prep.entity_words == {"Asia": "Asie"}
    assert list(prep.entity_roles) == ["Asia"]


def test_registered_graph_pack_name_recovers_the_qualified_source_surface() -> None:
    resolver = _Resolver({}, names=("World Bank (DAC)",))
    entities = _recover_entity_surfaces(
        [ExtractedEntity(text="World Bank", span="World Bank")],
        query="ODA from World Bank (DAC)",
        resolver=cast(EntityResolver, resolver),
    )

    assert entities == [ExtractedEntity(text="World Bank (DAC)", span="World Bank (DAC)")]


def test_qualified_graph_pack_name_preserves_an_ordinary_ambiguous_resolution() -> None:
    ambiguous = Ambiguous(candidates=(_ASIA, _DAC_ASIA))
    resolver = _Resolver({"World Bank (DAC)": ambiguous, "World Bank": Resolved(entity=_ASIA)})

    found = _lookup(
        [ExtractedEntity(text="World Bank (DAC)", span="World Bank (DAC)")],
        query="aid from World Bank (DAC)",
        resolver=resolver,
    )

    assert found == {"World Bank (DAC)": ambiguous}
    assert resolver.calls == [["World Bank (DAC)"]]


def _params() -> EntryParams:
    return EntryParams(
        deadline=time.monotonic() + 30,
        max_candidates=5,
        pac=True,
        stream_option_coverage=False,
        include_option_coverage=False,
        stream_observations=False,
        include_observations=False,
        include_skills=False,
        skills_index=None,
        coverage=empty_coverage_snapshot(),
        binder=GeminiBinder(),
    )


class _ExtractionLLM:
    def __init__(self, extraction: Extraction) -> None:
        self.extraction = extraction

    async def generate_structured(self, *, prompt, system, schema):
        assert schema is Extraction
        return self.extraction, None


async def _raw_pipeline_capture(
    query: str, extraction: Extraction
) -> tuple[InterpretationEvent, _ExtractionPrep, dict[str, Resolution]]:
    """Run the raw-text door up to its single-variable leg and resolve the prepared names
    as the leg's recall does."""
    captured: list[_ExtractionPrep] = []

    async def capture(prep: _ExtractionPrep, **kwargs: object):
        captured.append(prep)
        if False:
            yield None

    graph = FakeGraph()
    resolver = make_resolver(graph)
    with patch("search.engine.entry.raw_text._handle_single_variable", capture):
        async with scoped_concurrency():
            events = [
                event
                async for event in raw_text.handle(
                    RawTextInput(query=query),
                    graph=graph,
                    resolver=resolver,
                    admission=Admission(graph=graph),
                    llm=_ExtractionLLM(extraction),
                    sdmx=FakeSdmxClient(observations={}, availability={}),
                    materialise_graph=None,
                    materialise_sdmx=None,
                    start_ms=0,
                    params=_params(),
                )
            ]
        interpretation = next(event for event in events if isinstance(event, InterpretationEvent))
        (prep,) = captured
        resolutions = await resolve_names(
            prep.entities, resolver=resolver, surfaces=prep.entity_surfaces
        )
    return interpretation, prep, dict(zip(prep.entities, resolutions, strict=True))


def test_raw_pipeline_recovers_the_dac_group_before_validating_its_members() -> None:
    extraction = Extraction(
        kind="resolution",
        variables=["ODA"],
        entities=[
            ExtractedEntity(text="France", span="France", flow_role="origin", role_span="from"),
            ExtractedEntity(
                text="Asia",
                span="Asia (DAC)",
                flow_role="destination",
                role_span="to",
                members=True,
            ),
        ],
    )

    interpretation, prep, pre_resolved = asyncio.run(
        _raw_pipeline_capture("ODA from France to Asia (DAC) countries", extraction)
    )

    assert interpretation.entities == ["France", "Asia (DAC)"]
    assert prep.member_entities == frozenset({"Asia (DAC)"})
    assert set(prep.entity_roles) == {"France", "Asia (DAC)"}
    assert prep.entity_words == {"France": "France", "Asia (DAC)": "Asia (DAC)"}
    assert isinstance(pre_resolved["Asia (DAC)"], Group)


@pytest.mark.parametrize(
    "name",
    [
        "World Bank (DAC)",
        # A registry group the membership policy answers by the region `africa`.
        "Africa (DAC)",
    ],
)
def test_real_registry_recognizes_a_qualified_registered_name(name: str) -> None:
    resolver = make_resolver(FakeGraph())

    assert resolver.is_exact_registered_name(name)
    assert not resolver.is_exact_registered_name("World Bank (unregistered)")


def test_raw_pipeline_reads_a_translated_qualified_span_as_its_english_group() -> None:
    extraction = Extraction(
        kind="resolution",
        variables=["APD"],
        entities=[
            ExtractedEntity(text="France", span="France", flow_role="origin", role_span="de"),
            ExtractedEntity(text="Asia", span="Asie (DAC)", flow_role="destination", role_span="à"),
        ],
    )

    interpretation, prep, pre_resolved = asyncio.run(
        _raw_pipeline_capture("APD de la France à l'Asie (DAC)", extraction)
    )

    assert interpretation.entities == ["France", "Asia (DAC)"]
    assert prep.entity_words["Asia (DAC)"] == "Asie (DAC)"
    assert isinstance(pre_resolved["Asia (DAC)"], Group)


def test_raw_pipeline_keeps_a_qualified_pack_name_out_of_source_scan() -> None:
    extraction = Extraction(
        kind="resolution",
        variables=["ODA"],
        entities=[
            ExtractedEntity(
                text="World Bank",
                span="World Bank",
                flow_role="origin",
                role_span="from",
            )
        ],
    )

    interpretation, prep, pre_resolved = asyncio.run(
        _raw_pipeline_capture("ODA from World Bank (DAC)", extraction)
    )

    assert interpretation.entities == ["World Bank (DAC)"]
    assert prep.entities == ["World Bank (DAC)"]
    assert prep.entity_words == {"World Bank (DAC)": "World Bank (DAC)"}
    assert set(pre_resolved) == {"World Bank (DAC)"}


def test_raw_pipeline_keeps_a_members_reading() -> None:
    extraction = Extraction(
        kind="resolution",
        variables=["health spending"],
        entities=[ExtractedEntity(text="Africa", span="African", members=True)],
    )

    _, prep, _ = asyncio.run(
        _raw_pipeline_capture("health spending of African countries", extraction)
    )

    assert prep.member_entities == frozenset({"Africa"})
