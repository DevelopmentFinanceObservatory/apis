"""Tests the raw-text source scan over the committed dev catalogue build and the fixture
code index."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from code_index.shared import get_index
from entity_resolution import Admission
from materialise.roles import RoleAxis
from origin_catalogue import CatalogueSnapshot, bound_catalogue, current_catalogue
from origin_catalogue.artifact import load_catalogue
from scripts.stage_origin_catalogue_fixture import stage_fixture
from search.engine import config
from search.engine.binders import GeminiBinder
from search.engine.clients.graph import DetectedSV, Detection, settled_un
from search.engine.concurrency import scoped_concurrency
from search.engine.conjoin import combine_regions
from search.engine.entry import raw_text
from search.engine.envelope import now_ms
from search.engine.exclusion_read import (
    Exclusion,
    ExclusionRead,
    LeftOut,
    QueryReading,
    admit_exclusions,
)
from search.engine.extract import ExtractedEntity, Extraction, ExtractionResult
from search.engine.fanout import _FanOutResult, _run_variables_streaming
from search.engine.interpret import Detected
from search.engine.region import RegionResult, source_choice
from search.engine.region.codes import ORGANISATION_READ_AS_PARTY, ORGANISATION_READ_AS_SOURCE
from search.engine.region.phase_rank_bind import _role_bearing_outranks
from search.engine.region.result import AmbiguousEntity, RoleNotAdmittedEntity
from search.engine.region.source_choice import (
    ContestedOrganisation,
    OrganisationEvidence,
    OrganisationReading,
    OrganisationReadings,
    RecalledFamily,
    choose_reading,
    choose_readings,
    party_possible,
    read_as_party,
    read_as_source,
    read_organisations,
)
from search.engine.response_cache import ResponseCache
from search.engine.shape import ShapeDraft
from search.engine.source_code import SeriesCodeMembers, SourceCodeRefusal
from search.engine.source_scan import (
    find_organisations,
    scan_sources,
    source_phrases,
)
from search.engine.source_scope import (
    SourceDictionary,
    SourceRefusal,
    SourceScope,
    source_dictionary,
)
from search.heading.departure_fact import PartyReason
from search.models import GraphRef, NoDataReason, RawTextInput, SourceMention
from tests.engine._builders import conjoin_spec
from tests.engine._conjoin_fixtures import _ECON_SHAPE, _PERSON_SHAPE, _region
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_raw_text_handle_close_order import _params, _stub_region
from tests.engine.test_response_cache_flow import _make_scope
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_EVAL_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "origin_catalogue_eval"


@pytest.fixture(scope="module")
def catalogue(tmp_path_factory: pytest.TempPathFactory) -> CatalogueSnapshot:
    return load_catalogue(stage_fixture(_EVAL_FIXTURE_DIR, tmp_path_factory.mktemp("eval")))


@pytest.fixture(scope="module")
def dictionary(catalogue: CatalogueSnapshot) -> SourceDictionary:
    return source_dictionary(catalogue)


def _in(text: str) -> SourceMention:
    return SourceMention(text=text)


def _out(text: str) -> SourceMention:
    return SourceMention(text=text, polarity="exclude")


# Each row: the query, the extraction's variables and entity spans, and each variable's
# source mentions. Rows with an id after a corpus query carry that query's recorded
# extraction.
_SCANS = [
    # A table is a source wherever it stands.
    ("education aid to Kenya from the CRS", ["education aid"], ["Kenya", "CRS"], [[_in("CRS")]]),
    ("DAC2a ODA to Kenya", ["ODA"], ["Kenya"], [[_in("DAC2a")]]),
    (
        "GDP per capita current prices in Kenya from the April 2025 World Economic Outlook",
        ["GDP per capita"],
        ["Kenya"],
        [[_in("World Economic Outlook")]],
    ),
    # A short key matches only in capitals.
    ("who gives the most health aid to Kenya", ["health aid"], ["Kenya"], [[]]),
    # A hit inside a longer entity name is that name.
    (
        "aid to Kenya from Catholic Relief Services (CRS)",
        ["aid"],
        ["Kenya", "Catholic Relief Services (CRS)"],
        [[]],
    ),
    ("ODA from DAC countries", ["ODA"], ["DAC countries"], [[]]),
    # An organisation the entity pack names that nothing cues is read as the source, an
    # entity or not; the door also builds the reading with it as a party.
    (
        "Ghana debt service to the World Bank",
        ["debt service"],
        ["Ghana", "World Bank"],
        [[_in("World Bank")]],
    ),
    ("World Bank core contributions", ["World Bank core contributions"], [], [[_in("World Bank")]]),
    ("Ghana debt service to the World Bank", ["debt service"], ["Ghana"], [[_in("World Bank")]]),
    ("World Bank GNI of Denmark", ["World Bank GNI"], ["Denmark"], [[_in("World Bank")]]),
    ("which DAC donors met the 0.7% target in 2023", ["ODA as percentage of GNI"], ["DAC"], [[]]),
    (
        "OECD data on education aid to Kenya",
        ["education aid"],
        ["OECD", "Kenya"],
        [[_in("OECD")]],
    ),
    ("GDP of Kenya according to the IMF", ["GDP"], ["Kenya", "IMF"], [[_in("IMF")]]),
    ("Germany ODA per the DAC", ["ODA"], ["Germany", "DAC"], [[]]),
    ("ODA per DAC member", ["ODA"], ["DAC"], [[]]),
    ("ODA per the OECD member", ["ODA"], ["OECD"], [[]]),
    ("ODA to Kenya reported to the DAC", ["ODA"], ["Kenya", "DAC"], [[_in("DAC")]]),
    ("ODA from Germany as published by the DAC", ["ODA"], ["Germany", "DAC"], [[_in("DAC")]]),
    ("IMF GDP current prices Kenya", ["GDP"], ["Kenya"], [[_in("IMF")]]),
    # A whole measure, or words inside a longer code label, are the measure.
    ("adolescent birth rate in Bangladesh", ["adolescent birth rate"], ["Bangladesh"], [[]]),
    ("aid to UN peacekeeping from Norway", ["aid"], ["Norway"], [[]]),
    # Measure words name no table: "imputed multilateral" ODA is a DAC2a aid type. The
    # ONE table's own name does.
    (
        "Imputed multilateral ODA from Norway",
        ["imputed multilateral ODA"],
        ["Norway"],
        [[]],
    ),
    (
        "ONE Imputed Multilateral Sectors data for Kenya",
        ["Imputed Multilateral Sectors data"],
        ["Kenya"],
        [[_in("ONE Imputed Multilateral Sectors")]],
    ),
    # A source qualifies the measure holding its words, else the one it precedes, else the
    # nearest before it.
    (
        "CRS and imputed figures for education aid to Kenya",
        ["CRS education aid", "imputed education aid"],
        ["Kenya"],
        [[_in("CRS")], []],
    ),
    (
        "WEO inflation and population of Peru",
        ["inflation", "population"],
        ["Peru"],
        [[_in("WEO")], []],
    ),
    (
        "GDP from the WEO and population of Peru",
        ["GDP", "population"],
        ["Peru"],
        [[_in("WEO")], []],
    ),
    # One mention per wording and polarity.
    ("CRS aid to Kenya from the CRS", ["aid"], ["Kenya"], [[_in("CRS")]]),
]


@pytest.mark.parametrize(("query", "variables", "entity_spans", "mentions"), _SCANS)
def test_scan_finds_the_sources_a_query_names(
    dictionary: SourceDictionary,
    query: str,
    variables: list[str],
    entity_spans: list[str],
    mentions: list[list[SourceMention]],
):
    scanned = scan_sources(
        query,
        dictionary=dictionary,
        measures=variables,
        entity_spans=entity_spans,
        index=get_index(),
    )
    assert [list(found) for found in scanned.per_measure] == mentions


@pytest.mark.parametrize(
    ("query", "variables", "entity_spans", "source_entities"),
    [
        ("OECD data on education aid to Kenya", ["education aid"], ["OECD", "Kenya"], {"OECD"}),
        ("education aid to Kenya from the CRS", ["education aid"], ["Kenya", "CRS"], {"CRS"}),
        # The source reading reads a contested organisation as the source, an entity or not.
        (
            "Ghana debt service to the World Bank",
            ["debt service"],
            ["Ghana", "World Bank"],
            {"World Bank"},
        ),
        # The second "World Bank" belongs to "World Bank IDA", so the source covers the
        # entity's only occurrence.
        (
            "World Bank data on World Bank IDA disbursements to Kenya",
            ["IDA disbursements"],
            ["World Bank", "World Bank IDA", "Kenya"],
            {"World Bank"},
        ),
    ],
)
def test_an_entity_a_source_covers_is_no_entity(
    dictionary: SourceDictionary,
    query: str,
    variables: list[str],
    entity_spans: list[str],
    source_entities: set[str],
):
    """One span gets one reading: an entity read as a source leaves the part's entities."""
    scanned = scan_sources(
        query,
        dictionary=dictionary,
        measures=variables,
        entity_spans=entity_spans,
        index=get_index(),
    )
    assert scanned.source_entities == source_entities


def test_the_raw_text_door_reads_a_source_entity_as_the_source_alone(
    catalogue: CatalogueSnapshot,
):
    """ "OECD data on education aid to Kenya": the extraction makes OECD an entity, the scan
    a source. The part carries OECD as its source and Kenya as its only entity, so no role
    is asked of OECD."""
    query = "OECD data on education aid to Kenya"
    extraction = Extraction(
        kind="resolution",
        variables=["education aid"],
        entities=[
            ExtractedEntity(text="OECD", span="OECD"),
            ExtractedEntity(text="Kenya", span="Kenya", role="recipient", cue="to"),
        ],
    )
    with bound_catalogue(catalogue):
        prep = raw_text._prepare_extraction(
            extraction, ExtractionResult(extraction=extraction, qualifiers={}), query=query
        )

    assert prep.entities == ["Kenya"]
    assert set(prep.per_variable_entity_roles[0]) == {"Kenya"}
    assert prep.entity_words == {"Kenya": "Kenya"}
    assert prep.per_variable_sources == [(_in("OECD"),)]


_OWE_DEBT_AND_INTEREST = "what does Zambia owe the IMF in debt and interest"


@pytest.mark.parametrize(
    ("query", "words", "entity", "measure", "stated"),
    [
        # dq-ids-011: a role the extraction keeps states it, though the model misquoted its
        # cue ("owes" for "owe").
        pytest.param(
            "what does Zambia owe the IMF",
            "IMF",
            ExtractedEntity(text="IMF", span="IMF", role="creditor", cue="owes"),
            "debt",
            True,
            id="kept role, misquoted cue",
        ),
        # dq-mls-018: a cue the query holds states it with no role.
        pytest.param(
            "How much goes through the World Bank?",
            "World Bank",
            ExtractedEntity(text="World Bank", span="World Bank", cue="through"),
            "financial flows",
            True,
            id="held cue, no role",
        ),
        pytest.param(
            _OWE_DEBT_AND_INTEREST,
            "IMF",
            ExtractedEntity(text="IMF", span="IMF", cue="lends"),
            "debt",
            False,
            id="no role, cue not held",
        ),
        pytest.param(
            _OWE_DEBT_AND_INTEREST,
            "IMF",
            ExtractedEntity(text="IMF", span="IMF"),
            "debt",
            False,
            id="no role, no cue",
        ),
        # A cue that is the organisation's own name reads its role off the measure.
        pytest.param(
            _OWE_DEBT_AND_INTEREST,
            "IMF",
            ExtractedEntity(text="IMF", span="IMF", role="source", cue="IMF"),
            "debt",
            False,
            id="own name as cue",
        ),
        # An entity scoping only the other measure states no role in this part.
        pytest.param(
            _OWE_DEBT_AND_INTEREST,
            "IMF",
            ExtractedEntity(text="IMF", span="the IMF", role="creditor", cue="owe", variables=[1]),
            "debt",
            False,
            id="other part",
        ),
        pytest.param(
            _OWE_DEBT_AND_INTEREST,
            "IMF",
            ExtractedEntity(text="IMF", span="the IMF", role="creditor", cue="owe", variables=[1]),
            "interest",
            True,
            id="its own part",
        ),
    ],
)
def test_an_organisations_role_is_stated_by_a_kept_role_or_a_held_cue_in_its_part(
    query: str, words: str, entity: ExtractedEntity, measure: str, stated: bool
):
    """`query`'s extraction holds `entity` beside the measures "debt" and "interest", or
    "financial flows" alone, and is read for the part of `measure`."""
    variables = ["financial flows"] if measure == "financial flows" else ["debt", "interest"]
    extraction = Extraction(kind="resolution", variables=variables, entities=[entity])

    assert raw_text._role_stated(extraction, words, measure=measure, query=query) == stated


def test_the_party_reading_keeps_an_extracted_organisation_as_its_entity(
    catalogue: CatalogueSnapshot,
):
    """ "World Bank data on aid from the World Bank to Kenya": the first "World Bank" is the
    source of the numbers, and the extraction returned the second as the donor, which is
    contested. The source reading reads both as the source; the party reading keeps the
    extracted World Bank, with its origin role, and the first "World Bank" as its source."""
    query = "World Bank data on aid from the World Bank to Kenya"
    extraction = Extraction(
        kind="resolution",
        variables=["aid"],
        entities=[
            ExtractedEntity(text="World Bank", span="World Bank", role="source", cue="from"),
            ExtractedEntity(text="Kenya", span="Kenya", role="recipient", cue="to"),
        ],
    )
    with bound_catalogue(catalogue):
        prep = raw_text._prepare_extraction(
            extraction, ExtractionResult(extraction=extraction, qualifiers={}), query=query
        )

    assert prep.entities == ["Kenya"]
    assert prep.per_variable_sources == [(_in("World Bank"),)]
    assert prep.per_variable_party_readings is not None
    (party,) = prep.per_variable_party_readings
    assert party is not None
    assert [organisation.role_stated for organisation in party.contested] == [True]
    assert party.reading.entities == ["World Bank", "Kenya"]
    assert [role.axis for role in party.reading.entity_roles["World Bank"]] == ["origin"]
    scope = party.reading.source_scope
    assert isinstance(scope, SourceScope)
    assert scope.named


@pytest.mark.parametrize(
    ("query", "variables", "entity_spans", "contested"),
    [
        # Nothing cues the World Bank, an entity or not: it may be the creditor.
        ("Ghana debt service to the World Bank", ["debt service"], ["Ghana"], ["World Bank"]),
        (
            "Ghana debt service to the World Bank",
            ["debt service"],
            ["Ghana", "World Bank"],
            ["World Bank"],
        ),
        (
            "World Bank IDA concessional flows to Kenya",
            ["IDA concessional flows"],
            ["World Bank", "Kenya"],
            ["World Bank"],
        ),
        ("World Bank GNI of Denmark", ["World Bank GNI"], ["Denmark"], ["World Bank"]),
        ("IMF GDP current prices Kenya", ["GDP"], ["Kenya"], ["IMF"]),
        # A data word or a source cue makes it the source, and a table names no party.
        ("World Bank data on aid to Ghana", ["aid"], ["Ghana"], []),
        ("GDP of Kenya according to the IMF", ["GDP"], ["Kenya"], []),
        ("education aid to Kenya from the CRS", ["education aid"], ["Kenya"], []),
        # The OECD is a source the entity pack does not name, so it is no party here.
        ("OECD education aid to Kenya", ["education aid"], ["Kenya"], []),
    ],
)
def test_an_uncued_organisation_the_pack_names_is_contested(
    dictionary: SourceDictionary,
    query: str,
    variables: list[str],
    entity_spans: list[str],
    contested: list[str],
):
    scanned = scan_sources(
        query,
        dictionary=dictionary,
        measures=variables,
        entity_spans=entity_spans,
        index=get_index(),
    )
    assert [hit.text for hit in scanned.hits if hit.contested] == contested


@pytest.mark.parametrize(
    ("query", "variables", "entity_spans", "found"),
    [
        (
            "Global Fund disbursements to Kenya",
            ["Global Fund disbursements"],
            ["Kenya"],
            [("Global Fund", (0,))],
        ),
        # A short name matches only in capitals; "who" is a word.
        ("who funds GAVI in Kenya", ["funds"], ["Kenya"], [("GAVI", (0,))]),
        ("WHO funding to Kenya and GDP of Kenya", ["funding", "GDP"], ["Kenya"], []),
        # A source key, an entity's words and a longer entity are not recovered.
        ("Ghana debt service to the World Bank", ["debt service"], ["Ghana"], []),
        ("aid from the Global Fund to Malawi", ["aid"], ["Global Fund", "Malawi"], []),
        (
            "World Bank IDA concessional flows to Kenya",
            ["concessional flows"],
            ["World Bank IDA", "Kenya"],
            [],
        ),
        # A name is written as one: capitalised at both ends, or an acronym in capitals. The
        # pack's generic aliases ("international finance", "multilateral fund") name nothing
        # in lower case.
        ("international finance flows to Kenya", ["international finance flows"], ["Kenya"], []),
        (
            "multilateral fund disbursements to Ghana",
            ["multilateral fund disbursements"],
            ["Ghana"],
            [],
        ),
        (
            "International Finance Corporation investments in Kenya",
            ["investments"],
            ["Kenya"],
            [("International Finance Corporation", (0,))],
        ),
        ("IFC investments in Kenya", ["investments"], ["Kenya"], [("IFC", (0,))]),
        # A short name written as the pack stores it reads as one.
        ("Gavi disbursements to Kenya", ["disbursements"], ["Kenya"], [("Gavi", (0,))]),
        ("AfDB loans to Ghana", ["loans"], ["Ghana"], [("AfDB", (0,))]),
        ("IsDB financing to Senegal", ["financing"], ["Senegal"], [("IsDB", (0,))]),
        # A capital that only opens the sentence names nothing: no name of WHO holds "Who".
        ("Who funds GAVI in Kenya", ["funds"], ["Kenya"], [("GAVI", (0,))]),
        # Each attaches to the measures a source would: here the nearest before it.
        (
            "grants from GAVI to Kenya and GDP of Kenya",
            ["grants", "GDP"],
            ["Kenya"],
            [("GAVI", (0,))],
        ),
    ],
)
def test_find_organisations_reads_the_pack_names_no_source_or_entity_holds(
    dictionary: SourceDictionary,
    query: str,
    variables: list[str],
    entity_spans: list[str],
    found: list[tuple[str, tuple[int, ...]]],
):
    mentions = find_organisations(
        query,
        dictionary=dictionary,
        entity_spans=entity_spans,
        measures=variables,
        index=get_index(),
    )
    assert [(mention.text, mention.targets) for mention in mentions] == found


def _prep(query: str, extraction: Extraction, catalogue: CatalogueSnapshot) -> Any:
    with bound_catalogue(catalogue):
        return raw_text._prepare_extraction(
            extraction, ExtractionResult(extraction=extraction, qualifiers={}), query=query
        )


@pytest.mark.parametrize(
    ("entities", "role_stated"),
    [
        ([ExtractedEntity(text="Ghana", span="Ghana")], False),
        (
            [
                ExtractedEntity(text="Ghana", span="Ghana"),
                ExtractedEntity(
                    text="World Bank", span="the World Bank", role="creditor", cue="to"
                ),
            ],
            True,
        ),
        # Listed with no role and no cue, the World Bank states no role.
        (
            [
                ExtractedEntity(text="Ghana", span="Ghana"),
                ExtractedEntity(text="World Bank", span="World Bank"),
            ],
            False,
        ),
    ],
)
def test_the_raw_text_door_records_whether_the_extraction_states_the_organisation_s_role(
    catalogue: CatalogueSnapshot, entities: list[ExtractedEntity], role_stated: bool
):
    """ "Ghana debt service to the World Bank" contests the World Bank either way. Its contest
    records whether the extraction states the World Bank's role, which decides between two
    possible readings (`source_choice.choose_reading`)."""
    extraction = Extraction(kind="resolution", variables=["debt service"], entities=entities)
    prep = _prep("Ghana debt service to the World Bank", extraction, catalogue)

    (party,) = prep.per_variable_party_readings
    assert [(organisation.words, organisation.role_stated) for organisation in party.contested] == [
        ("World Bank", role_stated)
    ]


def test_the_raw_text_door_reads_an_uncued_world_bank_both_ways_and_a_cued_one_as_the_source(
    catalogue: CatalogueSnapshot,
):
    """ "Ghana debt service to the World Bank", with no entity for the World Bank, reads it
    as its source and holds a reading with the World Bank as an entity and no source, whose
    contest names IBRD and IDA and that the extraction states no role for it.
    "World Bank data on aid to Ghana" reads it as the source alone."""
    ghana = Extraction(
        kind="resolution",
        variables=["debt service"],
        entities=[ExtractedEntity(text="Ghana", span="Ghana")],
    )
    cued = Extraction(
        kind="resolution",
        variables=["aid"],
        entities=[ExtractedEntity(text="Ghana", span="Ghana", role="recipient", cue="to")],
    )

    contested = _prep("Ghana debt service to the World Bank", ghana, catalogue)
    source_only = _prep("World Bank data on aid to Ghana", cued, catalogue)

    assert contested.entities == ["Ghana"]
    assert contested.per_variable_sources == [(_in("World Bank"),)]
    (party,) = contested.per_variable_party_readings
    assert party.contested == (
        ContestedOrganisation(
            words="World Bank",
            parties=frozenset(
                {
                    "InternationalBankForReconstructionAndDevelopment",
                    "InternationalDevelopmentAssociation",
                }
            ),
            role_stated=False,
            mention=0,
            entity="World Bank",
        ),
    )
    assert party.reading.entities == ["Ghana", "World Bank"]
    assert party.reading.source_scope is None

    assert source_only.per_variable_sources == [(_in("World Bank"),)]
    assert source_only.per_variable_party_readings is None


def test_the_raw_text_door_reads_a_publisher_before_the_indicator_as_a_source_mention(
    catalogue: CatalogueSnapshot,
):
    """ "World Bank GNI of Denmark": the model leaves the World Bank in the measure. The
    part's source mention is the World Bank, and its party reading, which the part runs only
    where the data read the World Bank as a party, records that the extraction states no
    role for it."""
    extraction = Extraction(
        kind="resolution",
        variables=["World Bank GNI"],
        entities=[ExtractedEntity(text="Denmark", span="Denmark")],
    )
    prep = _prep("World Bank GNI of Denmark", extraction, catalogue)

    assert prep.per_variable_sources == [(_in("World Bank"),)]
    (party,) = prep.per_variable_party_readings
    assert [organisation.role_stated for organisation in party.contested] == [False]
    assert party.reading.entities == ["Denmark", "World Bank"]


def test_the_raw_text_door_recovers_the_global_fund_as_the_provider(
    catalogue: CatalogueSnapshot,
):
    """ "Global Fund disbursements to Kenya": the model leaves the Global Fund in the
    measure. The Global Fund is no source, so the door adds it as an entity, and it stands
    straight before the measure word, so it is the provider."""
    extraction = Extraction(
        kind="resolution",
        variables=["Global Fund disbursements"],
        entities=[ExtractedEntity(text="Kenya", span="Kenya", role="recipient", cue="to")],
    )
    prep = _prep("Global Fund disbursements to Kenya", extraction, catalogue)

    assert prep.entities == ["Kenya", "Global Fund"]
    assert [role.axis for role in prep.per_variable_entity_roles[0]["Global Fund"]] == ["origin"]
    assert prep.per_variable_sources == [()]
    assert prep.per_variable_party_readings is None


class _KeyingScope:
    """A cache scope that misses and keeps the layer-2 key a real scope builds from the
    lookup the door makes."""

    def __init__(self) -> None:
        self.key: str | None = None

    async def lookup_parsed(self, **kwargs):
        scope = _make_scope(ResponseCache(max_bytes=10_000_000, max_entry_bytes=10_000_000))
        await scope.lookup_parsed(**kwargs)
        self.key = scope._layer2_key
        await scope.release()
        return None

    def note_health(self, *, timed_out: bool, degraded_leg: bool) -> None:
        pass


def _layer2_key(query: str, extraction: Extraction, catalogue: CatalogueSnapshot) -> str | None:
    """The layer-2 key the raw-text door looks `query` up under, `extraction` its parse and
    no stretch of it left out."""
    scope = _KeyingScope()

    async def _extract(query, *, llm):
        return ExtractionResult(extraction=extraction, qualifiers={}), None

    async def _resolve_variable(variable, **kwargs):
        return _stub_region(variable)

    async def _combiner(regions, **kwargs):
        yield None

    async def _read_exclusions(query, *, llm):
        return ExclusionRead(exclusions=[]), None

    async def _run() -> None:
        graph = FakeGraph()
        async with scoped_concurrency():
            async for _ in raw_text.handle(
                RawTextInput(query=query),
                graph=graph,
                resolver=make_resolver(graph),
                admission=Admission(graph=graph),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                materialise_graph=None,
                materialise_sdmx=None,
                start_ms=0,
                params=_params(),
                scope=scope,  # ty: ignore[invalid-argument-type]
            ):
                pass

    with (
        bound_catalogue(catalogue),
        patch("search.engine.entry.raw_text.extract", _extract),
        patch("search.engine.entry.raw_text.read_exclusions", _read_exclusions),
        patch("search.engine.fanout.resolve_variable", _resolve_variable),
        patch("search.engine.entry.raw_text._stream_combined_regions", _combiner),
    ):
        asyncio.run(_run())
    return scope.key


def test_an_organisation_the_door_adds_with_no_role_keys_layer2_apart(
    catalogue: CatalogueSnapshot,
):
    """ "health aid and education aid with GAVI in Kenya": the door adds GAVI as an entity
    of both parts and grounds no role for it, so the parts' roles match the query without
    it, and a query of two parts keys on no query text. The layer-2 key holds the entities
    of the extraction the door prepared, so the two queries key apart."""
    extraction = Extraction(
        kind="resolution",
        variables=["health aid", "education aid"],
        entities=[ExtractedEntity(text="Kenya", span="Kenya")],
    )

    with_gavi = _layer2_key(
        "health aid and education aid with GAVI in Kenya", extraction, catalogue
    )
    without = _layer2_key("health aid and education aid in Kenya", extraction, catalogue)

    assert with_gavi is not None
    assert without is not None
    assert with_gavi != without


def test_a_lower_case_organisation_reaches_the_part_through_the_extraction(
    catalogue: CatalogueSnapshot, dictionary: SourceDictionary
):
    """ "global fund disbursements to kenya": a lower-case run names no organisation to the
    scan (`source_scan._names_an_organisation`), so the Global Fund reaches the part as the
    entity the extraction returns, once, and is no source."""
    query = "global fund disbursements to kenya"
    extraction = Extraction(
        kind="resolution",
        variables=["disbursements"],
        entities=[
            ExtractedEntity(text="global fund", span="global fund"),
            ExtractedEntity(text="kenya", span="kenya", role="recipient", cue="to"),
        ],
    )
    found = find_organisations(
        query,
        dictionary=dictionary,
        entity_spans=[],
        measures=extraction.variables,
        index=get_index(),
    )
    prep = _prep(query, extraction, catalogue)

    assert found == []
    assert prep.entities == ["global fund", "kenya"]
    assert prep.per_variable_sources == [()]
    assert prep.per_variable_party_readings is None


def _refusal(variable: str, reason: NoDataReason) -> RegionResult:
    return RegionResult(
        variable_text=variable,
        status="no_data",
        specs=(),
        no_data_reason=reason,
        warnings=(),
        timing_by_step={},
    )


def _stream_door(
    query: str,
    prep: Any,
    catalogue: CatalogueSnapshot,
    pipeline: Callable[..., Awaitable[RegionResult]],
    *,
    degrade_on_error: bool,
) -> list[RegionResult]:
    """The regions the raw-text fan-out answers `prep`'s parts with, `pipeline` standing in
    for `resolve_variable`."""
    graph = FakeGraph()

    async def _scenario() -> _FanOutResult:
        out = _FanOutResult()
        with bound_catalogue(catalogue), patch("search.engine.fanout.resolve_variable", pipeline):
            async for _event in _run_variables_streaming(
                prep.unique,
                out=out,
                per_variable_entities=prep.per_variable_entities,
                per_variable_entity_roles=prep.per_variable_entity_roles,
                per_variable_source_scopes=prep.per_variable_source_scopes,
                per_variable_party_readings=prep.per_variable_party_readings,
                entities=prep.entities,
                date_request=None,
                query=query,
                pac=False,
                graph=graph,
                resolver=make_resolver(graph),
                admission=Admission(graph=graph),
                llm=FakeLLM(responses={}),
                binder=GeminiBinder(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                base_timing={},
                pre_resolved=None,
                deadline=time.monotonic() + 3600,
                use_semaphore=False,
                coverage=empty_coverage_snapshot(),
                degrade_on_error=degrade_on_error,
                door="raw_text",
            ):
                pass
        return out

    return asyncio.run(_scenario()).regions


_IDS = "provenance/WorldBankInternationalDebtStatistics"
_WDI = "provenance/WorldBankWorldDevelopmentIndicators"
_CRS = "provenance/OECDCreditorReportingSystem"
_DAC2A = "provenance/OECDDAC2A"
_DAC1 = "provenance/OECDDAC1"
_WEO = "provenance/IMFWorldEconomicOutlook"
_FLOW: frozenset[RoleAxis] = frozenset({"origin", "destination"})
_DONOR: frozenset[RoleAxis] = frozenset({"origin"})
# The World Bank's tables a source reading of it is scoped to.
_WORLD_BANK_TABLES = frozenset({_IDS, _WDI})
# Where the index publishes IBRD and IDA as parties.
_WORLD_BANK_PARTIES: frozenset[tuple[str, RoleAxis]] = frozenset(
    {
        (_IDS, "origin"),
        (_CRS, "origin"),
        (_DAC2A, "origin"),
        (_DAC2A, "destination"),
    }
)
_IDS_DEBT = RecalledFamily(datasets=frozenset({_IDS}), axes=_FLOW)
_IDS_DEBT_STOCK = RecalledFamily(datasets=frozenset({_IDS}), axes=frozenset[RoleAxis]({"origin"}))
_WDI_SERIES = RecalledFamily(datasets=frozenset({_WDI}), axes=frozenset())
_DAC2A_FLOW = RecalledFamily(datasets=frozenset({_DAC2A}), axes=_FLOW)
_CRS_FLOW = RecalledFamily(datasets=frozenset({_CRS}), axes=_FLOW)
_WEO_GDP = RecalledFamily(datasets=frozenset({_WEO}), axes=frozenset())
_IDS_BY_DEBTOR = RecalledFamily(
    datasets=frozenset({_IDS}), axes=frozenset[RoleAxis]({"destination"})
)
# An IDS row the fixture index lists, with a creditor and a debtor, and a GDP series it
# does not list.
_IDS_BY_CREDITOR = "Amount_FinancialFlow_Buyback_PublicOrPubliclyGuaranteed_ByCreditor_Nominal"
_GDP_SERIES = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"


@pytest.mark.parametrize(
    (
        "measure",
        "scoped",
        "allowed",
        "role_stated",
        "role",
        "source_possible",
        "answered",
        "discloses",
    ),
    [
        # IDS debt service publishes IBRD and IDA as creditors and the World Bank's tables
        # hold it too, so both readings are possible. The extraction states the World Bank's
        # role: the party reading refuses entity_ambiguous.
        pytest.param(
            [_IDS_DEBT],
            [_IDS_DEBT],
            _WORLD_BANK_TABLES,
            True,
            "party",
            True,
            None,
            ORGANISATION_READ_AS_PARTY,
            id="Ghana debt service to the World Bank",
        ),
        # Both readings are possible and the extraction states the World Bank's role by the
        # cue "through", which names no direction.
        pytest.param(
            [_IDS_DEBT],
            [_IDS_DEBT],
            _WORLD_BANK_TABLES,
            True,
            "party",
            True,
            None,
            ORGANISATION_READ_AS_PARTY,
            id="How much goes through the World Bank",
        ),
        # GNI families are subject-only, so the World Bank is no party there, whatever the
        # extraction states.
        pytest.param(
            [_WDI_SERIES],
            [_WDI_SERIES],
            _WORLD_BANK_TABLES,
            False,
            "source",
            True,
            None,
            None,
            id="World Bank GNI of Denmark",
        ),
        pytest.param(
            [_WDI_SERIES],
            [_WDI_SERIES],
            _WORLD_BANK_TABLES,
            True,
            "source",
            True,
            None,
            None,
            id="World Bank GNI of Denmark, the World Bank an entity",
        ),
        # DAC2a publishes IBRD and IDA as recipients, and no World Bank table holds the
        # measure, so the party reading is the only one, whatever the extraction states.
        pytest.param(
            [_DAC2A_FLOW],
            [],
            _WORLD_BANK_TABLES,
            False,
            "party",
            False,
            None,
            ORGANISATION_READ_AS_PARTY,
            id="World Bank core contributions",
        ),
        pytest.param(
            [_DAC2A_FLOW],
            [_WDI_SERIES],
            _WORLD_BANK_TABLES,
            True,
            "party",
            True,
            None,
            ORGANISATION_READ_AS_PARTY,
            id="Norway contributions to the World Bank",
        ),
        # Both readings are possible and the extraction states no role for the World Bank:
        # the World Bank is the source.
        pytest.param(
            [_IDS_DEBT_STOCK],
            [_IDS_DEBT_STOCK],
            _WORLD_BANK_TABLES,
            False,
            "source",
            True,
            None,
            None,
            id="World Bank external debt owed to China",
        ),
        pytest.param(
            [_WDI_SERIES],
            [_WDI_SERIES],
            _WORLD_BANK_TABLES,
            False,
            "source",
            True,
            None,
            None,
            id="World Bank GDP of Taiwan",
        ),
        # The extraction states the IMF's role and recall's tail holds a debt family
        # by debtor, but the answer's GDP series has no flow axis: silent.
        pytest.param(
            [_WEO_GDP, _IDS_BY_DEBTOR],
            [_WEO_GDP],
            frozenset({_WEO}),
            True,
            "source",
            True,
            _GDP_SERIES,
            None,
            id="GDP of Kenya from the IMF",
        ),
        pytest.param(
            [_IDS_DEBT],
            [_IDS_DEBT],
            _WORLD_BANK_TABLES,
            True,
            "party",
            True,
            None,
            ORGANISATION_READ_AS_PARTY,
            id="Ghana debt service to Norway and the World Bank",
        ),
        # IDA is a CRS donor.
        pytest.param(
            [_CRS_FLOW],
            [_WDI_SERIES],
            _WORLD_BANK_TABLES,
            True,
            "party",
            True,
            None,
            ORGANISATION_READ_AS_PARTY,
            id="Kenya grants from the World Bank",
        ),
        # A creditor family recalled under the GDP family governs only from the top or at
        # SEARCH_UNKNOWN_SCORE_THRESHOLD (0.65), as the pass reads a flow family.
        pytest.param(
            [dataclasses.replace(_WEO_GDP, score=0.79), dataclasses.replace(_IDS_DEBT, score=0.61)],
            [_WEO_GDP],
            frozenset({_WEO}),
            True,
            "source",
            True,
            _GDP_SERIES,
            None,
            id="GDP over a creditor family at 0.61",
        ),
        pytest.param(
            [dataclasses.replace(_WEO_GDP, score=0.79), dataclasses.replace(_IDS_DEBT, score=0.70)],
            [_WEO_GDP],
            frozenset({_WEO}),
            True,
            "party",
            True,
            None,
            ORGANISATION_READ_AS_PARTY,
            id="GDP over a creditor family at 0.70",
        ),
        # DAC1 families lead "contributions" without the World Bank, and the DAC2a family
        # under them publishes it: the flow measure governs, so the party reading is possible.
        pytest.param(
            [
                RecalledFamily(datasets=frozenset({_DAC1}), axes=_DONOR, score=0.58),
                dataclasses.replace(_DAC2A_FLOW, score=0.52),
            ],
            [_WDI_SERIES],
            _WORLD_BANK_TABLES,
            True,
            "party",
            True,
            None,
            ORGANISATION_READ_AS_PARTY,
            id="Norway contributions to the World Bank under DAC1",
        ),
        # The scoped detection of "core contributions" leads with DAC1 families, and a
        # family the World Bank's tables publish trails at 0.53: no source reading.
        pytest.param(
            [
                RecalledFamily(datasets=frozenset({_DAC1}), axes=_DONOR, score=0.58),
                dataclasses.replace(_DAC2A_FLOW, score=0.52),
            ],
            [
                RecalledFamily(datasets=frozenset({_DAC1}), axes=_DONOR, score=0.58),
                dataclasses.replace(_WDI_SERIES, score=0.53),
            ],
            _WORLD_BANK_TABLES,
            False,
            "party",
            False,
            None,
            ORGANISATION_READ_AS_PARTY,
            id="World Bank core contributions under DAC1",
        ),
        # An organisation whose role the extraction states, read as the source, the only
        # reading the data allow, discloses itself where the answer's series has a flow
        # axis; one whose role it does not state is silent.
        pytest.param(
            [_WEO_GDP, RecalledFamily(datasets=frozenset({_WEO}), axes=_FLOW)],
            [_WDI_SERIES],
            _WORLD_BANK_TABLES,
            True,
            "source",
            True,
            _IDS_BY_CREDITOR,
            ORGANISATION_READ_AS_SOURCE,
            id="a flow measure that publishes no World Bank party",
        ),
        pytest.param(
            [_WEO_GDP, RecalledFamily(datasets=frozenset({_WEO}), axes=_FLOW)],
            [_WDI_SERIES],
            _WORLD_BANK_TABLES,
            False,
            "source",
            True,
            _IDS_BY_CREDITOR,
            None,
            id="a flow measure that publishes no World Bank party, no entity",
        ),
        # Neither reading is possible: the party reading refuses on the World Bank's words.
        pytest.param(
            [_WEO_GDP],
            [],
            frozenset(),
            False,
            "party",
            False,
            None,
            ORGANISATION_READ_AS_PARTY,
            id="neither reading possible",
        ),
    ],
)
def test_the_data_decide_whether_an_organisation_is_a_source_or_a_party(
    monkeypatch: pytest.MonkeyPatch,
    measure: list[RecalledFamily],
    scoped: list[RecalledFamily],
    allowed: frozenset[str],
    role_stated: bool,
    role: str,
    source_possible: bool,
    answered: str | None,
    discloses: str | None,
):
    """`answered` is the series the source reading's answer elected, if any. The rows'
    scores read against a governance threshold of 0.65."""
    monkeypatch.setattr(
        config, "_SETTINGS", config.settings().model_copy(update={"unknown_score_threshold": 0.65})
    )
    contested = dataclasses.replace(_WORLD_BANK, role_stated=role_stated)
    reading = choose_reading(
        measure=measure,
        scoped=scoped,
        allowed=allowed,
        published=_WORLD_BANK_PARTIES,
        role_stated=role_stated,
    )

    assert (reading.role, reading.source_possible) == (role, source_possible)
    if role == "party":
        answer = dataclasses.replace(
            _refusal("debt service", "entity_ambiguous"),
            ambiguous_entities=(AmbiguousEntity(text="World Bank", candidates=()),),
        )
        disclosed = read_as_party(answer, contested, reading)
    else:
        answer = dataclasses.replace(
            _refusal("debt service", "no_observations"),
            elected_stat_vars=[GraphRef(ref=answered, label=answered)] if answered else None,
        )
        disclosed = read_as_source(answer, contested)
    assert [warning.code for warning in disclosed.warnings] == ([discloses] if discloses else [])
    if discloses is None:
        return
    (fact,) = disclosed.departure_facts.values()
    assert (fact.code, fact.asked) == (discloses, "World Bank")
    if role == "party":
        # With no other source named, the extraction's entity chose the reading where the
        # source reading was possible, and the World Bank's own data leads nothing otherwise.
        assert reading.reason == ("named_as_party" if source_possible else "no_own_data")
        assert fact.party_reason == reading.reason


_DISBURSEMENTS = ([_CRS_FLOW], [_IDS_DEBT, _CRS_FLOW])


@pytest.mark.parametrize(
    ("named", "recalls", "role", "source_possible"),
    [
        # "CRS World Bank disbursements to Kenya": the CRS publishes IBRD and IDA as donors,
        # so naming it says whose data the part reads and the World Bank is a party in it,
        # though an IDS family, a World Bank table, leads the source reading's recall.
        pytest.param(
            {"CRS": frozenset({_CRS})},
            _DISBURSEMENTS,
            "party",
            False,
            id="CRS World Bank disbursements",
        ),
        # The WDI publishes neither as a party, so the source reading stands as it does
        # with no other source named.
        pytest.param(
            {"WDI": frozenset({_WDI})},
            _DISBURSEMENTS,
            "source",
            True,
            id="WDI and World Bank disbursements",
        ),
        pytest.param({}, _DISBURSEMENTS, "source", True, id="World Bank disbursements"),
        # No flow family governs GDP, so the World Bank can be no party there, and the CRS
        # listing it rules out nothing.
        pytest.param(
            {"CRS": frozenset({_CRS})},
            ([_WEO_GDP], [_WDI_SERIES]),
            "source",
            True,
            id="CRS World Bank GDP of Kenya",
        ),
    ],
)
def test_another_named_source_that_publishes_the_organisation_makes_it_a_party(
    named: dict[str, frozenset[str]],
    recalls: tuple[list[RecalledFamily], list[RecalledFamily]],
    role: str,
    source_possible: bool,
):
    """Each recall leads with a family that governs, and the extraction states no role for
    the World Bank."""
    measure, scoped = recalls
    reading = choose_reading(
        measure=measure,
        scoped=scoped,
        allowed=_WORLD_BANK_TABLES,
        named=named,
        published=_WORLD_BANK_PARTIES,
        role_stated=False,
    )

    assert (reading.role, reading.source_possible) == (role, source_possible)
    assert reading.listed_by == (("CRS",) if role == "party" else ())


# A table of the African Development Bank.
_AFDB = "provenance/AfDBStatistics"
# Where the index publishes the IMF as a party: an IDS creditor.
_IMF_PARTIES: frozenset[tuple[str, RoleAxis]] = frozenset({(_IDS, "origin")})


def _world_bank(*, role_stated: bool) -> OrganisationEvidence:
    return OrganisationEvidence(
        words="World Bank",
        role_stated=role_stated,
        allowed=_WORLD_BANK_TABLES,
        published=_WORLD_BANK_PARTIES,
    )


def _imf(
    *, role_stated: bool, published: frozenset[tuple[str, RoleAxis]] = _IMF_PARTIES
) -> OrganisationEvidence:
    return OrganisationEvidence(
        words="IMF", role_stated=role_stated, allowed=frozenset({_WEO}), published=published
    )


@pytest.mark.parametrize(
    ("organisations", "measure", "scoped", "readings"),
    [
        # The extraction states no role for the World Bank, and its IDS leads both
        # recalls, so it is the source. The IDS, the World Bank's table, lists the IMF as a
        # creditor, so the IMF is a party.
        pytest.param(
            [_world_bank(role_stated=False), _imf(role_stated=True)],
            [_IDS_DEBT],
            [_IDS_DEBT],
            [("source", None, ()), ("party", "named_source", ("World Bank",))],
            id="World Bank debt service from Kenya to the IMF",
        ),
        # The extraction states both roles, and the CRS publishes each as a donor.
        pytest.param(
            [
                _world_bank(role_stated=True),
                _imf(role_stated=True, published=frozenset({(_CRS, "origin")})),
            ],
            [_CRS_FLOW],
            [_WDI_SERIES],
            [("party", "named_as_party", ()), ("party", "no_own_data", ())],
            id="Kenya grants from the World Bank and the IMF",
        ),
        # Both are possible sources whose roles the extraction does not state, and each one's tables
        # list the other as a party: neither reading is more the question's, so both stay
        # sources.
        pytest.param(
            [
                dataclasses.replace(
                    _world_bank(role_stated=False),
                    published=_WORLD_BANK_PARTIES | {(_WEO, "origin")},
                ),
                _imf(role_stated=False),
            ],
            [_IDS_DEBT, RecalledFamily(datasets=frozenset({_WEO}), axes=_FLOW, score=0.9)],
            [_IDS_DEBT, RecalledFamily(datasets=frozenset({_WEO}), axes=_FLOW, score=0.9)],
            [("source", None, ()), ("source", None, ())],
            id="two sources listing each other",
        ),
        # GDP families have no flow, so the IMF can be no party, though the IDS, a World
        # Bank table, lists it as a creditor: both read as sources.
        pytest.param(
            [_world_bank(role_stated=False), _imf(role_stated=False)],
            [dataclasses.replace(_WDI_SERIES, score=0.9), dataclasses.replace(_WEO_GDP, score=0.9)],
            [dataclasses.replace(_WDI_SERIES, score=0.9), dataclasses.replace(_WEO_GDP, score=0.9)],
            [("source", None, ()), ("source", None, ())],
            id="World Bank and IMF GDP of Kenya",
        ),
        # The World Bank lists the IMF and the IMF lists the AfDB, each where a debt family
        # publishing both lenders governs. Only a source no other source lists names a
        # source for the others, so the IMF reads as a party and the AfDB stays a source.
        pytest.param(
            [
                _world_bank(role_stated=False),
                _imf(role_stated=False),
                OrganisationEvidence(
                    words="AfDB",
                    role_stated=False,
                    allowed=frozenset({_AFDB}),
                    published=frozenset({(_WEO, "origin")}),
                ),
            ],
            [RecalledFamily(datasets=frozenset({_IDS, _WEO}), axes=_FLOW, score=0.9)],
            [
                dataclasses.replace(_IDS_DEBT, score=0.9),
                RecalledFamily(datasets=frozenset({_WEO}), axes=_FLOW, score=0.9),
                RecalledFamily(datasets=frozenset({_AFDB}), axes=_FLOW, score=0.9),
            ],
            [
                ("source", None, ()),
                ("party", "named_source", ("World Bank",)),
                ("source", None, ()),
            ],
            id="a chain of listings",
        ),
    ],
)
def test_each_contested_organisation_takes_its_own_reading(
    monkeypatch: pytest.MonkeyPatch,
    organisations: list[OrganisationEvidence],
    measure: list[RecalledFamily],
    scoped: list[RecalledFamily],
    readings: list[tuple[str, PartyReason | None, tuple[str, ...]]],
):
    """Each row's readings give each organisation's role, reason and the sources listing
    it, read against a governance threshold of 0.65."""
    monkeypatch.setattr(
        config, "_SETTINGS", config.settings().model_copy(update={"unknown_score_threshold": 0.65})
    )
    chosen = choose_readings(organisations, measure=measure, scoped=scoped)

    assert [(r.role, r.reason, r.listed_by) for r in chosen] == readings


@pytest.mark.parametrize(
    ("hit", "role_stated", "role", "source_possible"),
    [
        # Outside the scope the IDS row seats IBRD and IDA, IDS creditors in the fixture
        # index, and inside it the World Bank's IDS publishes it, so the extraction decides.
        (_IDS_BY_CREDITOR, True, "party", True),
        (_IDS_BY_CREDITOR, False, "source", True),
        # No index row or catalogue entry records who publishes the series, so, as for
        # candidacy, any allowed provenance but a declared dataset may.
        ("Unrecorded_Series", False, "source", True),
        # A topic stands for no family.
        ("dc/topic/Unrecorded", False, "party", False),
    ],
)
def test_an_organisations_reading_reads_the_index_under_and_outside_its_scope(
    catalogue: CatalogueSnapshot,
    hit: str,
    role_stated: bool,
    role: str,
    source_possible: bool,
):
    """Each reading's pass text is detected under that pass's scope: the party reading's,
    unscoped and keeping the World Bank's words, and the source reading's, scoped to the
    World Bank's tables. Both recall `hit`."""
    detected: list[tuple[str, frozenset[str] | None]] = []
    type_of = "Topic" if hit.startswith("dc/topic/") else "StatisticalVariable"

    async def detect(_graph: Any, text: str, *, variable: str) -> Detected:
        detected.append((text, current_catalogue().scope))
        return _detection(
            text, DetectedSV(dcid=hit, score=0.9, type_of=type_of, sentence=None), variable=variable
        )

    async def read() -> OrganisationReadings:
        with bound_catalogue(catalogue), patch.object(source_choice, "detect_measure", detect):
            return await read_organisations(
                [dataclasses.replace(_WORLD_BANK, role_stated=role_stated)],
                party_text="World Bank debt buyback",
                party_variable="World Bank debt buyback",
                party_scope=None,
                source_text="debt buyback",
                source_variable="debt buyback",
                source_scope=_scope(("World Bank", _WORLD_BANK_TABLES)),
                graph=cast(Any, FakeGraph()),
            )

    readings = asyncio.run(read())

    (reading,) = readings.readings
    assert (reading.role, reading.source_possible) == (role, source_possible)
    # The readings carry their own pass's detection, which that pass recalls from.
    assert readings.detected is not None
    assert readings.detected.text == (
        "World Bank debt buyback" if role == "party" else "debt buyback"
    )
    assert sorted(detected, key=lambda pair: pair[1] is not None) == [
        ("World Bank debt buyback", None),
        ("debt buyback", _WORLD_BANK_TABLES),
    ]


# A CRS row the fixture index lists, with a donor and a recipient. IBRD and IDA are CRS donors.
_CRS_DISBURSEMENT = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode11330_Nominal"
)


def _detection(text: str, *hits: DetectedSV, variable: str | None = None) -> Detected:
    """`hits` as the graph's detection of `text` for `variable` (`text` where omitted), with
    no UN candidate."""
    return Detected(
        text=text,
        variable=text if variable is None else variable,
        detection=Detection(candidates=hits, un=settled_un(())),
    )


def _scope(*named: tuple[str, frozenset[str]]) -> SourceScope:
    """A scope including each of `named`, its words and the provenances it stands for."""
    return SourceScope(
        allowed=frozenset().union(*(tables for _, tables in named)),
        named=True,
        mentions=tuple(SourceMention(text=words, polarity="include") for words, _ in named),
        expansions=tuple(tables for _, tables in named),
    )


@pytest.mark.parametrize(
    ("source_scope", "party_scope", "hit", "role", "source_possible"),
    [
        # "CRS World Bank disbursements to Kenya": the CRS families lead the source reading's
        # recall but are no World Bank table, so the World Bank has no source reading, and
        # the CRS, where IDA is a donor, reads it as a party.
        pytest.param(
            _scope(("CRS", frozenset({_CRS})), ("World Bank", _WORLD_BANK_TABLES)),
            _scope(("CRS", frozenset({_CRS}))),
            _CRS_DISBURSEMENT,
            "party",
            False,
            id="CRS World Bank disbursements to Kenya",
        ),
        # An IDS creditor family, a World Bank table, leads both recalls, but the CRS the
        # part names publishes IBRD and IDA as donors, so the World Bank is a party in it.
        pytest.param(
            _scope(("CRS", frozenset({_CRS})), ("World Bank", _WORLD_BANK_TABLES)),
            _scope(("CRS", frozenset({_CRS}))),
            _IDS_BY_CREDITOR,
            "party",
            False,
            id="CRS World Bank disbursements with IDS leading",
        ),
        # A source reading whose scope refused runs no scoped search.
        pytest.param(
            SourceRefusal(
                reason="source_not_resolved", terms=("World Bank",), ambiguous=(), mentions=()
            ),
            None,
            _CRS_DISBURSEMENT,
            "party",
            False,
            id="a refused source scope",
        ),
    ],
)
def test_only_the_organisations_own_tables_make_its_source_reading_possible(
    catalogue: CatalogueSnapshot,
    source_scope: SourceScope | SourceRefusal,
    party_scope: SourceScope | None,
    hit: str,
    role: str,
    source_possible: bool,
):
    """Both recalls lead with `hit`; the extraction states no role for the World Bank."""
    detected: list[frozenset[str] | None] = []

    async def detect(_graph: Any, text: str, *, variable: str) -> Detected:
        detected.append(current_catalogue().scope)
        return _detection(
            text, DetectedSV(dcid=hit, score=0.8, type_of="StatisticalVariable", sentence=None)
        )

    async def read() -> OrganisationReadings:
        with bound_catalogue(catalogue), patch.object(source_choice, "detect_measure", detect):
            return await read_organisations(
                # The World Bank's mention follows the CRS's in the source reading's scope.
                [dataclasses.replace(_WORLD_BANK, role_stated=False, mention=1)],
                party_text="World Bank disbursements",
                party_variable="World Bank disbursements",
                party_scope=party_scope,
                source_text="disbursements",
                source_variable="disbursements",
                source_scope=source_scope,
                graph=cast(Any, FakeGraph()),
            )

    (reading,) = asyncio.run(read()).readings

    assert (reading.role, reading.source_possible) == (role, source_possible)
    assert reading.reason == (
        "source_unresolved" if isinstance(source_scope, SourceRefusal) else "named_source"
    )
    party_allowed = party_scope.allowed if party_scope is not None else None
    scoped = [source_scope.allowed] if isinstance(source_scope, SourceScope) else []
    assert sorted(detected, key=str) == sorted([party_allowed, *scoped], key=str)


@pytest.mark.parametrize(
    ("role", "role_stated", "calls", "answer", "warned"),
    [
        # The data read the World Bank as a party: the leg runs the party reading alone and
        # restates it.
        (
            "party",
            True,
            [(["Ghana", "World Bank"], False)],
            "entity_ambiguous",
            [ORGANISATION_READ_AS_PARTY],
        ),
        # A source reading of an organisation whose role the extraction states answers from
        # a flow series, so it discloses itself; one whose role it does not state is silent.
        (
            "source",
            True,
            [(["Ghana"], True)],
            "variable_not_resolved",
            [ORGANISATION_READ_AS_SOURCE],
        ),
        ("source", False, [(["Ghana"], True)], "variable_not_resolved", []),
    ],
)
def test_the_raw_text_door_runs_the_one_reading_the_data_choose(
    catalogue: CatalogueSnapshot,
    role: str,
    role_stated: bool,
    calls: list[tuple[list[str], bool]],
    answer: str,
    warned: list[str],
):
    """The leg decides its contested organisation's reading before any pass and runs that
    reading alone. The pipeline is a stub that refuses on the World Bank's words where it is
    an entity."""
    query = "Ghana debt service to the World Bank"
    extraction = Extraction(
        kind="resolution",
        variables=["debt service"],
        entities=[
            ExtractedEntity(text="Ghana", span="Ghana"),
            *(
                [ExtractedEntity(text="World Bank", span="World Bank", role="creditor", cue="to")]
                if role_stated
                else []
            ),
        ],
    )
    prep = _prep(query, extraction, catalogue)
    ran: list[tuple[list[str], bool]] = []

    async def _pipeline(variable: str, **kwargs: Any) -> RegionResult:
        ran.append((kwargs["entities"], kwargs["source_scope"] is not None))
        if "World Bank" in kwargs["entities"]:
            return dataclasses.replace(
                _refusal(variable, "entity_ambiguous"),
                ambiguous_entities=(AmbiguousEntity(text="World Bank", candidates=()),),
            )
        # The source reading elects a debt series by creditor and finds no data.
        return dataclasses.replace(
            _refusal(variable, "variable_not_resolved"),
            elected_stat_vars=[GraphRef(ref=_IDS_BY_CREDITOR, label=_IDS_BY_CREDITOR)],
        )

    async def _read(*_args: Any, **_kwargs: Any) -> OrganisationReadings:
        reading = OrganisationReading(role=cast(Any, role), source_possible=True)
        return OrganisationReadings(readings=(reading,), detected=None)

    with patch("search.engine.fanout.read_organisations", _read):
        (region,) = _stream_door(query, prep, catalogue, _pipeline, degrade_on_error=False)

    assert ran == calls
    assert region.no_data_reason == answer
    assert [warning.code for warning in region.warnings] == warned
    assert region.parsed_form is not None
    assert [entity.text for entity in region.parsed_form.entities or ()] == calls[0][0]


@pytest.mark.parametrize(
    ("query", "variable", "hit", "place", "extracted", "entities", "scoped", "handed", "warned"),
    [
        # The extraction states the World Bank's role and an IDS creditor family
        # publishes IBRD and IDA: the leg reads it as a party and its pass recalls from the
        # probe's party detection.
        pytest.param(
            "Ghana debt service to the World Bank",
            "debt service",
            _IDS_BY_CREDITOR,
            "Ghana",
            ("World Bank",),
            ["Ghana", "World Bank"],
            (),
            True,
            [("World Bank", ORGANISATION_READ_AS_PARTY)],
            id="Ghana debt service to the World Bank",
        ),
        # Each mention is read on its own evidence. The extraction states no role for the
        # World Bank, and its IDS leads the recall, so it is the source. The IDS lists the IMF
        # as a creditor, so it is a party. The mixed reading's pass detects its own text
        # after the probe's two detections.
        pytest.param(
            "World Bank debt service from Kenya to the IMF",
            "debt service",
            _IDS_BY_CREDITOR,
            "Kenya",
            ("IMF",),
            ["Kenya", "IMF"],
            ("World Bank",),
            False,
            [("IMF", ORGANISATION_READ_AS_PARTY)],
            id="World Bank debt service from Kenya to the IMF",
        ),
        # A series no index row lists recalls GDP, so neither organisation can be a party
        # and each one's tables may publish it: both read as sources, though the IDS, a World
        # Bank table, lists the IMF as a creditor.
        pytest.param(
            "World Bank and IMF GDP of Kenya",
            "GDP",
            "Unrecorded_Series",
            "Kenya",
            (),
            ["Kenya"],
            ("World Bank", "IMF"),
            True,
            [],
            id="World Bank and IMF GDP of Kenya",
        ),
    ],
)
def test_the_raw_text_door_reads_each_contested_organisation_on_its_own_evidence(
    catalogue: CatalogueSnapshot,
    query: str,
    variable: str,
    hit: str,
    place: str,
    extracted: tuple[str, ...],
    entities: list[str],
    scoped: tuple[str, ...],
    handed: bool,
    warned: list[tuple[str, str]],
):
    """Both probe detections recall `hit`. The extraction returns `place`, and the
    organisations `extracted` as creditors after "to". The pipeline is a stub that refuses on every
    organisation the reading holds as an entity. `scoped` names the source mentions whose
    tables scope the pass, and `handed` whether the pass recalls from a probe detection."""
    extraction = Extraction(
        kind="resolution",
        variables=[variable],
        entities=[
            ExtractedEntity(text=place, span=place),
            *(
                ExtractedEntity(text=words, span=words, role="creditor", cue="to")
                for words in extracted
            ),
        ],
    )
    prep = _prep(query, extraction, catalogue)
    (own_scope,) = prep.per_variable_source_scopes
    tables = {
        mention.text: expansion & own_scope.allowed
        for mention, expansion in zip(own_scope.mentions, own_scope.expansions, strict=True)
    }
    detections: list[str] = []
    ran: list[tuple[list[str], frozenset[str] | None, bool]] = []

    async def detect(_graph: Any, text: str, *, variable: str) -> Detected:
        detections.append(text)
        return _detection(
            text,
            DetectedSV(dcid=hit, score=0.9, type_of="StatisticalVariable", sentence=None),
        )

    async def _pipeline(variable: str, **kwargs: Any) -> RegionResult:
        scope = kwargs["source_scope"]
        ran.append(
            (kwargs["entities"], scope.allowed if scope else None, kwargs["detected"] is not None)
        )
        return dataclasses.replace(
            _refusal(variable, "entity_ambiguous"),
            ambiguous_entities=tuple(
                AmbiguousEntity(text=entity, candidates=())
                for entity in kwargs["entities"]
                if entity != place
            ),
        )

    with patch.object(source_choice, "detect_measure", detect):
        (region,) = _stream_door(query, prep, catalogue, _pipeline, degrade_on_error=False)

    expected = frozenset().union(*(tables[words] for words in scoped)) if scoped else None
    assert ran == [(entities, expected, handed)]
    assert len(detections) == 2
    assert [(fact.asked, fact.code) for fact in region.departure_facts.values()] == warned


@pytest.mark.parametrize("role", ["party", "source"])
def test_the_probe_detects_the_text_and_scope_the_chosen_pass_recalls(
    catalogue: CatalogueSnapshot, role: str
):
    """ "World Bank core contributions" with the World Bank left in the variable: the party
    reading recalls the variable whole and the source reading without the World Bank's
    words. The probe detects each reading's own text under its own scope, so the pass that
    runs takes the probe's detection as its recall's."""
    query = "World Bank core contributions"
    extraction = Extraction(
        kind="resolution", variables=["World Bank core contributions"], entities=[]
    )
    prep = _prep(query, extraction, catalogue)
    probed: dict[str, Any] = {}
    ran: list[tuple[str, object, object]] = []
    chosen = _detection("chosen")

    async def _pipeline(variable: str, **kwargs: Any) -> RegionResult:
        ran.append((kwargs["detect_query"], kwargs["source_scope"], kwargs["detected"]))
        return _refusal(variable, "variable_not_resolved")

    async def _read(_contested: Any, **kwargs: Any) -> OrganisationReadings:
        probed.update(kwargs)
        reading = OrganisationReading(role=cast(Any, role), source_possible=True)
        return OrganisationReadings(readings=(reading,), detected=chosen)

    with patch("search.engine.fanout.read_organisations", _read):
        _stream_door(query, prep, catalogue, _pipeline, degrade_on_error=False)

    assert probed["party_text"] == "World Bank core contributions"
    assert probed["source_text"] == "core contributions"
    # The pass recalls from the detection that chose it (`interpret.recall`).
    assert ran == [(probed[f"{role}_text"], probed[f"{role}_scope"], chosen)]


def test_each_reading_compiles_the_series_code_within_its_own_scope(
    catalogue: CatalogueSnapshot,
):
    """ "CRS DT.AMT.BLAT.CD to the World Bank": IDS alone holds the code. The source reading
    allows the World Bank's tables and reads the code's IDS series; the party reading allows
    the CRS alone, which holds no such code, so it refuses the code there."""
    query = "CRS DT.AMT.BLAT.CD to the World Bank"
    extraction = Extraction(kind="resolution", variables=["DT.AMT.BLAT.CD"], entities=[])
    prep = _prep(query, extraction, catalogue)

    (own,) = prep.per_variable_compiled_codes
    assert isinstance(own, SeriesCodeMembers)
    assert own.provenance == _IDS
    (party,) = prep.per_variable_party_readings
    assert isinstance(party.reading.source_code, SourceCodeRefusal)
    assert party.reading.source_code.reason == "source_code_not_held"


@pytest.mark.parametrize(
    ("reason", "listed_by", "message"),
    [
        (
            "named_as_party",
            (),
            "World Bank was read as a party to the flow, as the question names it, not as the "
            "source of the data.",
        ),
        (
            "named_source",
            ("CRS",),
            "World Bank was read as a party to the flow, since CRS lists it as a party.",
        ),
        (
            "source_unresolved",
            (),
            "World Bank was read as a party to the flow, since its words name no single "
            "source of data.",
        ),
        (
            "no_own_data",
            (),
            "World Bank was read as a party to the flow, since none of the data it publishes "
            "measures this.",
        ),
    ],
)
def test_a_party_reading_states_its_reason(
    reason: PartyReason, listed_by: tuple[str, ...], message: str
):
    reading = OrganisationReading(
        role="party",
        source_possible=reason == "named_as_party",
        reason=reason,
        listed_by=listed_by,
    )
    answer = dataclasses.replace(
        _refusal("disbursements", "entity_ambiguous"),
        ambiguous_entities=(AmbiguousEntity(text="World Bank", candidates=()),),
    )

    disclosed = read_as_party(answer, _WORLD_BANK, reading)

    (warning,) = disclosed.warnings
    assert warning.message == message
    (fact,) = disclosed.departure_facts.values()
    assert (fact.party_reason, fact.shown) == (reason, listed_by[0] if listed_by else None)


@pytest.mark.parametrize(
    ("query", "code", "role", "source_possible"),
    [
        # The IDS code names principal repayments by creditor, which the IDS, a World Bank
        # table, publishes with IBRD and IDA as creditors: both readings are possible and the
        # extraction states no role for the World Bank, so it is the source.
        ("World Bank DT.AMT.BLAT.CD Kenya", "DT.AMT.BLAT.CD", "source", True),
        # No index row holds the WDI code, so both readings refuse it and recall nothing.
        ("World Bank NY.GNP.PCAP.CD Kenya", "NY.GNP.PCAP.CD", "party", False),
    ],
)
def test_a_reading_with_a_series_code_is_judged_on_its_members_with_no_search(
    catalogue: CatalogueSnapshot, query: str, code: str, role: str, source_possible: bool
):
    """A reading whose code compiles recalls the code's StatVars and searches nothing, so the
    probe judges it on their families and makes no detection for it."""
    extraction = Extraction(kind="resolution", variables=[code], entities=[])
    prep = _prep(query, extraction, catalogue)
    (party,) = prep.per_variable_party_readings
    (own_code,) = prep.per_variable_compiled_codes
    (own_scope,) = prep.per_variable_source_scopes

    async def detect(_graph: Any, text: str, *, variable: str) -> Detected:
        raise AssertionError(f"a coded reading searched {text!r}")

    async def read() -> OrganisationReadings:
        with bound_catalogue(catalogue), patch.object(source_choice, "detect_measure", detect):
            return await read_organisations(
                party.contested,
                party_text="",
                party_variable="",
                party_scope=party.reading.source_scope,
                party_code=party.reading.source_code,
                source_text="",
                source_variable="",
                source_scope=own_scope,
                source_code=own_code,
                graph=cast(Any, FakeGraph()),
            )

    readings = asyncio.run(read())

    (reading,) = readings.readings
    assert (reading.role, reading.source_possible) == (role, source_possible)
    assert readings.detected is None


def test_a_part_with_no_contest_reads_once_with_no_probe(catalogue: CatalogueSnapshot):
    """A data word makes the World Bank the source, so the part holds no party reading and
    the leg makes no detection before its pass."""
    query = "World Bank data on aid to Ghana"
    extraction = Extraction(
        kind="resolution",
        variables=["aid"],
        entities=[ExtractedEntity(text="Ghana", span="Ghana")],
    )
    prep = _prep(query, extraction, catalogue)
    ran: list[list[str]] = []

    async def _pipeline(variable: str, **kwargs: Any) -> RegionResult:
        ran.append(kwargs["entities"])
        return _refusal(variable, "variable_not_resolved")

    async def _read(*_args: Any, **_kwargs: Any) -> OrganisationReadings:
        raise AssertionError("a part with no contest reads no organisation")

    with patch("search.engine.fanout.read_organisations", _read):
        (region,) = _stream_door(query, prep, catalogue, _pipeline, degrade_on_error=False)

    assert ran == [["Ghana"]]
    assert region.warnings == ()


_WORLD_BANK = ContestedOrganisation(
    words="World Bank",
    parties=frozenset(
        {"InternationalBankForReconstructionAndDevelopment", "InternationalDevelopmentAssociation"}
    ),
    role_stated=True,
    mention=0,
    entity="World Bank",
)


def _ida_spec() -> Any:
    return SimpleNamespace(
        definition=SimpleNamespace(
            identity=SimpleNamespace(
                entities={
                    "origin": ["InternationalDevelopmentAssociation"],
                    "destination": ["country/GHA"],
                }
            )
        )
    )


@pytest.mark.parametrize(
    ("answer", "discloses"),
    [
        # The party reading binds IDA, one of the World Bank's organisations.
        (
            dataclasses.replace(
                _refusal("debt service", "variable_not_resolved"), status="definite"
            ),
            True,
        ),
        # It refuses on the World Bank's own words.
        (
            dataclasses.replace(
                _refusal("debt service", "entity_ambiguous"),
                ambiguous_entities=(AmbiguousEntity(text="World Bank", candidates=()),),
            ),
            True,
        ),
        (
            dataclasses.replace(
                _refusal("debt service", "entity_not_resolved"),
                unresolved_entity_names=("World Bank",),
            ),
            True,
        ),
        # It refuses on something else: it says nothing of a party reading.
        (
            dataclasses.replace(
                _refusal("debt service", "entity_not_resolved"),
                unresolved_entity_names=("Taiwan",),
            ),
            False,
        ),
        (_refusal("debt service", "no_observations"), False),
    ],
)
def test_a_party_reading_discloses_itself_only_where_it_reads_the_organisation(
    answer: RegionResult, discloses: bool
):
    if answer.status == "definite":
        answer = dataclasses.replace(answer, specs=(_ida_spec(),))
    reading = OrganisationReading(role="party", source_possible=True)
    disclosed = read_as_party(answer, _WORLD_BANK, reading)
    codes = [warning.code for warning in disclosed.warnings]
    assert codes == ([ORGANISATION_READ_AS_PARTY] if discloses else [])


def _gdp_or_debt_shape(shape_id: str, *, roles: tuple[str, ...], score: float) -> ShapeDraft:
    """A shape the pass admitted, declaring `roles` beside its subject, recalled at
    `score`."""
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        slot_taxonomy={},
        declared_roles=("observationAbout", *roles),
        representative_score=score,
    )


def test_a_party_reading_the_pass_admits_no_flow_for_refuses_and_says_so(
    monkeypatch: pytest.MonkeyPatch, catalogue: CatalogueSnapshot
):
    """The pre-pass forecasts the pass (`source_choice.party_possible`): a debt family
    leading the recall at 0.60 governs it, so the World Bank, an entity, reads as a party. The
    pass admits only the GDP shape recalled at 0.58 and a debt shape at 0.50, which no
    longer governs, so no flow outranks the subject (`_role_bearing_outranks`). The part
    refuses `entity_role_not_admitted` on the World Bank and says it read it as a party."""
    monkeypatch.setattr(
        config, "_SETTINGS", config.settings().model_copy(update={"unknown_score_threshold": 0.65})
    )
    recall = [
        dataclasses.replace(_IDS_DEBT, score=0.60),
        dataclasses.replace(_WDI_SERIES, score=0.58),
        dataclasses.replace(_IDS_DEBT, score=0.50),
    ]
    admitted = [
        _gdp_or_debt_shape("gdp", roles=(), score=0.58),
        _gdp_or_debt_shape("debt", roles=("creditor",), score=0.50),
    ]
    reading = choose_reading(
        measure=recall,
        scoped=[_WDI_SERIES],
        allowed=_WORLD_BANK_TABLES,
        published=_WORLD_BANK_PARTIES,
        role_stated=True,
    )
    with bound_catalogue(catalogue):
        outranks = _role_bearing_outranks(admitted)
    refused = dataclasses.replace(
        _refusal("debt service", "entity_role_not_admitted"),
        role_not_admitted_entities=(
            RoleNotAdmittedEntity(
                text="World Bank",
                dcid="InternationalDevelopmentAssociation",
                graph_type="Organization",
            ),
        ),
    )

    disclosed = read_as_party(refused, _WORLD_BANK, reading)

    assert party_possible(recall, _WORLD_BANK_PARTIES)
    assert (reading.role, reading.reason) == ("party", "named_as_party")
    assert not outranks
    assert [warning.code for warning in disclosed.warnings] == [ORGANISATION_READ_AS_PARTY]


def _left_out(query: str, *stretches: str, things: Sequence[str] = ()) -> QueryReading:
    """`query` with each of `stretches` admitted, each leaving out `things`, else its words
    less its first."""
    return admit_exclusions(
        query,
        ExclusionRead(
            exclusions=[
                Exclusion(phrase=stretch, left_out=list(things) or stretch.split()[1:])
                for stretch in stretches
            ]
        ),
    )


# Each row: the query, the stretch it leaves out, the extraction's variables and entity spans
# of the blanked text, each variable's source mentions, and whether the stretch leaves out
# sources alone.
_STRETCHES = [
    (
        "aid to Kenya excluding the CRS",
        "excluding the CRS",
        ["aid"],
        ["Kenya"],
        [[_out("CRS")]],
        True,
    ),
    (
        "aid to Kenya, not from the CRS",
        "not from the CRS",
        ["aid"],
        ["Kenya"],
        [[_out("CRS")]],
        True,
    ),
    (
        "ODA to Kenya apart from DAC2a",
        "apart from DAC2a",
        ["ODA"],
        ["Kenya"],
        [[_out("DAC2a")]],
        True,
    ),
    ("AOD a Kenia sin el CRS", "sin el CRS", ["AOD"], ["Kenia"], [[_out("CRS")]], True),
    # A data word beside a source names nothing else.
    (
        "ODA to Kenya excluding CRS data",
        "excluding CRS data",
        ["ODA"],
        ["Kenya"],
        [[_out("CRS")]],
        True,
    ),
    (
        "GDP of Kenya excluding IMF figures",
        "excluding IMF figures",
        ["GDP"],
        ["Kenya"],
        [[_out("IMF")]],
        True,
    ),
    (
        "ODA to Kenya excluding the CRS database",
        "excluding the CRS database",
        ["ODA"],
        ["Kenya"],
        [[_out("CRS")]],
        True,
    ),
    # Every source of a list goes, and a source outside the stretch stays asked for.
    (
        "ODA to Kenya excluding CRS and DAC2a",
        "excluding CRS and DAC2a",
        ["ODA"],
        ["Kenya"],
        [[_out("CRS"), _out("DAC2a")]],
        True,
    ),
    (
        "ODA to Kenya, neither the CRS nor DAC2a",
        "neither the CRS nor DAC2a",
        ["ODA"],
        ["Kenya"],
        [[_out("CRS"), _out("DAC2a")]],
        True,
    ),
    (
        "debt of Kenya from IDS and not CRS",
        "not CRS",
        ["debt"],
        ["Kenya"],
        [[_in("IDS"), _out("CRS")]],
        True,
    ),
    (
        "ODA to Kenya excluding CRS and including DAC2a",
        "excluding CRS",
        ["ODA"],
        ["Kenya"],
        [[_out("CRS"), _in("DAC2a")]],
        True,
    ),
    (
        "ODA to Kenya without CRS, from DAC2a",
        "without CRS",
        ["ODA"],
        ["Kenya"],
        [[_out("CRS"), _in("DAC2a")]],
        True,
    ),
    # An item of the list naming no source leaves the stretch to its disclosure.
    (
        "aid to Kenya excluding the CRS, Norway and DAC2a",
        "excluding the CRS, Norway and DAC2a",
        ["aid"],
        ["Kenya"],
        [[_out("CRS"), _out("DAC2a")]],
        False,
    ),
    # A source word inside a longer thing left out is no source.
    (
        "external debt of Kenya without IMF credit",
        "without IMF credit",
        ["external debt"],
        ["Kenya"],
        [[]],
        False,
    ),
    ("aid to Kenya, not imputed", "not imputed", ["aid"], ["Kenya"], [[]], False),
    # A source attaches from the stretch's blank, to the nearest measure before it.
    (
        "GDP of Kenya excluding the WEO and IDS external debt of Kenya",
        "excluding the WEO",
        ["GDP", "external debt"],
        ["Kenya"],
        [[_out("WEO")], [_in("IDS")]],
        True,
    ),
]


@pytest.mark.parametrize(("query", "stretch", "measures", "spans", "mentions", "alone"), _STRETCHES)
def test_a_stretch_is_read_for_the_sources_it_leaves_out(
    dictionary: SourceDictionary, query, stretch, measures, spans, mentions, alone
):
    reading = _left_out(query, stretch)
    scan = scan_sources(
        reading.text,
        dictionary=dictionary,
        measures=measures,
        entity_spans=spans,
        index=get_index(),
        left_out=reading.left_out,
    )
    assert [list(part) for part in scan.per_measure] == mentions
    assert (stretch in scan.source_stretches) is alone


# Each row: a query, its extraction's variable, entity and dropped phrase, and whether a
# source the query asks for applies the phrase.
_SOURCE_PHRASE_ROWS = [
    (
        "Germany total ODA, as reported to the DAC",
        "total ODA",
        "Germany",
        "as reported to the DAC",
        True,
    ),
    ("Germany total ODA per the DAC", "total ODA", "Germany", "per the DAC", False),
    (
        "Germany total ODA, as published by the DAC",
        "total ODA",
        "Germany",
        "as published by the DAC",
        True,
    ),
    ("Germany total ODA including the DAC", "total ODA", "Germany", "including the DAC", True),
    ("aide à l'éducation au Kenya selon le SNPC", "education aid", "Kenya", "selon le SNPC", True),
    ("ODA to Kenya, the DAC2a figures", "ODA", "Kenya", "the DAC2a figures", True),
    ("education aid to Kenya from the CRS", "education aid", "Kenya", "from the CRS", True),
    (
        "Germany total ODA, preliminary figures as reported to the DAC",
        "total ODA",
        "Germany",
        "preliminary figures as reported to the DAC",
        False,
    ),
    (
        "Germany total ODA, preliminary data from the DAC",
        "total ODA",
        "Germany",
        "preliminary data from the DAC",
        False,
    ),
    (
        "Germany total ODA, net disbursements as reported to the DAC",
        "total ODA",
        "Germany",
        "net disbursements as reported to the DAC",
        False,
    ),
    (
        "Germany ODA on a grant-equivalent basis from the DAC",
        "ODA",
        "Germany",
        "on a grant-equivalent basis from the DAC",
        False,
    ),
    (
        "Germany total ODA, bilateral only, as reported to the DAC",
        "total ODA",
        "Germany",
        "bilateral only, as reported to the DAC",
        False,
    ),
    (
        "Germany ODA for humanitarian purposes reported to the DAC",
        "ODA",
        "Germany",
        "for humanitarian purposes reported to the DAC",
        False,
    ),
    (
        "Germany total ODA, excluding debt relief as reported to the DAC",
        "total ODA",
        "Germany",
        "excluding debt relief as reported to the DAC",
        False,
    ),
    (
        "Germany total ODA, revised figures from the DAC",
        "total ODA",
        "Germany",
        "revised figures from the DAC",
        False,
    ),
    (
        "Germany total ODA in constant prices from the DAC",
        "total ODA",
        "Germany",
        "in constant prices from the DAC",
        False,
    ),
    (
        "Germany total ODA in euros according to the DAC",
        "total ODA",
        "Germany",
        "in euros according to the DAC",
        False,
    ),
    (
        "Germany total ODA, forecasts according to the DAC",
        "total ODA",
        "Germany",
        "forecasts according to the DAC",
        False,
    ),
    (
        "Germany total ODA not reported to the DAC",
        "total ODA",
        "Germany",
        "not reported to the DAC",
        False,
    ),
    (
        "Germany total ODA reported outside the DAC",
        "total ODA",
        "Germany",
        "reported outside the DAC",
        False,
    ),
    (
        "Germany total ODA compared with the DAC",
        "total ODA",
        "Germany",
        "compared with the DAC",
        False,
    ),
    ("Germany total ODA relative to the DAC", "total ODA", "Germany", "relative to the DAC", False),
    (
        "Germany total ODA as a share of the DAC",
        "total ODA",
        "Germany",
        "as a share of the DAC",
        False,
    ),
    ("Germany total ODA, other than the DAC", "total ODA", "Germany", "other than the DAC", False),
    ("Germany total ODA, DAC data for 2020", "total ODA", "Germany", "DAC data for 2020", False),
    (
        "education aid to Kenya, the CRS commitments",
        "education aid",
        "Kenya",
        "the CRS commitments",
        False,
    ),
    (
        "education aid to Kenya from the CRS gross disbursements",
        "education aid",
        "Kenya",
        "from the CRS gross disbursements",
        False,
    ),
    (
        "education aid to Kenya, preliminary figures from the CRS",
        "education aid",
        "Kenya",
        "preliminary figures from the CRS",
        False,
    ),
    (
        "education aid to Kenya, commitments from the CRS",
        "education aid",
        "Kenya",
        "commitments from the CRS",
        False,
    ),
    (
        "education aid to Kenya, humanitarian projects in the CRS",
        "education aid",
        "Kenya",
        "humanitarian projects in the CRS",
        False,
    ),
    (
        "education aid to Kenya, private donors in the CRS",
        "education aid",
        "Kenya",
        "private donors in the CRS",
        False,
    ),
    (
        "education aid to Kenya, flows not in the CRS",
        "education aid",
        "Kenya",
        "flows not in the CRS",
        False,
    ),
    (
        "education aid to Kenya, flows missing from the CRS",
        "education aid",
        "Kenya",
        "flows missing from the CRS",
        False,
    ),
    (
        "education aid to Kenya, 2019 edition of the CRS",
        "education aid",
        "Kenya",
        "2019 edition of the CRS",
        False,
    ),
    (
        "education aid to Kenya, grants as reported to the CRS",
        "education aid",
        "Kenya",
        "grants as reported to the CRS",
        False,
    ),
    (
        "education aid to Kenya, loans from the CRS",
        "education aid",
        "Kenya",
        "loans from the CRS",
        False,
    ),
    (
        "education aid to Kenya, bilateral data from the CRS",
        "education aid",
        "Kenya",
        "bilateral data from the CRS",
        False,
    ),
    (
        "GDP of Kenya, the 2025 edition of the World Economic Outlook",
        "GDP",
        "Kenya",
        "the 2025 edition of the World Economic Outlook",
        False,
    ),
    (
        "GDP of Kenya, the latest vintage of the World Economic Outlook",
        "GDP",
        "Kenya",
        "the latest vintage of the World Economic Outlook",
        False,
    ),
    (
        "GDP of Kenya, staff projections from the World Economic Outlook",
        "GDP",
        "Kenya",
        "staff projections from the World Economic Outlook",
        False,
    ),
    (
        "GDP of Kenya, October update of the World Economic Outlook",
        "GDP",
        "Kenya",
        "October update of the World Economic Outlook",
        False,
    ),
    (
        "GDP of Kenya, from the April 2025 World Economic Outlook",
        "GDP",
        "Kenya",
        "from the April 2025 World Economic Outlook",
        False,
    ),
    (
        "aide à l'éducation au Kenya, chiffres provisoires selon le SNPC",
        "education aid",
        "Kenya",
        "chiffres provisoires selon le SNPC",
        False,
    ),
    (
        "aide à l'éducation au Kenya, engagements selon le SNPC",
        "education aid",
        "Kenya",
        "engagements selon le SNPC",
        False,
    ),
    (
        "ayuda a la educación en Kenia, datos preliminares según el CRS",
        "education aid",
        "Kenia",
        "datos preliminares según el CRS",
        False,
    ),
    (
        "ayuda a la educación en Kenia, compromisos según el CRS",
        "education aid",
        "Kenia",
        "compromisos según el CRS",
        False,
    ),
    (
        "ODA to Kenya, imputed shares from the DAC2a",
        "ODA",
        "Kenya",
        "imputed shares from the DAC2a",
        False,
    ),
    ("CRS education aid to Kenya", "education aid", "Kenya", "according to the CRS", False),
    ("aid to Kenya in fragile states", "aid", "Kenya", "in fragile states", False),
]


@pytest.mark.parametrize(("query", "variable", "entity", "dropped", "applied"), _SOURCE_PHRASE_ROWS)
def test_a_dropped_phrase_naming_only_a_source_is_applied_by_it(
    catalogue: CatalogueSnapshot,
    query: str,
    variable: str,
    entity: str,
    dropped: str,
    applied: bool,
):
    extraction = Extraction(
        kind="resolution",
        variables=[variable],
        entities=[ExtractedEntity(text=entity, span=entity)],
        dropped_qualifiers=[dropped],
    )
    with bound_catalogue(catalogue):
        prep = raw_text._prepare_extraction(
            extraction, ExtractionResult(extraction=extraction, qualifiers={}), query=query
        )
    assert bool(prep.qualifier_warnings) is not applied
    assert bool(prep.qualifier_facts) is not applied


def test_a_source_the_query_leaves_out_applies_no_phrase(catalogue: CatalogueSnapshot):
    """The stretch that leaves out the CRS is the exclusion's to judge, and the source it
    leaves out is no source the answer serves."""
    reading = _left_out("education aid to Kenya excluding the CRS", "excluding the CRS")
    with bound_catalogue(catalogue):
        scan = scan_sources(
            reading.text,
            dictionary=source_dictionary(catalogue),
            measures=["education aid"],
            entity_spans=["Kenya"],
            index=get_index(),
            left_out=reading.left_out,
        )
    assert [hit.polarity for hit in scan.hits] == ["exclude"]
    assert source_phrases(["excluding the CRS"], query=reading.text, hits=scan.hits) == set()
    assert (
        source_phrases(
            ["excluding the CRS"],
            query="education aid to Kenya excluding the CRS",
            hits=[dataclasses.replace(scan.hits[0], start=27, end=30)],
        )
        == set()
    )


@pytest.mark.parametrize(
    ("query", "variable", "dropped", "warned"),
    [
        ("Germany total ODA, as reported to the DAC", "total ODA", "as reported to the DAC", False),
        (
            "total ODA by Germany as reported to the DAC",
            "total official development assistance",
            "as reported to the DAC",
            False,
        ),
        ("total ODA from Germany reported to the DAC", "ODA", "reported to the DAC", False),
        ("ODA reported to the DAC by Germany", "official development assistance", None, False),
        # A source the dictionary does not hold is no mention, so its phrase still warns.
        (
            "Germany total ODA, as reported to Eurostat",
            "total ODA",
            "as reported to Eurostat",
            True,
        ),
        # A named source the phrase narrows to one edition is not what the answer serves.
        (
            "GDP of Kenya from the April 2025 World Economic Outlook",
            "GDP",
            "from the April 2025 World Economic Outlook",
            True,
        ),
    ],
)
def test_raw_door_warns_for_a_source_phrase_only_where_its_mention_does_not_apply_it(
    catalogue: CatalogueSnapshot, query: str, variable: str, dropped: str | None, warned: bool
):
    """Each row carries the extraction recorded live for its query (Germany's role aside)."""
    extraction = Extraction(
        kind="resolution",
        variables=[variable],
        entities=[ExtractedEntity(text="Germany", span="Germany")],
        dropped_qualifiers=[dropped] if dropped is not None else [],
    )
    with bound_catalogue(catalogue):
        prep = raw_text._prepare_extraction(
            extraction, ExtractionResult(extraction=extraction, qualifiers={}), query=query
        )
    assert bool(prep.qualifier_warnings) is warned
    assert bool(prep.qualifier_facts) is warned


def test_raw_door_reads_a_stretch_that_leaves_out_a_publisher(catalogue: CatalogueSnapshot):
    reading = _left_out(
        "education aid to Kenya excluding the World Bank", "excluding the World Bank"
    )
    extraction = Extraction(
        kind="resolution",
        variables=["education aid"],
        entities=[
            ExtractedEntity(text="Kenya", span="Kenya", role="recipient", cue="to"),
        ],
    )
    with bound_catalogue(catalogue):
        prep = raw_text._prepare_extraction(
            extraction,
            ExtractionResult(extraction=extraction, qualifiers={}),
            query=reading.text,
            left_out=reading.left_out,
        )
    assert prep.entities == ["Kenya"]
    assert prep.per_variable_sources == [(_out("World Bank"),)]
    assert prep.exclusions == ()


@pytest.mark.parametrize(
    ("query", "variables", "codes", "compiled"),
    [
        ("debt service of Kenya", ["debt service"], [None], [None]),
        (
            "Kenya's debt service, dt.tds.degg.cd",
            ["debt service dt.tds.degg.cd"],
            ["DT.TDS.DEGG.CD"],
            ["DT.TDS.DEGG.CD"],
        ),
        (
            "DT.NFL.MIDA.CD to Kenya",
            ["DT.NFL.MIDA.CD"],
            ["DT.NFL.MIDA.CD"],
            ["source_code_shares_variable"],
        ),
    ],
    ids=["no-code", "held", "shares-variable"],
)
def test_the_raw_text_door_reads_a_series_code_the_query_writes(
    catalogue: CatalogueSnapshot, query, variables, codes, compiled
):
    """The extraction names no series code. The scan finds a held one in the query, and the
    part compiles it as a parsed part's `source_code` compiles."""
    extraction = Extraction(
        kind="resolution",
        variables=variables,
        entities=[ExtractedEntity(text="Kenya", span="Kenya")],
    )
    with bound_catalogue(catalogue):
        prep = raw_text._prepare_extraction(
            extraction, ExtractionResult(extraction=extraction, qualifiers={}), query=query
        )

    assert prep.per_variable_source_codes == codes
    assert [
        code.reason if isinstance(code, SourceCodeRefusal) else code and code.code
        for code in prep.per_variable_compiled_codes
    ] == compiled


@pytest.mark.parametrize(
    ("query", "stretch", "measures", "targets"),
    [
        (
            "ODA to Kenya and GDP of Kenya excluding humanitarian aid",
            "excluding humanitarian aid",
            ["ODA", "GDP"],
            (1,),
        ),
        (
            "ODA excluding humanitarian aid to Kenya and GDP of Kenya",
            "excluding humanitarian aid",
            ["ODA", "GDP"],
            (0,),
        ),
        pytest.param(
            "ODA and OOF to Kenya excluding debt relief",
            "excluding debt relief",
            ["ODA", "OOF"],
            (0, 1),
            id="coordinated-measures",
        ),
        pytest.param(
            "ODA and OOF to Kenya, both excluding debt relief",
            "excluding debt relief",
            ["ODA", "OOF"],
            (0, 1),
            id="coordinated-measures-both",
        ),
        pytest.param(
            "aid and loans to Kenya excluding debt relief",
            "excluding debt relief",
            ["official development assistance", "loans"],
            (0, 1),
            id="a-measure-the-query-does-not-write",
        ),
        pytest.param(
            "health ODA to Kenya and ODA to Uganda excluding debt relief",
            "excluding debt relief",
            ["health ODA", "ODA"],
            (1,),
            id="a-measure-inside-another",
        ),
        pytest.param(
            "ODA to Uganda and health ODA to Kenya excluding debt relief",
            "excluding debt relief",
            ["ODA", "health ODA"],
            (1,),
            id="a-measure-inside-another-later",
        ),
        pytest.param(
            "ODA excluding debt relief and OOF to Kenya",
            "excluding debt relief",
            ["ODA", "OOF"],
            (0,),
            id="the-blank-parts-coordinated-measures",
        ),
        pytest.param(
            "excluding debt relief, ODA and OOF to Kenya",
            "excluding debt relief",
            ["ODA", "OOF"],
            (0, 1),
            id="before-every-measure",
        ),
    ],
)
def test_a_stretch_qualifies_the_measure_it_attaches_to(
    dictionary: SourceDictionary, query, stretch, measures, targets
):
    """A stretch qualifies every measure, but the measures of one clause when its blank lies
    in that clause, so only those parts disclose it."""
    reading = _left_out(query, stretch)
    scan = scan_sources(
        reading.text,
        dictionary=dictionary,
        measures=measures,
        entity_spans=["Kenya"],
        index=get_index(),
        left_out=reading.left_out,
    )
    assert scan.stretch_targets == (targets,)


def test_a_stretch_the_query_does_not_write_qualifies_every_measure(dictionary: SourceDictionary):
    scan = scan_sources(
        "ODA to Kenya and GDP of Kenya",
        dictionary=dictionary,
        measures=["ODA", "GDP"],
        entity_spans=["Kenya"],
        index=get_index(),
        left_out=[LeftOut("excluding the CRS", ("CRS",), None, "excluding the CRS")],
    )
    assert (scan.stretch_targets, scan.per_measure, scan.source_stretches) == (
        ((0, 1),),
        [(), ()],
        frozenset(),
    )


def test_a_stretch_departs_only_on_the_part_it_qualifies(catalogue: CatalogueSnapshot):
    """ "excluding humanitarian aid" qualifies GDP alone: the envelope warns while the GDP
    part does not leave it out, and only the GDP part says the search did not apply it."""
    query = "ODA to Kenya and GDP of Kenya excluding humanitarian aid"
    reading = _left_out(query, "excluding humanitarian aid")
    extraction = Extraction(
        kind="resolution",
        variables=["ODA", "GDP"],
        entities=[ExtractedEntity(text="Kenya", span="Kenya")],
    )
    with bound_catalogue(catalogue):
        prep = raw_text._prepare_extraction(
            extraction,
            ExtractionResult(extraction=extraction, qualifiers={}),
            query=reading.text,
            left_out=reading.left_out,
        )
    regions = [
        prep.onto(_region(conjoin_spec(shape, "x"), variable), i)
        for i, (shape, variable) in enumerate(((_PERSON_SHAPE, "ODA"), (_ECON_SHAPE, "GDP")))
    ]

    response, _ = combine_regions(
        regions,
        extra_warnings=prep.surviving(regions),
        start_ms=now_ms(),
        engine_build="test",
        max_candidates=10,
    )

    assert prep.exclusion_targets == ((1,),)
    assert [
        [d.asked for d in part.departures if d.code == "measure_not_applied"]
        for part in response.parts
    ] == [[], ["excluding humanitarian aid"]]
    assert [w.code for w in response.diagnostics.warnings] == ["query_qualifier_dropped"]


@pytest.mark.parametrize(
    ("entity", "refused_as"),
    [
        ("World Bank", "World Bank"),
        ("World Bank", "the World Bank"),
        ("World Bank", "world bank"),
        ("World bank", "World Bank"),
    ],
)
def test_a_refusal_on_the_organisation_s_words_says_it_was_read_as_a_party(entity, refused_as):
    """A refusal built by another path can name the organisation in other case or with a
    leading "the" than the extraction's entity holds; the part still says it read the
    organisation as a party."""
    reading = OrganisationReading(role="party", source_possible=False, reason="no_own_data")
    refused = dataclasses.replace(
        _refusal("debt service", "entity_role_not_admitted"),
        unresolved_entity_names=(refused_as,),
    )

    disclosed = read_as_party(refused, dataclasses.replace(_WORLD_BANK, entity=entity), reading)

    assert [warning.code for warning in disclosed.warnings] == [ORGANISATION_READ_AS_PARTY]
