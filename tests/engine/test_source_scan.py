"""Tests the raw-text source scan over the committed dev catalogue build and the fixture
code index."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import pytest

from code_index.shared import get_index
from origin_catalogue import CatalogueSnapshot, bound_catalogue
from origin_catalogue.artifact import load_catalogue
from scripts.stage_origin_catalogue_fixture import stage_fixture
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
from search.engine.source_code import SourceCodeRefusal
from search.engine.source_scan import scan_sources, source_phrases
from search.engine.source_scope import SourceDictionary, source_dictionary
from search.models import SourceMention
from tests.engine._builders import conjoin_spec
from tests.engine._conjoin_fixtures import _ECON_SHAPE, _PERSON_SHAPE, _region

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
    # An organisation that is an entity or lies in a measure's words is a party, unless a
    # data word follows it or a source cue ("according to", "reported to", "per the")
    # precedes it.
    ("Ghana debt service to the World Bank", ["debt service"], ["Ghana", "World Bank"], [[]]),
    (
        "World Bank IDA concessional flows to Kenya",
        ["IDA concessional flows"],
        ["World Bank", "Kenya"],
        [[]],
    ),
    ("World Bank core contributions", ["World Bank core contributions"], [], [[]]),
    ("which DAC donors met the 0.7% target in 2023", ["ODA as percentage of GNI"], ["DAC"], [[]]),
    (
        "OECD data on education aid to Kenya",
        ["education aid"],
        ["OECD", "Kenya"],
        [[_in("OECD")]],
    ),
    ("GDP of Kenya according to the IMF", ["GDP"], ["Kenya", "IMF"], [[_in("IMF")]]),
    ("Germany ODA per the DAC", ["ODA"], ["Germany", "DAC"], [[_in("DAC")]]),
    ("ODA per DAC member", ["ODA"], ["DAC"], [[]]),
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
        ("Ghana debt service to the World Bank", ["debt service"], ["Ghana", "World Bank"], set()),
        # The second "World Bank" is a party, so the entity stays.
        (
            "World Bank data on aid from the World Bank to Kenya",
            ["aid"],
            ["World Bank", "Kenya"],
            set(),
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
            ExtractedEntity(text="Kenya", span="Kenya", flow_role="destination", role_span="to"),
        ],
    )
    with bound_catalogue(catalogue):
        prep = raw_text._prepare_extraction(
            extraction, ExtractionResult(extraction=extraction, qualifiers={}), query=query
        )

    assert prep.entities == ["Kenya"]
    assert set(prep.entity_roles) == {"Kenya"}
    assert prep.entity_words == {"Kenya": "Kenya"}
    assert prep.per_variable_sources == [(_in("OECD"),)]


def test_the_raw_text_door_keeps_an_entity_read_as_a_source_only_once(
    catalogue: CatalogueSnapshot,
):
    """ "World Bank data on aid from the World Bank to Kenya": the first "World Bank" is the
    source of the numbers, the second the donor. The part keeps the World Bank as an entity
    with its origin role, and as its source."""
    query = "World Bank data on aid from the World Bank to Kenya"
    extraction = Extraction(
        kind="resolution",
        variables=["aid"],
        entities=[
            ExtractedEntity(
                text="World Bank", span="World Bank", flow_role="origin", role_span="from"
            ),
            ExtractedEntity(text="Kenya", span="Kenya", flow_role="destination", role_span="to"),
        ],
    )
    with bound_catalogue(catalogue):
        prep = raw_text._prepare_extraction(
            extraction, ExtractionResult(extraction=extraction, qualifiers={}), query=query
        )

    assert prep.entities == ["World Bank", "Kenya"]
    assert [role.axis for role in prep.entity_roles["World Bank"]] == ["origin"]
    assert prep.per_variable_sources == [(_in("World Bank"),)]


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
    ("Germany total ODA per the DAC", "total ODA", "Germany", "per the DAC", True),
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
            ExtractedEntity(text="Kenya", span="Kenya", flow_role="destination", role_span="to"),
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
