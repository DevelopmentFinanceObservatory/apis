"""Tests the raw-text source scan over the committed dev catalogue build and the fixture
code index."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from code_index.shared import get_index
from origin_catalogue import CatalogueSnapshot, bound_catalogue
from origin_catalogue.artifact import load_catalogue
from scripts.stage_origin_catalogue_fixture import stage_fixture
from search.engine.entry import raw_text
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
    # data word follows it or "according to" precedes it.
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


@pytest.mark.parametrize(
    ("phrase", "covered"),
    [
        ("the CRS", True),
        ("selon le SNPC", True),
        ("IMF", True),
        ("from the April 2025 World Economic Outlook", False),
        ("in fragile states", False),
    ],
)
def test_a_dropped_phrase_naming_only_a_source_is_applied_by_it(phrase: str, covered: bool):
    mentions = [_out("CRS"), _in("SNPC"), _in("IMF"), _in("World Economic Outlook")]
    assert (phrase in source_phrases([phrase], mentions)) is covered


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
    ],
)
def test_a_stretch_qualifies_the_measure_it_attaches_to(
    dictionary: SourceDictionary, query, stretch, measures, targets
):
    """A stretch qualifies a measure as a source does, from its blank, so only that part
    discloses it."""
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
        left_out=[LeftOut("excluding the CRS", ("CRS",), None)],
    )
    assert (scan.stretch_targets, scan.per_measure, scan.source_stretches) == (
        ((0, 1),),
        [(), ()],
        frozenset(),
    )
