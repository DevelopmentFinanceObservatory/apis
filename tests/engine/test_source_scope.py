"""Tests source mentions compiling to a part's scope over the committed dev catalogue build."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from origin_catalogue import CatalogueSnapshot
from origin_catalogue.artifact import load_catalogue
from origin_catalogue.registry import Alias
from scripts.stage_origin_catalogue_fixture import stage_fixture
from search.engine import source_scope
from search.engine.errors import EngineInputError
from search.engine.source_scope import (
    SourceCollection,
    SourceReference,
    SourceRefusal,
    SourceScope,
    build_source_dictionary,
    clear_source_dictionaries,
    compile_part_sources,
    compile_sources,
    source_dictionary,
    without_source_words,
)
from search.models import ParsedPart, SourceMention

_EVAL_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "origin_catalogue_eval"

IDS = "provenance/WorldBankInternationalDebtStatistics"
CRS = "provenance/OECDCreditorReportingSystem"
DAC1 = "provenance/OECDDAC1"
DAC2A = "provenance/OECDDAC2A"
DAC2B = "provenance/OECDDAC2B"
WEO = "provenance/IMFWorldEconomicOutlook"
IMPUTED = "provenance/ONEImputedMultilateralSectors"
CONVERSION_DAC = "provenance/ONEConversionTablesDAC"
CONVERSION_IMF = "provenance/ONEConversionTablesIMF"
WDI = "dc/base/WorldDevelopmentIndicators"
WORLD_BANK = "dc/s/WorldBank"
OECD = "dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd"
IMF = "source/InternationalMonetaryFund"
ONE_DATA = "source/ONEData"
WHO_SOURCE = "dc/s/WorldHealthOrganizationWho"
WHO_UN = "undata/p/WHO"


@pytest.fixture(scope="module")
def catalogue(tmp_path_factory: pytest.TempPathFactory) -> CatalogueSnapshot:
    return load_catalogue(stage_fixture(_EVAL_FIXTURE_DIR, tmp_path_factory.mktemp("eval")))


def _published_by(catalogue: CatalogueSnapshot, source: str) -> frozenset[str]:
    return frozenset(row.id for row in catalogue.provenance_rows() if row.source.ref == source)


def _every(catalogue: CatalogueSnapshot) -> frozenset[str]:
    return frozenset(row.id for row in catalogue.provenance_rows())


def _include(text: str) -> SourceMention:
    return SourceMention(text=text)


def _exclude(text: str) -> SourceMention:
    return SourceMention(text=text, polarity="exclude")


type Allowed = Callable[[CatalogueSnapshot], frozenset[str]]


def _only(*provenances: str) -> Allowed:
    return lambda _catalogue: frozenset(provenances)


# Each row: the part's mentions, the refs they resolve to, the allowed set, and `named`.
_SCOPES: list[tuple[str, list[SourceMention], list[str | None], Allowed, bool]] = [
    ("IDS", [_include("IDS")], [IDS], _only(IDS), True),
    ("CRS", [_include("crs")], [CRS], _only(CRS), True),
    ("full name", [_include("OECD Creditor Reporting System")], [CRS], _only(CRS), True),
    ("OECD", [_include("the OECD")], [OECD], lambda c: _published_by(c, OECD), True),
    (
        "World Bank",
        [_include("World Bank")],
        [WORLD_BANK],
        lambda c: _published_by(c, WORLD_BANK),
        True,
    ),
    ("WEO", [_include("WEO")], [WEO], _only(WEO), True),
    # The IMF names a source, not the ONE conversion table whose name ends "(IMF)".
    ("IMF", [_include("IMF")], [IMF], _only(WEO), True),
    ("DAC2a", [_include("DAC2a")], [DAC2A], _only(DAC2A), True),
    # A dataset label and its provenance's name match: the dataset contains the provenance.
    (
        "dataset over its provenance",
        [_include("World Development Indicators")],
        ["dc/d/WorldBank_WorldDevelopmentIndicators"],
        _only(WDI),
        True,
    ),
    ("registry alias", [_include("WDI")], [WDI], _only(WDI), True),
    # SNPC is the French name of the CRS (Système de notification des pays créanciers).
    ("French alias", [_include("SNPC")], [CRS], _only(CRS), True),
    # Naming an opt-in product opts into it.
    (
        "opt-in product named",
        [_include("ONE Imputed Multilateral Sectors")],
        [IMPUTED],
        _only(IMPUTED),
        True,
    ),
    # Naming its publisher leaves the opt-in product out.
    (
        "opt-in publisher named",
        [_include("ONE Data")],
        [ONE_DATA],
        _only(CONVERSION_DAC, CONVERSION_IMF),
        True,
    ),
    (
        "include and exclude",
        [_include("OECD"), _exclude("CRS")],
        [OECD, CRS],
        lambda c: _published_by(c, OECD) - {CRS},
        True,
    ),
    # Exclusion alone keeps every other admitted source, opt-in products included.
    ("exclude only", [_exclude("CRS")], [CRS], lambda c: _every(c) - {CRS}, False),
    # Excluding a publisher removes its opt-in products too.
    (
        "exclude opt-in publisher",
        [_exclude("ONE Data")],
        [ONE_DATA],
        lambda c: _every(c) - {CONVERSION_DAC, CONVERSION_IMF, IMPUTED},
        False,
    ),
    # Excluded words that name no admitted source remove nothing and keep no ref.
    ("unresolved exclusion", [_exclude("Global Fund")], [None], _every, False),
    (
        "include beside an unresolved exclusion",
        [_include("CRS"), _exclude("Global Fund")],
        [CRS, None],
        _only(CRS),
        True,
    ),
    ("ref", [SourceMention(ref=IDS)], [IDS], _only(IDS), True),
    ("ref over text", [SourceMention(text="CRS", ref=DAC1)], [DAC1], _only(DAC1), True),
    ("source ref", [SourceMention(ref=IMF)], [IMF], _only(WEO), True),
]


@pytest.mark.parametrize(
    ("mentions", "refs", "allowed", "named"),
    [row[1:] for row in _SCOPES],
    ids=[row[0] for row in _SCOPES],
)
def test_mentions_compile_to_a_scope(
    catalogue: CatalogueSnapshot,
    mentions: list[SourceMention],
    refs: list[str | None],
    allowed: Allowed,
    named: bool,
) -> None:
    pointers = [f"/sources/{i}" for i in range(len(mentions))]
    scope = compile_sources(mentions, catalogue=catalogue, pointers=pointers)
    assert isinstance(scope, SourceScope)
    assert (scope.allowed, scope.named) == (allowed(catalogue), named)
    assert [mention.ref for mention in scope.mentions] == refs
    assert [mention.text for mention in scope.mentions] == [m.text for m in mentions]
    assert [r.ref if isinstance(r, SourceReference) else r for r in scope.references] == refs


_REFUSALS: list[tuple[str, list[SourceMention], str, list[str]]] = [
    ("unknown word", [_include("XYZ statistics")], "source_not_resolved", ["XYZ statistics"]),
    # Two conversion tables share the words and neither contains the other.
    (
        "ambiguous word",
        [_include("ONE Conversion Tables")],
        "source_ambiguous",
        ["ONE Conversion Tables"],
    ),
    (
        "unmatched before ambiguous",
        [_include("ONE Conversion Tables"), _include("XYZ")],
        "source_not_resolved",
        ["XYZ"],
    ),
    # Excluded words that match nothing do not refuse; ambiguous ones still do.
    (
        "unresolved exclusion beside an unmatched include",
        [_include("XYZ statistics"), _exclude("Global Fund")],
        "source_not_resolved",
        ["XYZ statistics"],
    ),
    (
        "ambiguous exclusion",
        [_exclude("ONE Conversion Tables")],
        "source_ambiguous",
        ["ONE Conversion Tables"],
    ),
]


@pytest.mark.parametrize(
    ("mentions", "reason", "terms"),
    [row[1:] for row in _REFUSALS],
    ids=[row[0] for row in _REFUSALS],
)
def test_words_that_resolve_to_no_one_source_refuse(
    catalogue: CatalogueSnapshot, mentions: list[SourceMention], reason: str, terms: list[str]
) -> None:
    pointers = [f"/sources/{i}" for i in range(len(mentions))]
    refusal = compile_sources(mentions, catalogue=catalogue, pointers=pointers)
    assert isinstance(refusal, SourceRefusal)
    assert (refusal.reason, list(refusal.terms)) == (reason, terms)


def test_an_alias_of_an_unadmitted_ref_refuses(catalogue: CatalogueSnapshot) -> None:
    # Every registry alias names refs this build admits, so the alias here is made up.
    with _aliases_replaced(Alias(refs=("provenance/NotInThisBuild",), names=("Phantom",))):
        refusal = compile_sources([_include("Phantom")], catalogue=catalogue, pointers=["/s"])
    assert isinstance(refusal, SourceRefusal)
    assert (refusal.reason, list(refusal.terms)) == ("source_not_resolved", ["Phantom"])


def test_an_ambiguous_refusal_lists_the_candidates_and_pins_the_resolved_mentions(
    catalogue: CatalogueSnapshot,
) -> None:
    refusal = compile_sources(
        [_include("ONE Conversion Tables"), _exclude("CRS")],
        catalogue=catalogue,
        pointers=["/a", "/b"],
    )
    assert isinstance(refusal, SourceRefusal)
    (group,) = refusal.ambiguous
    assert [(c.ref, c.kind) for c in group.candidates] == [
        (CONVERSION_DAC, "provenance"),
        (CONVERSION_IMF, "provenance"),
    ]
    assert refusal.mentions == (
        _include("ONE Conversion Tables"),
        SourceMention(text="CRS", ref=CRS, polarity="exclude"),
    )


@pytest.mark.parametrize(
    ("mentions", "pointer"),
    [
        ([SourceMention(ref="provenance/Unknown")], "/sources/0"),
        ([_include("IDS"), SourceMention(ref="dc/base/WorldBank_IDS")], "/sources/1"),
        ([_include("CRS"), _exclude("IDS"), _exclude("crs")], "/sources/2"),
    ],
    ids=["unadmitted ref", "registry-excluded ref", "exclusion empties the scope"],
)
def test_an_unadmitted_ref_or_an_empty_scope_refuses_request_invalid(
    catalogue: CatalogueSnapshot, mentions: list[SourceMention], pointer: str
) -> None:
    pointers = [f"/sources/{i}" for i in range(len(mentions))]
    with pytest.raises(EngineInputError) as caught:
        compile_sources(mentions, catalogue=catalogue, pointers=pointers)
    assert (caught.value.code, caught.value.pointer) == ("request_invalid", pointer)


def test_no_mentions_compile_to_no_scope(catalogue: CatalogueSnapshot) -> None:
    assert compile_sources([], catalogue=catalogue, pointers=[]) is None


@pytest.mark.parametrize(
    ("variable", "mentions", "measure"),
    [
        ("DAC2a ODA", [_include("DAC2a")], "ODA"),
        ("ODA from DAC 2a", [_include("DAC 2a")], "ODA from"),
        ("CRS education aid", [_include("the CRS")], "education aid"),
        # Words that resolved to no source may be the measure's.
        ("Global Fund grants", [_exclude("Global Fund")], "Global Fund grants"),
        # A variable made only of source words stays.
        ("CRS", [_include("CRS")], "CRS"),
    ],
)
def test_a_variable_leaves_out_the_words_of_the_sources_its_scope_resolved(
    catalogue: CatalogueSnapshot, variable: str, mentions: list[SourceMention], measure: str
) -> None:
    pointers = [f"/sources/{i}" for i in range(len(mentions))]
    scope = compile_sources(mentions, catalogue=catalogue, pointers=pointers)
    assert isinstance(scope, SourceScope)
    assert without_source_words(variable, scope) == measure


def test_a_deprecated_provenance_compiles_as_the_equivalent_mention(
    catalogue: CatalogueSnapshot,
) -> None:
    deprecated = ParsedPart(variable_text="ODA", provenance=CRS)
    mention = ParsedPart(variable_text="ODA", sources=[SourceMention(ref=CRS)])
    assert compile_part_sources(
        deprecated, catalogue=catalogue, pointer_prefix="/p"
    ) == compile_part_sources(mention, catalogue=catalogue, pointer_prefix="/p")


@pytest.mark.parametrize(
    ("part", "pointer"),
    [
        (ParsedPart(variable_text="ODA", provenance="provenance/Unknown"), "/p/provenance"),
        (
            ParsedPart(
                variable_text="ODA",
                sources=[_include("CRS")],
                provenance="provenance/Unknown",
            ),
            "/p/provenance",
        ),
        (
            ParsedPart(variable_text="ODA", sources=[_exclude("CRS")], source_text="crs"),
            "/p/sources/0",
        ),
    ],
    ids=["provenance", "provenance after a mention", "exclusion before source_text"],
)
def test_a_part_refusal_points_at_the_field_the_mention_was_posted_in(
    catalogue: CatalogueSnapshot, part: ParsedPart, pointer: str
) -> None:
    with pytest.raises(EngineInputError) as caught:
        compile_part_sources(part, catalogue=catalogue, pointer_prefix="/p")
    assert caught.value.pointer == pointer


@contextmanager
def _aliases_replaced(*aliases: Alias) -> Iterator[None]:
    """Compile with the registry's aliases replaced by `aliases`."""
    clear_source_dictionaries()
    try:
        with patch.object(source_scope, "_aliases", lambda: aliases):
            yield
    finally:
        clear_source_dictionaries()


def _union(catalogue: CatalogueSnapshot, refs: list[str]) -> frozenset[str]:
    """The provenances of `refs`, each a provenance, a Dataset or a Source."""
    return frozenset(
        row.id
        for row in catalogue.provenance_rows()
        if {row.id, row.source.ref, row.dataset.ref if row.dataset else row.id} & set(refs)
    )


# Each row: the words, the label the collection takes from its alias's first name, and the
# references the alias gathers in this build.
_COLLECTIONS: list[tuple[str, str, str, list[str]]] = [
    ("DAC", "DAC", "DAC", [CRS, DAC1, DAC2A, DAC2B]),
    ("DAC with a qualifier", "the OECD DAC", "DAC", [CRS, DAC1, DAC2A, DAC2B]),
    # The base source and the UN agency, which the catalogue gives the same label.
    ("WHO", "WHO", "WHO", [WHO_SOURCE, WHO_UN]),
    ("WHO spelled out", "World Health Organization", "WHO", [WHO_SOURCE, WHO_UN]),
    ("WHO as named", "World Health Organization (WHO)", "WHO", [WHO_SOURCE, WHO_UN]),
]


@pytest.mark.parametrize(
    ("text", "label", "members"),
    [row[1:] for row in _COLLECTIONS],
    ids=[row[0] for row in _COLLECTIONS],
)
def test_an_alias_of_several_refs_is_one_collection_over_the_union_of_its_members(
    catalogue: CatalogueSnapshot, text: str, label: str, members: list[str]
) -> None:
    included = compile_sources([_include(text)], catalogue=catalogue, pointers=["/s"])
    assert isinstance(included, SourceScope)
    (collection,) = included.references
    assert isinstance(collection, SourceCollection)
    assert (collection.label, [member.ref for member in collection.members]) == (label, members)
    assert (included.allowed, included.named) == (_union(catalogue, members), True)
    # A collection has no ref to post back, so the mention keeps the user's words.
    assert included.mentions == (_include(text),)

    excluded = compile_sources([_exclude(text)], catalogue=catalogue, pointers=["/s"])
    assert isinstance(excluded, SourceScope)
    assert (excluded.allowed, excluded.named) == (
        _every(catalogue) - _union(catalogue, members),
        False,
    )


def test_a_collection_member_that_is_a_source_follows_the_opt_in_rule(
    catalogue: CatalogueSnapshot,
) -> None:
    """An included Source member leaves out its opt-in products; an excluded one removes
    them."""
    with _aliases_replaced(Alias(refs=(ONE_DATA, CRS), names=("custom",))):
        included = compile_sources([_include("custom")], catalogue=catalogue, pointers=["/s"])
        excluded = compile_sources([_exclude("custom")], catalogue=catalogue, pointers=["/s"])
    assert isinstance(included, SourceScope)
    assert included.allowed == {CONVERSION_DAC, CONVERSION_IMF, CRS}
    assert isinstance(excluded, SourceScope)
    assert excluded.allowed == _every(catalogue) - {CONVERSION_DAC, CONVERSION_IMF, IMPUTED, CRS}


def test_an_alias_keeps_only_the_refs_the_catalogue_admits(catalogue: CatalogueSnapshot) -> None:
    dac = {DAC1, DAC2A, DAC2B, CRS}
    one_left = build_source_dictionary(
        tuple(r for r in catalogue.provenance_rows() if r.id not in dac - {CRS})
    )
    none_left = build_source_dictionary(
        tuple(r for r in catalogue.provenance_rows() if r.id not in dac)
    )
    # One admitted ref names that reference, not a collection of one.
    named = one_left.keys["dac"]
    assert isinstance(named, tuple)
    assert [reference.ref for reference in named] == [CRS]
    assert "dac" not in none_left.keys


# Keys the catalogue's own names give to unrelated references that no alias overrides.
_DELIBERATELY_AMBIGUOUS = {
    "one conversion tables": "two provenances, the DAC and the IMF tables; neither contains the other",
}


def _ambiguous_keys(catalogue: CatalogueSnapshot) -> set[str]:
    return {
        key
        for key in source_dictionary(catalogue).keys
        if isinstance(
            compile_sources([_include(key)], catalogue=catalogue, pointers=["/s"]), SourceRefusal
        )
    }


def test_every_key_the_catalogue_gives_to_unrelated_references_is_overridden_or_listed(
    catalogue: CatalogueSnapshot,
) -> None:
    with _aliases_replaced():
        derived = _ambiguous_keys(catalogue)
    assert {"who", "world health organization", "world health organization who"} <= derived
    assert set(_DELIBERATELY_AMBIGUOUS) <= derived

    assert _ambiguous_keys(catalogue) == set(_DELIBERATELY_AMBIGUOUS)
