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
    _name_corrections,
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
from search.render.assumptions import NameCorrection
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_choice_options import _FERTILITY, _fertility_client, _ShapeZeroLLM
from tests.fixtures import FakeGraph, make_resolver


class _Resolver:
    """Small resolver double that records each batched lookup. Each of `names` is a
    registered name (`is_exact_registered_name`). `answers` holds each text's resolution,
    and a text it omits is not found, as the resolver finds no entity for most stretches of
    query words; a country's names are no registered names, and resolve as any text does."""

    def __init__(self, answers: dict[str, object], *, names: tuple[str, ...] = ()) -> None:
        self.answers = answers
        self.names = {name.casefold() for name in names}
        self.calls: list[list[str]] = []

    def is_exact_registered_name(self, text: str) -> bool:
        return text.casefold() in self.names

    async def resolve(self, texts: list[str]) -> list[object]:
        self.calls.append(texts)
        return [self.answers.get(text, NotFound()) for text in texts]


_ASIA = Entity(dcid="asia", name="Asia", kind="continent", graph_types=("Continent",))
_DAC_ASIA = Entity(
    dcid="DACAsiaUnallocated",
    name="Asia, unallocated by country (DAC)",
    kind="aggregate",
    graph_types=("StatisticalPopulation",),
)


def _recover(
    entities: list[ExtractedEntity], *, query: str, resolver: _Resolver
) -> list[ExtractedEntity]:
    """Recover the entities' surfaces as the raw-text door does before its scans."""
    return asyncio.run(
        _recover_entity_surfaces(
            entities,
            query=query,
            resolver=cast(EntityResolver, resolver),
            deadline=time.monotonic() + 30,
        )
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
    entities = _recover(
        [ExtractedEntity(text="Asia", span="Asia")],
        query="France ODA to Asia (DAC)",
        resolver=resolver,
    )

    assert entities == [ExtractedEntity(text="Asia (DAC)", span="Asia (DAC)")]


def test_registered_full_span_promotes_the_qualified_canonical_identity() -> None:
    resolver = _Resolver({}, names=("Asia (DAC)",))
    entities = _recover(
        [ExtractedEntity(text="Asia", span="Asia (DAC)")],
        query="France ODA to Asia (DAC)",
        resolver=resolver,
    )

    assert entities == [ExtractedEntity(text="Asia (DAC)", span="Asia (DAC)")]


@pytest.mark.parametrize("span", ["Asie", "Asie (DAC)"])
def test_translated_span_carries_its_qualifier_onto_the_english_name(span: str) -> None:
    resolver = _Resolver({}, names=("Asia (DAC)",))
    entities = _recover(
        [ExtractedEntity(text="Asia", span=span)],
        query="aide de la France à l'Asie (DAC)",
        resolver=resolver,
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

    recovered = _recover(entities, query=query, resolver=resolver)

    assert recovered == expected


def test_unknown_parenthetical_does_not_change_a_bare_place_identity() -> None:
    resolver = _Resolver({})
    entities = _recover(
        [ExtractedEntity(text="Asia", span="Asia")],
        query="aid to Asia (unknown grouping)",
        resolver=resolver,
    )

    assert entities == [ExtractedEntity(text="Asia", span="Asia")]


def test_explicit_qualified_and_unqualified_mentions_stay_distinct() -> None:
    resolver = _Resolver({}, names=("Asia (DAC)",))
    entities = _recover(
        [
            ExtractedEntity(text="Asia", span="Asia"),
            ExtractedEntity(text="Asia (DAC)", span="Asia (DAC)"),
        ],
        query="aid to Asia and Asia (DAC)",
        resolver=resolver,
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
    graph_types=("Country",),
)
_COG = Entity(
    dcid="country/COG", name="Republic of the Congo", kind="country", graph_types=("Country",)
)
_KOR = Entity(dcid="country/KOR", name="South Korea", kind="country", graph_types=("Country",))
_PRK = Entity(dcid="country/PRK", name="North Korea", kind="country", graph_types=("Country",))
_TTO = Entity(
    dcid="country/TTO", name="Trinidad and Tobago", kind="country", graph_types=("Country",)
)
_ATG = Entity(
    dcid="country/ATG", name="Antigua and Barbuda", kind="country", graph_types=("Country",)
)
_VCT = Entity(
    dcid="country/VCT",
    name="Saint Vincent and the Grenadines",
    kind="country",
    graph_types=("Country",),
)
_FSM = Entity(
    dcid="country/FSM",
    name="Federated States of Micronesia",
    kind="country",
    graph_types=("Country",),
)
_MICRONESIA = Entity(dcid="Micronesia", name="Micronesia", kind="subregion", graph_types=("Place",))
# Towns sharing the first word of a country's name, as the graph's text search finds them.
_TRINIDAD_TOWN = Entity(
    dcid="wikidataId/Q1001", name="Trinidad", kind="place_other", graph_types=("City",)
)
_ANTIGUA_TOWN = Entity(
    dcid="wikidataId/Q1002", name="Antigua Guatemala", kind="place_other", graph_types=("City",)
)
_SAINT_VINCENT_TOWN = Entity(
    dcid="wikidataId/Q1003", name="Saint-Vincent", kind="place_other", graph_types=("City",)
)
_CAN = Entity(dcid="country/CAN", name="Canada", kind="country", graph_types=("Country",))
_DEU = Entity(dcid="country/DEU", name="Germany", kind="country", graph_types=("Country",))
_GBR = Entity(dcid="country/GBR", name="United Kingdom", kind="country", graph_types=("Country",))
_USA = Entity(dcid="country/USA", name="United States", kind="country", graph_types=("Country",))
_NAIROBI = Entity(
    dcid="wikidataId/Q3870", name="Nairobi", kind="place_other", graph_types=("City",)
)
_NAIROBI_COUNTY = Entity(
    dcid="wikidataId/Q3335223", name="Nairobi County", kind="place_other", graph_types=("County",)
)
# Two towns named Canadian, in Oklahoma and Texas, which the graph pack holds.
_CANADIAN_OK = Entity(
    dcid="geoId/4011450", name="Canadian", kind="place_other", graph_types=("City",)
)
_CANADIAN_TX = Entity(
    dcid="geoId/4812412", name="Canadian", kind="place_other", graph_types=("City",)
)


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
            Ambiguous(candidates=(_COD, _COG)),
            id="congo",
        ),
        pytest.param(
            "aid to Korea",
            ExtractedEntity(text="South Korea", span="Korea"),
            {"Korea": Ambiguous(candidates=(_KOR, _PRK)), "South Korea": Resolved(entity=_KOR)},
            Ambiguous(candidates=(_KOR, _PRK)),
            id="korea",
        ),
        pytest.param(
            "aid to Trinidad",
            ExtractedEntity(text="Trinidad and Tobago", span="Trinidad"),
            {
                "Trinidad": Ambiguous(candidates=(_TTO, _TRINIDAD_TOWN)),
                "Trinidad and Tobago": Resolved(entity=_TTO),
            },
            Resolved(entity=_TTO),
            id="trinidad",
        ),
        pytest.param(
            "aid to Antigua",
            ExtractedEntity(text="Antigua and Barbuda", span="Antigua"),
            {
                "Antigua": Ambiguous(candidates=(_ATG, _ANTIGUA_TOWN)),
                "Antigua and Barbuda": Resolved(entity=_ATG),
            },
            Resolved(entity=_ATG),
            id="antigua",
        ),
        pytest.param(
            "aid to Saint Vincent",
            ExtractedEntity(text="Saint Vincent and the Grenadines", span="Saint Vincent"),
            {
                "Saint Vincent": Ambiguous(candidates=(_VCT, _SAINT_VINCENT_TOWN)),
                "Saint Vincent and the Grenadines": Resolved(entity=_VCT),
            },
            Resolved(entity=_VCT),
            id="saint_vincent",
        ),
        pytest.param(
            "aid to Micronesia",
            ExtractedEntity(text="Federated States of Micronesia", span="Micronesia"),
            {
                "Micronesia": Ambiguous(candidates=(_FSM, _MICRONESIA)),
                "Federated States of Micronesia": Resolved(entity=_FSM),
            },
            Resolved(entity=_FSM),
            id="micronesia",
        ),
        pytest.param(
            "German aid to Kenya",
            ExtractedEntity(text="Germany", span="German"),
            {"German": NotFound(), "Germany": Resolved(entity=_DEU)},
            Resolved(entity=_DEU),
            id="demonym_the_resolver_does_not_hold",
        ),
        pytest.param(
            "aid from the U.S. to Kenya",
            ExtractedEntity(text="United States", span="U.S."),
            {"U.S.": NotFound(), "United States": Resolved(entity=_USA)},
            Resolved(entity=_USA),
            id="abbreviation_the_resolver_does_not_hold",
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
def test_a_surface_and_its_name_settle_on_one_resolution(
    query: str, entity: ExtractedEntity, answers: dict[str, object], expected: Resolution
) -> None:
    """The query's words decide: a name that only adds words to its surface settles its
    readings only as their one place of its kind ("Trinidad and Tobago" for "Trinidad",
    beside a town), never between two of a kind ("Democratic Republic of the Congo" for
    "Congo"), and a name in other words answers where the surface finds nothing."""
    resolver = _Resolver(answers)

    found = _lookup([entity], query=query, resolver=resolver)

    assert found == {entity.text: expected}
    assert resolver.calls == [[entity.span, entity.text]]


@pytest.mark.parametrize(
    ("query", "name", "span", "answers", "expected"),
    [
        pytest.param(
            "aid to Congo",
            "Democratic Republic of the Congo",
            "Congo",
            {
                "Congo": Ambiguous(candidates=(_COD, _COG)),
                "Democratic Republic of the Congo": Resolved(entity=_COD),
            },
            Ambiguous(candidates=(_COD, _COG)),
            id="expanded_name",
        ),
        pytest.param(
            "aid to DR Congo",
            "Democratic Republic of the Congo",
            "DR Congo",
            {
                "DR Congo": Resolved(entity=_COD),
                "Democratic Republic of the Congo": Resolved(entity=_COD),
            },
            Resolved(entity=_COD),
            id="mention_keeps_its_choosing_word",
        ),
        pytest.param(
            "aid to South Korea",
            "Republic of Korea",
            "South Korea",
            {"South Korea": Resolved(entity=_KOR), "Republic of Korea": Resolved(entity=_KOR)},
            Resolved(entity=_KOR),
            id="mention_names_one_korea",
        ),
        pytest.param(
            "aid to Korea",
            "Democratic People's Republic of Korea",
            None,
            # The resolver reads "Korea" as South Korea.
            {
                "Korea": Resolved(entity=_KOR),
                "Democratic People's Republic of Korea": Resolved(entity=_PRK),
            },
            Resolved(entity=_PRK),
            id="mention_naming_another_place",
        ),
        pytest.param(
            "German aid to Kenya",
            "Germany",
            None,
            {"Germany": Resolved(entity=_DEU)},
            Resolved(entity=_DEU),
            id="demonym",
        ),
        pytest.param(
            "aid from the UK to Kenya",
            "United Kingdom",
            None,
            {"United Kingdom": Resolved(entity=_GBR)},
            Resolved(entity=_GBR),
            id="abbreviation",
        ),
    ],
)
def test_a_name_the_query_does_not_write_resolves_through_the_query_words(
    query: str,
    name: str,
    span: str | None,
    answers: dict[str, object],
    expected: Resolution,
) -> None:
    """An extraction that names a place in words the query does not write, and cites no
    span, gets the query's own words for it as its span, which resolution reads: the words
    it shares with the query, grown to the widest stretch around them that resolves to the
    name's place ("DR Congo", "South Korea"). Words that resolve to another place, and a
    name in other words, which shares none with the query, leave it to resolve as itself."""
    resolver = _Resolver(answers)

    recovered = _recover([ExtractedEntity(text=name)], query=query, resolver=resolver)
    found = _lookup(recovered, query=query, resolver=resolver)

    assert recovered == [ExtractedEntity(text=name, span=span)]
    assert found == {name: expected}


_COD_NAME = "Democratic Republic of the Congo"
_COG_NAME = "Republic of the Congo"
_KOR_NAME = "Republic of Korea"
_PRK_NAME = "Democratic People's Republic of Korea"


@pytest.mark.parametrize(
    ("query", "entities", "spans"),
    [
        pytest.param(
            "ODA from DRC to Republic of the Congo",
            [ExtractedEntity(text=_COD_NAME), ExtractedEntity(text=_COG_NAME, span=_COG_NAME)],
            [None, _COG_NAME],
            id="other_span_holds_the_words",
        ),
        pytest.param(
            "ODA from DR Congo to Congo",
            [ExtractedEntity(text=_COD_NAME), ExtractedEntity(text=_COG_NAME)],
            ["DR Congo", None],
            id="stretch_names_another_congo",
        ),
        pytest.param(
            "aid from South Korea to North Korea",
            [ExtractedEntity(text=_KOR_NAME), ExtractedEntity(text=_PRK_NAME)],
            ["South Korea", "North Korea"],
            id="each_korea_keeps_its_choosing_word",
        ),
        pytest.param(
            "aid to North and South Korea",
            [ExtractedEntity(text=_KOR_NAME), ExtractedEntity(text=_PRK_NAME)],
            ["South Korea", None],
            id="one_stretch_for_two_koreas",
        ),
        pytest.param(
            "aid to North and South Korea",
            [ExtractedEntity(text=_PRK_NAME), ExtractedEntity(text=_KOR_NAME)],
            [None, "South Korea"],
            id="one_stretch_for_two_koreas_named_in_turn",
        ),
        pytest.param(
            "ODA from DRC to Congo",
            [ExtractedEntity(text=_COD_NAME), ExtractedEntity(text=_COG_NAME)],
            [None, None],
            id="contested_word",
        ),
    ],
)
def test_a_recovered_span_takes_no_words_another_entity_names(
    query: str, entities: list[ExtractedEntity], spans: list[str | None]
) -> None:
    """Words another entity's span holds are that entity's mention, a stretch of query
    words is the span only of the place it resolves to, and a word both names hold
    chooses neither. A name with no words of its own resolves as itself: each Congo and
    each Korea once."""
    places = {_COD_NAME: _COD, _COG_NAME: _COG, _KOR_NAME: _KOR, _PRK_NAME: _PRK}
    resolver = _Resolver(
        {
            "Congo": Ambiguous(candidates=(_COD, _COG)),
            "DR Congo": Resolved(entity=_COD),
            "Korea": Resolved(entity=_KOR),
            "South Korea": Resolved(entity=_KOR),
            "North Korea": Resolved(entity=_PRK),
            **{name: Resolved(entity=place) for name, place in places.items()},
        }
    )

    recovered = _recover(entities, query=query, resolver=resolver)
    found = _lookup(recovered, query=query, resolver=resolver)

    assert [entity.span for entity in recovered] == spans
    assert found == {entity.text: Resolved(entity=places[entity.text]) for entity in entities}


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

    recovered = _recover([entity], query=query, resolver=resolver)
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
                    dcid="country/FRA", name="France", kind="country", graph_types=("Country",)
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
    entities = _recover(
        [ExtractedEntity(text="World Bank", span="World Bank")],
        query="ODA from World Bank (DAC)",
        resolver=resolver,
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


_KEN = Entity(dcid="country/KEN", name="Kenya", kind="country", graph_types=("Country",))
_FRA = Entity(dcid="country/FRA", name="France", kind="country", graph_types=("Country",))


@pytest.mark.parametrize(
    ("entity", "query", "answers", "expected"),
    [
        # The typed words name no entity, so the extraction's name corrected them.
        pytest.param(
            ExtractedEntity(text="Germany", span="Germny"),
            "ODA from Germny",
            {"Germany": Resolved(_DEU)},
            (NameCorrection(asked="Germny", ref="country/DEU"),),
            id="misspelling",
        ),
        # A listed alias of the entity names it: a translation, no correction.
        pytest.param(
            ExtractedEntity(text="Kenya", span="Kenia"),
            "ayuda a Kenia",
            {"Kenya": Resolved(_KEN), "Kenia": Resolved(_KEN)},
            (),
            id="alias",
        ),
        # A translation the query writes no words of carries a null span.
        pytest.param(
            ExtractedEntity(text="Germany", span=None),
            "Hilfe aus Deutschland",
            {"Germany": Resolved(_DEU)},
            (),
            id="null-span translation",
        ),
        # The typed words are the name, whatever their case.
        pytest.param(
            ExtractedEntity(text="Kenya", span="kenya"),
            "aid to kenya",
            {"Kenya": Resolved(_KEN)},
            (),
            id="same words",
        ),
        # A national adjective, whose cue is its own span, names its country.
        pytest.param(
            ExtractedEntity(text="France", span="French", flow_role="origin", role_span="French"),
            "French aid",
            {"France": Resolved(_FRA)},
            (),
            id="national adjective",
        ),
        # The words leave a choice the name settles: the entity is one of their candidates.
        pytest.param(
            ExtractedEntity(text="Trinidad and Tobago", span="Trinidad"),
            "aid to Trinidad",
            {
                "Trinidad": Ambiguous((_TTO, _TRINIDAD_TOWN)),
                "Trinidad and Tobago": Resolved(_TTO),
            },
            (),
            id="settled choice",
        ),
        # The words name another entity than the one the name settles on.
        pytest.param(
            ExtractedEntity(text="Kenya", span="Kenyan Republic"),
            "aid to the Kenyan Republic",
            {"Kenya": Resolved(_KEN), "Kenyan Republic": Resolved(_DEU)},
            (NameCorrection(asked="Kenyan Republic", ref="country/KEN"),),
            id="another entity",
        ),
    ],
)
def test_a_name_corrects_typed_words_that_list_no_name_or_alias_of_its_entity(
    entity: ExtractedEntity,
    query: str,
    answers: dict[str, object],
    expected: tuple[NameCorrection, ...],
) -> None:
    """The extraction's name corrects the query's words for it when those words resolve to
    no listed name or alias of the entity the name settles on."""
    resolver = _Resolver(answers)
    corrections = asyncio.run(
        _name_corrections(
            [entity],
            query=query,
            resolver=cast(EntityResolver, resolver),
            deadline=time.monotonic() + 30,
        )
    )
    assert corrections == expected


class _MeasuresFromKenia(_ShapeZeroLLM):
    """Reads every query as `variables` from Kenya, written "Kenia" in the query, and elects
    shape 0 with no bindings."""

    def __init__(self, variables: list[str]) -> None:
        self.variables = variables

    async def generate_structured(self, *, prompt, system, schema):
        if schema is Extraction:
            kenya = ExtractedEntity(
                text="Kenya", span="Kenia", flow_role="origin", role_span="from"
            )
            return Extraction(kind="resolution", variables=self.variables, entities=[kenya]), None
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


@pytest.mark.parametrize(
    "variables", [[_FERTILITY], [_FERTILITY, "population"]], ids=["one measure", "two measures"]
)
def test_a_dropped_direction_quotes_the_query_s_words(variables: list[str]) -> None:
    """A subject-only shape drops the direction the query states for a place. Every
    measure's warning and departure quote the query's words for it, which differ from the
    extracted name."""
    client = _fertility_client(_MeasuresFromKenia(variables))

    response = client.post(
        "/api/search/resolve",
        json={"input": {"kind": "raw_text", "query": f"{' and '.join(variables)} from Kenia"}},
    )

    assert response.status_code == 200, response.text
    parts = response.json()["parts"]
    assert len(parts) == len(variables)
    for part in parts:
        assert [w["message"] for w in part["warnings"] if w["code"] == "direction_discarded"] == [
            "Kenia was named as the provider; this shape reports the place as its own subject, "
            "so the direction was dropped."
        ]
        assert [d["asked"] for d in part["departures"]] == ["Kenia"]
