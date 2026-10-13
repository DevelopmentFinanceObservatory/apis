"""EntityResolver tests use resolvekit's composed pack and a fresh FakeGraphLookup per test.

Graph names differ from pack names to check that resolved names come from resolvekit.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable

import pytest
from cachetools import TLRUCache
from resolvekit import ResolutionContext, ResolutionStatus
from resolvekit.core.model.result import CandidateSummary

from entity_resolution import (
    ADMITTED_TYPES,
    Ambiguous,
    EntityResolver,
    Group,
    NotFound,
    NotHeld,
    Resolved,
    pack,
)
from entity_resolution.regions import identity_to_dcid
from entity_resolution.resolver import (
    _QUERY_ALIASES,
    _GateOutcome,
    _normalize_key,
    _parse_group_survivors,
    _resolve_batch_job,
    _ttu,
    _without_article_and_punctuation,
)
from graph_generation import FakeGeneration
from graph_generation.config import (
    GRAPH_CACHE_MAX_AGE_SECONDS,
    GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS,
)
from graph_generation.stamp import UNKNOWN_KEY

from .conftest import FakeGraphLookup

_UNUSED_GRAPH_NAME = ("graph-name-should-not-be-used",)


def _labels(result: NotFound) -> tuple[str, ...]:
    """A `NotFound`'s near matches by name, as a person reads them."""
    return tuple(entity.name for entity in result.near_matches)


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize(
    ("text", "dcid", "name"),
    [
        ("Kenya", "country/KEN", "Kenya"),
        ("France", "country/FRA", "France"),
        ("Türkiye", "country/TUR", "Turkey"),
        ("Côte d'Ivoire", "country/CIV", "Côte d'Ivoire"),
        ("Kenia", "country/KEN", "Kenya"),
        ("UK", "country/GBR", "United Kingdom"),
        ("DRC", "country/COD", "Congo [DRC]"),
        # Both names include the article in resolvekit's country pack.
        ("the Gambia", "country/GMB", "Gambia"),
        ("the Netherlands", "country/NLD", "Netherlands"),
        # AU is Australia's ISO 3166 alpha-2 code in the country pack.
        ("AU", "country/AUS", "Australia"),
    ],
)
def test_country_listed_alias_resolves(shared_resolver, text: str, dcid: str, name: str):
    graph = FakeGraphLookup(nodes={dcid: {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME}})
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve([text]))

    assert isinstance(result, Resolved), (text, result)
    assert result.entity.dcid == dcid
    assert result.entity.kind == "country"
    assert result.entity.graph_type == "Country"
    assert result.entity.name == name


def test_africa_resolves_continent_via_region_map(shared_resolver):
    graph = FakeGraphLookup(
        nodes={"africa": {"typeOf": ("Continent",), "name": _UNUSED_GRAPH_NAME}}
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve(["Africa"]))

    assert isinstance(result, Resolved)
    assert result.entity.dcid == "africa"
    assert result.entity.kind == "continent"
    assert result.entity.name == "Africa"


def test_sub_saharan_africa_resolves_subregion_via_region_map(shared_resolver):
    graph = FakeGraphLookup(
        nodes={"SubSaharanAfrica": {"typeOf": ("UNGeoRegion",), "name": _UNUSED_GRAPH_NAME}}
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve(["Sub-Saharan Africa"]))

    assert isinstance(result, Resolved)
    assert result.entity.dcid == "SubSaharanAfrica"
    assert result.entity.kind == "subregion"
    assert result.entity.name == "Sub-Saharan Africa"


def test_central_africa_resolves_subregion_via_query_alias(shared_resolver):
    """The alias maps Central Africa to Middle Africa, whose resolvekit score is 0.868.

    The country alias scores 0.935.
    """
    dcid = identity_to_dcid("m49/017")
    graph = FakeGraphLookup(nodes={dcid: {"typeOf": ("UNGeoRegion",), "name": _UNUSED_GRAPH_NAME}})
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    results = _run(resolver.resolve(["Central Africa", "central  africa"]))

    for result in results:
        assert isinstance(result, Resolved), result
        assert result.entity.dcid == dcid
        assert result.entity.kind == "subregion"
        assert result.entity.name == "Middle Africa"


def test_central_african_republic_still_resolves_country(shared_resolver):
    graph = FakeGraphLookup(
        nodes={"country/CAF": {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME}}
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve(["Central African Republic"]))

    assert isinstance(result, Resolved)
    assert result.entity.dcid == "country/CAF"
    assert result.entity.kind == "country"
    assert result.entity.name == "Central African Republic"


@pytest.mark.parametrize(
    ("text", "dcid", "kind"),
    [
        ("East Africa", "EasternAfrica", "subregion"),
        ("West Africa", "WesternAfrica", "subregion"),
        ("North Africa", "NorthernAfrica", "subregion"),
        ("Southern Africa", "SouthernAfrica", "subregion"),
        ("South Asia", "SouthernAsia", "subregion"),
        ("Southeast Asia", "SouthEasternAsia", "subregion"),
        ("Central America", "CentralAmerica", "subregion"),
        ("Latin America", "LatinAmericaAndCaribbean", "subregion"),
        ("Middle East", "WesternAsia", "subregion"),
        ("Caribbean", "Caribbean", "subregion"),
        ("Oceania", "oceania", "continent"),
        ("Sub-Saharan Africa", "SubSaharanAfrica", "subregion"),
    ],
)
def test_other_region_style_queries_resolve_without_a_table_entry(
    shared_resolver, text: str, dcid: str, kind: str
):
    """Each form resolves to its intended region without a query_aliases.json entry."""
    assert _normalize_key(text) not in _QUERY_ALIASES, text
    graph_type = "Continent" if kind == "continent" else "UNGeoRegion"
    graph = FakeGraphLookup(nodes={dcid: {"typeOf": (graph_type,), "name": _UNUSED_GRAPH_NAME}})
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve([text]))

    assert isinstance(result, Resolved), (text, result)
    assert result.entity.dcid == dcid
    assert result.entity.kind == kind


def _resolve_each_on_shared_thread(resolver, texts):
    context = ResolutionContext(entity_types=ADMITTED_TYPES)
    return {
        text: resolver.resolve(text, as_result=True, context=context, include_entity=True)
        for text in texts
    }


def _members_of_on_shared_thread(resolver, group):
    return resolver.members_of(group)


def test_query_aliases_keys_are_normalized_and_values_resolve_in_the_pack(shared_resolver):
    """Alias keys are normalized, and each row names a canonical pack name. Calls use the resolver thread."""
    raw_aliases = json.loads(pack.query_aliases_path().read_text(encoding="utf-8"))
    assert raw_aliases
    for key in raw_aliases:
        assert key == _normalize_key(key), key

    results = shared_resolver.run(
        _resolve_each_on_shared_thread,
        shared_resolver.resolver,
        [row["name"] for row in raw_aliases.values()],
    ).result()

    for value, result in results.items():
        assert result.status != ResolutionStatus.ERROR, (value, result)
        assert any(c.canonical_name == value for c in result.candidates), (value, result)


@pytest.mark.parametrize(
    ("text", "dcid", "name"),
    [
        ("African Development Bank", "AfricanDevelopmentBank", "African Development Bank"),
        ("AfDB", "AfricanDevelopmentBank", "African Development Bank"),
        ("IDA", "InternationalDevelopmentAssociation", "International Development Association"),
        ("EU Institutions", "EuropeanUnionInstitutions", "European Union institutions"),
        ("Asian Development Bank", "AsianDevelopmentBank", "Asian Development Bank"),
        # The exporter folds the ontology node's "AFC" alias into the base-schema node.
        ("AFC", "Africa_Finance_Corporation", "Africa Finance Corporation"),
        ("Africa Finance Corporation", "Africa_Finance_Corporation", "Africa Finance Corporation"),
        ("World Bank Group", "WorldBankGroup", "World Bank Group"),
        # Resolvekit generates no candidate for these stored aliases; the pack's alias index does.
        (
            "OAPEC",
            "OrganizationOfArabPetroleumExportingCountries",
            "Organization of Arab Petroleum Exporting Countries",
        ),
        ("C&A Foundation", "LaudesFoundation", "Laudes Foundation"),
        ("Omidyar Network Fund, Inc.", "OmidyarNetwork", "Omidyar Network"),
        (
            "Latin American Agribusiness Dev. Corp.",
            "LatinAmericanAgribusinessDevelopmentCorporation",
            "Latin American Agribusiness Development Corporation",
        ),
        (
            "African Export\u2013Import Bank",
            "AfricanExportImportBank",
            "African Export-Import Bank",
        ),
        ("Charity Projects Ltd", "ComicRelief", "Comic Relief"),
        ("Inter-American Investment Corporation", "IDBInvest", "IDB Invest"),
        # Without the alias index this canonical name parsed as the UN's member group.
        (
            "Joint United Nations Programme on HIV/AIDS",
            "JointUnitedNationsProgrammeOnHIVAIDS",
            "Joint United Nations Programme on HIV/AIDS",
        ),
    ],
)
def test_organisation_listed_alias_resolves(shared_resolver, text: str, dcid: str, name: str):
    graph = FakeGraphLookup(nodes={dcid: {"typeOf": ("Organization",), "name": _UNUSED_GRAPH_NAME}})
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve([text]))

    assert isinstance(result, Resolved), (text, result)
    assert result.entity.dcid == dcid
    assert result.entity.kind == "organisation"
    assert result.entity.name == name


@pytest.mark.parametrize(
    ("text", "dcid", "graph_type", "name"),
    [
        # The stored alias keeps its comma; the org normalizer strips punctuation.
        (
            "Africa, regional",
            "DACAfricaUnallocated",
            "DACRecipientUnallocatedByCountry",
            "Africa, unallocated by country (DAC)",
        ),
        (
            "other multilaterals",
            "DACOtherMultilaterals",
            "DACUnspecifiedParty",
            "other multilaterals (DAC)",
        ),
    ],
)
def test_dac_residual_resolves_aggregate(
    shared_resolver, text: str, dcid: str, graph_type: str, name: str
):
    graph = FakeGraphLookup(nodes={dcid: {"typeOf": (graph_type,), "name": _UNUSED_GRAPH_NAME}})
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve([text]))

    assert isinstance(result, Resolved), (text, result)
    assert result.entity.dcid == dcid
    assert result.entity.kind == "aggregate"
    assert result.entity.name == name


def test_world_bank_ambiguous_between_ibrd_and_ida(shared_resolver):
    ibrd = "InternationalBankForReconstructionAndDevelopment"
    ida = "InternationalDevelopmentAssociation"
    graph = FakeGraphLookup(
        nodes={
            ibrd: {"typeOf": ("Organization",), "name": _UNUSED_GRAPH_NAME},
            ida: {"typeOf": ("Organization",), "name": _UNUSED_GRAPH_NAME},
        }
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve(["World Bank"]))

    assert isinstance(result, Ambiguous)
    dcids = [c.dcid for c in result.candidates]
    assert set(dcids) == {ibrd, ida}
    assert len(dcids) == 2
    assert all(c.kind == "organisation" for c in result.candidates)


@pytest.mark.parametrize(
    ("text", "dcid", "name"),
    [
        ("IMF", "InternationalMonetaryFund", "International Monetary Fund"),
        ("WHO", "WorldHealthOrganization", "World Health Organization"),
    ],
)
def test_curated_acronym_resolves_to_the_pinned_parent(
    shared_resolver, text: str, dcid: str, name: str
):
    graph = FakeGraphLookup(nodes={dcid: {"typeOf": ("Organization",), "name": _UNUSED_GRAPH_NAME}})
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve([text]))

    assert isinstance(result, Resolved), (text, result)
    assert result.entity.dcid == dcid
    assert result.entity.name == name


def test_gate_invariant_no_listed_survivor_falls_to_fallback_ladder(shared_resolver):
    """Only a node listing the input as a name or alternate name can win the fallback ladder.

    Lagos resolves by alternate name. Nigria and Guinea Bissau remain NotFound.
    """
    graph = FakeGraphLookup(
        nodes={
            "geoId/lagos": {
                "typeOf": ("AdministrativeArea1",),
                "name": ("Lagos State",),
                "alternateName": ("Lagos",),
            }
        },
        arcs={("<-description->dcid", "Lagos"): ["geoId/lagos"]},
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    results = _run(resolver.resolve(["Lagos", "Nigria", "Guinea Bissau"]))

    lagos, nigria, guinea_bissau = results
    assert isinstance(lagos, Resolved)
    assert lagos.entity.dcid == "geoId/lagos"
    assert lagos.entity.kind == "place_other"
    assert lagos.entity.name == "Lagos State"
    assert isinstance(nigria, NotFound)
    assert _labels(nigria) == ("Nigeria",)
    assert isinstance(guinea_bissau, NotFound)
    assert _labels(guinea_bissau) == ("Guinea-Bissau",)


def test_trailing_period_typo_recovers_via_the_article_and_punctuation_retry(shared_resolver):
    """The retry strips the period and finds country/KEN. The graph fixture lacks that node.

    NotHeld, not Resolved, shows the retry found the candidate and the fallback ladder did not.
    """
    graph = FakeGraphLookup()
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve(["Kenya."]))

    assert isinstance(result, NotHeld), result
    assert result.dcid == "country/KEN"


@pytest.mark.parametrize(
    ("text", "near_matches"),
    [
        ("Lagos", ("Laos",)),
        # The alias index matches exactly, so a code no row stores stays unmatched.
        ("0915", ("European Union (15 members, 1995\u20132004)",)),
    ],
)
def test_gate_invariant_near_match_reported_but_never_wins(
    shared_resolver, text: str, near_matches: tuple[str, ...]
):
    """A rejected candidate is reported as a near match, not returned as the result."""
    graph = FakeGraphLookup()
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve([text]))

    assert isinstance(result, NotFound)
    assert _labels(result) == near_matches


def test_every_stored_name_keeps_its_owner_past_the_gate(shared_resolver):
    """Each name and alias in the committed records and curated map yields its own row as a survivor."""
    records = json.loads(pack.records_path().read_text(encoding="utf-8"))
    curated = json.loads(pack.curated_path().read_text(encoding="utf-8"))
    pairs = [
        (name, f"{pack.NAMESPACE_PREFIX}{record['dcid']}")
        for record in records
        for name in [
            record["name"],
            *(record.get("aliases") or []),
            *curated.get(record["dcid"], []),
        ]
    ]
    assert pairs

    outcomes = shared_resolver.run(
        _resolve_batch_job, shared_resolver, None, [name for name, _ in pairs]
    ).result()

    unreachable = [
        name
        for (name, owner), outcome in zip(pairs, outcomes, strict=True)
        if owner not in {survivor.entity_id for survivor in outcome.survivors}
    ]
    assert unreachable == []


def test_alias_shared_by_a_country_code_and_an_organisation_is_ambiguous(shared_resolver):
    """`CAF` is the Central African Republic's ISO code and an alias of the CAF bank."""
    bank = "DevelopmentBankOfLatinAmericaAndTheCaribbean"
    graph = FakeGraphLookup(
        nodes={
            "country/CAF": {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME},
            bank: {"typeOf": ("Organization",), "name": _UNUSED_GRAPH_NAME},
        }
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve(["CAF"]))

    assert isinstance(result, Ambiguous), result
    assert [c.dcid for c in result.candidates] == ["country/CAF", bank]


def test_european_union_falls_back_to_the_organisation_storing_the_alias(shared_resolver):
    """Without a graph node for the union, the group takes the organisation that stores
    `European Union` as an alias for its entity."""
    member_ids = shared_resolver.run(
        _members_of_on_shared_thread, shared_resolver.resolver, "European Union"
    ).result()
    nodes = {dcid: {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME} for dcid in member_ids}
    nodes["EuropeanUnionInstitutions"] = {"typeOf": ("Organization",), "name": _UNUSED_GRAPH_NAME}
    resolver = EntityResolver(graph=FakeGraphLookup(nodes=nodes), shared=shared_resolver)

    (result,) = _run(resolver.resolve(["European Union"]))

    assert isinstance(result, Group), result
    assert result.entity is not None
    assert result.entity.dcid == "EuropeanUnionInstitutions"
    assert [e.dcid for e in result.members] == sorted(member_ids)


@pytest.mark.parametrize("text", ["OECD", "G7"])
def test_geo_organization_grouping_resolves_group(shared_resolver, text: str):
    """The geo pack returns a group id, and the registry confirms the group and its members:
    a graph holding none of them still yields the Group, and no graph read names a member.

    Member ids come from resolvekit's offline members_of lookup.
    """
    member_ids = shared_resolver.run(
        _members_of_on_shared_thread, shared_resolver.resolver, text
    ).result()
    assert member_ids
    graph = FakeGraphLookup()
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve([text]))

    assert isinstance(result, Group), (text, result)
    assert result.entity is None
    assert [e.dcid for e in result.members] == sorted(member_ids)
    assert all(e.kind == "country" for e in result.members)
    read = {dcid for dcids, _ in graph.node_arcs_many_calls for dcid in dcids}
    assert not read & set(member_ids)


def _parse_group_survivors_on_shared_thread(resolver, text: str):
    context = ResolutionContext(entity_types=ADMITTED_TYPES)
    group_types = resolver._runner.available_group_types
    return _parse_group_survivors(resolver, text, context, group_types)


@pytest.mark.parametrize(
    ("text", "group_name", "reading"),
    [
        ("G7 countries", "G7", "entity"),
        ("the G7 countries", "G7", "entity"),
        ("G20 countries", "G20", "members"),
        ("OECD countries", "OECD", "entity"),
        ("European Union", "European Union", "entity"),
        ("EU countries", "European Union", "members"),
        ("EU member states", "European Union", "members"),
        ("European Union member states", "European Union", "members"),
        ("the EU member states.", "European Union", "members"),
        ("DAC members", "OECD Development Assistance Committee", "members"),
    ],
)
def test_group_phrase_recovers_the_bare_groups_membership(
    shared_resolver, text: str, group_name: str, reading: str
):
    """Long group phrases recover the bare group's membership.

    Parsing handles G7, G20, and OECD. Aliases handle EU and DAC phrases, and a group
    reached through an alias row that declares `members` says its text names them.
    """
    member_ids = shared_resolver.run(
        _members_of_on_shared_thread, shared_resolver.resolver, group_name
    ).result()
    assert member_ids
    graph = FakeGraphLookup(
        nodes={dcid: {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME} for dcid in member_ids}
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve([text]))

    assert isinstance(result, Group), (text, result)
    assert [e.dcid for e in result.members] == sorted(member_ids)
    assert result.reading == reading


def test_an_alias_that_declares_no_members_resolves_as_the_whole(shared_resolver):
    """ "Central Africa" resolves through an alias row with no `members`, so the result
    reads as the entity."""
    graph = FakeGraphLookup(
        nodes={"MiddleAfrica": {"typeOf": ("UNGeoRegion",), "name": _UNUSED_GRAPH_NAME}}
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve(["Central Africa"]))

    assert isinstance(result, Resolved), result
    assert result.entity.dcid == "MiddleAfrica"
    assert result.reading == "entity"


def test_group_node_dcids_are_the_nodes_a_group_resolves_to(shared_resolver):
    """The reverse lookup holds the dcid a group resolves to as its own node, and no
    member, country or continent."""
    member_ids = shared_resolver.run(
        _members_of_on_shared_thread, shared_resolver.resolver, "European Union"
    ).result()
    nodes = {dcid: {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME} for dcid in member_ids}
    nodes["EuropeanUnion"] = {"typeOf": ("ContinentalUnion",), "name": _UNUSED_GRAPH_NAME}
    graph = FakeGraphLookup(nodes=nodes)
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    async def resolve_and_look_up():
        (union,) = await resolver.resolve(["European Union"])
        return union, await resolver.group_node_dcids()

    union, group_nodes = _run(resolve_and_look_up())

    assert isinstance(union, Group) and union.entity is not None
    assert union.entity.dcid in group_nodes
    assert not group_nodes & {*(member.dcid for member in union.members), "africa"}
    # The one graph read is the resolve's confirmation: the lookup reads the packs alone.
    assert len(graph.node_arcs_many_calls) == 1


@pytest.mark.parametrize(
    "text",
    [
        "the EU",
        "the European Union",
        "european union.",
    ],
)
def test_article_prefixed_or_punctuated_eu_phrasing_recovers_the_groups_membership(
    shared_resolver, text: str
):
    """Parsing drops union spans. Retrying stripped phrases resolves the EU group."""
    member_ids = shared_resolver.run(
        _members_of_on_shared_thread, shared_resolver.resolver, "European Union"
    ).result()
    assert member_ids
    graph = FakeGraphLookup(
        nodes={dcid: {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME} for dcid in member_ids}
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve([text]))

    assert isinstance(result, Group), (text, result)
    assert [e.dcid for e in result.members] == sorted(member_ids)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("the EU", "EU"),
        ("The EU", "EU"),
        ("the European Union", "European Union"),
        ("european union.", "european union"),
        ("the G7", "G7"),
        ("the G7.", "G7"),
        ("Gambia", "Gambia"),
        ("the Gambia", "Gambia"),
        ("Theodore", "Theodore"),
        ("  the  EU  ", "EU"),
    ],
)
def test_without_article_and_punctuation(text: str, expected: str):
    """Retry-after-miss preserves Gambia. Theodore has no article boundary."""
    assert _without_article_and_punctuation(text) == expected


@pytest.mark.parametrize("text", ["Central Africa", "the Sudan region"])
def test_parse_group_survivors_rejects_a_country_span(shared_resolver, text: str):
    """Central Africa and the Sudan region parse as Country, outside the allowed group types."""
    survivors = shared_resolver.run(
        _parse_group_survivors_on_shared_thread, shared_resolver.resolver, text
    ).result()
    assert survivors == ()


def test_the_sudan_region_does_not_resolve_via_a_recovered_country_span(shared_resolver):
    """The unlisted phrase "the Sudan region" does not resolve through a country span."""
    graph = FakeGraphLookup(
        nodes={"country/SDN": {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME}}
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve(["the Sudan region"]))

    assert isinstance(result, NotFound), result


def test_world_bank_top_raw_candidate_is_a_group_the_gate_must_drop(shared_resolver):
    """Resolvekit ranks a low-income group first, but its names omit "World Bank".

    The listed-name gate drops it, leaving IBRD and IDA as ambiguous organisations.
    """
    raw = shared_resolver.run(
        _resolve_each_on_shared_thread, shared_resolver.resolver, ["World Bank"]
    ).result()["World Bank"]
    assert raw.candidates[0].entity_type == "geo.organization"

    ibrd = "InternationalBankForReconstructionAndDevelopment"
    ida = "InternationalDevelopmentAssociation"
    graph = FakeGraphLookup(
        nodes={
            ibrd: {"typeOf": ("Organization",), "name": _UNUSED_GRAPH_NAME},
            ida: {"typeOf": ("Organization",), "name": _UNUSED_GRAPH_NAME},
        }
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve(["World Bank"]))

    assert isinstance(result, Ambiguous), result
    assert {c.dcid for c in result.candidates} == {ibrd, ida}


def test_congo_ambiguous_with_no_fallback_call(shared_resolver):
    graph = FakeGraphLookup(
        nodes={
            "country/COD": {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME},
            "country/COG": {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME},
        }
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve(["Congo"]))

    assert isinstance(result, Ambiguous)
    assert {c.dcid for c in result.candidates} == {"country/COD", "country/COG"}
    assert graph.resolve_by_arc_calls == []


def test_ambiguous_candidates_ordered_by_dcid_not_entity_id(shared_resolver, monkeypatch):
    """Candidates sort by confidence then dcid, which can differ from entity-id order.

    The stub supplies a tied pair whose namespace stripping reverses that order.
    """
    import entity_resolution.resolver as resolver_module

    country = CandidateSummary(
        entity_id="country/AAA", confidence=0.9, canonical_name="AAA Land", pack_id="geo.countries"
    )
    org = CandidateSummary(
        entity_id="dfo_graph/ZZZOrg", confidence=0.9, canonical_name="ZZZ Org", pack_id="org"
    )

    def fake_batch_job(resolver, code_index, texts):
        return [_GateOutcome(survivors=(country, org), near_matches=())]

    monkeypatch.setattr(resolver_module, "_resolve_batch_job", fake_batch_job)

    graph = FakeGraphLookup(
        nodes={
            "country/AAA": {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME},
            "ZZZOrg": {"typeOf": ("Organization",), "name": _UNUSED_GRAPH_NAME},
        }
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve(["Tied Entities"]))

    assert isinstance(result, Ambiguous)
    assert [c.dcid for c in result.candidates] == ["ZZZOrg", "country/AAA"]


def test_batch_of_five_hits_makes_one_confirmation_call_and_no_fallback_calls(shared_resolver):
    graph = FakeGraphLookup(
        nodes={
            "country/KEN": {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME},
            "country/FRA": {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME},
            "africa": {"typeOf": ("Continent",), "name": _UNUSED_GRAPH_NAME},
            "AfricanDevelopmentBank": {"typeOf": ("Organization",), "name": _UNUSED_GRAPH_NAME},
            "InternationalDevelopmentAssociation": {
                "typeOf": ("Organization",),
                "name": _UNUSED_GRAPH_NAME,
            },
        }
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    results = _run(resolver.resolve(["Kenya", "France", "Africa", "AfDB", "IDA"]))

    assert all(isinstance(r, Resolved) for r in results)
    assert len(graph.node_arcs_many_calls) == 1
    assert graph.resolve_by_arc_calls == []


def test_batch_with_two_misses_makes_at_most_four_fallback_calls(shared_resolver):
    graph = FakeGraphLookup()
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    results = _run(resolver.resolve(["Nigria", "Guinea Bissau"]))

    assert all(isinstance(r, NotFound) for r in results)
    assert len(graph.resolve_by_arc_calls) <= 4


def test_fallback_ambiguous_when_the_answering_arc_names_two_held_dcids(shared_resolver):
    """Two held dcids list Twin Peaks, so the fallback result is ambiguous."""
    graph = FakeGraphLookup(
        nodes={
            "dcid/first": {"typeOf": ("Country",), "name": ("Twin Peaks",)},
            "dcid/second": {"typeOf": ("Country",), "name": ("Twin Peaks",)},
        },
        arcs={("<-description->dcid", "Twin Peaks"): ["dcid/first", "dcid/second"]},
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve(["Twin Peaks"]))

    assert isinstance(result, Ambiguous)
    assert [c.dcid for c in result.candidates] == ["dcid/first", "dcid/second"]


def test_fallback_resolves_to_the_one_candidate_the_graph_still_holds(shared_resolver):
    """Only one named dcid confirms, so the dangling candidate is dropped."""
    graph = FakeGraphLookup(
        nodes={"dcid/first": {"typeOf": ("Country",), "name": ("Twin Peaks",)}},
        arcs={("<-description->dcid", "Twin Peaks"): ["dcid/first", "dcid/second"]},
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve(["Twin Peaks"]))

    assert isinstance(result, Resolved)
    assert result.entity.dcid == "dcid/first"


def test_fallback_not_found_when_neither_named_dcid_confirms(shared_resolver):
    graph = FakeGraphLookup(
        arcs={("<-description->dcid", "Nigria"): ["dcid/first", "dcid/second"]},
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve(["Nigria"]))

    assert isinstance(result, NotFound)
    assert _labels(result) == ("Nigeria",)


def test_fallback_drops_a_description_match_and_keeps_the_listed_spelling(shared_resolver):
    """The arc answers both spellings, but only listed name "Nigeria" resolves."""
    graph = FakeGraphLookup(
        nodes={"country/NGA": {"typeOf": ("Country",), "name": ("Nigeria",)}},
        arcs={
            ("<-description->dcid", "Nigria"): ["country/NGA"],
            ("<-description->dcid", "Nigeria"): ["country/NGA"],
        },
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    typo, listed = _run(resolver.resolve(["Nigria", "Nigeria"]))

    assert isinstance(typo, NotFound)
    assert _labels(typo) == ("Nigeria",)
    assert isinstance(listed, Resolved)
    assert listed.entity.dcid == "country/NGA"


def test_fallback_resolves_a_text_the_answering_node_lists_as_an_alternate_name(shared_resolver):
    """Lagos resolves as a listed alternate. Lagoss stays NotFound."""
    graph = FakeGraphLookup(
        nodes={
            "geoId/lagos": {
                "typeOf": ("AdministrativeArea1",),
                "name": ("Lagos State",),
                "alternateName": ("Lagos",),
            }
        },
        arcs={
            ("<-description->dcid", "Lagos"): ["geoId/lagos"],
            ("<-description->dcid", "Lagoss"): ["geoId/lagos"],
        },
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    result, unlisted = _run(resolver.resolve(["Lagos", "Lagoss"]))

    assert isinstance(result, Resolved)
    assert result.entity.dcid == "geoId/lagos"
    assert result.entity.name == "Lagos State"
    assert isinstance(unlisted, NotFound)


def test_fallback_resolves_a_healthy_text_alongside_an_unnormalisable_one(shared_resolver):
    """The geo normaliser rejects ">" without interrupting Lagos's resolution in the batch."""
    graph = FakeGraphLookup(
        nodes={
            "geoId/lagos": {
                "typeOf": ("AdministrativeArea1",),
                "name": ("Lagos State",),
                "alternateName": ("Lagos",),
            }
        },
        arcs={
            ("<-description->dcid", "Lagos"): ["geoId/lagos"],
            ("<-description->dcid", ">"): ["geoId/lagos"],
        },
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    healthy, refused = _run(resolver.resolve(["Lagos", ">"]))

    assert isinstance(healthy, Resolved)
    assert healthy.entity.dcid == "geoId/lagos"
    assert isinstance(refused, NotFound)


def test_fallback_resolves_only_the_spelling_the_answering_node_lists(shared_resolver):
    """The normaliser keeps hyphens significant, so only the listed hyphenated name resolves."""
    graph = FakeGraphLookup(
        nodes={"country/GNB": {"typeOf": ("Country",), "name": ("Guinea-Bissau",)}},
        arcs={
            ("<-description->dcid", "Guinea-Bissau"): ["country/GNB"],
            ("<-description->dcid", "Guinea Bissau"): ["country/GNB"],
        },
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    spaced, hyphenated = _run(resolver.resolve(["Guinea Bissau", "Guinea-Bissau"]))

    assert isinstance(spaced, NotFound)
    assert _labels(spaced) == ("Guinea-Bissau",)
    assert isinstance(hyphenated, Resolved)
    assert hyphenated.entity.dcid == "country/GNB"


def test_lru_avoids_a_second_resolvekit_job_for_a_cached_text(shared_resolver, monkeypatch):
    import entity_resolution.resolver as resolver_module

    calls: list[list[str]] = []
    original = resolver_module._resolve_batch_job

    def counting(resolver, code_index, texts):
        calls.append(list(texts))
        return original(resolver, code_index, texts)

    monkeypatch.setattr(resolver_module, "_resolve_batch_job", counting)

    graph = FakeGraphLookup(
        nodes={"country/KEN": {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME}}
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    first = _run(resolver.resolve(["Kenya"]))
    second = _run(resolver.resolve(["Kenya"]))

    assert first == second
    assert len(calls) == 1


class _RaisesOnceThenDelegates(FakeGraphLookup):
    """Raise on the first node_arcs_many call, then delegate."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._raise_next = True

    def node_arcs_many(
        self,
        dcids,
        *,
        properties,
        deadline: float | None = None,
    ):
        if self._raise_next:
            self._raise_next = False
            raise RuntimeError("simulated graph failure")
        return super().node_arcs_many(dcids, properties=properties)


def test_an_exception_from_the_graph_is_not_cached(shared_resolver):
    graph = _RaisesOnceThenDelegates(
        nodes={"country/KEN": {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME}}
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    with pytest.raises(RuntimeError):
        _run(resolver.resolve(["Kenya"]))

    (result,) = _run(resolver.resolve(["Kenya"]))
    assert isinstance(result, Resolved)
    assert result.entity.dcid == "country/KEN"


def test_empty_and_whitespace_text_yields_not_found_without_any_graph_call(shared_resolver):
    graph = FakeGraphLookup()
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    results = _run(resolver.resolve(["", "   "]))

    assert all(isinstance(r, NotFound) and r.near_matches == () for r in results)
    assert graph.node_arcs_many_calls == []
    assert graph.resolve_by_arc_calls == []


def test_resolve_passes_the_callers_deadline_to_every_graph_read(shared_resolver):
    """The same deadline reaches resolve_by_arc, node_names_many, and node_arcs_many.

    Qqqlandia reaches the fallback ladder and triggers all three reads.
    """
    graph = FakeGraphLookup(
        nodes={"dcid/one": {"typeOf": ("Country",), "name": ("Qqqlandia",)}},
        arcs={("<-description{typeOf:Country}->dcid", "Qqqlandia"): ["dcid/one"]},
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)
    deadline = time.monotonic() + 5.0

    results = _run(resolver.resolve(["Qqqlandia"], deadline=deadline))

    assert [r.entity.dcid for r in results if isinstance(r, Resolved)] == ["dcid/one"]
    assert graph.deadline_calls, "no graph read recorded a deadline"
    assert {name for name, _ in graph.deadline_calls} == {
        "resolve_by_arc",
        "node_names_many",
        "node_arcs_many",
    }
    assert [recorded for _, recorded in graph.deadline_calls] == [deadline] * len(
        graph.deadline_calls
    )


def test_resolve_without_a_deadline_passes_none_to_every_graph_read(shared_resolver):
    """Calls without a deadline pass None to each graph read."""
    graph = FakeGraphLookup(nodes={"country/KEN": {"typeOf": ("Country",)}})
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    results = _run(resolver.resolve(["Kenya"]))

    assert all(isinstance(r, Resolved) for r in results)
    assert [recorded for _, recorded in graph.deadline_calls] == [None] * len(graph.deadline_calls)


def test_stale_write_race_lands_under_the_stamp_its_own_call_captured(shared_resolver):
    """A blocked gen-a read writes under its captured key after gen-b advances.

    Restoring gen-a serves Kenya without another graph read.
    """
    release_confirmation = threading.Event()
    confirmation_started = threading.Event()

    class BlockingGraph(FakeGraphLookup):
        def node_arcs_many(self, dcids, *, properties, deadline=None):
            if "country/KEN" in dcids:
                confirmation_started.set()
                assert release_confirmation.wait(timeout=5), "test deadlocked"
            return super().node_arcs_many(dcids, properties=properties, deadline=deadline)

    graph = BlockingGraph(
        nodes={"country/KEN": {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME}}
    )
    generation = FakeGeneration(token="gen-a")
    resolver = EntityResolver(graph=graph, shared=shared_resolver, generation=generation)

    async def request_a():
        return await resolver.resolve(["Kenya"])

    async def request_b():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, confirmation_started.wait, 5)
        generation.advance("gen-b")
        await resolver.resolve(["some other text not otherwise cached"])
        release_confirmation.set()

    async def race():
        task_a = asyncio.create_task(request_a())
        task_b = asyncio.create_task(request_b())
        (result_a,) = await task_a
        await task_b
        return result_a

    result_a = _run(race())

    assert isinstance(result_a, Resolved)
    assert result_a.entity.dcid == "country/KEN"

    calls_before = len(graph.node_arcs_many_calls)
    generation.advance("gen-a")
    (again,) = _run(resolver.resolve(["Kenya"]))
    assert isinstance(again, Resolved)
    assert len(graph.node_arcs_many_calls) == calls_before


def _counting_resolve_batch_job(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record each `_resolve_batch_job` call for cache-hit assertions."""
    import entity_resolution.resolver as resolver_module

    calls: list[list[str]] = []
    original = resolver_module._resolve_batch_job

    def counting(resolver, code_index, texts):
        calls.append(list(texts))
        return original(resolver, code_index, texts)

    monkeypatch.setattr(resolver_module, "_resolve_batch_job", counting)
    return calls


def test_hit_under_the_same_generation_token_avoids_a_second_resolvekit_job(
    shared_resolver, monkeypatch
):
    calls = _counting_resolve_batch_job(monkeypatch)
    graph = FakeGraphLookup(
        nodes={"country/KEN": {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME}}
    )
    generation = FakeGeneration(token="gen-a")
    resolver = EntityResolver(graph=graph, shared=shared_resolver, generation=generation)

    first = _run(resolver.resolve(["Kenya"]))
    second = _run(resolver.resolve(["Kenya"]))

    assert first == second
    assert len(calls) == 1


def test_advancing_the_generation_token_misses_a_previously_cached_text(
    shared_resolver, monkeypatch
):
    calls = _counting_resolve_batch_job(monkeypatch)
    graph = FakeGraphLookup(
        nodes={"country/KEN": {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME}}
    )
    generation = FakeGeneration(token="gen-a")
    resolver = EntityResolver(graph=graph, shared=shared_resolver, generation=generation)

    _run(resolver.resolve(["Kenya"]))
    generation.advance("gen-b")
    _run(resolver.resolve(["Kenya"]))

    assert len(calls) == 2


def test_unknown_and_known_generations_never_share_a_cache_entry(shared_resolver, monkeypatch):
    calls = _counting_resolve_batch_job(monkeypatch)
    graph = FakeGraphLookup(
        nodes={"country/KEN": {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME}}
    )
    generation = FakeGeneration()  # starts unknown
    resolver = EntityResolver(graph=graph, shared=shared_resolver, generation=generation)

    _run(resolver.resolve(["Kenya"]))  # unknown-generation key
    generation.advance("gen-a")
    _run(resolver.resolve(["Kenya"]))  # known-generation key

    assert len(calls) == 2


def test_stats_reports_entries_and_clear_empties_the_cache(shared_resolver):
    graph = FakeGraphLookup(
        nodes={"country/KEN": {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME}}
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    assert resolver.stats() == {"hits": 0, "misses": 0, "entries": 0}
    _run(resolver.resolve(["Kenya"]))
    assert resolver.stats() == {"hits": 0, "misses": 1, "entries": 1}
    _run(resolver.resolve(["Kenya"]))
    assert resolver.stats() == {"hits": 1, "misses": 1, "entries": 1}

    resolver.clear()
    assert resolver.stats() == {"hits": 1, "misses": 1, "entries": 0}


def test_ttu_uses_the_known_lifetime_for_a_real_token():
    assert _ttu(("gen-a", "kenya"), None, 100.0) == 100.0 + GRAPH_CACHE_MAX_AGE_SECONDS


def test_ttu_uses_the_unknown_lifetime_for_the_unknown_key():
    assert _ttu((UNKNOWN_KEY, "kenya"), None, 100.0) == (
        100.0 + GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS
    )


class _LockAssertingCache(TLRUCache):
    """A TLRUCache that asserts EntityResolver holds `_cache_lock` on every cache operation."""

    def __init__(self, *, owner: EntityResolver, maxsize: int, ttu: Callable[..., float]) -> None:
        super().__init__(maxsize=maxsize, ttu=ttu)
        self._owner = owner

    def _assert_locked(self) -> None:
        assert self._owner._cache_lock.locked(), "cache accessed without _cache_lock held"

    def get(self, key, default=None):
        self._assert_locked()
        return super().get(key, default)

    def __getitem__(self, key):
        self._assert_locked()
        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        self._assert_locked()
        super().__setitem__(key, value)

    def __contains__(self, key) -> bool:
        self._assert_locked()
        return super().__contains__(key)

    def __len__(self) -> int:
        self._assert_locked()
        return super().__len__()

    def clear(self) -> None:
        self._assert_locked()
        super().clear()


def test_every_cache_access_holds_the_lock(shared_resolver):
    """Miss, hit, stats, and clear exercise every cache access with the lock assertion active."""
    graph = FakeGraphLookup(
        nodes={"country/KEN": {"typeOf": ("Country",), "name": _UNUSED_GRAPH_NAME}}
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)
    resolver._cache = _LockAssertingCache(owner=resolver, maxsize=4096, ttu=_ttu)

    _run(resolver.resolve(["Kenya"]))
    _run(resolver.resolve(["Kenya"]))
    resolver.stats()
    resolver.clear()
