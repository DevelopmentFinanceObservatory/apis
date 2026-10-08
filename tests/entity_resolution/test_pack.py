"""Offline tests for the committed pack, its builder, and raw resolvekit rankings.

The session resolver combines the dfo_graph pack with bundled geo packs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from resolvekit import ResolutionContext, Resolver
from resolvekit.core.api import RoutingMode
from resolvekit.core.module_registry import register_module

from entity_resolution import pack

_MODULE_IDS = (
    "geo.countries",
    "geo.regions",
    "geo.continents",
    "geo.continental_unions",
    pack.MODULE_ID,
)
_FULL_CONTEXT = ResolutionContext(
    entity_types=frozenset(
        {
            "geo.country",
            "geo.subregion",
            "geo.continent",
            "geo.continental_union",
            "org.organisation",
            "org.dac_aggregate",
        }
    )
)


def _pack_id(dcid: str) -> str:
    return f"{pack.NAMESPACE_PREFIX}{dcid}"


@pytest.fixture(scope="session")
def composed_resolver(tmp_path_factory: pytest.TempPathFactory):
    cache_dir = tmp_path_factory.mktemp("dfo_graph_pack")
    os.environ["RESOLVEKIT_OFFLINE"] = "1"
    os.environ["RESOLVEKIT_CACHE_DIR"] = str(cache_dir)

    pack_dir = pack.build_pack(
        records_path=pack.records_path(), curated_path=pack.curated_path(), out_dir=cache_dir
    )
    register_module(pack.MODULE_ID, pack_dir)
    resolver = Resolver.lite(
        module_ids=list(_MODULE_IDS), routing_mode=RoutingMode.EXPLICIT, cache_size=0
    )
    yield resolver
    resolver.close()


def test_committed_records_have_expected_population():
    records = json.loads(pack.records_path().read_text())
    organisations = [r for r in records if r["kind"] == "organisation"]
    aggregates = [r for r in records if r["kind"] == "aggregate"]
    assert len(organisations) == 195
    assert len(aggregates) == 37

    dcids = [r["dcid"] for r in records]
    assert len(dcids) == len(set(dcids)), "committed records carry a duplicate dcid"
    dcids_by_name: dict[str, set[str]] = {}
    for r in records:
        dcids_by_name.setdefault(r["name"], set()).add(r["dcid"])
    duplicates = {name: ids for name, ids in dcids_by_name.items() if len(ids) > 1}
    assert not duplicates, f"committed records carry a duplicate name: {duplicates}"


def test_curated_aliases_name_only_known_dcids():
    records = json.loads(pack.records_path().read_text())
    known_dcids = {r["dcid"] for r in records}
    curated = json.loads(pack.curated_path().read_text())
    unknown = set(curated) - known_dcids
    assert not unknown, f"curated_aliases.json names dcid(s) absent from the records: {unknown}"


def test_build_pack_returns_directory_with_module_metadata(tmp_path: Path):
    pack_dir = pack.build_pack(
        records_path=pack.records_path(), curated_path=pack.curated_path(), out_dir=tmp_path
    )
    metadata = json.loads((pack_dir / "metadata.json").read_text())
    assert metadata["module_id"] == pack.MODULE_ID


def test_build_pack_returns_the_pack_it_built_beside_a_stale_one(tmp_path: Path):
    """A cache directory can hold packs other inputs built; the one returned is ours."""
    built = pack.build_pack(
        records_path=pack.records_path(), curated_path=pack.curated_path(), out_dir=tmp_path
    )
    stale = tmp_path / "byod" / "0000-stale"
    stale.mkdir()
    (stale / "metadata.json").write_text((built / "metadata.json").read_text())

    again = pack.build_pack(
        records_path=pack.records_path(), curated_path=pack.curated_path(), out_dir=tmp_path
    )

    assert again == built
    assert built.parent == tmp_path / "byod"


def test_build_pack_refuses_a_curated_alias_naming_an_unknown_dcid(tmp_path: Path):
    records_path = tmp_path / "records.json"
    records_path.write_text(
        json.dumps(
            [
                {
                    "dcid": "KnownOrg",
                    "name": "Known Org",
                    "kind": "organisation",
                    "aliases": [],
                    "dac_code": None,
                }
            ]
        )
    )
    curated_path = tmp_path / "curated.json"
    curated_path.write_text(json.dumps({"UnknownOrg": ["Alias"]}))

    with pytest.raises(pack.CuratedAliasError):
        pack.build_pack(
            records_path=records_path, curated_path=curated_path, out_dir=tmp_path / "out"
        )


@pytest.mark.parametrize(
    ("query", "expected_dcid"),
    [
        ("AfDB", "AfricanDevelopmentBank"),
        ("Africa, regional", "DACAfricaUnallocated"),
        ("913", "AfricanDevelopmentBank"),
        # A node's lowest DAC code is its code; every other one is an alias.
        ("915", "AsianDevelopmentBank"),
        ("20015", "AsianDevelopmentBank"),
        ("907", "InternationalMonetaryFund"),
        ("905", "InternationalDevelopmentAssociation"),
        ("996", "EuropeanUnionInstitutions"),
        ("IMF", "InternationalMonetaryFund"),
        ("WHO", "WorldHealthOrganization"),
        ("Asian Development Bank", "AsianDevelopmentBank"),
    ],
)
def test_pack_resolves_expected_top_candidate(composed_resolver, query: str, expected_dcid: str):
    result = composed_resolver.resolve(
        query, as_result=True, context=_FULL_CONTEXT, include_entity=True
    )
    assert result.candidates, f"{query!r} produced no candidates"
    assert result.candidates[0].entity_id == _pack_id(expected_dcid)


def test_asian_development_bank_is_answered_by_our_pack_not_a_loose_geo_region(composed_resolver):
    # _FULL_CONTEXT excludes loose geo.region entries, leaving this match to our pack.
    result = composed_resolver.resolve(
        "Asian Development Bank", as_result=True, context=_FULL_CONTEXT, include_entity=True
    )
    assert result.candidates[0].entity_id == _pack_id("AsianDevelopmentBank")


def test_world_bank_candidates_include_both_ibrd_and_ida(composed_resolver):
    result = composed_resolver.resolve(
        "World Bank", as_result=True, context=_FULL_CONTEXT, include_entity=True
    )
    candidate_ids = {c.entity_id for c in (result.candidates or [])}
    assert _pack_id("InternationalBankForReconstructionAndDevelopment") in candidate_ids
    assert _pack_id("InternationalDevelopmentAssociation") in candidate_ids


def test_region_ids_cover_every_resolvekit_subregion_and_continent(composed_resolver):
    rows = json.loads(pack.region_ids_path().read_text())
    assert len(rows) == 31  # 22 subregions + 9 continents

    for row in rows:
        assert composed_resolver.entity(row["resolvekit_id"]) is not None, (
            f"{row['resolvekit_id']!r} ({row['name']!r}) is not a real entity in the "
            "composed resolver"
        )

    africa_rows = [row for row in rows if row["name"] == "Africa"]
    assert len(africa_rows) == 1
    assert africa_rows[0]["dcid"] == "africa"
