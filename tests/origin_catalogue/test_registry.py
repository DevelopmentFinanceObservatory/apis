"""Tests the packaged registry's contents, its packaging, and validation refusals."""

from __future__ import annotations

import importlib.resources
import tomllib
from pathlib import Path

import pytest

from origin_catalogue import ScanLevel
from origin_catalogue.registry import (
    RegistryError,
    load_registry,
    packaged_registry,
    parse_registry,
    registry_hash,
)

_PYPROJECT = Path(__file__).resolve().parents[2] / "packages" / "dfo-data-tools" / "pyproject.toml"

_UN_SEGMENTS = {
    "undata/p/WHO": "who",
    "undata/p/UNICEF": "unicef",
    "undata/p/ILO": "ilo",
    "undata/p/UNHCR": "unhcr",
    "undata/p/UNESCO": "unesco",
    "undata/p/UNIDO": "unido",
    "undata/p/ITU": "itu",
    "undata/p/ECLAC": "eclac",
    "undata/p/IOM_DTM": "iomdtm",
    "undata/p/UNAIDS": "unaids",
    "undata/p/UNDP_HDRO": "undphdro",
    "undata/p/UNFPA": "unfpa",
    "undata/p/UNODC": "unodc",
    "undata/p/SDG": "sdg",
}


def test_registry_loads_through_importlib_resources() -> None:
    resource = importlib.resources.files("origin_catalogue") / "data" / "registry.toml"
    assert resource.is_file()
    assert packaged_registry().read_bytes() == resource.read_bytes()
    assert parse_registry(resource.read_bytes()) == load_registry()


def test_registry_module_is_declared_for_the_wheel() -> None:
    """uv_build ships a module's data files only for modules the build config names."""
    config = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    assert "origin_catalogue" in config["tool"]["uv"]["build-backend"]["module-name"]


def test_registry_holds_26_base_sources_and_14_un_agencies() -> None:
    registry = load_registry()
    assert len(registry.base) == 26
    assert {agency.provenance: agency.segment for agency in registry.un_system} == _UN_SEGMENTS


def test_registry_records_the_world_bank_exclusion_and_two_reviews() -> None:
    by_source = {entry.source: entry for entry in load_registry().base}

    (world_bank,) = by_source["dc/s/WorldBank"].exclusions or ()
    assert (world_bank.provenance, world_bank.reason) == (
        "dc/base/WorldBank_IDS",
        "superseded_by_dfo",
    )

    assert by_source["dc/s/UnitedNationsUn"].exclusions == ()
    oecd = by_source["dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd"]
    assert oecd.exclusions == ()
    unreviewed = [p for p, entry in by_source.items() if entry.exclusions is None]
    assert len(unreviewed) == 23


def test_sources_name_the_level_they_scan() -> None:
    registry = load_registry()
    by_source = {entry.source: entry for entry in registry.base}
    assert by_source["dc/s/StatisticsCanada"].scan_level == ScanLevel(
        place_type="AdministrativeArea1", within="country/CAN"
    )
    assert by_source["dc/s/SouthafricaBudget"].scan_level == ScanLevel(
        place_type="AdministrativeArea2", within="country/ZAF"
    )
    assert by_source["dc/s/EuropeanUnionEuEurostat"].scan_level == ScanLevel(
        place_type="EurostatNUTS2"
    )
    singapore = by_source["dc/s/DepartmentOfStatisticsSingapore"]
    assert (singapore.country, singapore.scan_level) == ("country/SGP", None)
    national = [entry for entry in registry.base if entry.country is not None]
    assert len(national) == 17
    levels = registry.scan_levels()
    assert len(levels) == 17
    assert levels["dc/s/UsBureauOfLaborStatisticsBls"] == (
        ScanLevel(place_type="State", within="country/USA"),
    )
    assert "dc/s/WorldBank" not in levels


def test_scan_type_without_a_country_scans_everywhere() -> None:
    registry = parse_registry(
        ('[[base]]\nsource = "dc/s/A"\nlabel = "A"\nscan_type = "EurostatNUTS2"\n' + _UN).encode()
    )
    assert registry.scan_levels() == {"dc/s/A": (ScanLevel(place_type="EurostatNUTS2"),)}


def test_registry_hash_is_the_sha256_of_the_file_bytes() -> None:
    raw = packaged_registry().read_bytes()
    assert registry_hash(raw) == registry_hash(bytes(raw))
    assert registry_hash(raw) != registry_hash(raw + b"\n")
    assert len(registry_hash(raw)) == 64


_BASE = """
[[base]]
source = "dc/s/WorldBank"
label = "World Bank"
"""
_UN = """
[[un_system]]
provenance = "undata/p/SDG"
segment = "sdg"
label = "SDG Global Database"
"""


@pytest.mark.parametrize(
    ("toml", "message"),
    [
        (_BASE + _BASE + _UN, "duplicate base source"),
        (_BASE + _UN + _UN, "duplicate UN provenance"),
        (
            _BASE
            + _UN
            + '[[un_system]]\nprovenance = "undata/p/SDG2"\nsegment = "sdg"\nlabel = "x"\n',
            "duplicate UN segment",
        ),
        (
            '[[base]]\nsource = "dc/s/A"\nlabel = "A"\n'
            'exclusions = [{ provenance = "dc/base/X", reason = "superseded_by_dfo" }]\n'
            '[[base]]\nsource = "dc/s/B"\nlabel = "B"\n'
            'exclusions = [{ provenance = "dc/base/X", reason = "superseded_by_dfo" }]\n' + _UN,
            "duplicate excluded provenance",
        ),
        (
            '[[base]]\nsource = "dc/s/A"\nlabel = "A"\n'
            'exclusions = [{ provenance = "dc/base/X", reason = "served_by", '
            'served_by = { data_commons = "un_system", provenance = "undata/p/SDG" } }]\n' + _UN,
            r"exclusions\.0\.reason\n  Input should be 'superseded_by_dfo'",
        ),
        (
            '[[base]]\nsource = "dc/s/A"\nlabel = "A"\n'
            'exclusions = [{ provenance = "dc/base/X", reason = "superseded_by_dfo", '
            'served_by = { data_commons = "dfo", provenance = "provenance/Y" } }]\n' + _UN,
            r"exclusions\.0\.served_by\n  Extra inputs are not permitted",
        ),
        (
            '[[base]]\nsource = "dc/s/A"\nlabel = "A"\n'
            'exclusions = [{ provenance = "undata/p/SDG", reason = "superseded_by_dfo" }]\n' + _UN,
            "is a UN provenance, not a base one",
        ),
        (
            '[[base]]\nsource = "dc/s/A"\nlabel = "A"\n'
            'exclusions = [{ provenance = "dc/base/X", reason = "withdrawn" }]\n' + _UN,
            "reason",
        ),
        ('[[base]]\nsource = "WorldBank"\nlabel = "World Bank"\n' + _UN, "source"),
        ('[[base]]\nsource = "dc/s/A"\nlabel = "A"\ncountry = "USA"\n' + _UN, "country"),
        ('[[base]]\nsource = "dc/s/A"\nlabel = "A"\nscan_type = "state"\n' + _UN, "scan_type"),
        (
            _BASE + '[[un_system]]\nprovenance = "undata/p/X"\nsegment = "sdgf"\nlabel = "x"\n',
            "reserves for another id family",
        ),
        (
            _BASE + '[[un_system]]\nprovenance = "undata/p/X"\nsegment = "UN_X"\nlabel = "x"\n',
            "segment",
        ),
        (_BASE + _UN + "extra = 1\n", "extra"),
        (_BASE + '[[base]]\nsource = "dc/s/A"\n' + _UN, "label"),
        ("[[base]\n", "not valid UTF-8 TOML"),
    ],
)
def test_parse_registry_refuses_malformed_entries(toml: str, message: str) -> None:
    with pytest.raises(RegistryError, match=message):
        parse_registry(toml.encode("utf-8"))


def test_parse_registry_accepts_a_minimal_registry() -> None:
    registry = parse_registry((_BASE + _UN).encode("utf-8"))
    assert [entry.source for entry in registry.base] == ["dc/s/WorldBank"]
    assert registry.base[0].exclusions is None
