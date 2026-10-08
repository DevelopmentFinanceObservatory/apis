"""Tests the packaged registry's contents, its packaging, and validation refusals."""

from __future__ import annotations

import importlib.resources
import tomllib
from collections import Counter
from pathlib import Path

import pytest

from origin_catalogue import ScanLevel
from origin_catalogue.registry import (
    DfoSupersededProvenance,
    RegistryError,
    UnSupersededProvenance,
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


def test_registry_records_two_supersessions_and_one_review() -> None:
    registry = load_registry()
    by_source = {entry.source: entry for entry in registry.base}

    assert by_source["dc/s/WorldBank"].exclusions == (
        DfoSupersededProvenance(provenance="dc/base/WorldBank_IDS", reason="superseded_by_dfo"),
    )

    sdg = UnSupersededProvenance(
        provenance="dc/base/UN_SDG",
        reason="superseded_by_un_system",
        successor="undata/p/SDG",
        variable_prefix="sdg/",
        code_delimiter=".",
        pair_delimiter="__",
        value_delimiter="--",
    )
    assert by_source["dc/s/UnitedNationsUn"].exclusions == (sdg,)
    oecd = by_source["dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd"]
    assert oecd.exclusions == ()
    unreviewed = [p for p, entry in by_source.items() if entry.exclusions is None]
    assert len(unreviewed) == 23


def test_un_agencies_exclude_the_series_derived_from_dfo_data() -> None:
    by_segment = {agency.segment: agency for agency in load_registry().un_system}
    counts = {
        segment: Counter(exclusion.dfo_provenance for exclusion in agency.exclusions)
        for segment, agency in by_segment.items()
        if agency.exclusions
    }
    assert counts == {
        "sdg": {
            "provenance/OECDCreditorReportingSystem": 13,
            "provenance/OECDDAC1": 16,
            "provenance/WorldBankInternationalDebtStatistics": 2,
        },
        "who": {"provenance/OECDCreditorReportingSystem": 2},
        "unicef": {"provenance/OECDDAC1": 4},
    }
    for segment, agency in by_segment.items():
        for exclusion in agency.exclusions:
            assert exclusion.series.startswith(f"undata/{segment}/SERIES-")
            assert exclusion.reason == "superseded_by_dfo"
    assert "undata/sdg/SERIES-DT_TDS_DECT" in {e.series for e in by_segment["sdg"].exclusions}
    # Climate finance, TOSSD and a public debt measure: DFO imports none of
    # their sources, so they stay admitted.
    assert not {e.series for e in by_segment["sdg"].exclusions} & {
        "undata/sdg/SERIES-DC_FIN_CLIMT",
        "undata/sdg/SERIES-DC_OSSD_GRT",
        "undata/sdg/SERIES-DP_DOD_DLD2_CR_CG_Z1",
    }


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
_WHO = """
[[un_system]]
provenance = "undata/p/WHO"
segment = "who"
label = "World Health Organization (WHO)"
"""


def _superseded(**fields: str) -> str:
    """A base entry whose one exclusion is superseded by the UN, with `fields`
    replacing the valid defaults. An empty value drops the key."""
    values = {
        "provenance": "dc/base/X",
        "reason": "superseded_by_un_system",
        "successor": "undata/p/SDG",
        "variable_prefix": "x/",
        "code_delimiter": ".",
        "pair_delimiter": "__",
        "value_delimiter": "--",
    } | fields
    body = ", ".join(f'{key} = "{value}"' for key, value in values.items() if value)
    return f'[[base]]\nsource = "dc/s/A"\nlabel = "A"\nexclusions = [{{ {body} }}]\n'


def _series(*series: str, reason: str = "superseded_by_dfo", extra: str = "") -> str:
    """The SDG agency excluding `series`, each naming a DFO provenance."""
    rows = ", ".join(
        f'{{ series = "{one}", reason = "{reason}", dfo_provenance = "provenance/Y"{extra} }}'
        for one in series
    )
    return _UN + f"exclusions = [{rows}]\n"


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
            "Input tag 'served_by' found using 'reason' does not match any of the expected tags",
        ),
        (
            '[[base]]\nsource = "dc/s/A"\nlabel = "A"\n'
            'exclusions = [{ provenance = "dc/base/X", reason = "superseded_by_dfo", '
            'served_by = { data_commons = "dfo", provenance = "provenance/Y" } }]\n' + _UN,
            r"exclusions\.0\.superseded_by_dfo\.served_by\n  Extra inputs are not permitted",
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
        (
            _superseded(successor="undata/p/WHO") + _UN,
            "names successor 'undata/p/WHO', which is not a registered UN provenance",
        ),
        (_superseded(successor="dc/base/Y") + _UN, "successor"),
        (
            _superseded(successor="") + _UN,
            r"exclusions\.0\.superseded_by_un_system\.successor\n  Field required",
        ),
        (
            _superseded(variable_prefix="") + _UN,
            r"exclusions\.0\.superseded_by_un_system\.variable_prefix\n  Field required",
        ),
        (_superseded(variable_prefix="x") + _UN, "variable_prefix"),
        (
            _superseded(code_delimiter="") + _UN,
            r"exclusions\.0\.superseded_by_un_system\.code_delimiter\n  Field required",
        ),
        (_superseded(code_delimiter="a") + _UN, "code_delimiter"),
        (_superseded(pair_delimiter="-") + _UN, "none may equal or hold another"),
        (_superseded(pair_delimiter=".") + _UN, "none may equal or hold another"),
        (
            _superseded(pair_delimiter="") + _UN,
            r"exclusions\.0\.superseded_by_un_system\.pair_delimiter\n  Field required",
        ),
        (
            _superseded(value_delimiter="") + _UN,
            r"exclusions\.0\.superseded_by_un_system\.value_delimiter\n  Field required",
        ),
        (
            _superseded() + _superseded(provenance="dc/base/Z").replace("dc/s/A", "dc/s/B") + _UN,
            "duplicate superseded variable prefix",
        ),
        (
            _superseded(reason="superseded_by_dfo") + _UN,
            r"exclusions\.0\.superseded_by_dfo\.successor\n  Extra inputs are not permitted",
        ),
        (
            _BASE + _series("undata/sdg/SERIES-A", reason="superseded_by_un_system"),
            r"exclusions\.0\.reason\n  Input should be 'superseded_by_dfo'",
        ),
        (
            _BASE + _series("undata/sdg/SERIES-A", extra=', successor = "undata/p/WHO"'),
            r"exclusions\.0\.successor\n  Extra inputs are not permitted",
        ),
        (
            _BASE + _WHO + _series("undata/who/SERIES-ODA_1_3"),
            r"'undata/p/SDG' excludes \['undata/who/SERIES-ODA_1_3'\], outside its own series",
        ),
        (_BASE + _series("undata/sdgx/SERIES-A"), "outside its own series"),
        (
            _BASE + _series("undata/sdg/SERIES-A", "undata/sdg/SERIES-A"),
            "duplicate excluded series",
        ),
        (
            _BASE + _UN + 'exclusions = [{ series = "undata/sdg/SERIES-A", '
            'reason = "superseded_by_dfo", dfo_provenance = "dc/base/WorldBank_IDS" }]\n',
            "dfo_provenance",
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
    assert registry.un_system[0].exclusions == ()


def test_parse_registry_accepts_both_supersessions() -> None:
    registry = parse_registry((_superseded() + _series("undata/sdg/SERIES-A")).encode("utf-8"))
    (supersession,) = registry.base[0].exclusions or ()
    assert isinstance(supersession, UnSupersededProvenance)
    assert (supersession.provenance, supersession.successor) == ("dc/base/X", "undata/p/SDG")
    assert registry.un_supersessions() == ((supersession, registry.un_system[0]),)
    (exclusion,) = registry.un_system[0].exclusions
    assert (exclusion.series, exclusion.dfo_provenance) == ("undata/sdg/SERIES-A", "provenance/Y")


def test_a_supersession_reads_a_variable_id_as_its_code_and_constraint_pairs() -> None:
    registry = parse_registry(
        (_superseded(code_delimiter=":", pair_delimiter="+", value_delimiter="=") + _UN).encode()
    )
    ((supersession, _agency),) = registry.un_supersessions()

    suffix = supersession.variable_suffix("x/A_B:AGE=Y15+SEX=F")
    assert suffix == "A_B:AGE=Y15+SEX=F"
    assert supersession.constraint_key(suffix) == (
        "A_B",
        frozenset({("AGE", "Y15"), ("SEX", "F")}),
    )
    assert supersession.constraint_key("A_B:SEX=F+AGE=Y15") == supersession.constraint_key(suffix)
    assert supersession.constraint_key("A_B") == ("A_B", frozenset())
    assert supersession.variable_suffix("y/A_B") is None
    assert supersession.variable_suffix("x/") is None
