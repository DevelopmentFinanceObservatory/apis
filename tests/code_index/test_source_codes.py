"""The `source_codes` block loads into `CodeIndex` lookups from a publisher's series code to
the StatVars it names, and back, and the loader refuses a code row the StatVar rows
contradict."""

from __future__ import annotations

import json

import pytest

from code_index import IndexFormatError, default_index_path, load_index, parse_index

from .conftest import FIXTURE_PATH

_IDS = "provenance/WorldBankInternationalDebtStatistics"
_WEO = "provenance/IMFWorldEconomicOutlook"
_OTHER = "provenance/OtherPublisher"


def _listed(dcid: str, *provenances: str) -> dict:
    return {
        "dcid": dcid,
        "population_type": "FinancialFlow",
        "measured_property": "amount",
        "stat_type": "measuredValue",
        "measurement_qualifier": "Nominal",
        "measurement_denominator": None,
        "observation_properties": ["financingRecipient"],
        "constraint_properties": {"financialFlowType": "ServicingPayment"},
        "provenances": list(provenances),
    }


def _document(source_codes: list[dict]) -> dict:
    """The CRS fixture plus four listed StatVars: an IDS aggregate and its by-creditor
    variant, a WEO StatVar, and one StatVar of a third publisher."""
    payload = json.loads(FIXTURE_PATH.read_text())
    payload["listed_statvars"] = [
        _listed("Ids_Aggregate", _IDS),
        _listed("Ids_ByCreditor", _IDS),
        _listed("Weo_Gdp", _WEO),
        _listed("Other_Series", _OTHER),
    ]
    payload["source_codes"] = source_codes
    return payload


_ROWS = [
    {"provenance": _IDS, "code": "DT.TDS.DEGG.CD", "dcids": ["Ids_Aggregate", "Ids_ByCreditor"]},
    {"provenance": _IDS, "code": "SHARED.CD", "dcids": ["Ids_Aggregate"]},
    {"provenance": _OTHER, "code": "SHARED.CD", "dcids": ["Other_Series"]},
    {"provenance": _WEO, "code": "NGDP", "dcids": ["Weo_Gdp"]},
    {"provenance": _WEO, "code": "NGDPD", "dcids": ["Weo_Gdp"]},
]


def test_a_code_names_its_statvars_by_provenance_and_each_statvar_its_codes():
    index = parse_index(_document(_ROWS))

    assert index.source_code_members("DT.TDS.DEGG.CD") == {
        _IDS: ("Ids_Aggregate", "Ids_ByCreditor")
    }
    assert index.source_code_members("dt.tds.degg.cd") == {
        _IDS: ("Ids_Aggregate", "Ids_ByCreditor")
    }
    assert index.source_code_members("SHARED.CD") == {
        _IDS: ("Ids_Aggregate",),
        _OTHER: ("Other_Series",),
    }
    assert index.source_code_members("NOT.HELD") == {}

    assert index.source_codes_of("Weo_Gdp") == ((_WEO, "NGDP"), (_WEO, "NGDPD"))
    assert index.source_codes_of("Ids_Aggregate") == (
        (_IDS, "DT.TDS.DEGG.CD"),
        (_IDS, "SHARED.CD"),
    )
    assert index.source_codes_of("Ids_ByCreditor") == ((_IDS, "DT.TDS.DEGG.CD"),)
    assert index.source_codes_of("Unknown") == ()
    assert index.source_code_provenances == {_IDS, _OTHER, _WEO}

    assert index.shared_codes(_WEO, "ngdp") == ("NGDPD",)
    assert index.shared_codes(_IDS, "DT.TDS.DEGG.CD") == ("SHARED.CD",)
    assert index.shared_codes(_IDS, "SHARED.CD") == ("DT.TDS.DEGG.CD",)
    assert index.shared_codes(_OTHER, "SHARED.CD") == ()
    assert index.shared_codes(_WEO, "DT.TDS.DEGG.CD") == ()
    assert index.shared_codes(_IDS, "NOT.HELD") == ()


def test_an_index_without_the_block_holds_no_code():
    index = parse_index(json.loads(FIXTURE_PATH.read_text()))

    assert index.source_codes == ()
    assert index.source_code_provenances == frozenset()
    assert index.source_code_members("DT.TDS.DEGG.CD") == {}


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {"provenance": _IDS, "code": "DT.NEW.CD", "dcids": ["Ids_Unknown"]},
            "no StatVar row",
        ),
        (
            {"provenance": _OTHER, "code": "DT.TDS.DEGG.CD", "dcids": ["Ids_Aggregate"]},
            "whose row carries provenances",
        ),
        ({"provenance": _IDS, "code": "dt.tds.degg.cd", "dcids": ["Ids_Aggregate"]}, "upper"),
        ({"provenance": _IDS, "code": " NGDP", "dcids": ["Ids_Aggregate"]}, "upper"),
        ({"provenance": _IDS, "code": "", "dcids": ["Ids_Aggregate"]}, "non-empty"),
        ({"provenance": _IDS, "code": "EMPTY.CD", "dcids": []}, "names no StatVar"),
        ({"provenance": _IDS, "code": "SHARED.CD", "dcids": ["Ids_ByCreditor"]}, "more than once"),
    ],
    ids=[
        "unknown-dcid",
        "provenance-off-row",
        "lower-case",
        "unstripped",
        "empty-code",
        "no-dcids",
        "pair-twice",
    ],
)
def test_refuses_a_code_row_the_statvar_rows_contradict(row, message):
    with pytest.raises(IndexFormatError, match=message):
        parse_index(_document([*_ROWS, row]))


def test_refuses_a_code_row_missing_a_field():
    with pytest.raises(IndexFormatError, match="malformed"):
        parse_index(_document([{"provenance": _IDS, "code": "NGDP"}]))


def test_the_packaged_index_names_the_debt_service_code_s_aggregate_and_by_creditor_series():
    index = load_index(default_index_path())

    assert index.source_code_members("DT.TDS.DEGG.CD") == {
        _IDS: (
            "Amount_FinancialFlow_ServicingPayment_IDSDebtorTypeGG_ByCreditor_Nominal",
            "Amount_FinancialFlow_ServicingPayment_IDSDebtorTypeGG_Nominal",
        )
    }
    assert index.shared_codes(_IDS, "DT.TDS.DEGG.CD") == ()
    assert index.shared_codes(_IDS, "DT.NFL.IMFC.CD") == ("DT.NFL.MIDA.CD",)
    assert {row.dcid for row in index.listed_statvars} == {
        dcid for row in index.source_codes for dcid in row.dcids
    }
