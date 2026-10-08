"""Test dataset declaration parsing, the checks that tie declarations to StatVar rows, and
absent-observation readings.

The packaged index declares CRS, OECD DAC1, OECD DAC2a, OECD DAC2b and the opt-in imputed
multilateral sectors table, each the provider's account of a flow. DAC1 shares the CRS flow and
commitment five-tuples, DAC2a shares the CRS and DAC1 flow five-tuple and the imputed table's,
and DAC2b shares the CRS, DAC1 and DAC2a flow five-tuple.
CRS and the imputed table read an absent observation as zero, which the rows in
`fixtures/absent_observation_evidence.json` support: each is one (financingProvider,
financingRecipient, year) cell read from the dev graph's observation table, where the source
reports an explicit zero for one member of a five-tuple and no row for most others. DAC1,
DAC2a, DAC2b and an undeclared source read as unknown."""

from __future__ import annotations

import json

import pytest

from code_index import (
    CodeIndex,
    DatasetDeclaration,
    IndexFormatError,
    IndexStamp,
    default_index_path,
    load_index,
    parse_index,
)
from code_index.query import row_five_tuple

from .conftest import FIXTURE_PATH

_CRS_PROVENANCE = "provenance/OECDCreditorReportingSystem"
_IMPUTED_PROVENANCE = "provenance/ONEImputedMultilateralSectors"
_DAC1_PROVENANCE = "provenance/OECDDAC1"
_DAC2A_PROVENANCE = "provenance/OECDDAC2A"
_DAC2B_PROVENANCE = "provenance/OECDDAC2B"
_EVIDENCE_PATH = FIXTURE_PATH.parent / "absent_observation_evidence.json"
_EMPTY_STAMP = IndexStamp(
    graph_endpoint="https://fake.example",
    exported_at="2026-09-15T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="",
)


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _with_imputed_dataset() -> dict:
    """The fixture plus a second, opt-in dataset holding the last row, moved to a five-tuple of
    its own. The dataset declares the ODA flow type that row carries as an invariant."""
    payload = _fixture()
    payload["datasets"].append(
        {
            "provenance": _IMPUTED_PROVENANCE,
            "label": "ONE Imputed Multilateral Sectors",
            "description": "A test table.",
            "opt_in": True,
            "absent_observation": "zero",
            "account": "provider",
            "rank": 10,
            "invariants": {"crsTypeOfFlow": "CRSTypeOfFlow10"},
        }
    )
    imputed = payload["statvars"][-1]
    imputed["provenances"] = [_IMPUTED_PROVENANCE]
    imputed["measured_property"] = "attributedAmount"
    return payload


def test_loader_parses_a_dataset_declaration(code_index: CodeIndex):
    assert code_index.datasets == {
        _CRS_PROVENANCE: DatasetDeclaration(
            provenance=_CRS_PROVENANCE,
            label="OECD Creditor Reporting System (CRS)",
            description="A test table.",
            opt_in=False,
            absent_observation="zero",
            account="provider",
            rank=0,
        )
    }


def test_loader_reads_each_rows_provenances_and_the_coverage_stamp(code_index: CodeIndex):
    assert {row.provenances for row in code_index.statvars} == {(_CRS_PROVENANCE,)}
    assert code_index.stamp.coverage_build == "fixture-coverage-build"
    assert code_index.stamp.statvars_without_coverage == 0


def test_parse_index_reads_a_document_as_load_index_reads_its_file(code_index: CodeIndex):
    parsed = parse_index(_fixture())

    assert parsed.statvars == code_index.statvars
    assert parsed.datasets == code_index.datasets
    assert parsed.stamp == code_index.stamp


@pytest.mark.parametrize(
    ("field", "value"),
    [("absent_observation", "not_reported"), ("account", "lender"), ("account", None)],
)
def test_loader_rejects_a_value_outside_a_fields_vocabulary(field: str, value: object):
    payload = _fixture()
    payload["datasets"][0][field] = value

    with pytest.raises(IndexFormatError, match=f"{field}={value!r}"):
        parse_index(payload)


def test_loader_rejects_an_opt_in_that_is_not_a_boolean():
    payload = _fixture()
    payload["datasets"][0]["opt_in"] = "false"

    with pytest.raises(IndexFormatError, match="opt_in='false'"):
        parse_index(payload)


@pytest.mark.parametrize("rank", ["0", 1.5, True, None])
def test_loader_rejects_a_rank_that_is_not_an_integer(rank: object):
    payload = _fixture()
    payload["datasets"][0]["rank"] = rank

    with pytest.raises(IndexFormatError, match=rf"rank={rank!r}, which is not an integer"):
        parse_index(payload)


def test_loader_rejects_a_duplicate_provenance():
    payload = _fixture()
    payload["datasets"].append(dict(payload["datasets"][0]))

    with pytest.raises(IndexFormatError, match="declared more than once"):
        parse_index(payload)


def test_loader_rejects_two_datasets_of_one_rank():
    payload = _with_imputed_dataset()
    payload["datasets"][-1]["rank"] = payload["datasets"][0]["rank"]

    with pytest.raises(IndexFormatError, match=rf"{_IMPUTED_PROVENANCE}.*share rank 0"):
        parse_index(payload)


@pytest.mark.parametrize(
    "field",
    [
        "provenance",
        "label",
        "description",
        "opt_in",
        "absent_observation",
        "account",
        "rank",
        "invariants",
    ],
)
def test_loader_rejects_a_dataset_missing_a_field(field: str):
    payload = _fixture()
    del payload["datasets"][0][field]

    with pytest.raises(IndexFormatError, match=field):
        parse_index(payload)


@pytest.mark.parametrize("field", ["label", "description"])
def test_loader_rejects_a_dataset_with_an_empty_label_or_description(field: str):
    payload = _fixture()
    payload["datasets"][0][field] = ""

    with pytest.raises(IndexFormatError, match=f"no {field}"):
        parse_index(payload)


def test_loader_rejects_a_document_missing_the_datasets_block():
    payload = _fixture()
    del payload["datasets"]

    with pytest.raises(IndexFormatError, match="datasets"):
        parse_index(payload)


def test_loader_rejects_a_row_missing_its_provenances():
    payload = _fixture()
    del payload["statvars"][0]["provenances"]

    with pytest.raises(IndexFormatError, match="provenances"):
        parse_index(payload)


@pytest.mark.parametrize("field", ["coverage_build", "statvars_without_coverage"])
def test_loader_rejects_a_stamp_missing_a_coverage_field(field: str):
    payload = _fixture()
    del payload["stamp"][field]

    with pytest.raises(IndexFormatError, match=field):
        parse_index(payload)


def test_loader_maps_each_five_tuple_to_the_dataset_publishing_it():
    index = parse_index(_with_imputed_dataset())

    imputed = index.statvars[-1]
    assert index.datasets[_IMPUTED_PROVENANCE].opt_in is True
    assert index.datasets_by_five_tuple[row_five_tuple(imputed)] == (_IMPUTED_PROVENANCE,)
    assert index.datasets_by_five_tuple[row_five_tuple(index.statvars[0])] == (_CRS_PROVENANCE,)


# The fixture's education row with no modality, on the same five-tuple as the imputed row.
_SHARED_ROW = 6


def _sharing_one_five_tuple(*, crs_rank: int, imputed_rank: int) -> CodeIndex:
    """The fixture with the imputed dataset on CRS's five-tuple: its own row, and an education
    row both datasets publish."""
    payload = _with_imputed_dataset()
    payload["datasets"][0]["rank"] = crs_rank
    payload["datasets"][-1]["rank"] = imputed_rank
    payload["statvars"][-1]["measured_property"] = payload["statvars"][0]["measured_property"]
    payload["statvars"][_SHARED_ROW]["provenances"] = [_CRS_PROVENANCE, _IMPUTED_PROVENANCE]
    return parse_index(payload)


@pytest.mark.parametrize(
    ("crs_rank", "imputed_rank", "order"),
    [
        (0, 10, (_CRS_PROVENANCE, _IMPUTED_PROVENANCE)),
        (10, 0, (_IMPUTED_PROVENANCE, _CRS_PROVENANCE)),
    ],
    ids=["crs-first", "imputed-first"],
)
def test_two_datasets_on_one_five_tuple_load_in_rank_order(
    crs_rank: int, imputed_rank: int, order: tuple[str, ...]
):
    index = _sharing_one_five_tuple(crs_rank=crs_rank, imputed_rank=imputed_rank)

    shared = index.statvars[_SHARED_ROW]
    assert shared.provenances == (_CRS_PROVENANCE, _IMPUTED_PROVENANCE)
    assert index.datasets_by_five_tuple[row_five_tuple(shared)] == order


def test_each_dataset_pins_its_own_rows_of_a_shared_five_tuple():
    """The imputed dataset's two education rows hold one purpose and one flow type, which
    CRS's health rows on the same five-tuple do not."""
    index = _sharing_one_five_tuple(crs_rank=0, imputed_rank=10)
    five_tuple = row_five_tuple(index.statvars[_SHARED_ROW])

    assert index.dataset_pins[_IMPUTED_PROVENANCE, five_tuple] == {
        "crsPurpose": "CRSPurposeCode11110",
        "crsTypeOfFinance": "CRSTypeOfFinance110",
        "crsTypeOfFlow": "CRSTypeOfFlow10",
    }
    assert index.dataset_pins[_CRS_PROVENANCE, five_tuple] == {
        "crsTypeOfFinance": "CRSTypeOfFinance110"
    }


def test_loader_rejects_a_provenance_no_dataset_declares():
    payload = _fixture()
    payload["statvars"][0]["provenances"] = [_CRS_PROVENANCE, "provenance/ThirdSource"]

    with pytest.raises(IndexFormatError, match=r"provenance/ThirdSource.*no dataset declares"):
        parse_index(payload)


def test_loader_rejects_a_declared_dataset_with_no_row():
    payload = _with_imputed_dataset()
    payload["statvars"][-1]["provenances"] = []

    with pytest.raises(IndexFormatError, match=rf"{_IMPUTED_PROVENANCE}.*no StatVar row"):
        parse_index(payload)


@pytest.mark.parametrize("held", [None, "CRSTypeOfFlow21"], ids=["absent", "other-code"])
def test_loader_rejects_a_row_without_its_datasets_declared_invariant(held: str | None):
    """An absent property fails the invariant whatever the code system's absent reading."""
    payload = _with_imputed_dataset()
    payload["statvars"][_SHARED_ROW]["provenances"] = [_CRS_PROVENANCE, _IMPUTED_PROVENANCE]
    constraints = payload["statvars"][_SHARED_ROW]["constraint_properties"]
    if held is None:
        del constraints["crsTypeOfFlow"]
    else:
        constraints["crsTypeOfFlow"] = held

    with pytest.raises(
        IndexFormatError, match=rf"{_IMPUTED_PROVENANCE}.*crsTypeOfFlow=.*invariant"
    ):
        parse_index(payload)


def test_loader_accepts_a_row_with_no_provenance():
    payload = _fixture()
    payload["statvars"][0]["provenances"] = []

    index = parse_index(payload)

    assert index.statvars[0].provenances == ()


def test_absent_observation_returns_zero_for_crs(code_index: CodeIndex):
    assert code_index.absent_observation(_CRS_PROVENANCE) == "zero"


def test_absent_observation_returns_unknown_for_an_undeclared_provenance(code_index: CodeIndex):
    assert code_index.absent_observation("provenance/SomeOtherSource") == "unknown"


def test_a_codeindex_built_with_no_datasets_at_all_reads_unknown():
    # Direct construction defaults to unknown, so an undeclared source needs no special case.
    index = CodeIndex(stamp=_EMPTY_STAMP, declarations=(), codes=(), statvars=())
    assert index.absent_observation(_CRS_PROVENANCE) == "unknown"
    assert index.stamp.coverage_build is None


def test_the_packaged_index_declares_crs_and_the_dac_tables_and_the_opt_in_imputed_dataset():
    index = load_index(default_index_path())

    assert {
        provenance: (dataset.opt_in, dataset.absent_observation, dataset.account, dataset.rank)
        for provenance, dataset in index.datasets.items()
    } == {
        _CRS_PROVENANCE: (False, "zero", "provider", 0),
        _IMPUTED_PROVENANCE: (True, "zero", "provider", 10),
        _DAC1_PROVENANCE: (False, "unknown", "provider", 20),
        _DAC2A_PROVENANCE: (False, "unknown", "provider", 30),
        _DAC2B_PROVENANCE: (False, "unknown", "provider", 40),
    }
    assert index.datasets[_DAC1_PROVENANCE].invariants == {}
    assert index.datasets[_DAC2A_PROVENANCE].invariants == {}
    assert index.datasets[_DAC2A_PROVENANCE].label == "OECD DAC2a"
    assert index.datasets[_DAC2B_PROVENANCE].invariants == {}
    assert index.datasets[_DAC2B_PROVENANCE].label == "OECD DAC2b"
    assert index.datasets[_CRS_PROVENANCE].invariants == {}
    assert index.datasets[_DAC1_PROVENANCE].label == "OECD DAC1"
    assert index.datasets[_IMPUTED_PROVENANCE].invariants == {
        "crsTypeOfFlow": "CRSTypeOfFlow10",
        "dac2aAidType": "DAC2aAidType106",
    }
    assert index.datasets[_CRS_PROVENANCE].label == "OECD Creditor Reporting System (CRS)"
    assert index.datasets[_IMPUTED_PROVENANCE].label == "ONE Imputed Multilateral Sectors"


def test_every_packaged_row_belongs_to_the_family_its_own_keys_select():
    """Each row's constraint properties select the family that holds it
    (`CodeIndex.family_of`), and that family's datasets are the row's own. CRS, DAC1, DAC2a
    and DAC2b share five-tuples under different codes, and DAC2a's one `attributedAmount` row
    declares fewer properties than the imputed rows, so each such five-tuple holds one family
    per dataset."""
    index = load_index(default_index_path())

    assert index.stamp.coverage_build
    uncovered = [row for row in index.statvars if not row.provenances]
    assert len(uncovered) == index.stamp.statvars_without_coverage
    for row in index.statvars:
        assert row.provenances == tuple(sorted(row.provenances))
        family = index.family_of(row_five_tuple(row), row.constraint_properties.keys())
        assert family is not None, row.dcid
        assert row in family.rows, row.dcid
        if row.provenances:
            assert row.provenances == tuple(sorted(family.datasets)), row.dcid
    shared = {
        tuple(family.datasets for family in families)
        for families in index.families_by_five_tuple.values()
        if len(families) > 1
    }
    assert shared == {
        ((_CRS_PROVENANCE,), (_DAC1_PROVENANCE,)),
        (
            (_CRS_PROVENANCE,),
            (_DAC1_PROVENANCE,),
            (_DAC2A_PROVENANCE,),
            (_DAC2B_PROVENANCE,),
        ),
        ((_DAC1_PROVENANCE,), (_DAC2A_PROVENANCE,)),
        ((_IMPUTED_PROVENANCE,), (_DAC2A_PROVENANCE,)),
    }


def test_the_packaged_index_pins_the_constant_slots_of_single_aid_type_five_tuples_only():
    """The imputed table holds one aid type, DAC1 publishes grant equivalents as a share of
    GNI for total ODA alone, and DAC2a publishes its ratios for total ODA alone and one
    attributed amount."""
    index = load_index(default_index_path())

    assert {
        (provenance, five_tuple.measured_property): pins
        for (provenance, five_tuple), pins in index.dataset_pins.items()
        if pins
    } == {
        (_IMPUTED_PROVENANCE, "attributedAmount"): {
            "crsTypeOfFlow": "CRSTypeOfFlow10",
            "dac2aAidType": "DAC2aAidType106",
        },
        (_DAC1_PROVENANCE, "grantEquivalent"): {"dac1AidType": "DAC1AidType1010"},
        (_DAC2A_PROVENANCE, "amount"): {"dac2aAidType": "DAC2aAidType206"},
        (_DAC2A_PROVENANCE, "attributedAmount"): {"dac2aAidType": "DAC2aAidType106"},
    }


@pytest.mark.parametrize(
    "evidence",
    json.loads(_EVIDENCE_PATH.read_text()),
    ids=lambda evidence: evidence["provenance"],
)
def test_a_dataset_read_as_zero_reports_explicit_zeros_beside_absent_cells(evidence: dict):
    """The register reading (`AbsentObservation`) holds for a source whose data carries an
    explicit zero for one member of a (roles, time) cell while other members of the same
    five-tuple have no row there."""
    index = load_index(default_index_path())
    rows = {row.dcid: row for row in index.statvars}
    observed = [rows[observation["variable"]] for observation in evidence["observations"]]
    five_tuples = {row_five_tuple(row) for row in observed}
    (five_tuple,) = five_tuples
    members = {row.dcid for row in index.statvars if row_five_tuple(row) == five_tuple}

    assert {row.provenances for row in observed} == {(evidence["provenance"],)}
    assert 0.0 in {observation["value"] for observation in evidence["observations"]}
    assert members - {row.dcid for row in observed}
    assert index.absent_observation(evidence["provenance"]) == "zero"


def test_the_packaged_index_holds_the_aid_type_code_row():
    index = load_index(default_index_path())

    assert index.declarations["dac2aAidType"].summable is False
    aid_type = index.code_by_dcid("DAC2aAidType106")
    assert aid_type is not None
    assert aid_type.code_system == "dac2aAidType"
    assert aid_type.label == "imputed multilateral ODA"
