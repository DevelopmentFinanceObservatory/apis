"""`load_index` reads the committed fixture into a `CodeIndex` with the
right shape and counts, and refuses a malformed document rather than
constructing a partial index."""

from __future__ import annotations

import gzip
import json

import pytest

from code_index import CodeIndex, IndexFormatError, default_index_path, load_index
from code_index.loader import load_measure_axes, parse_measure_axes
from code_index.query import CrossSystemParent, CyclicParent, DuplicateDcid, InvalidDeclaredDefault

from .conftest import FIXTURE_PATH


def test_loads_the_fixture_into_a_code_index(code_index: CodeIndex):
    assert isinstance(code_index, CodeIndex)
    assert len(code_index.codes) == 398
    assert len(code_index.statvars) == 9


def test_stamp_and_declarations_round_trip(code_index: CodeIndex):
    assert code_index.stamp.graph_endpoint == "https://dev.example/core/api"
    assert code_index.stamp.code_counts["crsPurpose"] == 289
    assert set(code_index.declarations) == {
        "crsPurpose",
        "crsTypeOfFinance",
        "crsCooperationModality",
        "crsTypeOfFlow",
    }


def test_declarations_carry_the_plain_name_and_graph_id_the_export_stamped(
    code_index: CodeIndex,
):
    assert {
        code_system: (declaration.plain_name, declaration.enum_dcid)
        for code_system, declaration in code_index.declarations.items()
    } == {
        "crsPurpose": ("sector", "CRSPurposeCodeEnum"),
        "crsTypeOfFinance": ("instrument", "CRSTypeOfFinanceEnum"),
        "crsCooperationModality": ("modality", "CRSCooperationModalityEnum"),
        "crsTypeOfFlow": ("flow", "CRSTypeOfFlowEnum"),
    }


def test_default_index_path_points_inside_the_package_tree():
    path = default_index_path()
    assert str(path).endswith("data/index.json.gz")


def test_the_packaged_index_declares_a_plain_name_and_graph_id_for_every_code_system():
    index = load_index(default_index_path())
    for declaration in index.declarations.values():
        assert declaration.plain_name
        assert declaration.enum_dcid
    assert index.declarations["crsPurpose"].enum_dcid == "CRSPurposeCodeEnum"


def test_the_packaged_index_labels_the_flow_type_and_netting_basis_axes():
    labels = {axis.axis: axis.label for axis in load_measure_axes(default_index_path()).axes}
    assert labels["financialFlowType"] == "flow type"
    assert labels["nettingBasis"] == "netting basis"


def test_the_packaged_index_defines_every_measure_label_and_noun_and_reads_code_definitions():
    measure_index = load_measure_axes(default_index_path())
    undefined = [
        label.label
        for label in (*measure_index.partition_labels, *measure_index.nouns)
        if not label.definition
    ]
    assert undefined == []
    grant = load_index(default_index_path()).code_by_dcid("CRSTypeOfFinance110")
    assert grant is not None
    assert grant.definition


def test_the_packaged_index_loads_and_agrees_with_its_own_stamp():
    # The stamped counts catch truncated exports that still parse as JSON.
    index = load_index(default_index_path())

    assert sum(index.stamp.code_counts.values()) == len(index.codes)
    assert index.stamp.statvar_shape_counts == {
        system: sum(system in row.constraint_properties for row in index.statvars)
        for system in index.declarations
    }
    assert all(
        any(system in row.constraint_properties for system in index.declarations)
        for row in index.statvars
    )
    assert index.codes
    assert index.statvars


def test_reads_a_gzipped_export_identically_to_a_plain_one(tmp_path, code_index: CodeIndex):
    # Compress the same fixture to isolate gzip detection and decompression from parsing.
    compressed = tmp_path / "index.json.gz"
    compressed.write_bytes(gzip.compress(FIXTURE_PATH.read_bytes()))

    from_gzip = load_index(compressed)

    assert from_gzip.stamp == code_index.stamp
    assert from_gzip.codes == code_index.codes
    assert from_gzip.statvars == code_index.statvars
    assert from_gzip.declarations == code_index.declarations


def test_reads_a_gzipped_export_whose_name_does_not_say_so(tmp_path):
    # Gzip detection uses content, so a `.json` filename still loads.
    misnamed = tmp_path / "index.json"
    misnamed.write_bytes(gzip.compress(FIXTURE_PATH.read_bytes()))

    assert len(load_index(misnamed).codes) == 398


@pytest.mark.parametrize(
    ("block", "key"), [("stamp", "content_hash"), ("declarations", "enum_dcid")]
)
def test_refuses_a_document_missing_a_required_key(tmp_path, block, key):
    payload = json.loads(FIXTURE_PATH.read_text())
    rows = payload[block] if isinstance(payload[block], list) else [payload[block]]
    del rows[0][key]
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload))
    with pytest.raises(IndexFormatError):
        load_index(broken)


@pytest.mark.parametrize(
    ("party_types", "parties", "error", "message"),
    [
        ([{"type": "DACGroup", "kind": "aggregate"}], [], IndexFormatError, "aggregate"),
        (
            [{"type": "Country", "kind": "party"}, {"type": "Country", "kind": None}],
            [],
            IndexFormatError,
            "more than once",
        ),
        ([], [{"dcid": "X", "kind": "aggregate", "totals": None}], IndexFormatError, "aggregate"),
        (
            [],
            [{"dcid": "X", "kind": "party", "totals": None, "axes": ["recipient"]}],
            IndexFormatError,
            "recipient",
        ),
        (
            [],
            [{"dcid": "country/FRA", "kind": "party", "totals": "europe"}],
            IndexFormatError,
            "only a total",
        ),
        (
            [],
            [
                {"dcid": "DACAfrica", "kind": "total", "totals": "africa"},
                {"dcid": "DACAfrica", "kind": "total", "totals": None},
            ],
            DuplicateDcid,
            "DACAfrica",
        ),
    ],
    ids=[
        "type-kind",
        "type-twice",
        "party-kind",
        "party-axis",
        "group-on-a-party",
        "party-twice",
    ],
)
def test_refuses_a_malformed_party_declaration(tmp_path, party_types, parties, error, message):
    payload = json.loads(FIXTURE_PATH.read_text())
    payload["party_types"] = party_types
    payload["parties"] = parties
    broken = tmp_path / "parties.json"
    broken.write_text(json.dumps(payload))
    with pytest.raises(error, match=message):
        load_index(broken)


def test_refuses_a_duplicate_code_dcid(tmp_path):
    payload = json.loads(FIXTURE_PATH.read_text())
    payload["codes"].append(dict(payload["codes"][0]))
    broken = tmp_path / "dup.json"
    broken.write_text(json.dumps(payload))
    with pytest.raises(DuplicateDcid):
        load_index(broken)


def test_refuses_a_duplicate_statvar_dcid(tmp_path):
    payload = json.loads(FIXTURE_PATH.read_text())
    payload["statvars"].append(dict(payload["statvars"][0]))
    broken = tmp_path / "dup_sv.json"
    broken.write_text(json.dumps(payload))
    with pytest.raises(DuplicateDcid):
        load_index(broken)


def test_refuses_a_code_that_is_its_own_parent(tmp_path):
    # A self-parent would keep the breadth-first descendants walk alive indefinitely.
    payload = json.loads(FIXTURE_PATH.read_text())
    for row in payload["codes"]:
        if row["dcid"] == "CRSPurposeCode120":
            row["parent"] = "CRSPurposeCode120"
    broken = tmp_path / "self_parent.json"
    broken.write_text(json.dumps(payload))
    with pytest.raises(CyclicParent):
        load_index(broken)


def test_refuses_a_parent_cycle_between_two_codes(tmp_path):
    # CRSPurposeCode121 is CRSPurposeCode120's child, so this link closes a two-code cycle.
    payload = json.loads(FIXTURE_PATH.read_text())
    for row in payload["codes"]:
        if row["dcid"] == "CRSPurposeCode120":
            row["parent"] = "CRSPurposeCode121"
    broken = tmp_path / "cycle.json"
    broken.write_text(json.dumps(payload))
    with pytest.raises(CyclicParent):
        load_index(broken)


def test_load_index_refuses_a_parent_from_a_different_code_system(tmp_path):
    """#852: `parent` names a `specializationOf` row in the child's code system.

    The cross-system target is otherwise valid, isolating the code-system check."""
    payload = json.loads(FIXTURE_PATH.read_text())
    for row in payload["codes"]:
        if row["dcid"] == "CRSPurposeCode11110":
            row["parent"] = "CRSTypeOfFinance110"
    broken = tmp_path / "cross_system_parent.json"
    broken.write_text(json.dumps(payload))
    with pytest.raises(CrossSystemParent) as excinfo:
        load_index(broken)
    assert "CRSPurposeCode11110" in str(excinfo.value)
    assert "CRSTypeOfFinance110" in str(excinfo.value)


# --- CodeSystemDeclaration.default (issue #968) ---


def _payload_with_flow_default(*, code: str, when_bound_under: list[str]) -> dict:
    payload = json.loads(FIXTURE_PATH.read_text())
    for row in payload["declarations"]:
        if row["code_system"] == "crsTypeOfFlow":
            row["default"] = {
                "code": code,
                "when_bound_under": when_bound_under,
                "when_named": [],
            }
    return payload


def test_loads_a_well_formed_declared_default(tmp_path):
    payload = _payload_with_flow_default(
        code="CRSTypeOfFlow10", when_bound_under=["CRSTypeOfFinance100"]
    )
    doc = tmp_path / "with_default.json"
    doc.write_text(json.dumps(payload))

    index = load_index(doc)

    default = index.declarations["crsTypeOfFlow"].default
    assert default is not None
    assert default.code == "CRSTypeOfFlow10"
    assert default.when_bound_under == ("CRSTypeOfFinance100",)
    # A missing default parses as None.
    assert index.declarations["crsPurpose"].default is None


def test_refuses_a_declared_default_whose_code_is_not_a_row_of_its_own_system(tmp_path):
    # CRSTypeOfFinance100 is a real code, but it belongs to crsTypeOfFinance,
    # not the crsTypeOfFlow system declaring the default.
    payload = _payload_with_flow_default(
        code="CRSTypeOfFinance100", when_bound_under=["CRSTypeOfFinance100"]
    )
    doc = tmp_path / "wrong_system_code.json"
    doc.write_text(json.dumps(payload))

    with pytest.raises(InvalidDeclaredDefault) as excinfo:
        load_index(doc)
    assert "CRSTypeOfFinance100" in str(excinfo.value)


def test_refuses_a_declared_default_whose_when_bound_under_names_an_unknown_dcid(tmp_path):
    payload = _payload_with_flow_default(code="CRSTypeOfFlow10", when_bound_under=["NotARealDcid"])
    doc = tmp_path / "unknown_scoping_dcid.json"
    doc.write_text(json.dumps(payload))

    with pytest.raises(InvalidDeclaredDefault) as excinfo:
        load_index(doc)
    assert "NotARealDcid" in str(excinfo.value)


def test_refuses_a_declared_default_scoped_under_its_own_code_system(tmp_path):
    # CRSTypeOfFlow21 is a real crsTypeOfFlow code -- the same system the
    # default itself declares -- so scoping under it would condition the
    # default on a binding of the very slot it defaults.
    payload = _payload_with_flow_default(
        code="CRSTypeOfFlow10", when_bound_under=["CRSTypeOfFlow21"]
    )
    doc = tmp_path / "self_scoped.json"
    doc.write_text(json.dumps(payload))

    with pytest.raises(InvalidDeclaredDefault) as excinfo:
        load_index(doc)
    assert "CRSTypeOfFlow21" in str(excinfo.value)


def test_the_packaged_index_declares_the_crs_type_of_flow_default():
    # #968: An open flow slot selects CRSTypeOfFlow10 for aid under CRSTypeOfFinance100.
    index = load_index(default_index_path())
    default = index.declarations["crsTypeOfFlow"].default
    assert default is not None
    assert default.code == "CRSTypeOfFlow10"
    assert default.when_bound_under == ("CRSTypeOfFinance100",)
    assert default.when_named == ("aid",)
    assert default.when_open is False
    assert default.note is None


@pytest.mark.parametrize(
    ("code_system", "code"),
    [
        ("dac1AidType", "DAC1AidType1010"),
        ("dac2aAidType", "DAC2aAidType206"),
        ("dac2bAidType", "DAC2bAidType296"),
    ],
)
def test_the_packaged_index_declares_the_table_total_as_the_open_aid_type_default(
    code_system: str, code: str
):
    """A DAC1 or DAC2a question that names no aid type reads total ODA, a DAC2b question net
    official non-concessional flows, and the answer says so."""
    default = load_index(default_index_path()).declarations[code_system].default
    assert default is not None
    assert default.code == code
    assert default.when_open is True
    assert default.note


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"when_open": True}, "fires on an open slot and carries no note"),
        ({"when_open": "yes", "note": "n"}, "when_open 'yes' is not a boolean"),
        ({"note": " "}, "note ' ' is not a non-empty string"),
    ],
    ids=["open-without-note", "open-not-boolean", "blank-note"],
)
def test_refuses_a_malformed_open_slot_default(tmp_path, fields, message):
    payload = _payload_with_flow_default(
        code="CRSTypeOfFlow10", when_bound_under=["CRSTypeOfFinance100"]
    )
    for row in payload["declarations"]:
        if row["code_system"] == "crsTypeOfFlow":
            row["default"].update(fields)
    doc = tmp_path / "malformed_open_default.json"
    doc.write_text(json.dumps(payload))

    with pytest.raises(IndexFormatError, match=message):
        load_index(doc)


def test_refuses_a_declared_default_with_no_when_named_key_at_all(tmp_path):
    payload = _payload_with_flow_default(
        code="CRSTypeOfFlow10", when_bound_under=["CRSTypeOfFinance100"]
    )
    for row in payload["declarations"]:
        if row["code_system"] == "crsTypeOfFlow":
            del row["default"]["when_named"]
    doc = tmp_path / "no_when_named_key.json"
    doc.write_text(json.dumps(payload))

    with pytest.raises(IndexFormatError):
        load_index(doc)


def test_loads_a_well_formed_declared_default_with_when_named(tmp_path):
    payload = _payload_with_flow_default(
        code="CRSTypeOfFlow10", when_bound_under=["CRSTypeOfFinance100"]
    )
    for row in payload["declarations"]:
        if row["code_system"] == "crsTypeOfFlow":
            row["default"]["when_named"] = ["aid"]
    doc = tmp_path / "with_when_named.json"
    doc.write_text(json.dumps(payload))

    index = load_index(doc)

    default = index.declarations["crsTypeOfFlow"].default
    assert default is not None
    assert default.when_named == ("aid",)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ("Aid", "when_named entry 'Aid' must be a non-empty"),
        ("", "when_named entry '' must be a non-empty"),
        # Returning the invalid entry itself would confuse null with no error; the validator
        # returns its index.
        (None, "when_named entry None must be a non-empty"),
    ],
    ids=["uppercase", "empty", "null"],
)
def test_refuses_a_declared_default_with_invalid_when_named_entry(tmp_path, entry, message):
    payload = _payload_with_flow_default(
        code="CRSTypeOfFlow10", when_bound_under=["CRSTypeOfFinance100"]
    )
    for row in payload["declarations"]:
        if row["code_system"] == "crsTypeOfFlow":
            row["default"]["when_named"] = [entry]
    doc = tmp_path / "invalid_when_named.json"
    doc.write_text(json.dumps(payload))

    with pytest.raises(IndexFormatError, match=message):
        load_index(doc)


def _measure_value(dcid: str, *, same_reading_as: str | None = None) -> dict:
    return {
        "dcid": dcid,
        "axis": "measurementQualifier",
        "label": dcid.lower(),
        "synonyms": [],
        "short_name": None,
        "matchable": True,
        "same_reading_as": same_reading_as,
    }


def _price_basis_document(*values: dict) -> dict:
    return {
        "measure_axes": [
            {
                "axis": "measurementQualifier",
                "kind": "shape",
                "role": "free",
                "prefer": ["Nominal", "RealValue", "PurchasingPowerParity"],
                "label": None,
            }
        ],
        "measure_values": list(values),
        "measure_partition_labels": [],
    }


def test_measure_index_reads_an_alias_into_its_canonical_reading():
    measure_index = parse_measure_axes(
        _price_basis_document(
            _measure_value("Nominal"),
            _measure_value("RealValue"),
            _measure_value("InflationAdjusted", same_reading_as="RealValue"),
            _measure_value("PurchasingPowerParity"),
        )
    )

    assert measure_index.canonical_reading("InflationAdjusted") == "RealValue"
    assert measure_index.canonical_reading("Nominal") == "Nominal"
    assert measure_index.reading_set("RealValue") == {"RealValue", "InflationAdjusted"}
    assert measure_index.reading_set("InflationAdjusted") == {"RealValue", "InflationAdjusted"}
    assert measure_index.reading_set("Nominal") == {"Nominal"}
    assert measure_index.reading_order("measurementQualifier") == (
        "Nominal",
        "RealValue",
        "InflationAdjusted",
        "PurchasingPowerParity",
    )
    assert measure_index.reading_order("nettingBasis") == ()


@pytest.mark.parametrize(
    ("values", "message"),
    [
        pytest.param(
            (_measure_value("InflationAdjusted", same_reading_as="RealValue"),),
            "not a 'measurementQualifier' measure value",
            id="canonical-missing",
        ),
        pytest.param(
            (
                _measure_value("Nominal"),
                _measure_value("RealValue", same_reading_as="Nominal"),
                _measure_value("InflationAdjusted", same_reading_as="RealValue"),
            ),
            "itself reads as",
            id="alias-chain",
        ),
    ],
)
def test_refuses_a_same_reading_as_that_names_no_canonical_value(values, message):
    with pytest.raises(IndexFormatError, match=message):
        parse_measure_axes(_price_basis_document(*values))


def test_the_packaged_index_files_inflation_adjusted_under_real_value():
    measure_index = load_measure_axes(default_index_path())

    assert measure_index.reading_order("measurementQualifier") == (
        "Nominal",
        "RealValue",
        "InflationAdjusted",
        "PurchasingPowerParity",
    )
