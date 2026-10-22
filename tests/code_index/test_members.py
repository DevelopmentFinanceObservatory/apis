"""Test index membership and minimal StatVar selection against the fixture index and a small family.

CRS rows cover undeclared modality. A synthetic Person/gender family tests aggregate reduction and
parity with the graph walk.
"""

from __future__ import annotations

import pytest

from code_index.members import (
    FamilyQuery,
    antichain_minimal_rows,
    family_for_five_tuple,
    identity_family_query,
    index_covers_slot_keys,
    listed_group,
)
from code_index.query import CodeIndex
from code_index.types import IndexStamp, StatVarRow
from statvar_members import FiveTuple, SlotBinding, enumerate_members
from tests.statvar_members.conftest import FakeMemberArcReader

_EDU_ODA_FIVE_TUPLE = FiveTuple(
    population_type="FinancialFlow",
    measured_property="amount",
    stat_type="measuredValue",
    measurement_qualifier="Nominal",
)
_EDU_ODA_SLOTS = {
    "crsPurpose": ("value", "CRSPurposeCode11110"),
    "crsTypeOfFinance": ("value", "CRSTypeOfFinance110"),
    "crsTypeOfFlow": ("value", "CRSTypeOfFlow10"),
    "financialFlowType": ("value", "Disbursement"),
    "nettingBasis": ("value", "Gross"),
    "crsCooperationModality": ("unbound",),
}
_EDU_ODA_DCIDS = {
    "FixtureStatVar_Amount_FinancialFlow_Disbursement_Gross_Nominal_EduODA_NoModality",
    "FixtureStatVar_Amount_FinancialFlow_Disbursement_Gross_Nominal_EduODA_GBS",
    "FixtureStatVar_Amount_FinancialFlow_Disbursement_Gross_Nominal_EduODA_CoreNGO",
}


def _minimal(code_index: CodeIndex, five_tuple: FiveTuple, slots: dict) -> tuple[StatVarRow, ...]:
    return antichain_minimal_rows(
        code_index, family_for_five_tuple(code_index, FamilyQuery(five_tuple, slots, None)).rows
    )


def _readings(code_index: CodeIndex) -> dict[str, str]:
    return {
        code_system: declaration.absent_reading
        for code_system, declaration in code_index.declarations.items()
    }


def test_slot_keys_cover_a_five_tuple_the_fixture_holds(code_index: CodeIndex):
    assert index_covers_slot_keys(code_index, _EDU_ODA_FIVE_TUPLE, _EDU_ODA_SLOTS.keys()) is True


def test_slot_keys_cover_no_row_of_a_measured_property_no_row_declares(code_index: CodeIndex):
    absent = FiveTuple(measured_property="nonexistentMeasure")
    assert index_covers_slot_keys(code_index, absent, _EDU_ODA_SLOTS.keys()) is False


def test_slot_keys_cover_no_row_for_a_denominator_unresolved_ask(code_index: CodeIndex):
    # Exported rows are resolved, so an unresolved-denominator tuple cannot match.
    unresolved = FiveTuple(
        population_type="FinancialFlow",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier="Nominal",
        denominator_unresolved=True,
    )
    assert index_covers_slot_keys(code_index, unresolved, _EDU_ODA_SLOTS.keys()) is False


def test_bound_and_unbound_slots_against_the_fixture_index(code_index: CodeIndex):
    rows = _minimal(code_index, _EDU_ODA_FIVE_TUPLE, _EDU_ODA_SLOTS)
    assert {row.dcid for row in rows} == _EDU_ODA_DCIDS


def test_not_declared_modality_keeps_every_row_instead_of_wrongly_reducing(code_index: CodeIndex):
    # The "not_declared" reading keeps the unlabelled row from dominating the coded rows.
    rows = _minimal(code_index, _EDU_ODA_FIVE_TUPLE, _EDU_ODA_SLOTS)
    assert len(rows) == 3


def test_a_bound_slot_narrows_to_exactly_the_matching_row(code_index: CodeIndex):
    slots = dict(_EDU_ODA_SLOTS, crsCooperationModality=("value", "CRSCooperationModalityA01"))
    rows = _minimal(code_index, _EDU_ODA_FIVE_TUPLE, slots)
    assert {row.dcid for row in rows} == {
        "FixtureStatVar_Amount_FinancialFlow_Disbursement_Gross_Nominal_EduODA_GBS"
    }


def test_a_five_tuple_the_index_does_not_carry_returns_no_members(code_index: CodeIndex):
    absent = FiveTuple(measured_property="nonexistentMeasure")
    assert _minimal(code_index, absent, {}) == ()


def test_a_constraint_property_outside_the_slot_vocabulary_is_excluded(code_index: CodeIndex):
    # Omitted `nettingBasis` is outside the slots, so rows declaring it are excluded.
    # `enumerate_members` applies the same rule on the graph-walked path.
    slots = {k: v for k, v in _EDU_ODA_SLOTS.items() if k != "nettingBasis"}
    rows = _minimal(code_index, _EDU_ODA_FIVE_TUPLE, slots)
    assert rows == ()


_COUNT_FIVE_TUPLE = FiveTuple(
    population_type="Person", measured_property="count", stat_type="measuredValue"
)
_GENDER_UNBOUND = {"gender": ("unbound",)}

_STAMP = IndexStamp(
    graph_endpoint="test",
    exported_at="2026-01-01T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="test",
)


def _person_family_index() -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=(),
        codes=(),
        statvars=(
            StatVarRow(
                dcid="Person",
                population_type="Person",
                measured_property="count",
                stat_type="measuredValue",
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=(),
                constraint_properties={},
                provenances=("provenance/Test",),
            ),
            StatVarRow(
                dcid="Person_Male",
                population_type="Person",
                measured_property="count",
                stat_type="measuredValue",
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=(),
                constraint_properties={"gender": "Male"},
                provenances=("provenance/Test",),
            ),
            StatVarRow(
                dcid="Person_Female",
                population_type="Person",
                measured_property="count",
                stat_type="measuredValue",
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=(),
                constraint_properties={"gender": "Female"},
                provenances=("provenance/Test",),
            ),
        ),
    )


def _person_family_reader() -> FakeMemberArcReader:
    def _arcs(gender: str | None) -> dict[str, tuple[str, ...]]:
        return {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": ("gender",) if gender else (),
            "observationProperties": (),
            **({"gender": (gender,)} if gender else {}),
        }

    return FakeMemberArcReader(
        arcs={
            "Person": _arcs(None),
            "Person_Male": _arcs("Male"),
            "Person_Female": _arcs("Female"),
        }
    )


def test_an_unconstrained_row_dominates_its_more_specific_siblings():
    rows = _minimal(_person_family_index(), _COUNT_FIVE_TUPLE, _GENDER_UNBOUND)
    assert {row.dcid for row in rows} == {"Person"}


def test_index_reduction_agrees_with_the_walk_on_a_genuinely_reducing_family():
    index_rows = _minimal(_person_family_index(), _COUNT_FIVE_TUPLE, _GENDER_UNBOUND)
    walked = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE, slots=_GENDER_UNBOUND, graph=_person_family_reader()
    )
    assert {row.dcid for row in index_rows} == {member.stat_var for member in walked.members}


def test_the_index_family_agrees_with_the_walk_and_keeps_the_dominated_rows():
    family = family_for_five_tuple(
        _person_family_index(), FamilyQuery(_COUNT_FIVE_TUPLE, _GENDER_UNBOUND, None)
    ).rows
    walked = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE, slots=_GENDER_UNBOUND, graph=_person_family_reader()
    )
    assert {row.dcid for row in family} == {"Person", "Person_Male", "Person_Female"}
    assert set(walked.family) == {row.dcid for row in family}


def test_index_reduction_agrees_with_the_walk_on_the_not_declared_edu_oda_family(
    code_index: CodeIndex,
):
    index_rows = _minimal(code_index, _EDU_ODA_FIVE_TUPLE, _EDU_ODA_SLOTS)

    def _crs_arcs(dcid: str, modality: str | None) -> dict[str, tuple[str, ...]]:
        constraint_props = (
            "crsPurpose",
            "crsTypeOfFinance",
            "crsTypeOfFlow",
            "financialFlowType",
            "nettingBasis",
        ) + (("crsCooperationModality",) if modality else ())
        return {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("FinancialFlow",),
            "measuredProperty": ("amount",),
            "statType": ("measuredValue",),
            "measurementQualifier": ("Nominal",),
            "measurementDenominator": (),
            "constraintProperties": constraint_props,
            "observationProperties": (),
            "crsPurpose": ("CRSPurposeCode11110",),
            "crsTypeOfFinance": ("CRSTypeOfFinance110",),
            "crsTypeOfFlow": ("CRSTypeOfFlow10",),
            "financialFlowType": ("Disbursement",),
            "nettingBasis": ("Gross",),
            **({"crsCooperationModality": (modality,)} if modality else {}),
        }

    reader = FakeMemberArcReader(
        arcs={
            "FixtureStatVar_Amount_FinancialFlow_Disbursement_Gross_Nominal_EduODA_NoModality": (
                _crs_arcs("NoModality", None)
            ),
            "FixtureStatVar_Amount_FinancialFlow_Disbursement_Gross_Nominal_EduODA_GBS": (
                _crs_arcs("GBS", "CRSCooperationModalityA01")
            ),
            "FixtureStatVar_Amount_FinancialFlow_Disbursement_Gross_Nominal_EduODA_CoreNGO": (
                _crs_arcs("CoreNGO", "CRSCooperationModalityB01")
            ),
        }
    )
    walked = enumerate_members(
        five_tuple=_EDU_ODA_FIVE_TUPLE,
        slots=_EDU_ODA_SLOTS,
        graph=reader,
        readings=_readings(code_index),
    )

    assert {row.dcid for row in index_rows} == {member.stat_var for member in walked.members}
    assert {row.dcid for row in index_rows} == _EDU_ODA_DCIDS


_IDS_FIVE_TUPLE = FiveTuple(
    population_type="FinancialPosition",
    measured_property="amount",
    stat_type="measuredValue",
    measurement_qualifier="Nominal",
)
_IDS_CONSTRAINTS = {
    "financialPositionType": "AmountOutstanding",
    "guaranteeStatus": "PublicOrPubliclyGuaranteed",
}
_IDS_SLOTS: dict[str, SlotBinding] = dict.fromkeys(_IDS_CONSTRAINTS, ("unbound",))


def _ids_twin_index(*, by_creditor_constraints: dict[str, str] = _IDS_CONSTRAINTS) -> CodeIndex:
    def _row(dcid: str, roles: tuple[str, ...], constraints: dict[str, str]) -> StatVarRow:
        return StatVarRow(
            dcid=dcid,
            population_type="FinancialPosition",
            measured_property="amount",
            stat_type="measuredValue",
            measurement_qualifier="Nominal",
            measurement_denominator=None,
            observation_properties=roles,
            constraint_properties=constraints,
            provenances=("provenance/Test",),
        )

    return CodeIndex(
        stamp=_STAMP,
        declarations=(),
        codes=(),
        statvars=(
            _row(
                "PPG_ByCreditor",
                ("financingProvider", "financingRecipient"),
                by_creditor_constraints,
            ),
            _row("PPG", ("financingRecipient",), _IDS_CONSTRAINTS),
        ),
    )


def _twin_family(index: CodeIndex, *role_keys: str) -> list[str]:
    query = FamilyQuery(_IDS_FIVE_TUPLE, _IDS_SLOTS, None, role_keys=frozenset(role_keys))
    return [row.dcid for row in family_for_five_tuple(index, query).rows]


def test_index_role_twins_resolve_to_the_total_when_no_creditor_is_bound():
    for keys in ((), ("destination",)):
        assert _twin_family(_ids_twin_index(), *keys) == ["PPG"]


def test_index_role_twins_resolve_to_the_breakdown_when_the_creditor_is_bound():
    assert _twin_family(_ids_twin_index(), "origin", "destination") == ["PPG_ByCreditor"]


@pytest.mark.parametrize(
    ("open_roles", "expected"),
    [((), ["PPG"]), (("origin",), ["PPG_ByCreditor"])],
    ids=["unnamed-creditor", "open-creditor"],
)
def test_an_identitys_open_creditor_resolves_the_twins_to_the_breakdown(open_roles, expected):
    """An open role joins the role keys the identity elects under, as a bound one does, so
    a debtor with every creditor asked for reads the by-creditor twin."""
    index = _ids_twin_index()
    identity: dict[str, object] = {
        "shape": {
            "kind": "statvar_schema",
            "population_type": _IDS_FIVE_TUPLE.population_type,
            "measured_property": _IDS_FIVE_TUPLE.measured_property,
            "stat_type": _IDS_FIVE_TUPLE.stat_type,
            "measurement_qualifier": _IDS_FIVE_TUPLE.measurement_qualifier,
        },
        "slots": [{"property": name, "binding": {"state": "unbound"}} for name in _IDS_SLOTS],
        "entities": {"destination": ["country/NGA"]},
    }
    if open_roles:
        identity["open_roles"] = list(open_roles)

    family = family_for_five_tuple(index, identity_family_query(index, identity))

    assert [row.dcid for row in family.rows] == expected


@pytest.mark.parametrize(
    ("role_keys", "open_role_keys", "expected", "elected_keys"),
    [
        (("destination",), (), ["PPG"], {"destination"}),
        (("destination",), ("origin",), ["PPG"], {"destination"}),
        (("origin", "destination"), (), ["PPG_ByCreditor"], {"origin", "destination"}),
    ],
    ids=["unnamed-creditor", "open-creditor", "bound-creditor"],
)
def test_a_total_answers_beside_a_by_creditor_row_of_other_constraint_values(
    role_keys, open_role_keys, expected, elected_keys
):
    """Equal constraint keys with another value are a different population, so the total
    does not displace the by-creditor row as a twin. The row declares a creditor the
    question does not ask for and leaves for the total, an open creditor no twin of the
    total places is set aside, and a bound creditor reads the row."""
    index = _ids_twin_index(
        by_creditor_constraints={**_IDS_CONSTRAINTS, "guaranteeStatus": "Private"}
    )
    query = FamilyQuery(
        _IDS_FIVE_TUPLE,
        _IDS_SLOTS,
        None,
        role_keys=frozenset(role_keys),
        open_role_keys=frozenset(open_role_keys),
    )

    family = family_for_five_tuple(index, query)

    assert [row.dcid for row in family.rows] == expected
    assert family.role_keys == elected_keys
    assert not family.tied


_FLOW_FIVE_TUPLE = FiveTuple(
    population_type="FinancialFlow",
    measured_property="amount",
    stat_type="measuredValue",
    measurement_qualifier="Nominal",
)
_NET_DISBURSEMENT = {"financialFlowType": "Disbursement", "nettingBasis": "Net"}
_DEBTOR = ("financingRecipient",)
_DEBTOR_AND_CREDITOR = ("financingProvider", "financingRecipient")
# Two listed IDS-like rows of each role set, the second narrowed by a guarantee status, and
# an index row of the same five-tuple the IDS slots cannot admit (it declares `crsPurpose`)
# beside one they can.
_LISTED_FAMILY: dict[str, tuple[tuple[str, ...], dict[str, str], bool]] = {
    "Ids_Net": (_DEBTOR, _NET_DISBURSEMENT, True),
    "Ids_Net_ByCreditor": (_DEBTOR_AND_CREDITOR, _NET_DISBURSEMENT, True),
    "Ids_Net_Ppg": (_DEBTOR, {**_NET_DISBURSEMENT, "guaranteeStatus": "Ppg"}, True),
    "Ids_Net_Ppg_ByCreditor": (
        _DEBTOR_AND_CREDITOR,
        {**_NET_DISBURSEMENT, "guaranteeStatus": "Ppg"},
        True,
    ),
    "Crs_Net_Education": (_DEBTOR_AND_CREDITOR, {**_NET_DISBURSEMENT, "crsPurpose": "Edu"}, False),
    "Crs_Net_Ppg": (_DEBTOR, {**_NET_DISBURSEMENT, "guaranteeStatus": "Ppg"}, False),
}


def _flow_row(dcid: str) -> StatVarRow:
    roles, constraints, _listed = _LISTED_FAMILY[dcid]
    return StatVarRow(
        dcid=dcid,
        population_type=_FLOW_FIVE_TUPLE.population_type,
        measured_property=_FLOW_FIVE_TUPLE.measured_property,
        stat_type=_FLOW_FIVE_TUPLE.stat_type,
        measurement_qualifier=_FLOW_FIVE_TUPLE.measurement_qualifier,
        measurement_denominator=None,
        observation_properties=roles,
        constraint_properties=constraints,
        provenances=("provenance/Listed",) if _listed else ("provenance/Test",),
    )


def _listed_index(*listed: str) -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=(),
        codes=(),
        statvars=[
            _flow_row(dcid) for dcid, (*_, is_listed) in _LISTED_FAMILY.items() if not is_listed
        ],
        listed_statvars=[_flow_row(dcid) for dcid in listed],
    )


def _listed_family_reader() -> FakeMemberArcReader:
    def _arcs(dcid: str) -> dict[str, tuple[str, ...]]:
        roles, constraints, _listed = _LISTED_FAMILY[dcid]
        return {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("FinancialFlow",),
            "measuredProperty": ("amount",),
            "statType": ("measuredValue",),
            "measurementQualifier": ("Nominal",),
            "measurementDenominator": (),
            "constraintProperties": tuple(constraints),
            "observationProperties": roles,
            **{prop: (value,) for prop, value in constraints.items()},
        }

    return FakeMemberArcReader(arcs={dcid: _arcs(dcid) for dcid in _LISTED_FAMILY})


@pytest.mark.parametrize(
    ("slots", "role_keys", "open_role_keys"),
    [
        ({"guaranteeStatus": ("unbound",)}, ("destination",), ()),
        ({"guaranteeStatus": ("unbound",)}, ("origin", "destination"), ()),
        ({"guaranteeStatus": ("unbound",)}, ("destination",), ("origin",)),
        ({"guaranteeStatus": ("value", "Ppg")}, ("destination",), ()),
        ({"guaranteeStatus": ("value", "Ppg")}, (), ()),
    ],
    ids=["total", "by-creditor", "open-creditor", "bound-guarantee", "no-roles"],
)
def test_a_listed_family_elects_the_walks_members_with_no_graph_read(
    slots, role_keys, open_role_keys
):
    """The index's listed and exported rows of a five-tuple stand for every StatVar the walk
    reads there, so the election over them is the walk's: same members, same family."""
    index = _listed_index("Ids_Net", "Ids_Net_ByCreditor", "Ids_Net_Ppg", "Ids_Net_Ppg_ByCreditor")
    full_slots: dict[str, SlotBinding] = {
        "financialFlowType": ("value", "Disbursement"),
        "nettingBasis": ("value", "Net"),
        **slots,
    }
    query = FamilyQuery(
        _FLOW_FIVE_TUPLE,
        full_slots,
        None,
        role_keys=frozenset(role_keys),
        open_role_keys=frozenset(open_role_keys),
    )

    listed = listed_group(index, query, member_cap=700)
    walked = enumerate_members(
        five_tuple=_FLOW_FIVE_TUPLE,
        slots=full_slots,
        graph=_listed_family_reader(),
        role_keys=role_keys,
        open_role_keys=open_role_keys,
    )

    assert listed is not None
    assert listed.members == walked.members
    assert set(listed.family) == set(walked.family)
    assert listed.elected_open_role_keys == walked.elected_open_role_keys


def test_a_family_with_no_listed_row_is_left_to_the_walk():
    """A five-tuple with no listed row, and one whose listed rows the query admits none of,
    return `None`, so the walk resolves them."""
    slots: dict[str, SlotBinding] = {
        "financialFlowType": ("value", "Disbursement"),
        "nettingBasis": ("value", "Net"),
        "guaranteeStatus": ("value", "Ppg"),
    }
    query = FamilyQuery(_FLOW_FIVE_TUPLE, slots, None, role_keys=frozenset({"destination"}))

    assert listed_group(_listed_index(), query, member_cap=700) is None
    assert listed_group(_listed_index("Ids_Net_ByCreditor"), query, member_cap=700) is None
    assert listed_group(_listed_index("Ids_Net_Ppg"), query, member_cap=700) is not None
