"""`elect_dataset` keeps the rows of the one dataset a five-tuple belongs to, and
`index_covers_slot_keys` decides whether the index resolves an identity.

A synthetic index puts each dataset on its own five-tuple, so no rule depends on a property
name. The packaged index pins the education-aid families' counts.
"""

from __future__ import annotations

import dataclasses

import pytest

from code_index import default_index_path, load_index
from code_index.members import (
    FamilyQuery,
    SourceNotInFamily,
    absent_member_reads_zero,
    antichain_minimal_rows,
    elect_dataset,
    family_for_five_tuple,
    identity_family_query,
    index_covers_slot_keys,
    index_family,
    named_dataset,
)
from code_index.query import CodeIndex
from code_index.types import (
    AbsentObservation,
    CodeSystemDeclaration,
    DatasetDeclaration,
    IndexStamp,
    StatVarRow,
)
from statvar_members import FiveTuple, SlotBinding

from .aid_identity import (
    AID_TYPE,
    CRS_MEMBER_COUNT,
    CRS_MEMBERS_WITHOUT_MODALITY,
    CRS_PROVENANCE,
    IMPUTED_MEMBER_COUNT,
    IMPUTED_MULTILATERAL,
    IMPUTED_PROVENANCE,
)

_PLAIN = "provenance/Plain"
_LINES = "provenance/Lines"
_TABLES = "provenance/Tables"
_LINE = "tableLine"

_FIVE_TUPLE = FiveTuple(
    population_type="TestFlow", measured_property="amount", stat_type="measuredValue"
)
_LINES_FIVE_TUPLE = dataclasses.replace(_FIVE_TUPLE, measured_property="lineAmount")

_STAMP = IndexStamp(
    graph_endpoint="test",
    exported_at="2026-01-01T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="test",
)


def _row(
    dcid: str,
    constraints: dict[str, str],
    *provenances: str,
    five_tuple: FiveTuple = _FIVE_TUPLE,
) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type=five_tuple.population_type,
        measured_property=five_tuple.measured_property or "amount",
        stat_type=five_tuple.stat_type,
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=("recipient",),
        constraint_properties=constraints,
        provenances=provenances,
    )


def _dataset(
    provenance: str, *, absent_observation: AbsentObservation = "zero"
) -> DatasetDeclaration:
    return DatasetDeclaration(
        provenance=provenance,
        label=provenance,
        synonyms=(),
        opt_in=False,
        absent_observation=absent_observation,
    )


def _not_declared(code_system: str, *, summable: bool = True) -> CodeSystemDeclaration:
    """A code system whose absent cell blocks dominance, as every packaged system reads."""
    return CodeSystemDeclaration(
        code_system=code_system,
        hierarchical=False,
        disjoint_from_children=False,
        summable=summable,
        absent_reading="not_declared",
        plain_name=code_system,
    )


def _index(
    *rows: StatVarRow,
    datasets: tuple[DatasetDeclaration, ...],
    declarations: tuple[CodeSystemDeclaration, ...] = (
        _not_declared("sector"),
        _not_declared(_LINE, summable=False),
    ),
) -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=declarations,
        codes=(),
        statvars=rows,
        datasets=datasets,
    )


_PLAIN_AND_LINES = (_dataset(_PLAIN), _dataset(_LINES))

_PLAIN_ROWS = (
    _row("plain_a", {"sector": "A"}, _PLAIN),
    _row("plain_b", {"sector": "B"}, _PLAIN),
)
_LINE_ROWS = (
    _row("line_a", {"sector": "A", _LINE: "L1"}, _LINES, five_tuple=_LINES_FIVE_TUPLE),
    _row("line_b", {"sector": "B", _LINE: "L1"}, _LINES, five_tuple=_LINES_FIVE_TUPLE),
)


def _dcids(rows: tuple[StatVarRow, ...]) -> set[str]:
    return {row.dcid for row in rows}


def _asks(
    bound: set[str], source: str | None = None, *, five_tuple: FiveTuple = _FIVE_TUPLE
) -> FamilyQuery:
    """A query on `five_tuple` binding each property in `bound` and naming `source`."""
    return FamilyQuery(five_tuple, dict.fromkeys(bound, ("value", "any")), source)


def _rows(
    index: CodeIndex,
    five_tuple: FiveTuple,
    slots: dict[str, SlotBinding],
    source: str | None = None,
) -> tuple[StatVarRow, ...]:
    return family_for_five_tuple(index, FamilyQuery(five_tuple, slots, source)).rows


# ---------------------------------------------------------------------------
# The election
# ---------------------------------------------------------------------------


def test_the_family_is_the_rows_of_the_dataset_its_five_tuple_belongs_to():
    index = _index(*_PLAIN_ROWS, *_LINE_ROWS, datasets=_PLAIN_AND_LINES)

    plain = elect_dataset(index, index.statvars, _asks(set()))
    lines = elect_dataset(index, index.statvars, _asks(set(), five_tuple=_LINES_FIVE_TUPLE))

    assert (plain.provenance, _dcids(plain.rows)) == (_PLAIN, {"plain_a", "plain_b"})
    assert (lines.provenance, _dcids(lines.rows)) == (_LINES, {"line_a", "line_b"})


@pytest.mark.parametrize(
    ("binding", "named"),
    [
        ({"state": "value", "ref": _LINES}, _LINES),
        ({"state": "set", "refs": [_LINES]}, _LINES),
        ({"state": "set", "refs": [_LINES, _LINES]}, _LINES),
        ({"state": "set", "refs": [_LINES, _TABLES]}, None),
        ({"state": "unbound"}, None),
        (None, None),
    ],
)
def test_the_provenance_facet_names_a_dataset_when_it_binds_one_distinct_ref(binding, named):
    facets = {} if binding is None else {"provenance": binding}

    assert named_dataset(facets) == named


def test_a_one_ref_set_and_a_value_ask_the_index_one_query():
    """Both spellings of the facet and of a slot hash to one `spec_id`, so they read one
    `FamilyQuery`."""
    index = _index(*_PLAIN_ROWS, *_LINE_ROWS, datasets=_PLAIN_AND_LINES)

    def identity(facet: dict, slot: dict) -> dict:
        return {
            "shape": {
                "kind": "statvar_schema",
                "population_type": _LINES_FIVE_TUPLE.population_type,
                "measured_property": _LINES_FIVE_TUPLE.measured_property,
                "stat_type": _LINES_FIVE_TUPLE.stat_type,
            },
            "slots": [{"property": _LINE, "binding": slot}],
            "facets": {"provenance": facet},
        }

    as_value = identity_family_query(
        index,
        identity({"state": "value", "ref": _LINES}, {"state": "value", "ref": "L1"}),
    )
    as_set = identity_family_query(
        index,
        identity({"state": "set", "refs": [_LINES]}, {"state": "set", "refs": ["L1"]}),
    )

    assert as_value.source == _LINES
    assert as_set == as_value


def test_a_named_source_that_publishes_the_five_tuple_elects_its_rows():
    index = _index(*_PLAIN_ROWS, *_LINE_ROWS, datasets=_PLAIN_AND_LINES)

    family = elect_dataset(index, _LINE_ROWS, _asks(set(), _LINES, five_tuple=_LINES_FIVE_TUPLE))

    assert (family.provenance, _dcids(family.rows)) == (_LINES, {"line_a", "line_b"})


def test_a_named_source_another_dataset_publishes_the_five_tuple_of_refuses():
    index = _index(*_PLAIN_ROWS, *_LINE_ROWS, datasets=_PLAIN_AND_LINES)

    with pytest.raises(SourceNotInFamily) as refusal:
        elect_dataset(index, _LINE_ROWS, _asks({_LINE}, _PLAIN, five_tuple=_LINES_FIVE_TUPLE))

    assert refusal.value.source == _PLAIN
    assert refusal.value.candidates == (_LINES,)


def test_rows_of_another_five_tuple_stay_out_of_the_family():
    """A measure family hands the election rows of several five-tuples. The query's own
    five-tuple decides the dataset, and another dataset's rows stay out."""
    index = _index(*_PLAIN_ROWS, *_LINE_ROWS, datasets=_PLAIN_AND_LINES)

    family = elect_dataset(index, (*_PLAIN_ROWS, *_LINE_ROWS), _asks(set()))

    assert _dcids(family.rows) == {"plain_a", "plain_b"}


def test_a_dataset_is_elected_without_a_declaration():
    index = _index(*_PLAIN_ROWS, datasets=())

    family = elect_dataset(index, index.statvars, _asks(set()))

    assert family.provenance == _PLAIN
    assert _dcids(family.rows) == {"plain_a", "plain_b"}


def test_a_row_with_no_provenance_joins_no_family():
    uncovered = _row("uncovered", {"sector": "C"})
    index = _index(*_PLAIN_ROWS, uncovered, datasets=_PLAIN_AND_LINES)

    family = elect_dataset(index, index.statvars, _asks(set()))
    assert _dcids(family.rows) == {"plain_a", "plain_b"}

    alone = elect_dataset(index, (uncovered,), _asks(set()))
    assert alone.provenance is None
    assert alone.rows == ()


def test_family_for_five_tuple_returns_the_elected_family():
    index = _index(*_PLAIN_ROWS, *_LINE_ROWS, datasets=_PLAIN_AND_LINES)
    unbound: dict[str, SlotBinding] = {"sector": ("unbound",), _LINE: ("unbound",)}
    bound: dict[str, SlotBinding] = {"sector": ("unbound",), _LINE: ("value", "L1")}

    assert _dcids(_rows(index, _FIVE_TUPLE, unbound)) == {"plain_a", "plain_b"}
    assert _dcids(_rows(index, _LINES_FIVE_TUPLE, bound)) == {"line_a", "line_b"}
    assert _dcids(_rows(index, _LINES_FIVE_TUPLE, unbound, _LINES)) == {"line_a", "line_b"}


# ---------------------------------------------------------------------------
# Role classes
# ---------------------------------------------------------------------------

_PROVIDER = ("provider",)
_PAIR = ("provider", "recipient")


def _role_row(
    dcid: str, constraints: dict[str, str], roles: tuple[str, ...], *provenances: str
) -> StatVarRow:
    return dataclasses.replace(_row(dcid, constraints, *provenances), observation_properties=roles)


def _asks_roles(*role_keys: str, bound: frozenset[str] = frozenset()) -> FamilyQuery:
    return dataclasses.replace(_asks(set(bound)), role_keys=frozenset(role_keys))


def test_a_provider_only_row_answers_a_provider_and_two_role_rows_answer_a_named_recipient():
    """A DAC 1-style line beside provider-and-recipient rows on one five-tuple, in one
    dataset: the line is the total over recipients."""
    rows = (
        _role_row("line", {"sector": "A"}, _PROVIDER, _PLAIN),
        _role_row("flow", {"sector": "A"}, _PAIR, _PLAIN),
    )
    index = _index(*rows, datasets=(_dataset(_PLAIN),))

    assert _dcids(elect_dataset(index, rows, _asks_roles("origin")).rows) == {"line"}
    assert _dcids(elect_dataset(index, rows, _asks_roles()).rows) == {"line"}
    assert _dcids(elect_dataset(index, rows, _asks_roles("origin", "destination")).rows) == {"flow"}
    assert _dcids(elect_dataset(index, rows, _asks_roles("destination")).rows) == {"flow"}


def test_role_keys_no_row_covers_keep_the_family_for_the_reproducibility_check():
    rows = (_role_row("flow", {"sector": "A"}, _PAIR, _PLAIN),)
    index = _index(*rows, datasets=(_dataset(_PLAIN),))

    assert _dcids(elect_dataset(index, rows, _asks_roles("observationAbout")).rows) == {"flow"}


def test_role_sets_the_election_cannot_order_stay_in_one_family():
    """The caller refuses a family that holds two role sets."""
    rows = (
        _role_row("line", {"sector": "A"}, _PROVIDER, _PLAIN),
        _role_row("total", {"sector": "A"}, ("recipient",), _PLAIN),
    )
    index = _index(*rows, datasets=(_dataset(_PLAIN),))

    assert _dcids(elect_dataset(index, rows, _asks_roles()).rows) == {"line", "total"}


# ---------------------------------------------------------------------------
# The index-path gate
# ---------------------------------------------------------------------------


def test_slot_keys_that_cover_a_row_take_the_index_whatever_their_values():
    index = _index(*_PLAIN_ROWS, *_LINE_ROWS, datasets=_PLAIN_AND_LINES)

    assert index_covers_slot_keys(index, _FIVE_TUPLE, {"sector"}) is True
    assert index_covers_slot_keys(index, _FIVE_TUPLE, {"sector", _LINE, "other"}) is True


def test_slot_keys_that_cover_no_row_leave_the_index():
    index = _index(*_LINE_ROWS, datasets=_PLAIN_AND_LINES)

    assert index_covers_slot_keys(index, _LINES_FIVE_TUPLE, set()) is False
    assert index_covers_slot_keys(index, _LINES_FIVE_TUPLE, {"sector"}) is False
    assert index_covers_slot_keys(index, _LINES_FIVE_TUPLE, {"sector", _LINE}) is True


def test_a_five_tuple_the_index_does_not_carry_is_not_covered():
    index = _index(*_PLAIN_ROWS, datasets=_PLAIN_AND_LINES)
    absent = FiveTuple(measured_property="nonexistentMeasure")

    assert index_covers_slot_keys(index, absent, {"sector"}) is False


# ---------------------------------------------------------------------------
# The education-aid families in the packaged index
# ---------------------------------------------------------------------------

_MODALITY = "crsCooperationModality"
_EDU_AID_FIVE_TUPLE = FiveTuple(
    population_type="FinancialFlow",
    measured_property="amount",
    stat_type="measuredValue",
    measurement_qualifier="Nominal",
)
_IMPUTED_FIVE_TUPLE = dataclasses.replace(_EDU_AID_FIVE_TUPLE, measured_property="attributedAmount")


def _education_aid_slots(index: CodeIndex, *, aid_type: SlotBinding) -> dict[str, SlotBinding]:
    return {
        _MODALITY: ("unbound",),
        "crsPurpose": ("set", index.descendants("CRSPurposeCode110")),
        "crsTypeOfFinance": ("unbound",),
        "crsTypeOfFlow": ("value", "CRSTypeOfFlow10"),
        AID_TYPE: aid_type,
        "financialFlowType": ("value", "Disbursement"),
        "nettingBasis": ("value", "Gross"),
    }


def _minimal(
    index: CodeIndex,
    slots: dict[str, SlotBinding],
    *,
    five_tuple: FiveTuple = _EDU_AID_FIVE_TUPLE,
    source: str | None = None,
) -> tuple[StatVarRow, ...]:
    return antichain_minimal_rows(index, _rows(index, five_tuple, slots, source))


def test_the_education_family_on_amount_is_the_crs_rows():
    index = load_index(default_index_path())

    rows = _minimal(index, _education_aid_slots(index, aid_type=("unbound",)))

    assert len(rows) == CRS_MEMBER_COUNT
    assert {row.provenances for row in rows} == {(CRS_PROVENANCE,)}


def test_the_education_members_without_a_modality_stay():
    index = load_index(default_index_path())

    rows = _minimal(index, _education_aid_slots(index, aid_type=("unbound",)))

    without_modality = [row for row in rows if _MODALITY not in row.constraint_properties]
    assert len(without_modality) == CRS_MEMBERS_WITHOUT_MODALITY


@pytest.mark.parametrize("aid_type", [("unbound",), ("value", IMPUTED_MULTILATERAL)])
def test_the_education_family_on_attributed_amount_is_the_imputed_rows(aid_type: SlotBinding):
    """The aid type holds one value on every imputed row, so binding it changes no member."""
    index = load_index(default_index_path())

    rows = _minimal(
        index, _education_aid_slots(index, aid_type=aid_type), five_tuple=_IMPUTED_FIVE_TUPLE
    )

    assert len(rows) == IMPUTED_MEMBER_COUNT
    assert {row.provenances for row in rows} == {(IMPUTED_PROVENANCE,)}


def test_the_crs_source_on_attributed_amount_refuses():
    index = load_index(default_index_path())
    slots = _education_aid_slots(index, aid_type=("unbound",))

    with pytest.raises(SourceNotInFamily) as refusal:
        _rows(index, _IMPUTED_FIVE_TUPLE, slots, CRS_PROVENANCE)

    assert refusal.value.candidates == (IMPUTED_PROVENANCE,)


# ---------------------------------------------------------------------------
# The family the index resolves, and how its dataset reads an absent member
# ---------------------------------------------------------------------------


def test_index_family_names_the_elected_dataset_beside_its_rows():
    index = _index(*_PLAIN_ROWS, *_LINE_ROWS, datasets=_PLAIN_AND_LINES)
    unbound: dict[str, SlotBinding] = {"sector": ("unbound",), _LINE: ("unbound",)}
    named: dict[str, SlotBinding] = {"sector": ("unbound",), _LINE: ("value", "L1")}

    plain = index_family(index, FamilyQuery(_FIVE_TUPLE, unbound, None))
    lines = index_family(index, FamilyQuery(_LINES_FIVE_TUPLE, named, None))

    assert plain is not None
    assert lines is not None
    assert (plain.provenance, _dcids(plain.rows)) == (_PLAIN, {"plain_a", "plain_b"})
    assert (lines.provenance, _dcids(lines.rows)) == (_LINES, {"line_a", "line_b"})
    assert _rows(index, _LINES_FIVE_TUPLE, named) == lines.rows


def test_index_family_is_none_when_the_slot_keys_cover_no_row():
    index = _index(*_PLAIN_ROWS, datasets=_PLAIN_AND_LINES[:1])

    assert index_family(index, FamilyQuery(_FIVE_TUPLE, {}, None)) is None
    assert (
        index_family(index, FamilyQuery(FiveTuple(population_type="Elsewhere"), {}, None)) is None
    )


def test_an_absent_member_reads_zero_only_under_a_dataset_that_declares_it():
    declares_nothing = _dataset(_LINES, absent_observation="unknown")
    index = _index(*_PLAIN_ROWS, *_LINE_ROWS, datasets=(_dataset(_PLAIN), declares_nothing))

    assert absent_member_reads_zero(index, _PLAIN) is True
    assert absent_member_reads_zero(index, _LINES) is False
    assert absent_member_reads_zero(index, "provenance/Undeclared") is False
    assert absent_member_reads_zero(index, None) is False
