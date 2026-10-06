"""`elect_dataset` keeps the rows of the one dataset elected among those publishing a
five-tuple, and `index_covers_slot_keys` decides whether the index resolves an identity.

Synthetic indexes put datasets on their own five-tuples and on shared ones, so no rule
depends on a property name. The packaged index pins the education-aid families' counts.
"""

from __future__ import annotations

import dataclasses

import pytest

from code_index import default_index_path, load_index
from code_index import members as members_module
from code_index.members import (
    FamilyQuery,
    SourceNotInFamily,
    absent_member_reads_zero,
    antichain_minimal_rows,
    bind_slot_value,
    canonical_identity,
    deciding_query,
    elect_dataset,
    elected_invariants,
    elected_pins,
    eligible_among,
    eligible_datasets,
    family_among,
    family_declares_roles,
    family_for_five_tuple,
    identity_family_query,
    index_covers_slot_keys,
    index_family,
    invariant_bindings,
    named_dataset,
    option_families,
    query_family,
)
from code_index.query import CodeIndex
from code_index.types import (
    AbsentObservation,
    CodeSystemDeclaration,
    DatasetDeclaration,
    IndexStamp,
    StatVarRow,
)
from spec_registry import recompute_identity
from statvar_members import FiveTuple, SlotBinding

from .aid_identity import (
    AID_TYPE,
    CRS_MEMBER_COUNT,
    CRS_MEMBERS_WITHOUT_MODALITY,
    CRS_PROVENANCE,
    IMPUTED_MEMBER_COUNT,
    IMPUTED_MULTILATERAL,
    IMPUTED_PROVENANCE,
    education_aid_identity,
    imputed_education_aid_identity,
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
    provenance: str,
    *,
    absent_observation: AbsentObservation = "zero",
    rank: int = 0,
    opt_in: bool = False,
    invariants: dict[str, str] | None = None,
) -> DatasetDeclaration:
    return DatasetDeclaration(
        provenance=provenance,
        label=provenance,
        description="A test table.",
        opt_in=opt_in,
        absent_observation=absent_observation,
        rank=rank,
        invariants=invariants or {},
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


@pytest.mark.parametrize(
    ("rows", "source"),
    [
        ((), _PLAIN),
        (_LINE_ROWS, None),
    ],
    ids=["named-publisher-no-rows", "no-publisher-row"],
)
def test_no_published_row_gives_an_empty_family(rows: tuple[StatVarRow, ...], source: str | None):
    """A named dataset that publishes the five-tuple but none of the rows, and rows no
    candidate publishes, leave the family empty without a refusal."""
    index = _index(*_PLAIN_ROWS, *_LINE_ROWS, datasets=_PLAIN_AND_LINES)

    family = elect_dataset(index, rows, _asks(set(), source))

    assert (family.provenance, family.rows) == (None, ())


def test_a_named_source_on_a_five_tuple_no_dataset_publishes_gives_an_empty_family():
    uncovered = _row("uncovered", {"sector": "C"})
    index = _index(uncovered, datasets=_PLAIN_AND_LINES)

    family = elect_dataset(index, (uncovered,), _asks(set(), _PLAIN))

    assert (family.provenance, family.rows) == (None, ())


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


def _asks_roles(
    *role_keys: str, bound: frozenset[str] = frozenset(), open_role_keys: tuple[str, ...] = ()
) -> FamilyQuery:
    return dataclasses.replace(
        _asks(set(bound)), role_keys=frozenset(role_keys), open_role_keys=frozenset(open_role_keys)
    )


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
# Several datasets on one five-tuple
# ---------------------------------------------------------------------------

_DONOR = ("financingProvider",)
_DONOR_RECIPIENT = ("financingProvider", "financingRecipient")


def _shared(*, plain_rank: int = 0, tables_rank: int = 1, tables_opt_in: bool = False) -> CodeIndex:
    """`_PLAIN` and `_TABLES` on one five-tuple: each its own sector row and one row both
    publish."""
    return _index(
        _row("plain_only", {"sector": "A"}, _PLAIN),
        _row("both", {"sector": "B"}, _PLAIN, _TABLES),
        _row("tables_only", {"sector": "C"}, _TABLES),
        datasets=(
            _dataset(_PLAIN, rank=plain_rank),
            _dataset(_TABLES, rank=tables_rank, opt_in=tables_opt_in),
        ),
    )


@pytest.mark.parametrize(
    ("plain_rank", "tables_rank", "elected", "dcids"),
    [
        (0, 1, _PLAIN, {"plain_only", "both"}),
        (1, 0, _TABLES, {"both", "tables_only"}),
    ],
    ids=["plain-first", "tables-first"],
)
def test_the_lowest_ranked_publisher_serves_a_shared_five_tuple(
    plain_rank: int, tables_rank: int, elected: str, dcids: set[str]
):
    index = _shared(plain_rank=plain_rank, tables_rank=tables_rank)

    family = elect_dataset(index, index.statvars, _asks(set()))

    assert (family.provenance, _dcids(family.rows)) == (elected, dcids)


@pytest.mark.parametrize("source", [_PLAIN, _TABLES])
def test_a_row_with_two_provenances_joins_the_family_of_either(source: str):
    index = _shared()

    family = elect_dataset(index, index.statvars, _asks(set(), source))

    assert family.provenance == source
    assert "both" in _dcids(family.rows)
    assert {row.dcid for row in index.statvars if source in row.provenances} == _dcids(family.rows)


@pytest.mark.parametrize(
    ("source", "excluded", "elected"),
    [
        (None, frozenset(), _PLAIN),
        (_TABLES, frozenset(), _TABLES),
        (None, frozenset({_PLAIN}), None),
    ],
    ids=["unnamed", "named", "only-default-excluded"],
)
def test_an_opt_in_dataset_beside_another_serves_only_a_query_that_names_it(
    source: str | None, excluded: frozenset[str], elected: str | None
):
    """`_TABLES` ranks first but opts in."""
    index = _shared(plain_rank=1, tables_rank=0, tables_opt_in=True)
    query = dataclasses.replace(_asks(set(), source), excluded=excluded)

    assert elect_dataset(index, index.statvars, query).provenance == elected


@pytest.mark.parametrize(
    ("excluded", "elected"),
    [(frozenset(), _TABLES), (frozenset({_TABLES}), None)],
    ids=["unnamed", "excluded"],
)
def test_a_five_tuple_only_opt_in_datasets_publish_elects_one_unnamed(
    excluded: frozenset[str], elected: str | None
):
    index = _index(
        _row("tables_only", {"sector": "C"}, _TABLES),
        datasets=(_dataset(_TABLES, opt_in=True),),
    )
    query = dataclasses.replace(_asks(set()), excluded=excluded)

    assert elect_dataset(index, index.statvars, query).provenance == elected


@pytest.mark.parametrize(
    ("excluded", "elected"),
    [
        (frozenset(), _PLAIN),
        (frozenset({_PLAIN}), _TABLES),
        (frozenset({_PLAIN, _TABLES}), None),
    ],
    ids=["none", "first", "both"],
)
def test_an_excluded_dataset_leaves_the_election_to_the_next(
    excluded: frozenset[str], elected: str | None
):
    index = _shared()
    query = dataclasses.replace(_asks(set()), excluded=excluded)

    assert elect_dataset(index, index.statvars, query).provenance == elected


def test_a_named_source_publishing_another_five_tuple_lists_every_publisher():
    index = _index(
        *_LINE_ROWS,
        *_shared(plain_rank=1, tables_rank=0).statvars,
        datasets=(_dataset(_PLAIN, rank=1), _dataset(_TABLES, rank=0), _dataset(_LINES, rank=2)),
    )

    with pytest.raises(SourceNotInFamily) as refusal:
        elect_dataset(index, index.statvars, _asks(set(), _LINES))

    assert refusal.value.candidates == (_TABLES, _PLAIN)


_CHANNEL = "channel"


def _split_index(*, sparse: bool) -> CodeIndex:
    """`_PLAIN` and `_TABLES` publish health rows under `sector`, and `_PLAIN` an education
    row. `sparse` adds one `_PLAIN` row that also carries `_CHANNEL`, which splits the two
    datasets into families of their own."""
    rows = [
        _row("plain_health", {"sector": "Health"}, _PLAIN),
        _row("plain_education", {"sector": "Education"}, _PLAIN),
        _row("tables_health", {"sector": "Health"}, _TABLES),
    ]
    if sparse:
        rows.append(_row("plain_health_ngo", {"sector": "Health", _CHANNEL: "NGO"}, _PLAIN))
    return _index(
        *rows,
        datasets=(_dataset(_PLAIN, rank=0), _dataset(_TABLES, rank=1)),
        declarations=(_not_declared("sector"), _not_declared(_CHANNEL)),
    )


def _health_identity(source: str | None) -> dict:
    """A saved identity binding `sector` to Health on `_FIVE_TUPLE`, naming `source`."""
    identity: dict = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": _FIVE_TUPLE.population_type,
            "measured_property": _FIVE_TUPLE.measured_property,
            "stat_type": _FIVE_TUPLE.stat_type,
        },
        "slots": [{"property": "sector", "binding": {"state": "value", "ref": "Health"}}],
        "entities": {"destination": ["country/KEN"]},
        "requested_window": {"kind": "all"},
    }
    if source is not None:
        identity["facets"] = {"provenance": {"state": "value", "ref": source}}
    return identity


@pytest.mark.parametrize("sparse", [False, True], ids=["one-family", "split"])
@pytest.mark.parametrize(
    ("source", "unbound", "row"),
    [
        (_PLAIN, None, "plain_health"),
        (_PLAIN, "strayProp", "plain_health"),
        (_TABLES, _CHANNEL, "tables_health"),
    ],
    ids=["declared-slots", "unbound-slot-no-row-declares", "unbound-slot-another-dataset-declares"],
)
def test_a_named_identity_keeps_its_dataset_when_a_sparse_property_splits_the_family(
    sparse: bool, source: str, unbound: str | None, row: str
):
    """A row of `_PLAIN` with one property more puts `_PLAIN` and `_TABLES` in two
    families. The identity naming a dataset resolves that dataset's row before and after,
    though `_TABLES`' family holds fewer properties, and whatever slot it leaves unbound:
    one no row declares, or one only the other dataset's rows carry."""
    index = _split_index(sparse=sparse)
    identity = _health_identity(source)
    if unbound is not None:
        identity["slots"].append({"property": unbound, "binding": {"state": "unbound"}})
    query = identity_family_query(index, identity)

    family = index_family(index, query)

    assert family is not None
    assert (family.provenance, _dcids(family.rows)) == (source, {row})
    assert index_family(index, deciding_query(index, query)) == family


@pytest.mark.parametrize("sparse", [False, True], ids=["one-family", "split"])
def test_a_named_identity_declaring_fewer_slots_than_any_row_is_left_to_the_graph_walk(
    sparse: bool,
):
    """No row's properties lie within the identity's keys, so the index does not decide it.
    `_PLAIN` carries every key, so the identity is not refused after the split either,
    though the keys alone select `_TABLES`' family, which holds fewer properties."""
    index = _split_index(sparse=sparse)
    identity = _health_identity(_PLAIN)
    identity["slots"] = []

    assert index_family(index, identity_family_query(index, identity)) is None


def test_a_named_identity_binding_a_property_its_dataset_lacks_refuses():
    """`_TABLES` carries no `_CHANNEL`, so an identity naming it with `_CHANNEL` bound
    refuses, naming `_PLAIN`, whose family the keys select."""
    index = _split_index(sparse=True)
    identity = _health_identity(_TABLES)
    identity["slots"].insert(0, {"property": _CHANNEL, "binding": {"state": "value", "ref": "NGO"}})

    with pytest.raises(SourceNotInFamily) as refusal:
        index_family(index, identity_family_query(index, identity))

    assert refusal.value.candidates == (_PLAIN,)


@pytest.mark.parametrize(
    "sparse",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                reason="the {sector} keys select the family with fewest properties, "
                "`_TABLES`', where a cover tie broken by serving order would keep `_PLAIN`",
            ),
        ),
    ],
    ids=["one-family", "split"],
)
def test_an_unnamed_identity_keeps_its_dataset_when_a_sparse_property_splits_the_family(
    sparse: bool,
):
    index = _split_index(sparse=sparse)

    family = index_family(index, identity_family_query(index, _health_identity(None)))

    assert family is not None
    assert (family.provenance, _dcids(family.rows)) == (_PLAIN, {"plain_health"})


def test_each_row_belongs_to_the_family_its_provenances_assign():
    """`_PLAIN`'s `sector`-only rows stay in `_PLAIN`'s family, though their keys alone
    select `_TABLES`' family, which holds fewer properties."""
    index = _split_index(sparse=True)
    sector_only = frozenset({"sector"})

    selected = index.family_of(_FIVE_TUPLE, sector_only)
    assert selected is not None
    assert selected.datasets == (_TABLES,)
    assert {dcid: family.datasets for dcid, family in index.family_by_dcid.items()} == {
        "plain_health": (_PLAIN,),
        "plain_education": (_PLAIN,),
        "plain_health_ngo": (_PLAIN,),
        "tables_health": (_TABLES,),
    }
    assert query_family(index, _asks({"sector"}, _PLAIN)) is index.family_by_dcid["plain_health"]
    assert query_family(index, _asks({"sector", _CHANNEL}, _TABLES)) is None
    assert query_family(index, _asks({"sector"}, "provenance/Elsewhere")) is None


def _role_index(*, crs_rank: int, dac1_rank: int) -> CodeIndex:
    """A CRS-like dataset of donor-and-recipient rows and a DAC 1-like dataset of donor-only
    rows on one five-tuple."""
    return _index(
        _role_row("crs", {"sector": "A"}, _DONOR_RECIPIENT, _PLAIN),
        _role_row("dac1", {"sector": "A"}, _DONOR, _TABLES),
        datasets=(_dataset(_PLAIN, rank=crs_rank), _dataset(_TABLES, rank=dac1_rank)),
    )


@pytest.mark.parametrize(
    ("crs_rank", "dac1_rank"), [(0, 1), (1, 0)], ids=["crs-first", "dac1-first"]
)
@pytest.mark.parametrize(
    ("role_keys", "open_role_keys", "elected"),
    [
        (("origin",), (), _TABLES),
        (("origin", "destination"), (), _PLAIN),
        (("destination",), (), _PLAIN),
        (("origin",), ("destination",), _PLAIN),
    ],
    ids=["donor", "donor-recipient", "recipient", "donor-open-recipient"],
)
def test_the_dataset_whose_rows_place_the_role_keys_exactly_serves_first(
    crs_rank: int,
    dac1_rank: int,
    role_keys: tuple[str, ...],
    open_role_keys: tuple[str, ...],
    elected: str,
):
    """A donor alone reads the donor-only table whatever the ranks. A recipient key drops the
    donor-only table, whose rows cover no recipient, even where `elect_role_class` would keep
    them for want of a covering row. An open recipient a row declares counts as a bound one."""
    index = _role_index(crs_rank=crs_rank, dac1_rank=dac1_rank)

    family = elect_dataset(
        index, index.statvars, _asks_roles(*role_keys, open_role_keys=open_role_keys)
    )

    assert family.provenance == elected


@pytest.mark.parametrize(
    ("donor_rank", "subject_rank"), [(0, 1), (1, 0)], ids=["donor-first", "subject-first"]
)
def test_an_open_role_no_row_declares_elects_under_the_bound_keys_alone(
    donor_rank: int, subject_rank: int
):
    """No row covers a donor with an open recipient, so the election reads the donor alone.
    The subject-only table, whose rows cover no donor, drops out whatever its rank, as it
    does when the recipient is unnamed."""
    index = _index(
        _role_row("dac1", {"sector": "A"}, _DONOR, _PLAIN),
        _role_row("subject", {"sector": "A"}, ("observationAbout",), _TABLES),
        datasets=(_dataset(_PLAIN, rank=donor_rank), _dataset(_TABLES, rank=subject_rank)),
    )

    for query in (_asks_roles("origin"), _asks_roles("origin", open_role_keys=("destination",))):
        assert elect_dataset(index, index.statvars, query).provenance == _PLAIN


_TOTAL_A = _role_row("total_a", {"sector": "A"}, _DONOR, _PLAIN)
_TWIN_A = _role_row("twin_a", {"sector": "A"}, _DONOR_RECIPIENT, _PLAIN)
_TOTAL_B = _role_row("total_b", {"sector": "B"}, _DONOR, _PLAIN)
_FLOW_C = _role_row("flow_c", {"sector": "C"}, _DONOR_RECIPIENT, _PLAIN)


@pytest.mark.parametrize(
    ("rows", "elected", "elected_keys"),
    [
        ((_TOTAL_A, _TOTAL_B, _FLOW_C), {"total_a", "total_b"}, {"origin"}),
        ((_TOTAL_A, _TWIN_A, _FLOW_C), {"twin_a", "flow_c"}, {"origin", "destination"}),
        ((_TOTAL_A, _TWIN_A, _TOTAL_B), {"twin_a"}, {"origin", "destination"}),
    ],
    ids=["totals-beside-one-flow", "one-total-twinned-beside-a-flow", "one-total-untwinned"],
)
def test_an_open_role_joins_where_a_total_or_its_twin_holds_it(
    rows: tuple[StatVarRow, ...], elected: set[str], elected_keys: set[str]
):
    """A donor with every recipient asked for reads the donor totals, and the open
    recipient joins where a twin of a total holds it: the by-recipient rows then answer,
    a flow of other constraint values among them. A by-recipient row that is a twin of no
    total does not stand for the totals beside it, and the dataset reads as it does for
    the donor alone. `DatasetFamily.role_keys` states which."""
    index = _index(*rows, datasets=(_dataset(_PLAIN),))
    query = _asks_roles("origin", open_role_keys=("destination",))

    family = elect_dataset(index, index.statvars, query)

    assert _dcids(family.rows) == elected
    assert family.role_keys == elected_keys
    assert not family.tied
    assert family == elect_dataset(index, index.statvars, _asks_roles(*elected_keys))


def test_without_an_exact_match_the_rank_decides_between_covering_datasets():
    rows = (
        _role_row("crs", {"sector": "A"}, _DONOR_RECIPIENT, _PLAIN),
        _role_row("crs_twin", {"sector": "A"}, _DONOR_RECIPIENT, _TABLES),
    )
    index = _index(*rows, datasets=(_dataset(_PLAIN, rank=1), _dataset(_TABLES, rank=0)))

    assert elect_dataset(index, rows, _asks_roles("origin")).provenance == _TABLES


_CREDITOR = ("creditor",)
_SERVICE_PUBLIC_CREDITOR = _role_row(
    "service_public", {"flow": "Service", "guarantee": "Public"}, _CREDITOR, _PLAIN, _TABLES
)
_SERVICE_PRIVATE_DEBTOR = _role_row(
    "private_debtor", {"flow": "Service", "guarantee": "Private"}, ("debtor",), _PLAIN
)
_SERVICE_PRIVATE_CREDITOR = _role_row(
    "private_creditor", {"flow": "Service", "guarantee": "Private"}, _CREDITOR, _TABLES
)

_TWO_PUBLISHER_CASES = {
    "a-tied-dataset-leaves-for-one-that-answers": (
        (
            _role_row("plain_pair", {"term": "Long"}, _PAIR, _PLAIN),
            _role_row(
                "plain_channel",
                {"flow": "Service", "guarantee": "Public"},
                ("provider", "channel"),
                _PLAIN,
            ),
            _role_row(
                "tables_pair",
                {"flow": "Service", "guarantee": "Public", "term": "Long"},
                _PAIR,
                _TABLES,
            ),
        ),
        ("origin",),
        (),
        (_TABLES, {"tables_pair"}, {"origin"}, False),
    ),
    "open-keys-one-dataset-cannot-order": (
        (
            _role_row("plain_provider", {"flow": "Service"}, _PROVIDER, _PLAIN),
            _role_row("plain_recipient", {"guarantee": "Public"}, ("recipient",), _PLAIN),
            _role_row(
                "tables_provider", {"flow": "Service", "guarantee": "Public"}, _PROVIDER, _TABLES
            ),
        ),
        (),
        ("origin", "destination"),
        (_PLAIN, {"plain_provider"}, {"origin"}, False),
    ),
    "a-shared-row-under-one-open-key": (
        (_SERVICE_PRIVATE_CREDITOR, _SERVICE_PUBLIC_CREDITOR, _SERVICE_PRIVATE_DEBTOR),
        (),
        ("origin", "destination"),
        (_PLAIN, {"service_public"}, {"origin"}, False),
    ),
    "a-shared-row-in-a-dataset-the-joined-key-ties": (
        (
            _SERVICE_PRIVATE_CREDITOR,
            _SERVICE_PUBLIC_CREDITOR,
            _SERVICE_PRIVATE_DEBTOR,
            _role_row(
                "plain_provider", {"flow": "Disbursement", "guarantee": "Public"}, _DONOR, _PLAIN
            ),
        ),
        (),
        ("origin", "destination"),
        (_TABLES, {"private_creditor", "service_public"}, {"origin"}, False),
    ),
}


@pytest.mark.parametrize(
    ("rows", "role_keys", "open_role_keys", "elected"),
    _TWO_PUBLISHER_CASES.values(),
    ids=_TWO_PUBLISHER_CASES,
)
def test_two_publishers_elect_one_family_for_a_question_and_the_keys_it_reports(
    rows: tuple[StatVarRow, ...],
    role_keys: tuple[str, ...],
    open_role_keys: tuple[str, ...],
    elected: tuple[str, set[str], set[str], bool],
):
    """`_PLAIN` ranks ahead of `_TABLES`. A dataset whose own election ties leaves for one
    that answers. With both axes open, the family is the one the open keys that joined
    elect on their own: `_PLAIN` reads its provider row once the recipient is set aside,
    and stays out where the origin alone still ties its creditor row with a provider row.
    Asking again with only the joined keys open returns the same family, which is the
    identity Search mints for the answer."""
    index = _index(
        *rows, datasets=(_dataset(_PLAIN, rank=0), _dataset(_TABLES, rank=1)), declarations=()
    )
    query = _asks_roles(*role_keys, open_role_keys=open_role_keys)

    family = elect_dataset(index, index.statvars, query)

    assert (family.provenance, _dcids(family.rows), family.role_keys, family.tied) == elected
    joined = dataclasses.replace(query, open_role_keys=family.role_keys - query.role_keys)
    assert elect_dataset(index, index.statvars, joined) == family


@pytest.mark.parametrize(
    ("datasets", "source", "excluded", "eligible"),
    [
        ((_TABLES, _PLAIN), None, frozenset(), (_PLAIN,)),
        ((_TABLES, _PLAIN), _TABLES, frozenset(), (_TABLES,)),
        ((_TABLES, _PLAIN), None, frozenset({_PLAIN}), ()),
        ((_TABLES,), None, frozenset(), (_TABLES,)),
        ((_TABLES,), None, frozenset({_TABLES}), ()),
    ],
    ids=["unnamed", "named", "excluded", "opt-in-alone", "opt-in-alone-excluded"],
)
def test_the_eligibility_rules_filter_the_datasets_a_caller_holds(
    datasets: tuple[str, ...],
    source: str | None,
    excluded: frozenset[str],
    eligible: tuple[str, ...],
):
    """`_TABLES` ranks first but opts in. A list holding it alone is not the family's
    publishers, so the filter reads the list it is given."""
    index = _shared(plain_rank=1, tables_rank=0, tables_opt_in=True)

    assert eligible_among(index, datasets, source=source, excluded=excluded) == eligible


@pytest.mark.parametrize(
    ("source", "excluded"),
    [(None, frozenset()), (_TABLES, frozenset()), (None, frozenset({_PLAIN}))],
    ids=["unnamed", "named", "excluded"],
)
def test_the_eligible_datasets_of_a_query_are_its_publishers_filtered(
    source: str | None, excluded: frozenset[str]
):
    index = _shared(plain_rank=1, tables_rank=0, tables_opt_in=True)
    query = dataclasses.replace(_asks(set(), source), excluded=excluded)

    assert eligible_datasets(index, query) == eligible_among(
        index, (_TABLES, _PLAIN), source=source, excluded=excluded
    )


@pytest.mark.parametrize(
    ("datasets", "declares"),
    [((_PLAIN, _TABLES), True), ((_PLAIN,), True), ((_TABLES,), False), ((), False)],
    ids=["family", "crs", "dac1", "none"],
)
def test_the_roles_a_family_declares_narrow_to_the_datasets_given(
    datasets: tuple[str, ...], declares: bool
):
    index = _role_index(crs_rank=0, dac1_rank=1)
    family = index.family_by_dcid["crs"]

    assert family_declares_roles(index, family, {"destination"}, datasets=datasets) is declares


@pytest.mark.parametrize(
    ("role_keys", "among", "elected"),
    [
        (("destination",), (_TABLES, _PLAIN), _PLAIN),
        (("origin",), (_TABLES, _PLAIN), _TABLES),
        (("origin",), (_PLAIN,), _PLAIN),
        (("destination",), (_TABLES,), _TABLES),
    ],
    ids=["recipient", "donor", "donor-without-the-donor-table", "recipient-without-its-table"],
)
def test_an_election_among_given_datasets_is_the_election_of_an_index_holding_them_alone(
    role_keys: tuple[str, ...], among: tuple[str, ...], elected: str
):
    """`_TABLES` ranks first and holds a recipient on one row of two, so it declares a donor
    and no recipient. `_PLAIN` holds both roles on every row. Among the datasets a caller
    gives, the election serves the dataset and rows an index holding only those datasets
    serves, whatever the datasets left out would serve."""
    rows = (
        _role_row("crs", {"sector": "A"}, _DONOR_RECIPIENT, _PLAIN),
        _role_row("dac1", {"sector": "A"}, _DONOR, _TABLES),
        _role_row("dac1_flow", {"sector": "B"}, _DONOR_RECIPIENT, _TABLES),
    )
    datasets = (_dataset(_PLAIN, rank=1), _dataset(_TABLES, rank=0))
    index = _index(*rows, datasets=datasets)
    alone = _index(
        *(row for row in rows if set(row.provenances) & set(among)),
        datasets=tuple(dataset for dataset in datasets if dataset.provenance in among),
    )
    query = dataclasses.replace(_asks_roles(*role_keys), slots={"sector": ("unbound",)})

    family = family_among(index, query, among)
    reference = index_family(alone, query)

    assert family is not None
    assert reference is not None
    assert family.provenance == elected
    assert (family.provenance, _dcids(family.rows)) == (
        reference.provenance,
        _dcids(reference.rows),
    )


@pytest.mark.parametrize(
    ("sector", "among", "served"),
    [("Health", (_PLAIN,), (_PLAIN, {"plain_health"})), ("Education", (_TABLES,), (None, set()))],
    ids=["own-family-rows", "no-row-among-the-given"],
)
def test_an_election_among_given_datasets_reads_their_family_whatever_the_keys_select(
    sector: str, among: tuple[str, ...], served: tuple[str | None, set[str]]
):
    """A `sector` key alone selects `_TABLES`' family, which is what the unnamed query
    elects. Among `_PLAIN` alone the election reads `_PLAIN`'s family, as a query naming it
    does. Among `_TABLES` alone it serves no dataset for a sector only `_PLAIN` holds."""
    index = _split_index(sparse=True)
    query = FamilyQuery(_FIVE_TUPLE, {"sector": ("value", sector)}, None)
    assert eligible_datasets(index, query) == (_TABLES,)

    family = family_among(index, query, among)

    assert family is not None
    assert (family.provenance, _dcids(family.rows)) == served


def _pinning_index() -> CodeIndex:
    """`_PLAIN` declares one `tableLine` value as an invariant and holds it on every row of
    the five-tuple, `_TABLES` holds two."""
    return _index(
        _row("plain_a", {"sector": "A", _LINE: "L1"}, _PLAIN),
        _row("plain_b", {"sector": "B", _LINE: "L1"}, _PLAIN),
        _row("tables_a", {"sector": "A", _LINE: "L1"}, _TABLES),
        _row("tables_b", {"sector": "B", _LINE: "L2"}, _TABLES),
        datasets=(_dataset(_PLAIN, rank=0, invariants={_LINE: "L1"}), _dataset(_TABLES, rank=1)),
    )


_PIN_CASES = {
    "elected": (None, frozenset(), {_LINE: "L1"}),
    "named-pinning": (_PLAIN, frozenset(), {_LINE: "L1"}),
    "named-unpinned": (_TABLES, frozenset(), {}),
    "excluded-pinning": (None, frozenset({_PLAIN}), {}),
}


def _pin_query(source: str | None, excluded: frozenset[str]) -> FamilyQuery:
    return FamilyQuery(
        _FIVE_TUPLE,
        {"sector": ("unbound",), _LINE: ("unbound",)},
        source,
        excluded=excluded,
    )


@pytest.mark.parametrize(("source", "excluded", "pins"), _PIN_CASES.values(), ids=_PIN_CASES)
def test_the_pins_are_those_of_the_dataset_serving_the_family(
    source: str | None, excluded: frozenset[str], pins: dict[str, str]
):
    """A query that names no dataset is canonical whether or not it binds the pin, as is one
    that names a dataset declaring nothing. One that names the elected dataset and leaves
    its invariant unbound binds it; one that binds it already is canonical."""
    index = _pinning_index()
    query = _pin_query(source, excluded)
    bound = dataclasses.replace(query, slots={**query.slots, _LINE: ("value", "L1")})

    assert index.dataset_pins[_PLAIN, _FIVE_TUPLE] == {_LINE: "L1"}
    assert index.dataset_pins[_TABLES, _FIVE_TUPLE] == {}
    assert elected_pins(index, query) == pins
    assert invariant_bindings(index, query) == (pins if source == _PLAIN else {})
    assert invariant_bindings(index, bound) == {}


def test_one_index_memoises_the_pins_of_each_query_apart():
    """`elected_pins` memoises per index and query. Queries that differ only in the dataset
    they name or exclude read their own pins from one index, and a caller that changes the
    returned pins leaves the memo as it was."""
    index = _pinning_index()

    for source, excluded, pins in (*_PIN_CASES.values(), *_PIN_CASES.values()):
        returned = elected_pins(index, _pin_query(source, excluded))
        assert returned == pins
        returned["sector"] = "changed"


def _held_refs(options) -> int:
    """The references one slot-options entry holds: its rows, its bucket positions, and
    each elected family with its rows, plus one for the entry."""
    return (
        1
        + len(options.rows)
        + sum(len(found) for found in options.buckets.values())
        + sum(1 + len(family.rows) for family in options.families.values())
    )


def test_the_slot_options_memo_holds_at_most_its_reference_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    """Each menu binds `finance` to its own set of 40 refs and reads 800 rows over 20
    sectors, about 2,400 references in all. With a budget of 6,000, reading 30 menus keeps
    at most two at any time, counted from what the memo holds. A menu read before each of
    the others stays, and an evicted menu elects the same families again."""
    budget = 6_000
    monkeypatch.setattr(members_module, "_SLOT_OPTIONS_REFS", budget)
    sectors = [f"S{n:02d}" for n in range(20)]
    finances = [f"F{n:02d}" for n in range(50)]
    index = _index(
        *(
            _row(f"{sector}_{finance}", {"sector": sector, "finance": finance}, _PLAIN)
            for sector in sectors
            for finance in finances
        ),
        datasets=(_dataset(_PLAIN),),
    )

    def menu(n: int) -> FamilyQuery:
        slots: dict[str, SlotBinding] = {
            "sector": ("unbound",),
            "finance": ("set", tuple((finances * 2)[n : n + 40])),
        }
        return FamilyQuery(_FIVE_TUPLE, slots, None)

    hot = menu(0)
    option_families(index, hot, "sector", sectors)
    memo = members_module._slot_options_cache[id(index)]
    (hot_options,) = memo.entries.values()
    first = option_families(index, menu(1), "sector", sectors)

    for n in range(2, 30):
        option_families(index, hot, "sector", sectors)
        option_families(index, menu(n), "sector", sectors)
        held = sum(_held_refs(options) for options in memo.entries.values())
        assert held == memo.weight <= budget
        assert any(options is hot_options for options in memo.entries.values())

    assert len(memo.entries) == 2
    again = option_families(index, menu(1), "sector", sectors)
    assert again == first
    assert again == {
        sector: family_for_five_tuple(index, bind_slot_value(index, menu(1), "sector", sector))
        for sector in sectors
    }
    assert all(len(family.rows) == 40 for family in again.values())


def test_binding_the_elected_pins_keeps_the_election():
    index = _pinning_index()
    open_query = FamilyQuery(_FIVE_TUPLE, {"sector": ("unbound",), _LINE: ("unbound",)}, _PLAIN)
    pinned_query = FamilyQuery(
        _FIVE_TUPLE, {"sector": ("unbound",), _LINE: ("value", "L1")}, _PLAIN
    )

    before = family_for_five_tuple(index, open_query)
    after = family_for_five_tuple(index, pinned_query)

    assert (after.provenance, _dcids(after.rows)) == (before.provenance, _dcids(before.rows))
    assert invariant_bindings(index, open_query) == {_LINE: "L1"}
    assert invariant_bindings(index, pinned_query) == {}


def _competing_index() -> CodeIndex:
    """`_TABLES` (rank 1) declares `alpha` A1 and `beta` B1 as invariants and holds them on
    every row; `_PLAIN` (rank 0) publishes a row of sector A beside it with `beta` B2."""
    return _index(
        _row("tables_a", {"alpha": "A1", "beta": "B1", "sector": "A"}, _TABLES),
        _row("tables_b", {"alpha": "A1", "beta": "B1", "sector": "B"}, _TABLES),
        _row("plain_a", {"alpha": "A1", "beta": "B2", "sector": "A"}, _PLAIN),
        datasets=(
            _dataset(_PLAIN, rank=0),
            _dataset(_TABLES, rank=1, invariants={"alpha": "A1", "beta": "B1"}),
        ),
        declarations=tuple(_not_declared(prop) for prop in ("alpha", "beta", "sector")),
    )


_OPEN: SlotBinding = ("unbound",)


@pytest.mark.parametrize(
    ("source", "sector", "alpha", "beta", "bound"),
    [
        (None, "A", ("value", "A1"), ("value", "B1"), {}),
        (None, "A", _OPEN, ("value", "B1"), {}),
        (_TABLES, "A", ("value", "A1"), ("value", "B1"), {}),
        (_TABLES, "A", _OPEN, ("value", "B1"), {"alpha": "A1"}),
        (_TABLES, "A", _OPEN, _OPEN, {"alpha": "A1", "beta": "B1"}),
        (_TABLES, "A", ("value", "A2"), _OPEN, {"beta": "B1"}),
        (_TABLES, "Z", _OPEN, _OPEN, {"alpha": "A1", "beta": "B1"}),
    ],
    ids=[
        "unnamed-bound",
        "unnamed-open",
        "named-bound",
        "named-alpha-open",
        "named-open",
        "named-other-value",
        "named-empty-family-open",
    ],
)
def test_the_canonical_query_binds_invariants_only_when_it_names_the_dataset(
    source: str | None,
    sector: str,
    alpha: SlotBinding,
    beta: SlotBinding,
    bound: dict[str, str],
):
    """`_TABLES` declares `alpha` A1 and `beta` B1. Named, the canonical query binds each it
    leaves unbound and keeps one bound to another value as posted; unnamed, it keeps the
    bindings it was posted with, since a later export could let `_PLAIN` serve. The rule
    reads the declaration alone, so a named question on sector Z, which no row holds today,
    binds them too. Each canonical query elects the query's family and is its own canonical
    form."""
    index = _competing_index()
    # `beta` before `alpha`, so the slots' order does not decide the result's order.
    query = FamilyQuery(
        _FIVE_TUPLE, {"beta": beta, "alpha": alpha, "sector": ("value", sector)}, source
    )
    canonical = dataclasses.replace(
        query,
        slots={**query.slots, **{prop: ("value", value) for prop, value in bound.items()}},
    )

    assert invariant_bindings(index, query) == bound
    assert list(invariant_bindings(index, query)) == sorted(bound)
    assert family_for_five_tuple(index, canonical) == family_for_five_tuple(index, query)
    assert invariant_bindings(index, canonical) == {}


@pytest.mark.parametrize("line", [("value", "L1"), ("unbound",)], ids=["bound", "open"])
def test_slot_keys_that_leave_membership_to_the_graph_imply_no_invariant(line: SlotBinding):
    """Every `_PLAIN` row carries `sector`, so a query naming `_PLAIN` that declares
    `tableLine` alone leaves membership to the graph walk, whose rows the loader never
    checked. Canonicalisation and every Search reader agree that it implies nothing."""
    index = _sector_index("A")
    query = FamilyQuery(_FIVE_TUPLE, {_LINE: line}, _PLAIN)

    assert index_covers_slot_keys(index, _FIVE_TUPLE, query.slots.keys()) is False
    assert elected_invariants(index, query) == {}
    assert invariant_bindings(index, query) == {}
    named = dataclasses.replace(query, slots={**query.slots, "sector": ("unbound",)})
    assert elected_invariants(index, named) == {_LINE: "L1"}


def test_an_invariant_keeps_its_rows_when_a_later_export_adds_a_lower_ranked_publisher():
    """In the first export `_TABLES` alone publishes the five-tuple and declares `P` V. The
    second adds `_PLAIN`, ranked first, with a row of `P` W. An identity posted with `P` V
    and no dataset keeps `P` bound, so it reads `_TABLES`' rows on both; one that names
    `_TABLES` binds `P` V whether posted with it bound or unbound, and reads them too. One
    posted with `P` unbound and no dataset stays unbound, and the rank serves `_PLAIN` once
    it publishes."""
    tables_rows = (
        _row("d_x", {"P": "V", "s": "X"}, _TABLES),
        _row("d_y", {"P": "V", "s": "Y"}, _TABLES),
    )
    tables = _dataset(_TABLES, rank=1, invariants={"P": "V"})
    declarations = (_not_declared("P"), _not_declared("s"))
    versions = (
        _index(*tables_rows, datasets=(tables,), declarations=declarations),
        _index(
            *tables_rows,
            _row("f_x", {"P": "W", "s": "X"}, _PLAIN),
            datasets=(_dataset(_PLAIN, rank=0), tables),
            declarations=declarations,
        ),
    )
    shape = {
        "kind": "statvar_schema",
        "population_type": _FIVE_TUPLE.population_type,
        "measured_property": _FIVE_TUPLE.measured_property,
        "stat_type": _FIVE_TUPLE.stat_type,
    }
    slots = [
        {"property": "P", "binding": {"state": "value", "ref": "V"}},
        {"property": "s", "binding": {"state": "unbound"}},
    ]
    unnamed = {"shape": shape, "slots": slots}
    named = {**unnamed, "facets": {"provenance": {"state": "value", "ref": _TABLES}}}
    open_unnamed = {
        "shape": shape,
        "slots": [{**slots[0], "binding": {"state": "unbound"}}, slots[1]],
    }
    open_named = {**open_unnamed, "facets": named["facets"]}
    bound_v = {"state": "value", "ref": "V"}
    tables_only = ({"d_x", "d_y"}, {"d_x", "d_y"})
    cases = (
        (unnamed, bound_v, tables_only),
        (named, bound_v, tables_only),
        (open_named, bound_v, tables_only),
        (open_unnamed, {"state": "unbound"}, ({"d_x", "d_y"}, {"f_x"})),
    )

    for posted, p_binding, rows in cases:
        for index, expected in zip(versions, rows, strict=True):
            canonical = canonical_identity(index, posted)
            family = family_for_five_tuple(index, identity_family_query(index, canonical))

            assert (canonical is posted) is (posted is not open_named)
            assert canonical["slots"][0]["binding"] == p_binding
            assert _dcids(family.rows) == expected


def _sector_index(*sectors: str) -> CodeIndex:
    """`_PLAIN` declares `tableLine` L1 as an invariant, and holds one row per sector of
    `sectors` without declaring the sector."""
    return _index(
        *(_row(f"plain_{sector}", {"sector": sector, _LINE: "L1"}, _PLAIN) for sector in sectors),
        datasets=(_dataset(_PLAIN, invariants={_LINE: "L1"}),),
    )


def test_an_explicit_binding_of_an_undeclared_constant_keeps_its_rows_on_a_later_export():
    """Every `_PLAIN` row holds sector A in the first export, but the sector is no invariant,
    so the canonical identity keeps its binding and still selects only sector A once the
    second export adds sector B. The declared invariant of the named dataset, posted
    unbound, is bound on both."""
    identity = {
        "shape": {
            "kind": "statvar_schema",
            "population_type": _FIVE_TUPLE.population_type,
            "measured_property": _FIVE_TUPLE.measured_property,
            "stat_type": _FIVE_TUPLE.stat_type,
        },
        "slots": [
            {"property": "sector", "binding": {"state": "value", "ref": "A"}},
            {"property": _LINE, "binding": {"state": "unbound"}},
        ],
        "facets": {"provenance": {"state": "value", "ref": _PLAIN}},
    }

    for index in (_sector_index("A"), _sector_index("A", "B")):
        canonical = canonical_identity(index, identity)
        family = family_for_five_tuple(index, identity_family_query(index, canonical))

        assert canonical["slots"] == [
            {"property": "sector", "binding": {"state": "value", "ref": "A"}},
            {"property": _LINE, "binding": {"state": "value", "ref": "L1"}},
        ]
        assert _dcids(family.rows) == {"plain_A"}


def test_canonicalising_runs_no_election(monkeypatch: pytest.MonkeyPatch):
    """`_TABLES` declares twelve invariants, and leaving any one unbound would let a `_PLAIN`
    row in and the rank serve it were the query not to name `_TABLES`. Canonicalising the
    named query that leaves all twelve unbound binds them from the declaration alone and
    elects no family."""
    props = [f"p{i:02d}" for i in range(12)]
    index = _index(
        _row("tables", dict.fromkeys(props, "V"), _TABLES),
        *(
            _row(f"plain_{prop}", {**dict.fromkeys(props, "V"), prop: "W"}, _PLAIN)
            for prop in props
        ),
        datasets=(
            _dataset(_PLAIN, rank=0),
            _dataset(_TABLES, rank=1, invariants=dict.fromkeys(props, "V")),
        ),
        declarations=tuple(_not_declared(prop) for prop in props),
    )
    open_slots: dict[str, SlotBinding] = dict.fromkeys(props, ("unbound",))
    query = FamilyQuery(_FIVE_TUPLE, open_slots, _TABLES)
    elections = 0
    elect = members_module.family_for_five_tuple

    def counted(index: CodeIndex, query: FamilyQuery):
        nonlocal elections
        elections += 1
        return elect(index, query)

    monkeypatch.setattr(members_module, "family_for_five_tuple", counted)

    assert invariant_bindings(index, query) == dict.fromkeys(props, "V")
    assert elections == 0


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


_NET_DISBURSEMENT: dict[str, SlotBinding] = {
    "financialFlowType": ("value", "Disbursement"),
    "nettingBasis": ("value", "Net"),
}
_CRS_OPEN_SLOTS = ("crsCooperationModality", "crsPurpose", "crsTypeOfFinance", "crsTypeOfFlow")


@pytest.mark.parametrize(
    ("open_slots", "as_role_keys", "dataset", "count"),
    [
        (("dac1AidType",), ("origin",), "provenance/OECDDAC1", 125),
        (
            _CRS_OPEN_SLOTS,
            ("origin", "destination"),
            "provenance/OECDCreditorReportingSystem",
            6461,
        ),
    ],
    ids=["dac1-provider-totals", "crs-flows"],
)
def test_an_open_recipient_reads_the_packaged_rows_its_dataset_declares_it_on(
    open_slots: tuple[str, ...], as_role_keys: tuple[str, ...], dataset: str, count: int
):
    """A donor's net disbursements to every recipient. DAC1 holds a recipient on two flows
    to the West Bank and Gaza Strip only, at constraint values no provider total holds,
    so its rows are its 125 provider totals, those of the donor alone, without the two
    flows. Every CRS row holds one, so its rows are those of a named recipient."""
    index = load_index(default_index_path())
    slots = {**dict.fromkeys(open_slots, ("unbound",)), **_NET_DISBURSEMENT}
    query = FamilyQuery(_EDU_AID_FIVE_TUPLE, slots, None, role_keys=frozenset({"origin"}))

    family = index_family(
        index, dataclasses.replace(query, open_role_keys=frozenset({"destination"}))
    )
    reference = index_family(index, dataclasses.replace(query, role_keys=frozenset(as_role_keys)))

    assert family is not None
    assert (family.provenance, len(family.rows)) == (dataset, count)
    assert family == reference


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


@pytest.mark.parametrize(
    ("slots", "source", "bound"),
    [
        (
            lambda index: _education_aid_slots(index, aid_type=("unbound",)),
            IMPUTED_PROVENANCE,
            {AID_TYPE: IMPUTED_MULTILATERAL},
        ),
        (lambda index: _education_aid_slots(index, aid_type=("unbound",)), None, {}),
        (lambda _index: {AID_TYPE: ("unbound",)}, IMPUTED_PROVENANCE, {}),
    ],
    ids=["covered-named", "covered-unnamed", "not-covered"],
)
def test_the_imputed_pins_are_bound_only_where_the_index_decides(slots, source, bound):
    """The imputed table declares the flow type and the aid type as invariants, so the
    canonical identity of a query that names it binds the aid type it leaves unbound, and
    one that names no dataset keeps it unbound. Slot keys no index row is covered by elect
    no family, so nothing is bound and the graph walk reads the slot."""
    index = load_index(default_index_path())
    query = FamilyQuery(_IMPUTED_FIVE_TUPLE, slots(index), source)

    assert invariant_bindings(index, query) == bound


def test_an_imputed_identity_posted_with_or_without_its_pins_is_one_canonical_identity():
    index = load_index(default_index_path())
    posted = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    unbound = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    for slot in unbound["slots"]:
        if slot["property"] in ("crsTypeOfFlow", AID_TYPE):
            slot["binding"] = {"state": "unbound"}

    canonical = canonical_identity(index, unbound)

    assert canonical == posted
    assert canonical_identity(index, posted) is posted


@pytest.mark.parametrize(
    "aid_type",
    [
        {"state": "value", "ref": IMPUTED_MULTILATERAL},
        {"state": "unbound"},
        {"state": "set", "refs": [IMPUTED_MULTILATERAL]},
        {"state": "exact", "ref": IMPUTED_MULTILATERAL},
    ],
    ids=["value", "unbound", "set-of-one", "exact"],
)
def test_every_form_of_a_named_identitys_invariant_reaches_one_spec_id(aid_type: dict):
    """The imputed table declares its aid type. An identity naming the table that binds it
    by value, leaves it unbound, binds a set of the one code or binds it `exact` reaches the
    spec_id of the value binding Search mints."""
    index = load_index(default_index_path())
    expected = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    posted = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    for slot in posted["slots"]:
        if slot["property"] == AID_TYPE:
            slot["binding"] = aid_type

    canonical = canonical_identity(index, posted)

    assert recompute_identity(canonical).spec_id == recompute_identity(expected).spec_id


def test_canonicalising_identities_on_five_tuples_the_index_does_not_carry_stores_nothing():
    """Every door canonicalises a posted identity, whatever five-tuple it names. The
    role-class memo holds the five-tuples the index carries, so posting new ones leaves its
    size as it was."""
    index = load_index(default_index_path())
    canonical_identity(index, imputed_education_aid_identity(source=IMPUTED_PROVENANCE))
    size = len(members_module._five_tuple_facts_cache.get(id(index), {}))

    for i in range(50):
        posted = education_aid_identity(source=CRS_PROVENANCE)
        posted["shape"]["measured_property"] = f"postedAmount{i}"
        assert canonical_identity(index, posted) is posted

    assert len(members_module._five_tuple_facts_cache.get(id(index), {})) == size


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
