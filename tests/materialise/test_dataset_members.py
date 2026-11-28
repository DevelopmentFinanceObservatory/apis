"""A declared dataset of member provenances elects, cuts and serves as one table under its id.

The synthetic `dataset/Ledger` splits one table's series between two provenances, as a
register filed by publisher does, and `dataset/Solo` is a Dataset of one member provenance
(`tests.materialise._ledger`). A facet of a member provenance reads as its dataset wherever
it is read, whatever dataset an election holds, so facet selection and group agreement key on
the dataset, rows carry its id, and no member provenance reaches a frame. A plain table, its
own sole member, is the control and keeps its bytes.
"""

from __future__ import annotations

import dataclasses
import io
import json
from dataclasses import replace

import pyarrow as pa
import pytest

from dc_sdmx import observation_schema
from graph_release.context import bound_release
from materialise import EnrichmentLookup, materialise_group
from materialise.columnar import dataset_cells, held_back_mask, held_back_through
from materialise.facets import (
    ElectedDataset,
    SeriesCounts,
    joint_dispatch_facet_ids,
    rank_candidates,
    read_as_datasets,
)
from materialise.formats import write_frame1
from materialise.generation import _select_member_facets, reads_absent_as_zero
from materialise.graph import Facet
from materialise.group_facets import select_group_facets
from materialise.held_runs import read_runs
from materialise.observation_rows import UN_OBSERVATION_ROLES, _variable_batch
from origin_catalogue import bound_catalogue
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index
from tests.origin_catalogue.fake_catalogue import dev_provenance_nodes, documenting

from ._group_support import (
    _BARE_A,
    _BARE_B,
    _CSV_HEADER,
    _ENTITY,
    _STAT_VAR_A,
    _STAT_VAR_B,
    _STAT_VAR_C,
    _csv_row,
    _request,
    _sdmx_client_for,
)
from ._ledger import LEDGER, MEMBERS, NORTH, PLAIN, SOLO, SOLO_MEMBER, SOUTH, bound_ledger_index
from .conftest import FakeGraphReader

_LEDGER_DATASET = ElectedDataset(
    provenance=LEDGER, provenances=(NORTH, SOUTH), recorded_provenances=(NORTH, SOUTH)
)
_SOLO_DATASET = ElectedDataset(
    provenance=SOLO, provenances=(SOLO_MEMBER,), recorded_provenances=(SOLO_MEMBER,)
)
_PLAIN_DATASET = ElectedDataset(
    provenance=PLAIN, provenances=(PLAIN,), recorded_provenances=(PLAIN,)
)

_ROWS_SCHEMA = pa.schema(
    [("provenance", pa.string()), ("measurementMethod", pa.string()), ("TIME_PERIOD", pa.string())]
)


@pytest.fixture(autouse=True)
def _ledger_index():
    with bound_ledger_index():
        yield


def _facet_id(provenance: str) -> str:
    """A facet id naming no provenance, as the graph's opaque ids name none."""
    return f"facet-{provenance.rsplit('/', 1)[-1]}"


def _facet(
    provenance: str, *, obs_count: int = 1, unit: str = "UNIT", method: str = "METH"
) -> Facet:
    suffix = "" if (unit, method) == ("UNIT", "METH") else f"-{unit}-{method}"
    return Facet(
        facet_id=f"{_facet_id(provenance)}{suffix}",
        provenance_id=provenance,
        measurement_method=method,
        unit=unit,
        obs_count=obs_count,
        earliest_date="2015",
        latest_date="2015",
        observation_period="P1Y",
    )


def _member(stat_var: str, dataset: ElectedDataset | None = _LEDGER_DATASET):
    request = _request(stat_var=stat_var, facet_bindings={}, index_sourced=True)
    return dataclasses.replace(request, dataset=dataset)


def _marked(north: str | None, south: str | None) -> dict[str, dict[str, tuple[str, ...]]]:
    """The Ledger's member provenances as the catalogue documents them, each marking the
    partial period given, or none."""
    crs = dev_provenance_nodes()["provenance/OECDCreditorReportingSystem"]
    unmarked = {prop: values for prop, values in crs.items() if prop != "partialObservationDate"}
    return {
        provenance: unmarked if mark is None else {**unmarked, "partialObservationDate": (mark,)}
        for provenance, mark in ((NORTH, north), (SOUTH, south))
    }


def test_the_dataset_of_provenances_is_built_from_the_index_member_map():
    members = {LEDGER: (NORTH, SOUTH)}

    dataset = ElectedDataset.of(
        LEDGER,
        recorded=(LEDGER, PLAIN),
        provenances_of=lambda dataset: members.get(dataset, (dataset,)),
    )

    assert dataset == ElectedDataset(
        provenance=LEDGER,
        provenances=(NORTH, SOUTH),
        recorded_provenances=(NORTH, SOUTH, PLAIN),
    )


@pytest.mark.parametrize(
    ("provenance", "dataset_id"),
    [
        pytest.param(NORTH, LEDGER, id="two_members"),
        pytest.param(SOLO_MEMBER, SOLO, id="one_member"),
        pytest.param(PLAIN, None, id="plain_table"),
    ],
)
def test_a_facet_reads_as_the_dataset_its_provenance_is_a_member_of(provenance, dataset_id):
    """The stamp reads the code index's member map alone: a Dataset of one member carries its
    id as a Dataset of two does, and a plain table's facet is returned as it is."""
    facet = _facet(provenance)

    (read,) = read_as_datasets([facet])

    assert read.dataset_id == dataset_id
    assert read.provenance_id == provenance
    if dataset_id is None:
        assert read is facet


@pytest.mark.parametrize(
    ("dataset", "provenances", "ref"),
    [
        pytest.param(_LEDGER_DATASET, (NORTH, SOUTH), LEDGER, id="two_members"),
        pytest.param(_SOLO_DATASET, (SOLO_MEMBER,), SOLO, id="one_member"),
        pytest.param(None, (NORTH, SOUTH), LEDGER, id="no_elected_dataset"),
        pytest.param(_PLAIN_DATASET, (PLAIN,), PLAIN, id="plain_table"),
    ],
)
def test_a_member_selects_its_dataset_and_dispatches_every_member_provenance(
    dataset, provenances, ref
):
    """The members hold different series of one variable, so the election reads them as one
    selection under the dataset's id with no disagreement to report, and the dispatch over the
    facets as read serves each. A StatVar minted with no elected dataset reads the same."""
    request = _member(_STAT_VAR_A, dataset)
    candidates = [_facet(provenance, obs_count=3 - i) for i, provenance in enumerate(provenances)]

    election = _select_member_facets(
        request,
        candidates,
        graph=FakeGraphReader(),
        breadth={_facet_id(p): 3 - i for i, p in enumerate(provenances)},
    )

    selection = election.selections["provenance"]
    assert selection.ref == ref
    assert selection.selection_reason in ("sole_candidate", "most_observations")
    assert selection.ranked_values == ()
    assert joint_dispatch_facet_ids(read_as_datasets(candidates), election, member=0) == tuple(
        _facet_id(p) for p in provenances
    )


def test_members_held_under_different_provenances_of_one_dataset_form_one_group():
    """Member A holds its series under one provenance and member B under the other. Keyed on
    provenance they would form two groups and leave one member unserved."""
    requests = [_member(_STAT_VAR_A), _member(_STAT_VAR_B)]
    candidates = [[_facet(NORTH, obs_count=2)], [_facet(SOUTH)]]

    group = select_group_facets(requests, candidates, graph=FakeGraphReader())

    assert group.election.selections["provenance"].ref == LEDGER
    assert group.unserved == ()
    assert group.dispatch_ids == ((_facet_id(NORTH),), (_facet_id(SOUTH),))


def test_the_unit_both_member_provenances_hold_serves_the_group():
    """Members A and B hold the dataset's series in one unit under different provenances,
    and member C a larger facet in another unit. The key every member but C holds is the
    dataset's in the first unit, so it serves A and B and leaves C out."""
    requests = [_member(_STAT_VAR_A), _member(_STAT_VAR_B), _member(_STAT_VAR_C)]
    candidates = [
        [_facet(NORTH)],
        [_facet(SOUTH)],
        [_facet(NORTH, obs_count=5, unit="OTHER")],
    ]

    group = select_group_facets(requests, candidates, graph=FakeGraphReader())

    assert group.election.selections["unit"].ref == "UNIT"
    assert group.unserved == (2,)


def test_under_national_currency_a_method_counts_the_series_of_both_member_provenances():
    """Each place's own currency leaves the unit out of the selection, so a selection counts
    the series of all its facets (`rank_candidates`). The method both provenances hold serves
    three plus two series, more than the four the other method holds under one."""
    candidates = read_as_datasets([_facet(NORTH), _facet(SOUTH), _facet(NORTH, method="OTHER")])
    counts = {_facet_id(NORTH): 3, _facet_id(SOUTH): 2, f"{_facet_id(NORTH)}-UNIT-OTHER": 4}

    ranked = rank_candidates(
        candidates,
        window=None,
        breadth=SeriesCounts(counts=counts, union_of=lambda ids: sum(counts[i] for i in ids)),
        national_currency=True,
    )

    assert ranked[0].measurement_method == "METH"


def test_a_group_of_one_dataset_reads_the_dataset_declaration_on_absent_members():
    requests = [_member(_STAT_VAR_A), _member(_STAT_VAR_B)]
    candidates = [[_facet(NORTH, obs_count=2)], [_facet(SOUTH)]]
    selections = select_group_facets(
        requests, candidates, graph=FakeGraphReader()
    ).election.selections
    read: list[str] = []

    def absent_reads_zero(dataset: str) -> bool:
        read.append(dataset)
        return dataset == LEDGER

    assert reads_absent_as_zero(selections, absent_reads_zero)
    assert read == [LEDGER]


def _mint(provenances: tuple[str, str], dataset: ElectedDataset | None):
    """A group mint of members A and B, A's series under the first of `provenances` and B's
    under the second."""
    pairs = ((_BARE_A, provenances[0]), (_BARE_B, provenances[1]))
    coverage = build_coverage_index(
        [
            (bare, _facet_id(provenance), "country/DONOR", _ENTITY, [2015])
            for bare, provenance in pairs
        ],
        role_names={
            bare: RoleNames(donor="donor", recipient="recipient") for bare in (_BARE_A, _BARE_B)
        },
        facet_metadata={
            _facet_id(provenance): FacetMetadata(
                provenance_id=provenance,
                measurement_method="METH",
                unit="UNIT",
                observation_period="P1Y",
            )
            for provenance in set(provenances)
        },
    )
    sdmx, _requests = _sdmx_client_for(
        {
            bare: _CSV_HEADER
            + _csv_row(
                stat_var=bare,
                year="2015",
                value="10",
                unit="UNIT",
                method="METH",
                provenance=provenance,
                facet_id=_facet_id(provenance),
            )
            for bare, provenance in pairs
        }
    )
    graph = FakeGraphReader()
    return materialise_group(
        [_member(_STAT_VAR_A, dataset), _member(_STAT_VAR_B, dataset)],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        parties={},
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )


# The content version this group mint of the plain table had under frame grammar 1.2.0,
# before the `reporting_organisation` column existed.
_PLAIN_CONTENT_VERSION = "cv_9810ab0f7801c7878de0ba0e2bd859dc"


def _frame1(result) -> bytes:
    sink = io.BytesIO()
    write_frame1(result.frame_chunks(), sink=sink)
    return sink.getvalue()


def _provenance_cells(frame1: bytes) -> set[str]:
    document = json.loads(frame1)
    column = document["columns"].index("provenance")
    return {row[column] for row in document["rows"]}


def _reporters(frame1: bytes) -> dict[str, str | None]:
    """Each row's `reporting_organisation` cell by its `stat_var`."""
    document = json.loads(frame1)
    columns = document["columns"]
    stat_var, reporter = columns.index("stat_var"), columns.index("reporting_organisation")
    return {row[stat_var]: row[reporter] for row in document["rows"]}


@pytest.mark.parametrize(
    ("provenances", "dataset", "ref"),
    [
        pytest.param((NORTH, SOUTH), _LEDGER_DATASET, LEDGER, id="two_members"),
        pytest.param((NORTH, SOUTH), None, LEDGER, id="no_elected_dataset"),
        pytest.param((SOLO_MEMBER, SOLO_MEMBER), _SOLO_DATASET, SOLO, id="one_member"),
    ],
)
def test_a_mint_of_a_dataset_serves_every_member_row_under_the_dataset_id(
    provenances, dataset, ref
):
    """Both members' rows are served, each carrying the dataset's id and naming the member
    that reported it, and no member provenance appears elsewhere in the frame, nor in its
    trace, its dependencies or its held runs."""
    result = _mint(provenances, dataset)

    counts = {entry["ref"]: entry["observation_count"] for entry in result.trace.resolved_stat_vars}
    assert counts == {_STAT_VAR_A: 1, _STAT_VAR_B: 1}
    assert not [w for w in result.warnings if w["code"] == "coverage_below_scope"]
    (provenance,) = [entry for entry in result.trace.facets if entry["axis"] == "provenance"]
    assert provenance["ref"] == ref
    frame1 = _frame1(result)
    assert _provenance_cells(frame1) == {ref}
    assert _reporters(frame1) == {_STAT_VAR_A: provenances[0], _STAT_VAR_B: provenances[1]}
    document = json.loads(frame1)
    reporter = document["columns"].index("reporting_organisation")
    rows = [row[:reporter] + row[reporter + 1 :] for row in document["rows"]]
    served = b"\n".join(
        [
            json.dumps(rows).encode(),
            json.dumps(result.trace.to_dict(), default=str).encode(),
            json.dumps(result.dependencies.as_json(), default=str).encode(),
            repr(result.held_runs).encode(),
        ]
    )
    assert [member for member in MEMBERS if member.encode() in served] == []


def test_a_mint_of_a_plain_table_keeps_its_bytes_under_an_index_declaring_datasets(
    _fixture_index_release,
):
    """A plain table's provenance is its own dataset, so it mints the frame and content
    version it mints under the session's index, which declares no Ledger. Its rows name no
    reporting organisation, so the frame has no such column and keeps the bytes and content
    version it had before frame grammar 1.3.0 added one."""
    under_ledger = _mint((PLAIN, PLAIN), _PLAIN_DATASET)
    with bound_release(_fixture_index_release):
        under_fixture = _mint((PLAIN, PLAIN), _PLAIN_DATASET)

    assert _frame1(under_ledger) == _frame1(under_fixture)
    assert under_ledger.content_version == under_fixture.content_version
    assert _provenance_cells(_frame1(under_ledger)) == {PLAIN}
    assert "reporting_organisation" not in json.loads(_frame1(under_ledger))["columns"]
    assert under_ledger.content_version == _PLAIN_CONTENT_VERSION


def test_the_read_writes_each_member_row_under_its_dataset_and_keeps_a_plain_batch():
    batch = pa.RecordBatch.from_pydict(
        {
            "provenance": [NORTH, SOUTH, SOLO_MEMBER, PLAIN, None],
            "measurementMethod": [None] * 5,
            "TIME_PERIOD": ["2015"] * 5,
        },
        schema=_ROWS_SCHEMA,
    )
    plain = pa.RecordBatch.from_pydict(
        {"provenance": [PLAIN], "measurementMethod": [None], "TIME_PERIOD": ["2015"]},
        schema=_ROWS_SCHEMA,
    )

    read = dataset_cells(batch)
    assert read["provenance"].to_pylist() == [LEDGER, LEDGER, SOLO, PLAIN, None]
    assert read["reporting_organisation"].to_pylist() == [NORTH, SOUTH, SOLO_MEMBER, None, None]
    assert dataset_cells(plain) is plain


def test_a_un_read_writes_each_member_row_under_its_dataset_and_names_the_member():
    """The UN deployment's rows, which the read lays out itself, carry the dataset's id and
    name their member as `dataset_cells` writes an SDMX row, and a plain table's name none."""
    series = [
        ("country/KEN", replace(_facet(provenance), dates=("2015",), values=(1.0,)))
        for provenance in (NORTH, PLAIN)
    ]

    batch = _variable_batch("undata/Stat_A", series, latest=False)

    assert batch.schema == observation_schema(UN_OBSERVATION_ROLES)
    assert batch["provenance"].to_pylist() == [LEDGER, PLAIN]
    assert batch["reporting_organisation"].to_pylist() == [NORTH, None]


@pytest.mark.parametrize(
    ("north", "south", "held", "through"),
    [
        pytest.param("2026", "2025", ["2025", "2026"], 2024, id="members-differ"),
        pytest.param("2026", None, ["2026"], 2025, id="one-member-marks"),
        pytest.param(None, None, [], None, id="none-marks"),
    ],
)
def test_a_dataset_holds_back_every_period_any_member_marks_partial(north, south, held, through):
    """The rows carry the dataset's id, and the cut judges the id by the earliest mark of its
    member provenances, so each member's rows are held back from the first period either
    marks."""
    periods = ["2024", "2024", "2025", "2025", "2026", "2026"]
    batch = pa.RecordBatch.from_pydict(
        {
            "provenance": [NORTH, SOUTH] * 3,
            "measurementMethod": [None] * 6,
            "TIME_PERIOD": periods,
        },
        schema=_ROWS_SCHEMA,
    )
    rows = pa.Table.from_batches([dataset_cells(batch)])

    with bound_catalogue(documenting(_marked(north, south))):
        mask = held_back_mask(rows, include=frozenset()).to_pylist()
        assert held_back_through(rows) == through

    assert sorted({period for period, cut in zip(periods, mask, strict=True) if cut}) == held


def test_a_partial_run_of_a_dataset_is_held_under_its_id_from_its_earliest_mark():
    north = dataclasses.replace(_facet(NORTH), latest_date="2026")

    with bound_catalogue(documenting(_marked("2026", "2025"))):
        (run,) = read_runs([([north], {north.facet_id or ""})])

    assert (run.kind, run.source, run.start_year, run.end_year) == ("partial", LEDGER, 2025, 2026)
