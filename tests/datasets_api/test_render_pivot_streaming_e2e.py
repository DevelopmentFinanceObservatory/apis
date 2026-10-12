"""Tests CSV byte parity across chunk splits for bare pivots, groupby-then-pivot, and a
non-outermost pivot index.

The groupby collapse streams first. Pivots handle rows whose index values recur in
non-adjacent chunks.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import pytest

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.types import StatVarRow
from datasets_api.mint_models import GroupbyOptions, MintOptions, PivotOptions
from datasets_api.render import render_and_store
from materialise.enrichment import EnrichmentLookup
from materialise.pipeline import ConcatenatedFrameChunks
from materialise.profile import OutputProfile, compute_output_profile
from materialise.reshape import PivotNotUniqueError, PivotOverCellCapError, Table

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._render_inputs import PREPARED_AT, SITE

_GRAPH = FakeMintGraphReader()
_CONTENT_VERSION = "sha256:" + "0" * 64


def _render(
    chunks: Iterable[Table],
    *,
    options: MintOptions,
    code_index: CodeIndex | None = None,
    output_profile: OutputProfile | None = None,
    pivot_max_output_cells: int = 5_000_000,
):
    artifact_store = FakeArtifactStore()
    rendered = render_and_store(
        frame_chunks=iter(chunks),
        options=options,
        spec={"spec_id": "pivot-streaming-e2e"},
        provenance_refs=["p1"],
        citations={},
        reproducing_call=None,
        content_version=_CONTENT_VERSION,
        artifact_store=artifact_store,
        entity_columns=("recipient",),
        enrichment=EnrichmentLookup(_GRAPH, type_kinds={}, parties={}),
        code_index=code_index,
        resolved_stat_var_refs=(),
        pivot_max_output_cells=pivot_max_output_cells,
        output_profile=output_profile,
        prepared_at=PREPARED_AT,
        site=SITE,
    )
    return rendered, artifact_store.fetch(rendered.artifact.url)


# --- Fixture 1: a bare pivot (no groupby at all). ---------------------------

_BARE_COLUMNS = ("stat_var", "recipient", "time_period", "value")
_BARE_ROWS = (
    ("dcid:MalariaGrantDisbursements", "country/KEN", "2020", 10.0),
    ("dcid:MalariaGrantDisbursements", "country/TGO", "2020", 20.0),
    ("dcid:MalariaGrantDisbursements", "country/KEN", "2021", 15.0),
)


def test_a_bare_pivot_matches_across_a_one_chunk_and_a_three_chunk_split():
    one_chunk = (Table(columns=_BARE_COLUMNS, rows=_BARE_ROWS),)
    three_chunks = tuple(Table(columns=_BARE_COLUMNS, rows=(row,)) for row in _BARE_ROWS)
    options = MintOptions(
        format="csv", bundle=False, pivot=PivotOptions(index="recipient", columns="time_period")
    )

    one_rendered, one_bytes = _render(one_chunk, options=options)
    three_rendered, three_bytes = _render(three_chunks, options=options)

    assert one_rendered.artifact.artifact_digest == three_rendered.artifact.artifact_digest
    assert one_bytes == three_bytes
    body = one_bytes.decode("utf-8")
    assert "country/KEN" in body
    assert "20.0" in body


def test_a_bare_pivot_never_builds_a_concatenated_frame_chunks(monkeypatch):
    """With no enrichment request, `_stream_bare_pivot` bypasses concatenation.
    The patched `from_chunks` raises if rendering calls it.
    """

    def _refuse(*_args: object, **_kwargs: object) -> ConcatenatedFrameChunks:
        raise AssertionError(
            "a bare pivot must never build a ConcatenatedFrameChunks over the raw rows"
        )

    monkeypatch.setattr(ConcatenatedFrameChunks, "from_chunks", _refuse)
    three_chunks = tuple(Table(columns=_BARE_COLUMNS, rows=(row,)) for row in _BARE_ROWS)
    options = MintOptions(
        format="csv", bundle=False, pivot=PivotOptions(index="recipient", columns="time_period")
    )

    rendered, fetched = _render(three_chunks, options=options)

    assert rendered.artifact.artifact_digest.startswith("sha256:")
    assert b"country/KEN" in fetched


def _profile(rows: tuple[tuple[object, ...], ...]) -> OutputProfile:
    return compute_output_profile(
        Table(columns=_BARE_COLUMNS, rows=rows),
        role_columns=("recipient",),
        enrichment=EnrichmentLookup(_GRAPH, type_kinds={}, parties={}),
    )


def test_a_stored_profile_over_the_cell_cap_refuses_before_reading_the_second_chunk():
    def chunks() -> Iterator[Table]:
        yield Table(columns=_BARE_COLUMNS, rows=_BARE_ROWS[:1])
        raise AssertionError("the pivot read past the first chunk")

    options = MintOptions(
        format="csv", bundle=False, pivot=PivotOptions(index="recipient", columns="time_period")
    )

    # Two recipients by three output columns is six cells.
    with pytest.raises(PivotOverCellCapError) as excinfo:
        _render(
            chunks(),
            options=options,
            output_profile=_profile(_BARE_ROWS),
            pivot_max_output_cells=5,
        )

    assert excinfo.value.count == 6


def test_a_non_unique_bare_pivot_names_the_stored_varying_column():
    rows = (
        ("dcid:SV_A", "country/KEN", "2020", 10.0),
        ("dcid:SV_B", "country/KEN", "2020", 20.0),
        ("dcid:SV_A", "country/TGO", "2021", 15.0),
    )
    options = MintOptions(
        format="csv", bundle=False, pivot=PivotOptions(index="recipient", columns="time_period")
    )

    with pytest.raises(PivotNotUniqueError) as excinfo:
        _render(
            (Table(columns=_BARE_COLUMNS, rows=rows),),
            options=options,
            output_profile=_profile(rows),
        )

    assert excinfo.value.varying == (("stat_var", 2),)


# --- Fixture 2: groupby (streamed collapse) then pivot. ---------------------

_STAMP = IndexStamp(
    graph_endpoint="https://example.test/core/api",
    exported_at="2026-09-09T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="deadbeef",
)
_PURPOSE_HEALTH = CodeRow(
    dcid="dcid:CRSPurposeCode122",
    code_system="crsPurpose",
    code="122",
    label="Basic health care",
    synonyms=(),
    parent=None,
)
_PURPOSE_MALARIA = CodeRow(
    dcid="dcid:CRSPurposeCode12250",
    code_system="crsPurpose",
    code="12250",
    label="Malaria control",
    synonyms=(),
    parent=None,
)
_FINANCE_GRANT = CodeRow(
    dcid="dcid:CRSTypeOfFinanceCode110",
    code_system="crsTypeOfFinance",
    code="110",
    label="Standard grant",
    synonyms=(),
    parent=None,
)
_FINANCE_LOAN = CodeRow(
    dcid="dcid:CRSTypeOfFinanceCode421",
    code_system="crsTypeOfFinance",
    code="421",
    label="Standard loan",
    synonyms=(),
    parent=None,
)

# Two declared-summable axes, matching `test_render_groupby_streaming_e2e
# .py`'s own fixture: `by=["crsPurpose"]` then collapses `crsTypeOfFinance`
# (the axis NOT named in `by`), merging SV_A/SV_B (same purpose, differing
# finance) into one row -- keeping `stat_var_crsPurpose` unique per
# recipient, exactly what the pivot below needs.
_DECLARATIONS = (
    CodeSystemDeclaration(
        code_system="crsPurpose",
        hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        plain_name="sector",
    ),
    CodeSystemDeclaration(
        code_system="crsTypeOfFinance",
        hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        plain_name="instrument",
    ),
)


def _statvar(dcid: str, *, purpose: str, finance: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="Amount",
        measured_property="amount",
        stat_type=None,
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=("recipient",),
        constraint_properties={"crsPurpose": purpose, "crsTypeOfFinance": finance},
    )


_INDEX = CodeIndex(
    stamp=_STAMP,
    declarations=_DECLARATIONS,
    codes=(_PURPOSE_HEALTH, _PURPOSE_MALARIA, _FINANCE_GRANT, _FINANCE_LOAN),
    statvars=(
        _statvar("SV_A", purpose=_PURPOSE_HEALTH.dcid, finance=_FINANCE_GRANT.dcid),
        _statvar("SV_B", purpose=_PURPOSE_HEALTH.dcid, finance=_FINANCE_LOAN.dcid),
        _statvar("SV_C", purpose=_PURPOSE_MALARIA.dcid, finance=_FINANCE_GRANT.dcid),
    ),
)

_GROUPBY_COLUMNS = ("stat_var", "recipient", "time_period", "value", "unit", "provenance")
# One recipient only (matching `test_render_groupby_streaming_e2e.py`'s own
# fixture): every resolved StatVar contributes a row for it, so the
# population check raises no `missing_members` refusal. In `row_sort_key`
# order (stat_var first, then role columns, then time_period): every
# `SV_A` row precedes every `SV_B` row, which precedes every `SV_C` row --
# `groupby_sum_chunks` requires non-decreasing order.
_GROUPBY_ROWS = (
    ("dcid:SV_A", "country/KEN", "2022", 10.0, "USD", "p1"),
    ("dcid:SV_B", "country/KEN", "2022", 5.0, "USD", "p1"),
    ("dcid:SV_C", "country/KEN", "2022", 7.0, "USD", "p1"),
)
_GROUPBY_RESOLVED_REFS = ("dcid:SV_A", "dcid:SV_B", "dcid:SV_C")


def _render_groupby_pivot(chunks: tuple[Table, ...]):
    artifact_store = FakeArtifactStore()
    rendered = render_and_store(
        frame_chunks=iter(chunks),
        options=MintOptions(
            format="csv",
            bundle=False,
            groupby=GroupbyOptions(by=["crsPurpose"], op="sum"),
            pivot=PivotOptions(index="recipient", columns="stat_var_crsPurpose"),
        ),
        spec={"spec_id": "groupby-pivot-streaming-e2e"},
        provenance_refs=["p1"],
        citations={},
        reproducing_call=None,
        content_version=_CONTENT_VERSION,
        artifact_store=artifact_store,
        entity_columns=("recipient",),
        enrichment=EnrichmentLookup(_GRAPH, type_kinds={}, parties={}),
        code_index=_INDEX,
        resolved_stat_var_refs=_GROUPBY_RESOLVED_REFS,
        pivot_max_output_cells=5_000_000,
        prepared_at=PREPARED_AT,
        site=SITE,
    )
    return rendered, artifact_store.fetch(rendered.artifact.url)


def test_a_groupby_then_pivot_matches_across_a_one_chunk_and_a_three_chunk_split():
    one_chunk = (Table(columns=_GROUPBY_COLUMNS, rows=_GROUPBY_ROWS),)
    three_chunks = tuple(Table(columns=_GROUPBY_COLUMNS, rows=(row,)) for row in _GROUPBY_ROWS)

    one_rendered, one_bytes = _render_groupby_pivot(one_chunk)
    three_rendered, three_bytes = _render_groupby_pivot(three_chunks)

    assert one_rendered.artifact.artifact_digest == three_rendered.artifact.artifact_digest
    assert one_bytes == three_bytes
    body = one_bytes.decode("utf-8")
    assert "country/KEN" in body
    assert "15.0" in body  # SV_A + SV_B merged under crsPurpose=health for KEN


# --- Fixture 3: a pivot indexed by a non-outermost sort column. -------------

# `row_sort_key` orders by `stat_var`, role columns, and `time_period`.
# A time period can recur after another StatVar's rows, so pivoting must handle non-adjacent rows.
_NON_OUTERMOST_COLUMNS = ("stat_var", "recipient", "time_period", "value")
_NON_OUTERMOST_ROWS = (
    ("dcid:StatVarAlpha", "country/KEN", "2020", 1.0),
    ("dcid:StatVarAlpha", "country/KEN", "2021", 2.0),
    ("dcid:StatVarBeta", "country/KEN", "2020", 3.0),
    ("dcid:StatVarBeta", "country/KEN", "2021", 4.0),
)


def test_a_pivot_whose_index_is_not_the_outermost_sort_column_matches_across_chunk_splits():
    one_chunk = (Table(columns=_NON_OUTERMOST_COLUMNS, rows=_NON_OUTERMOST_ROWS),)
    four_chunks = tuple(
        Table(columns=_NON_OUTERMOST_COLUMNS, rows=(row,)) for row in _NON_OUTERMOST_ROWS
    )
    options = MintOptions(
        format="csv", bundle=False, pivot=PivotOptions(index="time_period", columns="stat_var")
    )

    one_rendered, one_bytes = _render(one_chunk, options=options)
    four_rendered, four_bytes = _render(four_chunks, options=options)

    assert one_rendered.artifact.artifact_digest == four_rendered.artifact.artifact_digest
    assert one_bytes == four_bytes
    body = one_bytes.decode("utf-8")
    assert "2020" in body
    assert "2021" in body
