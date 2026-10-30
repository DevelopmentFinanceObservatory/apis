"""A groupby that collapses no axis can still stream into a pivot.

The no-summable-axis refusal depends on distinct StatVars collected across chunks.
Tests compare streamed results with `_reshape` over concatenated rows.
"""

from __future__ import annotations

import pytest

from code_index import CodeIndex, IndexStamp
from code_index.types import StatVarRow
from datasets_api.mint_models import GroupbyOptions, MintOptions, PivotOptions
from datasets_api.render import _RenderReads, _reshape, _stream_bare_pivot, render_and_store
from materialise.enrichment import EnrichmentLookup
from materialise.pipeline import ConcatenatedFrameChunks
from materialise.profile import OutputProfile, PivotFeasibility
from materialise.reshape import PivotNotUniqueError, RollupNotSummableError, Table

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._render_inputs import PREPARED_AT, SITE

_GRAPH = FakeMintGraphReader()
_STAMP = IndexStamp(
    graph_endpoint="https://example.test/core/api",
    exported_at="2026-09-09T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="deadbeef",
)


def _statvar(dcid: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="Amount",
        measured_property="amount",
        stat_type=None,
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=("recipient",),
        constraint_properties={},
    )


# No code system is declared summable at all, so `declared_axes_present`
# is empty regardless of `by` -- the "collapses nothing" case this test
# targets, distinct from `by` merely naming every declared axis.
_INDEX_TWO_CONTRIBUTORS = CodeIndex(
    stamp=_STAMP,
    declarations=(),
    codes=(),
    statvars=(_statvar("SV_A"), _statvar("SV_B")),
)
_INDEX_ONE_CONTRIBUTOR = CodeIndex(
    stamp=_STAMP, declarations=(), codes=(), statvars=(_statvar("SV_A"),)
)

_COLUMNS = ("stat_var", "recipient", "time_period", "value")
_TWO_CONTRIBUTOR_ROWS = (
    ("dcid:SV_A", "country/KEN", "2020", 10.0),
    ("dcid:SV_B", "country/KEN", "2021", 20.0),
)
_ONE_CONTRIBUTOR_ROWS = (
    ("dcid:SV_A", "country/KEN", "2020", 10.0),
    ("dcid:SV_A", "country/TZA", "2021", 20.0),
)


def _options() -> MintOptions:
    return MintOptions(
        format="csv",
        bundle=False,
        groupby=GroupbyOptions(by=["recipient"], op="sum"),
        pivot=PivotOptions(index="recipient", columns="time_period"),
    )


def _chunks(rows: tuple, bounds: list[int]) -> list[Table]:
    return [
        Table(columns=_COLUMNS, rows=rows[bounds[i] : bounds[i + 1]])
        for i in range(len(bounds) - 1)
    ]


def test_two_contributors_with_no_summable_axis_refuses_the_same_way_streamed():
    whole_table = Table(columns=_COLUMNS, rows=_TWO_CONTRIBUTOR_ROWS)

    try:
        _reshape(
            whole_table,
            options=_options(),
            entity_columns=("recipient",),
            reads=_RenderReads(
                enrichment=EnrichmentLookup(_GRAPH, parties=lambda: {}),
                known={},
                deadline=None,
            ),
            code_index=_INDEX_TWO_CONTRIBUTORS,
            resolved_stat_var_refs=("dcid:SV_A", "dcid:SV_B"),
            provenance_refs=("p1",),
            pivot_max_output_cells=5_000_000,
        )
        raise AssertionError("expected the whole-table call to refuse no_summable_axis")
    except RollupNotSummableError as whole_exc:
        assert whole_exc.precondition == "no_summable_axis"

    for bounds in ([0, len(_TWO_CONTRIBUTOR_ROWS)], [0, 1, len(_TWO_CONTRIBUTOR_ROWS)]):
        try:
            _stream_bare_pivot(
                iter(_chunks(_TWO_CONTRIBUTOR_ROWS, bounds)),
                options=_options(),
                entity_columns=("recipient",),
                reads=_RenderReads(
                    enrichment=EnrichmentLookup(_GRAPH, parties=lambda: {}),
                    known={},
                    deadline=None,
                ),
                code_index=_INDEX_TWO_CONTRIBUTORS,
                resolved_stat_var_refs=("dcid:SV_A", "dcid:SV_B"),
                provenance_refs=("p1",),
                subject_id="collapses-nothing-pivot-test",
                pivot_max_output_cells=5_000_000,
            )
            raise AssertionError("expected the streamed call to refuse no_summable_axis")
        except RollupNotSummableError as streamed_exc:
            assert streamed_exc.precondition == "no_summable_axis"


@pytest.mark.parametrize(
    "bounds",
    [[0, 2], [0, 1, 2]],
    ids=["stored_single_chunk", "fresh_multi_chunk"],
)
def test_no_summable_axis_refuses_before_a_stored_pivot_cap_on_every_chunking(bounds):
    """The stored profile says the pivot is over the cell cap. The first chunk of the fresh mint
    carries one StatVar, so only a read of every chunk sees the second and refuses first."""
    over_cap = PivotFeasibility(
        index="recipient",
        columns="time_period",
        duplicate_keys=False,
        homogeneous=True,
        unique=True,
        width=3,
        cells=6,
        varying=(),
    )
    profile = OutputProfile(
        schema_version=1, rows=2, columns=(), pivots=(over_cap,), enrichments=()
    )

    with pytest.raises(RollupNotSummableError) as refused:
        _stream_bare_pivot(
            iter(_chunks(_TWO_CONTRIBUTOR_ROWS, bounds)),
            options=_options(),
            entity_columns=("recipient",),
            reads=_RenderReads(
                enrichment=EnrichmentLookup(_GRAPH, parties=lambda: {}),
                known={},
                deadline=None,
            ),
            code_index=_INDEX_TWO_CONTRIBUTORS,
            resolved_stat_var_refs=("dcid:SV_A", "dcid:SV_B"),
            provenance_refs=("p1",),
            subject_id="collapses-nothing-pivot-test",
            pivot_max_output_cells=1,
            output_profile=profile,
        )

    assert refused.value.precondition == "no_summable_axis"


@pytest.mark.parametrize(
    "groupby",
    [None, GroupbyOptions(by=["recipient"], op="sum")],
    ids=["bare", "noop_groupby"],
)
def test_a_duplicate_key_pivot_names_the_varying_column_with_or_without_a_groupby(groupby):
    """A one-StatVar frame whose key repeats under two units refuses `pivot_not_unique` naming
    `unit`, whether the stored profile supplies it or the fold finds it."""
    columns = (*_COLUMNS[:-1], "unit", "value")
    rows = (
        ("dcid:SV_A", "country/KEN", "2020", "USD", 10.0),
        ("dcid:SV_A", "country/KEN", "2020", "EUR", 11.0),
    )
    facts = PivotFeasibility(
        index="recipient",
        columns="time_period",
        duplicate_keys=True,
        homogeneous=True,
        unique=False,
        width=2,
        cells=2,
        varying=(("unit", 2),),
    )
    profile = OutputProfile(schema_version=1, rows=2, columns=(), pivots=(facts,), enrichments=())
    options = MintOptions(
        format="csv",
        bundle=False,
        groupby=groupby,
        pivot=PivotOptions(index="recipient", columns="time_period"),
    )

    with pytest.raises(PivotNotUniqueError) as refused:
        _stream_bare_pivot(
            iter([Table(columns=columns, rows=rows)]),
            options=options,
            entity_columns=("recipient",),
            reads=_RenderReads(
                enrichment=EnrichmentLookup(_GRAPH, parties=lambda: {}),
                known={},
                deadline=None,
            ),
            code_index=_INDEX_ONE_CONTRIBUTOR,
            resolved_stat_var_refs=("dcid:SV_A",),
            provenance_refs=("p1",),
            subject_id="collapses-nothing-pivot-test",
            pivot_max_output_cells=5_000_000,
            output_profile=profile,
        )

    assert refused.value.varying == (("unit", 2),)


def test_one_contributor_with_no_summable_axis_pivots_successfully_and_matches_whole_table():
    whole_table = Table(columns=_COLUMNS, rows=_ONE_CONTRIBUTOR_ROWS)
    whole = _reshape(
        whole_table,
        options=_options(),
        entity_columns=("recipient",),
        reads=_RenderReads(
            enrichment=EnrichmentLookup(_GRAPH, parties=lambda: {}), known={}, deadline=None
        ),
        code_index=_INDEX_ONE_CONTRIBUTOR,
        resolved_stat_var_refs=("dcid:SV_A",),
        provenance_refs=("p1",),
        pivot_max_output_cells=5_000_000,
    )

    for bounds in ([0, len(_ONE_CONTRIBUTOR_ROWS)], [0, 1, len(_ONE_CONTRIBUTOR_ROWS)]):
        streamed = _stream_bare_pivot(
            iter(_chunks(_ONE_CONTRIBUTOR_ROWS, bounds)),
            options=_options(),
            entity_columns=("recipient",),
            reads=_RenderReads(
                enrichment=EnrichmentLookup(_GRAPH, parties=lambda: {}),
                known={},
                deadline=None,
            ),
            code_index=_INDEX_ONE_CONTRIBUTOR,
            resolved_stat_var_refs=("dcid:SV_A",),
            provenance_refs=("p1",),
            subject_id="collapses-nothing-pivot-test",
            pivot_max_output_cells=5_000_000,
        )
        assert streamed.table == whole.table


def test_render_and_store_never_builds_a_concatenated_frame_chunks_for_this_combination(
    monkeypatch,
):
    """A patched `from_chunks` raises if rendering uses the concatenation fallback."""

    def _refuse(*_args: object, **_kwargs: object) -> ConcatenatedFrameChunks:
        raise AssertionError(
            "a groupby-collapses-nothing-then-pivot request must never build a "
            "ConcatenatedFrameChunks over the raw rows"
        )

    monkeypatch.setattr(ConcatenatedFrameChunks, "from_chunks", _refuse)
    chunks = tuple(Table(columns=_COLUMNS, rows=(row,)) for row in _ONE_CONTRIBUTOR_ROWS)
    rendered = render_and_store(
        frame_chunks=lambda: iter(chunks),
        options=_options(),
        spec={"spec_id": "collapses-nothing-pivot-e2e"},
        provenance_refs=["p1"],
        citations={},
        reproducing_call=None,
        content_version="sha256:" + "0" * 64,
        artifact_store=FakeArtifactStore(),
        entity_columns=("recipient",),
        enrichment=EnrichmentLookup(_GRAPH, parties=lambda: {}),
        code_index=_INDEX_ONE_CONTRIBUTOR,
        resolved_stat_var_refs=("dcid:SV_A",),
        pivot_max_output_cells=5_000_000,
        groupby_fold_max_rows=2_000_000,
        groupby_row_fold_max_rows=700_000,
        prepared_at=PREPARED_AT,
        site=SITE,
    )
    assert rendered.artifact.artifact_digest.startswith("sha256:")
