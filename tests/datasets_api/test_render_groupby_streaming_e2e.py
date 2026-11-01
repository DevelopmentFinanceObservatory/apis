"""A groupby collapse renders to identical CSV bytes from one chunk or three through
`render_and_store`, and refuses a frame past its fold ceiling before folding.
"""

from __future__ import annotations

import contextvars
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from dfo_frame_grammar import Frame

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.types import StatVarRow
from datasets_api import render
from datasets_api.failures import classify_failure
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.render import render_and_store
from materialise.enrichment import EnrichmentLookup
from materialise.errors import OverCeilingError
from materialise.frame_build import ParquetFrameChunks, ParquetFrameSink, build_frame

from ._artifact_doubles import FakeArtifactStore
from ._frame_chunks import frame_chunks_of, frame_of_rows
from ._graph_doubles import FakeMintGraphReader
from ._render_inputs import PREPARED_AT, SITE

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

_GRAPH = FakeMintGraphReader()
_TABLE_COLUMNS = ("stat_var", "recipient", "time_period", "value", "unit", "provenance")
_ROWS = (
    ("dcid:SV_A", "country/KEN", "2022", 10.0, "USD", "p1"),
    ("dcid:SV_B", "country/KEN", "2022", 5.0, "USD", "p1"),
    ("dcid:SV_C", "country/KEN", "2022", 7.0, "USD", "p1"),
)
_RESOLVED_STAT_VAR_REFS = ("dcid:SV_A", "dcid:SV_B", "dcid:SV_C")


def _render(
    chunks: tuple[Frame, ...] | ParquetFrameChunks,
    *,
    groupby_fold_max_rows: int = 2_000_000,
    frame_rows: int | None = None,
    deadline: float | None = None,
):
    artifact_store = FakeArtifactStore()
    rendered = render_and_store(
        frame_chunks=chunks if isinstance(chunks, ParquetFrameChunks) else lambda: iter(chunks),
        options=MintOptions(
            format="csv", bundle=False, groupby=GroupbyOptions(by=["crsPurpose"], op="sum")
        ),
        spec={"spec_id": "groupby-streaming-e2e"},
        provenance_refs=["p1"],
        citations={},
        reproducing_call=None,
        content_version="sha256:" + "0" * 64,
        artifact_store=artifact_store,
        entity_columns=("recipient",),
        enrichment=EnrichmentLookup(_GRAPH, parties=lambda: {}),
        code_index=_INDEX,
        resolved_stat_var_refs=_RESOLVED_STAT_VAR_REFS,
        pivot_max_output_cells=5_000_000,
        groupby_fold_max_rows=groupby_fold_max_rows,
        frame_rows=frame_rows,
        deadline=deadline,
        prepared_at=PREPARED_AT,
        site=SITE,
    )
    return rendered, artifact_store.fetch(rendered.artifact.url)


def test_a_three_way_split_matches_the_single_chunk_artifact_digest(monkeypatch):
    one_rendered, one_bytes = _render((frame_of_rows(_TABLE_COLUMNS, _ROWS),))
    monkeypatch.setattr(render, "_RESHAPE_CHUNK_ROWS", 1)
    three_rendered, three_bytes = _render(_frames())

    assert one_rendered.artifact.artifact_digest == three_rendered.artifact.artifact_digest
    assert one_bytes == three_bytes
    body = one_bytes.decode("utf-8")
    assert "15.0" in body  # SV_A + SV_B merged under crsPurpose=health
    assert "7.0" in body  # SV_C stays apart (malaria)


def _frames() -> tuple[Frame, ...]:
    """`_ROWS` as three one-row Arrow frames."""
    return tuple(frame_chunks_of(_TABLE_COLUMNS, [(row,) for row in _ROWS])())


def _parquet_frame() -> ParquetFrameChunks:
    """`_ROWS` held as a one-row-group Parquet frame, as a mint holds its frame."""
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            dict(zip(_TABLE_COLUMNS, (*row[:4], "unit/USD", row[5]), strict=True)) for row in _ROWS
        ],
    )
    sink = ParquetFrameSink()
    sink.write(frame.table)
    return sink.finish(columns=frame.columns)


@pytest.mark.parametrize(
    ("chunks", "frame_rows", "fold_functions"),
    [
        pytest.param(
            _frames,
            None,
            ("widen_frame_table", "decode_table", "fold_groupby_sum_table"),
            id="arrow-slices-read",
        ),
    ],
)
def test_a_frame_past_the_fold_ceiling_refuses_before_any_fold(
    monkeypatch, chunks, frame_rows, fold_functions
):
    """A collapse of a frame of more rows than the fold ceiling refuses `over_ceiling` (`rows`)
    before any fold runs. A frame at the ceiling renders as an uncapped render does."""
    monkeypatch.setattr(render, "_RESHAPE_CHUNK_ROWS", 1)

    uncapped, uncapped_bytes = _render(chunks())
    at_ceiling, at_ceiling_bytes = _render(
        chunks(), frame_rows=frame_rows, groupby_fold_max_rows=len(_ROWS)
    )

    def unreached(*_args, **_kwargs):
        raise AssertionError("the fold ran")

    for name in fold_functions:
        monkeypatch.setattr(render, name, unreached)
    with pytest.raises(OverCeilingError) as refused:
        _render(chunks(), frame_rows=frame_rows, groupby_fold_max_rows=len(_ROWS) - 1)

    assert at_ceiling.artifact.artifact_digest == uncapped.artifact.artifact_digest
    assert at_ceiling_bytes == uncapped_bytes
    assert (refused.value.code, refused.value.ceiling_kind) == ("over_ceiling", "rows")
    assert (refused.value.count, refused.value.cap) == (len(_ROWS), len(_ROWS) - 1)


class _Overlap:
    """Counts the folds running at once. The first fold to start waits until a second render
    has reached the fold permit or the fold itself, so one permit is all that keeps them
    apart."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._arrived: set[int] = set()
        self._both_arrived = threading.Event()
        self.running = 0
        self.peak = 0

    def arrive(self) -> None:
        with self._lock:
            self._arrived.add(threading.get_ident())
            if len(self._arrived) == 2:
                self._both_arrived.set()

    def permits(self) -> threading.BoundedSemaphore:
        overlap = self

        class _Arriving(threading.BoundedSemaphore):
            def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
                overlap.arrive()
                return super().acquire(blocking, timeout)

            __enter__ = acquire

        return _Arriving(1)

    def fold(self, real):
        def counted(*args, **kwargs):
            self.arrive()
            with self._lock:
                self.running += 1
                self.peak = max(self.peak, self.running)
            # A deadlock breaker only: a second render arrives at once on either path.
            self._both_arrived.wait(timeout=10)
            try:
                return real(*args, **kwargs)
            finally:
                with self._lock:
                    self.running -= 1

        return counted


def test_two_collapses_never_fold_at_once(monkeypatch):
    """A collapse waits for the process's one fold permit while another folds, and then
    renders as it would alone."""
    monkeypatch.setattr(render, "_RESHAPE_CHUNK_ROWS", 1)
    _alone, alone_bytes = _render(_frames())
    overlap = _Overlap()
    permits = overlap.permits()
    monkeypatch.setattr(render, "_fold_permits", lambda: permits)
    monkeypatch.setattr(
        render, "fold_groupby_sum_table", overlap.fold(render.fold_groupby_sum_table)
    )

    # Each render runs in a copy of this context, which binds the graph release.
    contexts = [contextvars.copy_context() for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        rendered = list(pool.map(lambda context: context.run(_render, _frames()), contexts))

    assert overlap.peak == 1
    assert [body for _rendered, body in rendered] == [alone_bytes, alone_bytes]


@pytest.mark.parametrize(
    ("chunks", "frame_rows", "fold_max_rows", "code"),
    [
        pytest.param(_frames, None, 2_000_000, "timeout", id="wait-ends-at-the-deadline"),
        pytest.param(
            _frames, len(_ROWS), len(_ROWS) - 1, "over_ceiling", id="stated-rows-refuse-first"
        ),
        pytest.param(
            _parquet_frame, None, len(_ROWS) - 1, "over_ceiling", id="parquet-footer-refuses-first"
        ),
    ],
)
def test_a_collapse_waiting_for_a_held_fold_permit(
    monkeypatch, chunks, frame_rows, fold_max_rows, code
):
    """While another render holds the fold permit, a collapse of a frame under its ceiling
    waits no longer than its deadline and then answers timeout. A frame whose rows are known
    before it is read, stated or in a Parquet footer, and are past the ceiling refuses
    over_ceiling before it waits."""
    permits = threading.BoundedSemaphore(1)
    monkeypatch.setattr(render, "_fold_permits", lambda: permits)
    assert permits.acquire(blocking=False)

    with pytest.raises(Exception) as raised:
        _render(
            chunks(),
            groupby_fold_max_rows=fold_max_rows,
            frame_rows=frame_rows,
            deadline=time.monotonic() + 0.05,
        )

    if code == "over_ceiling":
        assert isinstance(raised.value, OverCeilingError)
        assert raised.value.count == len(_ROWS)
    else:
        problem = classify_failure(raised.value)
        assert problem is not None
        assert (problem.code, problem.retryable) == ("timeout", True)
