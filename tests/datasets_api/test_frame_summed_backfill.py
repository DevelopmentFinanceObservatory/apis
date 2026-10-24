"""`datasets-maintenance backfill-frame-summed` (`maintenance.run_backfill_frame_summed`) over
party-totals cuts minted with a groupby, their `frame_summed` cleared as on a row recorded
before migration 0038."""

from __future__ import annotations

import json
from typing import Any

import pytest

from datasets_api import maintenance
from datasets_api.maintenance import run_backfill_frame_summed

from .test_version_file_selector import (
    _CUT_FAMILY_IDENTITY,
    _cut_family_code_index,
    _cut_with_groupby,
    _drop_base_frame,
    _frame_summed,
    _mint,
)

_OUTPUT = {"format": "parquet", "bundle": False}


def _clear(version_store, cut: dict[str, Any], *columns: str) -> None:
    """Set each of `columns` to NULL on `cut`'s version row."""
    with version_store.transaction() as tx:
        for column in columns:
            tx.execute(
                f"UPDATE datasets.version_record SET {column} = NULL WHERE content_version = %s",
                (cut["content_version"],),
            )


def _recorded_without_frame_summed(
    by: list[str], *, flow_summable: bool
) -> tuple[Any, Any, dict[str, Any]]:
    """The version and artifact stores of a cut minted with a groupby by `by`, and the cut,
    its `frame_summed` cleared."""
    _, _, version_store, artifact_store, cut, _ = _cut_with_groupby(
        by, flow_summable=flow_summable, output=_OUTPUT
    )
    _clear(version_store, cut, "frame_summed")
    return version_store, artifact_store, cut


def _summed(version_store, cut: dict[str, Any]) -> bool | None:
    return _frame_summed(
        version_store,
        subject_id=cut["definition"]["spec_id"],
        content_version=cut["content_version"],
    )


def _backfilled(
    version_store, artifact_store, cut: dict[str, Any], *, flow_summable: bool, capsys
) -> tuple[bool | None, dict[str, Any]]:
    """Run the backfill under the `_CUT_FAMILY` index with `flow_summable`, and return the
    cut's `frame_summed` and the run's counts line."""
    capsys.readouterr()
    failed = run_backfill_frame_summed(
        store=version_store,
        artifact_store=artifact_store,
        code_index=_cut_family_code_index(flow_summable=flow_summable),
    )
    assert failed == 0
    [counts] = [
        line
        for line in map(json.loads, capsys.readouterr().out.splitlines())
        if line.get("event") == "datasets_frame_summed_backfilled"
    ]
    return _summed(version_store, cut), counts


@pytest.mark.parametrize("frame_hash", ["recorded", "unrecorded"])
@pytest.mark.parametrize(
    ("by", "flow_summable", "summed", "outcome"),
    [
        pytest.param(
            ["recipient", "testAxis"], False, False, "cut", id="groupby-leaves-the-cut-unchanged"
        ),
        pytest.param(["recipient"], False, False, "cut", id="groupby-keeps-an-unsummed-axis"),
        pytest.param(
            ["recipient", "testAxis"], True, True, "summed", id="groupby-sums-and-keeps-a-code-axis"
        ),
    ],
)
def test_a_frame_the_recut_or_its_collapse_reproduces_sets_frame_summed(
    by: list[str], flow_summable: bool, summed: bool, outcome: str, frame_hash: str, capsys
):
    """The base frame less the recorded cut is the cut's frame, or the recorded groupby's
    collapse of it is, matched on the recorded frame hash or, on a row recorded without one,
    on the content version minted from the hash alone. The value set is the one the mint
    recorded."""
    version_store, artifact_store, cut = _recorded_without_frame_summed(
        by, flow_summable=flow_summable
    )
    if frame_hash == "unrecorded":
        _clear(version_store, cut, "frame_sha256")

    frame_summed, counts = _backfilled(
        version_store, artifact_store, cut, flow_summable=flow_summable, capsys=capsys
    )

    assert frame_summed is summed
    assert counts[outcome] == 1


def test_a_collapse_the_current_index_no_longer_reproduces_stays_null(capsys):
    """The cut's frame is the collapse that summed `flowAxis`. Under an index that no longer
    declares `flowAxis` summable, neither the cut frame nor any collapse matches it."""
    version_store, artifact_store, cut = _recorded_without_frame_summed(
        ["recipient", "testAxis"], flow_summable=True
    )

    frame_summed, counts = _backfilled(
        version_store, artifact_store, cut, flow_summable=False, capsys=capsys
    )

    assert frame_summed is None
    assert counts["unmatched"] == 1


def test_a_cut_whose_base_frame_bytes_are_gone_stays_null(capsys):
    version_store, artifact_store, cut = _recorded_without_frame_summed(
        ["recipient"], flow_summable=False
    )
    _drop_base_frame(version_store, artifact_store, cut=cut)

    frame_summed, counts = _backfilled(
        version_store, artifact_store, cut, flow_summable=False, capsys=capsys
    )

    assert frame_summed is None
    assert counts["base_frame_gone"] == 1


def test_a_recorded_frame_summed_is_left_as_it_is(capsys):
    """A row holding `frame_summed`, here the opposite of what its frame holds, is neither
    read nor overwritten."""
    _, _, version_store, artifact_store, cut, _ = _cut_with_groupby(
        ["recipient", "testAxis"], flow_summable=True, output=_OUTPUT
    )
    with version_store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.version_record SET frame_summed = %s WHERE content_version = %s",
            (False, cut["content_version"]),
        )

    frame_summed, counts = _backfilled(
        version_store, artifact_store, cut, flow_summable=True, capsys=capsys
    )

    assert frame_summed is False
    assert counts["own_frame_known"] == counts["summed"] == counts["cut"] == 0


def test_a_value_set_after_the_row_was_read_is_not_overwritten(
    monkeypatch: pytest.MonkeyPatch, capsys
):
    """A row the new image records `frame_summed` on while the run matches its frame keeps
    the value the image wrote."""
    version_store, artifact_store, cut = _recorded_without_frame_summed(
        ["recipient", "testAxis"], flow_summable=True
    )
    matched = maintenance._matched_frame_summed

    def written_meanwhile(*args: Any, **kwargs: Any) -> maintenance.FrameSummedOutcome:
        with version_store.transaction() as tx:
            tx.execute(
                "UPDATE datasets.version_record SET frame_summed = %s WHERE content_version = %s",
                (False, cut["content_version"]),
            )
        return matched(*args, **kwargs)

    monkeypatch.setattr(maintenance, "_matched_frame_summed", written_meanwhile)

    frame_summed, counts = _backfilled(
        version_store, artifact_store, cut, flow_summable=True, capsys=capsys
    )

    assert counts["summed"] == 1
    assert frame_summed is False


def test_a_run_reads_every_page(monkeypatch: pytest.MonkeyPatch, capsys):
    """Two cuts, each a collapse of its own, read one row a page: both are set."""
    client, _, version_store, artifact_store, first, _ = _cut_with_groupby(
        ["recipient", "testAxis"], flow_summable=True, output=_OUTPUT
    )
    spec_id = first["definition"]["spec_id"]
    second = _mint(
        client,
        spec_id=spec_id,
        sidecar={"schema_version": 1, "spec_id": spec_id, "identity": _CUT_FAMILY_IDENTITY},
        options={
            **_OUTPUT,
            "without_party_totals": True,
            "groupby": {"by": ["recipient"], "op": "sum"},
        },
    )
    assert second["content_version"] != first["content_version"]
    _clear(version_store, first, "frame_summed")
    _clear(version_store, second, "frame_summed")
    monkeypatch.setattr(maintenance, "FRAME_SUMMED_PAGE", 1)

    _, counts = _backfilled(version_store, artifact_store, first, flow_summable=True, capsys=capsys)

    assert counts["summed"] == 2
    assert _summed(version_store, first) is True
    assert _summed(version_store, second) is True
