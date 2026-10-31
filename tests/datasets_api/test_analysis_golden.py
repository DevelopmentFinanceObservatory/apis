"""The analysis runner reproduces each recorded recipe's output from its recorded root frames
(`tests/fixtures/analysis_golden`, written by `datasets-maintenance record-recipe-golden`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from datasets_api.analysis import AnalysisRequest, run_analysis
from datasets_api.doors.publish import publish_built_frame
from datasets_api.maintenance import run_record_recipe_golden
from datasets_api.recipe_golden import cases, recorded_facts, replay_case
from datasets_api.recipe_replay import StoredRootSource
from dfo_analysis_engine.frame_ref import FrameRef

from ._analysis_doubles import (
    AID,
    AID_V1,
    POPULATION,
    POPULATION_V1,
    Stores,
    aid_frame,
    dataset,
    population_frame,
    source,
    step,
)
from ._frame_doubles import link_frame1_only
from ._render_inputs import ENRICHMENT, PREPARED_AT, SITE

_CASES = cases(Path(__file__).parents[1] / "fixtures" / "analysis_golden")


def test_the_golden_holds_cases() -> None:
    assert _CASES


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_the_runner_reproduces_the_recorded_output(case: Path) -> None:
    expected, computed = replay_case(case)

    assert recorded_facts(case) == expected
    assert computed == expected


def _stored_quotient(stores: Stores) -> FrameRef:
    """Store the aid and population roots and the quotient run over them, with its recipe, and
    return the run's pair."""
    for spec_id, content_version, frame in (
        (AID, AID_V1, aid_frame()),
        (POPULATION, POPULATION_V1, population_frame()),
    ):
        publish_built_frame(
            frame=frame,
            spec_id=spec_id,
            content_version=content_version,
            input_content_versions=[],
            artifact_store=stores.artifact_store,
            snapshot_reader=stores.snapshot_reader,
            version_writer=stores.version_writer,
            file_enrichment=ENRICHMENT,
            site=SITE,
            prepared_at=PREPARED_AT,
        )
    tree = step(
        "quotient_of",
        dataset(AID, AID_V1),
        dataset(POPULATION, POPULATION_V1),
        on={"destination": "place"},
    )
    return run_analysis(
        AnalysisRequest.model_validate({"tree": tree}),
        source=source(stores),
        stores=stores.analysis(),
    ).evaluation.ref


def _emitted(capsys: pytest.CaptureFixture[str], event: str) -> list[dict[str, Any]]:
    """The JSON lines the recording emitted under `event`."""
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if event in line]


def test_a_recorded_recipe_replays_to_its_stored_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stores = Stores()
    _stored_quotient(stores)

    for out, max_bytes in ((tmp_path / "kept", 2_000_000), (tmp_path / "skipped", 100)):
        run_record_recipe_golden(
            store=stores.store,
            artifact_store=stores.artifact_store,
            out=out,
            limit=50,
            max_bytes=max_bytes,
            origin="test",
            now=lambda: stores.now,
        )

    (case,) = cases(tmp_path / "kept")
    expected, computed = replay_case(case)
    assert computed == expected
    assert cases(tmp_path / "skipped") == []
    kept, skipped = _emitted(capsys, "datasets_recipe_golden_recorded")
    assert (kept["written"], kept["skipped"]) == (1, 0)
    assert (skipped["written"], skipped["skipped_by"]["past_max_bytes"]) == (0, 1)


def test_a_root_past_the_decode_bound_is_skipped_before_any_root_is_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root stored as a frame1 document recorded at 250 MB is past the decode bound, and the
    recording locates every root before it reads one, so the case is skipped unread."""
    stores = Stores()
    ref = _stored_quotient(stores)
    digest = link_frame1_only(
        stores.store, stores.artifact_store, subject_id=AID, content_version=AID_V1
    )
    with stores.store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_inventory SET size_bytes = %s WHERE artifact_digest = %s",
            (250_000_000, digest),
        )

    def unread(self, root):
        raise AssertionError("a root was read")

    monkeypatch.setattr(StoredRootSource, "read", unread)
    run_record_recipe_golden(
        store=stores.store,
        artifact_store=stores.artifact_store,
        out=tmp_path,
        limit=50,
        max_bytes=2_000_000,
        origin="test",
        now=lambda: stores.now,
    )

    assert cases(tmp_path) == []
    out = capsys.readouterr().out.splitlines()
    (skip,) = [json.loads(line) for line in out if "datasets_recipe_golden_skipped" in line]
    (recorded,) = [json.loads(line) for line in out if "datasets_recipe_golden_recorded" in line]
    assert (skip["spec_id"], skip["content_version"], skip["ceiling_kind"]) == (
        ref.spec_id,
        ref.content_version,
        "bytes",
    )
    assert recorded["skipped_by"] == {"not_held": 0, "past_max_bytes": 0, "over_ceiling": 1}
