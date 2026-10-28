"""The analysis runner reproduces each recorded recipe's output from its recorded root frames
(`tests/fixtures/analysis_golden`, written by `datasets-maintenance record-recipe-golden`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from datasets_api.analysis import AnalysisRequest, run_analysis
from datasets_api.doors.publish import publish_built_frame
from datasets_api.maintenance import run_record_recipe_golden
from datasets_api.recipe_golden import OUTPUT_FILE, cases, replay_case
from datasets_api.recipe_replay import frame_facts
from materialise.formats import read_canonical_frame

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
from ._render_inputs import ENRICHMENT, PREPARED_AT, SITE

_CASES = cases(Path(__file__).parents[1] / "fixtures" / "analysis_golden")


def test_the_golden_holds_cases() -> None:
    assert _CASES


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_the_runner_reproduces_the_recorded_output(case: Path) -> None:
    expected, computed = replay_case(case)

    assert frame_facts(read_canonical_frame((case / OUTPUT_FILE).read_bytes())) == expected
    assert computed == expected


def test_a_recorded_recipe_replays_to_its_stored_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stores = Stores()
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
    run_analysis(
        AnalysisRequest.model_validate({"tree": tree}),
        source=source(stores),
        stores=stores.analysis(),
    )

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
    lines = [line for line in capsys.readouterr().out.splitlines() if "golden_recorded" in line]
    assert ['"written": 1' in lines[0], '"skipped": 1' in lines[1]] == [True, True]
