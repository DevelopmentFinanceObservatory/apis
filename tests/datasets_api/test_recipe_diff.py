"""`maintenance diff-recipes`: each stored recipe computed again on its own root versions, read
from live store SQL and an in-memory bucket, compared with its stored frame, and nothing
written."""

from __future__ import annotations

import functools
import io
import json
from datetime import timedelta
from typing import Any

import pytest
from dfo_frame_grammar import Frame

from datasets_api import analysis, recipe_replay
from datasets_api.analysis import (
    AnalysisBounds,
    AnalysisRequest,
    compute_analysis,
    run_analysis,
)
from datasets_api.doors.publish import publish_built_frame
from datasets_api.maintenance import main, run_diff_recipes

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

_PER_PERSON = step(
    "quotient_of",
    dataset(AID, AID_V1),
    dataset(POPULATION, POPULATION_V1),
    on={"destination": "place"},
)
_SUMMED = step("aggregate", dataset(AID, AID_V1), group_by=["origin"], aggregation="sum")
_COMPARED = step("compare", dataset(AID, AID_V1), dataset(AID, AID_V1), on=["financingRecipient"])


def _store_root(stores: Stores, spec_id: str, content_version: str, frame: Frame) -> None:
    """Record a root version holding `frame` as its stored frame."""
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


def _analysed(stores: Stores, *trees: dict[str, Any]) -> None:
    for tree in trees:
        run_analysis(
            AnalysisRequest.model_validate({"tree": tree}),
            source=source(stores),
            stores=stores.analysis(),
        )


def _diff(stores: Stores) -> tuple[int, list[dict[str, Any]]]:
    report = io.StringIO()
    differs = run_diff_recipes(
        store=stores.store,
        artifact_store=stores.artifact_store,
        report=report,
        now=lambda: stores.now,
    )
    return differs, [json.loads(line) for line in report.getvalue().splitlines()]


def _counts(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    (line,) = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if "datasets_recipes_diffed" in line
    ]
    return line


def test_recipes_on_their_own_roots_are_equal(capsys: pytest.CaptureFixture[str]) -> None:
    stores = Stores()
    _store_root(stores, AID, AID_V1, aid_frame())
    _store_root(stores, POPULATION, POPULATION_V1, population_frame())
    _analysed(stores, _PER_PERSON, _COMPARED)

    differs, lines = _diff(stores)

    counts = _counts(capsys)
    assert differs == 0 and lines == []
    assert counts["equal"] == 2
    assert counts["families"]["compare"]["equal"] == 1
    assert counts["families"]["other"]["equal"] == 1


def test_a_root_whose_stored_frame_changed_differs(capsys: pytest.CaptureFixture[str]) -> None:
    stores = Stores()
    _store_root(stores, AID, AID_V1, aid_frame(scale=2.0))
    _store_root(stores, POPULATION, POPULATION_V1, population_frame())
    _analysed(stores, _PER_PERSON)

    differs, lines = _diff(stores)

    assert differs == 1
    (line,) = lines
    assert line["outcome"] == "differs" and line["family"] == "other"
    assert line["stored"]["output_digest"] != line["computed"]["output_digest"]
    assert _counts(capsys)["severity"] == "ERROR"


def test_a_root_no_longer_stored_is_unreadable(capsys: pytest.CaptureFixture[str]) -> None:
    stores = Stores()
    _store_root(stores, AID, AID_V1, aid_frame())
    _analysed(stores, _PER_PERSON)

    differs, lines = _diff(stores)

    assert differs == 0
    assert [line["outcome"] for line in lines] == ["roots_unreadable"]
    assert _counts(capsys)["roots_unreadable"] == 1


def test_a_stored_frame_past_retention_is_unreadable(capsys: pytest.CaptureFixture[str]) -> None:
    stores = Stores()
    _store_root(stores, AID, AID_V1, aid_frame())
    _store_root(stores, POPULATION, POPULATION_V1, population_frame())
    _analysed(stores, _PER_PERSON)
    stores.now += timedelta(days=31)

    differs, lines = _diff(stores)

    assert differs == 0
    assert [line["outcome"] for line in lines] == ["frame_unreadable"]
    assert _counts(capsys)["frame_unreadable"] == 1


def test_a_root_past_the_decode_bound_is_over_ceiling_and_compared_with_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A root stored as a frame1 document recorded at 250 MB refuses as it is located, before
    any root is read, so the replay compares nothing and differs from nothing."""
    stores = Stores()
    _store_root(stores, AID, AID_V1, aid_frame())
    _store_root(stores, POPULATION, POPULATION_V1, population_frame())
    _analysed(stores, _PER_PERSON)
    digest = link_frame1_only(
        stores.store, stores.artifact_store, subject_id=AID, content_version=AID_V1
    )
    with stores.store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_inventory SET size_bytes = %s WHERE artifact_digest = %s",
            (250_000_000, digest),
        )

    differs, lines = _diff(stores)

    assert differs == 0
    (line,) = lines
    assert (line["outcome"], line["computed"]) == ("over_ceiling", None)
    assert line["detail"].startswith("bytes: ")
    assert _counts(capsys)["severity"] == "INFO"


@pytest.mark.parametrize(
    ("bound", "outcome", "differs"),
    [
        pytest.param("root_rows", "over_ceiling", 0, id="root-rows-over-ceiling"),
        pytest.param("output_rows", "differs", 1, id="output-rows-differs"),
    ],
)
def test_a_row_cap_refusal_is_over_ceiling_unless_the_output_outgrew_the_stored_one(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    bound: str,
    outcome: str,
    differs: int,
) -> None:
    """Roots past the root-row cap refuse before anything is compared. An output past the rows
    the publish door stores differs, since the stored output was published under that cap."""
    stores = Stores()
    _store_root(stores, AID, AID_V1, aid_frame())
    _store_root(stores, POPULATION, POPULATION_V1, population_frame())
    _analysed(stores, _PER_PERSON)
    if bound == "root_rows":
        monkeypatch.setattr(
            recipe_replay,
            "compute_analysis",
            functools.partial(compute_analysis, bounds=AnalysisBounds(max_root_rows=1)),
        )
    else:
        monkeypatch.setattr(analysis, "PUBLISH_MAX_ROWS", 1)

    found, lines = _diff(stores)

    assert found == differs
    (line,) = lines
    assert (line["outcome"], line["computed"]) == (outcome, None)
    assert line["detail"].startswith(f"analysis_{bound}: ")
    assert _counts(capsys)[outcome] == 1


def _objects(stores: Stores) -> dict[str, dict[str, bytes]]:
    return {name: dict(bucket.objects) for name, bucket in stores.gcs.buckets.items()}


def test_a_diff_writes_to_no_store(capsys: pytest.CaptureFixture[str]) -> None:
    """A diff that reads changed and missing roots changes no database row and no object."""
    stores = Stores()
    _store_root(stores, AID, AID_V1, aid_frame(scale=2.0))
    _analysed(stores, _PER_PERSON, _SUMMED)
    changes, objects = stores.store.total_changes, _objects(stores)

    differs, lines = _diff(stores)

    assert differs == 1
    assert sorted(line["outcome"] for line in lines) == ["differs", "roots_unreadable"]
    assert stores.store.total_changes == changes
    assert _objects(stores) == objects


def test_the_command_exits_non_zero_when_a_recipe_differs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    seen: dict[str, Any] = {}

    def run(task: str, job: Any) -> int:
        seen["task"] = task
        seen["report"] = job.keywords["report"].name
        return 1

    monkeypatch.setattr("datasets_api.maintenance._run_over_artifacts", run)
    report = tmp_path / "diff.jsonl"

    assert main(["diff-recipes", "--report", str(report)]) == 1
    assert seen == {"task": "diff-recipes", "report": str(report)}


def _roots_swapped(node: Any) -> Any:
    """`node` with root indices 0 and 1 swapped."""
    if isinstance(node, dict):
        if "root" in node:
            return {**node, "root": 1 - node["root"]}
        return {key: _roots_swapped(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_roots_swapped(value) for value in node]
    return node


def test_a_recipe_storing_its_roots_in_another_order_is_equal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The run numbers its roots in its own order, so a stored root is matched to the root the
    run read by its record, not by its index."""
    stores = Stores()
    _store_root(stores, AID, AID_V1, aid_frame())
    _store_root(stores, POPULATION, POPULATION_V1, population_frame())
    _analysed(stores, _PER_PERSON)
    with stores.store.transaction() as tx:
        row = tx.fetch_one("SELECT recipe FROM datasets.frame_recipe", ())
        assert row is not None
        recipe = dict(row["recipe"])  # ty: ignore[no-matching-overload] -- the fake decodes JSON
        recipe["roots"].reverse()
        recipe["tree"] = _roots_swapped(recipe["tree"])
        tx.execute("UPDATE datasets.frame_recipe SET recipe = %s", (json.dumps(recipe),))

    differs, lines = _diff(stores)

    assert lines == []
    assert differs == 0
    assert _counts(capsys)["equal"] == 1
