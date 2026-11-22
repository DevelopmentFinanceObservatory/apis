"""`maintenance diff-recipes`: each stored recipe computed again on its own root versions, read
from live store SQL and an in-memory bucket, compared with its stored frame, and nothing
written."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import io
import json
from datetime import timedelta
from typing import Any

import pyarrow.parquet as pq
import pytest
from dfo_frame_grammar import Frame

from artifact_store import Tier
from datasets_api import analysis, recipe_replay
from datasets_api.analysis import (
    AnalysisBounds,
    AnalysisRequest,
    AnalysisResult,
    DerivedRoot,
    FrameUnavailable,
    RootExpired,
    RootRenderingUnreadable,
    compute_analysis,
    run_analysis,
)
from datasets_api.analysis_roots import frame_from_frame_layout, frame_ref_under
from datasets_api.doors.publish import publish_built_frame
from datasets_api.maintenance import main, run_diff_recipes
from datasets_api.mint_models import MintOptions
from datasets_api.recipe_replay import StoredRootSource
from datasets_api.render import STATISTICS_MEDIA_TYPE, artifact_rendering
from datasets_api.version_store import LiveVersionRecordWriter, rendering_key
from datasets_api.versions import RENDER_FORMAT, ArtifactRecord
from materialise.frame_build import build_frame

from ._analysis_doubles import (
    AID,
    AID_V1,
    POPULATION,
    POPULATION_V1,
    FakeRootSource,
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


_PURPOSE = "stat_var_crsPurpose"
_LABEL_CELL, _DCID_CELL = "Basic education", "dcid:CRS_11220"
_PURPOSE_ROOT = {"spec_id": AID, "content_version": AID_V1, "rendering": {"columns": [_PURPOSE]}}


def _purpose_file(cell: str) -> bytes:
    """A frame-layout file of two recipients, Kenya's purpose cell holding `cell`."""
    rows = [
        {
            "stat_var": "dcid:Amount_ODA",
            "financingRecipient": recipient,
            _PURPOSE: purpose,
            "time_period": "2022",
            "value": value,
            "unit": "USDollar",
        }
        for recipient, purpose, value in (("country/KEN", cell, 10.0), ("country/NGA", "x", 20.0))
    ]
    sink = io.BytesIO()
    pq.write_table(build_frame(role_keys=["financingRecipient", _PURPOSE], rows=rows).table, sink)
    return sink.getvalue()


def _link_purpose_file(stores: Stores, cell: str, *, render_format: str | None) -> str:
    """Link the file of `cell` to the aid version under the key of `render_format`, first seen
    one hour after the last file, so a later link is the newer file, and return its digest."""
    stores.now += timedelta(hours=1)
    stored = stores.artifact_store.put(_purpose_file(cell), tier=Tier.working)
    rendering = artifact_rendering(
        MintOptions.model_validate(
            {"format": "parquet", "bundle": False, "layout": "frame", "columns": [_PURPOSE]}
        ),
        site=SITE,
    )
    LiveVersionRecordWriter(store=stores.store, now=lambda: stores.now).record_artifact(
        subject_id=AID,
        content_version=AID_V1,
        artifact=ArtifactRecord(
            artifact_digest=stored.digest, rendering=rendering, media_type=STATISTICS_MEDIA_TYPE
        ),
    )
    with stores.store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.version_artifact SET rendering_key = %s WHERE artifact_digest = %s",
            (_link_key(rendering, render_format=render_format), stored.digest),
        )
        tx.execute(
            "UPDATE datasets.artifact_inventory SET created_at = %s WHERE artifact_digest = %s",
            (stores.now, stored.digest),
        )
    return stored.digest


def _link_key(rendering: dict[str, object], *, render_format: str | None) -> str:
    """The key `rendering` is linked under today, or for `None` the key a frame-layout file was
    linked under before its key carried a render format: the SHA-256 of its sorted JSON."""
    if render_format is None:
        return hashlib.sha256(json.dumps(rendering, sort_keys=True).encode()).hexdigest()
    assert render_format == RENDER_FORMAT
    return rendering_key(rendering)


class _FormatNamedSource(FakeRootSource):
    """`FakeRootSource` naming a root with columns as `analysis_roots.frame_ref_under` names it
    under `render_format`, or, given `artifact_digest`, as data-mcp names it by the digest of
    the file it read, recording the digest."""

    def __init__(
        self, *, render_format: str | None, artifact_digest: str | None = None, **held: Any
    ) -> None:
        super().__init__(**held)
        self.render_format = render_format
        self.artifact_digest = artifact_digest

    def resolve(self, root, *, latest, trace_id, path_words=None):
        resolved = super().resolve(root, latest=latest, trace_id=trace_id, path_words=path_words)
        options = resolved.record["options"]
        frame = frame_ref_under(
            spec_id=resolved.dataset.spec_id,
            content_version=resolved.dataset.content_version,
            columns=options.get("columns") or (),
            groupby=None,
            without_party_totals=False,
            enrichment_stamp=None,
            render_format=self.render_format,
        )
        record = {**resolved.record, "frame": frame.model_dump()}
        if self.artifact_digest is not None:
            preimage = f"{frame.content_version}:{self.artifact_digest}".encode()
            frame = frame.model_copy(
                update={"content_version": f"cv_{hashlib.sha256(preimage).hexdigest()[:32]}"}
            )
            record = {
                **record,
                "frame": frame.model_dump(),
                "artifact_digest": self.artifact_digest,
            }
        return dataclasses.replace(resolved, frame=frame, record=record)


def _purpose_recipe(
    stores: Stores, cell: str, *, render_format: str | None, artifact_digest: str | None = None
) -> AnalysisResult:
    """Store the recipe of a filter keeping the purpose `cell` over the aid root read with its
    purpose column, as a run named under `render_format`, or by `artifact_digest`, reading the
    file of `cell` stores it."""
    held = _FormatNamedSource(
        render_format=render_format,
        artifact_digest=artifact_digest,
        frames={(AID, AID_V1): frame_from_frame_layout(_purpose_file(cell))},
        newest={AID: AID_V1},
        recipes=stores.recipe_store,
    )
    keep = {
        "verb": "filter",
        "arguments": {
            "predicates": [{"kind": "category_match", "column": _PURPOSE, "values": [cell]}]
        },
        "inputs": {"data": _PURPOSE_ROOT},
    }
    return run_analysis(
        AnalysisRequest.model_validate({"tree": keep}), source=held, stores=stores.analysis()
    )


@pytest.mark.parametrize(
    ("cell", "render_format"),
    [
        pytest.param(_LABEL_CELL, None, id="label-cells-before-the-format"),
        pytest.param(_DCID_CELL, RENDER_FORMAT, id="dcid-cells-under-the-format"),
    ],
)
def test_a_root_reads_the_file_of_the_format_its_frame_names_with_both_held(
    capsys: pytest.CaptureFixture[str], cell: str, render_format: str | None
) -> None:
    """With a label-cell file linked before frame-layout keys carried a format and a newer
    dcid-cell file linked under `RENDER_FORMAT`, each recipe reads the file its frame pair
    names, so its predicate keeps the row it kept when it was stored."""
    stores = Stores()
    _store_root(stores, AID, AID_V1, aid_frame())
    _link_purpose_file(stores, _LABEL_CELL, render_format=None)
    _link_purpose_file(stores, _DCID_CELL, render_format=RENDER_FORMAT)
    stored = _purpose_recipe(stores, cell, render_format=render_format)

    differs, lines = _diff(stores)

    assert stored.evaluation.summary["rows"] == 1
    assert (differs, lines) == (0, [])
    assert _counts(capsys)["equal"] == 1


@pytest.mark.parametrize(
    ("render_format", "unreadable"),
    [
        pytest.param(None, RootExpired, id="its-format-s-file-gone"),
        pytest.param("another_format", RootRenderingUnreadable, id="a-format-not-read"),
    ],
)
def test_a_root_whose_format_s_file_is_not_held_is_unreadable(
    capsys: pytest.CaptureFixture[str],
    render_format: str | None,
    unreadable: type[FrameUnavailable],
) -> None:
    """A recipe whose frame pair names a format with no file held, or a format this build does
    not read, is unreadable, though a file of its rendering in `RENDER_FORMAT` is held."""
    stores = Stores()
    _store_root(stores, AID, AID_V1, aid_frame())
    _link_purpose_file(stores, _DCID_CELL, render_format=RENDER_FORMAT)
    stored = _purpose_recipe(stores, _LABEL_CELL, render_format=render_format).evaluation.ref

    differs, lines = _diff(stores)

    assert differs == 0
    assert [line["outcome"] for line in lines] == ["roots_unreadable"]
    assert _counts(capsys)["roots_unreadable"] == 1
    replay = StoredRootSource(
        snapshot_reader=stores.snapshot_reader,
        artifact_store=stores.artifact_store,
        recipes=stores.recipe_store,
        now=lambda: stores.now,
        fold_max_rows=1_000_000,
    )
    tree = DerivedRoot(spec_id=stored.spec_id, content_version=stored.content_version)
    with pytest.raises(unreadable):
        compute_analysis(AnalysisRequest(tree=tree), source=replay)


@pytest.mark.parametrize("cell", [_LABEL_CELL, _DCID_CELL])
def test_a_digest_named_root_reads_the_file_of_its_digest_with_both_held(
    capsys: pytest.CaptureFixture[str], cell: str
) -> None:
    """A root record carrying `artifact_digest`, its frame named by that digest as data-mcp
    names it, reads the file holding those bytes, the older or the newer, so its predicate keeps
    the row it kept when it was stored."""
    stores = Stores()
    _store_root(stores, AID, AID_V1, aid_frame())
    digests = {
        _LABEL_CELL: _link_purpose_file(stores, _LABEL_CELL, render_format=None),
        _DCID_CELL: _link_purpose_file(stores, _DCID_CELL, render_format=RENDER_FORMAT),
    }
    stored = _purpose_recipe(stores, cell, render_format=None, artifact_digest=digests[cell])

    differs, lines = _diff(stores)

    assert stored.evaluation.summary["rows"] == 1
    assert (differs, lines) == (0, [])
    assert _counts(capsys)["equal"] == 1


def test_a_digest_named_root_whose_file_is_gone_is_unreadable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A root record whose `artifact_digest` names a file the version no longer links is
    unreadable as expired, though a file of its rendering is held."""
    stores = Stores()
    _store_root(stores, AID, AID_V1, aid_frame())
    _link_purpose_file(stores, _DCID_CELL, render_format=RENDER_FORMAT)
    gone = stores.artifact_store.put(_purpose_file(_LABEL_CELL), tier=Tier.working).digest
    stored = _purpose_recipe(
        stores, _LABEL_CELL, render_format=None, artifact_digest=gone
    ).evaluation.ref

    differs, lines = _diff(stores)

    assert differs == 0
    assert [line["outcome"] for line in lines] == ["roots_unreadable"]
    assert _counts(capsys)["roots_unreadable"] == 1
    replay = StoredRootSource(
        snapshot_reader=stores.snapshot_reader,
        artifact_store=stores.artifact_store,
        recipes=stores.recipe_store,
        now=lambda: stores.now,
        fold_max_rows=1_000_000,
    )
    tree = DerivedRoot(spec_id=stored.spec_id, content_version=stored.content_version)
    with pytest.raises(RootExpired):
        compute_analysis(AnalysisRequest(tree=tree), source=replay)
