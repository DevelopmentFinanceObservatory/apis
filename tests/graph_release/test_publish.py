"""Pins the publisher: create-only parts and manifests, the pointer
precondition, `hold`, and pruning that never deletes a part a kept manifest
names. Each bucket test runs on `file://` and on the fake GCS client."""

from __future__ import annotations

import dataclasses
import gzip
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

import graph_release.publish
from artifact_publication.jobs import acquire_lease, release_lease
from artifact_publication.uri import (
    list_uri_object_times,
    read_uri_bytes,
    read_uri_bytes_and_generation,
    write_uri_bytes,
)
from graph_release.format import (
    Manifest,
    Pointer,
    StagedPart,
    manifest_uri,
    open_release,
    part_uri,
    pointer_uri,
    read_pointer,
    write_blob_part,
)
from graph_release.hold import LEASE_TTL_SECONDS, WRITE_MARGIN_SECONDS
from graph_release.hold import main as hold_main
from graph_release.publish import (
    KEEP_RELEASES,
    KEEP_SECONDS,
    HoldError,
    PartWriteError,
    PublishStatus,
    clear_hold,
    hold_release,
    prune_releases,
    publish_release,
)
from tests.artifact_publication.fixtures import gs_bucket  # noqa: F401
from tests.artifact_publication.support import BUCKET, FakeBlob
from tests.artifact_publication.test_jobs import _conditional_delete
from tests.graph_release.support import build_test_release, sample_tables


@dataclass
class Bucket:
    """A release root and a way to backdate an object under it."""

    root: str
    backdate: Callable[[str, float], None]


@pytest.fixture(params=["file", "gs"])
def bucket(request, tmp_path, monkeypatch) -> Bucket:
    if request.param == "file":
        root_dir = tmp_path / "releases"

        def backdate_file(relative: str, written: float) -> None:
            os.utime(root_dir / relative, (written, written))

        return Bucket(root=root_dir.as_uri(), backdate=backdate_file)

    objects = request.getfixturevalue("gs_bucket")
    monkeypatch.setattr(FakeBlob, "delete", _conditional_delete)

    def backdate_gs(relative: str, written: float) -> None:
        objects[f"releases/{relative}"].time_created = datetime.fromtimestamp(written, UTC)

    return Bucket(root=f"gs://{BUCKET}/releases", backdate=backdate_gs)


def _release(
    staging: Path, *, revision: str, blocks: object = "blocks"
) -> tuple[list[StagedPart], Manifest]:
    """A release whose `declarations` part every release shares and whose
    `code_index_blocks` part is `blocks`'s."""
    staged = [
        write_blob_part(
            data, staging_dir=staging, name=name, source="test", produced_by_execution="exec"
        )
        for name, data in (
            ("declarations", b"{}"),
            ("code_index_blocks", gzip.compress(json.dumps(blocks).encode(), mtime=0)),
        )
    ]
    manifest = Manifest(
        import_execution_id="exec",
        import_completed_at="2026-10-18T23:50:00+00:00",
        ingested_imports=("TestImport",),
        read_timestamp="2026-10-19T00:00:00+00:00",
        dc_serving_revision=revision,
        builder_code_hash="code",
        declarations_hash="declarations",
        parts=tuple(part.ref for part in staged),
    )
    return staged, manifest


def _generation(root: str) -> str | None:
    found = read_uri_bytes_and_generation(pointer_uri(root))
    return found[1] if found is not None else None


def _publish(root: str, staged: list[StagedPart], manifest: Manifest) -> PublishStatus:
    return publish_release(
        root, staged=staged, manifest=manifest, expected_pointer_generation=_generation(root)
    ).status


def test_a_publish_uploads_only_missing_parts_and_moves_the_pointer(bucket, tmp_path):
    staged, first = _release(tmp_path / "a", revision="rev-1", blocks="one")

    outcome = publish_release(
        bucket.root, staged=staged, manifest=first, expected_pointer_generation=None
    )

    assert outcome.status is PublishStatus.PUBLISHED
    assert outcome.release_id == first.release_id
    assert sorted(outcome.uploaded) == sorted(part.ref.file_name for part in staged)
    assert read_pointer(bucket.root) == Pointer(release_id=first.release_id)
    assert Manifest.from_json(read_uri_bytes(manifest_uri(bucket.root, first.release_id))) == first

    # A redeploy with no import: same parts, new revision, nothing uploaded.
    staged, second = _release(tmp_path / "b", revision="rev-2", blocks="one")
    outcome = publish_release(
        bucket.root,
        staged=staged,
        manifest=second,
        expected_pointer_generation=_generation(bucket.root),
    )

    assert outcome.status is PublishStatus.PUBLISHED
    assert outcome.uploaded == ()
    assert read_pointer(bucket.root) == Pointer(release_id=second.release_id)


def test_a_published_release_opens_with_its_tables_and_blobs(tmp_path):
    root = (tmp_path / "releases").as_uri()
    facets = sample_tables()["facets"]

    manifest = build_test_release(
        root, tables={"facets": facets}, blobs={"declarations": b'{"declarations": 1}'}
    )

    assert read_pointer(root) == Pointer(release_id=manifest.release_id)
    loaded = open_release(root, local_dir=tmp_path / "local")
    assert loaded.manifest == manifest
    assert loaded.tables["facets"]["facet_id"].to_pylist() == [f"f{n}" for n in range(10)]
    assert json.loads(loaded.blobs["declarations"]) == {"declarations": 1}


def test_an_existing_part_is_never_overwritten(bucket, tmp_path):
    staged, manifest = _release(tmp_path, revision="rev")
    target = part_uri(bucket.root, staged[0].ref)
    _publish(bucket.root, *_release(tmp_path / "seed", revision="seed", blocks="seed"))
    write_uri_bytes(target, b"written by another run")

    outcome = publish_release(
        bucket.root,
        staged=staged,
        manifest=manifest,
        expected_pointer_generation=_generation(bucket.root),
    )

    assert outcome.status is PublishStatus.PUBLISHED
    assert staged[0].ref.file_name not in outcome.uploaded
    assert read_uri_bytes(target) == b"written by another run"


def test_a_part_write_that_does_not_land_writes_no_manifest_or_pointer(
    bucket, tmp_path, monkeypatch
):
    _publish(bucket.root, *_release(tmp_path / "seed", revision="seed", blocks="seed"))
    before = read_uri_bytes(pointer_uri(bucket.root))
    staged, manifest = _release(tmp_path / "new", revision="rev", blocks="new")
    # Another writer lands bytes under a part name after the listing.
    monkeypatch.setattr(graph_release.publish, "list_uri_object_times", lambda _uri: {})
    write_uri_bytes(part_uri(bucket.root, staged[0].ref), b"written by another run")

    with pytest.raises(PartWriteError):
        _publish(bucket.root, staged, manifest)

    assert read_uri_bytes(pointer_uri(bucket.root)) == before
    assert read_uri_bytes_and_generation(manifest_uri(bucket.root, manifest.release_id)) is None


def test_an_existing_manifest_under_the_same_id_is_kept_and_pointed_at(bucket, tmp_path):
    staged, manifest = _release(tmp_path, revision="rev")
    earlier = dataclasses.replace(manifest, read_timestamp="2026-10-18T00:00:00+00:00")
    assert earlier.release_id == manifest.release_id
    assert _publish(bucket.root, staged, earlier) is PublishStatus.PUBLISHED
    _publish(bucket.root, *_release(tmp_path / "other", revision="other", blocks="other"))

    assert _publish(bucket.root, staged, manifest) is PublishStatus.PUBLISHED

    assert read_pointer(bucket.root) == Pointer(release_id=manifest.release_id)
    stored = read_uri_bytes(manifest_uri(bucket.root, manifest.release_id))
    assert Manifest.from_json(stored).read_timestamp == "2026-10-18T00:00:00+00:00"


@pytest.mark.parametrize("expected", ["stale", None])
def test_a_moved_pointer_is_left_alone_and_nothing_is_uploaded(bucket, tmp_path, expected):
    _publish(bucket.root, *_release(tmp_path / "a", revision="rev-1", blocks="one"))
    before = read_uri_bytes(pointer_uri(bucket.root))
    staged, manifest = _release(tmp_path / "b", revision="rev-2", blocks="two")

    outcome = publish_release(
        bucket.root, staged=staged, manifest=manifest, expected_pointer_generation=expected
    )

    assert outcome.status is PublishStatus.POINTER_MOVED
    assert outcome.uploaded == ()
    assert read_uri_bytes(pointer_uri(bucket.root)) == before
    assert read_uri_bytes_and_generation(manifest_uri(bucket.root, manifest.release_id)) is None


def test_a_held_pointer_refuses_a_publish_until_cleared(bucket, tmp_path):
    staged, first = _release(tmp_path / "a", revision="rev-1", blocks="one")
    _publish(bucket.root, staged, first)
    assert hold_release(bucket.root, first.release_id) == Pointer(
        release_id=first.release_id, hold=first.release_id
    )
    staged, second = _release(tmp_path / "b", revision="rev-2", blocks="two")

    outcome = publish_release(
        bucket.root,
        staged=staged,
        manifest=second,
        expected_pointer_generation=_generation(bucket.root),
    )

    assert outcome.status is PublishStatus.HELD
    assert outcome.uploaded == ()
    assert read_pointer(bucket.root) == Pointer(release_id=first.release_id, hold=first.release_id)

    assert clear_hold(bucket.root) == Pointer(release_id=first.release_id)
    assert _publish(bucket.root, staged, second) is PublishStatus.PUBLISHED
    assert read_pointer(bucket.root) == Pointer(release_id=second.release_id)


def test_a_hold_names_only_a_published_release(bucket, tmp_path):
    with pytest.raises(HoldError, match="no manifest"):
        hold_release(bucket.root, "0" * 64)
    with pytest.raises(HoldError, match="no release pointer"):
        clear_hold(bucket.root)
    assert read_pointer(bucket.root) is None


def test_staged_parts_must_match_the_manifest(tmp_path):
    staged, manifest = _release(tmp_path, revision="rev")

    with pytest.raises(ValueError, match="1 named part"):
        publish_release(
            (tmp_path / "releases").as_uri(),
            staged=staged[:1],
            manifest=manifest,
            expected_pointer_generation=None,
        )


@pytest.mark.parametrize("by_env", [False, True])
def test_the_hold_cli_sets_and_clears_the_hold(tmp_path, monkeypatch, capsys, by_env):
    releases = tmp_path / "releases"
    staged, manifest = _release(tmp_path / "staging", revision="rev")
    _publish(releases.as_uri(), staged, manifest)
    if by_env:
        monkeypatch.setenv("GRAPH_RELEASE_URI", releases.as_uri())
        target: list[str] = []
    else:
        monkeypatch.delenv("GRAPH_RELEASE_URI", raising=False)
        target = ["--local-dir", str(releases)]

    assert hold_main([*target, "set", manifest.release_id]) == 0
    assert read_pointer(releases.as_uri()) == Pointer(manifest.release_id, manifest.release_id)
    assert json.loads(capsys.readouterr().out)["event"] == "graph_release_hold"

    assert hold_main([*target, "clear"]) == 0
    assert read_pointer(releases.as_uri()) == Pointer(manifest.release_id)

    assert hold_main([*target, "set", "0" * 64]) == 1
    assert read_pointer(releases.as_uri()) == Pointer(manifest.release_id)

    # A running build holds the lease, so pruning may be under way.
    lease = acquire_lease(releases.as_uri(), build_id="build", ttl_seconds=60)
    assert lease is not None
    capsys.readouterr()
    assert hold_main([*target, "set", manifest.release_id]) == 1
    assert "a build holds the lease" in capsys.readouterr().err
    assert read_pointer(releases.as_uri()) == Pointer(manifest.release_id)
    assert release_lease(releases.as_uri(), lease)


@pytest.mark.parametrize(
    ("command", "elapsed", "status"),
    [
        ("set", LEASE_TTL_SECONDS - WRITE_MARGIN_SECONDS, 0),
        ("set", LEASE_TTL_SECONDS - WRITE_MARGIN_SECONDS + 1, 1),
        ("clear", LEASE_TTL_SECONDS - WRITE_MARGIN_SECONDS + 1, 1),
    ],
    ids=["set-at-the-deadline", "set-past-the-deadline", "clear-past-the-deadline"],
)
def test_the_hold_cli_writes_nothing_once_its_lease_is_too_short_for_the_write(
    tmp_path, capsys, command, elapsed, status
):
    releases = tmp_path / "releases"
    staged, first = _release(tmp_path / "a", revision="rev-1", blocks="one")
    _publish(releases.as_uri(), staged, first)
    staged, second = _release(tmp_path / "b", revision="rev-2", blocks="two")
    _publish(releases.as_uri(), staged, second)
    if command == "clear":
        hold_release(releases.as_uri(), second.release_id)
    before = read_pointer(releases.as_uri())
    # The first reading starts the lease; every later one is `elapsed` on.
    readings = iter([0.0])

    def clock() -> float:
        return next(readings, elapsed)

    args = ["set", first.release_id] if command == "set" else ["clear"]
    assert hold_main(["--local-dir", str(releases), *args], clock=clock) == status

    if status:
        assert read_pointer(releases.as_uri()) == before
        assert "deadline" in capsys.readouterr().err
    else:
        assert read_pointer(releases.as_uri()) == Pointer(first.release_id, first.release_id)
    # The lease is released either way, so the next build is not kept waiting.
    lease = acquire_lease(releases.as_uri(), build_id="build", ttl_seconds=60)
    assert lease is not None
    assert release_lease(releases.as_uri(), lease)


def _objects(root: str) -> set[str]:
    return set(list_uri_object_times(root))


def test_pruning_keeps_the_newest_ten_and_the_held_release_and_their_parts(bucket, tmp_path):
    now = time.time()
    old = now - KEEP_SECONDS - 3600
    manifests = []
    for index in range(KEEP_RELEASES + 4):
        staged, manifest = _release(tmp_path / str(index), revision=f"rev-{index}", blocks=index)
        assert _publish(bucket.root, staged, manifest) is PublishStatus.PUBLISHED
        manifests.append(manifest)
    for index, manifest in enumerate(manifests):
        written = old + index if index < len(manifests) - 1 else now
        bucket.backdate(f"manifests/{manifest.release_id}.json", written)
        for ref in manifest.parts:
            bucket.backdate(f"parts/{ref.file_name}", old)
    # The newest ten are the young manifest and releases 4 to 12. Release 0
    # stays because it is held, and 1 to 3 go with the parts only they name.
    hold_release(bucket.root, manifests[0].release_id)
    stray = part_uri(bucket.root, _release(tmp_path / "stray", revision="x", blocks="x")[0][1].ref)
    write_uri_bytes(stray, b"stray")
    bucket.backdate(stray.removeprefix(f"{bucket.root}/"), old)

    deleted = prune_releases(bucket.root, now=now)

    kept = {0, *range(len(manifests) - KEEP_RELEASES, len(manifests))}
    dropped = [manifests[index] for index in range(len(manifests)) if index not in kept]
    assert deleted == tuple(
        sorted(
            [f"manifests/{manifest.release_id}.json" for manifest in dropped]
            + [f"parts/{manifest.parts[0].file_name}" for manifest in dropped]
            + [stray.removeprefix(f"{bucket.root}/")]
        )
    )
    remaining = _objects(bucket.root)
    for index in kept:
        assert f"manifests/{manifests[index].release_id}.json" in remaining
        for ref in manifests[index].parts:
            assert f"parts/{ref.file_name}" in remaining
    assert prune_releases(bucket.root, now=now) == ()


def test_pruning_keeps_every_manifest_younger_than_a_day(bucket, tmp_path):
    for index in range(KEEP_RELEASES + 2):
        staged, manifest = _release(tmp_path / str(index), revision=f"rev-{index}", blocks=index)
        assert _publish(bucket.root, staged, manifest) is PublishStatus.PUBLISHED
    before = _objects(bucket.root)

    assert prune_releases(bucket.root, now=time.time()) == ()
    assert _objects(bucket.root) == before


def test_pruning_deletes_nothing_without_a_readable_pointer_and_manifests(bucket, tmp_path):
    staged, manifest = _release(tmp_path / "a", revision="rev", blocks="a")
    assert prune_releases(bucket.root, now=time.time()) == ()

    _publish(bucket.root, staged, manifest)
    stray = f"{bucket.root}/parts/{'f' * 64}.json"
    write_uri_bytes(stray, b"stray")
    write_uri_bytes(f"{bucket.root}/manifests/{'e' * 64}.json", b'{"format": 2}')
    before = _objects(bucket.root)

    assert prune_releases(bucket.root, now=time.time() + 2 * KEEP_SECONDS) == ()
    assert _objects(bucket.root) == before

    write_uri_bytes(pointer_uri(bucket.root), b"{")
    assert prune_releases(bucket.root, now=time.time() + 2 * KEEP_SECONDS) == ()
    assert _objects(bucket.root) == before
