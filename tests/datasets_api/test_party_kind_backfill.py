"""`datasets-maintenance backfill-party-kind-coverage`
(`maintenance.run_backfill_party_kind_coverage`) over a version whose output profile was
counted under the earlier rule, which read each party for no dataset."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any

from code_index import CodeIndex
from datasets_api import maintenance
from datasets_api.maintenance import run_backfill_party_kind_coverage

from . import test_serve_party_totals as party_totals
from ._artifact_doubles import FakeArtifactStore
from .test_version_file_selector import _PartyRig

_OTHER_DATASET = "provenance/OtherTable"


def _france_differs_by_dataset() -> CodeIndex:
    """The party-totals fixture index with France a total in another dataset that publishes
    it as a provider, so its kind read for no dataset is none, while each row of the frame,
    all of OECD DAC2a, reads it as a party."""
    index = party_totals._code_index()
    return CodeIndex(
        stamp=index.stamp,
        declarations=(),
        codes=(),
        statvars=index.statvars,
        parties=[
            replace(
                row,
                published=(*row.published, (_OTHER_DATASET, "origin")),
                dataset_kinds=((_OTHER_DATASET, "total"),),
            )
            if row.dcid == party_totals._FRANCE
            else row
            for row in index.parties.values()
        ],
    )


def _profiles(version_store, manifest: dict[str, Any]) -> tuple[Any, Any]:
    """The `output_profile` column and the body's `output_profile` of `manifest`'s row."""
    with version_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT output_profile, immutable_body FROM datasets.version_record "
            "WHERE content_version = %s",
            (manifest["content_version"],),
        )
    assert row is not None
    return row["output_profile"], row["immutable_body"].get("output_profile")


def _counted_under_the_earlier_rule(version_store, manifest: dict[str, Any]) -> dict[str, Any]:
    """Write the profile the earlier rule counted to `manifest`'s row and return it: France,
    whose kind differs between the datasets that publish it as a provider, missing."""
    profile, _ = _profiles(version_store, manifest)
    earlier = json.loads(json.dumps(profile))
    [entry] = [
        entry
        for entry in earlier["enrichments"]
        if entry["column"] == "provider" and entry["field"] == "party_kind"
    ]
    entry["covered"] -= 1
    entry["missing"] = [[party_totals._FRANCE, ""]]
    with version_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT immutable_body FROM datasets.version_record WHERE content_version = %s",
            (manifest["content_version"],),
        )
        assert row is not None
        body = dict(row["immutable_body"])
        body["output_profile"] = earlier
        tx.execute(
            "UPDATE datasets.version_record SET output_profile = %s, immutable_body = %s "
            "WHERE content_version = %s",
            (json.dumps(earlier), json.dumps(body), manifest["content_version"]),
        )
    return earlier


def _backfilled(
    rig: _PartyRig, index: CodeIndex, *, dry_run: bool, capsys, artifact_store=None, failed=0
) -> dict[str, Any]:
    capsys.readouterr()
    assert (
        run_backfill_party_kind_coverage(
            store=rig.version_store,
            artifact_store=artifact_store or rig.artifact_store,
            parties=index.parties,
            dry_run=dry_run,
        )
        == failed
    )
    [counts] = [
        line
        for line in map(json.loads, capsys.readouterr().out.splitlines())
        if line.get("event") == "datasets_party_kind_backfilled"
    ]
    assert counts["failed"] == failed
    return counts


@dataclass(frozen=True)
class _Minted:
    rig: _PartyRig
    index: CodeIndex
    manifest: dict[str, Any]
    minted: dict[str, Any]
    earlier: dict[str, Any]


def _minted_then_counted_under_the_earlier_rule() -> _Minted:
    """A version minted today whose stored profile is then rewritten as the earlier rule
    counted it. The minted profile covers both providers."""
    index = _france_differs_by_dataset()
    rig = _PartyRig()
    spec_id, sidecar = party_totals._spec_id_and_sidecar()
    manifest = party_totals._mint(
        rig.client(index), spec_id=spec_id, sidecar=sidecar, options={"format": "csv"}
    )
    minted, _ = _profiles(rig.version_store, manifest)
    assert [
        (entry["covered"], entry["entities"], entry["missing"])
        for entry in minted["enrichments"]
        if entry["column"] == "provider" and entry["field"] == "party_kind"
    ] == [(2, 2, [])]
    earlier = _counted_under_the_earlier_rule(rig.version_store, manifest)
    return _Minted(rig=rig, index=index, manifest=manifest, minted=minted, earlier=earlier)


def test_a_profile_counted_under_the_earlier_rule_is_counted_again_per_row(capsys):
    """A dry run counts the row it would rewrite and writes nothing. The run writes the count a
    mint makes today to the profile column and the body alike, and a second run finds nothing
    to change."""
    version = _minted_then_counted_under_the_earlier_rule()
    rig, index, manifest = version.rig, version.index, version.manifest

    dry = _backfilled(rig, index, dry_run=True, capsys=capsys)
    assert (dry["refreshed"], dry["unchanged"]) == (1, 0)
    assert _profiles(rig.version_store, manifest) == (version.earlier, version.earlier)

    run = _backfilled(rig, index, dry_run=False, capsys=capsys)
    assert run["refreshed"] == 1
    assert _profiles(rig.version_store, manifest) == (version.minted, version.minted)

    again = _backfilled(rig, index, dry_run=False, capsys=capsys)
    assert (again["refreshed"], again["unchanged"]) == (0, 1)


def _set_body_key(tx, manifest: dict[str, Any], key: str, value: object) -> None:
    """Another writer's change to one key of `manifest`'s body."""
    row = tx.fetch_one(
        "SELECT immutable_body FROM datasets.version_record WHERE content_version = %s",
        (manifest["content_version"],),
    )
    assert row is not None
    body = {**row["immutable_body"], key: value}
    tx.execute(
        "UPDATE datasets.version_record SET immutable_body = %s WHERE content_version = %s",
        (json.dumps(body), manifest["content_version"]),
    )


def test_a_body_write_another_writer_lands_after_the_backfill_reads_the_row_is_kept(
    monkeypatch, capsys
):
    """The backfill sets the body's `output_profile` alone, so a change another writer makes
    to the body between the backfill's read of the row and its write stays."""
    version = _minted_then_counted_under_the_earlier_rule()
    stored_profile = maintenance._stored_profile
    reads: list[None] = []

    def read_then_another_writer_lands(tx, subject_id, content_version):
        profile = stored_profile(tx, subject_id, content_version)
        reads.append(None)
        if len(reads) == 2:
            _set_body_key(tx, version.manifest, "other_writer", "kept")
        return profile

    monkeypatch.setattr(maintenance, "_stored_profile", read_then_another_writer_lands)

    run = _backfilled(version.rig, version.index, dry_run=False, capsys=capsys)

    assert run["refreshed"] == 1
    with version.rig.version_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT output_profile, immutable_body FROM datasets.version_record "
            "WHERE content_version = %s",
            (version.manifest["content_version"],),
        )
    assert row is not None
    assert row["output_profile"] == version.minted
    assert row["immutable_body"]["output_profile"] == version.minted
    assert row["immutable_body"]["other_writer"] == "kept"


def test_a_profile_changed_after_it_was_counted_is_left_to_the_writer_that_changed_it(
    monkeypatch, capsys
):
    """A profile that changes between the backfill's count and its write is counted
    `changed_under_run` and keeps the other writer's profile."""
    version = _minted_then_counted_under_the_earlier_rule()
    recount = maintenance.with_party_kind_coverage
    changed = {**version.earlier, "rows": version.earlier["rows"] + 1}

    def recount_then_another_writer_lands(*args, **kwargs):
        refreshed = recount(*args, **kwargs)
        with version.rig.version_store.transaction() as tx:
            tx.execute(
                "UPDATE datasets.version_record SET output_profile = %s WHERE content_version = %s",
                (json.dumps(changed), version.manifest["content_version"]),
            )
        return refreshed

    monkeypatch.setattr(maintenance, "with_party_kind_coverage", recount_then_another_writer_lands)

    run = _backfilled(version.rig, version.index, dry_run=False, capsys=capsys)

    assert (run["refreshed"], run["changed_under_run"]) == (0, 1)
    assert _profiles(version.rig.version_store, version.manifest) == (changed, version.earlier)


def test_a_version_whose_frame_no_tier_holds_is_counted_and_left_alone(capsys):
    version = _minted_then_counted_under_the_earlier_rule()

    run = _backfilled(
        version.rig,
        version.index,
        dry_run=False,
        capsys=capsys,
        artifact_store=FakeArtifactStore(),
    )

    assert (run["frame_gone"], run["refreshed"]) == (1, 0)
    assert _profiles(version.rig.version_store, version.manifest) == (
        version.earlier,
        version.earlier,
    )


def test_a_version_that_fails_is_logged_and_left_alone_and_the_task_exits_1(monkeypatch, capsys):
    """A version the recount raises on is logged with its error, keeps its profile and
    counts as failed, and the task exits non-zero."""
    version = _minted_then_counted_under_the_earlier_rule()

    def failing(*_args, **_kwargs):
        raise RuntimeError("frame unreadable")

    monkeypatch.setattr(maintenance, "with_party_kind_coverage", failing)

    run = _backfilled(version.rig, version.index, dry_run=False, capsys=capsys, failed=1)
    assert run["refreshed"] == 0
    assert _profiles(version.rig.version_store, version.manifest) == (
        version.earlier,
        version.earlier,
    )

    monkeypatch.setattr(maintenance, "MaintenanceSettings", _Settings)
    monkeypatch.setattr(
        maintenance, "LiveDatasetStore", lambda connections: version.rig.version_store
    )
    monkeypatch.setattr(maintenance, "shared_pool", lambda: object())
    monkeypatch.setattr(maintenance, "close_shared_pool", lambda: None)
    monkeypatch.setattr(maintenance, "LiveArtifactStore", lambda **_: version.rig.artifact_store)
    monkeypatch.setattr(maintenance, "get_release_service", lambda: None)
    monkeypatch.setattr(maintenance, "bound_served_release", lambda _service: nullcontext())
    monkeypatch.setattr(maintenance, "get_index", lambda: version.index)
    capsys.readouterr()

    assert maintenance.main(["backfill-party-kind-coverage"]) == 1
    [failure] = [
        line
        for line in map(json.loads, capsys.readouterr().out.splitlines())
        if line.get("event") == "datasets_party_kind_backfill_failed"
    ]
    assert failure["error"] == "RuntimeError: frame unreadable"


@dataclass(frozen=True)
class _Settings:
    artifact_bucket_working: str = "working-bucket"
    artifact_bucket_pinned: str = "pinned-bucket"
