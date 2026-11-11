"""`datasets-maintenance backfill-party-kind-coverage`
(`maintenance.run_backfill_party_kind_coverage`) over a version whose output profile was
counted under the earlier rule, which read each party for no dataset."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from code_index import CodeIndex
from datasets_api.maintenance import run_backfill_party_kind_coverage

from . import test_serve_party_totals as party_totals
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


def _backfilled(rig: _PartyRig, index: CodeIndex, *, dry_run: bool, capsys) -> dict[str, Any]:
    capsys.readouterr()
    failed = run_backfill_party_kind_coverage(
        store=rig.version_store,
        artifact_store=rig.artifact_store,
        parties=index.parties,
        dry_run=dry_run,
    )
    assert failed == 0
    [counts] = [
        line
        for line in map(json.loads, capsys.readouterr().out.splitlines())
        if line.get("event") == "datasets_party_kind_backfilled"
    ]
    return counts


def test_a_profile_counted_under_the_earlier_rule_is_counted_again_per_row(capsys):
    """A dry run counts the row it would rewrite and writes nothing. The run writes the count a
    mint makes today to the profile column and the body alike, and a second run finds nothing
    to change."""
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

    dry = _backfilled(rig, index, dry_run=True, capsys=capsys)
    assert (dry["refreshed"], dry["unchanged"]) == (1, 0)
    assert _profiles(rig.version_store, manifest) == (earlier, earlier)

    run = _backfilled(rig, index, dry_run=False, capsys=capsys)
    assert run["refreshed"] == 1
    assert _profiles(rig.version_store, manifest) == (minted, minted)

    again = _backfilled(rig, index, dry_run=False, capsys=capsys)
    assert (again["refreshed"], again["unchanged"]) == (0, 1)
