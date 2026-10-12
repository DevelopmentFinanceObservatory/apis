"""Compares a tier's bucket listing with `artifact_location` in both directions. Every pinned
digest needs an `artifact_reference` row, and any row, even soft-deleted, satisfies that. The
working tier has no such leg.

Object names come from `artifact_location.object_name`, which is the digest or, after a working
tier renewal, `<digest>#r<renewal>`. A renewal leaves the previous object in the bucket until the
lifecycle rule deletes it, so that object reports as orphaned until then.

A working row expired under `dataset_store.snapshot.working_tier_expired` is not servable, and
the working bucket's lifecycle rule deletes its object, so the sweep counts it as an expired
working location and leaves it out of the missing objects. Every pinned row and every unexpired
working row should have its object, so each one the bucket lacks is missing. An object stays out
of the orphaned objects while any row for its tier, expired or not, names it.

`datasets_api.maintenance` runs this read-only sweep daily. Bucket and database reads are
unsynchronised, so writes in flight can appear as missing or orphaned objects. Confirm a report
with another sweep after writes settle before acting on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import cast

from artifact_store import ArtifactStore, Tier
from dataset_store import DatasetStore
from dataset_store.snapshot import working_tier_expired


@dataclass(frozen=True)
class ReconciliationReport:
    tier: Tier
    orphaned_objects: frozenset[str]
    """In the bucket, no `artifact_location` row for this tier names it."""
    missing_objects: frozenset[str]
    """An `artifact_location` row for this tier that should have bytes names it, but the bucket
    does not have it. Expired working rows are left out."""
    expired_working_locations: frozenset[str]
    """Working tier only (always empty for pinned): object names of `artifact_location` rows
    expired under `working_tier_expired`, whether or not the bucket still has the object."""
    unreferenced_pinned_digests: frozenset[str]
    """Pinned tier only (always empty for working): an `artifact_location`
    row for this digest exists, but no `artifact_reference` row at all
    does -- see this module's own docstring for why "at all", not
    "no live one". Digests, not bucket object names: the operator act this
    leg feeds (a reference backfill, or a copy removal) keys on the
    digest."""


def sweep_tier(
    *, store: DatasetStore, artifact_store: ArtifactStore, tier: Tier, now: datetime
) -> ReconciliationReport:
    """Compares one tier's inventory and bucket state at `now` without making repairs."""
    with store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT object_name, stored_at FROM datasets.artifact_location WHERE tier = %s",
            (tier.value,),
        )
        unreferenced_rows = (
            tx.fetch_all(
                "SELECT artifact_digest FROM datasets.artifact_location al "
                "WHERE al.tier = %s AND NOT EXISTS ("
                "SELECT 1 FROM datasets.artifact_reference ar "
                "WHERE ar.artifact_digest = al.artifact_digest"
                ")",
                (tier.value,),
            )
            if tier == Tier.pinned
            else []
        )
    inventory_objects = frozenset(str(row["object_name"]) for row in rows)
    expired_objects = (
        frozenset(
            str(row["object_name"])
            for row in rows
            if working_tier_expired(cast(datetime, row["stored_at"]), now=now)
        )
        if tier == Tier.working
        else frozenset()
    )
    bucket_objects = artifact_store.list(tier=tier)
    return ReconciliationReport(
        tier=tier,
        orphaned_objects=bucket_objects - inventory_objects,
        missing_objects=inventory_objects - expired_objects - bucket_objects,
        expired_working_locations=expired_objects,
        unreferenced_pinned_digests=frozenset(
            str(row["artifact_digest"]) for row in unreferenced_rows
        ),
    )


def sweep(
    *, store: DatasetStore, artifact_store: ArtifactStore, now: datetime
) -> tuple[ReconciliationReport, ...]:
    """Both tiers at `now`, in `Tier`'s own declared order."""
    return tuple(
        sweep_tier(store=store, artifact_store=artifact_store, tier=tier, now=now) for tier in Tier
    )
