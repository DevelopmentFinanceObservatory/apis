"""Compares a tier's bucket listing with `artifact_location` in both directions. Every pinned
digest needs an `artifact_reference` row, and any row, even soft-deleted, satisfies that. The
working tier has no such leg.

Scripts and scheduled jobs run this read-only sweep. Bucket and database reads are unsynchronised,
so writes in flight can appear as missing or orphaned objects. Confirm a report with another sweep
after writes settle before acting on it.
"""

from __future__ import annotations

from dataclasses import dataclass

from artifact_store import ArtifactStore, Tier
from dataset_store import DatasetStore


@dataclass(frozen=True)
class ReconciliationReport:
    tier: Tier
    orphaned_objects: frozenset[str]
    """In the bucket, no `artifact_location` row for this tier names it."""
    missing_objects: frozenset[str]
    """An `artifact_location` row for this tier names it, but the bucket
    does not have it."""
    unreferenced_pinned_digests: frozenset[str]
    """Pinned tier only (always empty for working): an `artifact_location`
    row for this digest exists, but no `artifact_reference` row at all
    does -- see this module's own docstring for why "at all", not
    "no live one". Digests, not bucket object names: the operator act this
    leg feeds (a reference backfill, or a copy removal) keys on the
    digest."""


def sweep_tier(
    *, store: DatasetStore, artifact_store: ArtifactStore, tier: Tier
) -> ReconciliationReport:
    """Compares one tier's inventory and bucket state without making repairs."""
    with store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT object_name FROM datasets.artifact_location WHERE tier = %s", (tier.value,)
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
    bucket_objects = artifact_store.list(tier=tier)
    return ReconciliationReport(
        tier=tier,
        orphaned_objects=bucket_objects - inventory_objects,
        missing_objects=inventory_objects - bucket_objects,
        unreferenced_pinned_digests=frozenset(
            str(row["artifact_digest"]) for row in unreferenced_rows
        ),
    )


def sweep(
    *, store: DatasetStore, artifact_store: ArtifactStore
) -> tuple[ReconciliationReport, ...]:
    """Both tiers, in `Tier`'s own declared order."""
    return tuple(sweep_tier(store=store, artifact_store=artifact_store, tier=tier) for tier in Tier)
