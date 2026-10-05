"""Synchronous write registry for canonical definitions and collections.

`register_definition` recomputes the identity-core document, verifies the supplied id, then
inserts or byte-compares canonical JCS bytes. A collision commits no registry change, stores
both versions in a separate quarantine transaction, logs the stable alert marker, and raises.
Quarantined ids remain ambiguous even when a later registration matches either version.

`register_collection` registers members sequentially, then mints its id from sorted unique
member ids and upserts the collection and ordered members. Earlier member registrations remain
committed if a later member fails. SQL is schema-qualified. Async callers use a worker thread.
Reading belongs to `SpecReader`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from dataset_store import DatasetStore, PoolTimeoutError
from dfo_platform.store_errors import safe_diagnostics
from spec_registry.collection_id import (
    canonical_collection_bytes,
    mint_collection_id,
    sorted_unique_member_ids,
)
from spec_registry.errors import (
    IdentityCollision,
    IdentityQuarantined,
    SpecIdMismatch,
    SpecRegistryStoreError,
    SpecRegistryUnavailable,
)
from spec_registry.identity import recompute_identity

logger = logging.getLogger(__name__)


# Cloud Logging alert policies use this stable marker.
IDENTITY_COLLISION_QUARANTINED_MARKER = "identity_collision_quarantined"

# Identity version accepted by the spec-id projection.
_SPEC_IDENTITY_VERSION = "spec1"


@dataclass(frozen=True)
class RegisteredDefinition:
    """Verified spec id and version, with whether this call inserted the registry row."""

    spec_id: str
    schema_version: int
    inserted: bool


@dataclass(frozen=True)
class RegisteredCollection:
    """Minted collection id, ordered member ids, and whether this call inserted the row."""

    collection_id: str
    member_spec_ids: list[str]
    inserted: bool


@dataclass(frozen=True)
class MemberDefinition:
    """Input for one sequential member registration."""

    identity_definition: object
    schema_version: int
    supplied_spec_id: str


class SpecRegistry(Protocol):
    def register_definition(
        self,
        *,
        identity_definition: object,
        schema_version: int,
        supplied_spec_id: str,
        trace_id: str | None = None,
    ) -> RegisteredDefinition:
        """Recompute and verify identity, then insert or compare canonical bytes.

        Raises on mismatch or collision.
        """
        ...

    def register_collection(
        self,
        *,
        members: Sequence[MemberDefinition],
        trace_id: str | None = None,
    ) -> RegisteredCollection:
        """Register members in order, then mint and upsert the collection id.

        Member failures propagate. Earlier registrations commit even if a later member fails.
        Empty members and collection-id collisions raise their registry errors.
        """
        ...


class LiveSpecRegistry:
    """`DatasetStore` implementation of `SpecRegistry`."""

    def __init__(self, *, store: DatasetStore) -> None:
        self._store = store

    def register_definition(
        self,
        *,
        identity_definition: object,
        schema_version: int,
        supplied_spec_id: str,
        trace_id: str | None = None,
    ) -> RegisteredDefinition:
        recomputed = recompute_identity(identity_definition)
        if supplied_spec_id != recomputed.spec_id:
            raise SpecIdMismatch(supplied=supplied_spec_id, computed=recomputed.spec_id)

        # Store the canonical projection so readers reproduce the minted spec id.
        persisted_definition = {
            "schema_version": schema_version,
            "spec_id": recomputed.spec_id,
            "identity": recomputed.identity,
        }

        # Raise safe summaries outside the except block to keep driver messages out of context.
        # Pool timeouts map to retryable unavailability.
        # Other failures map to store errors.
        failure: str | None = None
        unavailable = False
        try:
            with self._store.transaction() as tx:
                inserted_row = tx.fetch_one(
                    "INSERT INTO datasets.spec_registry "
                    "(spec_id, identity_version, identity_bytes, identity_digest, trace_id) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (spec_id) DO NOTHING RETURNING spec_id",
                    (
                        recomputed.spec_id,
                        _SPEC_IDENTITY_VERSION,
                        recomputed.identity_bytes,
                        recomputed.identity_digest,
                        trace_id,
                    ),
                )
                inserted = inserted_row is not None
                already_quarantined = False
                if inserted:
                    held_bytes = recomputed.identity_bytes
                else:
                    existing = tx.fetch_one(
                        "SELECT identity_bytes FROM datasets.spec_registry WHERE spec_id = %s",
                        (recomputed.spec_id,),
                    )
                    assert existing is not None, (
                        "spec_registry row missing immediately after a conflicting insert"
                    )
                    held_bytes = cast(bytes, existing["identity_bytes"])

                collided = held_bytes != recomputed.identity_bytes
                if not collided:
                    # Quarantine entries require an existing id, so new ids skip this query.
                    if not inserted:
                        quarantine_row = tx.fetch_one(
                            "SELECT 1 FROM datasets.identity_quarantine "
                            "WHERE id_kind = %s AND subject_id = %s LIMIT 1",
                            ("spec", recomputed.spec_id),
                        )
                        already_quarantined = quarantine_row is not None
                    if not already_quarantined:
                        tx.execute(
                            "INSERT INTO datasets.spec_definition "
                            "(spec_id, schema_version, definition) VALUES (%s, %s, %s) "
                            "ON CONFLICT (spec_id, schema_version) DO NOTHING",
                            (recomputed.spec_id, schema_version, persisted_definition),
                        )
        except PoolTimeoutError as exc:
            failure = f"registering spec_id {recomputed.spec_id!r} failed: {safe_diagnostics(exc)}"
            unavailable = True
            inserted = False
            collided = False
            already_quarantined = False
        except Exception as exc:
            failure = f"registering spec_id {recomputed.spec_id!r} failed: {safe_diagnostics(exc)}"
            inserted = False
            collided = False
            already_quarantined = False

        if failure is not None:
            if unavailable:
                raise SpecRegistryUnavailable(failure)
            raise SpecRegistryStoreError(failure)

        if collided:
            self._quarantine(
                id_kind="spec",
                subject_id=recomputed.spec_id,
                held_bytes=held_bytes,
                incoming_bytes=recomputed.identity_bytes,
                trace_id=trace_id,
            )
            raise IdentityCollision(subject_id=recomputed.spec_id)

        if already_quarantined:
            raise IdentityQuarantined(subject_id=recomputed.spec_id, id_kind="spec")

        return RegisteredDefinition(
            spec_id=recomputed.spec_id, schema_version=schema_version, inserted=inserted
        )

    def register_collection(
        self,
        *,
        members: Sequence[MemberDefinition],
        trace_id: str | None = None,
    ) -> RegisteredCollection:
        registered_ids: list[str] = []
        for position, member in enumerate(members):
            try:
                registered = self.register_definition(
                    identity_definition=member.identity_definition,
                    schema_version=member.schema_version,
                    supplied_spec_id=member.supplied_spec_id,
                    trace_id=trace_id,
                )
            except SpecIdMismatch as exc:
                raise SpecIdMismatch(
                    supplied=exc.supplied, computed=exc.computed, position=position
                ) from exc
            registered_ids.append(registered.spec_id)

        sorted_members = sorted_unique_member_ids(registered_ids)
        collection_id = mint_collection_id(registered_ids)
        incoming_bytes = canonical_collection_bytes(registered_ids)

        failure: str | None = None
        unavailable = False
        try:
            with self._store.transaction() as tx:
                inserted_row = tx.fetch_one(
                    "INSERT INTO datasets.collection_registry (collection_id, trace_id) "
                    "VALUES (%s, %s) ON CONFLICT (collection_id) DO NOTHING "
                    "RETURNING collection_id",
                    (collection_id, trace_id),
                )
                inserted = inserted_row is not None
                if inserted:
                    for position, spec_id in enumerate(sorted_members):
                        tx.execute(
                            "INSERT INTO datasets.collection_member "
                            "(collection_id, position, spec_id) VALUES (%s, %s, %s)",
                            (collection_id, position, spec_id),
                        )
                    held_bytes = incoming_bytes
                else:
                    existing_rows = tx.fetch_all(
                        "SELECT spec_id FROM datasets.collection_member "
                        "WHERE collection_id = %s ORDER BY position",
                        (collection_id,),
                    )
                    existing_members = [cast(str, row["spec_id"]) for row in existing_rows]
                    held_bytes = canonical_collection_bytes(existing_members)

                collided = held_bytes != incoming_bytes
                # Quarantine entries require an existing id, so new ids skip this query.
                already_quarantined = False
                if not inserted and not collided:
                    quarantine_row = tx.fetch_one(
                        "SELECT 1 FROM datasets.identity_quarantine "
                        "WHERE id_kind = %s AND subject_id = %s LIMIT 1",
                        ("collection", collection_id),
                    )
                    already_quarantined = quarantine_row is not None
        except PoolTimeoutError as exc:
            failure = f"registering collection_id {collection_id!r} failed: {safe_diagnostics(exc)}"
            unavailable = True
            inserted = False
            collided = False
            already_quarantined = False
            held_bytes = incoming_bytes
        except Exception as exc:
            failure = f"registering collection_id {collection_id!r} failed: {safe_diagnostics(exc)}"
            inserted = False
            collided = False
            already_quarantined = False
            held_bytes = incoming_bytes

        if failure is not None:
            if unavailable:
                raise SpecRegistryUnavailable(failure)
            raise SpecRegistryStoreError(failure)

        if collided:
            self._quarantine(
                id_kind="collection",
                subject_id=collection_id,
                held_bytes=held_bytes,
                incoming_bytes=incoming_bytes,
                trace_id=trace_id,
            )
            raise IdentityCollision(subject_id=collection_id)

        if already_quarantined:
            raise IdentityQuarantined(subject_id=collection_id, id_kind="collection")

        return RegisteredCollection(
            collection_id=collection_id, member_spec_ids=sorted_members, inserted=inserted
        )

    def _quarantine(
        self,
        *,
        id_kind: str,
        subject_id: str,
        held_bytes: bytes,
        incoming_bytes: bytes,
        trace_id: str | None,
    ) -> None:
        """Quarantine both identities in a separate transaction after logging the alert marker.

        The marker is logged before insertion so alerts fire even if the insert fails.
        """
        logger.error(
            "%s id_kind=%s subject_id=%s trace_id=%s",
            IDENTITY_COLLISION_QUARANTINED_MARKER,
            id_kind,
            subject_id,
            trace_id,
        )

        # Raise safe diagnostics after the except block to keep driver messages out of context.
        failure: str | None = None
        try:
            with self._store.transaction() as tx:
                tx.execute(
                    "INSERT INTO datasets.identity_quarantine "
                    "(id_kind, subject_id, held_bytes, incoming_bytes, trace_id) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (id_kind, subject_id, held_bytes, incoming_bytes, trace_id),
                )
        except Exception as exc:
            failure = f"quarantining {id_kind} id {subject_id!r} failed: {safe_diagnostics(exc)}"

        if failure is not None:
            logger.error(
                "quarantine insert failed after the collision marker above: "
                "id_kind=%s subject_id=%s",
                id_kind,
                subject_id,
            )
            raise SpecRegistryStoreError(failure)
