"""Synchronous write registry for canonical definitions.

`register_definition` recomputes the identity-core document, verifies the supplied id, then
inserts or byte-compares canonical JCS bytes. A collision commits no registry change, stores
both versions in a separate quarantine transaction, logs the stable alert marker, and raises.
Quarantined ids remain ambiguous even when a later registration matches either version.
`register_served_definition` registers a served definition when absent and binds it to the
posted id it serves in `datasets.spec_alias`, in one transaction.
SQL is schema-qualified. Async callers use a worker thread. Reading belongs to `SpecReader`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, cast

from dataset_store import DatasetStore, PoolTimeoutError, Transaction
from dfo_platform.store_errors import safe_diagnostics
from spec_registry.errors import (
    IdentityCollision,
    IdentityQuarantined,
    SpecIdMismatch,
    SpecRegistryStoreError,
    SpecRegistryUnavailable,
)
from spec_registry.identity import RecomputedIdentity, recompute_identity
from spec_registry.reader import StoredDefinition, read_stored_definitions

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

    def register_served_definition(
        self,
        *,
        served: RecomputedIdentity,
        schema_version: int,
        posted_spec_id: str | None,
        trace_id: str | None = None,
    ) -> StoredDefinition:
        """Return the stored definition of the served identity `served`, registering it at
        `schema_version` when absent, and bind it to `posted_spec_id` when one is given.

        One transaction reads, registers and binds, and the binding is its last write. A
        binding already held adds no row, and a call that raises binds nothing.
        Raises `IdentityQuarantined` when the served id, the posted id or a served id already
        bound to either is quarantined, the rule every `SpecReader` read applies. Raises `IdentityCollision`, quarantining the served id, when the registry
        holds other identity bytes under it.
        """
        ...


@dataclass(frozen=True)
class _Registration:
    """What one registration attempt found: whether it inserted the registry row, the bytes
    the registry holds, and whether the id is quarantined."""

    inserted: bool
    held_bytes: bytes
    already_quarantined: bool


def _register_in(
    tx: Transaction,
    recomputed: RecomputedIdentity,
    *,
    schema_version: int,
    trace_id: str | None,
) -> _Registration:
    """Inside `tx`, insert the registry row of `recomputed` or read the bytes held under its id,
    and store its definition when the bytes match and the id is not quarantined."""
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

    if held_bytes == recomputed.identity_bytes:
        # Quarantine entries require an existing id, so new ids skip this query.
        if not inserted:
            quarantine_row = tx.fetch_one(
                "SELECT 1 FROM datasets.identity_quarantine "
                "WHERE id_kind = %s AND subject_id = %s LIMIT 1",
                ("spec", recomputed.spec_id),
            )
            already_quarantined = quarantine_row is not None
        if not already_quarantined:
            # Store the canonical projection so readers reproduce the minted spec id.
            tx.execute(
                "INSERT INTO datasets.spec_definition "
                "(spec_id, schema_version, definition) VALUES (%s, %s, %s) "
                "ON CONFLICT (spec_id, schema_version) DO NOTHING",
                (
                    recomputed.spec_id,
                    schema_version,
                    {
                        "schema_version": schema_version,
                        "spec_id": recomputed.spec_id,
                        "identity": recomputed.identity,
                    },
                ),
            )
    return _Registration(
        inserted=inserted, held_bytes=held_bytes, already_quarantined=already_quarantined
    )


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

        # Raise safe summaries outside the except block to keep driver messages out of context.
        # Pool timeouts map to retryable unavailability.
        # Other failures map to store errors.
        failure: str | None = None
        unavailable = False
        registration: _Registration | None = None
        try:
            with self._store.transaction() as tx:
                registration = _register_in(
                    tx, recomputed, schema_version=schema_version, trace_id=trace_id
                )
        except PoolTimeoutError as exc:
            failure = f"registering spec_id {recomputed.spec_id!r} failed: {safe_diagnostics(exc)}"
            unavailable = True
        except Exception as exc:
            failure = f"registering spec_id {recomputed.spec_id!r} failed: {safe_diagnostics(exc)}"

        if failure is not None:
            if unavailable:
                raise SpecRegistryUnavailable(failure)
            raise SpecRegistryStoreError(failure)

        assert registration is not None, "registration transaction returned no outcome"
        self._refuse_unregistered(recomputed, registration, trace_id=trace_id)
        return RegisteredDefinition(
            spec_id=recomputed.spec_id,
            schema_version=schema_version,
            inserted=registration.inserted,
        )

    def register_served_definition(
        self,
        *,
        served: RecomputedIdentity,
        schema_version: int,
        posted_spec_id: str | None,
        trace_id: str | None = None,
    ) -> StoredDefinition:
        failure: str | None = None
        unavailable = False
        quarantined_ids: set[str] = set()
        stored: StoredDefinition | None = None
        registration: _Registration | None = None
        try:
            with self._store.transaction() as tx:
                spec_ids = [served.spec_id]
                if posted_spec_id is not None:
                    spec_ids.append(posted_spec_id)
                # The read also checks the served ids already bound to each id.
                quarantined_ids, definitions = read_stored_definitions(tx, spec_ids)
                if not quarantined_ids:
                    stored = definitions.get(served.spec_id)
                    if stored is None:
                        registration = _register_in(
                            tx, served, schema_version=schema_version, trace_id=trace_id
                        )
                    else:
                        registration = _Registration(
                            inserted=False,
                            held_bytes=stored.identity_bytes,
                            already_quarantined=False,
                        )
                    # The binding is the last write, made only when nothing below raises.
                    if (
                        posted_spec_id is not None
                        and registration.held_bytes == served.identity_bytes
                        and not registration.already_quarantined
                    ):
                        tx.execute(
                            "INSERT INTO datasets.spec_alias (posted_spec_id, served_spec_id) "
                            "VALUES (%s, %s) "
                            "ON CONFLICT (posted_spec_id, served_spec_id) DO NOTHING",
                            (posted_spec_id, served.spec_id),
                        )
        except PoolTimeoutError as exc:
            failure = f"registering spec_id {served.spec_id!r} failed: {safe_diagnostics(exc)}"
            unavailable = True
        except Exception as exc:
            failure = f"registering spec_id {served.spec_id!r} failed: {safe_diagnostics(exc)}"

        if failure is not None:
            if unavailable:
                raise SpecRegistryUnavailable(failure)
            raise SpecRegistryStoreError(failure)

        if quarantined_ids:
            raise IdentityQuarantined(subject_id=sorted(quarantined_ids)[0], id_kind="spec")
        assert registration is not None, "served definition neither read nor registered"
        self._refuse_unregistered(served, registration, trace_id=trace_id)
        if stored is not None:
            return stored
        return StoredDefinition(
            spec_id=served.spec_id,
            schema_version=schema_version,
            identity=served.identity,
            identity_bytes=served.identity_bytes,
            identity_digest=served.identity_digest,
        )

    def _refuse_unregistered(
        self,
        recomputed: RecomputedIdentity,
        registration: _Registration,
        *,
        trace_id: str | None,
    ) -> None:
        """Quarantine and raise `IdentityCollision` when the registry holds other bytes under
        the id of `recomputed`, and raise `IdentityQuarantined` for a quarantined id."""
        if registration.held_bytes != recomputed.identity_bytes:
            self._quarantine(
                id_kind="spec",
                subject_id=recomputed.spec_id,
                held_bytes=registration.held_bytes,
                incoming_bytes=recomputed.identity_bytes,
                trace_id=trace_id,
            )
            raise IdentityCollision(subject_id=recomputed.spec_id)
        if registration.already_quarantined:
            raise IdentityQuarantined(subject_id=recomputed.spec_id, id_kind="spec")

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
