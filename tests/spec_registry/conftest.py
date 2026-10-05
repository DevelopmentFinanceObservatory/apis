"""Provides a domain-level SQL double over shared in-memory registry state.
Its lock serialises transactions for concurrent-insert tests."""

from __future__ import annotations

import json
import threading
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field


def shape_for(tag: str) -> dict:
    """A schema-valid described-arm `shape` union (`kind: "statvar_schema"`)
    carrying `tag` as its sole `population_type` member -- a convenience for
    tests that need a distinguishing, hash-affecting shape without spelling
    out all five schema members.
    """
    return {"kind": "statvar_schema", "population_type": tag}


def minimal_identity(**overrides: object) -> dict:
    """A minimal valid spec1 identity-core document, overridable per test --
    mirrors `tests/engine/test_identity_vectors.py`'s own `_minimal_document`
    shape.
    """
    doc: dict = {
        "identity_version": "spec1",
        "shape": shape_for("test_shape"),
        "slots": [],
        "entities": {},
        "requested_window": {"kind": "all"},
    }
    doc.update(overrides)
    return doc


def identity_of_encoded_size(*, encoded_bytes: int) -> dict:
    """A schema-valid `minimal_identity()` document whose compact JSON
    encoding (`json.dumps(..., separators=(",", ":"), ensure_ascii=False)`,
    the same encoding `spec_registry.identity.recompute_identity` measures
    against its byte cap) is exactly `encoded_bytes` long -- built by padding
    one `entities` role with filler refs, so a caller can construct a
    document sitting exactly at, or one byte past, that cap.

    For `n` refs of equal length `filler_length` (all but a shorter or
    longer last one to close the exact gap), a JSON array of ASCII strings
    with no internal whitespace encodes to `sum(len(ref)) + 3n + 1` bytes
    (2 bracket bytes, 2 quote bytes per ref, `n - 1` comma bytes).
    """
    document = minimal_identity(entities={"provider": []})
    base_length = len(
        json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    # `entities: {"provider": []}` costs 2 bytes ("[]") in that base; the
    # padded list below replaces it, so the list's own encoded length must
    # close the remaining gap.
    target_list_length = encoded_bytes - base_length + 2
    assert target_list_length >= 2, "encoded_bytes is smaller than the unpadded document"

    filler_length = 900  # comfortably under the kit's own 1,024-char string cap
    max_string_length = 1024
    max_values_per_role = 1024
    refs_count = 1
    while True:
        last_length = target_list_length - 3 * refs_count - 1 - (refs_count - 1) * filler_length
        if 0 <= last_length <= max_string_length:
            break
        refs_count += 1
        assert refs_count <= max_values_per_role, "encoded_bytes needs too many filler refs"

    refs = ["x" * filler_length] * (refs_count - 1) + ["x" * last_length]
    document["entities"] = {"provider": refs}

    actual = len(json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    assert actual == encoded_bytes, (
        f"padding arithmetic drifted: got {actual}, wanted {encoded_bytes}"
    )
    return document


def wire_definition_for(
    *, spec_id: str, identity: Mapping[str, object], schema_version: int = 1
) -> dict:
    """A minimal register-call wire definition envelope: `schema_version`
    and `spec_id` alongside the identity-core document under `identity` --
    matching the shape `tests/conftest.py::minimal_spec` already uses
    elsewhere in this repo for a spec's `definition` member.
    """
    return {"schema_version": schema_version, "spec_id": spec_id, "identity": dict(identity)}


@dataclass
class FakeRegistryState:
    spec_registry: dict[str, dict] = field(default_factory=dict)
    spec_definition: dict[tuple[str, int], dict] = field(default_factory=dict)
    collection_registry: dict[str, dict] = field(default_factory=dict)
    collection_member: dict[str, dict[int, str]] = field(default_factory=dict)
    identity_quarantine: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class FakeTransaction:
    """Recognises the closed set of literal SQL statements
    `spec_registry.registry.LiveSpecRegistry`/`spec_registry.reader.LiveSpecReader`
    issue, by table name and statement kind, and mutates `FakeRegistryState`
    in place -- never a generic SQL executor.
    """

    def __init__(self, state: FakeRegistryState) -> None:
        self._state = state

    def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        self._dispatch(sql, params)

    def fetch_one(self, sql: str, params: Sequence[object] = ()) -> Mapping[str, object] | None:
        rows = self._dispatch(sql, params)
        if not rows:
            return None
        if len(rows) > 1:
            raise ValueError(f"fetch_one got {len(rows)} rows for {sql!r}")
        return rows[0]

    def fetch_all(self, sql: str, params: Sequence[object] = ()) -> list[Mapping[str, object]]:
        return self._dispatch(sql, params)

    def _dispatch(self, sql: str, params: Sequence[object]) -> list[dict]:
        s = self._state

        if "INSERT INTO datasets.spec_registry" in sql:
            spec_id, identity_version, identity_bytes, identity_digest, trace_id = params
            if spec_id in s.spec_registry:
                return []
            s.spec_registry[spec_id] = {
                "identity_version": identity_version,
                "identity_bytes": identity_bytes,
                "identity_digest": identity_digest,
                "trace_id": trace_id,
            }
            return [{"spec_id": spec_id}]

        if "SELECT identity_bytes FROM datasets.spec_registry" in sql:
            (spec_id,) = params
            row = s.spec_registry.get(spec_id)
            return [{"identity_bytes": row["identity_bytes"]}] if row else []

        if "SELECT spec_id, identity_bytes, identity_digest FROM datasets.spec_registry" in sql:
            return [
                {
                    "spec_id": spec_id,
                    "identity_bytes": row["identity_bytes"],
                    "identity_digest": row["identity_digest"],
                }
                for spec_id, row in s.spec_registry.items()
                if spec_id in params
            ]

        if "INSERT INTO datasets.spec_definition" in sql:
            spec_id, schema_version, definition = params
            key = (spec_id, schema_version)
            if key not in s.spec_definition:
                s.spec_definition[key] = {"definition": definition}
            return []

        if "SELECT spec_id, schema_version, definition FROM datasets.spec_definition" in sql:
            if "AND schema_version = %s" in sql:
                *spec_ids, schema_version = params
                return [
                    {"spec_id": key[0], "schema_version": key[1], "definition": row["definition"]}
                    for key, row in s.spec_definition.items()
                    if key[0] in spec_ids and key[1] == schema_version
                ]
            return [
                {"spec_id": key[0], "schema_version": key[1], "definition": row["definition"]}
                for key, row in s.spec_definition.items()
                if key[0] in params
            ]

        if "SELECT collection_id FROM datasets.collection_registry" in sql:
            (collection_id,) = params
            return (
                [{"collection_id": collection_id}] if collection_id in s.collection_registry else []
            )

        if "INSERT INTO datasets.collection_registry" in sql:
            collection_id, trace_id = params
            if collection_id in s.collection_registry:
                return []
            s.collection_registry[collection_id] = {"trace_id": trace_id}
            s.collection_member.setdefault(collection_id, {})
            return [{"collection_id": collection_id}]

        if "INSERT INTO datasets.collection_member" in sql:
            collection_id, position, spec_id = params
            s.collection_member.setdefault(collection_id, {})[position] = spec_id
            return []

        if "SELECT spec_id FROM datasets.collection_member" in sql:
            (collection_id,) = params
            members = s.collection_member.get(collection_id, {})
            return [{"spec_id": members[position]} for position in sorted(members)]

        if "INSERT INTO datasets.identity_quarantine" in sql:
            id_kind, subject_id, held_bytes, incoming_bytes, trace_id = params
            s.identity_quarantine.append(
                {
                    "id_kind": id_kind,
                    "subject_id": subject_id,
                    "held_bytes": held_bytes,
                    "incoming_bytes": incoming_bytes,
                    "trace_id": trace_id,
                }
            )
            return []

        if "SELECT 1 FROM datasets.identity_quarantine" in sql:
            id_kind, subject_id = params
            hit = any(
                row["id_kind"] == id_kind and row["subject_id"] == subject_id
                for row in s.identity_quarantine
            )
            return [{"1": 1}] if hit else []

        if "SELECT subject_id FROM datasets.identity_quarantine" in sql:
            id_kind, *subject_ids = params
            return [
                {"subject_id": row["subject_id"]}
                for row in s.identity_quarantine
                if row["id_kind"] == id_kind and row["subject_id"] in subject_ids
            ]

        raise AssertionError(f"FakeTransaction has no handler for statement: {sql!r}")


class FakeDatasetStore:
    """Structurally satisfies `dataset_store.DatasetStore`. Build two
    instances over the SAME `FakeRegistryState` (via the `state=` constructor arg) to
    simulate two pool-borrowed connections seeing the same durable rows.
    """

    def __init__(self, state: FakeRegistryState | None = None) -> None:
        self.state = state if state is not None else FakeRegistryState()

    @contextmanager
    def transaction(self) -> Generator[FakeTransaction, None, None]:
        # Serialises each transaction's full body -- see module docstring.
        with self.state.lock:
            yield FakeTransaction(self.state)
