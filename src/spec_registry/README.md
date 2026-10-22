# `spec_registry`

`spec_registry` handles storage and identity validation for dataset specification definitions backed by PostgreSQL (`dataset_store`). It operates as an HTTP-free domain package consumed by `datasets_api` doors (`POST /api/datasets/register`, `POST /api/datasets`, `GET /api/datasets/{id}`, `GET /api/datasets/{id}/{content_version}`, and `POST /api/datasets/{id}/{content_version}/pin`) and materialisation workflows.

## Identity Architecture

### Specification Identity (`spec_id`)

A specification identity is a deterministic content hash computed over the identity-core definition.

Given an identity-core document, `spec_registry.identity.recompute_identity` projects and normalises fields via `dfo_spec_kit.identity.build_projection`, serialises to canonical JSON (JCS / RFC 8785) via `dfo_spec_kit.identity.canonicalize`, and hashes the result using SHA-256 via `dfo_spec_kit.identity.mint_spec_id`.

The identifier uses the `spec_` prefix followed by the first 32 lowercase hexadecimal characters of the SHA-256 digest of canonical JCS bytes (for example, `spec_0123456789abcdef0123456789abcdef`). Only fields within the identity-core document participate in identity computation. Envelope metadata (`schema_version`, execution options, presentation preferences) is excluded.

The compact UTF-8 JSON encoding of the identity document has a maximum size of 65,536 bytes (`_MAX_IDENTITY_DOCUMENT_BYTES`). Payloads exceeding this limit raise `dfo_spec_kit.identity.IdentityError` (`input_limit_exceeded`).

```
Identity Core Document -> Projection -> Canonical JCS Bytes -> SHA-256 -> spec_<hex[:32]>
```

## Storage Contracts

`spec_registry` interacts with PostgreSQL (`datasets` schema) using the following relational model:

| Table | Primary Key | Description |
| --- | --- | --- |
| `datasets.spec_registry` | `spec_id` | Canonical identity record storing `identity_version` (e.g., `spec1`), `identity_bytes` (JCS bytes), `identity_digest` (full SHA-256 hex digest), and registration timestamp. Written once, immutable. |
| `datasets.spec_definition` | `(spec_id, schema_version)` | Append-only store for complete canonical definitions storing `definition` JSONB derived from canonical `identity_bytes`. Keyed by version to allow definitions to evolve while preserving identity bytes. |
| `datasets.identity_quarantine` | `id` (bigserial) | Quarantine log for identity collision records (`id_kind` `spec`), storing `held_bytes`, `incoming_bytes`, and `trace_id`. |
| `datasets.spec_alias` | `(posted_spec_id, served_spec_id)` | One row per served id a posted id has resolved to, with `bound_at`. Both ids reference `datasets.spec_registry`. |

### Write Transactions (`SpecRegistry` / `LiveSpecRegistry`)

The `SpecRegistry` protocol defines the write interface:

```python
class SpecRegistry(Protocol):
    def register_definition(
        self,
        *,
        identity_definition: object,
        schema_version: int,
        supplied_spec_id: str,
        trace_id: str | None = None,
    ) -> RegisteredDefinition: ...

    def register_served_definition(
        self,
        *,
        served: RecomputedIdentity,
        schema_version: int,
        posted_spec_id: str | None,
        trace_id: str | None = None,
    ) -> StoredDefinition: ...
```

#### Registration Semantics and Collision Handling

1. **Identity Verification**: `register_definition` recomputes the identity from `identity_definition`. If `supplied_spec_id` differs from the computed `spec_id`, it raises `SpecIdMismatch`.
2. **Canonical Insertion**: Executes `INSERT ... ON CONFLICT (spec_id) DO NOTHING RETURNING spec_id` against `datasets.spec_registry`.
3. **Idempotency and Quarantine**:
   - On successful row insertion, the canonical definition is written to `datasets.spec_definition`.
   - If the row already exists, `identity_bytes` are read back and compared byte-for-byte with incoming canonical bytes.
   - When bytes match, the identifier is checked against `datasets.identity_quarantine`. If quarantined, it raises `IdentityQuarantined`. If unquarantined, it writes to `datasets.spec_definition` if missing and returns `RegisteredDefinition(inserted=False)`.
   - When bytes diverge, held and incoming bytes are written to `datasets.identity_quarantine` in an independent transaction. The system logs `identity_collision_quarantined` and raises `IdentityCollision`.

#### Served Definitions and Alias Bindings

`register_served_definition` returns the stored definition of a served identity, the canonical form of a posted identity under the code index. One transaction checks `datasets.identity_quarantine` for the served id, the posted id and each served id `datasets.spec_alias` already binds to either, reads the served definition, registers it with the steps above when absent, and inserts the `(posted_spec_id, served_spec_id)` row into `datasets.spec_alias` with `ON CONFLICT DO NOTHING` as its last write. A `posted_spec_id` of `None` binds nothing. A quarantined id among those checked raises `IdentityQuarantined`, the rule `read_definition` and `read_served_spec_ids` apply to a read under the posted id, so a posted id refuses here in every state where a read or an address under it refuses. The identity bytes the registry holds under the served id are compared with the incoming ones whether or not the definition is already stored, and a mismatch quarantines the served id and raises `IdentityCollision`. A call that raises inserts no alias row.

### Read Interface (`SpecReader` / `LiveSpecReader`)

The `SpecReader` protocol defines read-only retrieval:

```python
class SpecReader(Protocol):
    def read_definition(
        self, spec_id: str, *, schema_version: int | None = None
    ) -> StoredDefinition | None: ...

    def read_definitions(
        self, spec_ids: Sequence[str]
    ) -> dict[str, StoredDefinition]: ...

    def read_served_spec_ids(self, spec_id: str) -> tuple[str, ...] | None: ...
```

#### Read Semantics

- **Quarantine Check**: Every read checks `datasets.identity_quarantine` for the requested `spec_id` and for each served id `datasets.spec_alias` binds to it, in one statement of the read's transaction. When any of them is quarantined, `IdentityQuarantined` is raised, naming the quarantined id.
- **Missing Records**: Unknown identifiers return `None` (or omit the key in `read_definitions`).
- **Version Resolution**: When `schema_version` is omitted in `read_definition` or `read_definitions`, the reader selects the highest available `schema_version` for that `spec_id`.
- **Batch Reads**: `read_definitions` executes a single batched SQL query using `IN (...)`.
- **Served Ids**: `read_served_spec_ids` returns the served ids `datasets.spec_alias` binds to a posted id, newest `bound_at` first, an empty tuple for a registered id with no binding, and `None` for an unregistered id. It raises `IdentityQuarantined` under the quarantine check above. One transaction makes the three reads.

## Error Hierarchy and Wire Mapping

All domain exceptions inherit from `SpecRegistryError`. Store operations catch database driver exceptions, suppressing raw driver messages and exposing SQLSTATE codes when present.

| Exception | Cause / Condition | HTTP Status (in `datasets_api`) |
| --- | --- | --- |
| `SpecIdMismatch` | Client-provided `spec_id` differs from server-recomputed identity. | `422 Unprocessable Entity` (`spec_id_mismatch`) |
| `IdentityCollision` | Stored `identity_bytes` differ from incoming canonical bytes for the same id. Both are quarantined. | `500 Internal Server Error` (`identity_collision`) |
| `IdentityQuarantined` | Identifier is present in `datasets.identity_quarantine`. Requests for this ID are refused. | `500 Internal Server Error` (`identity_quarantined`) |
| `SpecRegistryUnavailable` | Database connection pool exhausted (`PoolTimeoutError`). Transient condition. | `503 Service Unavailable` (`service_unavailable`) |
| `SpecRegistryStoreError` | Database query failure or connection error. | `503 Service Unavailable` (`service_unavailable`) |

## Integration with `spec-kit` and `datasets_api`

```
+--------------------------------------------------------------------------------+
|                                  datasets_api                                  |
|  (Doors: /register, POST /api/datasets, /{id}, /{id}/{ver}, /{id}/{ver}/pin)  |
+---------------------------------------+----------------------------------------+
                                        |
                                        | Calls SpecRegistry / SpecReader
                                        v
+--------------------------------------------------------------------------------+
|                                 spec_registry                                  |
|  - recompute_identity()                - LiveSpecRegistry                      |
|                                        - LiveSpecReader                        |
|                                        - Quarantine handling                   |
+-----------------------+--------------------------------+-----------------------+
                        |                                |
                        | Uses canonicalization          | Reads/Writes
                        v                                v
+---------------------------------------+  +-------------------------------------+
|               spec-kit                |  |            dataset_store            |
|  - identity.build_projection          |  |  (PostgreSQL tables in              |
|  - identity.canonicalize              |  |   datasets.* schema)                |
|  - identity.mint_spec_id              |  +-------------------------------------+
+---------------------------------------+
```

### Integration Points

- **`spec-kit`**:
  - `spec_registry.identity` acts as an adapter over `dfo_spec_kit.identity`.
  - Normalisation, projection building, and canonical serialization conform to RFC 8785 (JSON Canonicalization Scheme) implemented in `spec-kit`.
  - Spec ID minting uses `dfo_spec_kit.identity.mint_spec_id`.
- **`datasets_api`**:
  - **Register Door (`POST /api/datasets/register`)**: Calls `LiveSpecRegistry.register_definition` on a dedicated `ThreadPoolExecutor` with semaphore-gated backpressure. Maps domain exceptions to standard RFC 9457 problem details envelopes.
  - **Definition Retrieval Door (`GET /api/datasets/{id}`)**: Uses `LiveSpecReader.read_definition` and `read_definitions` to resolve canonical definitions prior to rehydration and materialisation.
  - **Mint Door (`POST /api/datasets`)**: Persists definitions to the registry during artifact materialisation workflows.
  - **Threading**: All `spec_registry` operations are synchronous. Callers in asynchronous contexts dispatch calls to dedicated thread poo
