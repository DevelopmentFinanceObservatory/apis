stry`

`spec_registry` handles storage and identity validation for dataset specification and collection definitions backed by PostgreSQL (`dataset_store`). It operates as an HTTP-free domain package consumed by `datasets_api` doors (`POST /api/datasets/register`, `POST /api/datasets`, `GET /api/datasets/{id}`, `GET /api/datasets/{id}/{content_version}`, and `POST /api/datasets/{id}/{content_version}/pin`) and materialisation workflows.

## Identity Architecture

### Specification Identity (`spec_id`)

A specification identity is a deterministic content hash computed over the identity-core definition.

Given an identity-core document, `spec_registry.identity.recompute_identity` projects and normalises fields via `dfo_spec_kit.identity.build_projection`, serialises to canonical JSON (JCS / RFC 8785) via `dfo_spec_kit.identity.canonicalize`, and hashes the result using SHA-256 via `dfo_spec_kit.identity.mint_spec_id`.

The identifier uses the `spec_` prefix followed by the first 32 lowercase hexadecimal characters of the SHA-256 digest of canonical JCS bytes (for example, `spec_0123456789abcdef0123456789abcdef`). Only fields within the identity-core document participate in identity computation. Envelope metadata (`schema_version`, execution options, presentation preferences) is excluded.

The compact UTF-8 JSON encoding of the identity document has a maximum size of 65,536 bytes (`_MAX_IDENTITY_DOCUMENT_BYTES`). Payloads exceeding this limit raise `dfo_spec_kit.identity.IdentityError` (`input_limit_exceeded`).

```
Identity Core Document -> Projection -> Canonical JCS Bytes -> SHA-256 -> spec_<hex[:32]>
```

### Collection Identity (`collection_id`)

A collection identity is a deterministic content hash computed over an unordered set of member specifications.

The collection projection format is `{"identity_version": "col1", "members": [...]}`. Member `spec_id` values are normalised to Unicode NFC, de-duplicated, and sorted in ascending Unicode code point order (`sorted_unique_member_ids`).

The identifier uses the `col_` prefix followed by the first 32 lowercase hexadecimal characters of the SHA-256 digest over the canonical JCS bytes of the `col1` projection (`mint_collection_id`). A collection requires at least one member; empty member lists raise `EmptyCollectionMembers`.

```
Member spec_ids -> NFC Normalise & De-duplicate -> Codepoint Sort -> col1 Projection -> Canonical JCS -> SHA-256 -> col_<hex[:32]>
```

## Storage Contracts

`spec_registry` interacts with PostgreSQL (`datasets` schema) using the following relational model:

| Table | Primary Key | Description |
| --- | --- | --- |
| `datasets.spec_registry` | `spec_id` | Canonical identity record storing `identity_version` (e.g., `spec1`), `identity_bytes` (JCS bytes), `identity_digest` (full SHA-256 hex digest), and registration timestamp. Written once, immutable. |
| `datasets.spec_definition` | `(spec_id, schema_version)` | Append-only store for complete canonical definitions storing `definition` JSONB derived from canonical `identity_bytes`. Keyed by version to allow definitions to evolve while preserving identity bytes. |
| `datasets.collection_registry` | `collection_id` | Immutable registration row for collection identities. |
| `datasets.collection_member` | `(collection_id, position)` | Positional member mapping linking `collection_id` to member `spec_id` entries in canonical sort order. |
| `datasets.identity_quarantine` | `id` (bigserial) | Quarantine log for identity collision records (`spec` or `collection`), storing `held_bytes`, `incoming_bytes`, and `trace_id`. |

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

    def register_collection(
        self,
        *,
        members: Sequence[MemberDefinition],
        trace_id: str | None = None,
    ) -> RegisteredCollection: ...
```

#### Registration Semantics and Collision Handling

1. **Identity Verification**: `register_definition` recomputes the identity from `identity_definition`. If `supplied_spec_id` differs from the computed `spec_id`, it raises `SpecIdMismatch`.
2. **Canonical Insertion**: Executes `INSERT ... ON CONFLICT (spec_id) DO NOTHING RETURNING spec_id` against `datasets.spec_registry`.
3. **Idempotency and Quarantine**:
   - On successful row insertion, the canonical definition is written to `datasets.spec_definition`.
   - If the row already exists, `identity_bytes` are read back and compared byte-for-byte with incoming canonical bytes.
   - When bytes match, the identifier is checked against `datasets.identity_quarantine`. If quarantined, it raises `IdentityQuarantined`. If unquarantined, it writes to `datasets.spec_definition` if missing and returns `RegisteredDefinition(inserted=False)`.
   - When bytes diverge, held and incoming bytes are written to `datasets.identity_quarantine` in an independent transaction. The system logs `identity_collision_quarantined` and raises `IdentityCollision`.
4. **Collection Registration**: `register_collection` processes member definitions sequentially through `register_definition`. It derives `collection_id` over verified member `spec_id` values and executes the insert, byte comparison, and quarantine workflow for the collection and its member mappings in `datasets.collection_member`.

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

    def read_collection(
        self, collection_id: str
    ) -> StoredCollection | None: ...
```

#### Read Semantics

- **Quarantine Check**: Queries verify `datasets.identity_quarantine`. When the requested `spec_id` or `collection_id` is quarantined, `IdentityQuarantined` is raised immediately.
- **Missing Records**: Unknown identifiers return `None` (or omit the key in `read_definitions`).
- **Version Resolution**: When `schema_version` is omitted in `read_definition` or `read_definitions`, the reader selects the highest available `schema_version` for that `spec_id`.
- **Batch Reads**: `read_definitions` executes a single batched SQL query using `IN (...)`.

## Error Hierarchy and Wire Mapping

All domain exceptions inherit from `SpecRegistryError`. Store operations catch database driver exceptions, suppressing raw driver messages and exposing SQLSTATE codes when present.

| Exception | Cause / Condition | HTTP Status (in `datasets_api`) |
| --- | --- | --- |
| `SpecIdMismatch` | Client-provided `spec_id` differs from server-recomputed identity. | `422 Unprocessable Entity` (`spec_id_mismatch`) |
| `IdentityCollision` | Stored `identity_bytes` differ from incoming canonical bytes for the same id. Both are quarantined. | `500 Internal Server Error` (`identity_collision`) |
| `IdentityQuarantined` | Identifier is present in `datasets.identity_quarantine`. Requests for this ID are refused. | `500 Internal Server Error` (`identity_quarantined`) |
| `EmptyCollectionMembers` | Collection registration attempted with zero members. Wire requests validate non-empty members via schema. | Unhandled domain exception (HTTP door enforces 400 via `RegisterRequest`) |
| `SpecRegistryUnavailable` | Database connection pool exhausted (`PoolTimeoutError`). Transient condition. | `503 Service Unavailable` (`service_unavailable`) |
| `SpecRegistryStoreError` | Database query failure or connection error. | `500 Internal Server Error` |

## Integration with `dfo-spec-kit` and `datasets_api`

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
|  - mint_collection_id()                - LiveSpecReader                        |
|  - collection_id_projection()          - Quarantine handling                   |
+-----------------------+--------------------------------+-----------------------+
                        |                                |
                        | Uses canonicalization          | Reads/Writes
                        v                                v
+---------------------------------------+  +-------------------------------------+
|             dfo-spec-kit              |  |            dataset_store            |
|  - identity.build_projection          |  |  (PostgreSQL tables in              |
|  - identity.canonicalize              |  |   datasets.* schema)                |
|  - identity.mint_spec_id              |  +-------------------------------------+
+---------------------------------------+
```

### Integration Points

- **`dfo-spec-kit`**:
  - `spec_registry.identity` acts as an adapter over `dfo_spec_kit.identity`.
  - Normalisation, projection building, and canonical serialization conform to RFC 8785 (JSON Canonicalization Scheme) implemented in `dfo-spec-kit`.
  - Spec ID minting uses `dfo_spec_kit.identity.mint_spec_id`. Collection ID canonicalisation reuses `dfo_spec_kit.identity.canonicalize`.
- **`datasets_api`**:
  - **Register Door (`POST /api/datasets/register`)**: Calls `LiveSpecRegistry.register_definition` or `LiveSpecRegistry.register_collection` on a dedicated `ThreadPoolExecutor` with semaphore-gated backpressure. Maps domain exceptions to standard RFC 9457 problem details envelopes.
  - **Definition Retrieval Door (`GET /api/datasets/{id}`)**: Uses `LiveSpecReader.read_definition` (for `spec_` IDs) and `LiveSpecReader.read_collection` / `read_definitions` (for `col_` IDs) to resolve canonical definitions prior to rehydration and materialisation.
  - **Mint Door (`POST /api/datasets`)**: Persists definitions and collection groupings to the registry during artifact materialisation workflows.
  - **Threading**: All `spec_registry` operations are synchronous. Callers in asynchronous contexts dispatch calls to dedicated thread poo
