# artifact_store

`artifact_store` manages blob storage, content addressing, inventory metadata, and signed URL generation for dataset artifacts.

## Storage Model and Content Addressing

Artifacts are identified by their SHA-256 digest in the format `sha256:<hex_digest>`. The inventory location of each (digest, tier) names the GCS object that holds the bytes: the digest itself, or `<digest>#r<renewal>` after a working-tier renewal. `get()` hashes the bytes it reads against the digest whatever the object is called.

Artifacts are organized into two storage tiers:

| Tier | Enum | Purpose | Lifecycle Policy |
| --- | --- | --- | --- |
| Working | `Tier.working` | Ephemeral outputs from materialization | 30-day age-based deletion |
| Pinned | `Tier.pinned` | Preserved versions from snapshot and pin operations | 90-day soft delete, no automated expiry |

Buckets enforce uniform bucket-level access without per-object ACLs.

## Write Idempotency and Deduplication

The `put()` method uploads raw bytes with the precondition `if_generation_match=0`.

```python
from artifact_store import LiveArtifactInventory, LiveArtifactStore, Tier
from dataset_store import DatasetStore

# dataset_store is an initialized DatasetStore instance
inventory = LiveArtifactInventory(store=dataset_store)
store = LiveArtifactStore(
    buckets={
        Tier.working: "dev-fin-observatory-datasets-working",
        Tier.pinned: "dev-fin-observatory-datasets-pinned",
    },
    inventory=inventory,
)

stored = store.put(data=b"col1,col2\n1,2\n", tier=Tier.working)
```

1. The digest is computed as `sha256:<hexdigest>` over the raw bytes.
2. The blob is uploaded to the tier bucket under the digest with `if_generation_match=0`, so new bytes land with no inventory read.
3. If that name is taken, GCS returns HTTP 412 `PreconditionFailed`, and the store reads the tier's inventory location. When the location names a renewed object, the store uploads under that name the same way.
4. A 412 under the location's name is a deduplication hit, and the store sets `StoredArtifact.deduplicated = True`, unless the working location is past retention (see renewal below).
5. After the object exists in GCS, the store commits a record to the inventory repository, or, for a view from `deferred(batch)`, adds the record to the `InventoryBatch` for the caller to write in its own transaction (`inventory.write_entry`). The Datasets serve path writes every artifact's record in the transaction that links it to its version.
6. Server-side promotion via `copy()` uses the same `if_generation_match=0` precondition when copying an artifact between tiers (e.g. from `Tier.working` to `Tier.pinned`).

### Streaming writes: `open_put`

`open_put()` streams bytes into a sink while hashing them incrementally, so a caller never holds the whole artifact in memory. An artifact under 16 MiB stays buffered and commits in a single upload under its object name, the same path `put()` uses. One that crosses 16 MiB spills to a staging object in the working bucket's `staging/` prefix, then is server-side copied to its object name with `if_generation_match=0` once the digest is known. Staging objects are removed by the working bucket's one-day lifecycle rule on `staging/`, since neither this module nor the runtime identity it executes under holds a delete grant on either bucket.

## Inventory Tracking

Inventory state is stored in PostgreSQL through `LiveArtifactInventory` across two tables in the `datasets` schema (migrations `0003_artifact_inventory.sql` and `0025_artifact_location_renewal.sql`):

- `datasets.artifact_inventory`: Unique record per `artifact_digest` tracking `size_bytes` and `created_at`.
- `datasets.artifact_location`: Composite record on `(artifact_digest, tier)` tracking `bucket`, `object_name`, `stored_at`, and `renewal`, the highest renewal name `<digest>#r<renewal>` taken for it (0 when none).

Writes to `datasets.artifact_location` update `stored_at` only when `uploaded=True` (fresh write), and a fresh write in the location's bucket also moves `object_name` to the object it uploaded, since those bytes are the newest copy. That happens when the lifecycle rule has deleted the bare digest of a renewed location: the next write lands under the bare digest, and the location returns to it with its `renewal` kept. Deduplicated writes leave `stored_at` untouched to preserve the original retention window. A deduplication that finds no location row hit an object of unknown age, such as one whose inventory write failed, so the store reads the object's creation time (a metadata read) and records it as `stored_at`.

### Renewal by new object name

Runtime identities can only create objects in the working bucket, and its lifecycle rule deletes each object at age 30 with no timing guarantee. A write whose working location is past retention and whose object still exists therefore uploads the bytes again under `<digest>#r<renewal + 1>`, and `ArtifactInventory.renew()` moves the location to that name with `stored_at` set to now, only after the upload returns. An object with no location row whose creation time is past retention renews the same way under `<digest>#r1`, and `renew()` creates the row. A deduplication against expired bytes is never recorded, so it never becomes servable. A renewal name that is already taken holds this digest's bytes, since only the store writes `<digest>#r<n>` and it does so after hashing, left by a concurrent renewer or by a renewal whose `renew()` failed. The store reads that object's creation time: within retention, `renew()` moves the location to the name with `stored_at` set to that time and the write returns `deduplicated=True`; past retention, the write moves on to the next name, up to three. A location already at that renewal or later stays, so an uploader and an adopter of one name converge on one location, and a renewal whose inventory write failed is recovered by the next write. `get()` rehashes the bytes whatever the object is called. Pinned locations never renew.

The renewed object is safe for 30 days from its own creation. The previous object stays in the bucket until the lifecycle rule deletes it, and the reconciliation sweep reports it as orphaned until then.

## Signed URL Generation

`LiveArtifactStore.signed_location_url()` generates single-object V4 signed GET URLs using the IAM Credentials API (`signBlob`) for a location the caller already read.

```python
from datetime import timedelta

url = store.signed_location_url(
    "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    tier=Tier.working,
    bucket=location.bucket,
    object_name=location.object_name,
    ttl=timedelta(minutes=15),
    response_type="text/csv",
    response_disposition='attachment; filename="data.csv"',
)
```

Signing reads no inventory. A caller passes the bucket and object name of the location it holds, such as a snapshot read's location or the `StoredArtifact` a `put()` returned, so a renewed location is signed under its new object name.

### IAM signBlob Routing

Cloud Run workloads execute under a service account without downloadable private keys. The signing workflow routes signatures through IAM:

1. Application Default Credentials, discovered once per process and refreshed only when the token is missing or expired, provide an OAuth access token and service account email.
2. The `roles/iam.serviceAccountTokenCreator` role on the service account authorizes the `iamcredentials.googleapis.com` `signBlob` RPC.
3. `Blob.generate_signed_url` receives the access token and service account email, invoking the remote `signBlob` endpoint.

With `STORAGE_EMULATOR_HOST` set, which `google-cloud-storage` also reads to send every call to a GCS emulator, and `DFO_LOCAL_STACK=1`, which only the local stack sets, `sign_get_url` skips credentials and signing and returns the object's media link, with the `response-content-type` and `response-content-disposition` overrides in its query. The link is on `STORAGE_EMULATOR_LINK_BASE` when that is set and on the emulator otherwise. The local stack in `scripts/local-stack/` sets it to the API's origin, which serves the emulator's media path. `STORAGE_EMULATOR_HOST` set without `DFO_LOCAL_STACK=1` raises `StorageEmulatorRefused` when a `LiveArtifactStore` is built and on every signing, so a deployment that sets it by mistake fails at start-up.

### URL Constraints and Caching

- `SIGNED_URL_TTL_FLOOR`: Minimum TTL is 1 second (`timedelta(seconds=1)`). Sub-second TTL values and non-positive TTLs (`<= 0`) raise `ValueError` because V4 signing serializes expiry to whole seconds.
- `SIGNED_URL_TTL_CEILING`: Maximum TTL is 1 hour (`timedelta(hours=1)`). Requests exceeding 1 hour raise `ValueError`. The underlying GCS V4 protocol limit is `V4_MAX_TTL = timedelta(days=7)`.
- `SIGNED_URL_REUSE_WINDOW`: Generated URLs are cached in memory for 1 minute (`timedelta(minutes=1)`) keyed by `(tier, digest, bucket, object_name, ttl_seconds, response_type, response_disposition)`.
- Explicit Response Headers: `response_type` and `response_disposition` are mandatory parameters signed into the URL query string as `response-content-type` and `response-content-disposition`.

## ArtifactStore Protocol

The `ArtifactStore` protocol defines synchronous operations over content-addressed storage:

- `put(data: bytes, *, tier: Tier) -> StoredArtifact`: Writes bytes under their SHA-256 digest, or the location's object name when the digest is taken, with precondition-based idempotency and working-tier renewal.
- `get(*, artifact_digest: str, tier: Tier) -> bytes`: Reads the object the inventory location names and re-verifies integrity against the digest.
- `signed_location_url(digest: str, *, tier: Tier, bucket: str, object_name: str, ttl: timedelta, response_type: str, response_disposition: str) -> str`: Generates a cached V4 signed GET URL for a location the caller holds, without an inventory read.
- `copy(digest: str, *, from_tier: Tier, to_tier: Tier) -> StoredArtifact`: Copies a blob between tier buckets server-side.
- `list(*, tier: Tier) -> frozenset[str]`: Lists the object names, digests and renewal names, present in the tier bucket.
- `deferred(batch: InventoryBatch) -> ArtifactStore`: A view whose writes add their inventory records to `batch` in place of recording them; reads, signing and the URL cache are the store's own.

## Error Classification and Context Sanitization

Exceptions inherit from `ArtifactStoreError`. Driver exceptions and query strings are sanitized to avoid logging sensitive signature parameters (`X-Goog-Signature`, `X-Goog-Credential`).

To prevent Starlette's `BaseHTTPMiddleware` from resurrecting driver exceptions containing signed URLs via `__context__`, all wrap sites catch exceptions, extract safe metadata, and raise typed errors only after the `except` block exits.

| Exception | Cause |
| --- | --- |
| `ArtifactWriteError` | Storage write or server-side copy failed due to unexpected driver errors. |
| `ArtifactSigningError` | IAM `signBlob` or URL signature RPC failed. Contains `error_class_name` and HTTP `status_code`. |
| `ArtifactInventoryError` | Database write or query on inventory tables failed after bytes were stored. |
| `ArtifactRecordConflictError` | Incoming `bucket` or `size_bytes` conflicts with an existing record for the same digest. |
| `ArtifactLocationDriftError` | Recorded inventory location is not in the tier's configured bucket. |
| `ArtifactNotFoundError` | Artifact record or GCS object does not exist. |
| `ArtifactIntegrityError` | Bytes retrieved by `get()` do not match the expected SHA-256 digest. |

When a database write fails after GCS upload succeeds, `ArtifactInventoryError` is raised and the application logs an entry containing the marker:

```python
ARTIFACT_INVENTORY_WRITE_FAILED_MARKER = "artifact_inventory_write_failed"
```

## Testing Doubles

Unit and integration tests use test doubles from `tests.artifact_store.conftest` to run without GCP credentials or network access:

- `FakeArtifactStore`: In-memory `ArtifactStore` recording call order in `store.calls` and headers in `store.signed_headers`.
- `RecordingArtifactInventory`: Call-recording `ArtifactInventory` fake for verifying inventory commit sequencing without a database.
- `FakeGCSClient`, `FakeGCSBucket`, `FakeGCSBlob`: In-memory GCS doubles for testing `LiveArtifactStore` upload, deduplication, and copy behaviors.
- `FakeArtifactInventoryDatasetStore`: SQLite-backed `DatasetStore` double holding the inventory tables of migrations `0003_artifact_inventory.sql` and `0025_artifact_location_renewal.sql`.

```python
from datetime import timedelta

from artifact_store import Tier
from tests.artifact_store.conftest import FakeArtifactStore

store = FakeArtifactStore()

# Stores bytes in memory and simulates precondition deduplication
stored = store.put(b"test-payload", tier=Tier.working)
assert stored.deduplicated is False

# Repeated write returns deduplicated status
stored_again = store.put(b"test-payload", tier=Tier.working)
assert stored_again.deduplicated is True

# Reads back raw bytes
data = store.get(artifact_digest=stored.digest, tier=Tier.working)
assert data == b"test-payload"

# Returns deterministic mock signed URL
url = store.signed_location_url(
    stored.digest,
    tier=Tier.working,
    bucket=stored.bucket,
    object_name=stored.object_name,
    ttl=timedelta(minutes=10),
    response_type="text/plain",
    response_disposition="inline",
)
assert "X-Goog-Signature=fake0000signature" in url
```

## Configuration

Environment variables configure bucket bindings across services:

| Environment Variable | Setting Property | Required | Description |
| --- | --- | --- | --- |
| `DATASETS_ARTIFACT_BUCKET_WORKING` | `artifact_bucket_working` | Yes | GCS bucket name for `Tier.working` |
| `DATASETS_ARTIFACT_BUCKET_PINNED` | `artifact_bucket_pinned` | Yes | GCS bucket name for `Tier.pinned` |

The `LiveArtifactStore` constructor requires a dictionary mapping every `Tier` member to a bucket name:

```python
import os

from artifact_store import LiveArtifactStore, Tier

buckets = {
    Tier.working: os.environ["DATASETS_ARTIFACT_BUCKET_WORKING"],
    Tier.pinned: os.environ["DATASETS_ARTIFACT_BUCKET_PINNED"],
}
store = LiveArtifactStore(buckets=buckets, inventory=inventory)
```

`inventory` is the `ArtifactInventory` that records each stored artifact's digest, bucket and size.
