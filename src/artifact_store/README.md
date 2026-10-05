# artifact_store

`artifact_store` manages blob storage, content addressing, inventory metadata, and signed URL generation for dataset artifacts.

## Storage Model and Content Addressing

Artifacts are stored in Google Cloud Storage (GCS) keyed by their SHA-256 digest in the format `sha256:<hex_digest>`. The object name in GCS equals the content digest.

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
2. The blob is uploaded to the tier bucket with `if_generation_match=0`.
3. If the object exists, GCS returns HTTP 412 `PreconditionFailed`. The store treats this as a deduplication hit and sets `StoredArtifact.deduplicated = True`.
4. After the object exists in GCS, the store commits a record to the inventory repository.
5. Server-side promotion via `copy()` uses the same `if_generation_match=0` precondition when copying an artifact between tiers (e.g. from `Tier.working` to `Tier.pinned`).

### Streaming writes: `open_put`

`open_put()` streams bytes into a sink while hashing them incrementally, so a caller never holds the whole artifact in memory. An artifact under 16 MiB stays buffered and commits in a single upload under its digest, the same path `put()` uses. One that crosses 16 MiB spills to a staging object in the working bucket's `staging/` prefix, then is server-side copied to its digest name with `if_generation_match=0` once the digest is known. Staging objects are removed by the working bucket's one-day lifecycle rule on `staging/`, since neither this module nor the runtime identity it executes under holds a delete grant on either bucket.

## Inventory Tracking

Inventory state is stored in PostgreSQL through `LiveArtifactInventory` across two tables in the `datasets` schema (migration `0003_artifact_inventory.sql`):

- `datasets.artifact_inventory`: Unique record per `artifact_digest` tracking `size_bytes` and `created_at`.
- `datasets.artifact_location`: Composite record on `(artifact_digest, tier)` tracking `bucket`, `object_name`, and `stored_at`. Enforces `CHECK (object_name = artifact_digest)`.

Writes to `datasets.artifact_location` update `stored_at` only when `uploaded=True` (fresh write). Deduplicated writes leave `stored_at` untouched to preserve the original retention window.

Between `stored_at` + 30 days and the working bucket's lifecycle delete (age 30, applied within about a day), a re-render of the same bytes deduplicates against the old object and leaves its location expired. The version link stays unservable, so each request in that window renders again and a pin refuses. The first render after the lifecycle delete uploads afresh, resets `stored_at`, and the link serves again. An overwrite would close the window but needs `storage.objects.delete` on the working bucket, which the runtime identities do not hold.

## Signed URL Generation

`LiveArtifactStore.signed_get_url()` generates single-object V4 signed GET URLs using the IAM Credentials API (`signBlob`).

```python
from datetime import timedelta

url = store.signed_get_url(
    digest="sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    tier=Tier.working,
    ttl=timedelta(minutes=15),
    response_type="text/csv",
    response_disposition='attachment; filename="data.csv"',
)
```

### IAM signBlob Routing

Cloud Run workloads execute under a service account without downloadable private keys. The signing workflow routes signatures through IAM:

1. Application Default Credentials provide an OAuth access token and service account email.
2. The `roles/iam.serviceAccountTokenCreator` role on the service account authorizes the `iamcredentials.googleapis.com` `signBlob` RPC.
3. `Blob.generate_signed_url` receives the access token and service account email, invoking the remote `signBlob` endpoint.

### URL Constraints and Caching

- `SIGNED_URL_TTL_FLOOR`: Minimum TTL is 1 second (`timedelta(seconds=1)`). Sub-second TTL values and non-positive TTLs (`<= 0`) raise `ValueError` because V4 signing serializes expiry to whole seconds.
- `SIGNED_URL_TTL_CEILING`: Maximum TTL is 1 hour (`timedelta(hours=1)`). Requests exceeding 1 hour raise `ValueError`. The underlying GCS V4 protocol limit is `V4_MAX_TTL = timedelta(days=7)`.
- `SIGNED_URL_REUSE_WINDOW`: Generated URLs are cached in memory for 5 minutes (`timedelta(minutes=5)`) keyed by `(tier, digest, ttl_seconds, response_type, response_disposition)`.
- Explicit Response Headers: `response_type` and `response_disposition` are mandatory parameters signed into the URL query string as `response-content-type` and `response-content-disposition`.

## ArtifactStore Protocol

The `ArtifactStore` protocol defines synchronous operations over content-addressed storage:

- `put(data: bytes, *, tier: Tier) -> StoredArtifact`: Writes bytes under their SHA-256 digest with precondition-based idempotency.
- `get(*, artifact_digest: str, tier: Tier) -> bytes`: Reads raw bytes by content-addressed digest and re-verifies integrity against the digest.
- `signed_get_url(digest: str, *, tier: Tier, ttl: timedelta, response_type: str, response_disposition: str) -> str`: Generates a cached V4 signed GET URL.
- `copy(digest: str, *, from_tier: Tier, to_tier: Tier) -> StoredArtifact`: Copies a blob between tier buckets server-side.
- `list(*, tier: Tier) -> frozenset[str]`: Lists all object digests present in the tier bucket.

## Error Classification and Context Sanitization

Exceptions inherit from `ArtifactStoreError`. Driver exceptions and query strings are sanitized to avoid logging sensitive signature parameters (`X-Goog-Signature`, `X-Goog-Credential`).

To prevent Starlette's `BaseHTTPMiddleware` from resurrecting driver exceptions containing signed URLs via `__context__`, all wrap sites catch exceptions, extract safe metadata, and raise typed errors only after the `except` block exits.

| Exception | Cause |
| --- | --- |
| `ArtifactWriteError` | Storage write or server-side copy failed due to unexpected driver errors. |
| `ArtifactSigningError` | IAM `signBlob` or URL signature RPC failed. Contains `error_class_name` and HTTP `status_code`. |
| `ArtifactInventoryError` | Database write or query on inventory tables failed after bytes were stored. |
| `ArtifactRecordConflictError` | Incoming `bucket`, `object_name`, or `size_bytes` conflicts with an existing record for the same digest. |
| `ArtifactLocationDriftError` | Recorded inventory location does not match the configured bucket or digest object name. |
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
- `FakeArtifactInventoryDatasetStore`: SQLite-backed `DatasetStore` double running migration `0003_artifact_inventory.sql`.

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
url = store.signed_get_url(
    digest=stored.digest,
    tier=Tier.working,
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
