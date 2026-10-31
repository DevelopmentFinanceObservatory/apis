"""Datasets maintenance tasks, run as Cloud Run jobs under the runtime identity.

`datasets-maintenance <task>` runs one task and exits non-zero when it fails. Each task is a
subcommand registered in `main`.

`sweep` runs daily from Cloud Scheduler. It settles pin claims past their lease as
`sweep-export-claims` does, then sweeps both artifact tiers with the read-only
`reconciliation.sweep` at the current time, whatever the claims did, and writes one structured
`datasets_reconciliation_sweep` line per tier with its orphaned, missing, expired working and
unreferenced counts. Missing objects count only rows that should have bytes, so a working row
past `WORKING_TIER_RETENTION` counts as expired. For each non-empty orphaned, missing or
unreferenced leg the line also carries the sorted object names, digests for the unreferenced
leg, at most `SWEEP_NAMES_PER_LEG` of them, under `<leg>_sample`. The report repairs nothing. It
reads `DATASETS_ARTIFACT_BUCKET_WORKING`, `DATASETS_ARTIFACT_BUCKET_PINNED` and the runtime
database settings `dataset_store.provider.shared_pool` reads.

`sweep-export-claims` runs every five minutes from Cloud Scheduler. It settles each claim an
export or a pin left past its lease by itself (`retention_admission.sweep_copy_claims`), `held`
ones before `copying` ones, so a claim that fails leaves the rest settled. It starts no claim
whose worst case, a bucket read included, would run past `retention_admission.sweep_budget`, so
one run ends before the next. It writes one `datasets_copy_claims_swept` line with the settled,
failed and deferred counts, and exits non-zero when any failed or were deferred, so a backlog
alerts. Every settlement names the claim's generation, so a second run changes nothing. A refund
lands on the UTC day that spent the unit.

`age-operations` runs daily. It first fails every abandoned row of `datasets.operation`
(`Operation.abandoned`) as retryable `internal_error`, so a row nobody polls leaves `pending` or
`running`. It then deletes `done` and `failed` rows last updated more than
`WORKING_TIER_RETENTION` ago, the window the capabilities document publishes as
`operation_retention_days`, and writes one `datasets_operations_aged` line with both counts. It
reads the runtime database settings and `DATASETS_OPERATION_DEADLINE_SECONDS`, whose default is
the API's own.

`backfill-frame1` runs once per environment. For every version linking a pinned frame1 document
and no pinned canonical frame, it reads the document's frame (`store_errors.read_stored_frame`),
stores it as the canonical frame, checks the stored frame reads back as the same frame, copies
it to the pinned tier
and, in one transaction, links it under `render.CANONICAL_FRAME_RENDERING` and copies every
`artifact_reference` row the frame1 held under the version to it, so the sweep counts the
canonical frame referenced exactly as the frame1 was. A second run finds nothing to convert. It
writes one `datasets_frame1_backfill_failed` line per version it cannot convert and one
`datasets_frame1_backfilled` line with the converted and failed counts, exits 1 when any version
failed, and reads the settings `sweep` reads.

`settle` runs on every deploy, after the services move to the new image. Between the migration
job and the service move the previous image can still write a pin reference holding the raw
shared secret, so `settle` repeats the statements of migration 0026 that turn the secret into
its label (`SETTLE_STATEMENTS`). Each touches only a pin principal that is neither a label nor
an open caller's `ip:` key, so a repeat run changes nothing, and the table lock they run under
holds concurrent pins back until they finish. It runs them in one transaction, then writes one
`datasets_settled` line counting the pin references still holding a raw secret, the version rows holding no facts, and
the version rows whose output profile, in the column or the body, still names the party-kind
enrichment `kind`. Insert triggers repair the previous image's version rows, also for a request
an old revision finishes after this task: migration 0030's fills the facts of a row written with
only `immutable_body`, and migration 0033's renames the enrichment `party_kind`. The facts count
is therefore above 0 only for a row whose body holds neither `prepared_at` nor
`input_content_versions`, which `settle` leaves as it is, and the enrichment count only when
0033's trigger is missing. When any count is above 0 the line has `ERROR` severity and the task
exits 1. It reads only the runtime database settings.

`backfill-descriptions` runs on every deploy, after the services move and the code index they
run is recorded. It first fills the `continuations` of every published frame's version row
holding none, recorded before versions stored them or by the previous image until the services
move, oldest first, as a publish derives them (`run_backfill_continuations`), with one
`datasets_continuations_backfill_failed` `ERROR` line per version it cannot fill, tried again on
the next run, and one `datasets_continuations_backfilled` line counting the versions filled,
failed and left for the next run. It then writes a description record
with basis `backfill` for every dataset and frame version row that holds none: those made before
records existed, those the previous image writes until the services move, and those whose
stored record, of this image's schema version or an earlier one, fails validation, which it
replaces (`description_store.insert_description`). Collection versions are skipped, since a
record names only a `spec_` or `ops_` version. Versions recorded last come first, so the previous image's versions, whose artifacts every tier still holds, are
described before older ones that may fail for want of them. It reads each
version's stored facts, its series statistics artifact while a tier holds it, else its body's
inline statistics, an `ops_` version's stored frame, and the image's code index
(`code_index.shared.get_index`). Status counts come from the statistics through the fold a mint
uses, and a version's period count is unknown without a known grain. An `ops_` version whose
frame no tier holds gets a record whose series facts are unknown. Each record is inserted with
`ON CONFLICT DO NOTHING`, so a valid record a mint or publish wrote first stands and a second
run writes nothing. Versions that collapse no other come first, so a multi-contributor collapse is
derived from its base version's record (`backfill_record`), and from none when its base holds
no record. A stored warning code whose fact cannot be stated, a `withheld_by_registry` or
`un_system_unavailable` message naming nothing in a format its producer wrote or a choice,
conversion gap or projection the trace holds no step for, is left out of the record, which
serves its steps as a gap when the code was a step, with one
`datasets_description_backfill_warning_unstated` line naming the version and code. A version it
cannot backfill is skipped with one `ERROR` line `datasets_description_backfill_failed` naming
it, holding the traceback and saying whether the failure was transient, and is tried again on
the next run (`run_backfill_descriptions` gives the rule). It starts no version once
`BACKFILL_DESCRIPTIONS_BUDGET` has passed, so the run ends before the Job's timeout and the next
run carries on from the versions still without a record. The run ends with one
`datasets_descriptions_backfilled` line counting the versions written, failed permanently,
failed transiently and left for the next run. It exits 1 when the store is unreachable, or when
it read no version and at least one failed for a cause a retry clears (`_clears_on_retry`).
It reads the settings `sweep` reads.

`backfill-warnings [--dry-run]` rewrites each stored `coverage_below_scope` member sentence
that names members by ref into the counts and member fields a mint writes, in the `warnings`
column and in `immutable_body`, reading each row's members from the graph
(`warning_backfill`), one row per transaction, starting none once
`BACKFILL_DESCRIPTIONS_BUDGET` has passed. It writes one `datasets_warning_backfill_failed`
`ERROR` line per row it leaves, saying whether a later run can get past the failure, and one
`datasets_warnings_backfilled` line with the counts. It stops at a store outage
(`_clears_on_retry`), and exits 1 when a row failed transiently, lost the compare or was left.
A row refused for good is refused again on every run from what it stores, so it is reported
each time and sets no exit code. `--dry-run` writes nothing, adds one `datasets_warning_backfill_row` line
per row, and compares its split with the stored one on rows a mint already split. When no row
matches it opens no release and builds no client. It reads the runtime database settings,
`DatasetsSettings`, `DATASETS_GRAPH_TOKEN`, `DATASETS_SDMX_TIMEOUT_SECONDS`, the served graph
release and `ORIGIN_CATALOGUE_URI`.

`backfill-frame-summed` runs by hand once the deploy that applies migration 0038 has moved the
services to the new image, since the previous image records derived rows without `frame_summed`
until then. A later run is safe and fills any row the previous image wrote after an earlier one.
It fills `frame_summed` on each derived version row recorded without it whose record cannot say
what its own frame holds (`recorded_rendering.own_frame_mode`): a party-totals cut minted with a
groupby, which otherwise renders from its base frame alone. It reads the base version's stored
frame from any tier holding its bytes, cuts it as the version's trace recorded
(`party_totals_left_out`, an entry without a `provenance` cutting its parties from the rows of
every dataset, `PartyTotalsCut.of_trace_entries`) and hashes the cut frame. A frame is the version's own when its hash
equals the version's `frame_sha256`, or when the version minted from that hash
(`mint_version_from_digest`) is the row's `content_version`, which a derived version mints from
its frame hash alone, so a row recorded before versions stored the hash matches too. The cut
frame being the version's own sets `frame_summed` false. Otherwise it collapses the cut frame by
each groupby the version's files record, under the served graph release's code index, and the
collapse being the version's own sets it true. A row whose base frame no tier holds, or which
neither frame matches, as after a code index change, stays NULL. Rows are read
`FRAME_SUMMED_PAGE` at a time, and the update holds each row to `frame_summed IS NULL`, so no
run changes a value already set. It writes one `datasets_frame_summed_backfill_failed` `ERROR`
line per version that fails and one `datasets_frame_summed_backfilled` line counting each
`FrameSummedOutcome` and the failures, exits 1 when any version failed, and reads the settings
`sweep` reads. No deploy runs it, since a row that stays NULL would be read and cut again on
every deploy.

`diff-recipes --report PATH` runs by hand against an environment to check the analysis runner
against its stored frames. It computes every recipe in `datasets.frame_recipe` again on its own root versions through
the analysis runner (`recipe_replay.replay_recipe`) and compares the result frame's `cv_` and
output digest with the stored frame's. It writes nothing to any store: roots are read only where
their version already holds the frame or file (`recipe_replay.StoredRootSource`), nothing renders,
and nothing is published, pinned or recorded. It writes one JSON line per recipe that is not
equal to `PATH` (default `recipe-diff.jsonl`), each `{spec_id, content_version, family, outcome,
stored, computed, detail}`, and one `datasets_recipes_diffed` line holding the count of each
outcome (`equal`, `differs`, `roots_unreadable`, `frame_unreadable`) per family (`compare`,
`sum_of`, `other`) and in total. It exits 1 when any recipe differs, with that line at `ERROR`
severity. It reads the settings `sweep` reads.

`record-recipe-golden --out DIR [--limit N] [--max-bytes B] [--origin NAME]` writes up to `N`
(default 50) recipes whose roots and stored frame it can read, with those frames, into `DIR`, one
case directory each labelled `NAME` (default `dfo-dev`) (`recipe_golden.record_case`), skipping
a recipe whose frames together pass `B` bytes (default 2,000,000). It reads the stores as `diff-recipes` does and writes only under
`DIR`, one `datasets_recipe_golden_recorded` line counting the cases written and skipped.

`normalise-recipes [--dry-run]` runs by hand once, after the website's share migration job.
For each recipe in `datasets.frame_recipe` whose root `options.columns` are not sorted and
listed once, or whose root `options.groupby.by` is not sorted, as `analysis_roots.mint_options`
writes them, it sorts them, recomputes `normal_form_hash` and `replay_key` as the analysis route does, moves the card revisions drawing that frame from the
old hash to the new one in the Presentation database
(`presentation_store.store.rehash_derived_rendering`), then rewrites the recipe row. Each
database commits on its own, revisions first, so a run stopped between them leaves the recipe
to the next run. It skips a recipe whose stored hashes the stored recipe does not reproduce and
one whose new replay key another frame's recipe holds (`RecipeSkip`), with one
`datasets_recipe_normalise_skipped` line each. It writes one `datasets_recipes_normalised` line
with the counts and exits 1 when it skipped any. `--dry-run` counts and changes nothing, and a
second run changes nothing. It reads the runtime database settings of both stores.

`take-down-share SHARE_ID` runs by hand to take a share down for its owner, as for the ids the
website's `share_cleanup` table holds. It takes the share down as the owner's delete with its
write token does (`ShareStore.delete_share`) and needs no token. It writes one
`presentation_share_taken_down` line, or `presentation_share_take_down_not_found` when no share
is up at the id, as for one taken down already, and exits 0 for both, so a re-run changes
nothing. An id that is not a share id gets one `presentation_share_take_down_refused` `ERROR`
line and exits 1 without opening the store. It reads the Presentation store's runtime database
settings.

Each task builds only what it uses from the environment the Job sets, so none reads a setting
of the API surfaces, such as `DFO_PUBLIC_SITE_URL`.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from datetime import datetime, timedelta
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, cast

import pg8000.exceptions
import pyarrow as pa
import pyarrow.parquet as pq
from google.api_core.exceptions import RetryError, ServerError, TooManyRequests
from google.auth.exceptions import TransportError
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from artifact_publication.jobs import emit
from artifact_store import (
    ArtifactInventoryError,
    ArtifactNotFoundError,
    ArtifactStore,
    ArtifactStoreError,
    LiveArtifactInventory,
    LiveArtifactStore,
    Tier,
)
from code_index.query import CodeIndex
from code_index.shared import get_index
from dataset_store import DatasetStore, Transaction
from dataset_store.connection import default_is_broken
from dataset_store.errors import PoolTimeoutError
from dataset_store.provider import close_shared_pool, shared_pool
from dataset_store.snapshot import (
    WORKING_TIER_RETENTION,
    FrameLineage,
    LiveSnapshotReader,
    ManifestFacts,
    StoredArtifactLocation,
    StoredVersion,
    VersionState,
    select_servable_locations,
)
from dataset_store.store import LiveDatasetStore
from datasets_api.allowance import LiveAllowanceLedger
from datasets_api.analysis import FrameUnavailable
from datasets_api.config import (
    DEFAULT_OPERATION_DEADLINE_SECONDS,
    DatasetsConfigError,
    DatasetsSettings,
)
from datasets_api.description_record import backfill_record, computed_inputs
from datasets_api.description_store import (
    insert_description,
    read_description,
    validated_record,
)
from datasets_api.lineage_continuations import derived_continuations, lineage_inputs
from datasets_api.manifest import provenance_refs_of
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.operation_store import (
    Clock,
    delete_terminal_operations,
    fail_abandoned_operations,
    utc_now,
)
from datasets_api.party_totals import TRACE_KEY as PARTY_TOTALS_TRACE_KEY
from datasets_api.party_totals import PartyTotalsCut, cut_frame
from datasets_api.recipe_golden import DEFAULT_MAX_BYTES, record_case
from datasets_api.recipe_replay import ReplayOutcome, StoredRootSource, replay_recipe
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.reconciliation import sweep
from datasets_api.recorded_rendering import (
    derived_version_sums,
    own_frame_mode,
    recorded_mintings,
)
from datasets_api.render import (
    CANONICAL_FRAME_RENDERING,
    FrameMode,
    entity_roles,
    groupby_collapse_sha256,
    is_statistics_rendering,
    render_and_store_canonical_frame,
    stored_frame_location,
)
from datasets_api.retention_admission import sweep_copy_claims
from datasets_api.serve import refs_from_resolved_stat_vars
from datasets_api.store_errors import (
    DatasetsStoreUnavailable,
    StoredFrameUnreadable,
    read_stored_frame,
)
from datasets_api.version_store import LiveVersionRecordWriter, rendering_key
from datasets_api.versions import ArtifactRecord
from datasets_api.warning_backfill import (
    Readers,
    emit_nothing_backfilled,
    run_backfill_warnings,
    select_rows,
)
from dc_sdmx import SdmxClient
from description.facts import SCHEMA_VERSION, DescriptionRecord
from dfo_analysis_engine.recipe import normal_form, normal_form_hash, replay_key, root_identity
from dfo_platform.store_errors import extract_sqlstate, store_failure_is_permanent
from graph_release.context import bound_served_release
from graph_release.service import get_release_service
from materialise.content_version import mint_version_from_digest
from materialise.continued_series import ContinuedSeries, continued_series_of_trace
from materialise.formats import (
    FRAME1_MEDIA_TYPE,
    Frame,
    frame_entity_columns,
)
from materialise.reshape import RollupNotSummableError
from materialise.statistics import (
    SERIES_STATISTICS_SCHEMA,
    SeriesFigures,
    SeriesStatisticsSink,
)
from origin_catalogue import bound_catalogue
from origin_catalogue.graph import build_graph_client
from origin_catalogue.service import get_catalogue_service
from series_coverage.shared import get_coverage_service

if TYPE_CHECKING:
    from presentation_store import ShareStore


class MaintenanceSettings(BaseSettings):
    """The two artifact buckets, read under the API's own `DATASETS_` names."""

    model_config = SettingsConfigDict(env_prefix="DATASETS_")

    artifact_bucket_working: str = Field(min_length=1)
    artifact_bucket_pinned: str = Field(min_length=1)


SWEEP_NAMES_PER_LEG = 20
"""The most object names, or digests, a sweep line carries for one leg."""


def run_copy_claim_sweep(
    *, store: DatasetStore, artifact_store: ArtifactStore, now: Clock = utc_now
) -> int:
    """Settle the pin claims past their lease (`retention_admission.sweep_copy_claims`), emit
    the counts and return how many failed or were left for the next run, so a backlog fails the
    task."""
    settled, failed, deferred = sweep_copy_claims(
        allowances=LiveAllowanceLedger(store=store, limits={}),
        artifacts=artifact_store,
        now=now(),
    )
    emit(
        "datasets_copy_claims_swept",
        settled_claims=settled,
        failed_claims=failed,
        deferred_claims=deferred,
    )
    return failed + deferred


def run_sweep(*, store: DatasetStore, artifact_store: ArtifactStore, now: Clock = utc_now) -> int:
    """Settle expired pin claims, then sweep both tiers whatever that did, emit one line per
    tier with each leg's count and, for each non-empty leg, its first `SWEEP_NAMES_PER_LEG`
    names in sorted order, and return how many claims failed or were left."""
    try:
        failed = run_copy_claim_sweep(store=store, artifact_store=artifact_store, now=now)
    except Exception as exc:
        emit("datasets_maintenance_failed", severity="ERROR", task="sweep", error=str(exc))
        failed = 1
    for report in sweep(store=store, artifact_store=artifact_store, now=now()):
        legs = {
            "orphaned_objects": report.orphaned_objects,
            "missing_objects": report.missing_objects,
            "unreferenced_pinned_digests": report.unreferenced_pinned_digests,
        }
        emit(
            "datasets_reconciliation_sweep",
            tier=report.tier.value,
            expired_working_locations=len(report.expired_working_locations),
            **{leg: len(names) for leg, names in legs.items()},
            **{
                f"{leg}_sample": sorted(names)[:SWEEP_NAMES_PER_LEG]
                for leg, names in legs.items()
                if names
            },
        )
    return failed


def _sweep() -> int:
    return _run_over_artifacts("sweep", run_sweep)


def _sweep_copy_claims() -> int:
    return _run_over_artifacts("sweep-export-claims", run_copy_claim_sweep)


def _run_over_artifacts(task: str, run: Callable[..., int | None]) -> int:
    """Run `run` over the runtime store and both artifact buckets, closing the pool after.
    Exits 1 when `run` returns a non-zero failure count."""
    try:
        settings = MaintenanceSettings()
    except ValidationError as exc:
        emit("datasets_maintenance_failed", severity="ERROR", task=task, error=str(exc))
        return 1
    store = LiveDatasetStore(connections=shared_pool())
    try:
        failures = run(
            store=store,
            artifact_store=LiveArtifactStore(
                buckets={
                    Tier.working: settings.artifact_bucket_working,
                    Tier.pinned: settings.artifact_bucket_pinned,
                },
                inventory=LiveArtifactInventory(store=store),
            ),
        )
    finally:
        close_shared_pool()
    return 1 if failures else 0


class OperationAgeingSettings(BaseSettings):
    """The operation deadline, read under the name the API and the worker read it by."""

    model_config = SettingsConfigDict(env_prefix="DATASETS_")

    operation_deadline_seconds: int = Field(default=DEFAULT_OPERATION_DEADLINE_SECONDS, ge=1)


def run_age_operations(*, store: DatasetStore, deadline: timedelta, now: Clock = utc_now) -> None:
    """Fail abandoned operations, delete terminal ones past the published retention, and emit
    both counts."""
    failed = fail_abandoned_operations(store=store, deadline=deadline, now=now)
    deleted = delete_terminal_operations(store=store, older_than=WORKING_TIER_RETENTION, now=now)
    emit(
        "datasets_operations_aged",
        failed_abandoned_operations=failed,
        deleted_operations=deleted,
    )


def _age_operations() -> int:
    try:
        settings = OperationAgeingSettings()
    except ValidationError as exc:
        emit("datasets_maintenance_failed", severity="ERROR", task="age-operations", error=str(exc))
        return 1
    try:
        run_age_operations(
            store=LiveDatasetStore(connections=shared_pool()),
            deadline=timedelta(seconds=settings.operation_deadline_seconds),
        )
    finally:
        close_shared_pool()
    return 0


_LABELLED = "'^(secret-sha256:[0-9a-f]{16}|ip:[0-9a-f]{32})$'"
"""Every principal the pin door writes, as a POSIX pattern: a `Principal.label` or a caller key
(`dfo_platform.caller.address_key`). A pin principal it does not match is the raw secret the
image before migration 0026 stored."""

SETTLE_STATEMENTS: tuple[str, ...] = (
    "LOCK TABLE datasets.artifact_reference IN SHARE ROW EXCLUSIVE MODE",
    f"""UPDATE datasets.artifact_reference AS labelled
SET created_at = LEAST(labelled.created_at, raw.created_at),
    removed_at = CASE
        WHEN labelled.removed_at IS NULL OR raw.removed_at IS NULL THEN NULL
        ELSE GREATEST(labelled.removed_at, raw.removed_at)
    END
FROM datasets.artifact_reference AS raw
WHERE raw.reference_kind = 'pin'
  AND raw.principal !~ {_LABELLED}
  AND labelled.principal
      = 'secret-sha256:' || left(encode(sha256(convert_to(raw.principal, 'UTF8')), 'hex'), 16)
  AND labelled.artifact_digest = raw.artifact_digest
  AND labelled.subject_id = raw.subject_id
  AND labelled.content_version = raw.content_version
  AND labelled.reference_kind = raw.reference_kind""",
    f"""DELETE FROM datasets.artifact_reference AS raw
USING datasets.artifact_reference AS labelled
WHERE raw.reference_kind = 'pin'
  AND raw.principal !~ {_LABELLED}
  AND labelled.principal
      = 'secret-sha256:' || left(encode(sha256(convert_to(raw.principal, 'UTF8')), 'hex'), 16)
  AND labelled.artifact_digest = raw.artifact_digest
  AND labelled.subject_id = raw.subject_id
  AND labelled.content_version = raw.content_version
  AND labelled.reference_kind = raw.reference_kind""",
    f"""UPDATE datasets.artifact_reference
SET principal = 'secret-sha256:' || left(encode(sha256(convert_to(principal, 'UTF8')), 'hex'), 16)
WHERE reference_kind = 'pin'
  AND principal !~ {_LABELLED}""",
)
"""The statements of `0026_artifact_reference_principal_label.sql`, which the migration
explains, matching only a principal `_LABELLED` does not, so an open caller's `ip:` key and a
second run are left alone. The lock queues pins until the transaction ends: a pin landing between
the fold and the rewrite would otherwise make the rewrite collide with its row, or leave a raw
row the count reports."""


def run_settle(*, store: DatasetStore) -> int:
    """Run `SETTLE_STATEMENTS` in one transaction, emit the pin references holding a raw
    secret, the version rows holding no facts and the version rows whose output profile still
    names the enrichment `kind`, at `ERROR` when any exists, and return their number."""
    with store.transaction() as tx:
        for statement in SETTLE_STATEMENTS:
            tx.execute(statement)
        factless = _count(
            tx,
            "SELECT count(*) AS count FROM datasets.version_record "
            "WHERE prepared_at IS NULL AND input_content_versions IS NULL",
        )
        raw = _count(
            tx,
            "SELECT count(*) AS count FROM datasets.artifact_reference "
            f"WHERE reference_kind = 'pin' AND principal !~ {_LABELLED}",
        )
        legacy_party_kind = _count(
            tx,
            "SELECT count(*) AS count FROM datasets.version_record "
            "WHERE output_profile -> 'enrichments' @> '[{\"field\": \"kind\"}]' "
            "OR immutable_body -> 'output_profile' -> 'enrichments' "
            '@> \'[{"field": "kind"}]\'',
        )
    unsettled = factless + raw + legacy_party_kind
    emit(
        "datasets_settled",
        severity="ERROR" if unsettled else "INFO",
        factless_versions=factless,
        raw_pin_principals=raw,
        legacy_party_kind_profiles=legacy_party_kind,
    )
    return unsettled


def _count(tx: Transaction, sql: str) -> int:
    row = tx.fetch_one(sql)
    # `count(*)` without `GROUP BY` returns exactly one row.
    assert row is not None
    return int(cast(int, row["count"]))


def _settle() -> int:
    try:
        unsettled = run_settle(store=LiveDatasetStore(connections=shared_pool()))
    finally:
        close_shared_pool()
    return 1 if unsettled else 0


def _pinned_frame1_versions(store: DatasetStore) -> list[tuple[str, str, str]]:
    """`(subject_id, content_version, frame1 digest)` of every version linking a pinned frame1
    document and no pinned canonical frame."""
    with store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT va.subject_id AS subject_id, va.content_version AS content_version, "
            "va.artifact_digest AS artifact_digest "
            "FROM datasets.version_artifact va "
            "JOIN datasets.artifact_location al "
            "ON al.artifact_digest = va.artifact_digest AND al.tier = 'pinned' "
            "WHERE va.media_type = %s AND NOT EXISTS ("
            "SELECT 1 FROM datasets.version_artifact frame "
            "JOIN datasets.artifact_location pinned "
            "ON pinned.artifact_digest = frame.artifact_digest AND pinned.tier = 'pinned' "
            "WHERE frame.subject_id = va.subject_id "
            "AND frame.content_version = va.content_version "
            "AND frame.rendering_key = %s) "
            "ORDER BY va.subject_id, va.content_version",
            (FRAME1_MEDIA_TYPE, rendering_key(CANONICAL_FRAME_RENDERING)),
        )
    return [
        (str(row["subject_id"]), str(row["content_version"]), str(row["artifact_digest"]))
        for row in rows
    ]


def _stored_canonical_frame(
    frame: Frame, *, subject_id: str, content_version: str, artifact_store: ArtifactStore
) -> ArtifactRecord:
    """Store `frame`, read from a frame1 document, as the version's canonical frame, and check
    that the stored frame reads back as `frame`, so a conversion never changes content. The
    frame is compared, not the document's digest: a document that predates a column the frame
    holds has another digest than the frame's own frame1 bytes.

    Raises:
        ValueError: The stored canonical frame reads back as another frame.
    """
    record = render_and_store_canonical_frame(
        frame_chunks=(frame,),
        subject_id=subject_id,
        content_version=content_version,
        artifact_store=artifact_store,
    )
    stored = read_stored_frame(
        artifact_store.get(artifact_digest=record.artifact_digest, tier=Tier.working),
        media_type=record.media_type,
        digest=record.artifact_digest,
    )
    if stored != frame:
        raise ValueError(f"canonical frame {record.artifact_digest} reads back as another frame")
    return record


def run_backfill_frame1(*, store: DatasetStore, artifact_store: ArtifactStore) -> int:
    """Link a pinned canonical frame, holding the frame1's references, under every version
    holding only a pinned frame1 document, emit the counts, and return the number of versions
    it could not convert. A version that fails, its document unreadable included, emits
    `datasets_frame1_backfill_failed` naming it, and the run goes on to the next."""
    writer = LiveVersionRecordWriter(store=store)
    converted = failed = 0
    for subject_id, content_version, digest in _pinned_frame1_versions(store):
        try:
            frame = read_stored_frame(
                artifact_store.get(artifact_digest=digest, tier=Tier.pinned),
                media_type=FRAME1_MEDIA_TYPE,
                digest=digest,
            )
            record = _stored_canonical_frame(
                frame,
                subject_id=subject_id,
                content_version=content_version,
                artifact_store=artifact_store,
            )
            artifact_store.copy(record.artifact_digest, from_tier=Tier.working, to_tier=Tier.pinned)
            writer.record_replacement_artifact(
                subject_id=subject_id,
                content_version=content_version,
                artifact=ArtifactRecord(
                    artifact_digest=record.artifact_digest,
                    rendering=record.rendering,
                    media_type=record.media_type,
                ),
                replaces=digest,
            )
        except (
            ArtifactStoreError,
            DatasetsStoreUnavailable,
            StoredFrameUnreadable,
            ValueError,
        ) as exc:
            failed += 1
            emit(
                "datasets_frame1_backfill_failed",
                severity="ERROR",
                subject_id=subject_id,
                content_version=content_version,
                artifact_digest=digest,
                error=f"{type(exc).__name__}: {exc}",
            )
            continue
        converted += 1
    emit("datasets_frame1_backfilled", converted=converted, failed=failed)
    return failed


def _backfill_frame1() -> int:
    return _run_over_artifacts("backfill-frame1", run_backfill_frame1)


# The datasets maintenance Job's timeout is 3600 seconds. A backfill starts no version after
# this long, leaving the rest of the hour to the version in hand.
BACKFILL_DESCRIPTIONS_BUDGET = timedelta(minutes=45)


def _clears_on_retry(exc: BaseException) -> bool:
    """Whether a later run can get past `exc`. Only these can: the store unreachable (a broken
    connection, `default_is_broken`), its pool exhausted, a SQLSTATE `store_failure_is_permanent`
    holds transient, the artifact inventory's lookup failing in the store
    (`ArtifactInventoryError`, which wraps only a store failure), and the artifact store's
    transport failing, timing out or answering 429 or 5xx. Anything else, such as an object
    whose bytes no longer hash to its digest, is permanent."""
    if default_is_broken(exc) or isinstance(exc, PoolTimeoutError | ArtifactInventoryError):
        return True
    if isinstance(exc, DatasetsStoreUnavailable):
        return not store_failure_is_permanent(exc.sqlstate)
    if isinstance(exc, pg8000.exceptions.DatabaseError):
        sqlstate = extract_sqlstate(exc)
        return sqlstate is not None and not store_failure_is_permanent(sqlstate)
    return isinstance(exc, ServerError | TooManyRequests | RetryError | TransportError)


# Description rows checked for validity per query of `_invalid_record_versions`.
_RECORD_SCAN_PAGE = 500


def _unrecorded_versions(store: DatasetStore) -> list[tuple[str, str, datetime]]:
    """`(subject_id, content_version, recorded_at)` of every dataset or frame version row that
    holds no description record, or one that fails validation, those that collapse no other
    version first, each group recorded last first."""
    with store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT v.subject_id AS subject_id, v.content_version AS content_version, "
            "v.recorded_at AS recorded_at, v.base_content_version AS base_content_version "
            "FROM datasets.version_record v "
            "WHERE v.subject_kind = 'spec' AND NOT EXISTS ("
            "SELECT 1 FROM datasets.description_record d "
            "WHERE d.subject_id = v.subject_id AND d.content_version = v.content_version)"
        )
    versions = [*rows, *_invalid_record_versions(store)]
    versions.sort(key=lambda row: (str(row["subject_id"]), str(row["content_version"])))
    versions.sort(key=lambda row: cast(datetime, row["recorded_at"]), reverse=True)
    versions.sort(key=lambda row: row["base_content_version"] is not None)
    return [
        (str(row["subject_id"]), str(row["content_version"]), cast(datetime, row["recorded_at"]))
        for row in versions
    ]


def _invalid_record_versions(store: DatasetStore) -> list[Mapping[str, object]]:
    """The version rows whose description record, of this image's schema version or an
    earlier one, fails validation, each logged by `validated_record`, read
    `_RECORD_SCAN_PAGE` records at a time. A record of a later schema version is a later
    image's, which this image leaves alone."""
    invalid: list[Mapping[str, object]] = []
    after = ("", "")
    while True:
        with store.transaction() as tx:
            page = tx.fetch_all(
                "SELECT d.subject_id AS subject_id, d.content_version AS content_version, "
                "d.schema_version AS schema_version, d.record AS record, "
                "v.recorded_at AS recorded_at, "
                "v.base_content_version AS base_content_version "
                "FROM datasets.description_record d JOIN datasets.version_record v "
                "ON v.subject_id = d.subject_id AND v.content_version = d.content_version "
                "WHERE (d.subject_id, d.content_version) > (%s, %s) "
                "AND d.schema_version <= %s "
                "ORDER BY d.subject_id, d.content_version LIMIT %s",
                (*after, SCHEMA_VERSION, _RECORD_SCAN_PAGE),
            )
        invalid.extend(
            row
            for row in page
            if validated_record(
                row["record"],
                schema_version=cast(int, row["schema_version"]),
                id=str(row["subject_id"]),
                content_version=str(row["content_version"]),
            )
            is None
        )
        if len(page) < _RECORD_SCAN_PAGE:
            return invalid
        after = (str(page[-1]["subject_id"]), str(page[-1]["content_version"]))


def _held_bytes(
    artifact_store: ArtifactStore, location: StoredArtifactLocation | None
) -> bytes | None:
    """The bytes at `location`, or `None` when there is none or its object is gone. Any other
    artifact store failure raises, so a passing outage never reads as data no longer held."""
    if location is None:
        return None
    try:
        return artifact_store.get(
            artifact_digest=location.artifact_digest, tier=Tier(location.tier)
        )
    except ArtifactNotFoundError:
        return None


def _statistics_figures(data: bytes, *, continuing: Sequence[ContinuedSeries]) -> SeriesFigures:
    sink = SeriesStatisticsSink(continuing=continuing)
    sink.write(pq.read_table(pa.BufferReader(data), schema=SERIES_STATISTICS_SCHEMA))
    return sink.finish().figures


def _inline_statistics_figures(
    store: DatasetStore,
    *,
    subject_id: str,
    content_version: str,
    continuing: Sequence[ContinuedSeries],
) -> SeriesFigures | None:
    """The series figures of the per-series `statistics` list a body stored before the series
    statistics artifact existed, or `None` when the body holds no such list."""
    with store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT immutable_body FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            (subject_id, content_version),
        )
    body = None if row is None else row["immutable_body"]
    entries = body.get("statistics") if isinstance(body, Mapping) else None
    if not isinstance(entries, list):
        return None
    sink = SeriesStatisticsSink(continuing=continuing)
    sink.write(
        pa.Table.from_pylist(
            [_statistics_row(cast("Mapping[str, object]", entry)) for entry in entries],
            schema=SERIES_STATISTICS_SCHEMA,
        )
    )
    return sink.finish().figures


def _statistics_row(entry: Mapping[str, object]) -> dict[str, object]:
    """One inline statistics entry as a `series-statistics/1` row."""
    points = {
        name: cast("Mapping[str, object]", entry[name])
        for name in ("first", "latest", "minimum", "maximum")
    }
    change = cast("Mapping[str, object] | None", entry.get("change")) or {}
    return {
        "stat_var": entry["stat_var"],
        "entities": entry["entities"],
        "facet": json.dumps(entry["facet"], sort_keys=True, separators=(",", ":")),
        "observations": entry["observations"],
        **{f"{name}_value": point["value"] for name, point in points.items()},
        **{f"{name}_period": point["period"] for name, point in points.items()},
        "change_absolute": change.get("absolute"),
        "change_percent": change.get("percent"),
    }


def _stored_frame(
    artifact_store: ArtifactStore, servable: Sequence[StoredArtifactLocation]
) -> Frame | None:
    """The version's stored frame, its canonical frame or its frame1 document, while a tier
    holds it."""
    location = stored_frame_location(servable)
    data = _held_bytes(artifact_store, location)
    if location is None or data is None:
        return None
    return read_stored_frame(data, media_type=location.media_type, digest=location.artifact_digest)


def run_backfill_continuations(*, store: DatasetStore, deadline: datetime, now: Clock) -> None:
    """Fill `continuations` on every published frame's version row holding none, oldest
    first, starting none once `deadline` has passed, and emit the counts.

    Each version's continuations derive as its publish derives them
    (`lineage_continuations.derived_continuations`): from the versions its record's `retrieve`
    steps name, else from those its stored lineage lists. An older version is filled before the
    versions computed from it, which then take its entries as stored. The update holds the row
    to `continuations IS NULL`, so a second run, or a row a publish filled meanwhile, changes
    nothing. A version that fails is logged with its traceback and tried again on the next run.
    """
    reader = LiveSnapshotReader(store=store)
    with store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT subject_id, content_version FROM datasets.version_record "
            "WHERE input_content_versions IS NOT NULL AND continuations IS NULL "
            "ORDER BY recorded_at, subject_id, content_version"
        )
    filled = failed = remaining = 0
    for position, row in enumerate(rows):
        if now() >= deadline:
            remaining = len(rows) - position
            break
        subject_id, content_version = str(row["subject_id"]), str(row["content_version"])
        try:
            version = reader.read_version(subject_id, content_version)
            if version is None or not isinstance(version.facts, FrameLineage):
                continue
            with store.transaction() as tx:
                record = read_description(tx, id=subject_id, content_version=content_version)
            steps = None if record is None else record.steps
            entries = derived_continuations(
                lineage_inputs(
                    steps_inputs=None if steps is None else computed_inputs(steps),
                    lineage=version.facts,
                ),
                reader=reader,
            )
            with store.transaction() as tx:
                tx.execute(
                    "UPDATE datasets.version_record SET continuations = %s "
                    "WHERE subject_id = %s AND content_version = %s AND continuations IS NULL",
                    (json.dumps(entries), subject_id, content_version),
                )
        except Exception as exc:
            # One version that fails is skipped; the run carries on with the others.
            failed += 1
            emit(
                "datasets_continuations_backfill_failed",
                severity="ERROR",
                subject_id=subject_id,
                content_version=content_version,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            continue
        filled += 1
    emit("datasets_continuations_backfilled", filled=filled, failed=failed, remaining=remaining)


def run_backfill_descriptions(
    *,
    store: DatasetStore,
    artifact_store: ArtifactStore,
    code_index: CodeIndex,
    now: Clock = utc_now,
    budget: timedelta = BACKFILL_DESCRIPTIONS_BUDGET,
) -> int:
    """Write a backfilled description record for every version holding none, starting none
    once `budget` has passed, emit the counts, and return the number of versions that failed
    transiently when it read none, else 0.

    A version fails transiently when a later run can get past its failure (`_clears_on_retry`):
    the store or the artifact store unreachable, timing out or overloaded. Every other failure
    is permanent: building its record raises, or a stored row or object it reads is damaged.
    Each failure is logged with its traceback, and the
    version is skipped and tried again on the next run. Only a run that read no version and
    met a transient failure exits non-zero, so an outage fails it and versions that can never be
    described do not. A collapse whose base holds no record, as when the base failed earlier in
    the run, is logged with both content versions and counted as a permanent failure, so a
    later run derives it from the base's record."""
    reader = LiveSnapshotReader(store=store)
    deadline = now() + budget
    run_backfill_continuations(store=store, deadline=deadline, now=now)
    versions = _unrecorded_versions(store)
    backfilled = failed = transient = read = remaining = 0
    for position, (subject_id, content_version, recorded_at) in enumerate(versions):
        if now() >= deadline:
            remaining = len(versions) - position
            break
        try:
            state = reader.read_version_state(subject_id, content_version)
            if state is None:
                read += 1
                continue
            base = _base_record(store, state.version)
            if state.version.base_content_version is not None and base is None:
                failed += 1
                emit(
                    "datasets_description_backfill_base_unrecorded",
                    severity="ERROR",
                    subject_id=subject_id,
                    content_version=content_version,
                    base_content_version=state.version.base_content_version,
                )
                continue
            backfilled_at = now()
            servable = select_servable_locations(state.locations, now=backfilled_at)
            statistics = frame = None
            if isinstance(state.version.facts, FrameLineage):
                frame = _stored_frame(artifact_store, servable)
            else:
                data = _held_bytes(
                    artifact_store,
                    next((loc for loc in servable if is_statistics_rendering(loc.rendering)), None),
                )
                continuing = continued_series_of_trace(state.version.facts.trace)
                statistics = (
                    _statistics_figures(data, continuing=continuing)
                    if data is not None
                    else _inline_statistics_figures(
                        store,
                        subject_id=subject_id,
                        content_version=content_version,
                        continuing=continuing,
                    )
                )
            record, unstated_warnings = backfill_record(
                state.version,
                statistics=statistics,
                frame=frame,
                code_index=code_index,
                recorded_at=recorded_at,
                backfilled_at=backfilled_at,
                base=base,
                summed=_version_sums(state),
            )
            with store.transaction() as tx:
                written = insert_description(tx, record, recorded_at=backfilled_at)
        except Exception as exc:
            # One version that fails is skipped; the run carries on so every other version is
            # backfilled.
            failed += 1
            outage = _clears_on_retry(exc)
            if outage:
                transient += 1
            emit(
                "datasets_description_backfill_failed",
                severity="ERROR",
                subject_id=subject_id,
                content_version=content_version,
                transient=outage,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            continue
        read += 1
        if not written:
            continue
        backfilled += 1
        for code in unstated_warnings:
            emit(
                "datasets_description_backfill_warning_unstated",
                severity="WARNING",
                subject_id=subject_id,
                content_version=content_version,
                warning_code=code,
            )
    emit(
        "datasets_descriptions_backfilled",
        backfilled=backfilled,
        failed=failed,
        transient=transient,
        remaining=remaining,
        code_index=code_index.stamp.content_hash,
    )
    return 0 if read else transient


def _version_sums(state: VersionState) -> bool:
    """Whether the version `state` holds sums its definition's variables, as a backfilled
    record states it: a published frame always, a derived version as the frames that serve it
    read its own frame (`recorded_rendering.own_frame_mode`), and one whose record cannot say
    as the files linked under it read (`recorded_rendering.derived_version_sums`)."""
    version = state.version
    if isinstance(version.facts, FrameLineage) or version.base_content_version is None:
        return True
    renderings = [location.rendering for location in state.locations]
    mode = own_frame_mode(version, renderings=renderings)
    if mode is not None:
        return mode is FrameMode.SUMMED
    return derived_version_sums(version.facts.trace, renderings)


def _base_record(store: DatasetStore, version: StoredVersion) -> DescriptionRecord | None:
    """The record of the version `version` collapses, `None` when it collapses none or its
    base holds no record."""
    if version.base_content_version is None:
        return None
    with store.transaction() as tx:
        return read_description(
            tx, id=version.subject_id, content_version=version.base_content_version
        )


def _backfill_descriptions() -> int:
    # The run reads one code index, the served graph release's.
    with bound_served_release(get_release_service()):
        return _run_over_artifacts(
            "backfill-descriptions", partial(run_backfill_descriptions, code_index=get_index())
        )


class WarningBackfillSettings(BaseSettings):
    """The graph token and SDMX timeout a mint reads with, under the API's own names.
    `DatasetsBuildSettings` holds both but also reads `DFO_PUBLIC_SITE_URL`, which the job does
    not set."""

    model_config = SettingsConfigDict(env_prefix="DATASETS_")

    graph_token: str = ""
    sdmx_timeout_seconds: int = Field(default=90, ge=1)


def _backfill_warnings(dry_run: bool) -> int:
    """Run `warning_backfill.run_backfill_warnings` over the rows it rewrites, and with
    `dry_run` the rows in the new format. With none it opens no release and builds no client.
    It refuses to start without a graph token, since every read would then answer 401 or 403,
    which a later run can get past, and leave every row."""
    deadline = utc_now() + BACKFILL_DESCRIPTIONS_BUDGET
    try:
        backfill = WarningBackfillSettings()
    except ValidationError as exc:
        emit(
            "datasets_maintenance_failed",
            severity="ERROR",
            task="backfill-warnings",
            error=str(exc),
        )
        return 1
    if not backfill.graph_token:
        emit(
            "datasets_maintenance_failed",
            severity="ERROR",
            task="backfill-warnings",
            error="DATASETS_GRAPH_TOKEN is empty, so no graph or SDMX read can authenticate",
        )
        return 1
    store = LiveDatasetStore(connections=shared_pool())
    try:
        rows = select_rows(store, new_format=dry_run)
        if not rows:
            emit_nothing_backfilled(dry_run=dry_run)
            return 0
        try:
            settings = DatasetsSettings.from_env()
        except DatasetsConfigError as exc:
            emit(
                "datasets_maintenance_failed",
                severity="ERROR",
                task="backfill-warnings",
                error=str(exc),
            )
            return 1
        with ExitStack() as stack:
            stack.enter_context(bound_served_release(get_release_service()))
            stack.enter_context(bound_catalogue(get_catalogue_service().snapshot()))
            graph = build_graph_client(
                base_url=settings.graph_base_url,
                token=backfill.graph_token,
                un_base_url=settings.un_graph_base_url,
                un_availability=SdmxClient(
                    base_url=settings.un_graph_base_url, timeout=backfill.sdmx_timeout_seconds
                ),
                **settings.graph_cache_options(),
            )
            stack.callback(graph.close)
            sdmx = SdmxClient(
                base_url=settings.sdmx_base_url,
                token=backfill.graph_token,
                timeout=backfill.sdmx_timeout_seconds,
            )
            stack.callback(sdmx.close)
            unfinished = run_backfill_warnings(
                rows,
                store=store,
                readers=Readers(
                    graph=graph,
                    availability=sdmx,
                    code_index=get_index(),
                    coverage=get_coverage_service().snapshot().index,
                ),
                dry_run=dry_run,
                deadline=deadline,
                now=utc_now,
                clears_on_retry=_clears_on_retry,
            )
    finally:
        close_shared_pool()
    return 1 if unfinished else 0


class FrameSummedOutcome(StrEnum):
    """What `run_backfill_frame_summed` found for one derived version holding no
    `frame_summed`. `SUMMED` and `CUT` are set on the row; every other outcome leaves it NULL."""

    SUMMED = "summed"
    """The version's frame is its recorded groupby's collapse of the cut (`_is_own_frame`)."""
    CUT = "cut"
    """The version's frame is its base frame less its recorded cut (`_is_own_frame`)."""
    UNMATCHED = "unmatched"
    """Neither frame is the version's own, as when the code index changed since the mint."""
    BASE_FRAME_GONE = "base_frame_gone"
    """No tier holds the bytes of the base version's stored frame."""
    OWN_FRAME_KNOWN = "own_frame_known"
    """The record already says what the version's own frame holds
    (`recorded_rendering.own_frame_mode`), so the row is left alone."""


FRAME_SUMMED_PAGE = 100
"""Derived version rows `run_backfill_frame_summed` reads per query."""


def _derived_versions_without_frame_summed(
    store: DatasetStore, *, after: tuple[str, str]
) -> list[tuple[str, str]]:
    """`(subject_id, content_version)` of up to `FRAME_SUMMED_PAGE` derived version rows
    holding no `frame_summed`, past `after` in key order."""
    with store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT subject_id, content_version FROM datasets.version_record "
            "WHERE base_content_version IS NOT NULL AND frame_summed IS NULL "
            "AND (subject_id, content_version) > (%s, %s) "
            "ORDER BY subject_id, content_version LIMIT %s",
            (*after, FRAME_SUMMED_PAGE),
        )
    return [(str(row["subject_id"]), str(row["content_version"])) for row in rows]


def _is_own_frame(version: StoredVersion, *, frame_sha256: str) -> bool:
    """Whether the frame hashing to `frame_sha256` is derived `version`'s own: the hash it
    records, or the hash its `content_version` was minted from, alone."""
    if version.frame_sha256 is not None and frame_sha256 == version.frame_sha256:
        return True
    return mint_version_from_digest(frame_sha256) == version.content_version


def _matched_frame_summed(
    version: StoredVersion,
    *,
    renderings: Sequence[Mapping[str, object]],
    base_frame: Frame,
    code_index: CodeIndex,
) -> FrameSummedOutcome:
    """Match `version`'s frame (`_is_own_frame`) against `base_frame` less the party cut its
    trace records (`CUT`), else against a groupby its files record collapsing the cut frame
    (`SUMMED`), else `UNMATCHED`. The cut replays the trace's entries, so a party classified
    otherwise since the mint cuts as it did then. The collapse reads `code_index`."""
    # Only a dataset version names a base.
    assert isinstance(version.facts, ManifestFacts)
    trace = version.facts.trace
    entity_columns = entity_roles(frame_entity_columns(base_frame.columns))
    cut = cut_frame(
        lambda: iter((base_frame,)),
        PartyTotalsCut.of_trace_entries(
            cast("list[Mapping[str, object]]", trace[PARTY_TOTALS_TRACE_KEY])
        ),
        role_columns=entity_columns,
    )
    if _is_own_frame(version, frame_sha256=cut.frame_sha256):
        return FrameSummedOutcome.CUT
    groupbys: list[GroupbyOptions] = []
    for minting in recorded_mintings(renderings):
        if minting.groupby is not None and minting.groupby not in groupbys:
            groupbys.append(minting.groupby)
    for groupby in groupbys:
        try:
            collapsed = groupby_collapse_sha256(
                cut.frame_chunks,
                options=MintOptions(groupby=groupby, without_party_totals=True),
                code_index=code_index,
                entity_columns=entity_columns,
                resolved_stat_var_refs=refs_from_resolved_stat_vars(
                    cast("list[Mapping[str, object]]", trace.get("resolved_stat_vars", []))
                ),
                provenance_refs=provenance_refs_of(
                    cast("list[Mapping[str, object]]", trace.get("facets", []))
                ),
                subject_id=version.subject_id,
            )
        except RollupNotSummableError:
            # Under `code_index` the groupby does not sum this frame.
            continue
        if collapsed is not None and _is_own_frame(version, frame_sha256=collapsed):
            return FrameSummedOutcome.SUMMED
    return FrameSummedOutcome.UNMATCHED


def _backfill_frame_summed_of(
    subject_id: str,
    content_version: str,
    *,
    store: DatasetStore,
    reader: LiveSnapshotReader,
    artifact_store: ArtifactStore,
    code_index: CodeIndex,
) -> FrameSummedOutcome | None:
    """Set `frame_summed` on one derived version row when its frame hash matches
    (`_matched_frame_summed`), and return the outcome, `None` for a row no longer recorded."""
    state = reader.read_version_state(subject_id, content_version)
    if state is None:
        return None
    version = state.version
    renderings = [location.rendering for location in state.locations]
    if own_frame_mode(version, renderings=renderings) is not None:
        return FrameSummedOutcome.OWN_FRAME_KNOWN
    assert version.base_content_version is not None
    base_frame = _stored_frame(
        artifact_store,
        reader.read_artifact_locations(subject_id, version.base_content_version),
    )
    if base_frame is None:
        return FrameSummedOutcome.BASE_FRAME_GONE
    outcome = _matched_frame_summed(
        version, renderings=renderings, base_frame=base_frame, code_index=code_index
    )
    if outcome in (FrameSummedOutcome.SUMMED, FrameSummedOutcome.CUT):
        with store.transaction() as tx:
            tx.execute(
                "UPDATE datasets.version_record SET frame_summed = %s "
                "WHERE subject_id = %s AND content_version = %s AND frame_summed IS NULL",
                (outcome is FrameSummedOutcome.SUMMED, subject_id, content_version),
            )
    return outcome


def run_backfill_frame_summed(
    *, store: DatasetStore, artifact_store: ArtifactStore, code_index: CodeIndex
) -> int:
    """Set `frame_summed` on every derived version row holding none whose frame hash a re-cut
    of its base frame, or a collapse of that cut, reproduces, emit the count of each
    `FrameSummedOutcome`, and return the number of versions that failed.

    Rows are read `FRAME_SUMMED_PAGE` at a time in key order. The update holds the row to
    `frame_summed IS NULL`, so a second run, or a row the previous image left, changes no value
    already set. A version that fails is logged with its traceback and left NULL."""
    reader = LiveSnapshotReader(store=store)
    counts = dict.fromkeys(FrameSummedOutcome, 0)
    failed = 0
    after = ("", "")
    while True:
        page = _derived_versions_without_frame_summed(store, after=after)
        for subject_id, content_version in page:
            try:
                outcome = _backfill_frame_summed_of(
                    subject_id,
                    content_version,
                    store=store,
                    reader=reader,
                    artifact_store=artifact_store,
                    code_index=code_index,
                )
            except Exception as exc:
                # A version that fails is skipped and the run carries on with the others.
                failed += 1
                emit(
                    "datasets_frame_summed_backfill_failed",
                    severity="ERROR",
                    subject_id=subject_id,
                    content_version=content_version,
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
                continue
            if outcome is not None:
                counts[outcome] += 1
        if len(page) < FRAME_SUMMED_PAGE:
            break
        after = page[-1]
    emit(
        "datasets_frame_summed_backfilled",
        **{outcome.value: count for outcome, count in counts.items()},
        failed=failed,
        code_index=code_index.stamp.content_hash,
    )
    return failed


def _backfill_frame_summed() -> int:
    # The run reads one code index, the served graph release's.
    with bound_served_release(get_release_service()):
        return _run_over_artifacts(
            "backfill-frame-summed", partial(run_backfill_frame_summed, code_index=get_index())
        )


def stored_recipe_pairs(store: DatasetStore) -> list[tuple[str, str]]:
    """`(spec_id, content_version)` of every frame holding a recipe, oldest recipe first."""
    with store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT subject_id, content_version FROM datasets.frame_recipe "
            "ORDER BY recorded_at, subject_id, content_version"
        )
    return [(str(row["subject_id"]), str(row["content_version"])) for row in rows]


def _replay_source(
    store: DatasetStore, *, artifact_store: ArtifactStore, now: Clock
) -> StoredRootSource:
    return StoredRootSource(
        snapshot_reader=LiveSnapshotReader(store=store),
        artifact_store=artifact_store,
        recipes=LiveRecipeStore(store=store),
        now=now,
    )


def run_diff_recipes(
    *, store: DatasetStore, artifact_store: ArtifactStore, report: TextIO, now: Clock = utc_now
) -> int:
    """Replay every stored recipe, write each one that is not equal to `report` and emit the
    counts. Return how many differ."""
    counts = {family: dict.fromkeys(ReplayOutcome, 0) for family in ("compare", "sum_of", "other")}
    recipes = LiveRecipeStore(store=store)
    for spec_id, content_version in stored_recipe_pairs(store):
        record = recipes.read_recipe(spec_id=spec_id, content_version=content_version)
        if record is None:
            continue
        replay = replay_recipe(
            spec_id=spec_id,
            content_version=content_version,
            recipe=record.body,
            source=_replay_source(store, artifact_store=artifact_store, now=now),
        )
        counts[replay.family][replay.outcome] += 1
        if replay.outcome is not ReplayOutcome.EQUAL:
            report.write(json.dumps(replay.as_json(), sort_keys=True) + "\n")
    totals = {outcome: sum(by[outcome] for by in counts.values()) for outcome in ReplayOutcome}
    differs = totals[ReplayOutcome.DIFFERS]
    emit(
        "datasets_recipes_diffed",
        severity="ERROR" if differs else "INFO",
        **{outcome.value: count for outcome, count in totals.items()},
        families={
            family: {outcome.value: count for outcome, count in by.items()}
            for family, by in counts.items()
        },
    )
    return differs


def _diff_recipes(report: Path) -> int:
    with report.open("w", encoding="utf-8") as sink:
        return _run_over_artifacts("diff-recipes", partial(run_diff_recipes, report=sink))


def run_record_recipe_golden(
    *,
    store: DatasetStore,
    artifact_store: ArtifactStore,
    out: Path,
    limit: int,
    max_bytes: int,
    origin: str,
    now: Clock = utc_now,
) -> None:
    """Write up to `limit` readable recipes with their frames under `out`, one case each,
    labelled `origin`."""
    written = skipped = 0
    for spec_id, content_version in stored_recipe_pairs(store):
        if written >= limit:
            break
        try:
            kept = record_case(
                spec_id=spec_id,
                content_version=content_version,
                source=_replay_source(store, artifact_store=artifact_store, now=now),
                out=out / f"{spec_id}-{content_version}",
                origin=origin,
                max_bytes=max_bytes,
            )
        except FrameUnavailable:
            kept = False
        written += kept
        skipped += not kept
    emit("datasets_recipe_golden_recorded", written=written, skipped=skipped)


def _record_recipe_golden(out: Path, limit: int, max_bytes: int, origin: str) -> int:
    return _run_over_artifacts(
        "record-recipe-golden",
        partial(run_record_recipe_golden, out=out, limit=limit, max_bytes=max_bytes, origin=origin),
    )


class RecipeSkip(StrEnum):
    """Why `normalise-recipes` leaves a recipe whose root options are out of order as it is."""

    # The stored hashes differ from those the analysis route computes from the stored recipe,
    # as for a tree whose frames different engine builds computed.
    UNREPRODUCIBLE = "unreproducible"
    # Another frame's recipe holds the replay key the sorted recipe computes.
    REPLAY_KEY_HELD = "replay_key_held"


def _recipe_hashes(recipe: Mapping[str, Any]) -> tuple[str, str]:
    """The `normal_form_hash` and `replay_key` of `recipe`, computed as the analysis route
    computes them for the frame it runs (`datasets_api.analysis._recipe_hashes`)."""
    roots = recipe["roots"]
    normal_hash = normal_form_hash(normal_form(recipe["tree"], roots))
    key = replay_key(
        normal_form_hash=normal_hash,
        root_datasets=[
            root_identity(root["dataset"], enrichment_stamp=root.get("enrichment_stamp"))
            for root in roots
        ],
        execution_identities=[recipe["execution_identity"]],
    )
    return normal_hash, key


def _with_sorted_options(recipe: Mapping[str, Any]) -> dict[str, Any] | None:
    """`recipe` with each root's `options.columns` sorted and listed once and its
    `options.groupby.by` sorted, as `analysis_roots.mint_options` writes them, or `None` when
    every root's already are."""
    roots: list[Mapping[str, Any]] = []
    for root in recipe["roots"]:
        options = dict(root.get("options", {}))
        if (columns := options.get("columns")) is not None:
            options["columns"] = sorted(set(columns))
        if (groupby := options.get("groupby")) is not None:
            options["groupby"] = {**groupby, "by": sorted(groupby["by"])}
        if options != root.get("options", {}):
            root = {**root, "options": options}
        roots.append(root)
    if roots == recipe["roots"]:
        return None
    return {**recipe, "roots": roots}


def run_normalise_recipes(
    *, datasets: DatasetStore, presentation: DatasetStore, dry_run: bool = False
) -> int:
    """Sort the root columns and `groupby.by` of every stored recipe that lists them out of
    order, recompute its two hashes, and move the card revisions drawing its frame to the new
    `normal_form_hash`.
    Return how many recipes it skipped (`RecipeSkip`).

    The revisions move first, in the Presentation database, then the recipe row, in the
    Datasets one, so a run stopped between the two leaves the recipe to the next run. With
    `dry_run` it counts and changes nothing."""
    # Presentation is another surface's store, so its modules load for this task alone.
    from presentation_store.store import rehash_derived_rendering

    with datasets.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT subject_id, content_version, normal_form_hash, replay_key, recipe "
            "FROM datasets.frame_recipe ORDER BY recorded_at, subject_id, content_version"
        )
    normalised = revisions = 0
    skipped: dict[RecipeSkip, int] = dict.fromkeys(RecipeSkip, 0)
    for row in rows:
        spec_id, content_version = str(row["subject_id"]), str(row["content_version"])
        stored = cast(Mapping[str, Any], row["recipe"])
        sorted_recipe = _with_sorted_options(stored)
        if sorted_recipe is None:
            continue
        old_hash, old_key = str(row["normal_form_hash"]), str(row["replay_key"])
        new_hash, new_key = _recipe_hashes(sorted_recipe)
        skip: RecipeSkip | None = None
        if _recipe_hashes(stored) != (old_hash, old_key):
            skip = RecipeSkip.UNREPRODUCIBLE
        else:
            with datasets.transaction() as tx:
                holder = tx.fetch_one(
                    "SELECT subject_id, content_version FROM datasets.frame_recipe "
                    "WHERE replay_key = %s AND superseded_at IS NULL",
                    (new_key,),
                )
            if holder is not None:
                skip = RecipeSkip.REPLAY_KEY_HELD
        if skip is not None:
            skipped[skip] += 1
            emit(
                "datasets_recipe_normalise_skipped",
                severity="WARNING",
                spec_id=spec_id,
                content_version=content_version,
                reason=skip.value,
                replay_key=new_key,
            )
            continue
        normalised += 1
        revisions += rehash_derived_rendering(
            presentation,
            spec_id=spec_id,
            content_version=content_version,
            old_hash=old_hash,
            new_hash=new_hash,
            dry_run=dry_run,
        )
        if dry_run:
            continue
        with datasets.transaction() as tx:
            tx.execute(
                "UPDATE datasets.frame_recipe "
                "SET normal_form_hash = %s, replay_key = %s, recipe = %s "
                "WHERE subject_id = %s AND content_version = %s AND replay_key = %s",
                (
                    new_hash,
                    new_key,
                    {**sorted_recipe, "normal_form_hash": new_hash, "replay_key": new_key},
                    spec_id,
                    content_version,
                    old_key,
                ),
            )
    emit(
        "datasets_recipes_normalised",
        severity="WARNING" if any(skipped.values()) else "INFO",
        dry_run=dry_run,
        normalised=normalised,
        revisions=revisions,
        **{reason.value: count for reason, count in skipped.items()},
    )
    return sum(skipped.values())


def _normalise_recipes(dry_run: bool) -> int:
    # Presentation is another surface's store, so its modules load for this task alone.
    from presentation_store import provider as presentation_provider

    datasets = LiveDatasetStore(connections=shared_pool())
    try:
        presentation = LiveDatasetStore(connections=presentation_provider.shared_pool())
        try:
            skipped = run_normalise_recipes(
                datasets=datasets, presentation=presentation, dry_run=dry_run
            )
        finally:
            presentation_provider.close_shared_pool()
    finally:
        close_shared_pool()
    return 1 if skipped else 0


def run_take_down_share(*, share_store: ShareStore, share_id: str) -> None:
    """Take the share `share_id` down as its owner's delete does, and log whether one was up."""
    # Presentation is another surface's store, so its modules load for this task alone.
    from presentation_store import ShareNotFound

    try:
        share_store.delete_share(share_id=share_id)
    except ShareNotFound:
        emit("presentation_share_take_down_not_found", share_id=share_id)
        return
    emit("presentation_share_taken_down", share_id=share_id)


def _take_down_share(share_id: str) -> int:
    from presentation_store import LiveShareStore
    from presentation_store import provider as presentation_provider
    from presentation_store.ids import SHARE_ID_PATTERN

    if not SHARE_ID_PATTERN.fullmatch(share_id):
        emit(
            "presentation_share_take_down_refused",
            severity="ERROR",
            share_id=share_id,
            reason=f"not a share id ({SHARE_ID_PATTERN.pattern})",
        )
        return 1
    presentation = LiveDatasetStore(connections=presentation_provider.shared_pool())
    try:
        run_take_down_share(share_store=LiveShareStore(store=presentation), share_id=share_id)
    finally:
        presentation_provider.close_shared_pool()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="datasets-maintenance")
    tasks = parser.add_subparsers(dest="task", required=True)
    tasks.add_parser(
        "sweep", help="report orphaned and missing artifact objects per tier"
    ).set_defaults(run=_sweep)
    tasks.add_parser(
        "sweep-export-claims",
        help="settle the pin claims of exports and pins past their lease, refunding where no copy stands",
    ).set_defaults(run=_sweep_copy_claims)
    tasks.add_parser(
        "age-operations",
        help="fail abandoned operations and delete done and failed ones past their retention",
    ).set_defaults(run=_age_operations)
    tasks.add_parser(
        "backfill-frame1", help="link a pinned canonical frame for every pinned frame1 document"
    ).set_defaults(run=_backfill_frame1)
    tasks.add_parser(
        "settle", help="label raw pin principals and count version rows holding no facts"
    ).set_defaults(run=_settle)
    tasks.add_parser(
        "backfill-descriptions",
        help="write a description record for every version made before records existed",
    ).set_defaults(run=_backfill_descriptions)
    warnings = tasks.add_parser(
        "backfill-warnings",
        help="rewrite stored member sentences into the counts and fields a mint writes",
    )
    warnings.add_argument("--dry-run", action="store_true")
    warnings.set_defaults(run=_backfill_warnings)
    tasks.add_parser(
        "backfill-frame-summed",
        help="record whether each derived version recorded without it stores its collapse",
    ).set_defaults(run=_backfill_frame_summed)
    diff = tasks.add_parser(
        "diff-recipes",
        help="compute every stored recipe again on its own roots and compare its output",
    )
    diff.add_argument("--report", type=Path, default=Path("recipe-diff.jsonl"))
    diff.set_defaults(run=_diff_recipes)
    golden = tasks.add_parser(
        "record-recipe-golden",
        help="write readable recipes with their root and output frames as golden cases",
    )
    golden.add_argument("--out", type=Path, required=True)
    golden.add_argument("--limit", type=int, default=50)
    golden.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    golden.add_argument("--origin", default="dfo-dev")
    golden.set_defaults(run=_record_recipe_golden)
    normalise = tasks.add_parser(
        "normalise-recipes",
        help="sort each stored recipe's root columns and move its cards to the new hash",
    )
    normalise.add_argument("--dry-run", action="store_true")
    normalise.set_defaults(run=_normalise_recipes)
    take_down = tasks.add_parser(
        "take-down-share", help="take a share down as its owner's delete does, with no token"
    )
    take_down.add_argument("share_id")
    take_down.set_defaults(run=_take_down_share)
    args = parser.parse_args(argv)
    options = {key: value for key, value in vars(args).items() if key not in ("task", "run")}
    return args.run(**options)


if __name__ == "__main__":
    sys.exit(main())
