"""Render the minting options analysis roots read from a dataset's head before a reader asks.

A cold analysis root that reads a groupby sum or a party-totals cut pays the render of that
rendering from the dataset version's stored frame (`current.head_rendering`), seconds on a large
dataset. The render depends only on the dataset version, the minting options and the stamp of
the code index and render format (`versions.rendering_stamp`), and once a root has rendered it a
later root reads the version it recorded with no render
(`analysis_roots.LiveRootSource._recorded_rendering`). A new head, a new code index or a new
render format leaves every such record behind, so the first reader after each pays the render
again.

Each head row of a derived version's link records when an analysis root last read that file as
the rendering made from that head (`datasets.version_artifact_head.read_at`,
`mark_root_read`): when a root renders it, and when a root reads it recorded, at most once per
`READ_MARK_INTERVAL`. A stored answer is read after its roots resolve, so a repeated request
marks its roots too.

`render_ahead` renders, from one head of a dataset, each rendering of the dataset's links that
an analysis root read from any head since `read_since`, latest read first, up to `limit`, and
skips each one a link already records as rendered from that head under the serving stamp, so a
second run renders nothing. It renders as a root does (`current.head_rendering`), under a
materialise permit, so the render writes the link and its head row with the serving stamp and
the first reader of that head reads it with no render. The render leaves the new row's `read_at`
unset, so a rendering nobody reads ages out of the window and is no longer rendered ahead. A
rendering that mints nothing, as a sum that leaves the frame as it is, is linked under the head
itself with no head row, and a root reads it from that link (`render_ahead` skips it once
linked). Only bare frame-layout Parquet renderings are marked, since an analysis root reads no
other.

Two callers run it, neither on a request thread:

- The datasets worker, after an operation whose run materialised a raw version, renders ahead
  from that version under the worker's one materialise permit, after the operation completes, so
  the caller polling it is not held by renders of other roots
  (`operations.worker.run_operation`). A rebuild on a request thread queues a render-ahead
  operation for the worker instead (`operations.render_ahead_queue`), one per dataset and head.
  `DATASETS_RENDER_AHEAD_PER_HEAD` caps each, and the run's work deadline bounds it.
- `datasets-maintenance render-ahead`, run by hand, renders ahead from the head of every dataset
  with a read in the window, for a deploy or graph release that changes the code index or
  `versions.RENDER_FORMAT`, capped per head and in total (`DATASETS_RENDER_AHEAD_MAX`).

Each run first reads the dataset's head and renders nothing when the version it was asked to
render from is no longer the head.

A rendering whose render fails is logged and passed over, and the next run tries it again.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from dataset_store import DatasetStore
from datasets_api.current import DefinitionDeps, head_rendering
from datasets_api.manifest_models import Manifest
from datasets_api.render import (
    complete_rendering,
    is_incomplete_rendering,
    is_statistics_rendering,
    rendering_options,
)
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.version_store import current_rendering_keys
from datasets_api.versions import HeadVersion, ReadRendering, VersionLister, rendering_stamp
from dfo_platform.cloud_logging import JSON_FIELDS_ATTR
from dfo_platform.store_errors import safe_diagnostics

logger = logging.getLogger(__name__)

READ_MARK_INTERVAL = timedelta(hours=1)
"""The longest a head row's recorded read lags the last read. A root that reads a rendering whose
row records a read within it writes nothing."""

_RENDERED_AHEAD_EVENT = "datasets_rendered_ahead"
_RENDER_AHEAD_FAILED_EVENT = "datasets_render_ahead_failed"


def mark_root_read(
    lister: VersionLister,
    *,
    subject_id: str,
    head_content_version: str,
    manifest: Manifest,
    enrichment_digest: str | None,
    now: datetime,
) -> None:
    """Record `now` as the last read of the file `manifest` lists, a rendering an analysis root
    read as made from `head_content_version`, unless its head row records a read within
    `READ_MARK_INTERVAL`. A manifest of the head itself, which a rendering that minted nothing
    serves, records nothing, since its link holds no head row. A store failure is logged, and the
    read goes on."""
    artifact = next(
        (a for a in manifest.artifacts if not is_statistics_rendering(a.rendering)), None
    )
    if artifact is None or manifest.content_version == head_content_version:
        return
    try:
        lister.mark_rendering_read(
            subject_id,
            content_version=manifest.content_version,
            head_content_version=head_content_version,
            rendering_keys=current_rendering_keys(artifact.rendering, enrichment_digest),
            read_at=now,
            unless_read_since=now - READ_MARK_INTERVAL,
        )
    except DatasetsStoreUnavailable:
        logger.warning(
            "Datasets: recording a read of %r at %r from head %r failed; the read goes on",
            subject_id,
            manifest.content_version,
            head_content_version,
            exc_info=True,
        )


def read_recently(link_read_at: datetime | None, *, now: datetime) -> bool:
    """Whether a head row whose last recorded read is `link_read_at` records one within
    `READ_MARK_INTERVAL` of `now`, so a read now writes nothing."""
    return link_read_at is not None and link_read_at >= now - READ_MARK_INTERVAL


@dataclass(frozen=True)
class RenderedAhead:
    """What one `render_ahead` call did: the renderings read in the window that the head did not
    record (`due`), and of those the ones it `rendered`, the ones that `failed`, and the ones it
    left for the cap or the deadline (`left`). `head_moved` says the version it was asked to
    render from was not the dataset's head, so it rendered nothing."""

    head_moved: bool = False
    due: int = 0
    rendered: int = 0
    failed: int = 0
    left: int = 0
    failures: tuple[str, ...] = field(default=(), repr=False)


def _canonical(rendering: Mapping[str, object]) -> str:
    return json.dumps(dict(rendering), sort_keys=True, default=str)


def _recorded_from(
    versions: Sequence[HeadVersion],
    own: Iterable[Mapping[str, object]],
    *,
    head: str,
    stamp: str,
) -> set[str]:
    """The renderings, in `_canonical` form, a link records as made from `head` under `stamp`,
    and every complete rendering of `own`, the files linked under `head` itself, where a
    rendering that minted nothing from it is linked."""
    recorded = {
        _canonical(rendering) for rendering in own if not is_incomplete_rendering(rendering)
    }
    for version in versions:
        for link in version.links:
            if not is_incomplete_rendering(link.rendering) and (
                link.rendered_under.get(head) == stamp
            ):
                recorded.add(_canonical(link.rendering))
    return recorded


def _due(
    read: Sequence[ReadRendering], *, recorded: set[str], deps: DefinitionDeps
) -> list[tuple[ReadRendering, Mapping[str, object]]]:
    """Each of `read` this build renders as minting options, with the complete rendering it
    renders, that `recorded` does not hold, in the order of `read`."""
    due: list[tuple[ReadRendering, Mapping[str, object]]] = []
    seen: set[str] = set()
    for entry in read:
        rendering = complete_rendering(entry.rendering)
        key = _canonical(rendering)
        if key in recorded or key in seen:
            continue
        seen.add(key)
        options = rendering_options(rendering, site=deps.settings.site)
        if options is None or (options.groupby is None and not options.without_party_totals):
            continue
        due.append((entry, rendering))
    return due


def render_ahead(
    subject_id: str,
    *,
    head_content_version: str,
    deps: DefinitionDeps,
    read_since: datetime,
    limit: int,
    deadline: float | None = None,
) -> RenderedAhead:
    """Render from `head_content_version` each rendering of `subject_id`'s links an analysis
    root read since `read_since` and no link records as made from that head under the serving
    stamp, latest read first, at most `limit`, starting none once the monotonic `deadline` has
    passed. When `head_content_version` is not the dataset's head, as after a concurrent writer
    advanced it or a later rebuild moved it, it renders nothing. See the module docstring.

    Raises:
        DatasetsStoreUnavailable: A store read or write failed, which stops the run.
    """
    lister = deps.version_lister
    if lister is None or limit <= 0:
        return RenderedAhead()
    head = deps.head_store.read_head(subject_id)
    if head is None or head.content_version != head_content_version:
        logger.info(
            "Datasets render ahead of %r from %r skipped: the head is %r",
            subject_id,
            head_content_version,
            None if head is None else head.content_version,
        )
        return RenderedAhead(head_moved=True)
    read = lister.list_read_renderings(subject_id, read_since=read_since)
    if not read:
        return RenderedAhead()
    stamp = rendering_stamp(deps.code_index.stamp.content_hash)
    recorded = _recorded_from(
        lister.list_head_versions(subject_id, head_content_version=head_content_version),
        (
            location.rendering
            for location in deps.snapshot_reader.read_artifact_locations(
                subject_id, head_content_version
            )
        ),
        head=head_content_version,
        stamp=stamp,
    )
    due = _due(read, recorded=recorded, deps=deps)
    rendered = 0
    failures: list[str] = []
    started = 0
    for entry, rendering in due:
        if started >= limit or (deadline is not None and time.monotonic() >= deadline):
            break
        started += 1
        options = rendering_options(rendering, site=deps.settings.site)
        assert options is not None
        try:
            manifest = head_rendering(
                subject_id=subject_id,
                content_version=head_content_version,
                deps=deps,
                options=options,
                trace_id=None,
                coverage_snapshot=deps.coverage.snapshot(),
            )
        except DatasetsStoreUnavailable:
            raise
        except Exception as exc:
            failures.append(_failed(subject_id, head_content_version, entry, exc))
            continue
        if manifest is None:
            failures.append(
                _failed(subject_id, head_content_version, entry, "its stored frame is not servable")
            )
            continue
        rendered += 1
    result = RenderedAhead(
        due=len(due),
        rendered=rendered,
        failed=len(failures),
        left=len(due) - started,
        failures=tuple(failures),
    )
    logger.info(
        "Datasets rendered ahead subject_id=%s head=%s due=%d rendered=%d failed=%d left=%d",
        subject_id,
        head_content_version,
        result.due,
        result.rendered,
        result.failed,
        result.left,
        extra={
            JSON_FIELDS_ATTR: {
                "event": _RENDERED_AHEAD_EVENT,
                "subject_id": subject_id,
                "head_content_version": head_content_version,
                "due": result.due,
                "rendered": result.rendered,
                "failed": result.failed,
                "left": result.left,
            }
        },
    )
    return result


def _failed(subject_id: str, head: str, entry: ReadRendering, cause: object) -> str:
    """Log one failed render ahead and return its cause."""
    reason = f"{type(cause).__name__}: {cause}" if isinstance(cause, Exception) else str(cause)
    logger.warning(
        "Datasets render ahead of %r from head %r failed: %s",
        subject_id,
        head,
        reason,
        extra={
            JSON_FIELDS_ATTR: {
                "event": _RENDER_AHEAD_FAILED_EVENT,
                "subject_id": subject_id,
                "head_content_version": head,
                "rendering": dict(entry.rendering),
                "error": reason,
            }
        },
    )
    return reason


def heads_read_since(store: DatasetStore, *, read_since: datetime) -> list[tuple[str, str]]:
    """Each dataset's `(subject_id, head content_version)` whose links an analysis root read
    since `read_since`, by the dataset's latest read, latest first.

    Raises:
        DatasetsStoreUnavailable: The read failed.
    """
    failure: str | None = None
    rows: list[Mapping[str, object]] = []
    try:
        with store.transaction() as tx:
            rows = tx.fetch_all(
                "SELECT sh.subject_id, sh.content_version, MAX(vah.read_at) AS read_at "
                "FROM datasets.subject_head sh "
                "JOIN datasets.version_artifact_head vah ON vah.subject_id = sh.subject_id "
                "WHERE vah.read_at >= %s "
                "GROUP BY sh.subject_id, sh.content_version "
                "ORDER BY MAX(vah.read_at) DESC, sh.subject_id",
                (read_since,),
            )
    except Exception as exc:
        failure = f"listing the heads read since {read_since} failed: {safe_diagnostics(exc)}"
    if failure is not None:
        raise DatasetsStoreUnavailable(failure)
    return [(str(row["subject_id"]), str(row["content_version"])) for row in rows]


__all__ = [
    "READ_MARK_INTERVAL",
    "RenderedAhead",
    "heads_read_since",
    "mark_root_read",
    "read_recently",
    "render_ahead",
]
