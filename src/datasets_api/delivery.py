"""One `datasets.delivery` log line per successful data-serving door call.

A door that serves data opens its `door_call` with `delivers=True`, which binds a `Delivery`
for the call. The door, or code it runs on an executor, which carries the request's context,
fills it from facts already in hand: `record_manifest` from a manifest, `record_version` from a
stored version, `record_file` from a stored version and the file it serves, `record_export` from
an export's facts, `record_derived_manifest` from a derived frame's ids, `record_derived_file`
from a rendered derived file, `record_analysis` from an analysis answer and `record_rows_served`
from what left. Each `record_*` function logs a fault of its own at WARNING and returns, so a
recording fault never fails the response. Once the door has built its response, it logs the line
through `DoorCall.delivered` (`datasets_api.timings`), which logs only once a spec id is recorded.
A mint or definition read that answers 202 logs nothing, and each operations poll that serves
the done manifest logs a line under its own request's identity.

The line carries the request's `dfo_platform.caller.DeliveryIdentity` beside the facts, as
structured fields (`dfo_platform.cloud_logging.JSON_FIELDS_ATTR`) and a short message:

- `door`, `trace_id`, and `admission`, `caller`, `caller_source`, `upstream`, `surface`,
  `client`, `user_agent`, `turn_id`, `mcp_client` from the identity;
- `spec_id` (an `ops_` id on the derived manifest, derived file and analysis doors),
  `content_version`, and `export_id` on the export doors;
- `measures`, the sorted bare StatVar dcids of the manifest's `resolved_stat_vars`, only where
  the door holds a manifest (mint, definition, snapshot, operations poll), else `None`, cut
  to `MEASURES_CAP` with `measures_truncated` set; `measure_count` wherever the StatVars are
  known;
- `series_count`, `rows_available` (the version's `coverage.rows`, an export's rows, or a
  derived file's data rows) and `rows_served`, the rows that left: observation points, a data
  file's rows, an export link's rows, a derived file's rows, and 0 on the manifest doors (an
  operations poll included), an export prepare, an analysis answer and a series statistics
  file. A file whose rendering reshapes rows (`RESHAPING_KEYS`) or leaves total rows out (a
  bare long CSV) stores no row count of its own, so its `rows_served` is `None`. A call that
  records none, including a file whose row count raised, still logs its line with null;
- `revalidated`, true on a definition read answered 304 to a matching `If-None-Match`, which
  sends no body and so can be left out of a count of deliveries, and false otherwise.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dataset_store.snapshot import ManifestFacts
from dfo_platform.caller import current_delivery_identity
from dfo_platform.cloud_logging import JSON_FIELDS_ATTR
from dfo_platform.trace import get_trace_id
from materialise.generation import bare_stat_var_ref

if TYPE_CHECKING:
    from dataset_store.snapshot import StoredArtifactLocation, StoredVersion
    from datasets_api.derived import DerivedFile
    from datasets_api.manifest_models import Manifest

logger = logging.getLogger("datasets_api.delivery")

MEASURES_CAP = 200
"""The most StatVars one line lists. Cloud Logging drops an entry over 256 KB."""

RESHAPING_KEYS = frozenset({"groupby", "pivot", "without_party_totals"})
"""The rendering keys of a file whose rows are not the version's rows one for one. A
`columns` selection keeps every row, so it is absent."""


@dataclass
class Delivery:
    """What one door call served, filled as the door comes to each fact. `rows_served` stays
    `None` until a `record_*` function sets it, so a call that records no row count logs null."""

    spec_id: str | None = None
    content_version: str | None = None
    export_id: str | None = None
    measures: tuple[str, ...] | None = None
    measure_count: int | None = None
    series_count: int | None = None
    rows_available: int | None = None
    rows_served: int | None = None
    revalidated: bool = False


_current: ContextVar[Delivery | None] = ContextVar("datasets_delivery", default=None)


@contextmanager
def delivering() -> Iterator[Delivery]:
    """Bind a fresh `Delivery` for the block, which the `record_*` functions fill."""
    delivery = Delivery()
    token = _current.set(delivery)
    try:
        yield delivery
    finally:
        _current.reset(token)


def _never_raises[**P](record: Callable[P, None]) -> Callable[P, None]:
    """`record`, logging any exception it raises at WARNING and returning."""

    @functools.wraps(record)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            record(*args, **kwargs)
        except Exception as exc:
            # A delivery fact must not fail the response it describes.
            logger.warning("datasets.delivery %s failed: %s", guarded.__name__, type(exc).__name__)

    return guarded


def _stat_vars(trace: Mapping[str, Any]) -> tuple[str, ...] | None:
    entries = trace.get("resolved_stat_vars")
    if not isinstance(entries, list):
        return None
    refs = {
        bare_stat_var_ref(entry["ref"])
        for entry in entries
        if isinstance(entry, Mapping) and isinstance(entry.get("ref"), str)
    }
    return tuple(sorted(refs))


def _record(
    *,
    definition: Mapping[str, Any],
    content_version: str,
    coverage: Mapping[str, Any],
    series_count: int | None,
    trace: Mapping[str, Any],
    measures: bool,
) -> None:
    delivery = _current.get()
    if delivery is None:
        return
    raw_spec_id = definition.get("spec_id")
    rows = coverage.get("rows")
    stat_vars = _stat_vars(trace)
    # Every field is computed before the first assignment, so a fault leaves no partial line.
    delivery.spec_id = raw_spec_id if isinstance(raw_spec_id, str) else delivery.spec_id
    delivery.content_version = content_version
    delivery.rows_available = rows if isinstance(rows, int) else None
    delivery.series_count = series_count
    delivery.measure_count = len(stat_vars) if stat_vars is not None else None
    if measures:
        delivery.measures = stat_vars


@_never_raises
def record_manifest(manifest: Manifest) -> None:
    """Record the version `manifest` describes, with its StatVars, on the bound delivery. It
    serves no rows, so `rows_served` is 0."""
    _record(
        definition={"spec_id": manifest.definition.spec_id},
        content_version=manifest.content_version,
        coverage={"rows": manifest.coverage.rows},
        series_count=manifest.series_count,
        trace=manifest.trace,
        measures=True,
    )
    delivery = _current.get()
    if delivery is not None:
        delivery.rows_served = 0


@_never_raises
def record_version(stored: StoredVersion) -> None:
    """Record the stored dataset version `stored`, without its StatVar list, on the bound
    delivery. A published frame's lineage records only its ids."""
    facts = stored.facts
    if not isinstance(facts, ManifestFacts):
        _record(
            definition={"spec_id": stored.subject_id},
            content_version=stored.content_version,
            coverage={},
            series_count=None,
            trace={},
            measures=False,
        )
        return
    definition = facts.definition
    if not isinstance(definition.get("spec_id"), str):
        definition = {**definition, "spec_id": stored.subject_id}
    _record(
        definition=definition,
        content_version=stored.content_version,
        coverage=facts.coverage,
        series_count=facts.series_count,
        trace=facts.trace,
        measures=False,
    )


@_never_raises
def record_file(stored: StoredVersion, location: StoredArtifactLocation) -> None:
    """Record the stored version `stored` and the file at `location` it serves. Rows served are 0
    for a series statistics file, the version's rows for a file that keeps them one for one, and
    `None` otherwise, as the location stores no row count of its own. A fault in that count
    leaves `None`, so the line still logs, with `rows_served` null."""
    record_version(stored)
    delivery = _current.get()
    if delivery is None:
        return
    delivery.rows_served = _file_rows(location.rendering, rows=delivery.rows_available)


def _file_rows(rendering: Mapping[str, object], *, rows: int | None) -> int | None:
    if rendering.get("statistics") is not None:
        return 0
    if not RESHAPING_KEYS.isdisjoint(rendering):
        return None
    # A bare long CSV leaves the total rows out (`datasets_api.render`).
    bare_long_csv = (
        rendering.get("bundle") is False
        and rendering.get("format") == "csv"
        and rendering.get("layout") != "frame"
    )
    return None if bare_long_csv else rows


@_never_raises
def record_export(facts: Mapping[str, Any], *, served: bool) -> None:
    """Record the export whose stored facts are `facts`. A `served` export, signed for its
    link, served its rows; a prepared one served none."""
    delivery = _current.get()
    if delivery is None:
        return
    frame = facts.get("frame")
    if isinstance(frame, Mapping):
        spec_id, content_version = frame.get("spec_id"), frame.get("content_version")
        delivery.spec_id = spec_id if isinstance(spec_id, str) else None
        delivery.content_version = content_version if isinstance(content_version, str) else None
    export_id = facts.get("export_id")
    delivery.export_id = export_id if isinstance(export_id, str) else None
    rows = facts.get("rows")
    delivery.rows_available = rows if isinstance(rows, int) else None
    delivery.rows_served = delivery.rows_available if served else 0


@_never_raises
def record_derived_manifest(*, spec_id: str, content_version: str) -> None:
    """Record the derived frame `spec_id` (an `ops_` id) at `content_version`, whose manifest the
    call served. The manifest states no row count, so `rows_available` stays `None`, and it serves
    no rows."""
    delivery = _current.get()
    if delivery is None:
        return
    delivery.spec_id = spec_id
    delivery.content_version = content_version
    delivery.rows_served = 0


@_never_raises
def record_derived_file(*, spec_id: str, content_version: str, file: DerivedFile) -> None:
    """Record the derived frame `spec_id` (an `ops_` id) at `content_version`, served whole as
    `file`. `rows_available` is the file's own data rows, the rows it serves."""
    delivery = _current.get()
    if delivery is None:
        return
    delivery.spec_id = spec_id
    delivery.content_version = content_version
    delivery.rows_available = file.rows
    delivery.rows_served = file.rows


@_never_raises
def record_analysis(answer: Mapping[str, Any]) -> None:
    """Record the frame an analysis `answer` (`doors.analysis.AnalysisReady`) names by its ref,
    which serves none of its rows."""
    delivery = _current.get()
    if delivery is None:
        return
    member = answer["ref"]["members"][0]
    spec_id = member["spec"]["spec_id"]
    content_version = member["content_version"]
    rows = answer["rows"]
    measure_count = answer["stat_var_count"]
    # Every field is read before the first assignment, so a fault leaves no partial line.
    delivery.spec_id = spec_id
    delivery.content_version = content_version
    delivery.rows_available = rows
    delivery.measure_count = measure_count
    delivery.rows_served = 0


@_never_raises
def record_rows_served(rows: int) -> None:
    """Record the rows the call served on the bound delivery."""
    delivery = _current.get()
    if delivery is not None:
        delivery.rows_served = rows


def log_delivery(door: str, delivery: Delivery) -> None:
    """Log `delivery`'s line for `door`, if it recorded a spec id. It never raises."""
    if delivery.spec_id is None:
        return
    try:
        identity = current_delivery_identity()
        measures = delivery.measures
        truncated = measures is not None and len(measures) > MEASURES_CAP
        fields: dict[str, object] = {
            "door": door,
            "trace_id": get_trace_id(),
            "admission": identity.admission,
            "caller": identity.caller,
            "caller_source": identity.caller_source,
            "upstream": identity.upstream,
            "surface": identity.surface.value,
            "client": identity.client,
            "user_agent": identity.user_agent,
            "turn_id": identity.turn_id,
            "mcp_client": identity.mcp_client,
            "spec_id": delivery.spec_id,
            "content_version": delivery.content_version,
            "export_id": delivery.export_id,
            "measures": list(measures[:MEASURES_CAP]) if measures is not None else None,
            "measures_truncated": truncated,
            "measure_count": delivery.measure_count,
            "series_count": delivery.series_count,
            "rows_available": delivery.rows_available,
            "rows_served": delivery.rows_served,
            "revalidated": delivery.revalidated,
        }
        logger.info(
            "datasets.delivery door=%s spec_id=%s content_version=%s surface=%s "
            "caller_source=%s admission=%s measure_count=%s rows_served=%s revalidated=%s",
            door,
            delivery.spec_id,
            delivery.content_version,
            identity.surface.value,
            identity.caller_source,
            identity.admission,
            delivery.measure_count,
            delivery.rows_served,
            delivery.revalidated,
            extra={JSON_FIELDS_ATTR: fields},
        )
    except Exception as exc:
        # A log line must not fail the response it describes.
        logger.warning("datasets.delivery line failed: %s", type(exc).__name__)


__all__ = [
    "MEASURES_CAP",
    "RESHAPING_KEYS",
    "Delivery",
    "delivering",
    "log_delivery",
    "record_analysis",
    "record_derived_file",
    "record_derived_manifest",
    "record_export",
    "record_file",
    "record_manifest",
    "record_rows_served",
    "record_version",
]
