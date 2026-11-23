"""The heads door serves `GET /api/datasets/heads?spec_id=...`, the stored-head check: for each
spec the query names, the dataset version a `latest: "stored"` read of it would serve from its
stored head now, or null. It is open to every caller.

The query names 1 to `HEADS_MAX` distinct `spec_id`s, each `spec_` and 32 lowercase hexadecimal
digits, and no other parameter, so an added parameter cannot split the edge cache. Anything else
refuses `request_invalid` (400) at the parameter at fault. The 200 is `DatasetHeads`,
one `DatasetHead` per asked spec in query order.

Each spec resolves to its served id as a stored read resolves it
(`kinds.spec_id.served_spec_id`, the id `resolve_served_definition` gives, computed in memory),
and the head of that id is judged by the stored read's own rule
(`current.stored_head_servable`), so a head past the recheck window and inside
`head_serve_stale_max_age_seconds` answers its version. `content_version` is null for an
unregistered or quarantined spec, one whose stored identity a stored read refuses to resolve, a spec with no head, a head keyed on an import the bound
release no longer carries, a head past that ceiling, and a head whose version has no record or
is revoked, or whose base version is, under the served id or the asked id.

The check reads and writes nothing else: it submits no head check, queues no rebuild, renews
no file and records no read. One store transaction reads every asked spec's definition,
bindings, quarantine, head and version state (`HeadReader.read_stored_heads`). The read runs on
the surface's read executor, and a full executor refuses retryable `service_unavailable` (503).
The 200 carries `Cache-Control: public, max-age=300`, so an edge cache answers a repeated check
for up to five minutes. A refusal carries `no-store`.

The door mounts before the definition door, so `heads` never reads as a dataset id
(`datasets_api.app`).
"""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from typing import Annotated, Final

from dfo_spec_kit.identity import IdentityError
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from code_index.composites import CompositeNotServable
from code_index.query import CodeIndex
from datasets_api.backpressure import Backpressure
from datasets_api.current import stored_head_servable
from datasets_api.deps import DepsCell
from datasets_api.kinds.spec_id import served_spec_id
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.timings import count, door_call, phase, run_with_timings
from datasets_api.versions import HeadReader, StoredHeadRead
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from dfo_platform.problem import Refusal
from spec_registry.reader import StoredDefinition

logger = logging.getLogger(__name__)

HEADS_MAX: Final = 64
HEADS_CACHE_CONTROL: Final = "public, max-age=300"

_SPEC_ID_PATTERN: Final = r"^spec_[0-9a-f]{32}$"
# `fullmatch` with no anchors: `$` also matches before a trailing newline.
_SPEC_ID = re.compile(r"spec_[0-9a-f]{32}")


class HeadsRequestInvalid(Refusal):
    """A query naming no `spec_id`, more than `HEADS_MAX`, one twice, one outside the id
    grammar, or any other parameter. Maps to `request_invalid` (400) at the parameter it
    names."""

    code = "request_invalid"

    def __init__(self, detail: str, *, pointer: str = "/spec_id") -> None:
        self.detail = detail
        self.pointer = pointer
        super().__init__(detail)


class HeadsBackpressure(Backpressure):
    """A full read executor maps to retryable `service_unavailable` (503)."""

    code = "service_unavailable"

    detail = "the heads door is at its concurrent-read capacity right now; retry"


class DatasetHead(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_id: str
    content_version: str | None = Field(
        description=(
            'the version a `latest: "stored"` read of the spec would serve from its stored '
            "head now. Null when that read would not serve from the head: no head, a head "
            "keyed on a replaced import, a head past the serve-stale ceiling, a revoked "
            "version, or a spec DFO has not registered"
        )
    )


class DatasetHeads(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heads: list[DatasetHead] = Field(description="one entry per asked spec, in query order")


def check_query(parameters: Iterable[str]) -> None:
    """Refuse a query naming any parameter other than `spec_id`, so an added parameter cannot
    make another cache key for one answer.

    Raises:
        HeadsRequestInvalid: A parameter other than `spec_id`, at that parameter.
    """
    unknown = next((name for name in parameters if name != "spec_id"), None)
    if unknown is not None:
        raise HeadsRequestInvalid(
            f"{unknown!r} is not a parameter of this route; name only spec_id",
            pointer=f"/{unknown}",
        )


def check_spec_ids(spec_ids: Sequence[str] | None) -> list[str]:
    """`spec_ids` as a list, when it names 1 to `HEADS_MAX` distinct ids in the grammar.

    Raises:
        HeadsRequestInvalid: It names none, more than `HEADS_MAX`, one twice, or one outside
            the grammar.
    """
    if not spec_ids:
        raise HeadsRequestInvalid("name at least one spec_id")
    if len(spec_ids) > HEADS_MAX:
        raise HeadsRequestInvalid(f"name at most {HEADS_MAX} spec_id values")
    malformed = next((spec_id for spec_id in spec_ids if not _SPEC_ID.fullmatch(spec_id)), None)
    if malformed is not None:
        raise HeadsRequestInvalid(
            f"{malformed!r} is not a spec id: `spec_` and 32 lowercase hexadecimal digits"
        )
    if len(set(spec_ids)) != len(spec_ids):
        raise HeadsRequestInvalid("name each spec_id once")
    return list(spec_ids)


def head_content_version(
    read: StoredHeadRead | None,
    *,
    now: datetime,
    recheck_interval: timedelta,
    serve_stale_max_age: timedelta,
) -> str | None:
    """The version a stored read serves from the head `read` holds at `now`, or `None` when
    the spec is unregistered or quarantined (`read` is `None`), its head does not serve by the
    stored read's rule (`stored_head_servable`), or the head's version is withdrawn."""
    if read is None or read.withdrawn:
        return None
    head = read.head
    if not stored_head_servable(
        head,
        now=now,
        recheck_interval=recheck_interval,
        serve_stale_max_age=serve_stale_max_age,
    ):
        return None
    return head.content_version


def _served_or_none(stored: StoredDefinition, *, code_index: CodeIndex) -> str | None:
    """`stored`'s served id (`served_spec_id`), or `None` when its stored identity has none: a
    composite named in a provenance set (`CompositeNotServable`) or an identity the kit
    refuses (`IdentityError`). A stored read of it refuses with the same error, so it serves
    from no head."""
    try:
        return served_spec_id(stored, code_index=code_index)
    except (CompositeNotServable, IdentityError):
        return None


def read_dataset_heads(
    *,
    spec_ids: Sequence[str],
    head_reader: HeadReader,
    code_index: CodeIndex,
    now: datetime,
    recheck_interval: timedelta,
    serve_stale_max_age: timedelta,
) -> DatasetHeads:
    """The stored-head answer for `spec_ids`, checked distinct and in the grammar, in their
    order, from one store read."""
    count("db_transactions", 0)
    with phase("heads_read"):
        reads = head_reader.read_stored_heads(
            spec_ids, served_id=functools.partial(_served_or_none, code_index=code_index)
        )
    return DatasetHeads(
        heads=[
            DatasetHead(
                spec_id=spec_id,
                content_version=head_content_version(
                    reads.get(spec_id),
                    now=now,
                    recheck_interval=recheck_interval,
                    serve_stale_max_age=serve_stale_max_age,
                ),
            )
            for spec_id in spec_ids
        ]
    )


def build_heads_door(*, deps: DepsCell) -> DoorContribution:
    """Build the heads door over the surface's dependency cell. Reads run on the read
    executor."""
    router = APIRouter()

    @router.get("/api/datasets/heads", response_model=DatasetHeads)
    async def read_heads(
        request: Request,
        spec_id: Annotated[
            list[str] | None,
            # Declared for the OpenAPI document only. `check_spec_ids` enforces it, so every
            # malformed query refuses `request_invalid` at `/spec_id`.
            Query(
                json_schema_extra={
                    "minItems": 1,
                    "maxItems": HEADS_MAX,
                    "uniqueItems": True,
                    "items": {"type": "string", "pattern": _SPEC_ID_PATTERN},
                }
            ),
        ] = None,
    ) -> JSONResponse:
        current = deps.get()
        settings = current.settings
        with door_call("heads", logger=logger) as call:
            check_query(request.query_params.keys())
            asked = check_spec_ids(spec_id)
            answer = await current.read_executor.run(
                functools.partial(
                    run_with_timings,
                    read_dataset_heads,
                    call.timings,
                    spec_ids=asked,
                    head_reader=current.head_store,
                    code_index=current.code_index,
                    now=current.now(),
                    recheck_interval=timedelta(seconds=settings.head_recheck_seconds),
                    serve_stale_max_age=timedelta(
                        seconds=settings.head_serve_stale_max_age_seconds
                    ),
                ),
                refuse=HeadsBackpressure,
            )
        return JSONResponse(
            answer.model_dump(mode="json"),
            headers={"cache-control": HEADS_CACHE_CONTROL, "Server-Timing": call.server_timing},
        )

    return DoorContribution(
        router=router,
        registry_endpoints=route_registry_endpoints(router, "datasets.heads"),
        classify=classify_datasets_refusal,
        exception_types=(HeadsRequestInvalid, HeadsBackpressure, DatasetsStoreUnavailable),
        endpoint_exemplars={
            "datasets.heads": (
                HeadsRequestInvalid("exemplar"),
                HeadsBackpressure(),
                DatasetsStoreUnavailable("exemplar"),
            )
        },
    )


__all__ = [
    "HEADS_CACHE_CONTROL",
    "HEADS_MAX",
    "DatasetHead",
    "DatasetHeads",
    "HeadsBackpressure",
    "HeadsRequestInvalid",
    "build_heads_door",
    "check_query",
    "check_spec_ids",
    "head_content_version",
    "read_dataset_heads",
]
