"""The versions door serves `GET /api/datasets/{id}/versions?limit=&cursor=`, one page of a
`spec_` dataset's stored versions, newest first. It is open to every caller.

The path id resolves to the served ids the registry binds to it, and versions recorded under
the posted id itself list too, as the snapshot door reads a version
(`doors.snapshot.read_under_served_id`). A version recorded under several of those ids lists
once. An unregistered id refuses `spec_not_found` (404), and a quarantined path id, or a
quarantined served id bound to it, refuses `identity_quarantined`. A version revoked, or whose
base version is revoked, under any of those ids is left out of the list, since a revocation
takes a version down everywhere and the snapshot and files doors refuse it.

Each version states its `rendering`: `plain` for a raw version, `groupby` with the `groupby` a
collapse was summed by, `without_party_totals` for a party-totals cut that sums nothing, and
`not_recorded` for a derived version with no linked file recording its options. A version
whose frame leaves party totals out says so in `without_party_totals`. The options are read as
the frame refresh reads them (`datasets_api.recorded_rendering`).

`limit` runs from 1 to `VERSIONS_PAGE_MAX` and defaults to `VERSIONS_PAGE_DEFAULT`. The cursor
holds the last listed version's `recorded_at` and `content_version`, the list's sort key, and
the next page starts after it, so a version recorded between two page reads lists on no later
page of that walk. A cursor is bound to the path id, and one that does not decode or was issued
for another id refuses `request_invalid` at `/cursor`. `complete` is true on the page that
reaches the end of the list, and `next_cursor` is null there.

Reads run on the surface's read executor; a full executor refuses retryable
`service_unavailable` (503). The door mounts before the snapshot door, whose
`GET /api/datasets/{id}/{content_version}` would otherwise read `versions` as a content
version (`datasets_api.app`).
"""

from __future__ import annotations

import base64
import binascii
import functools
import hashlib
import json
import logging
from datetime import datetime
from typing import Annotated, Final, Literal

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_serializer,
    model_serializer,
)

# Register the route converter before building routes.
from datasets_api import routing  # noqa: F401
from datasets_api.backpressure import Backpressure
from datasets_api.codes_snapshot import SpecNotFound
from datasets_api.deps import DepsCell
from datasets_api.headers import no_store_headers
from datasets_api.mint_models import GroupbyOptions
from datasets_api.recorded_rendering import recorded_minting
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.subject_ids import served_subject_ids
from datasets_api.timings import door_call, run_with_timings
from datasets_api.versions import ListedVersion, VersionLister, VersionsPageKey
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from dfo_platform.problem import Refusal
from spec_registry import IdentityQuarantined
from spec_registry.reader import SpecReader

logger = logging.getLogger(__name__)

VERSIONS_PAGE_DEFAULT: Final = 100
VERSIONS_PAGE_MAX: Final = 500

_CURSOR_DIGEST_CHARS: Final = 16


class VersionsRequestInvalid(Refusal):
    """A `limit` out of range, or a cursor that does not decode or belongs to another id. Maps
    to `request_invalid` (400) at the parameter it names."""

    code = "request_invalid"

    def __init__(self, pointer: str, detail: str) -> None:
        self.pointer = pointer
        self.detail = detail
        super().__init__(detail)


class VersionsBackpressure(Backpressure):
    """A full read executor maps to retryable `service_unavailable` (503)."""

    code = "service_unavailable"

    detail = "the versions door is at its concurrent-read capacity right now; retry"


class DatasetVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_version: str
    base_content_version: str | None = Field(
        description=(
            "the raw version a derived version (a full groupby collapse, a party-totals cut, "
            "or both) was rendered from; null on every other version"
        )
    )
    rendering: Literal["plain", "groupby", "without_party_totals", "not_recorded"] = Field(
        description=(
            "plain: a raw version, which is also what a groupby sum or a party-totals cut "
            "that left the frame unchanged is stored under. groupby: a collapse, summed by "
            "`groupby`, of the frame or of its party-totals cut when `without_party_totals` "
            "is true. without_party_totals: the frame without its party totals, unsummed. "
            "not_recorded: a derived version with no linked file recording its options"
        )
    )
    groupby: GroupbyOptions | None = Field(
        description=(
            "the groupby a collapse was summed by, as its file's rendering records it; null "
            "unless `rendering` is groupby"
        )
    )
    without_party_totals: bool = Field(
        default=False,
        description=(
            "true when the derived version's frame leaves out the group totals of its open "
            "party roles, as its file's rendering records it; absent when false"
        ),
    )

    @model_serializer(mode="wrap")
    def _without_party_totals_when_true(self, handler: SerializerFunctionWrapHandler):
        """Leave a false `without_party_totals` out, so a version that keeps its party totals
        lists without the field."""
        dumped = handler(self)
        if not self.without_party_totals:
            dumped.pop("without_party_totals", None)
        return dumped

    @field_serializer("groupby")
    def _groupby_as_recorded(self, groupby: GroupbyOptions | None):
        """Leave unset members out, as the rendering a mint records does
        (`render.artifact_rendering`), so the value equals the `groupby` the mint was sent."""
        return None if groupby is None else groupby.model_dump(mode="json", exclude_none=True)


class DatasetVersionsPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    spec_id: str
    versions: list[DatasetVersion] = Field(
        description="stored versions that are not revoked, newest first"
    )
    next_cursor: str | None = Field(
        description="continues the list after this page's last version; null when complete"
    )
    complete: bool = Field(description="true when this page reaches the end of the list")


def read_versions_page(
    *,
    id: str,
    limit: int,
    cursor: str | None,
    spec_reader: SpecReader,
    version_lister: VersionLister,
) -> DatasetVersionsPage:
    """One page of the versions `id` serves, newest first, revoked versions left out.

    Raises:
        VersionsRequestInvalid: `limit` is out of range, or `cursor` does not decode or
            belongs to another id.
        SpecNotFound: `id` is not registered.
        IdentityQuarantined: `id`, or a served id bound to it, is quarantined.
    """
    if not 1 <= limit <= VERSIONS_PAGE_MAX:
        raise VersionsRequestInvalid("/limit", f"limit must be between 1 and {VERSIONS_PAGE_MAX}")
    binding = _cursor_binding(id)
    after = _decode_cursor(cursor, binding) if cursor is not None else None
    subject_ids = served_subject_ids(id, spec_reader=spec_reader)
    # One more row than the page says whether the list continues past it.
    listed = version_lister.list_servable_versions(subject_ids, after=after, limit=limit + 1)
    page, complete = listed[:limit], len(listed) <= limit
    return DatasetVersionsPage(
        spec_id=id,
        versions=[_dataset_version(version) for version in page],
        next_cursor=None if complete else _encode_cursor(binding, page[-1]),
        complete=complete,
    )


def _dataset_version(version: ListedVersion) -> DatasetVersion:
    minting = (
        recorded_minting(version.renderings) if version.base_content_version is not None else None
    )
    if version.base_content_version is None:
        rendering = "plain"
    elif minting is None:
        rendering = "not_recorded"
    else:
        rendering = "groupby" if minting.groupby is not None else "without_party_totals"
    return DatasetVersion(
        content_version=version.content_version,
        base_content_version=version.base_content_version,
        rendering=rendering,
        groupby=minting.groupby if minting is not None else None,
        without_party_totals=minting is not None and minting.without_party_totals,
    )


def _cursor_binding(id: str) -> str:
    return hashlib.sha256(id.encode()).hexdigest()[:_CURSOR_DIGEST_CHARS]


def _encode_cursor(binding: str, last: ListedVersion) -> str:
    """Unpadded base64url over `{"b": binding, "t": recorded_at, "v": content_version}`, the
    last version a page listed."""
    payload = {"b": binding, "t": last.recorded_at.isoformat(), "v": last.content_version}
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str, binding: str) -> VersionsPageKey:
    """The sort key of the last version the cursor's page listed."""
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise VersionsRequestInvalid("/cursor", "the cursor does not decode") from None
    if not isinstance(payload, dict):
        raise VersionsRequestInvalid("/cursor", "the cursor does not decode")
    recorded_at, content_version = payload.get("t"), payload.get("v")
    if not isinstance(recorded_at, str) or not isinstance(content_version, str):
        raise VersionsRequestInvalid("/cursor", "the cursor does not decode")
    try:
        at = datetime.fromisoformat(recorded_at)
    except ValueError:
        raise VersionsRequestInvalid("/cursor", "the cursor does not decode") from None
    if payload.get("b") != binding:
        raise VersionsRequestInvalid("/cursor", "the cursor belongs to another dataset")
    return VersionsPageKey(recorded_at=at, content_version=content_version)


def build_versions_door(*, deps: DepsCell) -> DoorContribution:
    """Build the versions door over the surface's dependency cell. Reads run on the read
    executor."""
    router = APIRouter()

    @router.get("/api/datasets/{id:dfo_spec_id}/versions", response_model=DatasetVersionsPage)
    async def list_versions(
        id: str,
        limit: Annotated[
            int,
            # Declared for the OpenAPI document only: `read_versions_page` enforces the range,
            # so an out-of-range value keeps its `request_invalid` refusal and `/limit` pointer.
            Query(json_schema_extra={"minimum": 1, "maximum": VERSIONS_PAGE_MAX}),
        ] = VERSIONS_PAGE_DEFAULT,
        cursor: str | None = None,
    ) -> JSONResponse:
        current = deps.get()
        with door_call("versions", logger=logger) as call:
            page = await current.read_executor.run(
                functools.partial(
                    run_with_timings,
                    read_versions_page,
                    call.timings,
                    id=id,
                    limit=limit,
                    cursor=cursor,
                    spec_reader=current.spec_reader,
                    version_lister=current.version_lister,
                ),
                refuse=VersionsBackpressure,
            )
        return JSONResponse(
            page.model_dump(mode="json"),
            headers=no_store_headers(server_timing=call.server_timing),
        )

    return DoorContribution(
        router=router,
        registry_endpoints=route_registry_endpoints(router, "datasets.snapshot"),
        classify=classify_datasets_refusal,
        exception_types=(
            VersionsRequestInvalid,
            SpecNotFound,
            # The path id resolves through the spec registry, which raises it for a quarantined
            # path id or served id bound to it.
            IdentityQuarantined,
            VersionsBackpressure,
        ),
        exception_exemplars=(
            VersionsRequestInvalid("/cursor", "exemplar"),
            SpecNotFound(id="spec_exemplar"),
            IdentityQuarantined(subject_id="spec_exemplar", id_kind="spec"),
            VersionsBackpressure(),
        ),
    )


__all__ = [
    "VERSIONS_PAGE_DEFAULT",
    "VERSIONS_PAGE_MAX",
    "DatasetVersion",
    "DatasetVersionsPage",
    "VersionsBackpressure",
    "VersionsRequestInvalid",
    "build_versions_door",
    "read_versions_page",
]
