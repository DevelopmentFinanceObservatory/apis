"""The description door serves a version's description and pages one open dimension's values.

Routes:

- `GET /api/datasets/{id}/{content_version}/description` answers the version's description
  (`description.compose.describe_version`) in the `spec-kit` repository's description schema.
- `GET /api/datasets/{id}/{content_version}/description/values?property=&limit=&cursor=` pages
  the values the version's record holds for one open dimension, with their variable counts.

`id` is a `spec_` id with a `cv_` content version, or an `ops_` id with an `ocv_` one; one
`dfo_version_subject_id` converter routes both. A `spec_` path id resolves to the served ids the
registry binds to it, and a record kept under the posted id answers too
(`doors.snapshot.read_under_served_id`). An `ops_` id and its version must match the frame
recipe route's grammar (`publish_models.OpsSpecId`, `OcvContentVersion`), else the request is
refused `request_invalid` at `/id` or `/content_version`.

Each request reads the version's state and revocation in one transaction
(`read_servable_version`), then its record. A version that it or its base is revoked under
refuses `snapshot_revoked` (410) before `If-None-Match` is compared, so a 304 never outlives a
revocation. An unknown version, and a version that holds no record, refuse
`snapshot_not_found` (404); an unregistered `spec_` id refuses `spec_not_found` (404).

A description costs the record read, for a computed version one read of the records of the
inputs it was computed from, code-index lookups for codes, one batched label read for the places,
organisations, variables and unindexed properties the version's record names and those each
input's title names (`EnrichmentLookup.lookup`), and the source citations with their documents
(`materialise.provenance_doc.read_source_citations`), which give each source's current name,
publisher and latest download date and read nothing for a source the origin catalogue lists.
Each input is named by its title (`description.compose.version_title`). Both reads share one deadline, `_GRAPH_READ_SECONDS` (10 s)
from the start of the first. A failed, expired or size-capped graph read leaves the words at the labels the record kept at mint
(`at_mint`) and the dates it could not read as gaps, and that render answers `no-store`. Any
other render answers an `ETag` over its body and `Cache-Control: max-age=60, must-revalidate`.
The body carries no signed URL; its data link is `/api/datasets/{id}/{content_version}`.

A values page reads the record and the code index alone, and answers with the description's
cache policy. Values are ordered by the variables holding each, most first, with the value's
dcid breaking ties, an order the stored record fixes. `limit` runs from 1 to `VALUES_PAGE_MAX`.
The cursor holds the last value's count and dcid, and the next page starts after that key; it
is bound to the subject, content version and property, and one that does not decode or was
issued for another refuses `request_invalid` at `/cursor`. `complete` is true on the page that
reaches the end of the list. A property the record does not hold open, or holds with
unknown values, refuses `request_invalid` at `/property`.

Reads run on the surface's read executor; a full executor refuses retryable
`service_unavailable` (503).
"""

from __future__ import annotations

import base64
import binascii
import bisect
import functools
import hashlib
import json
import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Final

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import TypeAdapter, ValidationError
from pydantic.experimental.missing_sentinel import MISSING

from code_index.query import CodeIndex
from code_index.shared import get_measure_index
from dataset_store.snapshot import SnapshotNotFound, SnapshotRevoked, read_servable_version
from datasets_api import routing  # noqa: F401 -- registers the route converter
from datasets_api.codes_snapshot import SpecNotFound
from datasets_api.deps import Deps, DepsCell
from datasets_api.description_record import named_refs, title_named, unlabelled
from datasets_api.doors.snapshot import read_under_served_id
from datasets_api.headers import no_store_headers
from datasets_api.publish_models import OcvContentVersion, OpsSpecId
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.schema import (
    DESCRIPTION_OPENAPI_SCHEMA,
    DescribedProperty,
    DescribedValue,
    DescriptionValuesPage,
)
from datasets_api.timings import door_call, run_with_timings
from dc_graph import GraphCapExceededError, GraphReadError
from description.blocks import Words
from description.compose import describe_version, version_title
from description.facts import DescriptionRecord, DimensionValue
from description.steps import ComposedUnit, SimpleUnit, Unit
from description.vocabulary import SourceDocumentFacts, Vocabulary, build_vocabulary
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from dfo_platform.problem import Refusal
from dfo_platform.static_response import dynamic_public_json
from materialise import EnrichmentLookup
from materialise.facets import unit_currency_code
from materialise.graph import GraphReader
from materialise.provenance_doc import read_source_citations
from origin_catalogue import current_catalogue
from origin_catalogue.context import current_words_build
from spec_registry import IdentityQuarantined

logger = logging.getLogger(__name__)

DESCRIPTION_CACHE_CONTROL: Final = "max-age=60, must-revalidate"
"""A compliant cache serves a description for a minute, then revalidates it, so it stops
serving a revoked version within a minute."""

VALUES_PAGE_DEFAULT: Final = 100
VALUES_PAGE_MAX: Final = 1_000

# Bounds the label and source documentation reads of one description.
_GRAPH_READ_SECONDS: Final = 10.0
_CURSOR_DIGEST_CHARS: Final = 16

_DESCRIPTION_PATH = "/api/datasets/{id:dfo_version_subject_id}/{content_version}/description"
_VALUES_PATH = f"{_DESCRIPTION_PATH}/values"

_OPS_ID: TypeAdapter[str] = TypeAdapter(OpsSpecId)
_OCV: TypeAdapter[str] = TypeAdapter(OcvContentVersion)


class DescriptionRequestInvalid(Refusal):
    """A malformed `ops_` address, or a values page query the record cannot answer. Maps to
    `request_invalid` (400) at the parameter it names."""

    code = "request_invalid"

    def __init__(self, pointer: str, detail: str) -> None:
        self.pointer = pointer
        self.detail = detail
        super().__init__(detail)


class DescriptionBackpressure(Refusal):
    """A full read executor maps to retryable `service_unavailable` (503)."""

    code = "service_unavailable"
    retryable = True
    detail = "the description door is at its concurrent-read capacity right now; retry"


@dataclass(frozen=True, slots=True)
class RenderedDescription:
    """A description, and whether a failed graph read left words or dates out of it."""

    body: dict[str, Any]
    degraded: bool


def read_record(*, id: str, content_version: str, deps: Deps) -> DescriptionRecord:
    """The record of a servable version: under a served id for a `spec_` id, or the frame's own
    for an `ops_` id.

    Raises:
        DescriptionRequestInvalid: An `ops_` id or its version is malformed.
        SpecNotFound: A `spec_` id is unregistered.
        IdentityQuarantined: A `spec_` id, or a served id bound to it, is quarantined.
        SnapshotNotFound: The version is unknown or holds no record.
        SnapshotRevoked: The version or its base is revoked.
    """
    if id.startswith("ops_"):
        _check_frame_address(id, content_version)
        return _read_servable_record(id, content_version, revoked_under=(), deps=deps)
    return read_under_served_id(
        id=id,
        spec_reader=deps.spec_reader,
        read=lambda subject_id, subject_ids: _read_servable_record(
            subject_id, content_version, revoked_under=subject_ids, deps=deps
        ),
    )


def _check_frame_address(id: str, content_version: str) -> None:
    for pointer, adapter, value in (
        ("/id", _OPS_ID, id),
        ("/content_version", _OCV, content_version),
    ):
        try:
            adapter.validate_python(value)
        except ValidationError:
            raise DescriptionRequestInvalid(
                pointer, f"{value!r} is not a published frame's {pointer.removeprefix('/')}"
            ) from None


def _read_servable_record(
    subject_id: str, content_version: str, *, revoked_under: tuple[str, ...], deps: Deps
) -> DescriptionRecord:
    read_servable_version(
        deps.snapshot_reader,
        subject_id=subject_id,
        content_version=content_version,
        now=deps.now(),
        revoked_under=revoked_under,
    )
    record = deps.description_store.read_description(id=subject_id, content_version=content_version)
    if record is None:
        raise SnapshotNotFound(id=subject_id, content_version=content_version)
    return record


def render_description(*, id: str, content_version: str, deps: Deps) -> RenderedDescription:
    """The version's description in today's words (`read_record` lists the refusals)."""
    record = read_record(id=id, content_version=content_version, deps=deps)
    inputs = _input_records(record, deps=deps)
    vocabulary, degraded = _vocabulary(
        record,
        inputs=inputs,
        graph=deps.graph,
        enrichment=deps.enrichment,
        index=deps.code_index,
    )
    return RenderedDescription(body=describe_version(record, vocabulary), degraded=degraded)


def _input_records(record: DescriptionRecord, *, deps: Deps) -> list[DescriptionRecord]:
    """The records of the versions a computed version was computed from, as stored, in input
    order, read in one transaction. An input without an id or without a record is left out,
    and its words fall back to those its step stored."""
    if record.computed is None:
        return []
    versions = [
        (version.id, version.content_version)
        for version in record.computed.inputs
        if version.id is not None
    ]
    if not versions:
        return []
    held = deps.description_store.read_descriptions(versions=versions)
    return [held[version] for version in versions if version in held]


def _vocabulary(
    record: DescriptionRecord,
    *,
    inputs: Sequence[DescriptionRecord],
    graph: GraphReader,
    enrichment: EnrichmentLookup,
    index: CodeIndex,
) -> tuple[Vocabulary, bool]:
    """Today's words for `record`, and whether a graph read failed. A failed read leaves its
    labels or source documents out, so the composer falls back to the record's own. Each of
    `inputs`, the records of the versions `record` was computed from, is labelled by its title
    (`version_title`), worded from the same label read. The vocabulary's revision names the
    graph generation stamped as the reads start and the build of the catalogue the request is
    bound to, with the bound release (`current_words_build`)."""
    generation = graph.generation_stamp().token
    deadline = time.monotonic() + _GRAPH_READ_SECONDS
    degraded = False
    labels: dict[str, str] = {}
    indexed = _record_vocabulary(index)
    named = sorted(
        _graph_named(record, index=index, indexed=indexed).union(
            *(title_named(held, index=index, indexed=indexed) for held in inputs)
        )
    )
    if named:
        try:
            fields = enrichment.lookup(named, deadline=deadline)
        except (GraphReadError, GraphCapExceededError):
            logger.warning("description label read failed", exc_info=True)
            degraded = True
        else:
            labels = {ref: label for ref, read in fields.items() if (label := read.get("label"))}
    sources: dict[str, SourceDocumentFacts] = {}
    refs = [source.ref for source in record.evidence.sources]
    catalogue = current_catalogue()
    if refs:
        try:
            citations = read_source_citations(
                graph,
                refs,
                catalogue=catalogue,
                deadline=deadline,
                documented=True,
            )
        except (GraphReadError, GraphCapExceededError):
            logger.warning("description source read failed", exc_info=True)
            degraded = True
        else:
            for ref, citation in citations.items():
                degraded |= citation.documentation_read_failed
                if citation.name is not None and citation.name != ref:
                    labels.setdefault(ref, citation.name)
                if (doc := citation.documentation) is not None:
                    sources[ref] = SourceDocumentFacts(
                        publisher=doc.source.ref if doc.source is not None else None,
                        last_downloaded=doc.last_update,
                        next_release=doc.next_expected_update,
                    )

    def words(labels: dict[str, str]) -> Vocabulary:
        return build_vocabulary(
            index=index,
            measure_index=get_measure_index(),
            labels=labels,
            sources=sources,
            currency_units=_currency_units(record),
            graph_generation=generation,
            catalogue_build=current_words_build(),
        )

    vocabulary = words(labels)
    if inputs:
        titles = {
            held.id: title
            for held in inputs
            if (title := version_title(held, vocabulary)) is not None
        }
        vocabulary = words(labels | titles)
    return vocabulary, degraded


def _graph_named(record: DescriptionRecord, *, index: CodeIndex, indexed: Vocabulary) -> set[str]:
    """The dcids the record names whose label the code index does not give (`named_refs`):
    places, organisations and variables. A fixed or open property is read too when no role,
    code system or measure axis (`indexed`) names it (`unlabelled`)."""
    variables = record.variables
    properties = (
        []
        if variables is None
        else [
            *(fixed.property for fixed in variables.fixed),
            *(dimension.property for dimension in variables.dimensions or []),
        ]
    )
    return unlabelled(named_refs(record), properties, index=index, indexed=indexed)


def _currency_units(record: DescriptionRecord) -> dict[str, str]:
    """Each currency unit the record names, with its ISO 4217 code, read from the unit's dcid
    as `datasets_api.unit_labels.unit_currency` reads it, with no graph read."""
    units: dict[str, str] = {}
    for unit in _units(record):
        ref = unit.ref
        if ref is MISSING:
            continue
        iso = unit_currency_code(ref)
        if iso is not None:
            units[ref] = iso
    return units


def _units(record: DescriptionRecord) -> Iterator[SimpleUnit]:
    held: list[Unit] = [count.unit for count in record.series.by_source_unit or []]
    for step in record.known_steps:
        held.extend(unit for unit in (step.unit, step.converted_from) if unit is not MISSING)
    for unit in held:
        sides = (unit.numerator, unit.denominator) if isinstance(unit, ComposedUnit) else (unit,)
        yield from (side for side in sides if isinstance(side, SimpleUnit))


def read_values_page(
    *,
    id: str,
    content_version: str,
    prop: str,
    limit: int,
    cursor: str | None,
    deps: Deps,
) -> DescriptionValuesPage:
    """One page of the values the version's record holds for the open dimension `prop`.

    Raises:
        DescriptionRequestInvalid: `limit` is out of range, `prop` is not a dimension the
            record holds open with known values, or `cursor` does not decode or belongs to
            another list.
        Otherwise the refusals of `read_record`.
    """
    if not 1 <= limit <= VALUES_PAGE_MAX:
        raise DescriptionRequestInvalid("/limit", f"limit must be between 1 and {VALUES_PAGE_MAX}")
    record = read_record(id=id, content_version=content_version, deps=deps)
    values = _open_dimension(record, prop)
    words = Words.for_record(record, _record_vocabulary(deps.code_index))
    ranked = sorted(values, key=_value_key)
    binding = _cursor_binding(record, prop)
    start = 0
    if cursor is not None:
        start = bisect.bisect_right(ranked, _decode_cursor(cursor, binding), key=_value_key)
    page = ranked[start : start + limit]
    complete = start + limit >= len(ranked)
    return DescriptionValuesPage(
        property=DescribedProperty.model_validate(words.property(prop, "property")),
        items=[
            DescribedValue(
                ref=value.value, label=words.name(value.value), variables=value.variables
            )
            for value in page
        ],
        next_cursor=None if complete else _encode_cursor(binding, page[-1]),
        complete=complete,
    )


def _value_key(value: DimensionValue) -> tuple[int, str]:
    return (-value.variables, value.value)


def _open_dimension(record: DescriptionRecord, prop: str) -> list[DimensionValue]:
    dimensions = record.variables.dimensions if record.variables is not None else None
    for dimension in dimensions or []:
        if dimension.property == prop and dimension.values is not None:
            return dimension.values
    raise DescriptionRequestInvalid(
        "/property",
        f"{prop!r} is not a dimension whose values version {record.content_version} holds",
    )


def _record_vocabulary(index: CodeIndex) -> Vocabulary:
    """The words the code index gives, with no graph read: a values page words its values by
    their code labels, else the labels the record kept. A values page carries no `built_from`,
    so no graph generation or catalogue build joins the revision."""
    return build_vocabulary(
        index=index,
        measure_index=get_measure_index(),
        labels={},
        sources={},
        currency_units={},
        graph_generation=None,
        catalogue_build=None,
    )


def _cursor_binding(record: DescriptionRecord, prop: str) -> str:
    encoded = json.dumps([record.id, record.content_version, prop], separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:_CURSOR_DIGEST_CHARS]


def _encode_cursor(binding: str, last: DimensionValue) -> str:
    """Unpadded base64url over `{"b": binding, "v": variables, "d": dcid}`, the last value a
    page returned."""
    payload = {"b": binding, "v": last.variables, "d": last.value}
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str, binding: str) -> tuple[int, str]:
    """The sort key (`_value_key`) of the last value the cursor's page returned."""
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise DescriptionRequestInvalid("/cursor", "the cursor does not decode") from None
    if not isinstance(payload, dict):
        raise DescriptionRequestInvalid("/cursor", "the cursor does not decode")
    variables, dcid = payload.get("v"), payload.get("d")
    if (
        not isinstance(variables, int)
        or isinstance(variables, bool)
        or variables < 0
        or not isinstance(dcid, str)
    ):
        raise DescriptionRequestInvalid("/cursor", "the cursor does not decode")
    if payload.get("b") != binding:
        raise DescriptionRequestInvalid(
            "/cursor", "the cursor belongs to another version or property"
        )
    return (-variables, dcid)


def _cached(payload: object, request: Request, *, server_timing: str) -> Response:
    response = dynamic_public_json(payload, request, cache_control=DESCRIPTION_CACHE_CONTROL)
    response.headers["Server-Timing"] = server_timing
    return response


def build_description_door(*, deps: DepsCell) -> DoorContribution:
    """Build the description door over the surface's dependency cell. Reads run on the read
    executor."""
    router = APIRouter()

    @router.get(
        _DESCRIPTION_PATH,
        response_class=JSONResponse,
        responses={
            200: {
                "description": "the version's description",
                "content": {"application/json": {"schema": DESCRIPTION_OPENAPI_SCHEMA}},
            }
        },
    )
    async def get_description(id: str, content_version: str, request: Request) -> Response:
        current = deps.get()
        with door_call("description", logger=logger) as call:
            rendered = await current.read_executor.run(
                functools.partial(
                    run_with_timings,
                    render_description,
                    call.timings,
                    id=id,
                    content_version=content_version,
                    deps=current,
                ),
                refuse=DescriptionBackpressure,
            )
        if rendered.degraded:
            return JSONResponse(
                rendered.body, headers=no_store_headers(server_timing=call.server_timing)
            )
        return _cached(rendered.body, request, server_timing=call.server_timing)

    @router.get(_VALUES_PATH, response_model=DescriptionValuesPage)
    async def get_description_values(
        id: str,
        content_version: str,
        request: Request,
        prop: Annotated[str, Query(alias="property")],
        limit: int = VALUES_PAGE_DEFAULT,
        cursor: str | None = None,
    ) -> Response:
        current = deps.get()
        with door_call("description_values", logger=logger) as call:
            page = await current.read_executor.run(
                functools.partial(
                    run_with_timings,
                    read_values_page,
                    call.timings,
                    id=id,
                    content_version=content_version,
                    prop=prop,
                    limit=limit,
                    cursor=cursor,
                    deps=current,
                ),
                refuse=DescriptionBackpressure,
            )
        return _cached(page.model_dump(mode="json"), request, server_timing=call.server_timing)

    return DoorContribution(
        router=router,
        registry_endpoints=route_registry_endpoints(router, "datasets.description"),
        classify=classify_datasets_refusal,
        exception_types=(
            DescriptionRequestInvalid,
            SpecNotFound,
            IdentityQuarantined,
            SnapshotNotFound,
            SnapshotRevoked,
            DescriptionBackpressure,
        ),
        exception_exemplars=(
            DescriptionRequestInvalid("/property", "exemplar"),
            SpecNotFound(id="spec_exemplar"),
            IdentityQuarantined(subject_id="spec_exemplar", id_kind="spec"),
            SnapshotNotFound(id="spec_exemplar", content_version="cv_exemplar"),
            SnapshotRevoked(id="spec_exemplar", content_version="cv_exemplar", reason=None),
            DescriptionBackpressure(),
        ),
    )
