"""Response models document Presentation routes in OpenAPI. Handlers build response bodies.
`response_model` validates them, while `responses` documents already-built `Response` objects.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from datasets_api.manifest_models import RetentionModel

__all__ = [
    "AppliedDefaultModel",
    "CapabilitiesResponse",
    "ChangedFieldModel",
    "ComponentReadResponse",
    "ComposedUnitModel",
    "ConvertedUnitModel",
    "CountedStatVarModel",
    "CreateReplyResponse",
    "FactSummaryModel",
    "FactSummarySeriesEntry",
    "FrameDimensionMatch",
    "FrameDimensionSummary",
    "FrameDimensionValuesResponse",
    "FrameFilteredViewResponse",
    "FrameStructuralViewResponse",
    "FrameViewResponse",
    "ReleaseDocumentResponse",
    "ShareCardModel",
    "ShareFrameModel",
    "ShareReadResponse",
    "ShareSlotModel",
    "ShareWriteResponse",
    "SharedResponse",
    "WindowModel",
    "WriteReplyResponse",
]

# -- fact summary (assembler.reply) -----------------------------------------


class FactSummarySeriesEntry(BaseModel):
    """One `fact_summary.series[]` entry. It reports one visible series'
    identity, point count and formatted boundary values."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(description="this series' own id, null when none could be assigned")
    label: str = Field(description="this series' own display label")
    coordinates: dict[str, object] = Field(description="this series' own dimension coordinates")
    point_count: int = Field(description="how many of this series' points carry an observed value")
    first_value: str | None = Field(
        description="this series' earliest observed value, formatted, null when it has none"
    )
    last_value: str | None = Field(
        description="this series' latest observed value, formatted, null when it has none"
    )


class WindowModel(BaseModel):
    """A period window, its two boundary labels at the frame's grain."""

    model_config = ConfigDict(extra="forbid")

    start: str = Field(description="this window's own start period label")
    end: str = Field(description="this window's own end period label")


class AppliedDefaultModel(BaseModel):
    """One `fact_summary.defaults[]` entry: a choice the component made where its intent
    left one open. An entity-mode bar_chart whose select lists no values draws every value
    of its one varying dimension, largest first."""

    model_config = ConfigDict(extra="forbid")

    pointer: str = Field(description="a JSON Pointer to the intent field this default fills")
    detail: str = Field(
        description="what the component chose and how to narrow it, in plain language"
    )


class ConvertedUnitModel(BaseModel):
    """A converted unit, as the frame's `unit` cell states it."""

    model_config = ConfigDict(extra="forbid")

    currency: str = Field(description="the ISO 4217 code of the currency the values are in")
    prices: Literal["current", "constant"] = Field(description="the price basis of the values")
    base_year: int | None = Field(
        default=None, description="the base year of constant prices, absent for current prices"
    )


class CountedStatVarModel(BaseModel):
    """One side of a composed unit whose measure declares no unit, named by its StatVar."""

    model_config = ConfigDict(extra="forbid")

    stat_var: str = Field(description="the StatVar's `dcid:`-prefixed ref, as the frame holds it")
    label: str = Field(
        description="the StatVar's graph name, or its ref without `dcid:` when the graph has none"
    )


class ComposedUnitModel(BaseModel):
    """The unit of a quotient of two measures, read `<numerator> per <denominator>`.

    Each side is a unit's graph name (its ref when the graph has none), a converted unit, or
    the StatVar a unit-less side counts."""

    model_config = ConfigDict(extra="forbid")

    numerator: str | ConvertedUnitModel | CountedStatVarModel = Field(
        description="the unit of the measure divided"
    )
    denominator: str | ConvertedUnitModel | CountedStatVarModel = Field(
        description="the unit of the measure divided by"
    )


class FactSummaryModel(BaseModel):
    """The closed, token-capped fact summary a revision reports."""

    model_config = ConfigDict(extra="forbid")

    changed: list[str] = Field(
        description="the patch's own changed field names, empty for a create or a no-op write"
    )
    series: list[FactSummarySeriesEntry] = Field(
        description="every visible series this revision resolves, capped"
    )
    series_cut: int = Field(
        description="how many series the series cap and the token cap together dropped from "
        "this summary"
    )
    window: WindowModel | None = Field(
        description="this component's own [start, end] period window, null where its type "
        "defines none"
    )
    unit: str | ConvertedUnitModel | ComposedUnitModel | None = Field(
        description="this component's own unit, the value its resolved payload carries: a "
        "unit's graph name (its ref when the graph has none), a converted unit, or a composed "
        "unit. A line_chart or bar_chart reports its first series' unit and a table the unit "
        "every row shares. Null where the type defines none, the rows declare none, or a "
        "table's rows differ."
    )
    change: dict[str, object] | None = Field(
        description="the stat_card's derived comparison between its two basis periods, null "
        "for every other member and for a stat_card whose intent carries no change"
    )
    source_refs: list[str] | None = Field(
        description="every distinct raw provenance ref a source_card's frame carries. It is "
        "null for every other member."
    )
    defaults: list[AppliedDefaultModel] = Field(
        description="the choices this component made where its intent left one open, empty "
        "when it made none"
    )


# -- write reply (presentation_api.doors.components) -----------------------

_RESOLVED_PAYLOAD_DESCRIPTION = (
    "the revision's stored resolved payload, verbatim: the catalogue envelope "
    "{schema_version, catalog_version, component_id, revision, provenance, component}, "
    "validated against the catalogue release's envelope schema (dfo-spec-kit "
    "envelope.schema.json) when minted. GET /api/presentation/release serves the pinned "
    "release's envelope_schema and the component_schemas its component field $refs by $id; "
    "every payload this door serves carries that pinned catalog_version."
)


class ChangedFieldModel(BaseModel):
    """One entry of `changed[]`. `prior` and `new` are present only for a
    scalar field the write set changed. An array or object field carries
    `field` alone."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(description="the intent field this entry reports")
    prior: object | None = Field(
        default=None, description="the field's value before the write, for a disclosed scalar"
    )
    new: object | None = Field(
        default=None, description="the field's value after the write, for a disclosed scalar"
    )


class WriteReplyResponse(BaseModel):
    """`PATCH /api/presentation/components/{component_id}`'s 200 body. A
    create reply carries this same shape plus `affordance`
    (`CreateReplyResponse` below). A revise reply carries no `affordance`
    key."""

    model_config = ConfigDict(extra="forbid")

    component_id: str = Field(description="the revised component's own id")
    revision: int = Field(description="the revision this write minted, or the head left unchanged")
    catalog_version: str = Field(description="the catalogue release this revision minted under")
    composition_id: str | None = Field(
        description="the composition this component belongs to, null outside one"
    )
    composition_revision: int | None = Field(
        description="the composition's current head revision, null outside one"
    )
    changed: list[ChangedFieldModel] = Field(
        description="the patch's own fields, each paired with its prior and new value. This "
        "array is empty when the write was a no-op."
    )
    fact_summary: FactSummaryModel = Field(
        description="the closed, token-capped fact summary this revision reports"
    )
    resolved_payload: dict[str, object] = Field(description=_RESOLVED_PAYLOAD_DESCRIPTION)


class CreateReplyResponse(WriteReplyResponse):
    """`POST /api/presentation/components`'s 201 body. It carries
    `WriteReplyResponse`'s fields plus this door's own `affordance` line
    naming the freshly minted component's revise and read routes."""

    component_id: str = Field(description="the newly minted component's own id")
    revision: int = Field(description="the first revision a create mints, always 1")
    changed: list[ChangedFieldModel] = Field(
        description="always empty, since a create carries no prior intent to compare against"
    )
    affordance: str = Field(
        description="this door's own line naming the revise and read routes for this component"
    )
    write_token: str = Field(
        description="the cmw_ token that grants revising this component, sent back as the "
        "x-dfo-write-token header. A server-minted token appears only in this reply; a token "
        "the create sent as write_token is echoed, and a replayed create returns it again."
    )
    composition_write_token: str | None = Field(
        default=None,
        description="the cpw_ token that grants creating components into the composition "
        'this create started with composition_id "new", sent back as the x-dfo-write-token '
        "header. Present only on that create.",
    )


# -- component read (presentation_api.doors.components) --------------------


class ComponentReadResponse(BaseModel):
    """`GET /api/presentation/components/{component_id}`'s 200 body."""

    model_config = ConfigDict(extra="forbid")

    component_id: str = Field(description="the read component's own id")
    revision: int = Field(description="the revision this read reports")
    catalog_version: str = Field(description="the catalogue release this revision minted under")
    composition_id: str | None = Field(
        description="the composition this component belongs to, null outside one"
    )
    composition_revision: int | None = Field(
        description="the composition's current head revision, null outside one"
    )
    stored_intent: dict[str, object] = Field(
        description="this revision's own stored intent, exactly as minted"
    )
    fact_summary: FactSummaryModel = Field(
        description="the closed, token-capped fact summary this revision reports"
    )
    resolved_payload: dict[str, object] = Field(description=_RESOLVED_PAYLOAD_DESCRIPTION)
    retention: RetentionModel | None = Field(
        default=None,
        description="how long the frame this reply draws can still be fetched, counted from "
        "when DFO stored it: the stored frame on a read, the named frame on a render. A card's "
        "frame stays in the 30-day working tier until a share first includes the card. Once "
        "`held_until` has passed the frame can refuse `frame_unavailable`; fetch the card's "
        "data with latest=True to draw from current data. Null for a card that draws no frame",
    )


# -- shares (presentation_api.doors.shares) ---------------------------------


class ShareWriteResponse(BaseModel):
    """`PUT /api/presentation/shares/{share_id}`'s 200 body: where the replaced share lives."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="the share's id, 8 letters and digits")
    url: str = Field(description="the share's page on the public website, /s/<id>")
    embed_url: str = Field(description="the share's embeddable page, /embed/<id>")
    data_urls: list[str] = Field(
        description="one JSON data URL per slot, in slot order. A slot keeps its URL across "
        "replacements, so data_urls[0] always reads the share's first slot"
    )
    revision: int = Field(description="the share revision this write made, 1 on a create")
    latest: bool = Field(
        description="true for a live share, which follows new data; false for a snapshot, "
        "which keeps the versions its cards drew when it was made"
    )


class SharedResponse(ShareWriteResponse):
    """`POST /api/presentation/shares`'s 201 body."""

    write_token: str = Field(
        description="the shw_ token that grants replacing and deleting this share, sent back as "
        "the x-dfo-write-token header. It appears only in this reply; keep it as a secret"
    )


class ShareFrameModel(BaseModel):
    """The frame a slot's card revision draws."""

    model_config = ConfigDict(extra="forbid")

    spec_id: str = Field(description="the frame's spec_ or ops_ id")
    content_version: str = Field(description="the frame's cv_ or ocv_ content version")


class ShareCardModel(BaseModel):
    """The card revision a slot shows."""

    model_config = ConfigDict(extra="forbid")

    component_id: str = Field(description="the card's cmp_ id")
    revision: int = Field(description="the card revision the slot shows")
    frame: ShareFrameModel | None = Field(
        description="the frame the card revision draws, null for a card that draws none or an "
        "imported slot whose frame was not recorded"
    )


class ShareSlotModel(BaseModel):
    """One numbered slot of a share."""

    model_config = ConfigDict(extra="forbid")

    slot: int = Field(description="the slot's number, from 1")
    card: ShareCardModel | None = Field(
        description="the card the slot shows, null for a slot a replacement with fewer cards "
        "left empty"
    )
    data_url: str = Field(description="the slot's JSON data URL")


class ShareReadResponse(BaseModel):
    """`GET /api/presentation/shares/{share_id}`'s 200 body. It never carries the write
    token."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="the share's id, 8 letters and digits")
    kind: Literal["card", "canvas", "search", "conversation"] = Field(
        description="what the share publishes"
    )
    mode: Literal["live", "snapshot"] = Field(
        description="live follows new data; snapshot keeps the versions its cards drew when it "
        "was made"
    )
    title: str | None = Field(description="the share's title, null when it has none")
    revision: int = Field(description="the share's latest revision")
    slots: list[ShareSlotModel] = Field(
        description="every slot the share has had, in order. Search and conversation shares "
        "have none"
    )
    payload: object = Field(
        description="website-owned JSON stored with the share as given, null when there is none"
    )
    created_at: str = Field(description="when the share was made, as an ISO 8601 timestamp")


# -- meta (presentation_api.doors.meta) -------------------------------------


class CapabilitiesResponse(BaseModel):
    """`GET /api/presentation/capabilities`'s body. It carries this
    surface's own hashed `capability_version` beside the unhashed
    `catalog_version`."""

    model_config = ConfigDict(extra="forbid")

    capability_version: str = Field(
        description="a content hash over every capability-bearing field of the release "
        "this deployment serves"
    )
    catalog_version: str = Field(description="the catalogue release this deployment serves")
    build: str = Field(
        description="the commit sha the image was built from, the same value the "
        "x-dfo-commit response header carries"
    )


# -- frame view (presentation_api.doors.frames) -----------------------------


class FrameDimensionSummary(BaseModel):
    """One dimension of a frame's structural view (`GET .../frames/{...}`
    with neither `dimension` nor `query` given). Its values are capped to
    the per-dimension listing cap."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="this dimension's own column name")
    values: list[str] = Field(description="this dimension's own values, capped")
    total_count: int = Field(description="this dimension's own value count before capping")
    truncated: bool = Field(description="whether this dimension's values list was capped")
    plural: bool = Field(
        description=(
            "whether this frame's series hold more than one value of this dimension, a null "
            "counting as one; the estimates continuing a series hold its method, so they "
            "leave measurementMethod not plural"
        )
    )


class FrameDimensionMatch(BaseModel):
    """One dimension of a frame's query-filtered view (`?query=` given,
    `dimension` absent). Its values are filtered to those matching the
    query."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="this dimension's own column name")
    values: list[str] = Field(description="this dimension's own values matching the query")
    matched_count: int = Field(description="how many of this dimension's values matched the query")


class FrameStructuralViewResponse(BaseModel):
    """`GET /api/presentation/frames/{spec_id}/{content_version}`'s body
    with neither `dimension` nor `query` given. It reports the frame's own
    structural view, every dimension capped."""

    model_config = ConfigDict(extra="forbid")

    dimensions: list[FrameDimensionSummary] = Field(
        description="every dimension this frame declares"
    )
    period_grain: str = Field(description="this frame's own period grain")
    coverage: tuple[str, str] | None = Field(
        description="this frame's own [start, end] period coverage, null if it has none"
    )
    series_count: int = Field(description="how many distinct series this frame resolves")


class FrameDimensionValuesResponse(BaseModel):
    """`GET /api/presentation/frames/{spec_id}/{content_version}?dimension=`'s
    body. It reports one dimension's full value list, read fresh and
    uncapped. Given alongside `?query=`, `values` and `total_count`
    describe only the values matching the query."""

    model_config = ConfigDict(extra="forbid")

    dimension: str = Field(description="the dimension this response reports")
    values: list[str] = Field(
        description="every value this dimension carries, uncapped, or only the values "
        "matching `query` when it is given"
    )
    total_count: int = Field(
        description="how many values this dimension carries, or how many match `query` when "
        "it is given"
    )


class FrameFilteredViewResponse(BaseModel):
    """`GET /api/presentation/frames/{spec_id}/{content_version}?query=`'s
    body (`dimension` absent). It reports every dimension this frame
    declares, filtered to values matching the query, counts kept."""

    model_config = ConfigDict(extra="forbid")

    dimensions: list[FrameDimensionMatch] = Field(
        description="every dimension this frame declares, filtered"
    )
    period_grain: str = Field(description="this frame's own period grain")
    coverage: tuple[str, str] | None = Field(
        description="this frame's own [start, end] period coverage, null if it has none"
    )


FrameViewResponse = (
    FrameStructuralViewResponse | FrameDimensionValuesResponse | FrameFilteredViewResponse
)


# -- release (presentation_api.doors.release) -------------------------------


class ReleaseDocumentResponse(BaseModel):
    """`GET /api/presentation/release`'s body. It carries the pinned
    catalogue release's own identity and the vocabulary a client needs
    to mount tools without a second, out-of-band fetch of the catalogue
    bundle."""

    model_config = ConfigDict(extra="forbid")

    catalog_version: str = Field(description="the catalogue release this deployment serves")
    schema_version: int = Field(
        description="the envelope schema version this release resolves against"
    )
    member_types: list[str] = Field(
        description="every component type this release's catalogue defines"
    )
    constraint_ids: list[str] = Field(
        description="every constraint id this release's registry enforces"
    )
    retry_modes: dict[str, str] = Field(
        description="`retry_modes` is the headline mode per code, and the per-response "
        "`retryable` boolean is the authoritative value for every code. `schema_invalid` and "
        "`frame_unavailable` vary by cause, and `request_invalid` is always false although its "
        "headline mode is `correct_and_retry`."
    )
    envelope_schema: dict[str, object] = Field(
        description="the envelope JSON Schema this release's assembler validates a minted "
        "envelope against; every create/revise/read body's `resolved_payload` conforms to it"
    )
    component_schemas: dict[str, dict[str, object]] = Field(
        description="each member type's own component JSON Schema, keyed by member type. "
        "`envelope_schema`'s `component` field `$ref`s these schemas by their own `$id`, which "
        "otherwise resolves nowhere, so a client validating `resolved_payload` resolves those "
        "`$ref`s against this map"
    )
    intent_schemas_tolerant: dict[str, dict[str, object]] = Field(
        description="each member type's own tolerant intent schema"
    )
    minimal_intents: dict[str, dict[str, object]] = Field(
        description="each member type's own minimal legal intent"
    )
