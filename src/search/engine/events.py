"""Typed Search pipeline events and their SSE frame serializer.

`resolve_events` yields these transport-neutral models. `engine.app` maps them to SSE.
The module lives under `search.engine` because `ExtractedDate` is engine-specific and the
engine package requires the `engine` extra.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dfo_platform import ProblemDetails
from search.engine.extract import ExtractedDate
from search.engine.frame import Frame
from search.models import (
    Departure,
    PipelineStepName,
    ResolveResponse,
    RoleAxis,
    SentenceToken,
    Slot,
    SlotKey,
    SlotOption,
    SlotOptions,
    SpecObservations,
    Warning,
    departures_in_precedence_order,
)

__all__ = [
    "DoneEvent",
    "DoneTelemetry",
    "EntityOptionsEvent",
    "ErrorEvent",
    "Event",
    "EventSink",
    "FacetOptionsEvent",
    "FrameEvent",
    "InterpretationEvent",
    "LateDepartures",
    "ObservationsEvent",
    "PlaceOptionsEvent",
    "ResponseEvent",
    "SentenceEvent",
    "ShapeEvent",
    "SlotOptionsEvent",
    "StepEvent",
    "serialize_sse",
]


class InterpretationEvent(BaseModel):
    """Extraction outcome; fires once, right after the extract step succeeds.

    It carries the request's door-time reading. Each part's completed frame follows after
    its recall (`FrameEvent`)."""

    model_config = ConfigDict(frozen=True)
    kind: Literal["interpretation"] = "interpretation"
    variables: list[str]
    entities: list[str]
    dates: list[ExtractedDate]
    expected_results: int = Field(
        description="== len(variables) after the SEARCH_MAX_VARIABLES clamp"
    )


class FrameEvent(BaseModel):
    """One part's completed frame: the interpretation's second firing, per part.

    Fires once per part on the raw_text and parsed doors, after recall and before the
    part's `shape` event, once the recalled shapes' vocabularies have read its words. The
    frame adds the post-recall fields to the door-time ones: the codes the words match in
    each recalled vocabulary, the normalisation form, and what the frame could not settle.
    Every field read from the question's words carries who stated it (`stated_by`). A part
    that ends before its shapes are read (an unresolved entity, a source refusal, no shape
    recalled) and a part answered from the response cache send none, and so does the
    dataset_resubmit door, which sends no interpretation.
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal["frame"] = "frame"
    variable_index: int = Field(
        description="the index of the part this event describes in the final response's parts list"
    )
    frame: Frame = Field(description="the part's frame, door-time and post-recall fields")
    denominators_on_offer: list[str] = Field(
        description=(
            "the distinct measurementDenominator dcids the part's eligible shapes carry, "
            "sorted; empty when none divides"
        )
    )


class ShapeEvent(BaseModel):
    """Per-variable skeleton, fired after admissibility and the menu cap, while bind runs.

    Carries the top admissible reading as a best-effort projection: the bind call runs
    concurrently and may elect another reading, and the status rule may answer with
    another. The ``sentence`` event carries the part's answer once its plan is minted.
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal["shape"] = "shape"
    provisional: Literal[True] = Field(
        default=True,
        description=(
            "always true: the shape is the top admissible reading while bind runs, and the "
            "part's sentence event names what it answers with"
        ),
    )
    variable_index: int = Field(
        description="the index of the part this event describes in the final response's parts list"
    )
    # Display slug from a standard or named shape, distinct from `BaseIdentity.shape`.
    shape_id: str
    label: str
    slot_keys: list[SlotKey]
    sentence_plan: list[SentenceToken] = Field(
        description="skeleton form: empty-text slot tokens, nothing bound yet"
    )
    slot_options: list[SlotOptions] = Field(
        description=(
            "candidate values with labels; coverage always null "
            "(filled later by trailing slot_options events)"
        )
    )
    refine_supported: bool = Field(
        description=(
            "true when this shape's variables support construct-and-confirm "
            "refinement on resubmit; false for ratio-backed (_AsAFractionOf_) "
            "standard shapes, where no sibling StatVar can ever be constructed"
        )
    )


class SentenceEvent(BaseModel):
    """A part's sentence, sent once per part, just ahead of the `response` event of a
    resolution that ran. A response served from the response cache goes out without one.

    It carries the sentence of the Spec the part answers with: the definite answer, the
    first candidate of a candidates part, or the flagged match a refusal carries. It is
    built from the response, so its `spec_id` equals the one the `response` event states for
    the part, a Spec two parts share after the same-shape merge included. A part with none
    of these (a refusal that flags nothing, an unsupported part) sends none. Each
    candidate's own sentence ships inside the `response` event's
    `CandidateSet.specs[*].resolved.rendered_sentence`/`sentence_plan`.
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal["sentence"] = "sentence"
    variable_index: int = Field(
        description="the index of the part this event describes in the final response's parts list"
    )
    spec_id: str
    sentence_plan: list[SentenceToken]
    slots: list[Slot] = Field(description="the bound slots this sentence renders")


class StepEvent(BaseModel):
    """One pipeline stage boundary that ran. Powers 'N of M complete'.

    variable_index is None for a request-level step that runs once ahead of the
    per-variable fan-out (extract is the only one today) and an int for every
    per-variable step (recall/shape/bind/materialise/answer).
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal["step"] = "step"
    variable_index: int | None = Field(
        default=None,
        description=(
            "the index of the part this event describes in the final response's "
            "parts list; None for a request-level step"
        ),
    )
    step: PipelineStepName
    ran: bool
    ms: int | None = None


class ResponseEvent(BaseModel):
    """The full ResolveResponse — byte-identical to the buffered JSON body."""

    model_config = ConfigDict(frozen=True)
    kind: Literal["response"] = "response"
    response: ResolveResponse


class SlotOptionsEvent(BaseModel):
    """Per-option consequence counts (trailing per-slot coverage): emitted after
    `response` (and before `done`) on an SSE request that set
    ResolveOptions.include_option_coverage. request.py's
    ResolveOptions.include_option_coverage documents the buffered/resubmit
    inline-embed alternative (no trailing events there — coverage ships directly
    on response's own Spec.resolved.slot_options instead).
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal["slot_options"] = "slot_options"
    spec_id: str = Field(
        description=(
            "the Spec this slot belongs to; matches one of `response`'s parts' specs "
            "(a definite part's own spec, or one of a candidates part's "
            "candidates.specs). Join on this, not variable_index"
        )
    )
    key: SlotKey = Field(description="the slot identity this option set belongs to")
    options: list[SlotOption] = Field(
        description=(
            "the same candidate values as response's own matching SlotOptions.options, "
            "with coverage now populated where the probe could confirm it; an option "
            "the probe could not confirm keeps coverage=null (honest, not omitted)"
        )
    )
    truncated: bool = Field(
        description=(
            "taxonomy truncation of `options` (mirrors SlotOptions.truncated) — "
            "NOT whether the coverage probe itself finished; a slot with "
            "truncated=true can still carry fully-probed coverage on the options "
            "it does list"
        )
    )


class EntityOptionsEvent(BaseModel):
    """Per-entity-option consequence counts (trailing directional-axis coverage): emitted
    after `response` (and before `done`) on an SSE request that set
    ResolveOptions.include_option_coverage, mirroring SlotOptionsEvent for the
    directional-entity (origin or destination axis) side. Dev-finance only in v1.
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal["entity_options"] = "entity_options"
    spec_id: str = Field(
        description=(
            "the Spec this entity option set belongs to, the spec of one of "
            "`response`'s parts; join on this field"
        )
    )
    entity_ref: str | None = Field(
        default=None,
        description=(
            "dcid of the currently-bound directional entity this option set replaces; "
            "None means this role axis is open -- no entity is bound on it, and "
            "`options` names its own candidate member set rather than a swap "
            "target for a bound one (mirrors EntityOptions.entity_ref)"
        ),
    )
    role: RoleAxis = Field(description="which flow axis this entity is bound on")
    options: list[SlotOption] = Field(
        description=(
            "the same candidate entities as response's own matching EntityOptions.options, "
            "with coverage now populated where the probe could confirm it; an option the "
            "probe could not confirm keeps coverage=null (honest, not omitted)"
        )
    )
    truncated: bool = Field(
        description=(
            "candidate-set truncation of `options` (mirrors EntityOptions.truncated) -- "
            "NOT whether the coverage probe itself finished"
        )
    )


class FacetOptionsEvent(BaseModel):
    """Per-facet-axis candidate values (trailing facet coverage): emitted after
    `response` (and before `done`) on an SSE request that set
    ResolveOptions.include_option_coverage, mirroring SlotOptionsEvent/
    EntityOptionsEvent for the facet-axis (provenance/measurementMethod/unit)
    side. Unlike either, there is no separate confirm step behind this event --
    every option's SlotOption.coverage stays absent (buffered) or null (SSE,
    per SlotOption's own wrap-serializer rule), never populated.
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal["facet_options"] = "facet_options"
    spec_id: str = Field(
        description=(
            "the Spec this facet option set belongs to; matches one of `response`'s "
            "parts' specs (a definite part's own spec, or one of a candidates part's "
            "candidates.specs). Join on this, not variable_index"
        )
    )
    # Keep `axis` open in JSON Schema. Adding a facet axis stays additive.
    axis: str = Field(description="the facet axis this option set belongs to (see FacetAxis)")
    options: list[SlotOption] = Field(
        description="the same candidate values as response's own matching FacetOptions.options"
    )
    truncated: bool = Field(description="mirrors FacetOptions.truncated")


class PlaceOptionsEvent(BaseModel):
    """Prefetched observationAbout place candidates for one Spec (trailing):
    emitted after `response` (and before `done`) on an SSE request that set
    ResolveOptions.include_option_coverage, for a definite part whose Spec
    binds an observationAbout place. Sibling of SlotOptionsEvent/
    EntityOptionsEvent/FacetOptionsEvent under the same gate, but keyed on
    the OTHER door entirely: `options` and `truncated` are the /options
    door's own `OptionsSet.items`/`OptionsSet.truncated`
    (`search.models.OptionsResponse.options`), built by
    `region.place_options.build_place_options_events` from exactly the
    identity `region.pipeline`/`region.resubmit_promote` already resolved
    -- `resolve_options_target` and `compute_options`, the same pair a POST
    to /api/search/options would run for this Spec's own identity.
    The point is to let a client open the "Select place" popover on this
    anchor with zero network round trip: the anchor's peers and children
    candidate sets are already on the wire by the time a user could click.

    One event per (spec_id, scope) pair that succeeded; up to two per Spec
    (`peers` and `children`), fired independently of each other -- a scope
    that came back refused (e.g. `unsupported_peers_anchor_type`,
    `children_anchor_required`) or failed (a graph transport error) is
    dropped rather than emitted with an error or a placeholder, mirroring
    slot_options' own fail-open posture for a probe failure. No event at
    all fires for a Spec whose identity binds no observationAbout place --
    there is no anchor for a client to pop a place editor open on.
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal["place_options"] = "place_options"
    spec_id: str = Field(
        description=(
            "the Spec this place option set belongs to; matches one of `response`'s "
            "parts' specs (a definite part's own spec, or one of a candidates part's "
            "candidates.specs). Join on this, not variable_index"
        )
    )
    role: Literal["observationAbout"] = Field(
        description=(
            "always observationAbout: the subject-place role this event previews, "
            "never a directional (origin/destination) axis -- see EntityOptionsEvent "
            "for that side"
        )
    )
    scope: Literal["peers", "children"] = Field(
        description=(
            "which candidate universe this option set enumerates -- peers (places "
            "sharing the anchor's own type, under Earth) or children (places "
            "contained in the anchor), mirroring OptionsFor.scope"
        )
    )
    anchor_ref: str = Field(
        description=(
            "dcid of the bound observationAbout entity this option set was derived "
            "for (the first entry of identity.entities['observationAbout'] when more "
            "than one is bound); a client swaps to one of `options` to replace it"
        )
    )
    options: list[SlotOption] = Field(
        description=(
            "the same candidate places as a POST to /api/search/options would "
            "return in OptionsResponse.options.items for this identity, role, and "
            "scope, coverage populated (include_coverage=true on every derivation)"
        )
    )
    truncated: bool = Field(description="mirrors OptionsResponse.options.truncated for this scope")


class ObservationsEvent(BaseModel):
    """Trailing observation values for one Spec: emitted after `response` (and
    before `done`) on an SSE request. Unlike SlotOptionsEvent/EntityOptionsEvent
    (opt-in via include_option_coverage, off by default), observations are
    emitted only when ResolveOptions.include_observations is true. This event
    is available only for a part that committed to a definite Spec; a
    request naming several variables can commit several parts, each firing
    its own ObservationsEvent, joined by spec_id. A candidates part keeps its
    candidate metadata and coverage without observation fetches. Mirrors the
    same transport split: the buffered JSON body embeds observations directly
    on Spec.resolved.observations instead, and the response event's own
    Spec.resolved.observations stays null or omitted on SSE.

    Every definite part that requested observations gets exactly one of these
    events: `observations` always carries a real SpecObservations, whose own
    `omitted` field (search.models.ObservationsOmission) names why when there
    was nothing to show -- a fetch that timed out past the request deadline
    synthesises one too (enrichment.py's deadline handling), so an absent
    event is a protocol violation a client can treat as a bug report, not a
    state to interpret.
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal["observations"] = "observations"
    spec_id: str = Field(
        description=(
            "the Spec this observation payload belongs to; matches one of `response`'s "
            "parts' own definite spec (or, on a resubmit, the resubmitted Spec). "
            "Join on this, not variable_index"
        )
    )
    observations: SpecObservations = Field(
        description=(
            "the same payload shape Spec.resolved.observations carries on the buffered path"
        )
    )


class DoneTelemetry(BaseModel):
    """Telemetry block carried in the terminal ``done`` event."""

    model_config = ConfigDict(frozen=True)
    llm_usage: dict[str, int] | None = None


class LateDepartures(BaseModel):
    """A streamed definite part's complete departures once its observations are read."""

    model_config = ConfigDict(frozen=True)
    part_index: int = Field(
        ge=0,
        description="the index in `response.parts` of the definite part these departures replace",
    )
    spec_id: str = Field(
        description=(
            "the spec_id the part answers with; a part folded into an earlier one shares "
            "that part's spec_id, so `part_index` names the part"
        )
    )
    departures: list[Departure] = Field(
        description=(
            "the part's complete `departures` once its observations are read, built by "
            "the same rules as the list it shipped, one entity_unavailable per role, in "
            "DepartureCode precedence order; a client replaces the part's list with it"
        ),
    )

    @field_validator("departures")
    @classmethod
    def _departures_in_precedence_order(cls, value: list[Departure]) -> list[Departure]:
        return departures_in_precedence_order(value)


class DoneEvent(BaseModel):
    """Terminal event on success OR soft-timeout (partials already folded into ``response``)."""

    model_config = ConfigDict(frozen=True)
    kind: Literal["done"] = "done"
    elapsed_ms: int
    timed_out: bool = False
    degraded_leg: bool = Field(
        default=False,
        exclude=True,
        description=(
            "a leg of the answer failed and was backfilled; read in process by the buffered "
            "route, which reports it as the X-Search-Degraded-Leg header. Never serialised"
        ),
    )
    trace_id: str = Field(
        description="this request's trace_id, shared with diagnostics and every problem-details body"
    )
    telemetry: DoneTelemetry
    warnings: list[Warning] = Field(
        default_factory=list,
        description=(
            "late-arriving diagnostics from trailing enrichment probes, e.g. "
            "OPTION_COVERAGE_PROBE_FAILED or OPTION_COVERAGE_TIMED_OUT. Never "
            "present in `response`'s own diagnostics.warnings, since a trailing "
            "probe's outcome isn't known until after `response` has shipped. Empty "
            "unless the request opted into trailing enrichment (SSE + "
            "include_option_coverage) and a probe failed or was deadline-cancelled."
        ),
    )
    late_departures: list[LateDepartures] = Field(
        default_factory=list,
        description=(
            "the complete departures of each streamed definite part whose observations "
            "read, after `response` has shipped, changes them: modelled or projected values "
            "(modelled_values) and places the UN front door refused (entity_unavailable, "
            "merged with the part's own for that role). One entry per such part, named by "
            "`part_index`, present only when the list differs from the one the part shipped; "
            "a folded part and the part it folded into each get their own entry, built from "
            "their own words and warnings. A client replaces the part's `departures` with it. Empty unless the request streamed "
            "observations. A buffered answer's part already carries the complete list"
        ),
    )


class ErrorEvent(ProblemDetails):
    """Terminal failure — route-synthesized only; the pipeline generator never
    yields this. Reuses ProblemDetails verbatim (RFC 9457 base fields plus the
    platform registry's stable extensions) so the SSE terminal error and a
    non-2xx JSON failure can never drift apart; adds only the `kind` SSE
    discriminator.
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal["error"] = "error"


Event = Annotated[
    InterpretationEvent
    | FrameEvent
    | ShapeEvent
    | SentenceEvent
    | StepEvent
    | ResponseEvent
    | SlotOptionsEvent
    | EntityOptionsEvent
    | FacetOptionsEvent
    | PlaceOptionsEvent
    | ObservationsEvent
    | DoneEvent
    | ErrorEvent,
    Field(discriminator="kind"),
]

EventSink = Callable[[Event], Awaitable[None]]


def serialize_sse(event: Event) -> str:
    """Serialize one event to an SSE frame: `event:` line + `data:` JSON line + blank line."""
    # Omit optional nulls. The SSE surface context preserves meaningful
    # `coverage=null`, which buffered JSON omits.
    payload = event.model_dump_json(exclude_none=True, context={"surface": "sse"})  # type: ignore[union-attr]
    return f"event: {event.kind}\ndata: {payload}\n\n"  # type: ignore[union-attr]
