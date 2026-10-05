from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    computed_field,
    field_validator,
)

from dfo_platform import Cause

from .base import StrictModel
from .departure import Departure, departures_in_precedence_order
from .envelope import Diagnostics, QueryEcho, Timing, Warning
from .grammar import GraphRef, SentenceToken
from .outcomes import CandidateSet, NoData
from .request import _OPERATIONS_DESCRIPTION, Operation, ParsedPart
from .spec import Spec
from .vocab import (
    SCHEMA_VERSION,
    DeclinedReason,
    StatusLiteral,
    _require_schema_version_int,
)

PartRestatementReason = Literal[
    "no_phrase_inputs",
    "direction_unnameable",
    "inputs_exceed_caps",
]
"""Why a Part carries no `restatement` (see `_PartBase.restatement_reason`).

`no_phrase_inputs`: the part was never resolved from a phrase, so it has no
per-part resolution inputs to echo at all. The `dataset_resubmit` door's
`identity`, `candidates` and `placeholder` entry kinds post an identity, a
candidate set, or a stub outcome rather than a phrase, and an identity entry
re-resolves an already-identified spec the same way.

`direction_unnameable`: the leg's direction came from wording alone. Retrieval
found the role wording in a leg's own role query, so the engine could not echo
that leg's entities bare without reversing them, and no door-supplied role on
the elected shape named the axis an echo would need.

`inputs_exceed_caps`: one of the leg's own resolution inputs sits outside the
field bounds `ParsedPart` accepts, so an echo could only be truncated and
would misstate what was resolved.

Defined here rather than in `.vocab` because no request-side model shares it:
this vocabulary is the response's own account of a null `restatement`.
"""


class _PartBase(StrictModel):
    """Fields every part variant carries, whatever its status."""

    variable_text: str | None = Field(
        default=None,
        description=(
            "the variable text this part answers, in the order asked. This part's "
            "position in Resolution.parts equals the same variable_index every "
            "SSE event keyed by variable_index uses (ShapeEvent, StepEvent, "
            "SentenceEvent). Every dataset_resubmit entry kind (identity, "
            "candidates, placeholder, and parsed through its nested ParsedPart) "
            "requires its own posted variable_text, so this field is always the "
            "posted text echoed back onto the rebuilt part, never null. Null only "
            "when the whole posted query was empty or whitespace-only (raw_text "
            "door), which refuses variable_not_resolved before extraction runs and "
            "so has no text to name."
        ),
    )
    warnings: list[Warning] = Field(
        default_factory=list,
        description=(
            "this part's own run-health signals. A part folded into an earlier part's "
            "spec (see DefinitePart.spec) carries binding_folded alone. The merged "
            "spec's own resolved warnings live only on the first part that carries it. "
            "An unfolded part carries the warnings raised while resolving this part, "
            "plus observations_omitted on a candidates part when response_options."
            "include_observations was set."
        ),
    )
    timing_ms: Timing | None = Field(
        default=None,
        description=(
            "this part's own latency breakdown, where by_step names only the phases "
            "this part ran and total sums them. Excludes the shared, once-per-request "
            "steps (e.g. extract), which diagnostics.timing_ms.by_step carries once "
            "for the whole request; distinct too from diagnostics.timing_ms.total, "
            "the whole request's wall-clock elapsed time."
        ),
    )
    rendered_sentence: str | None = Field(
        default=None,
        description=(
            "a human-readable string when response_options.include_sentence is set; null "
            "or omitted otherwise. For candidates, a header naming how many readings "
            "there are ('3 readings.'), then each shown candidate's heading and a byline "
            "led by the source and span that set it apart; for no_data, a heading in "
            "the same claim form naming the entity, measure, and counterpart that were "
            "checked. Either ends with the text of this part's departures, in order. "
            "For a definite part this stays null or omitted: the confirmation sentence "
            "(heading and byline) lives at spec.resolved.rendered_sentence, and its "
            "departures reach the envelope's composed sentence."
        ),
    )
    sentence_plan: list[SentenceToken] | None = Field(
        default=None,
        description=(
            "the ordered token form of rendered_sentence, present under the same "
            "condition (response_options.include_sentence); null or omitted otherwise. "
            "For a definite part this stays null or omitted. spec.resolved.sentence_plan "
            "carries the confirmation sentence's own plan instead."
        ),
    )
    restatement: ParsedPart | None = Field(
        default=None,
        description=(
            "the ParsedPart that reproduces this part's own resolution inputs -- the "
            "same entities, resolved time window, unit text, and source text the "
            "engine actually used to resolve this one leg -- so a client holding this "
            "part alone can copy it straight into a fresh ParsedInput.parts entry and "
            "re-resolve it faithfully, without flattening it into the whole query's "
            "variable_text the way variable_text alone would. Canonical, not verbatim: "
            "for a raw_text part, time_text is the equivalent phrase this door's own "
            "time grammar recognises (e.g. '2015 to 2020'), not the user's original "
            "words, since the engine's date extraction keeps only the structured "
            "window and discards the phrasing. Null whenever this part has no faithful "
            "echo to give; restatement_reason names which case nulled it. On a "
            "dataset_resubmit entry's identity/candidates/placeholder kinds, which "
            "post an identity or stub rather than a phrase and have no per-part "
            "resolution inputs to echo -- a "
            "dataset_resubmit parsed entry, genuinely re-resolved from its own posted "
            "ParsedPart, carries a real one like any parsed-door part. Also null when "
            "this leg's entity was bound to a directional role axis "
            "(origin/destination) and no door-supplied role named that axis, since "
            "an echo without the axis would resubmit the request with that entity "
            "reversed. Also null when one of this leg's own inputs does not fit "
            "ParsedPart's own field caps, since a truncated echo would misrepresent "
            "what was actually resolved."
        ),
    )
    restatement_reason: PartRestatementReason | None = Field(
        default=None,
        description=(
            "why this part carries no restatement, present exactly when restatement "
            "is null and absent whenever it is present; the two fields are always "
            "one-null-one-set, so a client reads restatement when it is there and "
            "restatement_reason when it is not, and never has to guess at a bare "
            "null. 'no_phrase_inputs': this part was never resolved from a phrase "
            "(a dataset_resubmit identity/candidates/placeholder entry), so there "
            "were no resolution inputs to echo. "
            "'direction_unnameable': the leg's direction came from its own wording "
            "alone and no door-supplied role named the axis, so echoing its entities "
            "would have reversed them. 'inputs_exceed_caps': one of the leg's own "
            "resolution inputs sits outside ParsedPart's own field bounds. A "
            "dfo-apis wire field outside the spec-kit response contract."
        ),
    )
    operations: list[Operation] = Field(default_factory=list, description=_OPERATIONS_DESCRIPTION)
    departures: list[Departure] = Field(
        default_factory=list,
        description=(
            "every way this part's answer differs from what its request asked, "
            "ordered by how much each changes the number: closest_match, "
            "measure_not_applied, operation_not_applied, breakdown_not_applied, "
            "entity_unavailable, year_unavailable, modelled_values, "
            "group_read_as_members, direction_discarded, scope_limited. [] when the answer is what was asked. Present "
            "whether or not include_sentence is set, so a client that renders its "
            "own sentence still reads them; under include_sentence each one's text "
            "also closes the sentence that carries this part (see Departure.text). "
            "A warning on the same part reports run health and stays separate"
        ),
    )

    @field_validator("departures")
    @classmethod
    def _departures_in_precedence_order(cls, value: list[Departure]) -> list[Departure]:
        return departures_in_precedence_order(value)


class DefinitePart(_PartBase):
    """One unambiguous answer was resolved for this part."""

    status: Literal["definite"] = Field(description="one answer was resolved")
    spec: Spec = Field(
        description=(
            "the resolved Spec answering this part. Several parts sharing one five-tuple "
            "shape carry the identical spec verbatim (the same spec_id); a part that "
            "shares the spec without minting it carries a part-scoped binding_folded "
            "warning"
        )
    )


class CandidatesPart(_PartBase):
    """This part's variable was genuinely ambiguous; two or more distinct specs compete."""

    status: Literal["candidates"] = Field(
        description="this part's variable was genuinely ambiguous"
    )
    candidates: CandidateSet = Field(description="2..N competing Specs")


class NoDataPart(_PartBase):
    """Nothing could be returned for this part."""

    status: Literal["no_data"] = Field(description="nothing to return")
    no_data: NoData = Field(description="the reason")


class Unsupported(StrictModel):
    """The request was valid and fully understood, but not supported.

    Fires for a text-door condition the request grammar or this deployment
    cannot honour: the unit binder recognises but cannot represent a unit
    phrase at all -- PPP terms, per-capita, a percent/share unit, or a scale
    word (million/billion/thousand) -- code unit_request_unsupported, or a
    recognised facet axis this deployment cannot bind (code
    facet_binding_unsupported). A unit phrase the request form COULD
    represent, but that this text failed to parse cleanly (two conflicting
    currencies, price bases, or base years in one ask; an incomplete "base
    year" cue with no year value), is a different outcome entirely: it lands
    no_data with reason unit_not_resolved, never here. Distinct from every
    refusal: the request was never malformed or unresolvable, so this is a
    200, not a problem-details body.
    """

    code: str = Field(description="the reason, a code from the platform registry")
    pointer: str = Field(
        description=(
            "a JSON Pointer into `definition` (not the request body) naming the "
            "binding at fault -- a text-door request has no structured body to "
            "point into"
        )
    )
    detail: str = Field(description="human-readable explanation")
    cause: Cause | None = Field(
        default=None,
        description=(
            "why this refusal fired, present only for code=recipe_not_declared "
            "(the structured door's same-named ProblemDetails extension); "
            "omitted on every other code"
        ),
    )
    definition: dict[str, object] = Field(
        description=(
            "the canonical definition as interpreted -- {schema_version, "
            "identity: {identity_version, shape, slots, entities, facets, "
            "requested_window}} -- without spec_id, since this was never a "
            "resolvable construction"
        )
    )


class UnsupportedPart(_PartBase):
    """This part's variable was valid and understood, but not supported."""

    status: Literal["unsupported"] = Field(description="valid, understood, not supported")
    unsupported: Unsupported = Field(description="the reason and definition")


Part = Annotated[
    DefinitePart | CandidatesPart | NoDataPart | UnsupportedPart,
    Field(discriminator="status"),
]


class Skill(StrictModel):
    """One skill's front matter on the wire, keyed by id in
    Resolution.skills. Mirrors skills_index.SkillFrontMatter's four
    members -- the search API's own wire model, not that dataclass itself,
    so it carries its own field descriptions the way every other response
    model does.
    """

    id: str = Field(description="the skill's stable id; the key this entry is filed under")
    name: str = Field(description="the skill's declared name")
    description: str = Field(description="the skill's declared description")
    content_version: str = Field(
        description=(
            "sha256: over the skill's manifest, SKILL.md then its references in path "
            "order, so any edit to any file in the bundle changes it"
        ),
    )


class _AnswerBase(BaseModel):
    """Fields every resolve response carries, whatever its kind."""

    schema_version: Literal[1] = Field(
        default=SCHEMA_VERSION,
        description="contract schema version; exactly 1 for v1. Required on every response.",
    )
    _validate_schema_version = field_validator("schema_version", mode="before")(
        _require_schema_version_int
    )
    query_echo: QueryEcho = Field(description="what the response was understood to be answering")
    diagnostics: Diagnostics = Field(description="run health, warnings, timing, build stamp")


class Scope(StrictModel):
    """What a catalogue or declined answer was scoped to. Titled apart from the
    datasets API's own `Scope`, which the composed app's OpenAPI also publishes."""

    model_config = ConfigDict(extra="forbid", title="AnswerScope")

    place: GraphRef | None = Field(
        default=None,
        description=(
            "the country the question named, when the engine confirmed it as a "
            "country; null when the question named no place or named one that is "
            "not a confirmed country"
        ),
    )


class Resolution(_AnswerBase):
    """A resolve response that resolved the question into data: one part per
    variable asked, in query order, each a complete outcome carrying its own
    status and payload.

    A single-variable query is one part, so `status` reads that one part's
    status directly. A query in which extraction found no variable is also
    one part, whose `variable_text` is the whole query text.

    `status` below is a computed field that rides in every serialised
    response body without being a settable input, so this class stays a
    plain `BaseModel`, which allows unknown fields. `StrictModel` forbids them,
    which would make `Resolution.model_validate` reject its own `model_dump()`
    output, breaking every round-trip a client or this codebase does on a
    fetched response body. A posted `status` is ignored and recomputed.

    Usage::

        resp = RESOLVE_RESPONSE_ADAPTER.validate_python(obj)
        if resp.kind == "resolution":
            resp.status          # "definite" | "candidates" | "no_data" | "unsupported" | "mixed"
            resp.parts[0].status  # this part's own status
    """

    kind: Literal["resolution"] = Field(
        default="resolution",
        description="discriminator for a response that resolved the question into data",
    )
    parts: Annotated[list[Part], Field(min_length=1)] = Field(
        description="one entry per variable asked, in query order"
    )
    rendered_sentence: str | None = Field(
        default=None,
        description=(
            "the envelope-level confirmation sentence describing the whole query, "
            "composed across every part's own sentence in query order; present under "
            "the same response_options.include_sentence condition each part's own "
            "rendered_sentence observes, null or omitted otherwise. When every "
            "contributing part is definite, their headings join into one sentence. A "
            "qualifier axis (where, when, or source, each a wire value of axis, "
            "plus origin, identified by role=='origin' instead since SentenceAxis "
            "has no 'origin' member) whose bound value agrees across every "
            "contributing part is stated once after the joined phrases, a possessive "
            "lead every part shares opens the sentence once, and the byline follows "
            "once. A qualifier that differs stays inline in each part's own phrase. "
            "The response falls back to one sentence per part, each part's own text "
            "left exactly as that part rendered it, when a part is still resolving, "
            "when the parts' bylines differ, or when a differing qualifier's "
            "connective opens with a comma (a period range). A part still resolving "
            "contributes up to three tokens of its own: an optional prefix text "
            "token, its own variable_text as an editable slot, and a text token "
            "stating the outcome. A lone part that is not definite composes to "
            "exactly that part's own sentence. After the headings and bylines, each "
            "part's departures follow in part order as segment='departure' tokens "
            "carrying that part's part_index. This is the one sentence that writes "
            "out a definite part's departures."
        ),
    )
    sentence_plan: list[SentenceToken] | None = Field(
        default=None,
        description=(
            "the ordered token form of rendered_sentence, present under the same "
            "condition; null or omitted otherwise. Composed the same way "
            "rendered_sentence is, from every part's own sentence_plan -- see "
            "SentenceToken.part_index for how a client recovers which parts[i] each "
            "variable phrase and attached qualifier belongs to, and which tokens are "
            "shared skeleton (an inter-phrase connective, a hoisted shared qualifier, "
            "the terminal '.', or the shared ' ' the per-part fallback puts between "
            "two sentences) rather than any "
            "one part's own."
        ),
    )
    skills: dict[str, Skill] | None = Field(
        default=None,
        description=(
            "every skill named by any Spec.resolved.skills across this response's "
            "parts, keyed by id -- each id any Spec carries is a key here, and each "
            "key's front matter is carried once regardless of how many Specs name "
            "it. Present under response_options.include_skills; null when the "
            "option is off, {} when it is on and no Spec matched anything."
        ),
    )

    @computed_field(  # type: ignore[prop-decorator]
        description=(
            "the parts' shared status when every part agrees, or 'mixed' when two or "
            "more parts differ; derived from parts, never independently settable"
        )
    )
    @property
    def status(self) -> Literal[StatusLiteral, "mixed"]:
        statuses = {part.status for part in self.parts}
        return next(iter(statuses)) if len(statuses) == 1 else "mixed"


class CatalogueAnswer(_AnswerBase):
    """A resolve response to a question that asks what DFO holds and names
    neither an indicator nor a subject, such as "what data do you have?". It
    carries the country the question named, when the resolver confirms one."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["catalogue"] = Field(
        default="catalogue",
        description="discriminator for a response to a question about what DFO holds",
    )
    scope: Scope = Field(description="the place the question named, when it named a country")


class DeclinedAnswer(_AnswerBase):
    """A resolve response to a question search does not answer."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["declined"] = Field(
        default="declined",
        description="discriminator for a response to a question search does not answer",
    )
    reason: DeclinedReason = Field(
        description=(
            "why search declined: 'out_of_scope' when no part of the question "
            "asks for a published statistic about a place, a population or a "
            "financial flow; 'requires_analysis' when the subject is in scope but "
            "the question asks for an argument and names no data to show"
        )
    )
    scope: Scope = Field(description="the place the question named, when it named a country")


ResolveResponse = Annotated[
    Resolution | CatalogueAnswer | DeclinedAnswer,
    Field(discriminator="kind"),
]
"""One resolve response, discriminated on `kind`: a `Resolution` carries parts
and a computed `status`; a `CatalogueAnswer` and a `DeclinedAnswer` carry a
scope and no parts."""

RESOLVE_RESPONSE_ADAPTER: TypeAdapter[ResolveResponse] = TypeAdapter(ResolveResponse)
"""Validates and serialises a `ResolveResponse` and publishes its JSON schema."""
