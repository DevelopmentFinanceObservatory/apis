from __future__ import annotations

import unicodedata
from typing import Annotated, Literal

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from .base import StrictModel
from .entity import EntityGroup
from .identity import BaseIdentity
from .vocab import (
    SCHEMA_VERSION,
    EntityRoleKey,
    NoDataReason,
    OperationAxis,
    OperationVerb,
    Ordering,
    _require_schema_version_int,
)

# ---------------------------------------------------------------------------
# Request root: three input arms, discriminated on kind
# ---------------------------------------------------------------------------


class RawTextInput(StrictModel):
    """Natural-language query input. The engine extracts, resolves, and grounds it."""

    kind: Literal["raw_text"] = Field(
        default="raw_text", description="discriminator for natural-language query input"
    )
    query: str = Field(description="the user's natural-language request")


# Parsed model caps mirror search.engine.config but remain local literals so
# importing search.models does not require the optional engine package. A parity
# test checks these static values against config.py.
_PARSED_MAX_PARTS = 6
_PARSED_MAX_PART_ENTITIES = 16
_PARSED_MAX_VARIABLE_TEXT_CHARS = 200
_PARSED_MAX_ENTITY_TEXT_CHARS = 128
_PARSED_MAX_ROLE_TEXT_CHARS = 64
_PARSED_MAX_TIME_TEXT_CHARS = 100
_PARSED_MAX_UNIT_TEXT_CHARS = 100
_PARSED_MAX_SOURCE_TEXT_CHARS = 200
_PARSED_MAX_RAW_QUERY_CHARS = 2000
# At least extract's MAX_OPERATIONS, the most one query can carry, so every part
# the engine echoes posts back unrefused. A test pins the relation.
_PARSED_MAX_PART_OPERATIONS = 8
# Equal to config's max_dropped_qualifier_length, the length a dropped qualifier keeps;
# extract discards an operation whose phrase is longer. A test pins the relation.
_PARSED_MAX_OPERATION_PHRASE_CHARS = 200


# The latest year a forecast's `target_year` names. Extract reads a target year from
# query text under the same bound, so every operation it echoes posts back unrefused.
MAX_FORECAST_TARGET_YEAR = 2200

# Control characters (newlines among them) and the Unicode line and paragraph
# separators, U+2028 and U+2029, which `str.splitlines` also breaks on.
_LINE_BREAKING_CATEGORIES = frozenset({"Cc", "Zl", "Zp"})


class Operation(StrictModel):
    """One computation the question asks for over a part's data, which search
    does not perform: the part answers with the data the operation needs, and
    the operation rides on the part for a client to perform or hand on."""

    verb: OperationVerb = Field(
        description=(
            "what the question asks to compute: rank, aggregate, growth, share_of, "
            "explain, forecast, or report"
        )
    )
    phrase: str = Field(
        min_length=1,
        max_length=_PARSED_MAX_OPERATION_PHRASE_CHARS,
        description=(
            "the query wording that asks for the operation, e.g. 'top 5 donors'; one line "
            "with no control characters or line separators, and not blank"
        ),
    )
    over: OperationAxis | None = Field(
        default=None,
        description=(
            "the axis the operation runs across (place, origin, destination, "
            "category, or year); null when the verb names none, as explain does. "
            "The part leaves this axis open, so its data covers every value the "
            "operation needs"
        ),
    )
    target_year: int | None = Field(
        default=None,
        ge=1000,
        le=MAX_FORECAST_TARGET_YEAR,
        description=(
            "the year a forecast projects to, e.g. 2050 for 'forecast aid to Ethiopia to "
            f"2050'; set only on a forecast, a four-digit year up to {MAX_FORECAST_TARGET_YEAR}. "
            "The raw_text door sets it only to a year after the current one, and leaves "
            "it null when the question names none; a posted value is taken as given "
            "within the bound, whatever today's date. It is kept apart from the part's "
            "time_text, the years whose data the part fetches, and does not affect "
            "resolution"
        ),
    )

    @model_validator(mode="after")
    def _target_year_only_on_a_forecast(self) -> Operation:
        if self.target_year is not None and self.verb != "forecast":
            raise ValueError("target_year is set only on a forecast")
        return self

    @field_validator("phrase")
    @classmethod
    def _phrase_is_one_line(cls, phrase: str) -> str:
        """A posted phrase reaches the bind prompt as one list line, so a newline, a
        Unicode line or paragraph separator, or another control character would let
        caller text start a prompt line of its own, and a blank phrase renders as an
        empty one. A no-break space passes."""
        if any(unicodedata.category(char) in _LINE_BREAKING_CATEGORIES for char in phrase):
            raise ValueError(
                "phrase must not contain control characters, newlines or line separators"
            )
        if not phrase.strip():
            raise ValueError("phrase must not be blank")
        return phrase


_OPERATIONS_DESCRIPTION = (
    "the operations the question asks for over this part's data, in query order; "
    "the engine echoes them onto the part it resolves. Empty when the question "
    "asks for none"
)


class ParsedEntity(StrictModel):
    """One entity mentioned in a parsed part, with an optional caller-supplied role hint."""

    text: str = Field(
        min_length=1,
        max_length=_PARSED_MAX_ENTITY_TEXT_CHARS,
        description="the entity's text as the caller identified it",
    )
    role_text: str | None = Field(
        default=None,
        min_length=1,
        max_length=_PARSED_MAX_ROLE_TEXT_CHARS,
        description=(
            "the caller's free-form role hint for this entity, when available; "
            "recognised words (case-insensitive) are 'from'/'to' and synonyms such "
            "as donor/provider/funder/source/creditor/lender/exporter for the "
            "'from' side, recipient/beneficiary/debtor/borrower/importer for the "
            "'to' side -- unrecognised text warns role_text_not_recognised and "
            "the entity is treated as undirected"
        ),
    )
    dcid: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "an optional caller-pinned dcid for this entity, naming exactly which "
            "graph node `text` refers to -- the door a client with a candidate's "
            "dcid in hand (e.g. from NoData.ambiguous_entities) posts through, "
            "instead of relying on name resolution to disambiguate "
            "`text` on its own. When present, this entity's name resolution is "
            "skipped: the pinned dcid seeds the entry's pre_resolved map directly, "
            "still confirmed against the graph. An unconfirmable dcid (absent from "
            "the graph, or present but type-less) refuses invalid_role_ref (422). "
            "`text` stays required and is not re-derived from the dcid; it is still "
            "what a rendered sentence and any further restatement echo back."
        ),
    )
    members: bool | None = Field(
        default=None,
        description=(
            "true asks for the members of the set `text` names, each as its own series: "
            "the countries of a continent or UN subregion, or the members of a group. It "
            "has no effect on a single place. Text whose wording names members, such as "
            "'EU member states', reads as true without the flag. With a pinned `dcid` it "
            "is honoured for a Continent or UNGeoRegion node, and refuses request_invalid "
            "for a group's own node, whose members a pin cannot supply. One part names a "
            "set once: the same `text` posted with both values refuses request_invalid"
        ),
    )

    @field_validator("members")
    @classmethod
    def _false_is_unset(cls, members: bool | None) -> bool | None:
        """An entity asks for members or does not: a posted false is the field left out."""
        return members or None


class ParsedPart(StrictModel):
    """One variable ask within a parsed request, with its own entities, time, unit, and source."""

    variable_text: str = Field(
        min_length=1,
        max_length=_PARSED_MAX_VARIABLE_TEXT_CHARS,
        description="the variable's text as the caller identified it",
    )
    entities: list[ParsedEntity] | None = Field(
        default=None,
        max_length=_PARSED_MAX_PART_ENTITIES,
        description=(
            "entities associated with this part's variable, when available; null, "
            "omitted, or an empty list all mean no entity was named for this part, "
            "resolving the variable globally/entity-less -- the same meaning a "
            "raw_text query with no extracted place carries"
        ),
    )
    time_text: str | None = Field(
        default=None,
        min_length=1,
        max_length=_PARSED_MAX_TIME_TEXT_CHARS,
        description=(
            "this part's time expression, when available; recognised forms include a "
            "bare four-digit year, 'YYYY to YYYY' or 'YYYY-YYYY', 'from YYYY to YYYY', "
            "'between YYYY and YYYY', a latest end as in 'between YYYY and the latest' "
            "or 'YYYY to latest', 'since YYYY', 'latest', 'latest year available', "
            "'most recent', and 'last N years'. Anything else resolves dateless with a "
            "time_text_not_parsed warning, whose message lists every form"
        ),
    )
    unit_text: str | None = Field(
        default=None,
        min_length=1,
        max_length=_PARSED_MAX_UNIT_TEXT_CHARS,
        description=(
            "this part's requested unit or measurement basis, when available; "
            "recognised phrasing names a currency (ISO 4217 code or curated name/"
            "alias), a price basis (current or constant prices), and/or a base "
            "year (e.g. '2015 US dollars', 'constant prices, 2015 base year'). A "
            "phrase this text failed to parse cleanly resolves no_data with "
            "unit_not_resolved; a phrase the request grammar cannot express at "
            "all resolves unsupported with unit_request_unsupported. A phrase "
            "the grammar CAN express is checked against this deployment's own "
            "unit data: a currency/prices/base_year combination no unit this "
            "deployment carries expresses at all resolves unsupported with "
            "facet_binding_unsupported; one this deployment can express but that "
            "matches no candidate observation resolves no_data with "
            "facet_not_available; a match resolves definite, applying the "
            "completed request to the resolved spec's identity and carrying "
            "request_basis on resolved.facets.unit"
        ),
    )
    source_text: str | None = Field(
        default=None,
        min_length=1,
        max_length=_PARSED_MAX_SOURCE_TEXT_CHARS,
        description=(
            "this part's requested data source as free text, when available; free text "
            "never binds a source, so it always resolves unsupported with code "
            "facet_binding_unsupported. Name a declared dataset in `provenance` instead"
        ),
    )
    # A dataset dcid is a source; it shares the source text's length cap.
    provenance: str | None = Field(
        default=None,
        min_length=1,
        max_length=_PARSED_MAX_SOURCE_TEXT_CHARS,
        description=(
            "the dcid of the declared dataset this part answers from, when the caller "
            "names one; the served request schema lists the declared datasets under this "
            "field, each with its label, description and opt-in flag. A dcid that is not "
            "declared refuses request_invalid with a pointer to this field. Absent, the "
            "part answers from the dataset the code index elects for its identity"
        ),
    )
    operations: list[Operation] = Field(
        default_factory=list,
        max_length=_PARSED_MAX_PART_OPERATIONS,
        description=_OPERATIONS_DESCRIPTION,
    )
    open_roles: (
        list[Annotated[str, Field(min_length=1, max_length=_PARSED_MAX_ROLE_TEXT_CHARS)]] | None
    ) = Field(
        default=None,
        max_length=2,
        description=(
            "the roles this part asks for member by member, at most one word per flow "
            "axis, e.g. ['creditor'] for 'debt service paid to each creditor'. Each is "
            "a role_text word: creditor, donor, recipient, debtor, origin, destination, "
            "or a synonym. Where the answer's family declares the role, the answer "
            "holds one series per member of it and its identity states the axis in "
            "open_roles; a role named in neither entities nor open_roles reads as the "
            "total over that role. An unrecognised word, or a role one of this part's "
            "entities binds through role_text, refuses request_invalid. An operation "
            "whose over is origin or destination opens that axis too"
        ),
    )


class ParsedInput(StrictModel):
    """Pre-parsed input: one or more variable parts supplied directly by the caller,
    skipping natural-language extraction.
    """

    kind: Literal["parsed"] = Field(
        default="parsed", description="discriminator for pre-parsed input"
    )
    parts: list[ParsedPart] = Field(
        min_length=1,
        max_length=_PARSED_MAX_PARTS,
        description=(
            "one entry per variable ask; conjunctions pre-split by the caller. Parts "
            "that are byte-identical across every field collapse to one before "
            "resolution, so the response's own parts list can hold fewer entries "
            "than this list."
        ),
    )
    raw_query: str | None = Field(
        default=None,
        min_length=1,
        max_length=_PARSED_MAX_RAW_QUERY_CHARS,
        description="original query text, for echo only; does not affect resolution",
    )


_ENTITY_GROUPS_DESCRIPTION = (
    "the named group each role's entities were asked for as, keyed by entity role "
    "key. Post a prior Spec's resolved.entity_groups beside any identity derived "
    "from that Spec. A group is kept when it posts a dcid and the graph's members "
    "of that node are exactly the identity's entities on the role; any other group, "
    "including one on a role the identity binds no entity on, is dropped"
)

ResubmitEntityGroups = Annotated[
    dict[EntityRoleKey, EntityGroup] | None,
    Field(default=None, description=_ENTITY_GROUPS_DESCRIPTION),
]
"""The named groups posted beside one resubmitted identity: request context for its
roles, outside the identity and its spec_id."""


class ResubmitIdentityPayload(StrictModel):
    """One identity to re-materialise.

    Not a request input of its own: the transport that carries it is
    `ResubmitIdentityEntry` (inside a `DatasetResubmitInput`), and this model holds
    that entry's identity and its entity groups for the engine to resolve.

    `identity` is the complete definition. A no-op resubmit posts the
    `Spec.definition.identity` it received byte-for-byte. The engine derives the
    identity's StatVar members itself: from the packaged code index when it decides
    the identity, else from the graph walk the Datasets mint runs.

    `entity_groups` names the group a role's entities were asked for as. The engine
    keeps a group the graph confirms for the identity's entities on that role
    (`resubmit_request.resubmit_named_groups`).
    """

    identity: BaseIdentity = Field(
        description=(
            "the complete identity. Post a prior Spec.definition.identity "
            "byte-for-byte for a no-op resubmit; change nested members, including "
            "entities.origin or entities.destination, to edit that definition"
        )
    )
    entity_groups: ResubmitEntityGroups


# ---------------------------------------------------------------------------
# dataset_resubmit carries one ordered entry per response part through the shared
# combiner. Every entry supplies variable_text for sentence composition; parsed
# entries carry it inside ParsedPart.
# ---------------------------------------------------------------------------


class ResubmitIdentityEntry(StrictModel):
    """One part re-resolved from a single identity -- the payload
    `ResubmitIdentityPayload` carries, plus the text echo and operations every
    dataset_resubmit entry needs.
    """

    entry_kind: Literal["identity"] = Field(
        default="identity",
        description="discriminator for a single-identity dataset_resubmit entry",
    )
    variable_text: str = Field(
        min_length=1,
        max_length=_PARSED_MAX_VARIABLE_TEXT_CHARS,
        description="this part's variable text, echoed back onto the rebuilt part",
    )
    identity: BaseIdentity = Field(
        description=(
            "the complete identity. Post a prior Spec.definition.identity "
            "byte-for-byte for a no-op resubmit; change nested members, including "
            "entities.origin or entities.destination, to edit that definition"
        )
    )
    entity_groups: ResubmitEntityGroups
    operations: list[Operation] = Field(
        default_factory=list,
        max_length=_PARSED_MAX_PART_OPERATIONS,
        description=_OPERATIONS_DESCRIPTION,
    )


class ResubmitCandidate(StrictModel):
    """One candidate identity within a `candidates` dataset_resubmit entry. The
    engine derives its StatVar members as it does for an identity entry."""

    identity: BaseIdentity = Field(description="this candidate's complete identity")
    entity_groups: ResubmitEntityGroups


_RESUBMIT_MAX_CANDIDATES = 50
"""Structural ceiling of 50 candidates. The deployment-specific, environment-configurable
``SEARCH_MAX_CANDIDATES`` limit is enforced by the engine.
"""


class ResubmitCandidatesEntry(StrictModel):
    """One part re-materialised from a posted set of competing candidate
    identities, through the promote path, in posted order, always. `ordering`
    names the display rule this set already used, restated rather than
    recomputed: neither ordering rule is reproducible from posted state alone
    (`intent_rank` is one LLM call's output; `broadest_first`'s loose-last
    band comes from candidate_scores that do not round-trip) -- see
    `Ordering` and `outcomes.CandidateSet`. This entry carries no
    `max_candidates` field: `CandidateSet.max_candidates` on the response is
    not posted back and is always server-computed, never client-supplied.
    """

    entry_kind: Literal["candidates"] = Field(
        default="candidates",
        description="discriminator for a candidate-set dataset_resubmit entry",
    )
    variable_text: str = Field(
        min_length=1,
        max_length=_PARSED_MAX_VARIABLE_TEXT_CHARS,
        description="this part's variable text, echoed back onto the rebuilt part",
    )
    ordering: Ordering = Field(
        description=(
            "how `candidates` below was ordered when this set was first produced; "
            "echoed back verbatim onto the rebuilt candidates part, never recomputed"
        )
    )
    truncated: bool = Field(
        description=(
            "echoed back verbatim onto the rebuilt candidates part -- see "
            "CandidateSet.truncated for what this flags"
        )
    )
    candidates: list[ResubmitCandidate] = Field(
        min_length=2,
        max_length=_RESUBMIT_MAX_CANDIDATES,
        description=(
            "the competing candidates to re-materialise, in posted order, always -- "
            "the rebuilt CandidateSet.specs preserves this exact order. Refused "
            "request_invalid, at the engine rather than here, when the posted "
            "count exceeds this deployment's own ceiling -- readable in advance as "
            "caps.max_candidates from GET /api/search/capabilities"
        ),
    )
    operations: list[Operation] = Field(
        default_factory=list,
        max_length=_PARSED_MAX_PART_OPERATIONS,
        description=_OPERATIONS_DESCRIPTION,
    )


class ResubmitPlaceholderEntry(StrictModel):
    """One part restated as a stub from a no_data/unsupported outcome the
    client already holds -- zero graph round-trips, zero LLM calls. Carried
    forward unresolved so its sibling parts' composed sentence and fold can
    still be computed across the whole dataset; the engine stamps a
    part-scoped `part_restated` info warning on the rebuilt part.
    """

    entry_kind: Literal["placeholder"] = Field(
        default="placeholder", description="discriminator for a stub dataset_resubmit entry"
    )
    variable_text: str = Field(
        min_length=1,
        max_length=_PARSED_MAX_VARIABLE_TEXT_CHARS,
        description="this part's variable text, echoed back onto the rebuilt part",
    )
    status: Literal["no_data", "unsupported"] = Field(
        description="which stub outcome to restate for this part"
    )
    reason: NoDataReason | None = Field(
        default=None,
        description=(
            "the NoData.reason to restate onto the rebuilt part; required when "
            "status is no_data, refused when status is unsupported"
        ),
    )
    code: str | None = Field(
        default=None,
        description=(
            "the reason, a code from the platform registry, to restate onto the "
            "rebuilt part's Unsupported.code; required when status is unsupported, "
            "refused when status is no_data"
        ),
    )
    operations: list[Operation] = Field(
        default_factory=list,
        max_length=_PARSED_MAX_PART_OPERATIONS,
        description=_OPERATIONS_DESCRIPTION,
    )

    @model_validator(mode="after")
    def _check_status_reason_code_pairing(self) -> ResubmitPlaceholderEntry:
        if self.status == "no_data":
            if self.reason is None:
                raise ValueError("reason is required when status is no_data")
            if self.code is not None:
                raise ValueError("code must not be posted when status is no_data")
        else:
            if self.code is None:
                raise ValueError("code is required when status is unsupported")
            if self.reason is not None:
                raise ValueError("reason must not be posted when status is unsupported")
        return self


class ResubmitParsedEntry(StrictModel):
    """One part genuinely re-resolved through the parsed door's own per-part
    path, from a client-edited restatement of a sibling's own resolution
    inputs (`Part.restatement`) -- re-parsed, where `placeholder` only
    restates a stub. `part` already carries a required `variable_text` and
    its own `operations`, so this entry needs no separate top-level echo
    fields the way the other three kinds do. Costs one LLM bind per edit, the
    door's one entry kind that touches the LLM client at all.
    """

    entry_kind: Literal["parsed"] = Field(
        default="parsed",
        description="discriminator for a re-parsed dataset_resubmit entry",
    )
    part: ParsedPart = Field(
        description=(
            "this part's resolution inputs -- normally a prior response part's "
            "own Part.restatement, posted back verbatim or edited, and resolved "
            "through the same per-part machinery the parsed door uses"
        )
    )


ResubmitEntry = Annotated[
    ResubmitIdentityEntry
    | ResubmitCandidatesEntry
    | ResubmitPlaceholderEntry
    | ResubmitParsedEntry,
    Field(discriminator="entry_kind"),
]


class DatasetResubmitInput(StrictModel):
    """Re-resolve every part of a multi-variable result in one request, so an
    edit to one part cannot leave the other parts' composed sentence or fold
    stale (see `envelope.make_response` / `conjoin.combine_regions`, the
    combiner every entry kind here is routed through).

    `entries` supplies one entry per response part, in response part order;
    each entry edits against its own base, never a dataset-level one. There
    is no spec registry, so every entry carries its full identity and this
    door reads no server-side state.
    """

    kind: Literal["dataset_resubmit"] = Field(
        default="dataset_resubmit",
        description="discriminator for dataset-level re-resolution across every part",
    )
    entries: list[ResubmitEntry] = Field(
        min_length=1,
        max_length=_PARSED_MAX_PARTS,
        description=(
            "one entry per response part, in response part order; a single-entry "
            "request whose entry is an identity re-resolves exactly that one part"
        ),
    )
    raw_query: str | None = Field(
        default=None,
        min_length=1,
        max_length=_PARSED_MAX_RAW_QUERY_CHARS,
        description="original query text, for echo only; does not affect resolution",
    )


ResolveInput = Annotated[
    RawTextInput | ParsedInput | DatasetResubmitInput,
    Field(discriminator="kind"),
]


class ResolveOptions(StrictModel):
    """Optional resolution parameters the caller may supply."""

    include_option_coverage: bool | None = Field(
        default=None,
        description=(
            "when true, the engine probes each top-level interpretation and candidate "
            "Spec's slot_options for observation coverage (e.g. 'Kenya — 205 obs') and "
            "populates SlotOption.coverage; false/absent (the default) never fires the extra "
            "probes, so this is opt-in latency, not a hidden cost on the hot path. "
            "Delivery differs by transport, for every input kind: over SSE "
            "(Accept: text/event-stream), coverage arrives "
            "via trailing slot_options/entity_options events emitted after the "
            "response event, and the response event's own Spec.resolved.slot_options / "
            "entity_options stay coverage=null; on the buffered JSON body, coverage "
            "is embedded directly in the returned Spec. Spec.resolved.facet_options is "
            "computed whether or not this is set; over SSE with this set, it arrives "
            "in trailing facet_options events instead of the response event."
        ),
    )
    include_observations: bool | None = Field(
        default=None,
        description=(
            "when true (the default is false -- absent also means false), the engine populates "
            "Spec.resolved.observations with the resolved answer's observation values: the "
            "full series when small enough, otherwise a bounded latest-points preview. "
            "This per-spec fetch runs only after the engine commits to a definite "
            "spec, including a promoted candidate; a multi-candidate response keeps "
            "per-candidate metadata and coverage without fetching observations. Delivery mirrors "
            "include_option_coverage's transport split: over SSE, observations arrive "
            "via a trailing observations event emitted after the response event, and "
            "the response event's own Spec.resolved.observations stays null or omitted; "
            "on the buffered JSON body, observations are embedded directly in the "
            "returned Spec."
        ),
    )
    include_skills: bool | None = Field(
        default=None,
        description=(
            "when true (the default is false -- absent also means false), the engine "
            "matches every resolved Spec's bound graph nodes (its shape's structural "
            "anchors and the dcids its bound constraint slots name) and its selected "
            "provenance (facets.provenance.ref, when one provenance serves every series) "
            "against the shared skills index and populates Spec.resolved.skills with the "
            "matching skill ids: skills matched through a graph node first, "
            "nearest-attachment-first, then skills matched only through the provenance, "
            "by id; [] when nothing matched. The "
            "response's own skills map then carries every named skill's front matter "
            "once, keyed by id. When the shared skills index has not yet built a "
            "first snapshot, the response carries the skills_index_unavailable "
            "diagnostics warning instead and no Spec gets a populated skills field; "
            "when a snapshot is available but its own build may have left out a "
            "published skill or served one excluded or understated (a read stopped "
            "at a budget or bound, the build's content budget ran out before a "
            "skill's turn, or a build step failed), the response instead "
            "carries the skills_index_incomplete "
            "diagnostics warning alongside whatever matches it did serve; a "
            "part with no resolved spec (no_data, unsupported) carries the "
            "part-scoped skills_not_matchable warning, since it binds no graph nodes "
            "to match against."
        ),
    )


class ResolveRequest(StrictModel):
    """The single request type for the Search resolve endpoint."""

    schema_version: Literal[1] = Field(
        default=SCHEMA_VERSION,
        description="contract version the caller targets",
    )
    _validate_schema_version = field_validator("schema_version", mode="before")(
        _require_schema_version_int
    )
    input: ResolveInput = Field(description="tagged union; kind discriminates")
    response_options: ResolveOptions | None = Field(
        default=None, description="optional resolution options"
    )
