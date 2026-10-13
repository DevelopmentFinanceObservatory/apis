from __future__ import annotations

from typing import Annotated

from pydantic import (
    Field,
    model_validator,
)

from dfo_platform import Cause

from .base import StrictModel
from .grammar import GraphRef
from .identity import IdentityCurrency
from .request import ParsedInput
from .spec import NearestRealSpec, Spec
from .trace import ResolvedQualifier
from .vocab import (
    _UNRESOLVED_NO_DATA_REASONS,
    NoDataCategory,
    NoDataGate,
    NoDataReason,
    Ordering,
    SourceRefKind,
)

# Refusals whose flagged_match records the elected spec without a gate: each refuses before,
# or regardless of, an availability verdict.
_UNGATED_FLAGGED_MATCH_REASONS: frozenset[NoDataReason] = frozenset(
    {"role_incompatible", "qualifier_not_bound", "denominator_not_available"}
)

# ---------------------------------------------------------------------------
# CandidateSet and NoData
# ---------------------------------------------------------------------------


class CandidateSet(StrictModel):
    """A set of 2..N competing Specs, presented in the order named by ``ordering``.

    ``ordering="intent_rank"``: the near-neighbor margin gate downgraded a
    would-be "definite" answer to this candidates response -- a genuinely close
    call the model's own elected shape didn't flag. Order is the engine's own
    estimate of intent match, best first: the elected shape, then the
    near-winning reading that triggered the downgrade, then any remaining
    also_plausible/cascade alternates in bind's own ranked order. Which of
    those readings is present is a separate question from the order they sit
    in -- a reading carrying no data for the requested entity is dropped, the
    elected shape and the near-winning reading included, and the pool-widening
    cascade's next-best distinct reading takes the vacated slot. So
    ``intent_rank`` names the ranking rule, not a fixed cast of positions.

    ``ordering="broadest_first"``: this candidates response was ambiguous from
    the start (bind's own also_plausible; the gate never ran). Order is
    descending breadth (more candidate StatVars grouped under the shape),
    with spec_id lexicographic as the deterministic tie-break.

    Both ordering rules apply within relevance bands. Candidates that pass the
    retrieval/near-neighbor quality gate come first; loose candidates remain
    visible afterward, preserving their base order within that band.

    Each candidate carries its own heading and byline inside its own
    ``resolved.rendered_sentence``/``resolved.sentence_plan``. There is no separate parallel sentence array, and a
    candidate's Spec is a returned Spec like any other. The part's own
    sentence opens with how many readings there are and states, for each
    shown candidate, the source and span that set it apart.
    """

    ordering: Ordering = Field(
        description=(
            "how `specs` below is ordered within relevance bands -- see the two orderings "
            "described above"
        )
    )
    max_candidates: int = Field(description="the maximum number of candidates applied")
    specs: Annotated[list[Spec], Field(min_length=2)] = Field(
        description=(
            "2 <= len <= max_candidates; full Specs, ordered per `ordering` above within "
            "relevance bands"
        )
    )
    truncated: bool = Field(
        description=(
            "false when every eligible, semantically distinct candidate shape found "
            "for this query is present in specs; true when the engine's own candidate "
            "budget dropped an eligible shape before it could be considered, or "
            "max_candidates itself cut the ordered list short"
        )
    )


class AmbiguousEntityCandidate(StrictModel):
    """One graph node an ambiguous query wording could name -- one entry of
    an AmbiguousEntityGroup's own `candidates`. Mirrors engine.region.result.
    AmbiguousEntityCandidate field-for-field, with `label` in place of that
    dataclass's `name` (this contract's own vocabulary for a display
    string, matching ResolvedEntity.label)."""

    ref: str = Field(description="the candidate's own dcid")
    label: str = Field(description="the candidate's display name")
    kind: str = Field(
        description=(
            "the candidate's entity kind: country, continent, subregion, "
            "continental_union, organisation, aggregate, place_other, or other"
        )
    )


class AmbiguousEntityGroup(StrictModel):
    """One query wording Ambiguous resolved to more than one graph
    candidate, paired with the candidates it could name -- one entry of
    NoData.ambiguous_entities."""

    text: str = Field(
        description=(
            "the ambiguous query wording; equals some restatement.entities[i].text on the same part"
        )
    )
    candidates: list[AmbiguousEntityCandidate] = Field(
        description=(
            "the graph candidates this wording could name -- 2+ from a real "
            "Ambiguous resolution, by construction of the resolver's own "
            "Ambiguous outcome (2+ candidates graph-confirm and none can be "
            "preferred); not constrained to a minimum here since region."
            "AmbiguousEntity, this field's own engine-side source, carries no "
            "such constraint of its own"
        )
    )


class AmbiguousSourceCandidate(StrictModel):
    """One reading of an ambiguous source wording -- one entry of an
    AmbiguousSourceGroup's own `candidates`."""

    ref: str = Field(
        description="the source's reference, which a client posts back as a SourceMention's ref"
    )
    label: str = Field(description="the source's display name in the origin catalogue")
    kind: SourceRefKind = Field(
        description="what the reference names: a provenance, a catalogue Dataset, or a publishing Source"
    )


class AmbiguousSourceGroup(StrictModel):
    """One source wording that names two or more unrelated admitted sources, paired with
    the sources it could name -- one entry of NoData.ambiguous_sources."""

    text: str = Field(
        description=(
            "the ambiguous wording; equals some restatement.sources[i].text on the part, or "
            "its restatement.source_code for source_code_ambiguous"
        )
    )
    candidates: list[AmbiguousSourceCandidate] = Field(
        description=(
            "the admitted sources this wording names, two or more: provenances first, "
            "then datasets and sources"
        )
    )


class UnitAlternative(StrictModel):
    """One currency a unit_not_resolved refusal offers in place of the unit
    phrase it could not resolve -- one entry of NoData.unit_alternatives."""

    currency: IdentityCurrency = Field(
        description=(
            "the value a caller resubmits as facets.unit.request.currency, as "
            "written: an ISO 4217 code for an ambiguous currency name, or a role "
            "shape for a role-relative currency phrase beside entities on two roles"
        )
    )
    label: str = Field(description="the currency's display name, or the role's reader-facing noun")


class NoDataDenominator(StrictModel):
    """The denominators a denominator_not_available refusal compared: the one the question
    named and the one its flagged match divides by -- NoData.denominator."""

    asked: GraphRef | None = Field(
        default=None,
        description=(
            "the denominator the question named, as the search read it from the bases "
            "the candidate series divide by; null when the question named none, or named "
            "one that no candidate series divides by"
        ),
    )
    found: GraphRef | None = Field(
        default=None,
        description=(
            "the denominator the flagged match divides by; null when it states none "
            "(a level such as an amount in US dollars, or a series whose name alone "
            "gives its base)"
        ),
    )


class NoData(StrictModel):
    """Outcome when no grounded result can be returned."""

    reason: NoDataReason = Field(
        description=(
            "why nothing could be returned, in contract resolution order. "
            '"entity_not_resolved" means a named entity could not be resolved to a '
            'graph node; "entity_ambiguous" means a named entity matched more than '
            "one graph candidate and none could be preferred over the others, "
            'distinct from "entity_not_resolved", where no node matched at all; '
            '"entity_role_not_admitted" means a named entity resolved to exactly '
            "one graph node, but no role any candidate shape declares admits that "
            "node's graph type -- the entity is real, just not the kind of thing "
            "this query's data is reported for; "
            '"entity_set_too_large" means every named entity resolved, but the places '
            "the request names, counting the members of each group or region among "
            "them and summed over the roles they sit on, exceed what the engine binds "
            "in one request -- distinct from every other entity_* reason, where one "
            "entity, not the size of the set, is what failed; "
            '"variable_not_resolved" means no StatVar matched the '
            "variable text (e.g. an absent posted stat_var_dcid or entity, or no "
            'measure match at all); "role_not_resolved" means the query names a '
            'role the engine could not resolve; "unit_not_resolved" means the unit '
            "binder could not classify a token in the requested unit phrase against "
            'its lexicons; "no_matching_slice" means the measure matched fine, but '
            "the requested combination has no corresponding data series -- the "
            "constructed/derived variable does not exist in the graph at all. "
            "Covers an identity resubmit refine edit with no confirmable sibling "
            "StatVar for the edited slice, and a dev-finance scheme x purpose x "
            "recipient combination that was never published as a series, on "
            'raw-text and resubmit alike; "binding_mismatch" means the variable '
            "resolved and graph-confirmed StatVars exist, but none satisfies the "
            "requested slot bindings, either because a resolved value on a bound "
            "property mismatched or because the StatVar's value on a bound "
            "property is present but could not be read (an unresolvable dcid); "
            'distinct from "no_matching_slice" (the combination was never '
            'published at all) and "variable_not_resolved" (no StatVar matched '
            "at all); "
            '"denominator_not_available" means the '
            "query asks for a normalised form (per person, per area, a rate or a share) "
            "or names a denominator, and the matched series is not in that form or "
            'divides by another base; "no_observations" means a StatVar WAS confirmed to exist but '
            "has no observation at the resolved entity (it exists, but not data "
            'here); "facet_not_available" means observations exist at the entity, '
            'none under the requested facet values; "role_incompatible" means the '
            "role is real, and the StatVar does not declare it; "
            '"role_signature_incompatible" means the resolved StatVars carry '
            "incompatible role signatures and no partition into candidates was "
            'possible; "source_code_not_held" means no variable held under the part\'s '
            "source_code by a source search serves can answer the part: no served source "
            "holds the code, the part's sources exclude every source that does, or the "
            "code's variables cannot answer the part as asked; "
            '"source_code_ambiguous" means two of the part\'s sources hold its source_code; '
            '"source_code_shares_variable" means the source_code\'s variables answer other '
            "codes of its source too, so search cannot serve the code alone; "
            '"qualifier_not_bound" means the variable resolved to a '
            "series family, but a narrowing term the request's text named "
            "bound no constraint slot: no code system in the packaged index "
            "carries it, or its registered rows landed in no binding"
        )
    )
    cause: Cause | None = Field(
        default=None,
        description=(
            "why this refusal fired, present only for reason=rate_coverage_gap "
            "(the structured door's same-named ProblemDetails extension); "
            "omitted on every other reason"
        ),
    )
    category: NoDataCategory = Field(
        default="absent",
        description=(
            "derived from reason, never independently settable: "
            '"entity_not_resolved", "entity_ambiguous", "entity_set_too_large", '
            '"entity_role_not_admitted", "variable_not_resolved", "role_not_resolved", '
            '"source_not_resolved", "source_ambiguous", "source_code_not_held", '
            '"source_code_ambiguous", "source_code_shares_variable", and '
            '"unit_not_resolved" are '
            '"unresolved" (the request could not be read into the graph); every other '
            'code is "absent" (the read request names a place holding nothing)'
        ),
    )
    message: str | None = Field(
        default=None,
        description=(
            "the plain-language sentence for reason, naming the query terms it concerns "
            "where the reason has them (the indicator, place, or category wording), so a "
            "client can show a person the reason beside its code. Every no_data part the "
            "engine returns sets it. "
            "Clients must not parse it"
        ),
    )
    nearest_real: list[NearestRealSpec] | None = Field(
        default=None,
        description=(
            "optional; up to two real, grounded specs found by relaxing one constraint "
            "at a time and re-confirming against the live graph. Each carries its own "
            "relaxation naming the definition part relaxed and the value(s) it dropped. "
            "Shaped by response_options and banded by confidence like any returned spec. "
            "Distinct from flagged_match (the elected variable itself) and alternate_match "
            "(a different variable). nearest_real varies the SAME variable's own "
            "constraints, never the variable."
        ),
    )
    gate: NoDataGate | None = Field(
        default=None,
        description=(
            "optional; the failed quality gate's code from the platform registry, one of "
            "the three registry gate codes (no_member_has_data_at_place, "
            "no_member_asserts_bound_properties, only_alternate_has_data). The single home "
            "for the code regardless of which near-match field carries the Spec: present "
            "exactly when flagged_match or alternate_match is present, except a "
            "role_incompatible refusal before or during materialise whose flagged_match "
            "records the confirmed elected identity without a gate (role incompatibility "
            "prevents an availability verdict), a qualifier_not_bound refusal, whose "
            "flagged_match records the graph-confirmed elected spec without a gate "
            "because the qualifier guard refuses before an availability verdict exists, "
            "or a denominator_not_available refusal, whose flagged_match records the "
            "elected spec the form or denominator check refused. "
            "Absent on every other no_data outcome. At most one of "
            "flagged_match and alternate_match is present at a time."
        ),
    )
    flagged_match: Spec | None = Field(
        default=None,
        description=(
            "optional; the elected variable's own Spec, present whenever gate fired and a "
            "groundable identity exists for the elected shape, or for a role_incompatible, "
            "qualifier_not_bound or denominator_not_available refusal that records its "
            "graph-confirmed identity without a gate: role incompatibility prevents an "
            "availability verdict, the qualifier guard refuses before one exists, and the "
            "denominator check refuses the series for its form, not its data. "
            "See gate for which check failed. "
            "available_at_place is False and confidence is low on "
            "gate=no_member_has_data_at_place; both stay at their not-assessed default "
            "(None) on gate=no_member_asserts_bound_properties, since there is no resolved "
            "place to be unavailable at. Absent on gate=only_alternate_has_data, where no "
            "groundable Spec for the elected shape exists; alternate_match carries the "
            "substitute the engine found instead. Distinct from nearest_real, up to two "
            "constraint-RELAXED specs that DO have data nearby. flagged_match is the "
            "elected variable itself, unavailable as the request asked for it."
        ),
    )
    alternate_match: Spec | None = Field(
        default=None,
        description=(
            "optional; a categorically different, data-bearing variable the request did "
            "not elect, present exactly when gate == 'only_alternate_has_data'. "
            "elected_count still counts the variable the request actually asked for; "
            "alternate_match is the one substitute the engine found nearby with "
            "observations of its own. Distinct from flagged_match (the elected variable "
            "itself, present on the other two gates) and from nearest_real (up to two "
            "constraint-relaxed variants of the SAME variable). alternate_match always "
            "names a different variable."
        ),
    )
    elected_count: int | None = Field(
        default=None,
        description=(
            "optional; how many StatVars the request actually elected. Present whenever the "
            "elected shape's variable identity was graph-confirmed, whatever no_data reason "
            "or gate the outcome ultimately carries. None when the outcome stops short of a "
            "graph-confirmed identity altogether (variable_not_resolved, role_not_resolved, "
            "role_signature_incompatible, and binding_mismatch all resolve no StatVar to "
            "count). Counts the SAME variable as flagged_match whenever flagged_match is "
            "present. On gate=only_alternate_has_data, where flagged_match is absent, "
            "elected_count still counts the variable the request elected, distinct from "
            "alternate_match's different variable."
        ),
    )
    entity_terms: list[str] = Field(
        default_factory=list,
        description=(
            "the query entity text responsible for an entity-resolution outcome, in "
            "query order: every entity the engine could not resolve for "
            "entity_not_resolved, every ambiguous wording for entity_ambiguous, or "
            "every resolved-but-unadmitted entity for entity_role_not_admitted. Empty "
            "for every other no_data reason. This structured list is the client-facing "
            "source of those terms; rendered_sentence may quote the same wording for "
            "people, but clients must not parse it."
        ),
    )
    ambiguous_entities: list[AmbiguousEntityGroup] = Field(
        default_factory=list,
        description=(
            "every ambiguous wording responsible for an entity_ambiguous outcome, "
            "in query order, each paired with the graph candidates it could name. "
            "Empty for every other no_data reason. Invariant: every group's `text` "
            "equals some restatement.entities[i].text on the same part. Additive to "
            "entity_terms, which carries the same wordings as bare strings; "
            "rendered_sentence is unaffected by this field and quotes the same "
            "wordings and candidate names for people, but clients must not parse it."
        ),
    )
    source_terms: list[str] = Field(
        default_factory=list,
        description=(
            "the source wordings responsible for a source-resolution outcome, in query "
            "order: every included wording that matched no admitted source for "
            "source_not_resolved, "
            "every ambiguous wording for source_ambiguous, and the part's source_code for "
            "source_code_not_held, source_code_ambiguous and source_code_shares_variable. "
            "Empty for every other no_data reason. rendered_sentence may quote the same wording for people, but clients "
            "must not parse it."
        ),
    )
    ambiguous_sources: list[AmbiguousSourceGroup] = Field(
        default_factory=list,
        description=(
            "every ambiguous source wording responsible for a source_ambiguous outcome, in "
            "query order, each paired with the admitted sources it could name, and for "
            "source_code_ambiguous the part's source_code paired with the provenances "
            "holding it. Empty for every other no_data reason. A client posts the chosen "
            "candidate's ref on that mention of data_query, or for a source code as a new "
            "mention in its sources, and posts the part back."
        ),
    )
    unit_alternatives: list[UnitAlternative] | None = Field(
        default=None,
        description=(
            "the currencies a unit_not_resolved outcome offers, in the binder's "
            "order: the candidate ISO codes for an ambiguous currency name with no "
            "declared alias (XOF, XAF and CHF for 'francs'), or the role shapes "
            "for a role-relative currency phrase beside entities on two roles. "
            "Absent on every other outcome, and on a unit_not_resolved outcome "
            "the binder offers no candidates for"
        ),
    )
    unbound_qualifiers: list[ResolvedQualifier] = Field(
        default_factory=list,
        description=(
            "the ResolvedQualifier entries responsible for a qualifier_not_bound "
            "outcome, in query order: every entry from "
            "flagged_match.resolved.trace.query_qualifiers whose disposition is not "
            '"bound" ("unbound": a named code row exists but no final binding covers '
            'it; "no_vocabulary": the text names no code at all), then, for a part '
            "naming or leaving out sources, each source it could not bind, an "
            "\"unbound\" entry under the source's label whose named holds the source's "
            "references. Empty for every other no_data reason. This structured list "
            "is the client-facing source of which terms went unrecognised, alongside "
            "each one's own disposition; "
            "rendered_sentence may quote the same wording for people, but clients "
            "must not parse it."
        ),
    )
    denominator: NoDataDenominator | None = Field(
        default=None,
        description=(
            "the denominator the question named and the one the flagged match divides "
            "by, set on every denominator_not_available refusal the engine returns "
            "after electing a series; null for every other no_data reason. A client "
            "can offer the flagged match, or fetch its nearest_real level and the "
            "denominator asked and divide them"
        ),
    )
    data_query: ParsedInput | None = Field(
        default=None,
        description=(
            "the parsed form of this part, one part long, set when reason is "
            "entity_ambiguous, source_ambiguous or source_code_ambiguous: its variable "
            "text, entities with "
            "role_text, time, unit, sources, and operations. On entity_ambiguous every "
            "entity that resolved to one graph node is pinned to its dcid and each "
            "ambiguous wording left unpinned. A named group stays unpinned too, so it "
            "resolves to the same members again. On source_ambiguous every source "
            "mention that resolved carries its ref and each ambiguous wording carries "
            "none. On source_code_ambiguous it carries the code and the part's mentions. "
            "A client pins the candidate the user picks from ambiguous_entities or "
            "ambiguous_sources and posts the part back, through the parsed door "
            "or as a dataset_resubmit parsed entry. It is set when the part has no "
            "restatement too: an entity whose direction the question leaves unnamed "
            "carries no role_text. Null for every other no_data reason, and when the "
            "part's inputs exceed the ParsedPart bounds"
        ),
    )

    @model_validator(mode="after")
    def _derive_category(self) -> NoData:
        self.category = "unresolved" if self.reason in _UNRESOLVED_NO_DATA_REASONS else "absent"
        return self

    @model_validator(mode="after")
    def _validate_unit_alternatives(self) -> NoData:
        if self.unit_alternatives is not None and self.reason != "unit_not_resolved":
            raise ValueError("unit_alternatives belongs to reason='unit_not_resolved' only")
        return self

    @model_validator(mode="after")
    def _validate_denominator(self) -> NoData:
        if self.denominator is not None and self.reason != "denominator_not_available":
            raise ValueError("denominator belongs to reason='denominator_not_available' only")
        return self

    @model_validator(mode="after")
    def _validate_gate(self) -> NoData:
        if self.gate == "only_alternate_has_data":
            if self.alternate_match is None or self.flagged_match is not None:
                raise ValueError("only_alternate_has_data requires alternate_match")
        elif self.gate is not None:
            if self.flagged_match is None or self.alternate_match is not None:
                raise ValueError(f"{self.gate} requires flagged_match")
        elif (
            self.reason not in _UNGATED_FLAGGED_MATCH_REASONS
            or self.flagged_match is None
            or self.alternate_match is not None
        ) and (self.flagged_match is not None or self.alternate_match is not None):
            raise ValueError("a near-match spec requires gate")
        return self
