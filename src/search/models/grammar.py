from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    Field,
    model_validator,
)

from .base import StrictModel
from .vocab import Axis, DataCommonsId, RoleAxis, SentenceAxis, SentenceSegment, ValueKind

# ---------------------------------------------------------------------------
# Leaf types
# ---------------------------------------------------------------------------


class GraphRef(StrictModel):
    """Every graph identifier in the contract."""

    dcid: str = Field(description="the graph identifier")
    label: str = Field(description="plain-language display name, read from the graph")


class SourceRef(StrictModel):
    """The organisation behind a provenance: the Source node its `source` arc
    names on our Data Commons and on Data Commons by Google, or the agency
    provenance itself on the UN System Data Commons."""

    ref: str = Field(description="the source's dcid")
    label: str = Field(description="the source's display name")


class DataCommonsRef(StrictModel):
    """The Data Commons a provenance's data comes from."""

    id: DataCommonsId = Field(description="'dfo', 'base' or 'un_system'")
    label: str = Field(
        description=(
            "the Data Commons' display name: Development Finance Observatory, Data "
            "Commons by Google, or UN System Data Commons"
        )
    )


class ProvenanceRef(GraphRef):
    """A provenance on the wire: its dcid and label, the source behind it, and
    the Data Commons its data comes from. `label`, `source` and `data_commons`
    come from the origin catalogue's provenance table. A provenance the table
    does not list carries its graph label and neither `source` nor
    `data_commons`, and is omitted when the graph has no label for it."""

    source: SourceRef | None = Field(
        default=None, description="the organisation behind the provenance"
    )
    data_commons: DataCommonsRef | None = Field(
        default=None, description="the Data Commons the provenance's data comes from"
    )


class TimeWindow(StrictModel):
    """A date window. Absence of a bound means open-ended in that direction."""

    start_year: int | None = Field(default=None, description="absent means open-ended start")
    end_year: int | None = Field(
        default=None, description="absent means open-ended end (to latest)"
    )

    @model_validator(mode="after")
    def at_least_one_bound(self) -> TimeWindow:
        if self.start_year is None and self.end_year is None:
            raise ValueError("TimeWindow needs at least one of start_year or end_year.")
        return self


RequestedWindow = TimeWindow | Literal["latest"]
"""An explicit ``latest`` request stays distinct from a bounded window and no date signal,
so resubmission resolves the latest year again instead of pinning the original year.
"""


def in_window(date_str: str, window: TimeWindow | None) -> bool:
    """Return whether a date falls within the window. Include unparseable dates by default."""
    if window is None:
        return True
    try:
        year = int(str(date_str)[:4])
    except (ValueError, TypeError):
        return True  # non-year date strings: include by default
    if window.start_year is not None and year < window.start_year:
        return False
    return not (window.end_year is not None and year > window.end_year)


class SlotValue(StrictModel):
    """One taxonomy member or facet value bound to a slot."""

    ref: GraphRef | None = Field(
        default=None,
        description="the graph object bound, when the value is a graph object",
    )
    value_kind: ValueKind = Field(description="what kind of value this is")
    time_window: TimeWindow | None = Field(
        default=None, description="present iff value_kind == time_window"
    )
    literal: str | None = Field(default=None, description="present iff value_kind == literal")


class SlotKey(StrictModel):
    """A slot's typed identity: axis, constraint property, and human label."""

    axis: Axis = Field(description="the slot grammar axis")
    property: GraphRef | None = Field(
        default=None,
        description=(
            "the constraint property that defines this slot when one exists; "
            "null or omitted for when, source, and the where-axis slot that binds "
            "the subject/destination entity directly (no constraint property backs it)"
        ),
    )
    label: str = Field(description="human label, e.g. purpose, donor")

    def as_ref(self) -> str:
        """Return the ``axis:property_dcid`` key used by sentence tokens and slot options."""
        return f"{self.axis}:{self.property.dcid if self.property else ''}"


# ---------------------------------------------------------------------------
# The four Binding arms
# ---------------------------------------------------------------------------


class BindingValue(StrictModel):
    """A slot bound to exactly one taxonomy member."""

    kind: Literal["value"] = Field(default="value", description="binding kind discriminator")
    value: SlotValue = Field(description="exactly one taxonomy member")


class BindingSet(StrictModel):
    """A slot bound to two or more taxonomy members, all in scope."""

    kind: Literal["set"] = Field(default="set", description="binding kind discriminator")
    values: Annotated[list[SlotValue], Field(min_length=2)] = Field(
        description="two or more members, all in scope (a union)"
    )
    set_ref: GraphRef | None = Field(
        default=None,
        description="the group the set was drawn from, when one exists",
    )


class BindingUnbound(StrictModel):
    """A slot explicitly open to the whole slot in scope, NOT null or missing."""

    kind: Literal["unbound"] = Field(
        default="unbound", description="explicit whole slot in scope, NOT null/missing"
    )


Binding = Annotated[
    BindingValue | BindingSet | BindingUnbound,
    Field(discriminator="kind"),
]


class Expansion(StrictModel):
    """Codes a hierarchical value/set binding resolved to beyond its own bound
    value(s): the descendant codes admitted by walking the code index's
    `specializationOf` closure for a dimension the index declares
    `hierarchical` (code_index.query.CodeIndex.descendants).

    Additive only. Never a substitute for `binding`, and never fed into spec
    identity -- `BindingValue`/`BindingSet` are the sole identity-bearing wire
    form (see Slot.expansion's own docstring for why: identity is minted from
    `BindingSet.values`, built from the query's own drafted values, before
    resolution ever widens them). `added` never repeats a dcid already in the
    binding's own value(s); it holds only what expansion contributed on top.
    """

    added: Annotated[list[GraphRef], Field(min_length=1)] = Field(
        description=(
            "descendant codes admitted beyond the slot's own bound value(s), in "
            "the code index's descendant-walk order. Never empty -- a slot whose "
            "expansion added nothing carries no Expansion at all (Slot.expansion "
            "is None), rather than one with added=[]"
        )
    )


class Slot(StrictModel):
    """One slot in the chosen shape, with its current binding state."""

    key: SlotKey = Field(description="the slot's typed identity (axis + property + label)")
    binding: Binding = Field(description="exactly one of the three binding states")
    expansion: Expansion | None = Field(
        default=None,
        description=(
            "codes the resolver admitted beyond this slot's own bound value(s), "
            "when the bound value's code system is declared hierarchical; None "
            "for every other slot, including one bound on a flat code system "
            "(e.g. crsTypeOfFlow) or a property the deployment's code index does "
            "not carry at all. Reports resolution-time breadth only -- `binding` "
            "alone is what the query asked for and what spec identity hashes."
        ),
    )
    summable: bool = Field(
        default=False,
        description=(
            "true when the datasets door's `groupby` sums the family's members "
            "over this slot into one complete series: the slot's property is a "
            "code system the deployment's code index declares summable "
            "(capabilities `summable_axes`), and the dataset the family is "
            "served from declares that an observation it carries no row for "
            "reads as zero, so a member with no rows adds nothing. False for "
            "every other property, for the property that classifies a "
            "dataset's rows, and for every slot of a family whose dataset "
            "declares no such reading, where a member with no rows refuses the "
            "sum. Independent of the binding, so an unbound, set or "
            "value-bound slot reads alike. Spec identity does not hash it."
        ),
    )


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


class ShapeUpstream(StrictModel):
    """The one source, and the Data Commons serving it, behind every member of
    an upstream shape."""

    source: SourceRef = Field(description="the organisation behind every member's data")
    data_commons: DataCommonsRef = Field(description="the Data Commons serving that data")


class Shape(StrictModel):
    """The structural family chosen for the query."""

    shape_id: str = Field(description="stable id of this structural family")
    label: str = Field(description="plain-language family name, e.g. development finance flows")
    population_type: GraphRef | None = Field(
        default=None,
        description=(
            "the graph populationType; None when the StatVar does not assert it "
            "(custom-DC imports may omit any five-tuple arc)"
        ),
    )
    measured_property: GraphRef | None = Field(
        default=None,
        description=(
            "the graph measuredProperty; None when the StatVar does not assert it "
            "(custom-DC imports may omit any five-tuple arc)"
        ),
    )
    stat_type: GraphRef | None = Field(
        default=None,
        description=(
            "the graph statType, e.g. measuredValue; None when the StatVar does "
            "not assert it (custom-DC imports may omit any five-tuple arc)"
        ),
    )
    measurement_qualifier: GraphRef | None = Field(
        default=None,
        description="structural anchor; present only when the measured property is qualified",
    )
    measurement_denominator: GraphRef | None = Field(
        default=None,
        description=(
            "a structural anchor, present only on a ratio or per-capita measure. Holds the "
            "measurementDenominator arc's target dcid for a resolved StatVar or for a "
            "dangling target that itself carries a dcid. Is None both when no denominator "
            "arc exists at all and when the arc's target carries no dcid. A null value here "
            "does not by itself establish that no denominator arc exists. Read "
            "measurement_denominator_unresolved alongside it to tell the two apart"
        ),
    )
    measurement_denominator_unresolved: bool = Field(
        default=False,
        description=(
            "true when the measurementDenominator arc is present and its target is a "
            "ProvisionalNode the graph defines no real StatVar for, whether or not that "
            "target carries a dcid. False when no denominator arc exists at all, or when "
            "the arc names a confirmed StatVar. A client should read this flag before "
            "trusting measurement_denominator on its own. A null ref is consistent with "
            "either no arc (this flag false) or a dcid-less dangling target (this flag "
            "true), and a populated ref is consistent with either a confirmed StatVar "
            "(this flag false) or a dcid-bearing dangling target (this flag true)"
        ),
    )
    slot_keys: list[SlotKey] = Field(
        description="the constraint and facet slots this family exposes, in display order"
    )
    member_count: int = Field(
        description=(
            "the size of the resolved population behind this response: how many "
            "StatVars the shape's current bindings admit, narrowing as a slot "
            "binds and widening again as one is cleared"
        )
    )
    member_digest: str | None = Field(
        default=None,
        description=(
            "lowercase sha256 hex of the RFC 8785 (JCS) canonical JSON of the "
            "sorted list of bare dcids (no 'dcid:' prefix) in the population "
            "member_count names. Present only when that population is exact "
            "(a true total, not a floor over a capped recall sample) and is "
            "the family the code index resolves for the spec's identity; null "
            "otherwise. /api/datasets states the same family for the minted "
            "spec as trace.family_member_digest, so a caller compares the two "
            "to verify the surfaces agree on membership -- see docs/"
            "conformance-search-api-v1.md's clause next to C75."
        ),
    )
    refine_supported: bool = Field(
        default=False,
        description=(
            "true unless the shape's StatVar(s) are ratio-backed (with "
            "_AsAFractionOf_ tokens), which construct-and-confirm sibling "
            "StatVars cannot refine"
        ),
    )
    data_commons: list[DataCommonsRef] = Field(
        default_factory=list,
        description=(
            "every Data Commons holding admitted data for a member StatVar, in the "
            "order dfo, base, un_system; a Data Commons by Google variable lists dfo "
            "as well when the Observatory's own imports hold it. Empty when the "
            "engine derived the shape without its members"
        ),
    )
    upstream: ShapeUpstream | None = Field(
        default=None,
        description=(
            "the one source and Data Commons every member's data comes through, "
            "present only for a shape outside the Observatory's own imports whose "
            "members the origin catalogue places under exactly one source"
        ),
    )


# ---------------------------------------------------------------------------
# SentenceToken: the ordered token plan the confirmation sentence is built from
# ---------------------------------------------------------------------------


class SentenceToken(StrictModel):
    """One token in the ordered sentence plan: a "text" token (invariant
    wording such as a connective, a bare separator, the terminal period, or a
    departure sentence) or a "slot" token (a slot- or entity-bound phrase, or
    the wording of an ambiguous entity).

    ``rendered_sentence`` (the flat string) is the plain concatenation of every
    token's ``text``, in order. The two are computed together and cannot drift.
    A "slot" token with nothing bound appears too, with ``text=""``: the same
    token structure serves a loading skeleton (before binding is known) and the
    final sentence (after it is). The first non-empty token's text is
    capitalised.

    Every token belongs to one ``segment``. The heading states what the answer
    is: actor, measure noun, counterpart, and period, in possessive order, one
    slot token per axis ("Germany's gross ODA disbursements to Kenya, 1973 to
    2024"). A unit that changes the magnitude (per capita, percent of GDP, PPP
    international dollars) stays in the heading. The byline follows it: the
    latest-year label when one applies, the plain unit with its price basis
    ("current US dollars", "constant 2022 US dollars") as a slot token with
    axis "unit", the source, and, when the identity binds measurementMethod, the
    method as a slot token with axis "method". Departures come last and sit only on a part's or
    the envelope's sentence, so ``spec.resolved.sentence_plan`` carries none.
    """

    kind: Literal["text", "slot"] = Field(
        description=(
            "'text' is invariant wording: a connective carrying its own "
            "surrounding whitespace (' for ', ' to '), a possessive suffix "
            "(\"'s\"), a bare ' '/', ' separator, the terminal '.', or a departure "
            "sentence; 'slot' is a slot- or entity-bound phrase, or on an "
            "entity_ambiguous no_data plan the ambiguous wording itself, which a "
            "client hangs the candidate menu on. A client that does not recognise "
            "a kind should render its text as plain text regardless."
        )
    )
    text: str = Field(
        description="the rendered text for this token; empty string for a slot with nothing bound"
    )
    axis: SentenceAxis | None = Field(
        default=None,
        description=(
            "the grammatical axis this token fills, when kind=='slot': one of the "
            "request-slot axes, 'unit' for the byline's unit of measure, or "
            "'method' for the byline's measurement method, which no request slot "
            "carries. The unit token's binding_ref is 'unit:', and "
            "its options are the Spec's resolved.alternative_measures entries with "
            "axis 'unit': the price-basis siblings (current and constant prices) "
            "where the graph holds both. The method token appears only when the "
            "identity binds measurementMethod (a ref, a null ref, or a set), after "
            "the source token; its binding_ref is 'method:', its text the bound "
            "method's label without a trailing full stop or 'No stated method' for "
            "a null ref, and its options the Spec's facet_options entry with axis "
            "'measurementMethod'"
        ),
    )
    segment: SentenceSegment = Field(
        default="heading",
        description=(
            "the part of the sentence this token belongs to: 'heading' (what the "
            "answer is), 'byline' (latest-year label, unit, source, and a bound "
            "method), or "
            "'departure' (the text of one Departure on the same part, never on a "
            "Spec's own plan). A client styles each segment as a block; the plain "
            "join of every token's text still equals rendered_sentence"
        ),
    )
    property_dcid: str | None = Field(
        default=None,
        description=(
            "the constraint property dcid this token renders, when one exists; "
            "null or omitted for property-less axes (when/source), for a shape-label "
            "fallback phrase with no backing slot, and for entity-role tokens with no "
            "backing where slot. Set on a 'destination' entity-role token when the shape "
            "declares a where slot backing that axis (e.g. dev-finance)."
        ),
    )
    role: RoleAxis | None = Field(
        default=None,
        description=(
            "origin/destination when this token renders a directional entity role "
            "(the flow's two endpoints read back), or the directional scope an unbound "
            "where slot resolved to. entity_refs is the discriminator: non-empty on the "
            "entity read-back, empty on the unbound-where-slot fallback. binding_ref is "
            "not: an entity read-back carries binding_ref too whenever the shape declares "
            "a where slot backing the 'destination' axis (e.g. dev-finance), the same "
            "condition under which the unbound-where-slot fallback always carries it, so "
            "a present binding_ref value never separates the two. Both kinds join to the "
            "Spec.resolved.entity_options entry on the same role. A third token sits on "
            "an entity_ambiguous no_data plan, which has no Spec: the ambiguous wording, "
            "with the role the request gave it (null when it gave none), no entity_refs "
            "and no binding_ref, and text equal to one NoData.ambiguous_entities[i].text "
            "on the same part. The client titles that wording's candidate menu by this "
            "role and pins the pick into NoData.data_query."
        ),
    )
    binding_ref: str | None = Field(
        default=None,
        description=(
            "join key correlating this token to its Spec.resolved.slots / SlotOptions "
            "entry: 'axis:property_dcid', i.e. SlotKey.as_ref() (matching Slot.key "
            "and SlotOptions.key). "
            "Set for a slot-backed token, and also for a 'destination' entity-role token "
            "when the shape declares a where slot backing that axis (e.g. "
            "dev-finance) -- see entity_refs for the entity side of that token. "
            "The byline's unit token carries 'unit:', which no slot backs; its "
            "options are the resolved.alternative_measures entries with axis 'unit'. "
            "The byline's method token carries 'method:', which no slot backs either. "
            "Null or omitted for plain connective text, a shape-label fallback phrase "
            "with no backing slot, and 'origin'/unbacked-'destination' entity-role tokens."
        ),
    )
    entity_refs: list[str] = Field(
        default_factory=list,
        description=(
            "entity dcids this token renders, set on the entity read-back, one of the "
            "two token kinds role marks (see role's own docstring for the other, the "
            "unbound-where-slot fallback, which leaves entity_refs empty even though it "
            "too carries role) -- a directional entity read-back has no "
            "Spec.resolved.slots entry of its own, though a 'destination' token may "
            "additionally carry binding_ref when a backing where slot exists (e.g. "
            "dev-finance); one dcid per entity in the token's group, in the same order "
            "as the entities joined into this token's text -- e.g. a 'destination' "
            "token reading 'Ethiopia, Kenya, and Uganda' carries all three dcids, so a "
            "client can tell a 3-entity answer from a 1-entity one without parsing "
            "rendered text. Empty when role is unset, and this emptiness, not "
            "binding_ref, is the join rule that separates the two role-bearing kinds."
        ),
    )
    states: list[str] = Field(
        default_factory=list,
        description=(
            "the refs a heading or byline slot token states, so a client edits each "
            "from this token: its own binding_ref, then each binding the renderer "
            "left silent because this token's text already says it. Refs read as "
            "binding_ref does ('how:crsTypeOfFlow', 'where:', 'origin:', 'unit:'), "
            "plus 'axis:<property dcid>' for a measure axis no slot carries "
            "('axis:nettingBasis'). The measure noun states the measure axes its "
            "label or authored noun carries, the how values it names, and the "
            "agreed open axes beside it; a StatVar name or graph name states the how "
            "values it folds. The byline's unit token states 'unit:' and, when it "
            "names a price basis, the measurementQualifier axis. The byline's source "
            "token states 'source:', which is also its binding_ref where the answer "
            "binds a source slot. The byline's method token states 'method:', its "
            "binding_ref, and while the identity binds no method no token states "
            "'method:'. Empty on a token "
            "with empty text, on text tokens, and outside heading and byline slot "
            "tokens. Within one part, no ref appears on two tokens with text. A "
            "bound slot, an unbound slot with options, or a role that no token "
            "states is stated nowhere in the sentence (a boolean member, an "
            "adjectival fold dropped whole, an open role with no count), and a "
            "client offers it apart from the sentence with its current value. On "
            "the envelope plan a token with no part_index states its refs for "
            "every part it stands for"
        ),
    )
    distinguishes: bool | None = Field(
        default=None,
        description=(
            "true when this token's text differs from the same-bucket text on "
            "another candidate in the same CandidateSet; null or omitted on "
            "every other token, including every token on a Spec rendered "
            "outside a candidate set"
        ),
    )
    part_index: int | None = Field(
        default=None,
        description=(
            "this token's position in Resolution.parts, when this plan is the "
            "envelope-level composed sentence (Resolution.sentence_plan) and this "
            "token belongs to exactly one part's own contribution -- a variable phrase "
            "token, or a qualifier attached to it because it did not agree across every "
            "part, or a token of a part composed as its own sentence. Null carries "
            "three distinct readings a client must not conflate: on "
            "a part's own plan (DefinitePart.spec.resolved.sentence_plan, or the "
            "candidates/no_data/unsupported sentence_plan on _PartBase) the field has "
            "no meaning at all and is never set; on a composer-minted connective, it "
            "means the token belongs to the sentence as a whole rather than to any "
            "one part (the Oxford-joined ' and '/', ', the plan's own terminal '.', "
            "and the shared ' ' the per-part fallback puts between two sentences are "
            "all such tokens); and on a token every contributing part states alike and "
            "the join states once (a hoisted qualifier on the where, when, or source "
            "axis, each a wire value of axis, or the origin role, identified by "
            "role=='origin' instead since SentenceAxis has no 'origin' member; a "
            "shared possessive lead; the shared byline; or a heading or byline token "
            "of a part a later part folded into, binding_folded, which both parts "
            "state) it means the token no longer describes a single variable."
        ),
    )
