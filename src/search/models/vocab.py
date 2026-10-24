from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, get_args

from dfo_platform import RoleAxis  # noqa: F401 -- re-exported by search.models

SCHEMA_VERSION = 1


def _require_schema_version_int(value: object) -> object:
    """Accept only an exact integer. Reject booleans and floats before Literal[1] coerces them."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("schema_version must be the integer 1")
    return value


# ---------------------------------------------------------------------------
# After GA, changes to these v1 contract values require a major version bump.
# Before GA, approved breaking migrations may update v1 (conformance C3/C124).
# ---------------------------------------------------------------------------
AnswerKind = Literal["resolution", "catalogue", "declined"]
StatusLiteral = Literal["definite", "candidates", "no_data", "unsupported"]
DeclinedReason = Literal["out_of_scope", "requires_analysis"]
OperationVerb = Literal["rank", "aggregate", "growth", "share_of", "explain", "forecast", "report"]
OperationAxis = Literal["place", "origin", "destination", "category", "year"]
BindingKind = Literal["value", "set", "unbound"]
Axis = Literal["what", "how", "where", "when", "source"]
# SentenceAxis is the axis a sentence token fills: every request-slot axis plus
# "unit", the byline's unit of measure, and "method", the byline's measurement method,
# which no request slot carries.
SentenceAxis = Literal[Axis, "unit", "method"]
SentenceSegment = Literal["coverage", "heading", "byline", "departure", "assumption"]
CoverageKind = Literal["exact", "breadth", "bare"]

# BindingProvenance records why a slot has its value, independently of its
# BindingKind. "binder" is the default LLM election. "text_recognized" is a
# graph code named by the query. "measure_elected" comes from the elected
# measure partition. "declared_default" fills an open slot from its code
# system's declared default when another slot binds within that subtree, or
# from the default its shape's population type declares.
BindingProvenance = Literal["binder", "text_recognized", "measure_elected", "declared_default"]

# ---------------------------------------------------------------------------
# FacetAxis describes observation facets. Axis describes request slots.
# Keep them separate: measurementMethod and unit have no request-slot axis.
# ---------------------------------------------------------------------------
FacetAxis = Literal["provenance", "measurementMethod", "unit"]

_FACET_AXIS_SCHEMA_VERSION: dict[str, int] = {
    "provenance": 1,
    "measurementMethod": 1,
    "unit": 1,
}
"""Every facet axis name any published schema knows, mapped to the
schema_version that introduced it. All three ship at schema_version 1.
A posted axis absent from this map is unrecognised (`request_invalid`/400);
one present but mapped to a version greater than the running SCHEMA_VERSION
is recognised but not yet published here (`unsupported_schema_version`/422)
-- see `BaseIdentity`'s `mode="before"` validator, which is this map's only
reader. The higher-version branch has no real member to exercise yet, so
test coverage monkeypatches one synthetic entry into this map rather than
naming a real axis."""

EntryPath = Literal["raw_text", "parsed", "dataset_resubmit"]
ValueKind = Literal["entity", "enum_value", "time_window", "source", "literal"]
PipelineStepName = Literal["extract", "recall", "shape", "bind", "materialise", "answer"]
DataCommonsId = Literal["dfo", "base", "un_system"]
"""The Data Commons a provenance's data comes from: the Development Finance
Observatory's own imports, Data Commons by Google, or the UN System Data
Commons. Mirrors `origin_catalogue.OriginId`."""
SourceRefKind = Literal["provenance", "dataset", "source"]
"""What a source reference names: one provenance, a catalogue Dataset (`dc/d/*`) and the
provenances that are part of it, or a publishing Source and every provenance it
publishes."""
SourcePolarity = Literal["include", "exclude"]
"""Whether a source mention asks for its source or asks to leave it out."""
EntityRoleKey = Literal["observationAbout", "origin", "destination"]
"""The role-key vocabulary a resolved entity plays: the subject role, or one
of the engine's two flow axes. ``origin`` and ``destination`` name axes, never
a StatVar's declared property; declared labels remain presentation data."""
NoDataReason = Literal[
    "source_not_resolved",
    "source_ambiguous",
    "source_code_not_held",
    "source_code_ambiguous",
    "source_code_shares_variable",
    "entity_not_resolved",
    "entity_ambiguous",
    "entity_set_too_large",
    "entity_role_not_admitted",
    "variable_not_resolved",
    "role_not_resolved",
    "unit_not_resolved",
    "rate_coverage_gap",
    "no_matching_slice",
    "binding_mismatch",
    "denominator_not_available",
    "no_observations",
    "facet_not_available",
    "role_incompatible",
    "role_signature_incompatible",
    "qualifier_not_bound",
    "timed_out",
]
NoDataCategory = Literal["unresolved", "absent"]
QuantityForm = Literal["per_person", "per_head_base", "per_area", "rate", "share"]
"""A normalised form a question can ask for: per person, per a numbered count of people
(`per 100,000 people`), per area, a rate (one quantity to another) or a share of a
whole."""
DepartureCode = Literal[
    "closest_match",
    "measure_not_applied",
    "operation_not_applied",
    "breakdown_not_applied",
    "entity_unavailable",
    "year_unavailable",
    "modelled_values",
    "group_read_as_members",
    "whole_total_read",
    "whole_total_unpublished",
    "direction_discarded",
    "direction_inferred",
    "scope_limited",
]
"""Every way an answer can differ from what its request asked (see
`models.departure.Departure`), declared in precedence order, so the departure
that changes the number most comes first. A closed vocabulary of its own. No
departure code is a `Warning.code` or a refusal code, so the platform
code registry does not list them."""
DEPARTURE_ORDER: tuple[DepartureCode, ...] = get_args(DepartureCode)
"""`DepartureCode` in precedence order. `_PartBase.departures` follows it."""
AssumptionKind = Literal["declared_default", "open_role", "place_scope"]
"""Every kind of value an answer fills that its request did not ask for (see
`models.assumption.Assumption`), in the order a part's `assumptions` lists them. A closed
vocabulary of its own, like `DepartureCode`, outside the platform code registry."""
ObservationsOmissionReason = Literal[
    "observations_timed_out",
    "observations_no_matching_source",
    "observations_windowed_scope_unsupported",
    "observations_converted_scope_unsupported",
    "observations_no_dispatchable_facet",
    "observations_bulk_fetch_failed",
    "observations_probe_failed",
    "observations_preview_too_large",
    "observations_nothing_to_probe",
    "observations_clients_unavailable",
    "observations_no_points",
]
"""Every reason `SpecObservations.omitted.reason` (models/observations.py's
`ObservationsOmission`) can carry -- one closed vocabulary, kept in lock-step
by hand with `search.engine.region.codes`' own `OBSERVATIONS_*` string
constants (this package cannot import the engine package; `NoDataReason`
above is this same models/engine split's own precedent). Reason equals
warning code: every one of these is also registered in
`search.engine.codes.WARNING_CODES`, so a consumer reads one vocabulary
regardless of whether it is looking at `omitted.reason` or a `Warning.code`
that rides the same cause."""
NoDataGate = Literal[
    "no_member_has_data_at_place",
    "no_member_asserts_bound_properties",
    "only_alternate_has_data",
]
NO_DATA_RESOLUTION_ORDER: tuple[NoDataReason, ...] = (
    "source_not_resolved",
    "source_ambiguous",
    "source_code_not_held",
    "source_code_ambiguous",
    "source_code_shares_variable",
    "entity_not_resolved",
    "entity_ambiguous",
    "entity_set_too_large",
    "entity_role_not_admitted",
    "variable_not_resolved",
    "role_not_resolved",
    "role_incompatible",
    "role_signature_incompatible",
    "unit_not_resolved",
    "rate_coverage_gap",
    "qualifier_not_bound",
    "no_matching_slice",
    "binding_mismatch",
    "denominator_not_available",
    "no_observations",
    "facet_not_available",
    "timed_out",
)
"""The contract's own no_data resolution order: source (a part's source words
compile before anything else resolves, so an unmatched wording, then an ambiguous
one, comes first), source code (the part's series code compiles at the door against the
sources it allows, and a held code whose StatVars cannot answer the part refuses
source_code_not_held where the part would refuse variable_not_resolved, after the entity
reasons), entity (entity_ambiguous and
entity_set_too_large and entity_role_not_admitted sit at the entity position,
right after entity_not_resolved, in that order: the four run by how far the
entity got, from no identity, to several candidate identities, to one
confirmed identity whose expansion is unusable, to one confirmed identity
whose role is wrong. An elected shape whose places exceed the bound gets
`entity_set_too_large` only once elected, after the variable reasons),
variable, role (the two
role-compatibility codes sit at the role position; a whole population the request
directs to the other side refuses role_incompatible once every entity resolves,
before the variable reasons, since the conflict holds on every shape), unit (`rate_coverage_gap`,
currency-conversion's own no_data reason, sits immediately after
`unit_not_resolved`. `recipe_not_declared`, the other currency-conversion
code, maps to `unsupported_code` rather than `no_data_reason` on the `raw_text`/`parsed`
resolve doors, so a StatVar with no declared recipe surfaces as an
`unsupported` part instead of reaching this order), slice
(binding_mismatch sits at the slice position), denominator, observations,
facet, then timed_out, which only a part the request's deadline cut short carries, since
it never ran far enough to meet another reason. When several reasons could apply, the first
one in this order is what `reason` reports. This documents the contract sequence; it is not the
enforcement mechanism. Each failure site in the pipeline returns immediately
(region/pipeline.py, region/resubmit.py), so this constant can drift from
actual behaviour if a check is reordered without updating it. Every site that picks one
reason from several applies it through `elect_no_data_reason`."""


def elect_no_data_reason(reasons: Iterable[NoDataReason]) -> NoDataReason:
    """The reason `NO_DATA_RESOLUTION_ORDER` ranks first among `reasons`.

    The result depends only on the set of reasons, and adding a reason ranked lower never
    changes it.

    Raises:
        ValueError: `reasons` is empty, or holds a string outside the order.
    """
    return min(reasons, key=NO_DATA_RESOLUTION_ORDER.index)


_UNRESOLVED_NO_DATA_REASONS: frozenset[NoDataReason] = frozenset(
    {
        "source_not_resolved",
        "source_ambiguous",
        "source_code_not_held",
        "source_code_ambiguous",
        "source_code_shares_variable",
        "entity_not_resolved",
        "entity_ambiguous",
        "entity_set_too_large",
        "entity_role_not_admitted",
        "variable_not_resolved",
        "role_not_resolved",
        "unit_not_resolved",
        "timed_out",
    }
)
"""The NoDataReason values NoData._derive_category maps to category="unresolved"
(the request could not be read into the graph); every other reason maps to
"absent" (the read request names a place holding nothing). An explicit set
rather than a naming-convention check (e.g. a "_not_resolved" suffix), since
not every reason in this category shares that suffix: source_ambiguous, the three
source_code_* reasons, entity_ambiguous, entity_set_too_large and entity_role_not_admitted
do not."""
_FLAG_BEARING_NO_DATA_REASONS: frozenset[NoDataReason] = frozenset(
    {"no_observations", "no_matching_slice", "qualifier_not_bound"}
)
"""The NoDataReason values that carry a graph-confirmed variable identity,
whether the resolve path that minted them reached a Materialised result or
returned a bare NoDataDraft carrying one of these reasons directly (the
role-bearing resolve path's own no_observations relaxation, for one).
qualifier_not_bound reaches this set a third way: the guard that mints it
runs only after materialise has already confirmed the elected shape's
members. NoData.elected_count's own derivation
(search.engine.region.state._elected_stat_vars) keys off this same set, so
elected_count is set for exactly these reasons and no others. Every other
reason stops short of a graph-confirmed identity, so elected_count stays
None there."""
Confidence = Literal["high", "medium", "low"]
Ordering = Literal["broadest_first", "intent_rank"]
CandidateBasis = Literal["structural", "nominated", "twin", "direction", "breakdown"]
"""Why the status rule offers a reading as a candidate beside the elected one:
`structural`, a reading of the same population type and measured property that differs
only on a stat type, price basis or denominator the question leaves open; `nominated`, a
reading the bind model named as a genuine reading of the phrase that the question does not
rule out; `twin`, an equivalent reading of the elected one that answers where the elected
one holds no data; `direction`, the same reading with the other direction for a place the
question names without one, where both directions hold data; `breakdown`, the same measure
broken down by another set of slots, offered when its family holds no total and the question
names no breakdown that serves a member."""
PreviewOrder = Literal["ranked", "fixed"]
"""How a role-bearing bulk preview chose its series: `ranked` reads every series and shows
the first by `search.observations.series_rank`; `fixed` reads a selection made before any
value is known (`search.observations.fixed_selection`)."""
FacetBasis = Literal["bound", "declared_default", "fallback"]
"""Why a facet axis (or, on a unit-request's own member, a currency/prices/
base_year value) carries the value it does: 'bound' when the request itself
constrained it, 'declared_default' when an unbound axis was completed from a
declared deployment policy, 'fallback' when neither applies and the engine's
own selection (facets holding a point in the requested years first, then
DFO's facets for a variable both DFO and base Data Commons hold, then the
most places or role tuples covered there, then the most requested years the
span covers, then most observations, then the latest observation, then a
deterministic tie-break;
or, for a unit-request member, the binder's fixed fallback constant)
supplied it. Shared by ResolvedFacet.basis
(axis-level) and ResolvedFacet.request_basis (per-member, unit axis only) --
the same three-value vocabulary either way."""
PartyKind = Literal["party", "total", "residual"]
"""What one party of a flow is beside the other parties on its side: a `party` holds figures
of its own, a `total` holds other parties' figures ("Africa, Total", the G7), so a sum across a
flow role that counts it counts its members twice, and a `residual` holds figures no other party
does ("Africa, unallocated by country"). Mirrors `code_index.types.PartyKind` one-for-one."""
PartyClass = Literal["official", "private", "residual"]
"""Whose money a flow party's figures count: `private` for a private provider of development
finance, `official` for any other party, and `residual` for a residual. A whole population
covers classes (`EntityGroup.whole_of`): "Official Donors, Total" covers every `official`
donor. Mirrors `code_index.types.PartyClass` one-for-one."""
SlotOptionRelation = Literal["child", "parent", "root"]
"""Where a level-scoped `SlotOption` sits in its slot's menu. On a bound slot,
`"child"`/`"parent"` narrow or widen one step from the bound value in a
hierarchical code system's taxonomy, and a bound root's menu adds every other root
as `"root"`. On an unbound hierarchical slot, `"root"`
is a classification root and `"child"` one of its children, named by
`SlotOption.parent`. Every value of a flat code system
(`code_index.CodeSystemDeclaration.hierarchical` is `False`) is `"root"`.
Mirrors `code_index.LevelOption.relation`'s own vocabulary one-for-one."""
