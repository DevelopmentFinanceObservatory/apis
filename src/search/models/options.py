from __future__ import annotations

from typing import Literal

from pydantic import (
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from .base import StrictModel
from .coverage import Coverage
from .grammar import SlotKey, SlotValue
from .vocab import PartyKind, RoleAxis, SlotOptionRelation

# ---------------------------------------------------------------------------
# SlotOptions: the taxonomy a slot was bound against, surfaced to the client
# ---------------------------------------------------------------------------

SlotOptionsSource = Literal["observed", "seed"]


class SlotOption(StrictModel):
    """One candidate value the binder had available for a slot."""

    value: SlotValue = Field(description="one taxonomy member or observed facet value")
    coverage: Coverage | None = Field(
        default=None,
        description=(
            "this option's observation footprint if it were bound instead of the "
            "current value; populated when response_options.include_option_coverage is true"
        ),
    )
    relation: SlotOptionRelation | None = Field(
        default=None,
        description=(
            "where this option sits in a level-scoped constraint-slot menu on a "
            "code-index-carried taxonomy: 'child' or 'parent' of the slot's bound "
            "value, and 'root' for every other classification root beside a bound "
            "root; for an unbound hierarchical slot, 'root' for a classification "
            "root and 'child' for a child of the root named in parent; 'root' for "
            "every value of a flat code system. None on every other surface this "
            "type serves -- entity_options, facet_options, and the options door's "
            "OptionsSet -- none of which classify their candidates by level"
        ),
    )
    parent: str | None = Field(
        default=None,
        description=(
            "dcid of this option's parent in a hierarchical code system, for a "
            "level-scoped constraint-slot option that has one; None for a root, "
            "for a flat code system's values, and on every surface where "
            "relation is None"
        ),
    )
    member_count: int | None = Field(
        default=None,
        description=(
            "how many StatVars this index holds coded directly to this option's "
            "value (not a subtree rollup), for a level-scoped constraint-slot "
            "option; None on every other surface this type serves -- "
            "entity_options, facet_options, and the options door's OptionsSet -- "
            "none of which count members from the code index"
        ),
    )
    served_rows: int | None = Field(
        default=None,
        ge=0,
        description=(
            "the rows this answer's file would hold if this option were bound instead of "
            "the current value, in the terms of ResolvedSpec.download.rows: one per "
            "series and year with a point inside the requested window, or under 'latest' "
            "one per series holding its StatVar's latest year. A series is a served StatVar "
            "carrying the option or one of "
            "its code-index descendants (the widening the binder applies to a bound "
            "value) at one entity, or donor and recipient pair. An unbound slot counts "
            "the answer's own family; a bound slot counts the family with that slot left "
            "unbound and every other binding kept. An option this answer would serve "
            "nothing for is omitted from the menu, except the slot's current value. None "
            "where the option is not counted: on entity_options, facet_options and the "
            "options door, and on a bound slot whose family needs graph reads to count, "
            "where a trailing slot_options event can supply it. A StatVar the coverage artifact "
            "does not hold adds no rows: the count covers the pick's other StatVars, and an "
            "option all of whose StatVars are missing from the artifact is omitted"
        ),
    )
    served_rows_exact: bool | None = Field(
        default=None,
        description=(
            "true when served_rows equals the download.rows a pick of this option would "
            "state; false when it is an upper bound: a counted StatVar whose series sit in "
            "more than one facet group (provenance, method and unit) in the window, of "
            "which the pick's read elects one; an answer read in bulk whose pick's "
            "StatVars sit in more than one facet, since a bulk read elects one; under "
            "'latest', a StatVar with a facet observed more often than yearly; an answer "
            "that pins a facet value; a unit request whose conversion cannot be read as "
            "download.rows_exact describes, that converts a counted series from a currency "
            "the answer does not serve, or whose options the coverage probe counts (a "
            "bound slot of a subject-place family, a candidates reading); or a series "
            "whose facets carry no complete date list. Null exactly when served_rows is null"
        ),
    )
    party_kind: PartyKind | None = Field(
        default=None,
        description=(
            "on an origin or destination entity option, what the party is beside the others "
            "on its side: 'party' (a country, territory or organisation), 'total' (other "
            "parties' figures, so a "
            "sum across the role that counts it counts its members twice) or 'residual' "
            "(figures no other party holds), as the code index lists it. Absent for a party "
            "the index does not list, and on every other option"
        ),
    )

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo):
        # Unannotated on purpose: pydantic publishes a wrap serializer's return annotation as
        # the model's serialization schema, and `dict[str, Any]` reads as an open object.
        # coverage=null carries two different meanings depending on surface, and a
        # bare field value can't distinguish them -- so the serialisation context
        # has to. On the SSE trailing slot_options/entity_options events, the
        # published event schema promises "coverage=null (honest, not omitted)"
        # for an option the probe couldn't confirm; re-add it there when a parent's
        # exclude_none stripped it. On the buffered body, coverage is frequently
        # never probed at all (include_option_coverage=false) -- emitting null
        # there would read as "we probed and found nothing", a semantic lie, and
        # contradicts the buffered {ref, label} item shape -- so leave it omitted.
        # serialize_sse (events.py) is the only caller that passes this context.
        data = handler(self)
        if "coverage" not in data and info.context and info.context.get("surface") == "sse":
            data["coverage"] = None
        return data


class SlotOptions(StrictModel):
    """The full candidate set one slot was bound against.

    The taxonomy the binder chose from and the options a client can offer the user
    are the same artifact; this is that artifact, surfaced per slot. ``key`` matches
    the ``SlotKey`` of the corresponding entry in ``Spec.resolved.slots`` (join on axis and
    ``property.ref``).
    """

    key: SlotKey = Field(description="the slot identity this option set belongs to")
    options: list[SlotOption] = Field(
        description=(
            "candidate values, in taxonomy order; a dcid that failed to resolve a "
            "label is omitted rather than fabricated. For a level-scoped "
            "constraint slot (relation is populated), this is the bound value's "
            "children and parent (for a bound root, its children and every other "
            "root), or, when the slot is unbound, each "
            "classification root followed by its children -- never deeper levels. "
            "A counted menu omits every option its answer would serve nothing for "
            "except the slot's current value (see SlotOption.served_rows)"
        )
    )
    truncated: bool = Field(
        description=(
            "false when options is the full known set for this slot at its "
            "current level (a curated seed taxonomy, a level-scoped taxonomy "
            "sourced from the packaged code index, or an observed taxonomy from "
            "the same source, rather than the recalled candidates); true when "
            "options is only the values observed among the resolved candidates "
            "(bounded by the recall cut, not exhaustive)"
        )
    )
    source: SlotOptionsSource = Field(
        description=(
            "'seed' for a hand-curated family taxonomy (e.g. dev-finance scheme/"
            "purpose); 'observed' for the union of values seen on the confirmed "
            "StatVars or resolved query entities"
        )
    )


# ---------------------------------------------------------------------------
# EntityOptions: candidates for a directional entity axis
# ---------------------------------------------------------------------------


class EntityOptions(StrictModel):
    """Candidates for one directional entity axis.

    Parallel to SlotOptions, but for a directional *entity* (an observation
    endpoint) rather than a constraint-property value. A role-bearing flow may
    expose either its origin or destination axis here when that axis was left
    open. The entry's ``role`` is the routing key a client uses to promote a
    selected candidate into ``identity.entities`` on a dataset_resubmit entry.
    """

    entity_ref: str | None = Field(
        default=None,
        description=(
            "dcid of the currently-bound directional entity; joins to an entry "
            "in SentenceToken.entity_refs. None means this role axis is open -- "
            "no entity is bound on it, and `options` is this axis's own "
            "candidate member set rather than a set the current binding could "
            "be swapped among. An open role axis omits this key entirely on both "
            "the buffered body and the SSE entity_options event, rather than "
            "emitting null -- the opposite of SlotOption.coverage's own "
            "explicit-null convention on that same SSE channel"
        ),
    )
    role: RoleAxis = Field(
        description=(
            "the flow axis these candidates apply to; present even when entity_ref "
            "is omitted so a selected candidate can be posted under the same "
            "identity.entities role"
        )
    )
    options: list[SlotOption] = Field(
        description="candidate entities, reusing SlotOption for shape parity with slot_options"
    )
    truncated: bool = Field(
        description=(
            "true when options is capped or deliberately count-only, so it is not the "
            "complete known candidate set; false when every known candidate is listed"
        )
    )
    source: SlotOptionsSource = Field(
        description=(
            "'seed' for a hand-curated candidate set; 'observed' for candidates "
            "already confirmed while resolving this flow axis"
        )
    )


# ---------------------------------------------------------------------------
# FacetOptions: the candidate set a facet axis was chosen from
# ---------------------------------------------------------------------------


class FacetOptions(StrictModel):
    """The candidate set one facet axis (see ResolvedFacet) was chosen from:
    every distinct value seen on the resolved candidates' own observation
    facets, so a client can offer "what else could I ask for" alongside the
    served answer.

    Parallel to SlotOptions/EntityOptions, reusing SlotOption for shape
    parity. Unlike either, source is always "observed" -- a facet axis has
    no curated seed taxonomy -- and no option here ever carries a populated
    SlotOption.coverage: there is no separate confirm step for a facet
    option, the observed count already ranks the list.
    """

    # The open string keeps facet-axis additions additive in JSON Schema.
    axis: str = Field(description="the facet axis this option set belongs to (see FacetAxis)")
    options: list[SlotOption] = Field(
        description=(
            "candidate values seen on the resolved candidates' own unfiltered "
            "observation facets, ranked by in-window observation-point count "
            "descending then ref ascending; a dcid the graph can't label falls "
            "back to the dcid itself as its own label rather than being "
            "dropped -- every value here already came off a confirmed "
            "observation facet, unlike a curated seed candidate. On the "
            "measurementMethod and unit axes, an option with no value.ref stands "
            "for the facets that declare no value on the axis, as Data Commons "
            "omits the member; it is listed only beside a declared value, ranks "
            "last among equal counts, and is pinned by posting identity.facets."
            "<axis> as {state: 'value', ref: null}"
        )
    )
    truncated: bool = Field(
        description=(
            "true when options is bounded by SEARCH_MAX_OPTIONS rather than "
            "the full distinct set observed"
        )
    )
    source: SlotOptionsSource = Field(
        description="always 'observed' -- a facet axis has no curated seed taxonomy"
    )
