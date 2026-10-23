from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import (
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from dfo_platform.roles import FlowAccount

from .base import StrictModel
from .coverage import Coverage, Headline
from .entity import EntityGroup, ResolvedEntity
from .grammar import DataCommonsRef, GraphRef, SentenceToken, Shape, Slot, SourceRef, TimeWindow
from .identity import BaseIdentity
from .observations import SpecObservations
from .options import EntityOptions, FacetOptions, SlotOptions
from .statvar import ResolvedStatVars, SentenceFacts, StatVar
from .trace import ResolutionTrace
from .vocab import (
    SCHEMA_VERSION,
    Axis,
    CandidateBasis,
    Confidence,
    EntityRoleKey,
    FacetBasis,
    RoleAxis,
)


class ResolvedFacet(StrictModel):
    """One facet axis's resolved value on a Spec (see ResolvedSpec.facets):
    which value backed the answer's observations, and why.

    basis is always present. ref/label are present when one distinct value
    served every carried series; selection carries "per_series" in their
    place when the carried series select different values on this axis --
    never alongside ref/label. request_basis joins either shape, present
    only on the unit axis, only when unit was bound through the request form
    (currency/prices/base_year), one basis value per member the request
    named.

    Fields are flat and optional, validated by a model validator, rather
    than a discriminated ref/selection union. basis and request_basis are
    shared by both the single-ref and per-series shapes, so a union would
    duplicate them across arms for a mechanical gain (moving the
    ref/selection XOR to the type level) without expressing the
    request_basis-only-on-unit constraint, since that constraint is about
    which AXIS a ResolvedFacet lives under (see ResolvedSpec.facets), not
    its ref/selection shape. This is also response-only data, never
    deserialized from untrusted client bytes, so a runtime validator is
    the right tool for catching an illegal engine-side construction.
    """

    basis: FacetBasis = Field(
        description=(
            "'bound' when this axis was constrained by the request, "
            "'declared_default' when an unbound axis was completed from a "
            "declared deployment policy, 'fallback' when neither applies and "
            "the deployment policy supplied the value: facets holding a point in "
            "the requested years first, then DFO's facets for a variable both DFO "
            "and base Data Commons hold, then the most places or role tuples "
            "covered there, then the most requested years the span covers, then "
            "most observations, then the latest observation, then a deterministic "
            "tie-break"
        )
    )
    ref: str | None = Field(
        default=None,
        description=(
            "the axis's resolved dcid, present when one distinct value served "
            "every carried series; absent when selection is 'per_series'"
        ),
    )
    label: str | None = Field(
        default=None,
        description="best-effort display label for ref; present only alongside ref",
    )
    currency: str | None = Field(
        default=None,
        pattern=r"^[A-Z]{3}$",
        description=(
            "unit axis only, alongside ref: the ISO 4217 code of the currency ref is "
            "denominated in, from the graph's currency units (CurrencyUSD, USDollar and the "
            "base Data Commons Currency members such as INR); absent when ref is no "
            "currency. The sentence reads it: US dollars go to the byline's unit token, any "
            "other currency stays in the heading"
        ),
    )
    source: SourceRef | None = Field(
        default=None,
        description=(
            "provenance axis only, alongside ref: the organisation behind the "
            "provenance, from the origin catalogue"
        ),
    )
    data_commons: DataCommonsRef | None = Field(
        default=None,
        description=(
            "provenance axis only, alongside ref: the Data Commons the provenance's data comes from"
        ),
    )
    documentation: str | None = Field(
        default=None,
        description=(
            "provenance axis only, alongside ref: the Graph API route of the provenance's "
            "document, GET /api/graph/documentation/{dcid}"
        ),
    )
    account: FlowAccount | None = Field(
        default=None,
        description=(
            "provenance axis only, alongside ref: whose account of a flow the provenance "
            "gives, when ref is a dataset the code index declares; 'provider' when it "
            "reports flows as their providers record them"
        ),
    )
    selection: Literal["per_series"] | None = Field(
        default=None,
        description=(
            "'per_series' when the carried series select different values on "
            "this axis; present in place of ref/label, never alongside them"
        ),
    )
    request_basis: dict[str, FacetBasis] | None = Field(
        default=None,
        description=(
            "per-member basis (currency/prices/base_year) for a unit axis "
            "bound through the request form; present only on the unit axis, "
            "alongside either ref/label or selection"
        ),
    )

    @model_validator(mode="after")
    def _ref_label_selection_shape(self) -> ResolvedFacet:
        if self.label is not None and self.ref is None:
            raise ValueError("label is only meaningful alongside ref")
        if (
            self.source is not None
            or self.data_commons is not None
            or self.documentation is not None
            or self.currency is not None
            or self.account is not None
        ) and self.ref is None:
            raise ValueError(
                "source, data_commons, documentation, currency and account are only "
                "meaningful alongside ref"
            )
        if (self.ref is not None) == (self.selection is not None):
            raise ValueError("exactly one of ref or selection must be present")
        return self


class ResolvedScope(StrictModel):
    """The full member set of an entity axis this Spec's request left unnamed
    (see ResolvedSpec.scope).

    ``axis`` is observationAbout when place defaulting expanded a place-less
    request, and origin or destination when directional-role defaulting
    expanded a role-bearing request. ``members`` lists the entities on that axis
    the answer's read plan serves, the ones a coverage breadth dimension for the
    same axis counts. members is the single-element ["Earth"] list when the query
    resolved as a worldwide total.
    The response-level diagnostics may include place_scope_defaulted or
    role_scope_defaulted for such resolution, but those warnings are not a
    per-Spec one-to-one representation of this field. The scope lives outside
    BaseIdentity, the spec hash, and resolved.entities: a resubmit of the
    same definition rediscovers it from the live graph.
    """

    axis: EntityRoleKey = Field(
        description="the role-keyed axis the request left unnamed and the engine defaulted"
    )
    members: list[ResolvedEntity] = Field(
        description=(
            "every entity on axis this Spec's read plan serves, sorted by ref; each carries "
            "role == axis and a best-effort label/entity_type from a batched graph read, "
            "None when the graph has neither"
        )
    )


class SpecDefinition(StrictModel):
    """The canonical definition a Spec was resolved from: `{schema_version,
    spec_id, identity}`. This is the form the registry stores and permalinks
    carry, so it is self-contained. A definition lifted out of its Spec still
    names which spec it is.

    `identity` is the identity-core projection (see `BaseIdentity`) this
    Spec's spec_id was minted from, so hashing `identity` again reproduces
    `spec_id` by construction.

    `spec_id` is carried here as well as on `Spec.spec_id` deliberately: the
    duplication is what makes the definition independently meaningful once
    detached from the response that delivered it. The one exception is an
    `unsupported` response's definition, which omits it: the request was
    valid and understood but never became a resolvable construction, so no
    spec_id was ever legitimately minted for it.
    """

    schema_version: Literal[1] = Field(
        default=SCHEMA_VERSION,
        description="contract schema version this definition was published under; exactly 1",
    )
    spec_id: str = Field(
        description=(
            "the spec_id minted from `identity`; equals the sibling Spec.spec_id, "
            "and is what a detached definition (registry row, permalink) is keyed by"
        )
    )
    identity: BaseIdentity = Field(
        description=(
            "the identity-core projection this Spec's spec_id was minted from, "
            "serialised once and never re-derived; post it back byte-for-byte as "
            "ResubmitIdentityPayload.identity to edit this Spec. Directional edits "
            "change identity.entities.origin or identity.entities.destination directly"
        )
    )


class MeasureAxisValue(StrictModel):
    """One measure axis and the value a measure partition holds on it (see
    ResolvedSpec.measure_axis_values)."""

    axis: str = Field(
        description=(
            "the declared measure-axis identifier (code_index.types.MeasureAxis.axis), "
            "e.g. 'measuredProperty' or a constraint-property dcid such as "
            "'financialFlowType'"
        )
    )
    value: GraphRef = Field(
        description="the value's dcid and the measure index's label for it, e.g. Gross"
    )


class SpecAlternativeMeasure(StrictModel):
    """One other measure this Spec could equally answer with (see
    ResolvedSpec.alternative_measures): either another populated partition
    from the same family as the one this Spec's own served rows exhibit, or
    a genuinely distinct also_plausible candidate whose only difference from
    this Spec is measured_property (measurement_denominator held equal),
    folded in here instead of left as a second candidates entry. Carries its
    own identity and a label that disambiguates it.

    `definition` is minted by editing this Spec's own already-minted
    identity onto this partition's measure axes (retrieval.bound_values.
    mint_sibling_identity), an index-scan projection, not a resolution.
    Carries identity and a label only: no stat_vars, no coverage, no trace.
    ResolvedSpec's own required fields have no defaults, so producing one as
    a Spec would mean fabricating a resolution that never ran.

    Post `definition` back as `identity` on a dataset_resubmit identity entry to
    switch onto this measure: the identity's own what/how slots (this
    partition's per-partition-declared axes included, e.g. CRS cooperation
    modality) name the family the packaged code index's own election
    recovers members from, the same recovery a resubmit of this Spec's own
    identity uses.

    `axis` names the sentence token whose menu offers the entry. A partition
    that differs from this Spec's only in price basis (`measurementQualifier`
    `Nominal` or `RealValue`) is a unit option, axis "unit", offered by the
    byline's unit token (binding_ref "unit:"). Every other entry is a measure
    option, axis "what". The measure menu holds one entry per measure
    whatever its price basis, so no two "what" entries differ only in price
    basis.
    """

    definition: SpecDefinition = Field(
        description="the alternative measure's own {schema_version, spec_id, identity}"
    )
    label: str = Field(
        description=(
            "readable text for this option, distinct from every other entry's on the "
            "same axis. A 'what' entry names the measure without its price basis "
            "('net disbursements'), and differs from measure_label; a 'unit' entry "
            "names the prices it switches to ('constant prices')"
        )
    )
    axis: Literal["what", "unit"] = Field(
        description=(
            "the sentence token whose menu offers this entry: 'what', the measure "
            "noun, or 'unit', the byline's unit token (SentenceToken.axis 'unit', "
            "binding_ref 'unit:'), for a partition that differs from this Spec's "
            "only in price basis"
        )
    )
    measure_axis_values: list[MeasureAxisValue] = Field(
        description=(
            "every axis this measure's partition holds a value on, with that value, in "
            "measure-axis declaration order; the same projection as "
            "ResolvedSpec.measure_axis_values"
        )
    )
    served_rows: int | None = Field(
        default=None,
        ge=1,
        description=(
            "the rows a resubmit of `definition` would write at this Spec's bound roles, "
            "in the terms of download.rows: one per series and year with a point inside "
            "the requested window, or under 'latest' one per series holding its StatVar's "
            "latest year, a series being a served StatVar at one entity, or donor and "
            "recipient pair. A Spec that merges several answers counts at every entity its "
            "answers bind. A StatVar the coverage artifact does not hold adds no rows, and "
            "a measure that would serve none, including one whose StatVars are all missing "
            "from the artifact, is not listed. None where the count needs graph reads (a "
            "subject-place family, or a family the code index does not decide), or where "
            "the merged answers read different years or coverage"
        ),
    )
    served_rows_exact: bool | None = Field(
        default=None,
        description=(
            "true when served_rows equals the download.rows of a resubmit of "
            "`definition`; false when it is an upper bound: a counted StatVar whose series "
            "sit in more than one facet group (provenance, method and unit) in the window; "
            "an answer read in bulk whose measure's StatVars sit in more than one facet; "
            "under 'latest', a StatVar with a facet observed more often than yearly; an "
            "answer that pins a facet value; or a unit request whose conversion cannot be "
            "read as download.rows_exact describes, or that converts a counted series from "
            "a currency the answer does not serve. Null exactly when served_rows is null"
        ),
    )


class RelatedMeasure(StrictModel):
    """One other StatVar from the StatVarGroup this Spec's one StatVar belongs to (see
    ResolvedSpec.related_measures), with data at the Spec's places.

    `definition` is minted from the member's own five-tuple and constraint values, with this
    Spec's entities and requested window and no facets, since a unit or source this Spec
    pins belongs to its own measure. Post `definition` back as `identity` on a
    dataset_resubmit identity entry to switch onto this measure.
    """

    stat_var: GraphRef = Field(
        description=(
            "the StatVar this measure reads, labelled with its graph name, or its dcid when it "
            "has none"
        )
    )
    definition: SpecDefinition = Field(
        description="the measure's own {schema_version, spec_id, identity}"
    )


class OpenAxisValueCount(StrictModel):
    """One declared value on an open constraint axis and how many served members
    declare it (see OpenConstraintAxisFact.values)."""

    value: GraphRef = Field(description="the declared value")
    count: int = Field(description="how many distinct served members declare this value")


class OpenConstraintAxisFact(StrictModel):
    """One constraint axis the query left open, and what this Spec's own
    served StatVar members exhibit on it (see ResolvedSpec.open_axis_facts):
    whether they agree on a single declared value or differ, what that
    agreed value is when one exists, how many distinct values the axis
    actually served, and how many declare none at all.
    """

    property: str = Field(description="the constraint property dcid this fact is about")
    varies: bool = Field(
        description=(
            "False when every served member that declares a value for this axis "
            "declares the same one (including when none of them declare one at all, "
            "since declaring nothing is itself an agreeing value here, the same rule "
            "resolved.alternative_measures' own observed-partition read applies); "
            "True when they differ"
        )
    )
    value: GraphRef | None = Field(
        default=None,
        description=(
            "the single value every served member that declares one for this axis "
            "agrees on; present only when varies is False and at least one member "
            "declares a value, absent when varies is True or every member declares "
            "nothing"
        ),
    )
    distinct_count: int = Field(
        description=(
            "how many distinct declared values the served members actually exhibit "
            "for this axis, not counting a member that declares nothing. 0 when "
            "every member declares nothing, 1 when varies is False and value is set. "
            "Counted over the served members' own declarations, independent of this "
            "response's observation window: a query's applied when-slot narrows which "
            "observation points are returned, not which constraint values a member "
            "declares, so this count is never scoped to that window the way an "
            "observation-count field on this contract is"
        )
    )
    undeclared_count: int = Field(
        description=(
            "how many served members declare no value at all for this axis. A member "
            "whose arc names the axis but never resolved to a dcid counts here too, the "
            "same as a member that never asserted the axis at all"
        )
    )
    values: list[OpenAxisValueCount] = Field(
        default_factory=list,
        description=(
            "every declared value with the number of served members declaring it, "
            "highest count first, then by dcid. Each served StatVar counts once, also "
            "after a merge of answers that serve it more than once, so the counts plus "
            "undeclared_count sum to the distinct served members. Labels as for value"
        ),
    )


class RoleAxisFact(StrictModel):
    """One directional role axis (origin or destination) a role-bearing
    Spec's query left open, and what this Spec's own confirmation already
    knows about its member scope (see ResolvedSpec.role_axis_facts).

    Unlike OpenConstraintAxisFact, a role axis has no declared value of its
    own that could vary across served members -- the axis is a place a flow
    entity binds, not a constraint members assert -- so this carries no
    varies/undeclared_count of its own; it carries only what a role axis
    genuinely has: a population size and whether that population is known.
    """

    role: RoleAxis = Field(description="which flow axis this fact is about")
    member_count: int | None = Field(
        description=(
            "how many entities this axis' open scope resolves to. None when the "
            "count could not be established without a read this response chose not "
            "to make (e.g. a discovered opposite axis whose own role index was "
            "omitted after a cap trip); the fact still publishes in that case, "
            "naming the axis as open with an explicit unknown rather than staying "
            "silent about it"
        )
    )
    members_available: bool = Field(
        description=(
            "whether the member list this count summarises is itself known and "
            "could be enumerated (e.g. through entity_options or the options door), "
            "not just counted. False exactly when member_count is None"
        )
    )


class DownloadLicence(StrictModel):
    """The licence of the provenance a Spec's file cites. Its terms page is on the
    provenance's source page."""

    label: str = Field(
        description=(
            "the licence's name: the `name` of the licenseType node the provenances of "
            "the provenance's source-index entry agree on"
        )
    )


class DownloadFacts(StrictModel):
    """What a Datasets mint of this Spec's definition writes in its default file (no
    grouping), known before the first download. Rows and series count what the answer's
    read plan serves over the identity's whole window: every year for no window, the
    range for a range, and each series' latest year for `latest`."""

    rows: int | None = Field(
        default=None,
        description=(
            "the rows the definition's series serve: one per series and date, totals "
            "included. A long-layout CSV or XLSX keeps the rows where a role the Datasets "
            "options document lists in long_layout_totals holds a total out of its main "
            "table, so that table can hold fewer. Null when it could understate: a served "
            "member the coverage artifact does not hold"
        ),
    )
    rows_exact: bool = Field(
        description=(
            "true when rows equals the mint's coverage.rows. Under a unit request rows "
            "counts only the rows the conversion keeps: at constant prices each entity's "
            "rows in the years its deflator holds, none when it lacks the base year. False "
            "when rows is an upper bound: a unit request whose conversion cannot be read "
            "from the loaded conversion tables (none loaded, served facets under several "
            "recipes or in a unit that is no currency, or no single deflator role), or a "
            "series whose facets carry no complete date list. False when rows is null"
        )
    )
    series: int | None = Field(
        default=None,
        description=(
            "the series in the file: distinct (StatVar, entities) with a point, the "
            "mint's series_count. Null when rows is null"
        ),
    )
    window: TimeWindow | None = Field(
        default=None,
        description="the first and last years of the file's points; null when rows is null",
    )
    licence: DownloadLicence | None = Field(
        default=None,
        description=(
            "the licence of the one provenance the answer serves, from the origin "
            "catalogue; null when it serves several, or the catalogue names no single "
            "licence for it"
        ),
    )


class ConversionCurrency(StrictModel):
    """One currency a unit request can convert this answer to, with the years each price
    basis converts. A request whose tables leave out some of the years the answer serves
    shows the years stated here, and departs `year_unavailable` naming the table."""

    code: str = Field(description="the ISO 4217 code a unit request names the currency by")
    noun: str = Field(
        description=(
            "the plural English noun the sentence's unit token names the currency by "
            "('euros', 'Guatemalan quetzales')"
        )
    )
    current_years: TimeWindow | None = Field(
        description=(
            "the first and last years a current-price request in this currency shows, the "
            "longest run of consecutive years, inside the years the answer serves "
            "(download.window), in which the recipe's exchange-rate table holds a rate for "
            "this currency and for the native currency of every facet the answer serves. "
            "It equals download.window when the conversion keeps every served year. With "
            "no download.window, the run the tables hold. Null when no year converts"
        )
    )
    constant_years: TimeWindow | None = Field(
        description=(
            "the first and last years a constant-price request in this currency shows, at "
            "any base year from base_year_min to base_year_max, the longest run of "
            "consecutive years, read as current_years is, in which the exchange-rate table "
            "holds a rate for the native currency of every facet the answer serves and the "
            "deflator table a value for at least one entity the answer serves on the "
            "recipe's deflator role, bound or open. The conversion leaves out an entity's "
            "rows in the years its deflator lacks, and the answer names them "
            "(deflator_missing_omitted). Null when no year converts or no base year exists"
        )
    )
    base_year_min: int | None = Field(
        description=(
            "the first base year a constant-price request in this currency converts at. "
            "With base_year_max it spans the run of consecutive years, ending at the latest, in "
            "which the deflator table holds an actual value, the exchange-rate table a rate "
            "for the currency, and the deflator table a value for at least one entity the "
            "answer serves on the recipe's deflator role. The conversion leaves out every "
            "row of an entity whose deflator lacks the base year, and the answer names it "
            "(deflator_missing_omitted). Null when no such year exists"
        )
    )
    base_year_max: int | None = Field(
        description="the last base year a constant-price request in this currency converts at"
    )


class UnitConversion(StrictModel):
    """The conversions a unit request can ask of this answer: the currencies, years and base
    years the conversion recipe of its served members converts. A request posts one as the
    identity's `facets.unit` (`IdentityFacetValueRequest`)."""

    currencies: list[ConversionCurrency] = Field(
        description=(
            "each currency a unit request accepts that converts at least one served year "
            "at current prices, at a constant-price base year, or both, sorted by code"
        )
    )
    default_base_year: int = Field(
        description=(
            "the base year a constant-price request that names none completes to: the "
            "deflatorBaseYear every conversion table declares"
        )
    )
    role_currencies: dict[EntityRoleKey, str] = Field(
        default_factory=dict,
        description=(
            "the ISO 4217 code of the national currency (the graph's nationalCurrencyOf "
            "arc) of the one entity bound on each role, among `currencies`. A role bound "
            "to several entities, or to one no single currency names, is absent. A "
            "request may name the role ({'role': 'origin'}) or the code"
        ),
    )


class MembersNotStated(LookupError):
    """A Spec rebuilt from its wire form holds no served members, sentence facts or built
    shape label: the engine states them in `build_spec`, and the wire carries none. Reading
    one there is a programming error, raised so the reader does not answer from an empty
    list."""

    def __init__(self, fact: str) -> None:
        super().__init__(f"ResolvedSpec.{fact} is not stated on a Spec rebuilt from its wire form")


class ResolvedSpec(StrictModel):
    """What the graph held at resolution time: the engine's report of what it
    did with the ask. Meant for display and diagnostics, never hashed;
    `Spec.spec_id` depends only on `Spec.definition.identity`. Beside the
    contract's own `resolved` member list (`stat_vars`, `inapplicable_slots`,
    `unresolved_slots`, `entities`, `coverage`, `headline`, `facets`, `observations`,
    `rendered_sentence`/`sentence_plan`, `slot_options`/`entity_options`,
    `trace`, `confidence`), this also carries the engine's own supersets
    (`shape`, `slots`, `available_at_place`) that slot_options/sentence_plan
    join against and the eval harness scores.
    """

    shape: Shape = Field(description="the structural family chosen")
    slots: list[Slot] = Field(
        description=(
            "one entry per slot the shape defines, each in a binding state. Its what/how "
            "entries are the identity's slots, so a what/how property it leaves out is one "
            "no served member asserts"
        )
    )
    inapplicable_slots: list[str] = Field(
        default_factory=list,
        description=(
            "the slots whose property none of the resolved StatVars declares, listed "
            "by property dcid; empty when every slot applies. Joins to "
            "resolved.slots[*].key.property.ref. Applicability is graph "
            "state, so it is reported here and the slot itself reads 'unbound' in the "
            "definition either way. It is never a member of BaseIdentity, never hashed."
        ),
    )
    unresolved_slots: list[str] = Field(
        default_factory=list,
        description=(
            "the slots whose property is present but unresolved on at least one "
            "resolved StatVar (the graph asserted a value for this axis but could "
            "not resolve it to a dcid), listed by property dcid; empty when no such "
            "slot exists. Joins to resolved.slots[*].key.property.ref, and is "
            "disjoint from inapplicable_slots: a property is either never asserted by "
            "any resolved StatVar (inapplicable_slots), present but unreadable on one "
            "(this field), or neither (a genuinely open axis). The slot itself reads "
            "'unbound' in the definition in every one of these cases; it is never a "
            "member of BaseIdentity, never hashed."
        ),
    )
    stat_vars: ResolvedStatVars = Field(
        description="resolved StatVar(s) inside the region, and the scope they were drawn from"
    )
    entities: list[ResolvedEntity] = Field(
        description="resolved entities the data is about; may be empty if where is unbound"
    )
    entity_groups: dict[str, EntityGroup] | None = Field(
        default=None,
        description=(
            "the named group each role's entities stand for, keyed by the role key "
            "the entities above carry: each group whose members are exactly the "
            "entities definition.identity binds on that role. The heading names a "
            "role by its group and member count ('the 27 EU member states') in place "
            "of its members. None when no role was asked for as a group"
        ),
    )
    coverage: Coverage = Field(description="observation footprint")
    headline: Headline | None = Field(
        default=None,
        description=(
            "exact latest-year summary for a role-bearing answer (see Headline); "
            "None for a non-role-bearing Spec, which coverage above already answers exactly "
            "or approximately enough that no separate latest-year summary is computed"
        ),
    )
    download: DownloadFacts | None = Field(
        default=None,
        description=(
            "the rows, series, years and licence of the file a mint of this Spec writes "
            "(see DownloadFacts). None when the Spec has no read plan: a no-data answer, "
            "or same-shape parts merged into one Spec"
        ),
    )
    coverage_stamp: str | None = Field(
        default=None,
        description=(
            "the coverage artifact's own data-version stamp this Spec's role-bearing "
            "population and headline were confirmed against (series_coverage.CoverageManifest"
            ".stamp); None for a non-role-bearing Spec, which never consults coverage. "
            "The graph-derived analogue of this stamp (graph_generation) is not itself a "
            "wire field; this is coverage's own, published because coverage is a second, "
            "independently-versioned data source an answer can be stale against"
        ),
    )
    scope: ResolvedScope | None = Field(
        default=None,
        description=(
            "the full member set of an entity axis this Spec's resolution defaulted (see "
            "ResolvedScope): observationAbout for place defaulting, or origin/destination "
            "for directional-role defaulting. Response-level diagnostics may carry "
            "place_scope_defaulted or role_scope_defaulted, but warning presence is not an "
            "exact per-Spec equivalence; None when this Spec did not default an entity axis"
        ),
    )
    facets: dict[str, ResolvedFacet] = Field(
        default_factory=dict,
        description=(
            "the facet axis values of the series this Spec's read plan serves, the "
            "series its observations carry (see ResolvedFacet); the same whether or not "
            "observations are included or deferred to a trailing SSE event. Sparse, "
            "meaning an axis no served series carries is omitted, never a null "
            "placeholder. {} when the Spec serves no series"
        ),
    )
    observations: SpecObservations | None = Field(
        default=None,
        description=(
            "the observation values behind coverage above; None when not resolved for this "
            "Spec (e.g. no entity to probe, or response_options.include_observations is false), when "
            "ResolveOptions.include_observations opts into deferred SSE delivery (populated "
            "instead via a trailing observations event, mirroring slot_options/entity_options' "
            "own coverage-probe transport split), or for a candidates-status entry, "
            "nearest_real, or flagged_match spec (this iteration only populates it on a "
            "primary definite answer)."
        ),
    )
    slot_options: list[SlotOptions] = Field(
        default_factory=list,
        description=(
            "the candidate set each bound slot was chosen from, one entry per slot "
            "with a non-empty taxonomy; omits when/source and any slot with no "
            "observed or seeded values. spec_id does not depend on this field, so "
            "it never affects permalinks or dedup."
        ),
    )
    entity_options: list[EntityOptions] = Field(
        default_factory=list,
        description=(
            "candidate sets for directional origin or destination axes. An entry whose "
            "entity_ref is omitted describes an axis the role-bearing query left open; "
            "its role selects the identity.entities key for a resubmit, and the matching "
            "role_axis_facts entry reports that axis's member count. An entry with an "
            "entity_ref describes alternatives to a bound axis. Empty when the family has "
            "no directional entity-option concept. spec_id does not depend on this field."
        ),
    )
    facet_options: list[FacetOptions] = Field(
        default_factory=list,
        description=(
            "the candidate set each facet axis (provenance/measurementMethod/unit) was "
            "chosen from, one entry per axis with at least one value observed on the "
            "resolved candidates' unfiltered facets; an axis with no observed value is "
            "omitted, never an empty entry. Computed for every definite Spec and every "
            "candidate. Over SSE with response_options.include_option_coverage set, "
            "the entries arrive as trailing facet_options events and this list stays "
            "empty. Unlike slot_options, "
            "this has no bind-time baseline to fall back to. spec_id does not depend on "
            "this field."
        ),
    )
    rendered_sentence: str | None = Field(
        default=None,
        description=(
            "the editable confirmation sentence for this spec, present on every Spec "
            "a part shows; None on an equivalent_readings entry. It holds the heading and the "
            "byline only (see SentenceToken.segment): 'Germany's gross ODA "
            "disbursements to Kenya, 1973 to 2024. Current US dollars · OECD "
            "Creditor Reporting System (CRS).' It states what this Spec is and "
            "nothing about the request that produced it, so it reads the same after "
            "a dataset_resubmit and serves as the Spec's label. The part that "
            "carries this Spec states how it departs from the request "
            "(_PartBase.departures)"
        ),
    )
    sentence_plan: list[SentenceToken] | None = Field(
        default=None,
        description=(
            "the ordered token form of rendered_sentence, tying its tokens to "
            "definition parts; present under the same condition, None otherwise. "
            "Every token's segment is 'heading' or 'byline'"
        ),
    )
    trace: ResolutionTrace = Field(
        description="consolidated record of how the query was understood"
    )
    confidence: Confidence | None = Field(
        default=None,
        description=(
            "qualitative, relative retrieval confidence for this interpretation; None when not "
            "assessed. Bands are heuristic and may be retuned without a schema-version bump."
        ),
    )
    basis: CandidateBasis | None = Field(
        default=None,
        description=(
            "why the part offers this Spec as one of its candidates: 'structural', a reading "
            "that differs from the elected one only on a stat type, price basis or "
            "denominator the request leaves open; 'nominated', a reading the model named as "
            "a genuine reading of the request's phrase; 'twin', an equivalent reading of the "
            "elected one, offered where the elected one holds no data; 'direction', the same "
            "reading with the other direction for a place the request names without one, "
            "which a client shows as a direction prompt; 'breakdown', the same measure broken "
            "down by another set of slots, offered when its family holds no total and the "
            "request names no breakdown that serves a member. The elected reading, listed first, "
            "carries none. None on a definite answer and on every Spec outside "
            "CandidateSet.specs."
        ),
    )
    available_at_place: bool | None = Field(
        default=None,
        description=(
            "whether this spec has lifetime availability at the resolved place, independent of "
            "the coverage window. True = confirmed at place; False = the variable matched but "
            "has no observation here (the flagged_match case); None = not assessed (no place "
            "entity, or a named family such as dev-finance). CoverageExact.has_data may be "
            "False for a requested window even when this field is True."
        ),
    )
    equivalent_readings: list[Spec] | None = Field(
        default=None,
        description=(
            "other StatVar(s) confirmed to encode the same real-world indicator as this "
            "Spec. Each entry is a full Spec carrying identity and the material needed to "
            "re-resolve onto it: its definition.identity, spec_id, and resolved "
            "StatVar dcids provide everything ResubmitIdentityPayload requires. "
            "Its rendered_sentence, sentence_plan, facet_options, and observations come "
            "from re-resolving onto it on a dataset_resubmit identity entry, not "
            "from this response. This "
            "Spec's own answer stands on its own regardless. Switching to a named entry "
            "is optional, offered for its source vocabulary's own coverage. None when no "
            "such encoding is known. Never a member of BaseIdentity, never hashed, and "
            "never populated on its own entries in turn. An entry's own "
            "equivalent_readings is always empty."
        ),
    )
    skills: list[str] | None = Field(
        default=None,
        description=(
            "ids of the skills this Spec's bound graph nodes and selected provenance call "
            "for, matched against the shared skills index, most specific first: skills matched "
            "through a graph node by hop, then breadth, and skills matched only through the "
            "provenance last; present under the include_skills option. None = the request did "
            "not ask; [] = it asked and nothing matched."
        ),
    )
    alternative_measures: list[SpecAlternativeMeasure] | None = Field(
        default=None,
        description=(
            "every other measure this Spec could equally answer with -- either another "
            "populated partition from the same family as the one this Spec's own "
            "served rows exhibit (retrieval.bound_values.observed_measure_family), or "
            "a genuinely distinct also_plausible candidate whose only difference from "
            "this Spec is measured_property (measurement_denominator held equal), "
            "folded in here instead of left as a second candidates entry -- each with "
            "its own identity, a disambiguating label, and the axis of the sentence "
            "token whose menu offers it: 'what' entries are the measure switch beside "
            "measure_label below, 'unit' entries the price-basis options of the "
            "byline's unit token (binding_ref 'unit:'). A partition or folded "
            "candidate with no authored label is left out. None when this Spec's shape "
            "carries no measure family at all and no measure-sibling candidate was "
            "folded in, or its served rows span more than one partition; a served "
            "partition with every sibling unlabelled, or with none, still publishes "
            "[]. Never a member of BaseIdentity, never hashed."
        ),
    )
    measure_label: str | None = Field(
        default=None,
        description=(
            "the readable label for the measure partition this Spec's own served rows "
            "exhibit, composed the same way as each 'what' alternative_measures "
            "entry's label: what a switch built from those entries would show as "
            "currently selected. "
            "It names no price basis ('gross disbursements', not 'gross "
            "disbursements (nominal)'). Current or constant prices belong to the "
            "unit, which the byline's unit token states. None under the same "
            "condition alternative_measures is None"
        ),
    )
    measure_label_axes: list[str] = Field(
        default_factory=list,
        description=(
            "the declared measure-axis identifiers (code_index.types.MeasureAxis.axis, "
            "e.g. 'measuredProperty', 'measurementQualifier', or a constraint-property "
            "dcid such as a financial-flow-type or netting-basis axis) that measure_label "
            "was composed from. Every axis the served partition carries a value on is "
            "included here, whether or not its value appears in measure_label's own "
            "text. A renderer uses this to keep a bound how-slot sharing one of these "
            "property dcids from repeating what measure_label already said. Always [] "
            "when measure_label is None."
        ),
    )
    measure_axis_values: list[MeasureAxisValue] = Field(
        default_factory=list,
        description=(
            "each axis in measure_label_axes with the value the served partition holds "
            "on it, in the same order, including values the query never named (a "
            "declared default such as Gross or Nominal). Labels come from the measure "
            "index. Always [] when measure_label is None. Never a member of "
            "BaseIdentity, never hashed."
        ),
    )
    related_group: GraphRef | None = Field(
        default=None,
        description=(
            "the StatVarGroup related_measures come from: the smallest group this Spec's one "
            "StatVar belongs to that holds 2 to 40 members, ties broken by group dcid. None "
            "when related_measures is None or []"
        ),
    )
    related_measures: list[RelatedMeasure] | None = Field(
        default=None,
        description=(
            "up to 5 other members of related_group with data at every place this Spec "
            "names (and in its requested years when it names a range), ordered by the years "
            "they share with this Spec's StatVar there, then by dcid. Only a definite Spec "
            "that serves exactly one StatVar, has no measure family (measure_label is None) "
            "and names places with no roles carries a list; [] when no group qualifies or no "
            "other member has data there. None for any other Spec, and when the graph reads "
            "failed or ran past their deadline. Never a member of BaseIdentity, never hashed."
        ),
    )
    open_axis_facts: list[OpenConstraintAxisFact] | None = Field(
        default=None,
        description=(
            "one entry per constraint axis this Spec's own served StatVar members "
            "declare (an arc naming the property, resolved or not) that the query "
            "left unbound, each naming the axis's property and what those members "
            "exhibit on it: whether they agree on a value "
            "(varies=False) or differ (varies=True), and undeclared_count for how many "
            "declare none at all. None when this Spec's served members declare no "
            "constraint property the query left open at all. This is distinct from an open "
            "axis every member declares nothing for, which still publishes its own "
            "entry (varies=False, undeclared_count equal to the served member count). "
            "Never a member of BaseIdentity, never hashed."
        ),
    )
    role_axis_facts: list[RoleAxisFact] | None = Field(
        default=None,
        description=(
            "one entry per directional role axis (origin/destination) this "
            "role-bearing Spec's query left open, naming the axis and the member "
            "population its own confirmation already resolved (see RoleAxisFact). "
            "None when this Spec is not role-bearing, or is role-bearing but every "
            "role axis its shape declares carries a bound entity. Never a member of "
            "BaseIdentity, never hashed."
        ),
    )
    unit_conversion: UnitConversion | None = Field(
        default=None,
        description=(
            "the currencies and price bases a unit request can convert this answer to. "
            "Null when the served members' unit is no currency, no conversion recipe "
            "converts them (a stock, a series not in nominal prices, or a provenance that "
            "declares no recipe), the recipe's tables lack the native currency's rate in a "
            "served year, so nothing converts, or no conversion catalogue is loaded, which "
            "the part's unit_conversion_unavailable warning states. Never a member of "
            "BaseIdentity, never hashed."
        ),
    )
    # The named entities the answer's read plan serves no point at, each with the role it
    # was asked on (`ResolvedSpec.unserved_entities`), and the years the served series
    # hold over every date (`ResolvedSpec.data_years`). The part states them in its
    # departures, so they reach the wire in `departures` alone, and a response read back
    # from the cache holds neither. Equality compares the wire state (`__eq__`).
    _unserved_entities: tuple[tuple[str, str], ...] = PrivateAttr(default=())
    _data_years: tuple[int, int] | None = PrivateAttr(default=None)
    # Whether the served series include points continuing a series past its last period
    # (`ResolvedSpec.continued`), which the part states in a `modelled_values` departure.
    _continued: bool = PrivateAttr(default=False)
    # The sets the request named whose every member a role binds, each with its role and
    # member dcids (`ResolvedSpec.named_sets`). The part's departures count a set's members
    # without values against it. `entity_groups` carries the one set a role's entities equal.
    _named_sets: tuple[tuple[str, EntityGroup, frozenset[str]], ...] = PrivateAttr(default=())
    # The members of those sets the request also named on their own, by role
    # (`ResolvedSpec.named_alone`).
    _named_alone: frozenset[tuple[str, str]] = PrivateAttr(default=frozenset())
    # The total a dataset publishes for the group each role reads as its members, by role
    # (`ResolvedSpec.group_totals`). `entity_groups` names a total only when the role binds
    # it, and a group with no node of its own carries no dcid to look one up by.
    _group_totals: tuple[tuple[str, str], ...] = PrivateAttr(default=())
    # The members the answer serves and what the renderer reads of them, set by
    # `engine.spec_build.build_spec`. A Spec rebuilt from its wire form holds neither, and
    # reading one there raises `MembersNotStated`.
    _members: tuple[StatVar, ...] | None = PrivateAttr(default=None)
    _sentence_facts: SentenceFacts | None = PrivateAttr(default=None)
    # Facts the engine reads and the wire does not carry: the code pairs the measure label
    # names, which the sentence renderer keeps silent, and the two versions a same-shape
    # merge requires its members to share (`same_shape._check_merge_gates`).
    _measure_label_named_values: tuple[tuple[str, str], ...] = PrivateAttr(default=())
    _unit_lexicon_version: str | None = PrivateAttr(default=None)
    _declaration_version: str | None = PrivateAttr(default=None)
    # The label `shape` was built with before `engine.spec_build.build_spec` labelled it for
    # the members this Spec serves (`ResolvedSpec.built_shape_label`).
    _built_shape_label: str | None = PrivateAttr(default=None)

    @property
    def unserved_entities(self) -> tuple[tuple[str, str], ...]:
        """The `(role, dcid)` pairs among `entities` that the answer's read plan was asked
        to read and serves no point at on that role, in `entities` order
        (`read_plan.unserved_entities`). An entity on two roles can be unserved on one and
        served on the other. The part names them in an `entity_unavailable` departure
        (`engine.departures.spec_departure_facts`)."""
        return self._unserved_entities

    @property
    def named_sets(self) -> tuple[tuple[str, EntityGroup, frozenset[str]], ...]:
        """The sets the request named whose every member a role of `entities` binds, as
        `(role, group, member dcids)` (`engine.spec_build.role_named_sets`). A role that
        binds a set's members beside other places carries no `entity_groups` entry and
        still names the set here, so an `entity_unavailable` departure counts its members
        without values against the set (`engine.departures.unavailable_entity_facts`). A
        set the role binds at its own node is served there and is left out."""
        return self._named_sets

    @property
    def named_alone(self) -> frozenset[tuple[str, str]]:
        """The members of `named_sets` the request also named on their own, as `(role,
        dcid)` on the role key it named each on. On that role a departure names such a
        member as the request did and counts only the rest under its set
        (`engine.departures.unavailable_entity_facts`)."""
        return self._named_alone

    @property
    def group_totals(self) -> dict[str, str]:
        """The total a dataset publishes for the group each role reads as its members, keyed
        by role (`engine.spec_build.role_group_totals`). The departure saying what the members
        leave out reads the group's residuals from it (`engine.departures`)."""
        return dict(self._group_totals)

    @property
    def data_years(self) -> tuple[int, int] | None:
        """The first and last years the served series hold over every date, unclipped by
        the requested window (`read_plan.data_years`), `None` when unknown. Each StatVar's
        `data_date_range` is clipped to the window; a `year_unavailable` departure states
        this span as where the data starts and ends."""
        return self._data_years

    @property
    def continued(self) -> bool:
        """Whether the served series include points from a facet continuing a series past
        its last period, the same source's estimates (`read_plan.serves_continuation`).
        Coverage counts those points, and a `modelled_values` departure marks them
        (`engine.departures.spec_departure_facts`), with or without observations."""
        return self._continued

    @property
    def members(self) -> tuple[StatVar, ...]:
        """Every StatVar the answer serves, in served order, uncapped.

        Raises:
            MembersNotStated: the Spec was rebuilt from its wire form.
        """
        if self._members is None:
            raise MembersNotStated("members")
        return self._members

    @property
    def sentence_facts(self) -> SentenceFacts:
        """What the sentence renderer reads of the served members (`SentenceFacts.of`).

        Raises:
            MembersNotStated: the Spec was rebuilt from its wire form without the facts.
        """
        if self._sentence_facts is None:
            raise MembersNotStated("sentence_facts")
        return self._sentence_facts

    @property
    def measure_label_named_values(self) -> tuple[tuple[str, str], ...]:
        """The `(constraint property dcid, code dcid)` pairs outside the measure partition
        that `measure_label`'s authored text names, such as `(dac2aAidType,
        DAC2aAidType106)` for 'imputed multilateral ODA'. The renderer keeps a how slot
        carrying one of these values silent, since the measure label already says it.
        Empty when `measure_label` is None."""
        return self._measure_label_named_values

    @property
    def unit_lexicon_version(self) -> str | None:
        """The unit binder's lexicon version (`engine.unit.bind.LEXICON_VERSION`) that
        produced this Spec's unit facet; None when no unit request was bound."""
        return self._unit_lexicon_version

    @property
    def declaration_version(self) -> str | None:
        """The packaged code index's `content_hash` when the index decided this Spec's
        membership or declares a bound slot's code system
        (`retrieval.bound_values.declaration_version`); None otherwise."""
        return self._declaration_version

    @property
    def built_shape_label(self) -> str:
        """The label `shape` came to `engine.spec_build.build_spec` with. `shape.label` is
        that label composed for the members this Spec serves, so a Spec built over other
        members of the same shape (`same_shape.collapse_same_shape`) labels them from this
        one.

        Raises:
            MembersNotStated: the Spec was rebuilt from its wire form.
        """
        if self._built_shape_label is None:
            raise MembersNotStated("built_shape_label")
        return self._built_shape_label

    def __eq__(self, other: object) -> bool:
        """Equal when the wire state is: the private read facts above are inputs to the
        part's departures, which a cached response carries already written."""
        if not isinstance(other, ResolvedSpec):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def stating_unserved(self, entities: tuple[tuple[str, str], ...]) -> ResolvedSpec:
        """A copy whose `unserved_entities` are the `(role, dcid)` pairs `entities`."""
        stated = self.model_copy()
        stated._unserved_entities = entities
        return stated

    def stating_named_sets(
        self,
        named_sets: tuple[tuple[str, EntityGroup, frozenset[str]], ...],
        named_alone: frozenset[tuple[str, str]],
    ) -> ResolvedSpec:
        """A copy whose `named_sets` are `named_sets` and whose `named_alone` are
        `named_alone`."""
        stated = self.model_copy()
        stated._named_sets = named_sets
        stated._named_alone = named_alone
        return stated

    def stating_group_totals(self, group_totals: tuple[tuple[str, str], ...]) -> ResolvedSpec:
        """A copy whose `group_totals` are `group_totals`."""
        stated = self.model_copy()
        stated._group_totals = group_totals
        return stated

    def stating_data_years(self, years: tuple[int, int] | None) -> ResolvedSpec:
        """A copy whose `data_years` are `years`."""
        stated = self.model_copy()
        stated._data_years = years
        return stated

    def stating_continued(self, continued: bool) -> ResolvedSpec:
        """A copy whose `continued` is `continued`."""
        stated = self.model_copy()
        stated._continued = continued
        return stated

    def stating_members(self, members: Sequence[StatVar]) -> ResolvedSpec:
        """A copy whose `members` are `members` and whose `sentence_facts` are theirs."""
        stated = self.model_copy()
        stated._members = tuple(members)
        stated._sentence_facts = SentenceFacts.of(members)
        return stated

    def stating_sentence_facts(self, facts: SentenceFacts) -> ResolvedSpec:
        """A copy whose `sentence_facts` are `facts`, for a Spec rebuilt from its wire form
        that the renderer replays (the sentence golden). Its `members` stay unstated."""
        stated = self.model_copy()
        stated._sentence_facts = facts
        return stated

    def stating_measure_label_named_values(
        self, named_values: tuple[tuple[str, str], ...]
    ) -> ResolvedSpec:
        """A copy whose `measure_label_named_values` are `named_values`."""
        stated = self.model_copy()
        stated._measure_label_named_values = named_values
        return stated

    def stating_unit_lexicon_version(self, version: str | None) -> ResolvedSpec:
        """A copy whose `unit_lexicon_version` is `version`."""
        stated = self.model_copy()
        stated._unit_lexicon_version = version
        return stated

    def stating_declaration_version(self, version: str | None) -> ResolvedSpec:
        """A copy whose `declaration_version` is `version`."""
        stated = self.model_copy()
        stated._declaration_version = version
        return stated

    def stating_built_shape_label(self, label: str) -> ResolvedSpec:
        """A copy whose `built_shape_label` is `label`."""
        stated = self.model_copy()
        stated._built_shape_label = label
        return stated

    @field_validator("facets")
    @classmethod
    def _request_basis_only_on_unit_axis(
        cls, value: dict[str, ResolvedFacet]
    ) -> dict[str, ResolvedFacet]:
        for axis, facet in value.items():
            if facet.request_basis is not None and axis != "unit":
                raise ValueError(
                    f"request_basis is only meaningful on the unit axis, got axis {axis!r}"
                )
        return value

    @field_validator("equivalent_readings")
    @classmethod
    def _equivalent_readings_entries_do_not_nest(
        cls, value: list[Spec] | None
    ) -> list[Spec] | None:
        if value:
            for twin in value:
                if twin.resolved.equivalent_readings:
                    raise ValueError(
                        "equivalent_readings entries must not carry their own "
                        "equivalent_readings -- twin annotations do not nest"
                    )
        return value


class Spec(StrictModel):
    """The unit of meaning: one named region of the graph.

    Reused as the definite answer, each candidate, and NoData's own
    flagged_match/alternate_match (a plain Spec on both). See NoData's own
    docstring for the failed-gate code each carries, NoData.gate, kept
    structurally outside every identity path. NearestRealSpec (below)
    subclasses Spec for NoData.nearest_real, adding the relaxation each
    suggestion carries. That stays structurally outside every identity path
    the same way (spec_id is computed from bindings, never from the model),
    so it cannot perturb spec_id.
    """

    spec_id: str = Field(
        description=(
            "deterministic hash of (shape + canonical slot bindings); "
            "permalink-safe, dedup key, golden-comparison handle, cache key"
        )
    )
    definition: SpecDefinition = Field(
        description=(
            "the canonical definition this Spec was resolved from: {schema_version, "
            "identity}, the same shape Unsupported.definition uses. definition.identity "
            "is the identity-core projection this spec_id was minted from, serialised "
            "once and never re-derived; hashing it reproduces spec_id by construction. "
            "Post definition.identity back byte-for-byte as ResubmitIdentityPayload.identity "
            "to edit this Spec. Directional entities remain nested in identity.entities."
        )
    )
    resolved: ResolvedSpec = Field(
        description=(
            "the engine's report of what it did with the ask: stat_vars, entities, "
            "coverage, trace, and every other display/diagnostic field. Never hashed; "
            "see ResolvedSpec"
        )
    )


class Relaxation(StrictModel):
    """Which definition part a nearest_real suggestion relaxed to find data.

    Captured at the point a constraint binding is dropped and the resulting,
    widened request is re-confirmed against the live graph (see
    NearestRealSpec). Names the definition part relaxed (axis + the
    constraint property it carried) and the value(s) it dropped.
    """

    axis: Axis = Field(description="the slot axis whose binding was relaxed")
    property_dcid: str | None = Field(
        default=None,
        description="the constraint property dcid the relaxed slot carried, when one exists",
    )
    dropped_value_dcids: list[str] = Field(
        description="the taxonomy member dcid(s) the relaxed binding dropped"
    )


class NearestRealSpec(Spec):
    """A nearest_real suggestion: a full Spec found by relaxing one constraint
    binding and re-confirming the result against the live graph.

    Graph-re-confirmed only, never fabricated. See relaxation for which
    binding was dropped to find it.
    """

    relaxation: Relaxation = Field(
        description="the definition part this suggestion relaxed, and the value(s) it dropped"
    )
