from __future__ import annotations

from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from .base import StrictModel
from .grammar import DataCommonsRef, ProvenanceRef, SourceRef
from .vocab import EntityRoleKey, ObservationsOmissionReason, PartyKind, PreviewOrder

# ---------------------------------------------------------------------------
# SpecObservations: the observation values behind a Spec's coverage
# ---------------------------------------------------------------------------


class ObservationPoint(StrictModel):
    """One (date, value) pair from a resolved StatVar's observation series."""

    # `_serialize` writes `value` on every dump, so the served schema requires it.
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    date: str = Field(description="the observation date, e.g. '2018'")
    value: float | None = Field(
        default=None,
        description="the observed value; null when the graph reports a date with no value",
    )

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler):
        # Unannotated on purpose. Pydantic publishes a wrap serializer's return annotation as
        # the model's serialization schema, and `dict[str, Any]` reads as an open object.
        # Null marks a reported observation point without a measurement. Both serializers
        # use exclude_none, so restore it to preserve series length and gaps.
        data = handler(self)
        if "value" not in data:
            data["value"] = None
        return data


class SeriesFacetRef(StrictModel):
    """One axis's value on a single series' selected observation facet (see
    ObservationSeries.facet)."""

    ref: str = Field(description="the facet's dcid for this axis")
    label: str | None = Field(
        default=None,
        description="best-effort display label for ref; None when unresolved, never fabricated",
    )
    source: SourceRef | None = Field(
        default=None,
        description=(
            "provenance axis only: the organisation behind the provenance, from the "
            "origin catalogue; absent for a provenance the catalogue does not list"
        ),
    )
    data_commons: DataCommonsRef | None = Field(
        default=None,
        description=(
            "provenance axis only: the Data Commons the provenance's data comes "
            "from; absent for a provenance the catalogue does not list"
        ),
    )
    documentation: str | None = Field(
        default=None,
        description=(
            "provenance axis only: the Graph API route of the provenance's document, "
            "GET /api/graph/documentation/{dcid}"
        ),
    )


class SeriesStatVarRef(StrictModel):
    """A series' StatVar identity ({ref, label} -- mirrors SeriesFacetRef's shape)."""

    ref: str = Field(description="the StatVar dcid these points observe")
    label: str | None = Field(
        default=None,
        description=(
            "a row's own distinguishing label: composed from ref's own constraint "
            "values when it has any beyond its shape's shared five-tuple, else "
            "ref's plain resolved display name; None when unresolved, never fabricated"
        ),
    )


class SeriesEntityRef(StrictModel):
    """One entity in a series' role tuple (see ObservationSeries.entities)."""

    role: EntityRoleKey = Field(description="observationAbout, origin, or destination")
    ref: str = Field(description="the entity's dcid")
    label: str | None = Field(
        default=None,
        description="best-effort display label for ref; None when unresolved, never fabricated",
    )
    party_kind: PartyKind | None = Field(
        default=None,
        description=(
            "on an origin or destination entity, what the party is beside the others on its "
            "side: 'party' (a country, territory or organisation), 'total' (other parties' "
            "figures, so a sum "
            "across the role that counts it counts its members twice) or 'residual' (figures "
            "no other party holds), as the code index lists it. Absent for a party the index "
            "does not list, and on observationAbout"
        ),
    )


class SeriesAxisRef(StrictModel):
    """One constraint of a series' StatVar (see ObservationSeries.axes)."""

    property: str = Field(description="the constraint property's dcid, e.g. 'crsPurpose'")
    ref: str = Field(description="the dcid of the value the StatVar holds on property")
    label: str = Field(
        description=(
            "display label for ref: its graph label, else the dcid's tail split into "
            "lower-case words"
        )
    )


class SeriesConversionFallbackExample(StrictModel):
    """One entity a converted series' recipe substituted with its own
    fallback entity, because the substituted entity carried no row at all
    in the deflator table (see ObservationSeriesConversion.fallback_examples)."""

    entity: str = Field(description="the entity dcid absent from the deflator table")
    fallback_entity: str = Field(description="the recipe's fallback entity substituted for it")


class ObservationSeriesConversion(StrictModel):
    """What produced a converted series' points: the recipe and table
    versions, and the fallback/projected markers a caller reads off them.
    Modelled on materialise.convert.ConversionRecord, whose own docstring
    this field's members mirror. Omits ConversionRecord.native_currency:
    a caller trusting or reproducing the number needs what recipe and
    tables produced it, not what currency it started from -- facet.unit
    already names the currency the points are actually in.
    """

    recipe_id: str = Field(description="the ConversionRecipe.recipe_id that produced these points")
    table_provenance: SeriesFacetRef = Field(
        description=(
            "the Provenance node both the exchange-rate and (when present) deflator "
            "tables loaded under"
        )
    )
    exchange_rate_table_version: str = Field(description="the exchange-rate table's own version")
    deflator_table_version: str | None = Field(
        default=None,
        description="the deflator table's own version; absent for a current-prices conversion",
    )
    fallback_count: int = Field(
        description="rows whose deflator-role entity was substituted with the recipe's fallback entity"
    )
    fallback_examples: list[SeriesConversionFallbackExample] = Field(
        default_factory=list,
        description="a capped, sorted sample of the substituted (entity, fallback_entity) pairs",
    )
    projected_count: int = Field(
        description="rows whose rate/deflator term was sourced from a projected table row"
    )
    projected_examples: list[str] = Field(
        default_factory=list,
        description="a capped, sorted sample of the affected 'entity@year' keys",
    )


class ObservationSeries(StrictModel):
    """One (stat_var, role tuple) pair's observation points, in chronological order."""

    stat_var: SeriesStatVarRef = Field(description="the StatVar these points observe")
    entities: list[SeriesEntityRef] = Field(
        description=(
            "the role tuple this series observes -- one entry for a standard "
            "single-role series (role='observationAbout'), two for a dev-finance "
            "bilateral series (role='origin' and role='destination')"
        )
    )
    axes: list[SeriesAxisRef] = Field(
        default_factory=list,
        description=(
            "one entry per constraint the StatVar declares with a resolved value, ordered "
            "by property dcid; empty for a StatVar with no constraints. A client names "
            "each property from the Spec's slots"
        ),
    )
    points: list[ObservationPoint] = Field(
        description=(
            "chronological observation points from the series' lead observation facet "
            "(see SpecObservations), plus the points of the same source's estimates "
            "continuing the series past the lead's last period when it has any -- the "
            "`modelled_values` departure marks those -- filtered to the spec's requested "
            "window when one exists; the full "
            "in-window series when SpecObservations.truncated is false, else at most "
            "the latest points (bounded by the server's preview-points setting and its "
            "per-series row limit)"
        )
    )
    total_obs: int = Field(
        description=(
            "this series' SELECTED facet's in-window point count (before "
            "truncation) -- a per-series figure alongside "
            "SpecObservations.total_obs's aggregate across every series. For a "
            "scope-defaulted bulk latest-only series (no requested window), this "
            "is the coverage probe's all-time obs_count for that entity's "
            "selected facet -- the in-window count too, since no window narrows it"
        )
    )
    source_refs: list[ProvenanceRef] = Field(
        default_factory=list,
        description=(
            "provenance of the lead observation facet these points were read from "
            "(the facet selected for this series -- see SpecObservations)"
        ),
    )
    facet: dict[str, SeriesFacetRef] = Field(
        default_factory=dict,
        description=(
            "this series' lead observation facet, per axis (provenance/"
            "measurementMethod/unit); the estimates continuing the series keep the "
            "lead's entry; sparse -- an axis the facet carries no value "
            "for is absent, never a null placeholder. provenance here always agrees "
            "with source_refs, since both derive from the same lead facet"
        ),
    )
    conversion: ObservationSeriesConversion | None = Field(
        default=None,
        description=(
            "what recipe and tables converted this series' points to the requested "
            "currency; absent on a series served in its native facet"
        ),
    )
    # Whether a point the series serves carries a measurement method classed estimated
    # (`materialise.method_status.status_of`), as the estimates continuing a reported series
    # past its last period do. The part states it in a `modelled_values` departure
    # (`engine.departures`), so it reaches the wire there alone, and a series read back from
    # the cache does not hold it. Equality compares the wire state (`__eq__`).
    _holds_estimates: bool = PrivateAttr(default=False)

    @property
    def holds_estimates(self) -> bool:
        """Whether a served point carries a method classed estimated."""
        return self._holds_estimates

    def stating_estimates(self, held: bool) -> ObservationSeries:
        """A copy whose `holds_estimates` is `held`."""
        stated = self.model_copy()
        stated._holds_estimates = held
        return stated

    def __eq__(self, other: object) -> bool:
        """Equal when the wire state is: `holds_estimates` is an input to the part's
        departures, which a cached response carries already written."""
        if not isinstance(other, ObservationSeries):
            return NotImplemented
        return self.__dict__ == other.__dict__


class ObservationsOmission(StrictModel):
    """Why a SpecObservations payload carries no series, made unconditional:
    every _fetch_observations call that reaches a "nothing to show" outcome
    sets this field instead of leaving the whole payload absent, so a client
    is never left guessing which of at least a dozen distinct causes
    produced an empty preview.

    reason is a warning code from search.engine.region.codes (see
    ObservationsOmissionReason's own docstring for the closed vocabulary and
    why it is duplicated here rather than imported) -- the SAME code the
    accompanying Warning carries when one is also emitted, so a client reads
    one vocabulary rather than two. Not every reason accompanies a Warning:
    a genuinely uninteresting omission (nothing was ever there to probe)
    carries no diagnostic Warning of its own, but always carries this field.

    The converse also holds, and the pairing is not exhaustive: a reason
    arriving as a bare Warning with `omitted` absent means partial coverage,
    not absence -- a fetch that delivered some series would never set this
    field, but can warn (`observations_probe_failed` on a partially hard-
    failed fetch, and `observations_timed_out` on one some of whose pairs
    crossed the deadline).

    Whenever `omitted` is set, SpecObservations.series is `[]`, total_obs is
    `0`, and truncated is `false`, so a client reading `series ?? []` and
    ignoring `omitted` sees an ordinary empty result, with no schema_version
    bump needed to stay compatible.

    pairs/cap: populated only for reason=="observations_preview_too_large" --
    the (stat_var, role_tuple) pair (or pair x elected-combination
    dispatch-item, or distinct-stat_var) count that was compared against the
    ceiling it crossed, and that ceiling itself. Lets a client render "this
    selection has {pairs} series, above the {cap} the preview fetches"
    without re-deriving either number.

    dispatched: how many (stat_var, role_tuple) pairs this call actually
    dispatched. Populated for reason=="observations_no_points",
    "observations_probe_failed", "observations_no_matching_source", and
    "observations_timed_out" -- every reason a dispatch actually happened
    for; None for a reason reached before dispatch (e.g.
    "observations_nothing_to_probe").

    empty: dispatched pairs that carried no data on their own facet.
    Populated only for reason=="observations_no_points".

    hard_failed: dispatched pairs whose fetch itself failed. Populated for
    reason=="observations_no_points" (always 0 there -- a nonzero count
    means reason is "observations_probe_failed" instead) and for
    "observations_probe_failed" itself (the nonzero count that produced
    that reason); None elsewhere.

    facet_rejected: dispatched pairs a bound facet_filter (or currency
    conversion) rejected. Populated for reason=="observations_no_points"
    (always 0 there -- a nonzero count means reason is
    "observations_no_matching_source" instead) and for
    "observations_no_matching_source" itself; None elsewhere.

    Every field left None where the branch that produced this omission has
    no meaningful count for it.
    """

    reason: ObservationsOmissionReason = Field(
        description="machine-readable cause, equal to the accompanying Warning's code when one fires"
    )
    message: str = Field(description="human-readable explanation, safe to render directly")
    pairs: int | None = Field(
        default=None,
        description="observations_preview_too_large only: the count that was too large",
    )
    cap: int | None = Field(
        default=None, description="observations_preview_too_large only: the ceiling it exceeded"
    )
    dispatched: int | None = Field(
        default=None,
        description=(
            "observations_no_points/observations_probe_failed/"
            "observations_no_matching_source/observations_timed_out: pairs actually dispatched"
        ),
    )
    empty: int | None = Field(
        default=None,
        description="observations_no_points only: dispatched pairs that carried no data on their own facet",
    )
    hard_failed: int | None = Field(
        default=None,
        description=(
            "observations_no_points (always 0) and observations_probe_failed: dispatched "
            "pairs whose fetch itself failed"
        ),
    )
    facet_rejected: int | None = Field(
        default=None,
        description=(
            "observations_no_points (always 0) and observations_no_matching_source: "
            "dispatched pairs a bound facet_filter rejected"
        ),
    )


class SpecObservations(StrictModel):
    """The observation values behind a Spec's coverage.

    An opt-in bounded preview, or the full series when the resolved answer is
    small enough to ship in full -- see truncated. Describes the SAME (stat_var,
    entity) selection Spec.resolved.coverage already reports, so the numbers here never
    describe a different variable than what coverage summarises.

    Each series carries one lead observation facet's points, never a union
    across facets: a StatVar/entity pair can have several reporting facets
    (e.g. commitments vs disbursements, current vs constant prices), and
    merging their points chronologically would interleave same-dated points
    from incompatible series into one meaningless line. The one exception is
    the same source's estimates continuing the lead past its last period: the
    series carries their points after the lead's, `facet` names the lead, and
    the `modelled_values` departure marks the estimates. One facet group is
    selected per StatVar instead, across every place and role tuple the answer
    names: among facets holding a point in the requested years (all facets when
    none does), DFO's facets first for a variable both DFO and base Data
    Commons hold, then the one covering the most places or role tuples there,
    then the most requested years, then the most observations, then the latest
    observation, then a deterministic tie-break.
    """

    series: list[ObservationSeries] = Field(
        description="one entry per (stat_var, role tuple) pair with observations"
    )
    total_obs: int = Field(
        description=(
            "total observation count across every series' SELECTED facet, "
            "window-filtered to the spec's requested window when one exists -- "
            "known even when truncated is true and points are truncated. For a "
            "windowed per-pair answer, this is the chosen facet's in-window "
            "point count, NOT a raw facet obs_count sum and NOT a union across "
            "every facet of the pair -- see ObservationSeries. For a "
            "scope-defaulted bulk latest-only answer (no requested window), "
            "this sums the coverage probe's all-time obs_count per delivered "
            "entity instead: with no window to narrow it, that raw obs_count "
            "IS the in-window count"
        )
    )
    truncated: bool = Field(
        description=(
            "false when every series above carries its FULL point set (the whole "
            "answer's point count was within the server's inline threshold and each "
            "series within its row limit); true when a series was truncated to its "
            "latest points only, its total_obs still counting every point"
        )
    )
    omitted: ObservationsOmission | None = Field(
        default=None,
        description=(
            "why this payload carries no series, when it doesn't; absent when series "
            "is genuinely populated. See ObservationsOmission's own docstring"
        ),
    )
    order: PreviewOrder | None = Field(
        default=None,
        description=(
            "how a role-bearing bulk preview chose the series it shows when it shows fewer "
            "than the answer holds: 'ranked' when it read every series and shows the most "
            "recent and largest (latest year, then latest value, then identity), 'fixed' "
            "when it shows a selection made before any value was read, StatVars newest "
            "year first and taken round-robin; absent when every series is shown or the "
            "preview is not a role-bearing bulk one"
        ),
    )


def omitted_observations(
    reason: ObservationsOmissionReason,
    message: str,
    *,
    pairs: int | None = None,
    cap: int | None = None,
    dispatched: int | None = None,
    empty: int | None = None,
    hard_failed: int | None = None,
    facet_rejected: int | None = None,
) -> SpecObservations:
    """Build an omitted payload with empty series, zero observations, and ``truncated=False``.

    Centralizing these values keeps them consistent with the ``omitted`` reason and counts.
    """
    return SpecObservations(
        series=[],
        total_obs=0,
        truncated=False,
        omitted=ObservationsOmission(
            reason=reason,
            message=message,
            pairs=pairs,
            cap=cap,
            dispatched=dispatched,
            empty=empty,
            hard_failed=hard_failed,
            facet_rejected=facet_rejected,
        ),
    )
