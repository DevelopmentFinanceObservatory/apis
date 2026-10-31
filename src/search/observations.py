"""Observation preview caps, the preview-mode rule, and the assembly from series in hand to
`SpecObservations`, shared by both observation surfaces.

The live preview (`search.engine`) and the frozen observations door (`datasets_api`) build the
same `SpecObservations` shape. This module is the base-package home for what they share. It
imports nothing from `search.engine`, so a process without the engine extra loads it.

`ObservationSettings` owns the preview caps, read from `SEARCH_*` environment variables.
Consumers call `observation_settings()` at the point of use, so a test fixture can replace the
object and every reader sees the replacement.

A caller decides the preview's shape with `preview_mode`, then builds it from its series reads
(`SeriesRead`, keyed by `SeriesKey`) with the matching assembly: `assemble_per_series`,
`assemble_role_bulk` or `assemble_scope_bulk`, or `observations_too_large` for an omitted
preview. Every assembly reads its labels in one `LabelSource.node_labels_batch` call and lets
the label source's errors propagate. Its flow entities' party kinds come from the code index
alone (`materialise.enrichment.read_party_kinds`), with no read.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from code_index.composites import DECLARED_COMPOSITES
from code_index.query import CodeIndex
from description.blocks import documentation_link
from materialise.enrichment import read_party_kinds
from materialise.roles import axis_for_role
from materialise.series_conversion import SeriesConversion, SeriesKey
from origin_catalogue import ProvenanceRow, current_catalogue
from search.models import (
    DataCommonsRef,
    EntityRoleKey,
    ObservationPoint,
    ObservationSeries,
    ObservationSeriesConversion,
    PreviewOrder,
    ProvenanceRef,
    SeriesAxisRef,
    SeriesConversionFallbackExample,
    SeriesEntityRef,
    SeriesFacetRef,
    SeriesStatVarRef,
    SourceRef,
    SpecObservations,
    omitted_observations,
)
from search.render.labels import _humanize_ref, join_labels

logger = logging.getLogger(__name__)

# CurrencyUnit dcids use the `Currency{iso_code}` convention.
CURRENCY_DCID_PREFIX = "Currency"


class ObservationSettings(BaseSettings):
    """Observation preview caps, read from `SEARCH_*` environment variables.

    Field names are the `SEARCH_` env var name lowercased with that prefix stripped.
    """

    model_config = SettingsConfigDict(env_prefix="SEARCH_", extra="forbid")

    # Caps (stat_var, role_tuple) preview pairs.
    # Overflow omits the whole preview as observations_preview_too_large.
    # Partial subsets would misstate which members have data.
    obs_preview_max_pairs: int = 120

    # Latest points kept per series above obs_inline_max. Older points are dropped.
    obs_preview_points: int = 20

    # Total points allowed in full before every series is truncated to obs_preview_points.
    obs_inline_max: int = 2000

    # Display cap on role-bearing (StatVar, role tuple) series. SDMX row and byte caps
    # separately bound the fetch.
    role_bulk_series_max: int = 20

    # Rows a role-bearing bulk read may return and still be read whole and ranked
    # (`role_bulk_order`). Above it the preview reads a fixed selection. About 300 bytes a
    # row on the wire, so the default reads about 3 MB.
    role_bulk_rank_max_rows: int = 10_000

    @model_validator(mode="after")
    def _validate_bounds(self) -> ObservationSettings:
        """Raises ValueError when the settings are built."""
        if self.obs_preview_max_pairs < 1:
            raise ValueError(
                f"SEARCH_OBS_PREVIEW_MAX_PAIRS must be >= 1; got {self.obs_preview_max_pairs!r}"
            )
        if self.obs_preview_points < 1:
            raise ValueError(
                f"SEARCH_OBS_PREVIEW_POINTS must be >= 1; got {self.obs_preview_points!r}"
            )
        if self.obs_inline_max < 1:
            raise ValueError(f"SEARCH_OBS_INLINE_MAX must be >= 1; got {self.obs_inline_max!r}")
        if self.role_bulk_series_max < 1:
            raise ValueError(
                f"SEARCH_ROLE_BULK_SERIES_MAX must be >= 1; got {self.role_bulk_series_max!r}"
            )
        if self.role_bulk_rank_max_rows < 0:
            raise ValueError(
                f"SEARCH_ROLE_BULK_RANK_MAX_ROWS must be >= 0; got {self.role_bulk_rank_max_rows!r}"
            )
        return self


# Built on first read, after `search.engine.config` has loaded `.env` into the environment. A
# test fixture installs a different instance with
# `monkeypatch.setattr(search.observations, "_SETTINGS", ObservationSettings(...))`.
_SETTINGS: ObservationSettings | None = None


def observation_settings() -> ObservationSettings:
    """Return the process-wide `ObservationSettings` object, building it on first use."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = ObservationSettings()
    return _SETTINGS


def _source_and_origin(row: ProvenanceRow) -> tuple[SourceRef, DataCommonsRef]:
    return (
        SourceRef(ref=row.source.ref, label=row.source.label),
        DataCommonsRef(id=row.data_commons.id, label=row.data_commons.label),
    )


def unlisted_provenances(dcids: Sequence[str]) -> list[str]:
    """The distinct provenance dcids the bound catalogue has no row for, in first-seen
    order. A caller adds these ids to its graph label batch."""
    catalogue = current_catalogue()
    return [dcid for dcid in dict.fromkeys(dcids) if catalogue.provenance(dcid) is None]


def provenance_ref(dcid: str, labels: Mapping[str, str]) -> ProvenanceRef | None:
    """The provenance ref for `dcid`, with the catalogue row's name, source and Data Commons.
    A provenance the catalogue does not list takes its graph label from `labels` and has no
    source or Data Commons. `None` when neither names it."""
    row = current_catalogue().provenance(dcid)
    if row is not None:
        source, origin = _source_and_origin(row)
        return ProvenanceRef(ref=dcid, label=row.name, source=source, data_commons=origin)
    label = labels.get(dcid)
    return None if label is None else ProvenanceRef(ref=dcid, label=label)


def provenance_refs(dcids: Sequence[str], labels: Mapping[str, str]) -> list[ProvenanceRef]:
    """`provenance_ref` for each distinct dcid in first-seen order, omitting unnamed ones."""
    return [
        ref for dcid in dict.fromkeys(dcids) if (ref := provenance_ref(dcid, labels)) is not None
    ]


def provenance_facet_ref(dcid: str, labels: Mapping[str, str]) -> SeriesFacetRef:
    """A series' provenance-axis facet ref for `dcid`, with the catalogue row's members and the
    route of its document. A provenance the catalogue does not list takes its graph label from
    `labels`, if any. A declared composite (`code_index.composites`), which neither the graph
    nor the catalogue holds, takes its declared label and no document route."""
    composite = DECLARED_COMPOSITES.get(dcid)
    if composite is not None:
        return SeriesFacetRef(ref=dcid, label=composite.label)
    row = current_catalogue().provenance(dcid)
    documentation = documentation_link(dcid)
    if row is None:
        return SeriesFacetRef(ref=dcid, label=labels.get(dcid), documentation=documentation)
    source, origin = _source_and_origin(row)
    return SeriesFacetRef(
        ref=dcid, label=row.name, source=source, data_commons=origin, documentation=documentation
    )


type SeriesRank = tuple[bool, int, bool, float, SeriesKey]


def series_rank(key: SeriesKey, latest: tuple[str, float | None] | None) -> SeriesRank:
    """The one order every preview shows series in, from a series' identity `key` and its
    latest `(date, value)` point.

    The newest latest year comes first, then the largest latest value, then `key`. A latest
    point with no value follows the valued ones of its year, and a series with no point
    comes last.
    """
    if latest is None:
        return (True, 0, True, 0.0, key)
    date, value = latest
    return (False, -int(date[:4]), value is None, -(value or 0.0), key)


def role_bulk_order(*, rows: int, converts_to_role_currency: bool) -> PreviewOrder:
    """`ranked` when a role-bearing bulk read of `rows` rows fits `role_bulk_rank_max_rows`,
    else `fixed`. A conversion into each entity's own currency is always `fixed`, since its
    values are in different currencies and do not compare."""
    if converts_to_role_currency or rows > observation_settings().role_bulk_rank_max_rows:
        return "fixed"
    return "ranked"


def member_year_order(years: Mapping[str, int | None]) -> list[str]:
    """The StatVars of `years` newest year first, undated last, then by dcid."""
    return sorted(years, key=lambda sv: (years[sv] is None, -(years[sv] or 0), sv))


def fixed_selection[T](columns: Sequence[Sequence[T]], limit: int) -> list[T]:
    """The first `limit` items of `columns` taken round-robin (the first item of each
    column in column order, then the second of each, and so on). Columns are StatVars in
    `member_year_order`, each holding its series in identity order, so one StatVar cannot
    fill the preview while others have series."""
    picked: list[T] = []
    depth = 0
    while len(picked) < limit and any(depth < len(column) for column in columns):
        picked.extend(column[depth] for column in columns if depth < len(column))
        depth += 1
    return picked[:limit]


def member_row_label(
    sv: str, constraints: Mapping[str, str], labels: Mapping[str, str]
) -> str | None:
    """Label a row from its member constraint values, sorted by property dcid.

    Use graph labels when available. Members without constraints keep their resolved
    StatVar label. No graph read occurs here.
    """
    if not constraints:
        return labels.get(sv)
    value_labels = [
        _humanize_ref(value_dcid, labels.get(value_dcid))
        for _prop, value_dcid in sorted(constraints.items())
    ]
    return join_labels(value_labels)


def as_entity_role_key(role: str) -> EntityRoleKey | None:
    """Map a declared role to its closed observation wire key, or None if it has no axis.

    Derive the key from `axis_for_role`, since shape role names are open while `EntityRoleKey`
    is fixed to the wire contract. An unmapped role is logged and omitted.
    """
    if role == "observationAbout":
        return "observationAbout"
    axis = axis_for_role(role)
    if axis is not None:
        return axis
    logger.warning(
        "role %r has no flow axis to derive a wire role key from; its entity "
        "is omitted from the observation series",
        role,
    )
    return None


def _is_iso_currency(currency: str) -> bool:
    """Whether `currency` is an upper-case ISO 4217 alpha code (`"USD"`), as
    opposed to a spec role key (`"observationAbout"`, `"origin"`,
    `"destination"`). Mirrors `materialise.facets._is_iso_currency`."""
    return len(currency) == 3 and currency.isalpha() and currency.isupper()


def currency_unit_ref(currency: str | None) -> str | None:
    """The `CurrencyUnit` dcid a unit request for `currency` converts to, or None for a role
    currency, which resolves a different currency per series."""
    if currency is None or not _is_iso_currency(currency):
        return None
    return f"{CURRENCY_DCID_PREFIX}{currency}"


# ---------------------------------------------------------------------------
# Preview mode
# ---------------------------------------------------------------------------


class PreviewMode(StrEnum):
    """The shape an observation preview takes."""

    PER_SERIES = "per_series"
    """Every series in full, or its latest points above the inline cap (`assemble_per_series`)."""
    ROLE_BULK = "role_bulk"
    """A role-bearing answer's ranked or fixed selection of series (`assemble_role_bulk`)."""
    SCOPE_BULK = "scope_bulk"
    """One latest point per place of a defaulted place scope (`assemble_scope_bulk`)."""
    TOO_LARGE = "too_large"
    """No series. The preview is omitted (`observations_too_large`)."""


def preview_mode(
    *,
    role_bearing: bool,
    places_defaulted: bool,
    bound_pairs: int,
    series_count: Callable[[], int],
) -> PreviewMode:
    """The preview mode of an answer, against `obs_preview_max_pairs` (the cap).

    Args:
        role_bearing: the answer binds no `observationAbout` place and reads its entities on
            flow roles.
        places_defaulted: the answer names no place and reads a defaulted place set of more
            than one place, which is never the worldwide total alone. Read only for a standard
            answer.
        bound_pairs: the answer's StatVars times every combination of the entities it binds
            on its roles. Read only for a role-bearing answer.
        series_count: the series the answer holds in the years it reads. Called at most once,
            and only when the flags and `bound_pairs` leave the mode open, so a costly count is
            skipped when they decide.

    Returns:
        `SCOPE_BULK` for a standard answer over defaulted places, whatever its size.
        `ROLE_BULK` for a role-bearing answer whose bound pairs or series exceed the cap.
        `TOO_LARGE` for any other answer whose series exceed the cap. `PER_SERIES` otherwise.
    """
    if not role_bearing and places_defaulted:
        return PreviewMode.SCOPE_BULK
    cap = observation_settings().obs_preview_max_pairs
    if role_bearing and bound_pairs > cap:
        return PreviewMode.ROLE_BULK
    if series_count() > cap:
        return PreviewMode.ROLE_BULK if role_bearing else PreviewMode.TOO_LARGE
    return PreviewMode.PER_SERIES


# Pair-count and elected-mode dispatch-item caps apply before any fetch. The omission carries
# the crossed count and limit.
OBSERVATIONS_PREVIEW_TOO_LARGE = "observations_preview_too_large"
# A scope-defaulted bulk preview cannot honour a requested window because it returns one
# latest point per place. This fires before any read, regardless of scope size.
OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED = "observations_windowed_scope_unsupported"
# The Datasets observations door omits the bulk preview of a converted version. The live
# preview converts its bulk reads (`region.observations.bulk_conversion`) and uses this code
# only for StatVars that conversion left out: a warning naming them, or an omission when none
# converted. Windowed scope-defaulted previews use OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED
# first.
OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED = "observations_converted_scope_unsupported"

PREVIEW_OMISSION_REASONS: frozenset[str] = frozenset(
    {
        OBSERVATIONS_PREVIEW_TOO_LARGE,
        OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED,
        OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED,
    }
)
"""The omission reasons `observations_too_large`, `observations_windowed_scope` and
`observations_converted_bulk` return. The Datasets observations door puts all three on the
wire. The live preview puts the first two, and the third when a bulk conversion leaves
StatVars out (`bulk_conversion.left_out_omission`). Registry-binding tests use this served-code set."""

_TOO_LARGE_MESSAGE = (
    "observations fetch skipped: the requested member x entity combination is too large to preview."
)
"""The omission message of a preview with more series than the cap."""


def observations_too_large(count: int) -> SpecObservations:
    """The omitted preview of an answer holding `count` series, above `obs_preview_max_pairs`:
    no series, reason `observations_preview_too_large`, with `count` and the cap."""
    return omitted_observations(
        OBSERVATIONS_PREVIEW_TOO_LARGE,
        _TOO_LARGE_MESSAGE,
        pairs=count,
        cap=observation_settings().obs_preview_max_pairs,
    )


def observations_windowed_scope() -> SpecObservations:
    """The omitted `SCOPE_BULK` preview of an answer reading a date range: one latest point
    per place cannot honour the range. Reason `observations_windowed_scope_unsupported`."""
    return omitted_observations(
        OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED,
        "the observation preview was not fetched: a scope-wide "
        "preview can only show one latest point per place, which "
        "cannot honour the requested date range. Drop the date "
        "range or query a specific place to see observations.",
    )


_CONVERTED_BULK_MESSAGES = {
    PreviewMode.SCOPE_BULK: (
        "the observation preview was not fetched: a scope-wide "
        "preview merges facets across every place and cannot "
        "apply the requested unit conversion. Query a specific "
        "place to see converted observations."
    ),
    PreviewMode.ROLE_BULK: (
        "the observation preview was not fetched: a "
        "role-bearing bulk preview merges facets across "
        "every series and cannot apply the requested unit "
        "conversion. Name a specific entity on the open "
        "role to see converted observations."
    ),
}


def observations_converted_bulk(mode: PreviewMode) -> SpecObservations:
    """The Datasets observations door's omitted bulk preview (`SCOPE_BULK` or `ROLE_BULK`) of
    a version whose unit request converts. The live preview converts the same bulk reads
    (`region.observations.bulk_conversion`). Reason `observations_converted_scope_unsupported`.
    Raises `KeyError` for any other mode."""
    return omitted_observations(
        OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED, _CONVERTED_BULK_MESSAGES[mode]
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


class LabelSource(Protocol):
    """A graph client that labels nodes. Absent dcids are left out of the result."""

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]: ...


@dataclass(frozen=True)
class SeriesRead:
    """One series in hand, keyed elsewhere by its `SeriesKey`.

    `points` are `(date, value)` pairs in date order. `total_rows` counts the series' rows
    before any row limit, so a series holding fewer points than that is cut. `facets` maps
    each facet axis the series' facet declares (`provenance`, `measurementMethod`, `unit`) to
    its ref. `conversion` describes a converted series, whose facet `unit` is the native unit
    the rows converted from. `holds_estimates` is whether a point's own method is classed
    estimated (`materialise.method_status.status_of`), as a continuation's points are, while
    `facets` states the facet the series was elected by.
    """

    points: Sequence[tuple[str, float | None]]
    total_rows: int
    facets: Mapping[str, str] = field(default_factory=dict)
    conversion: SeriesConversion | None = None
    holds_estimates: bool = False


def facet_value_label(axis: str, label: str) -> str:
    """A facet value's display label on `axis`. A measurementMethod label drops one trailing
    full stop, since the graph names some methods as sentences ("World Bank Subnational
    Population Estimates.") and the byline and option lists state them mid-sentence."""
    return label.removesuffix(".") if axis == "measurementMethod" else label


def series_facet_map(
    facets: Mapping[str, str], labels: Mapping[str, str]
) -> dict[str, SeriesFacetRef]:
    """A series' wire facet map from its axis refs. The provenance axis takes its label, source
    and Data Commons from the origin catalogue, else its label from `labels`; other axes take
    `labels` (`facet_value_label`). An unlabelled ref keeps `label=None`."""
    return {
        axis: provenance_facet_ref(ref, labels)
        if axis == "provenance"
        else SeriesFacetRef(
            ref=ref,
            label=None if (label := labels.get(ref)) is None else facet_value_label(axis, label),
        )
        for axis, ref in facets.items()
    }


def _wire_conversion(
    conversion: SeriesConversion, labels: Mapping[str, str]
) -> ObservationSeriesConversion:
    """`conversion` on the wire, its table provenance described by the origin catalogue."""
    return ObservationSeriesConversion(
        recipe_id=conversion.recipe_id,
        table_provenance=provenance_facet_ref(conversion.table_provenance_id, labels),
        exchange_rate_table_version=conversion.exchange_rate_table_version,
        deflator_table_version=conversion.deflator_table_version,
        fallback_count=conversion.fallback_count,
        fallback_examples=[
            SeriesConversionFallbackExample(entity=entity, fallback_entity=fallback_entity)
            for entity, fallback_entity in conversion.fallback_examples
        ],
        projected_count=conversion.projected_count,
        projected_examples=list(conversion.projected_examples),
    )


def _read_labels(
    reads: Mapping[SeriesKey, SeriesRead],
    *,
    label_source: LabelSource,
    member_constraints: Mapping[str, Mapping[str, str]],
    target_currency_ref: str | None = None,
) -> dict[str, str]:
    """Every label `reads` show, in one batch: provenances the catalogue does not list, role
    entities, StatVars, method and unit refs, member constraint values and the target
    currency. No batch when nothing needs a label."""
    provenance_dcids = [
        *(ref for read in reads.values() if (ref := read.facets.get("provenance")) is not None),
        *(
            read.conversion.table_provenance_id
            for read in reads.values()
            if read.conversion is not None
        ),
    ]
    label_dcids = list(
        dict.fromkeys(
            [
                *unlisted_provenances(provenance_dcids),
                *(dcid for _sv, role_tuple in reads for _role, dcid in role_tuple),
                *(sv for sv, _role_tuple in reads),
                *(
                    ref
                    for read in reads.values()
                    for axis in ("measurementMethod", "unit")
                    if (ref := read.facets.get(axis)) is not None
                ),
                *(
                    value
                    for sv, _role_tuple in reads
                    for value in member_constraints.get(sv, {}).values()
                ),
                *([target_currency_ref] if target_currency_ref is not None else []),
            ]
        )
    )
    return label_source.node_labels_batch(label_dcids) if label_dcids else {}


def _series(
    key: SeriesKey,
    read: SeriesRead,
    points: Sequence[tuple[str, float | None]],
    *,
    labels: Mapping[str, str],
    index: CodeIndex,
    member_constraints: Mapping[str, Mapping[str, str]],
    target_currency_ref: str | None = None,
) -> ObservationSeries:
    """One wire series for `read`, showing `points`. Each entity on a flow role carries the
    party kind `index` lists for it in the series' provenance, or in every dataset for a
    series whose facet names none (`materialise.enrichment.read_party_kinds`).
    Its axes are its StatVar's `member_constraints`, each value labelled from `labels`, else by
    its humanised dcid (`_humanize_ref`), as the StatVar's row label names it. A converted
    series reports the target currency as its unit when there is one, and no unit otherwise.
    It holds estimates when `read` does (`ObservationSeries.holds_estimates`)."""
    sv, role_tuple = key
    constraints = member_constraints.get(sv, {})
    provenance = read.facets.get("provenance")
    party_kinds = read_party_kinds(
        role_tuple, parties=index.parties, datasets=() if provenance is None else (provenance,)
    )
    facet_map = series_facet_map(read.facets, labels)
    conversion = None
    if read.conversion is not None:
        if target_currency_ref is not None:
            facet_map["unit"] = SeriesFacetRef(
                ref=target_currency_ref, label=labels.get(target_currency_ref)
            )
        else:
            facet_map.pop("unit", None)
        conversion = _wire_conversion(read.conversion, labels)
    return ObservationSeries(
        stat_var=SeriesStatVarRef(ref=sv, label=member_row_label(sv, constraints, labels)),
        entities=[
            SeriesEntityRef(
                role=role_key,
                ref=dcid,
                label=labels.get(dcid),
                party_kind=party_kinds.get((role, dcid)),
            )
            for role, dcid in role_tuple
            if (role_key := as_entity_role_key(role)) is not None
        ],
        axes=[
            SeriesAxisRef(property=prop, ref=value, label=_humanize_ref(value, labels.get(value)))
            for prop, value in sorted(constraints.items())
        ],
        points=[ObservationPoint(date=date, value=value) for date, value in points],
        total_obs=read.total_rows,
        source_refs=provenance_refs([provenance] if provenance is not None else [], labels),
        facet=facet_map,
        conversion=conversion,
    ).stating_estimates(read.holds_estimates)


def _latest_point(points: Sequence[tuple[str, float | None]]) -> tuple[str, float | None] | None:
    return points[-1] if points else None


def _ranked(
    series: Sequence[tuple[SeriesKey, ObservationSeries]],
) -> list[ObservationSeries]:
    """`series` in `series_rank` order, grouped by unit: series in one unit stay together,
    and each unit's group stands where its first series ranks, so values are compared only
    within a unit."""

    def rank(keyed: tuple[SeriesKey, ObservationSeries]) -> SeriesRank:
        key, row = keyed
        return series_rank(key, (row.points[-1].date, row.points[-1].value) if row.points else None)

    by_unit: dict[str | None, list[ObservationSeries]] = {}
    for _key, row in sorted(series, key=rank):
        unit = row.facet.get("unit")
        by_unit.setdefault(unit.ref if unit is not None else None, []).append(row)
    return [row for rows in by_unit.values() for row in rows]


def assemble_per_series(
    reads: Mapping[SeriesKey, SeriesRead],
    *,
    label_source: LabelSource,
    index: CodeIndex,
    member_constraints: Mapping[str, Mapping[str, str]],
    target_currency_ref: str | None,
) -> SpecObservations:
    """The per-series preview (`PreviewMode.PER_SERIES`) of `reads`.

    Each read becomes one series, labelled from one `label_source` batch: its StatVar by
    `member_row_label` over `member_constraints[stat_var]`, its role entities, and its facet
    and source refs. Each origin and destination entity carries the party kind `index` lists
    for it (`_series`). A converted read shows `target_currency_ref` (the `CurrencyUnit` a
    currency request converts to, `currency_unit_ref`) as its unit, or no unit for a role
    currency. Series sort by `series_rank` within groups of one unit (`_ranked`).

    `total_obs` sums every read's `total_rows`. Above `obs_inline_max` every series keeps
    its last `obs_preview_points` points. `truncated` is set then, or when any read holds
    fewer points than its `total_rows`.

    Raises whatever `label_source.node_labels_batch` raises.
    """
    labels = _read_labels(
        reads,
        label_source=label_source,
        member_constraints=member_constraints,
        target_currency_ref=target_currency_ref,
    )
    settings = observation_settings()
    total_obs = sum(read.total_rows for read in reads.values())
    over_inline = total_obs > settings.obs_inline_max
    row_limited = any(len(read.points) < read.total_rows for read in reads.values())
    series = _ranked(
        [
            (
                key,
                _series(
                    key,
                    read,
                    read.points[-settings.obs_preview_points :] if over_inline else read.points,
                    labels=labels,
                    index=index,
                    member_constraints=member_constraints,
                    target_currency_ref=target_currency_ref,
                ),
            )
            for key, read in reads.items()
        ]
    )
    return SpecObservations(
        series=series, total_obs=total_obs, truncated=over_inline or row_limited
    )


def role_bulk_shown(
    reads: Mapping[SeriesKey, SeriesRead], *, order: PreviewOrder
) -> list[SeriesKey]:
    """The keys `assemble_role_bulk` shows, at most `role_bulk_series_max`, in
    `series_rank` order.

    `ranked` shows the first keys by `series_rank`. `fixed` shows `fixed_selection` over
    each StatVar's keys in identity order, StatVars in `member_year_order` by the year of
    their newest point. A live fixed read holds only the series it selected, so this keeps
    them all. The Datasets observations door holds every series and selects the same way.
    """
    limit = observation_settings().role_bulk_series_max

    def rank(key: SeriesKey) -> SeriesRank:
        return series_rank(key, _latest_point(reads[key].points))

    if order == "ranked":
        return sorted(reads, key=rank)[:limit]
    by_sv: dict[str, list[SeriesKey]] = {}
    for key in sorted(reads):
        by_sv.setdefault(key[0], []).append(key)
    years = {
        sv: max(
            (int(reads[key].points[-1][0][:4]) for key in keys if reads[key].points), default=None
        )
        for sv, keys in by_sv.items()
    }
    picked = fixed_selection([by_sv[sv] for sv in member_year_order(years)], limit)
    return sorted(picked, key=rank)


def assemble_role_bulk(
    reads: Mapping[SeriesKey, SeriesRead],
    *,
    label_source: LabelSource,
    index: CodeIndex,
    member_constraints: Mapping[str, Mapping[str, str]],
    order: PreviewOrder,
    unread_obs: int = 0,
    total_series: int | None = None,
    target_currency_ref: str | None = None,
) -> SpecObservations:
    """The role-bearing bulk preview (`PreviewMode.ROLE_BULK`) of `reads`.

    Shows the series `role_bulk_shown` picks under `order`, in `series_rank` order,
    labelled as `assemble_per_series` labels them from one `label_source` batch over the
    shown series. A converted read shows
    `target_currency_ref` as its unit, or no unit for a role currency, as
    `assemble_per_series` shows it. The result states `order` when it shows fewer series
    than `total_series`, the series the answer holds (every read when None).

    `total_obs` sums every read's `total_rows`, shown or not, plus `unread_obs`, the
    observations of series the caller counted without reading. Above `obs_inline_max` shown
    points, every shown series keeps its last `obs_preview_points` points and `truncated` is
    set.

    Raises whatever `label_source.node_labels_batch` raises.
    """
    settings = observation_settings()
    shown = {key: reads[key] for key in role_bulk_shown(reads, order=order)}
    labels = _read_labels(
        shown,
        label_source=label_source,
        member_constraints=member_constraints,
        target_currency_ref=target_currency_ref,
    )
    truncated = sum(len(read.points) for read in shown.values()) > settings.obs_inline_max
    series = [
        _series(
            key,
            read,
            read.points[-settings.obs_preview_points :] if truncated else read.points,
            labels=labels,
            index=index,
            member_constraints=member_constraints,
            target_currency_ref=target_currency_ref,
        )
        for key, read in shown.items()
    ]
    total_obs = sum(read.total_rows for read in reads.values()) + unread_obs
    cut = len(shown) < (len(reads) if total_series is None else total_series)
    return SpecObservations(
        series=series, total_obs=total_obs, truncated=truncated, order=order if cut else None
    )


def assemble_scope_bulk(
    reads: Mapping[SeriesKey, SeriesRead],
    *,
    label_source: LabelSource,
    index: CodeIndex,
    member_constraints: Mapping[str, Mapping[str, str]],
    target_currency_ref: str | None = None,
) -> SpecObservations:
    """The defaulted-scope bulk preview (`PreviewMode.SCOPE_BULK`) of `reads`, each keyed at
    one `observationAbout` place.

    Each series shows its latest point alone, labelled as `assemble_per_series` labels them
    from one `label_source` batch, and sorted by `series_rank` within groups of one unit
    (`_ranked`). A converted read shows `target_currency_ref` as its unit, or no unit
    for a role currency, as `assemble_per_series` shows it. `total_obs` sums every read's
    `total_rows`, and the preview is `truncated` when `total_obs` exceeds the series shown,
    each holding one point.

    Raises whatever `label_source.node_labels_batch` raises.
    """
    labels = _read_labels(
        reads,
        label_source=label_source,
        member_constraints=member_constraints,
        target_currency_ref=target_currency_ref,
    )
    series = _ranked(
        [
            (
                key,
                _series(
                    key,
                    read,
                    read.points[-1:],
                    labels=labels,
                    index=index,
                    member_constraints=member_constraints,
                    target_currency_ref=target_currency_ref,
                ),
            )
            for key, read in reads.items()
        ]
    )
    total_obs = sum(read.total_rows for read in reads.values())
    return SpecObservations(series=series, total_obs=total_obs, truncated=total_obs > len(series))
