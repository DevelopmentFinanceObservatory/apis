"""Builds `WriteReply` and its token-capped fact summary.

The summary covers changed fields, capped series, window and unit where
applicable, observed values, change, source refs, and the defaults the
component applied where the intent left a choice open. Values use the formatter
profile of the release that minted the revision. This module defines the summary
shape outside catalogue schemas. `resolved_payload` is the stamped envelope
stored by the revision, including the stored head on a no-op. Write and readback
summaries use the same builder and coordinates. Readbacks always have an empty
`changed` list."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from dfo_frame_grammar import STAT_VAR_COLUMN, UNIT_COLUMN

from assembler.errors import ReleaseMismatch, escape_pointer_segment
from assembler.frame_view import facet_dimension_names
from assembler.limits import FACT_SUMMARY_SERIES_CAP, FACT_SUMMARY_TOKEN_CAP
from assembler.select import has_arrayed_dimension
from materialise.unit_cells import unit_identity

logger = logging.getLogger(__name__)

__all__ = [
    "ChangedField",
    "WriteReply",
    "applied_defaults",
    "build_changed_fields",
    "build_fact_summary",
    "build_reply",
    "format_value",
    "series_entry",
]

# Log this marker when rows mix missing and present `series_id` values,
# any id is invalid, or rows sharing an id disagree on `dimensions`.
# Grouping then falls back to row dimensions, as it does when all ids are absent.
# Alert policies depend on this stable string.
TABLE_ROW_SERIES_ID_INCONSISTENT_MARKER = "table_row_series_id_inconsistent"

# Log this marker when stored coordinates contain a malformed or duplicate id.
# Building an id map would silently keep the last duplicate and assign its
# coordinates to every rebuilt series carrying that id. A stored revision is
# immutable, so coordinates are withheld for every series.
# Alert policies depend on this stable string.
SERIES_COORDINATES_INCONSISTENT_MARKER = "series_coordinates_inconsistent"

_Scalar = (str, int, float, bool, type(None))


@dataclass(frozen=True)
class ChangedField:
    """One patched field. Scalar values are disclosed. Arrays and objects use `None` with
    `disclosed=False`. A scalar `null` keeps `disclosed=True`."""

    field: str
    prior: object | None
    new: object | None
    disclosed: bool = True


@dataclass(frozen=True)
class WriteReply:
    component_id: str
    revision: int
    catalog_version: str
    composition_id: str | None
    composition_revision: int | None
    changed: tuple[ChangedField, ...]
    fact_summary: Mapping[str, object]
    resolved_payload: Mapping[str, object]
    # Set only on the create that minted them; revises and no-ops carry `None`.
    write_token: str | None = None
    composition_write_token: str | None = None


def build_changed_fields(
    *, current_intent: Mapping[str, object] | None, patch: Mapping[str, object]
) -> tuple[ChangedField, ...]:
    """Returns patch fields with prior and new scalar values. Composite values are withheld
    with `disclosed=False`. The merger rejects create-only keys. Creates return
    an empty tuple."""
    if current_intent is None:
        return ()
    fields = []
    for field, new_value in patch.items():
        prior_value = current_intent.get(field)
        if isinstance(prior_value, _Scalar) and isinstance(new_value, _Scalar):
            fields.append(ChangedField(field=field, prior=prior_value, new=new_value))
        else:
            fields.append(ChangedField(field=field, prior=None, new=None, disclosed=False))
    return tuple(fields)


def build_reply(
    *,
    composition_id: str | None,
    composition_revision: int | None,
    changed: tuple[ChangedField, ...],
    stored_intent: Mapping[str, object],
    resolved_payload: Mapping[str, object],
    series_coordinates: Sequence[Mapping[str, object]] | None,
    formatter_profile: Mapping[str, object],
    write_token: str | None = None,
    composition_write_token: str | None = None,
) -> WriteReply:
    component_id = cast(str, resolved_payload["component_id"])
    revision = cast(int, resolved_payload["revision"])
    catalog_version = cast(str, resolved_payload["catalog_version"])
    component = cast("Mapping[str, object]", resolved_payload["component"])
    fact_summary = build_fact_summary(
        component=component,
        member=cast(str, component["type"]),
        stored_intent=stored_intent,
        formatter_profile=formatter_profile,
        series_coordinates=series_coordinates,
        component_id=component_id,
        revision=revision,
        changed=changed,
        catalog_version=catalog_version,
    )
    return WriteReply(
        component_id=component_id,
        revision=revision,
        catalog_version=catalog_version,
        composition_id=composition_id,
        composition_revision=composition_revision,
        changed=changed,
        fact_summary=fact_summary,
        resolved_payload=resolved_payload,
        write_token=write_token,
        composition_write_token=composition_write_token,
    )


def format_value(*, value: float, profile: Mapping[str, object]) -> str:
    significant_digits = cast(int, profile.get("significant_digits", 3))
    return f"{value:.{significant_digits}g}"


def series_entry(
    *,
    id: object,
    label: object,
    coordinates: Mapping[str, object] | Sequence[tuple[str, object]],
    points: Sequence[Mapping[str, object]],
    profile: Mapping[str, object],
) -> dict[str, object]:
    """One `fact_summary.series[]` entry: `id`/`label`/`coordinates` as
    given, plus `point_count` and the formatted first/last observed values
    derived from `points`. Called once per series by `build_fact_summary`."""
    observed = [point for point in points if "value" in point]
    return {
        "id": id,
        "label": label,
        "coordinates": dict(coordinates),
        "point_count": len(observed),
        "first_value": (
            format_value(value=cast(float, observed[0]["value"]), profile=profile)
            if observed
            else None
        ),
        "last_value": (
            format_value(value=cast(float, observed[-1]["value"]), profile=profile)
            if observed
            else None
        ),
    }


def _is_valid_series_id(value: object) -> bool:
    """Stored ids are valid when they are non-empty strings.

    Invalid ids trigger coordinate withholding. Stat-card summaries emit `None`."""
    return isinstance(value, str) and bool(value)


def _resolve_stored_series_coordinates(
    *,
    series_list: Sequence[Mapping[str, object]],
    stored: Sequence[Mapping[str, object]] | None,
    component_id: str,
    revision: int,
) -> list[Mapping[str, object]]:
    """Matches stored coordinates by id, or by position for id-less equal-length lists.

    Duplicate or malformed ids log a warning and withhold all coordinates.
    Missing coordinates become empty mappings."""
    if not stored:
        return [{} for _ in series_list]

    non_null_ids = [entry.get("id") for entry in stored if entry.get("id") is not None]
    malformed_ids = [one_id for one_id in non_null_ids if not _is_valid_series_id(one_id)]
    if malformed_ids:
        logger.warning(
            "%s component_id=%s revision=%s reason=malformed_non_null_id ids=%s",
            SERIES_COORDINATES_INCONSISTENT_MARKER,
            component_id,
            revision,
            non_null_ids,
        )
        return [{} for _ in series_list]
    valid_ids = cast("list[str]", non_null_ids)
    if len(valid_ids) != len(set(valid_ids)):
        logger.warning(
            "%s component_id=%s revision=%s reason=duplicate_non_null_id ids=%s",
            SERIES_COORDINATES_INCONSISTENT_MARKER,
            component_id,
            revision,
            valid_ids,
        )
        return [{} for _ in series_list]

    by_id = {
        cast(str, entry["id"]): entry.get("coordinates") or {}
        for entry in stored
        if entry.get("id") is not None
    }
    position_matches = len(stored) == len(series_list)
    resolved: list[Mapping[str, object]] = []
    for index, one in enumerate(series_list):
        series_id = one.get("id")
        if series_id is not None:
            resolved.append(cast("Mapping[str, object]", by_id.get(cast(str, series_id), {})))
        elif position_matches:
            resolved.append(cast("Mapping[str, object]", stored[index].get("coordinates") or {}))
        else:
            resolved.append({})
    return resolved


def build_fact_summary(
    *,
    component: Mapping[str, object],
    member: str,
    stored_intent: Mapping[str, object],
    formatter_profile: Mapping[str, object],
    series_coordinates: Sequence[Mapping[str, object]] | None,
    component_id: str,
    revision: int,
    changed: tuple[ChangedField, ...],
    catalog_version: str,
) -> dict[str, object]:
    """Builds summaries from the resolved payload and its series coordinates.

    Older revisions without coordinates read back `{}` per series."""
    source_refs: object | None = None
    if member in ("line_chart", "bar_chart"):
        series_list = cast("Sequence[Mapping[str, object]]", component.get("series") or ())
        axis_key = "category_axis" if member == "bar_chart" else "x_axis"
        # The release requires this axis key. A missing key marks an unsupported
        # stored shape and returns 503 `release_mismatch` with retry mode
        # `retry_after_delay`. Migration 0006 or a later release supplies it.
        if axis_key not in component:
            raise ReleaseMismatch(
                catalog_version=catalog_version,
                detail=(
                    f"stored {member} component is missing the '{axis_key}' key; "
                    "its payload shape does not match the release's schema"
                ),
            )
        axis = cast("Mapping[str, object]", component[axis_key])
        # Period mode carries a window. Entity mode reports one shared period.
        window: Mapping[str, object] | None = cast(
            "Mapping[str, object] | None", axis.get("window")
        )
        unit: object | None = series_list[0].get("unit") if series_list else None
    elif member == "stat_card":
        # The summary uses the sole `series_coordinates` entry for the series id.
        # The component stores only its headline point and label.
        raw_stat_card_id = series_coordinates[0].get("id") if series_coordinates else None
        stat_card_id = (
            cast(str, raw_stat_card_id) if _is_valid_series_id(raw_stat_card_id) else None
        )
        series_list = (
            {
                "id": stat_card_id,
                "label": component.get("label"),
                "points": ({"period": component.get("period"), "value": component.get("value")},),
            },
        )
        window = None
        unit = component.get("unit")
    elif member == "table":
        series_list = _table_series_from_rows(
            component=component,
            component_id=component_id,
            revision=revision,
        )
        window = None
        unit = None
    elif member == "source_card":
        # Binds no series. Its `source_refs[]` names every distinct raw
        # provenance ref the resolved frame's own rows carry.
        series_list = ()
        window = None
        unit = None
        source_refs = component.get("source_refs")
    else:
        series_list = ()
        window = None
        unit = component.get("unit")

    coordinates_by_index = _resolve_stored_series_coordinates(
        series_list=series_list,
        stored=series_coordinates,
        component_id=component_id,
        revision=revision,
    )
    if member == "table":
        unit = _table_unit(
            component=component, series_list=series_list, coordinates=coordinates_by_index
        )
    per_series = [
        series_entry(
            id=one.get("id"),
            label=one.get("label"),
            coordinates=coordinates_by_index[index],
            points=cast("Sequence[Mapping[str, object]]", one.get("points") or ()),
            profile=formatter_profile,
        )
        for index, one in enumerate(list(series_list)[:FACT_SUMMARY_SERIES_CAP])
    ]

    summary: dict[str, object] = {
        "changed": [entry.field for entry in changed],
        "series": per_series,
        "series_cut": max(0, len(series_list) - FACT_SUMMARY_SERIES_CAP),
        "window": window,
        "unit": unit,
        "change": component.get("change"),
        "source_refs": source_refs,
        "defaults": applied_defaults(
            component=component, member=member, stored_intent=stored_intent
        ),
    }
    return _cap_tokens(summary=summary, token_cap=FACT_SUMMARY_TOKEN_CAP)


def applied_defaults(
    *, component: Mapping[str, object], member: str, stored_intent: Mapping[str, object]
) -> list[dict[str, str]]:
    """The choices the component made where its intent left one open, each as the intent
    pointer it fills and a sentence saying how to narrow it.

    An entity-mode bar chart whose select lists no values compared every value of its
    category dimension, largest first (`assembler.mint.builders`). The stored intent and
    payload both carry what this reads, so a write and a later read agree, and
    `assembler.pipeline.write` mints a new revision when a revise changes them."""
    if member != "bar_chart":
        return []
    axis = cast("Mapping[str, object]", component["category_axis"])
    select = cast("Mapping[str, object]", stored_intent["select"])
    if axis["mode"] != "entity" or has_arrayed_dimension(select=select):
        return []
    dimension = cast(str, axis["category_dimension"])
    count = len(cast("Sequence[object]", component["series"]))
    return [
        {
            "pointer": f"/select/{escape_pointer_segment(dimension)}",
            "detail": (
                f"select leaves {dimension!r} unbound, so this bar_chart draws all {count} of "
                f"its values at {axis['period']}, largest first; list the {dimension!r} "
                "values to keep in select to narrow it"
            ),
        }
    ]


def _table_unit(
    *,
    component: Mapping[str, object],
    series_list: Sequence[Mapping[str, object]],
    coordinates: Sequence[Mapping[str, object]],
) -> object | None:
    """Returns a table-level unit when every row agrees. Mixed or empty tables return
    `None`, leaving each row's unit as the reference. Rows compare by `unit_identity`: by
    their series' frame `unit` coordinate, so two units whose labels read alike differ, and
    by the canonical JSON of the row's unit for a series stored without coordinates, which
    makes structured-unit equality independent of key order."""
    rows = cast("Sequence[Mapping[str, object]]", component.get("rows") or ())
    frame_unit_keys = {
        cast("str", one["id"]): held.get(UNIT_COLUMN)
        for one, held in zip(series_list, coordinates, strict=True)
        if one.get("id") is not None and held
    }
    units_by_identity = {
        unit_identity(
            series_id=row.get("series_id"),
            payload_unit=row.get("unit"),
            frame_unit_keys=frame_unit_keys,
        ): row.get("unit")
        for row in rows
    }
    return next(iter(units_by_identity.values())) if len(units_by_identity) == 1 else None


def _table_series_from_rows(
    *,
    component: Mapping[str, object],
    component_id: str,
    revision: int,
) -> tuple[Mapping[str, object], ...]:
    """Rebuilds table series for write and readback summaries.

    Rows group by valid, consistent `series_id`. Others group by
    `dimensions` and have no id. Labels join entity-role dimensions in sorted
    order, falling back to `stat_var` when none are present. Role keys come from the first row."""
    rows = cast("Sequence[Mapping[str, object]]", component.get("rows") or ())
    dimension_keys = (
        list(cast("Mapping[str, object]", rows[0].get("dimensions") or {})) if rows else []
    )
    excluded = {STAT_VAR_COLUMN} | facet_dimension_names()
    role_keys = sorted(key for key in dimension_keys if key not in excluded)

    chunks = _table_row_chunks_by_series_id(rows=rows, component_id=component_id, revision=revision)
    if chunks is not None:
        return tuple(
            _table_series_entry(id=series_id, rows=chunk_rows, role_keys=role_keys)
            for series_id, chunk_rows in chunks
        )

    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    order: list[tuple[object, ...]] = []
    for row in rows:
        dimensions = cast("Mapping[str, object]", row.get("dimensions") or {})
        key = tuple(dimensions.get(name) for name in dimension_keys)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    return tuple(
        _table_series_entry(id=None, rows=groups[key], role_keys=role_keys) for key in order
    )


def _table_row_chunks_by_series_id(
    *,
    rows: Sequence[Mapping[str, object]],
    component_id: str,
    revision: int,
) -> list[tuple[str, Sequence[Mapping[str, object]]]] | None:
    """Groups rows by `series_id` in first-seen order when every id is a non-empty string and
    each id has consistent dimensions. No ids returns `None`. Mixed or malformed
    ids and inconsistent dimensions log a warning before returning `None`."""
    raw_ids = [row.get("series_id") for row in rows]
    present = [one for one in raw_ids if one is not None]
    if not present:
        return None
    if len(present) != len(raw_ids) or any(not isinstance(one, str) or not one for one in present):
        logger.warning(
            "%s component_id=%s revision=%s reason=mixed_or_malformed_series_id",
            TABLE_ROW_SERIES_ID_INCONSISTENT_MARKER,
            component_id,
            revision,
        )
        return None

    ids = cast("list[str]", raw_ids)
    chunks: dict[str, list[Mapping[str, object]]] = {}
    order: list[str] = []
    for row, series_id in zip(rows, ids, strict=True):
        if series_id not in chunks:
            chunks[series_id] = []
            order.append(series_id)
        chunks[series_id].append(row)

    for series_id in order:
        chunk_rows = chunks[series_id]
        first_dimensions = chunk_rows[0].get("dimensions")
        if any(row.get("dimensions") != first_dimensions for row in chunk_rows):
            logger.warning(
                "%s component_id=%s revision=%s reason=heterogeneous_series_id_chunk series_id=%s",
                TABLE_ROW_SERIES_ID_INCONSISTENT_MARKER,
                component_id,
                revision,
                series_id,
            )
            return None

    return [(series_id, tuple(chunks[series_id])) for series_id in order]


def _table_series_entry(
    *, id: str | None, rows: Sequence[Mapping[str, object]], role_keys: Sequence[str]
) -> Mapping[str, object]:
    return {
        "id": id,
        "label": _table_row_label(
            dimensions=cast("Mapping[str, object]", rows[0].get("dimensions") or {})
            if rows
            else {},
            role_keys=role_keys,
        ),
        "points": tuple(
            {"period": row["period"], "value": row["value"]} for row in rows if "value" in row
        ),
    }


def _table_row_label(*, dimensions: Mapping[str, object], role_keys: Sequence[str]) -> str:
    """`bind._label`'s rule, replayed over a stored table row's own
    `dimensions` map: the label-mapped value of every entity-role
    dimension the row carries (`role_keys`, already excluding `stat_var`
    and the facet-identity columns and already sorted into the same order
    `bind._label` joins in), joined with `" / "`. Falls back to the row's
    own `stat_var` value when it carries no entity-role dimension.
    `assembler.mint.builders`'s table `dimensions` values are already
    label-mapped, so no further lookup against `ResolvedFrame.labels` is
    needed here."""
    values = [dimensions[key] for key in role_keys if dimensions.get(key) is not None]
    if not values:
        stat_var_value = dimensions.get(STAT_VAR_COLUMN)
        values = [stat_var_value] if stat_var_value is not None else []
    return " / ".join(str(value) for value in values)


def _approx_tokens(summary: Mapping[str, object]) -> int:
    return len(json.dumps(summary, default=str)) // 4


def _cap_tokens(*, summary: dict[str, object], token_cap: int) -> dict[str, object]:
    """Drops tail series until the summary fits the budget, recording each drop in
    `series_cut` and preserving every field on listed series."""
    series = cast("list[object]", summary["series"])
    while _approx_tokens(summary) > token_cap and series:
        series.pop()
        summary["series_cut"] = cast(int, summary["series_cut"]) + 1
    return summary
