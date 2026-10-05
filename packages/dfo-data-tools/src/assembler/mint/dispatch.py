"""Builds and validates type-specific components, provenance, and versioned envelopes."""

from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from assembler.bind import BoundSeries
from assembler.catalogue import CatalogueRelease
from assembler.mint.builders import (
    _build_bar_chart,
    _build_line_chart,
    _build_note_narration,
    _build_source_card,
    _build_stat_card,
    _build_table,
)
from assembler.mint.provenance import (
    _build_all_sources,
    _build_frameless_provenance,
    _build_provenance,
    _build_sources,
)
from assembler.mint.validate import _validate, component_registry
from assembler.ports import ResolvedFrame
from assembler.resolve import ResolvedSeries


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


@dataclass(frozen=True)
class MintedComponent:
    """Minted component and provenance, plus observed periods for stat-card
    validation."""

    component: Mapping[str, object]
    provenance: Mapping[str, object]
    observed_periods: frozenset[str]


def mint_component(
    *,
    member: str,
    merged_intent: Mapping[str, object],
    resolved_frame: ResolvedFrame | None,
    series: Sequence[ResolvedSeries],
    bound_series: Sequence[BoundSeries],
    release: CatalogueRelease,
    clock: Callable[[], datetime.datetime] = _utc_now,
) -> MintedComponent:
    if member == "note_narration":
        # No frame, no sources, nothing to pin -- the write-side grammar
        # carries no `frame_ref` field for this member.
        component, observed_periods = _build_note_narration(merged_intent=merged_intent)
        _validate(
            schema=release.component_schemas[member],
            instance=component,
            legal_fields=tuple(
                sorted(
                    cast(
                        Mapping[str, object],
                        release.component_schemas[member].get("properties", {}),
                    )
                )
            ),
        )
        provenance = _build_frameless_provenance(clock=clock)
        return MintedComponent(
            component=component, provenance=provenance, observed_periods=observed_periods
        )

    assert resolved_frame is not None  # every member but note_narration resolves a frame

    if member == "source_card":
        # Every source the frame's own rows carry, not just the ones a
        # resolved series happened to reference -- source_card binds no
        # series of its own.
        sources, _ = _build_all_sources(resolved_frame=resolved_frame)
    else:
        sources, source_id_by_raw_ref = _build_sources(
            bound_series=bound_series, resolved_frame=resolved_frame
        )

    if member == "line_chart":
        component, observed_periods = _build_line_chart(
            merged_intent=merged_intent,
            resolved_frame=resolved_frame,
            series=series,
            bound_series=bound_series,
            source_id_by_raw_ref=source_id_by_raw_ref,
        )
    elif member == "bar_chart":
        component, observed_periods = _build_bar_chart(
            merged_intent=merged_intent,
            resolved_frame=resolved_frame,
            series=series,
            bound_series=bound_series,
            source_id_by_raw_ref=source_id_by_raw_ref,
        )
    elif member == "table":
        component, observed_periods = _build_table(
            merged_intent=merged_intent,
            resolved_frame=resolved_frame,
            series=series,
            bound_series=bound_series,
            source_id_by_raw_ref=source_id_by_raw_ref,
        )
    elif member == "stat_card":
        component, observed_periods = _build_stat_card(
            merged_intent=merged_intent,
            resolved_frame=resolved_frame,
            series=series,
            bound_series=bound_series,
        )
    elif member == "source_card":
        component, observed_periods = _build_source_card(sources=sources)
    else:
        raise AssertionError(f"mint_component has no builder for catalogue member {member!r}")

    _validate(
        schema=release.component_schemas[member],
        instance=component,
        legal_fields=tuple(
            sorted(
                cast(Mapping[str, object], release.component_schemas[member].get("properties", {}))
            )
        ),
    )

    provenance = _build_provenance(resolved_frame=resolved_frame, sources=sources, clock=clock)
    return MintedComponent(
        component=component, provenance=provenance, observed_periods=observed_periods
    )


def stamp_envelope(
    *, minted: MintedComponent, component_id: str, revision: int, release: CatalogueRelease
) -> dict[str, object]:
    envelope = {
        "schema_version": release.schema_version,
        "catalog_version": release.catalog_version,
        "component_id": component_id,
        "revision": revision,
        "provenance": minted.provenance,
        "component": minted.component,
    }
    _validate(
        schema=release.envelope_schema,
        instance=envelope,
        legal_fields=tuple(
            sorted(cast("Mapping[str, object]", release.envelope_schema.get("properties", {})))
        ),
        registry=component_registry(release),
    )
    return envelope
