"""The `describe_frame` readback: `get_frame_view`, called by
`assembler.pipeline`'s facade and, through it, by the frames door. Serves
a resolved frame's structural view -- whole, one dimension in full, or
filtered by a substring query -- with no write-side dependencies at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from assembler.errors import BindingInvalid
from assembler.frame_view import dimension_columns, hashable_cell
from assembler.ports import FrameRef, FrameStore, ResolvedFrame, StructuralView


def get_frame_view(
    *,
    frame_ref: FrameRef,
    dimension: str | None = None,
    query: str | None = None,
    frame_store: FrameStore,
) -> Mapping[str, object]:
    """Returns the structural view, one full dimension listing, or substring-filtered listings.

    Dimension listings read frame rows. Filters match case-insensitively."""
    resolved_frame = frame_store.load(frame_ref=frame_ref)

    if dimension is not None:
        return _dimension_values(resolved_frame=resolved_frame, dimension=dimension, query=query)
    if query is not None:
        return _structural_view_filtered(resolved_frame=resolved_frame, query=query)
    return _structural_view_payload(view=resolved_frame.structural_view)


def _structural_view_payload(*, view: StructuralView) -> dict[str, object]:
    return {
        "dimensions": [
            {
                "name": one.name,
                "values": list(one.values),
                "total_count": one.total_count,
                "truncated": one.truncated,
                "plural": one.plural,
            }
            for one in view.dimensions
        ],
        "period_grain": view.period_grain,
        "coverage": list(view.coverage) if view.coverage else None,
        "series_count": view.series_count,
    }


def _dimension_values(
    *, resolved_frame: ResolvedFrame, dimension: str, query: str | None
) -> dict[str, object]:
    columns = tuple(cast("Sequence[str]", resolved_frame.frame["columns"]))
    legal_dimensions = dimension_columns(columns=columns)
    if dimension not in legal_dimensions:
        raise BindingInvalid(
            pointer="/dimension",
            dimension=dimension,
            legal_values=tuple(sorted(legal_dimensions)),
            truncated=False,
            detail=f"{dimension!r} is not a dimension this frame declares",
        )
    index = columns.index(dimension)
    rows = cast("Sequence[Sequence[object]]", resolved_frame.frame["rows"])
    # Render structured units before set construction, matching `DimensionView.values`.
    # Binding preserves the raw unit object in the wire payload.
    values = sorted(
        {cast(str, hashable_cell(row[index])) for row in rows if row[index] is not None}
    )
    if query:
        needle = query.lower()
        values = [value for value in values if needle in value.lower()]
    return {"dimension": dimension, "values": values, "total_count": len(values)}


def _structural_view_filtered(*, resolved_frame: ResolvedFrame, query: str) -> dict[str, object]:
    columns = tuple(cast("Sequence[str]", resolved_frame.frame["columns"]))
    rows = cast("Sequence[Sequence[object]]", resolved_frame.frame["rows"])
    needle = query.lower()
    dimensions = []
    for name in dimension_columns(columns=columns):
        index = columns.index(name)
        # Match the rendered result text; structured units use this same text as
        # their substring target and set key.
        texts = (cast(str, hashable_cell(row[index])) for row in rows if row[index] is not None)
        values = sorted({text for text in texts if needle in text.lower()})
        dimensions.append({"name": name, "values": values, "matched_count": len(values)})
    coverage = resolved_frame.structural_view.coverage
    return {
        "dimensions": dimensions,
        "period_grain": resolved_frame.structural_view.period_grain,
        "coverage": list(coverage) if coverage else None,
    }
