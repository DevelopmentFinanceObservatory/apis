"""Builds grammar-valid frame1 fixtures and matching `ResolvedFrame` objects.

`Frame.build` applies the same NFC, sort-key, and duplicate-grain checks as
production. `malaria_ethiopia_*` supplies the observations used by the
catalogue's worked line-chart fixture."""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Mapping, Sequence

from dfo_frame_grammar import Frame

from assembler.frame_view import compute_structural_view
from assembler.ports import FrameRef, ResolvedFrame

__all__ = [
    "FIXTURE_RETRIEVED_ON",
    "build_frame_doc",
    "build_resolved_frame",
    "malaria_ethiopia_frame_ref",
    "malaria_ethiopia_resolved_frame",
    "malaria_ethiopia_rows",
]

# The `retrieved_on` every fixture frame carries unless a test names its
# own: the `retrieved` date the catalogue bundle's worked fixtures state.
FIXTURE_RETRIEVED_ON = datetime.date(2026, 6, 12)


def build_frame_doc(
    *, role_keys: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> dict[str, object]:
    """Builds a grammar-valid frame1 document (`{frame_version, columns,
    rows}`) from row mappings keyed by column name, through
    `dfo_frame_grammar.Frame.build` -- so each fixture follows the production grammar."""
    frame = Frame.build(role_keys=role_keys, rows=rows)
    return {
        "frame_version": "frame1",
        "columns": list(frame.columns),
        "rows": [list(row) for row in frame.rows],
    }


def build_resolved_frame(
    *,
    frame_ref: FrameRef,
    role_keys: Sequence[str],
    rows: Iterable[Mapping[str, object]],
    labels: Mapping[str, str] | None = None,
    manifest: Mapping[str, object] | None = None,
    retrieved_on: datetime.date = FIXTURE_RETRIEVED_ON,
) -> ResolvedFrame:
    doc = build_frame_doc(role_keys=role_keys, rows=rows)
    structural_view = compute_structural_view(frame=doc)
    return ResolvedFrame(
        frame_ref=frame_ref,
        frame=doc,
        structural_view=structural_view,
        manifest=manifest or {"schema_version": 1},
        labels=labels or {},
        retrieved_on=retrieved_on,
    )


def malaria_ethiopia_frame_ref() -> FrameRef:
    """The `frame_ref` the catalogue bundle's worked line_chart fixture
    carries (`fixtures/line_chart/intent.json`)."""
    return FrameRef(
        spec_id="spec_9f2e41a7c03b58d2a4c1f6b2e8d35790",
        content_version="cv_f4a81c22d9075e3ba61c2ee98f83b405",
    )


def malaria_ethiopia_rows() -> tuple[dict[str, object], ...]:
    """Observations for the worked line-chart series, excluding 2018.

    Binding derives its `status: "missing"` point from the absent row."""
    values = {
        "2015": 412.3,
        "2016": 455.1,
        "2017": 431.8,
        "2019": 410.0,
        "2020": 398.2,
        "2021": 405.6,
        "2022": 420.1,
        "2023": 433.7,
        "2024": 441.9,
    }
    return tuple(
        {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": "country/ETH",
            "time_period": period,
            "value": value,
            "unit": None,
            "measurementMethod": None,
            "provenance": "src_oecd_dac",
        }
        for period, value in values.items()
    )


def malaria_ethiopia_resolved_frame() -> ResolvedFrame:
    return build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(),
        role_keys=("recipient",),
        rows=malaria_ethiopia_rows(),
        labels={"country/ETH": "Ethiopia"},
    )
