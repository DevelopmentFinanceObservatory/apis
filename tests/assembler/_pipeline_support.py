"""Shared intent and frame helpers for assembler pipeline tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from dfo_frame_grammar import UNIT_COLUMN, unit_cell_to_json

from assembler.frame_view import compute_structural_view
from assembler.ports import FrameRef, ResolvedFrame
from assembler.testing import (
    FIXTURE_RETRIEVED_ON,
    build_frame_doc,
    build_resolved_frame,
    malaria_ethiopia_frame_ref,
    malaria_ethiopia_resolved_frame,
)
from description.facts import DescriptionRecord


def _stored_resolved_frame(
    *,
    frame_ref: FrameRef,
    role_keys: Sequence[str],
    rows: Iterable[Mapping[str, object]],
    labels: Mapping[str, str] | None = None,
) -> ResolvedFrame:
    """Build a resolved frame whose `unit` cells are frame1 JSON, as a stored frame's are.

    `Frame.build` holds a converted or composed unit as an object."""
    doc = build_frame_doc(role_keys=role_keys, rows=rows)
    unit_index = cast("list[str]", doc["columns"]).index(UNIT_COLUMN)
    for row in cast("list[list[object]]", doc["rows"]):
        row[unit_index] = unit_cell_to_json(row[unit_index])
    return ResolvedFrame(
        frame_ref=frame_ref,
        frame=doc,
        structural_view=compute_structural_view(frame=doc, continuation=None),
        manifest={"schema_version": 1},
        labels=labels or {},
        retrieved_on=FIXTURE_RETRIEVED_ON,
    )


def _malaria_intent(**overrides):
    frame = malaria_ethiopia_resolved_frame()
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": frame.frame_ref.spec_id,
            "content_version": frame.frame_ref.content_version,
        },
        "select": {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": "country/ETH",
        },
        "window": {"start": "2015", "end": "2024"},
        "title": "Malaria grant disbursements to Ethiopia",
    }
    intent.update(overrides)
    return frame, intent


def _stat_card_intent(**overrides):
    frame = malaria_ethiopia_resolved_frame()
    intent = {
        "type": "stat_card",
        "frame_ref": {
            "spec_id": frame.frame_ref.spec_id,
            "content_version": frame.frame_ref.content_version,
        },
        "select": {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": "country/ETH",
        },
    }
    intent.update(overrides)
    return frame, intent


def _bar_chart_intent(**overrides):
    frame = malaria_ethiopia_resolved_frame()
    intent = {
        "type": "bar_chart",
        "frame_ref": {
            "spec_id": frame.frame_ref.spec_id,
            "content_version": frame.frame_ref.content_version,
        },
        "select": {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": "country/ETH",
        },
        "window": {"start": "2015", "end": "2024"},
        "title": "Malaria grant disbursements to Ethiopia",
    }
    intent.update(overrides)
    return frame, intent


def _table_intent(**overrides):
    frame = malaria_ethiopia_resolved_frame()
    intent = {
        "type": "table",
        "frame_ref": {
            "spec_id": frame.frame_ref.spec_id,
            "content_version": frame.frame_ref.content_version,
        },
        "select": {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": "country/ETH",
        },
        "window": {"start": "2015", "end": "2024"},
        "title": "Malaria grant disbursements to Ethiopia",
    }
    intent.update(overrides)
    return frame, intent


def _multi_recipient_frame():
    data = {
        "country/ETH": {"2020": 100.0, "2021": 110.0, "2022": 120.0},
        "country/GHA": {"2020": 200.0, "2021": 210.0, "2022": 220.0},
        "country/KEN": {"2020": 300.0, "2021": 310.0, "2022": 320.0},
    }
    return build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": recipient,
                "time_period": period,
                "value": value,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_oecd_dac",
            }
            for recipient, values in data.items()
            for period, value in values.items()
        ],
    )


def _donor_and_recipient_frame():
    """Build rows with a constant donor and varying recipient for label tests."""
    data = {
        "country/ETH": {"2020": 100.0, "2021": 110.0},
        "country/GHA": {"2020": 200.0, "2021": 210.0},
    }
    return build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(),
        role_keys=("donor", "recipient"),
        rows=[
            {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "donor": "org/GlobalFund",
                "recipient": recipient,
                "time_period": period,
                "value": value,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_oecd_dac",
            }
            for recipient, values in data.items()
            for period, value in values.items()
        ],
        labels={
            "org/GlobalFund": "Global Fund",
            "country/ETH": "Ethiopia",
            "country/GHA": "Ghana",
        },
    )


def computed_record(
    *, frame_ref: FrameRef, steps: Sequence[Mapping[str, object]]
) -> DescriptionRecord:
    """The record Datasets writes for the `ops_` version `frame_ref` when it is published with
    the malaria version retrieved, then `steps`."""
    malaria = malaria_ethiopia_frame_ref()
    return DescriptionRecord.model_validate(
        {
            "id": frame_ref.spec_id,
            "content_version": frame_ref.content_version,
            "schema_version": 2,
            "basis": "mint",
            "index_stamp": None,
            "variables": None,
            "computed": {
                "inputs": [{"id": malaria.spec_id, "content_version": malaria.content_version}],
                "composed": None,
            },
            "series": {
                "series": 2,
                "rows": None,
                "periods": None,
                "span": {"start": "2018", "end": "2022"},
                "grain": "year",
                "latest_period": "2022",
                "series_reaching_latest": None,
                "entities_by_role": None,
                "by_source_unit": [{"source": None, "unit": None, "series": 2}],
                "status": {"estimated": None, "provisional": None},
            },
            "steps": [
                {
                    "id": "s1",
                    "op": "retrieve",
                    "input": {
                        "id": malaria.spec_id,
                        "content_version": malaria.content_version,
                    },
                },
                *steps,
            ],
            "caveats": [],
            "evidence": {
                "prepared": "2026-10-28",
                "title": "Computed from malaria grants",
                "sources": [],
                "source_release": None,
                "labels": [],
                "membership": None,
                "builds": {
                    "origin_catalogue": None,
                    "code_index": None,
                    "coverage": None,
                    "method_status": None,
                },
            },
            "late_fields": [],
        }
    )
