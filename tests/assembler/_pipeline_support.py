"""Shared intent and frame helpers for assembler pipeline tests."""

from __future__ import annotations

from assembler.testing import (
    build_resolved_frame,
    malaria_ethiopia_frame_ref,
    malaria_ethiopia_resolved_frame,
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
